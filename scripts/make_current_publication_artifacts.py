#!/usr/bin/env python3
"""Render the current four-cohort paper from completed, immutable results.

Historical dataset outputs remain available to their original analysis scripts.
This is the publication entrypoint after the exploratory Kumar2024 addition;
it performs no fitting, resampling, or parameter selection.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
KUMAR = RESULTS / "eeg" / "kumar2024"
SUPPLEMENTAL = KUMAR / "supplemental"
TABLES = ROOT / "build" / "tables"
FIGURES = ROOT / "build" / "figures"
sys.path.insert(0, str(ROOT / "analysis"))


def read_json(path: Path):
    return json.loads(path.read_text())


def kumar_results():
    data = read_json(KUMAR / "summary.json")
    subjects = [dict(subject=s["subject"], subject_means=s,
                     screening=dict(methods=s["full_target_screening"]))
                for s in data["subject_means"]]
    if len(subjects) != 18 or {s["subject"] for s in subjects} != set(range(1, 19)):
        raise ValueError("Kumar2024 publication requires all 18 subjects")
    if any(s["subject_means"]["n_splits"] != 30 or s["subject_means"]["n_outer"] != 50 for s in subjects):
        raise ValueError("Incomplete Kumar2024 split or coverage results")
    return subjects, read_json(KUMAR / "summary.json")


def verify_supplemental():
    if read_json(SUPPLEMENTAL/"verification.json").get("status") != "passed":
        raise ValueError("Kumar2024 supplementary results have not passed independent verification")


def paired_inference(differences):
    d = np.asarray(differences, dtype=float)
    n, mean = len(d), float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    radius = float(t.ppf(.975, n - 1) * se)
    lcb = float(mean - t.ppf(.95, n - 1) * se)
    return dict(n_subjects=n, mean_diff=mean, ci95_low=mean-radius,
                ci95_high=mean+radius, one_sided_lcb95=lcb,
                p_t=float(t.sf((mean+.02)/se, n-1)),
                noninferior_at_delta=lcb > -.02)


def current_cohorts():
    legacy_ni = read_json(RESULTS / "eeg/noninferiority/summary.json")
    specs = [
        ("Ma2020", "ma2020", "eeg/ma2020/downstream_summary.csv", None),
        ("Stieger2021", "stieger2021", "eeg/stieger2021/downstream_summary.csv", None),
        ("BNCI2014_004", "bnci2014_004", "eeg/bnci2014_004/downstream_summary.csv", "primary"),
    ]
    rows = []
    for name, key, path, analysis in specs:
        frame = pd.read_csv(RESULTS / path)
        if analysis:
            frame = frame[frame.analysis == analysis]
        selected = frame.set_index("method")
        all_row, ref = selected.loc["all_sources"], selected.loc["pair_ref"]
        rows.append(dict(dataset=name, key=key, candidate_count=float(all_row.mean_set_size),
                         retained_count=float(ref.mean_set_size),
                         accuracy_ref=float(ref.mean_balanced_accuracy),
                         accuracy_all=float(all_row.mean_balanced_accuracy),
                         inference=legacy_ni[key]))
    subjects, summary = kumar_results()
    methods = summary["overall"]["methods"]
    differences = [s["subject_means"]["methods"]["pair_ref"]["balanced_accuracy"]
                   - s["subject_means"]["methods"]["all_sources"]["balanced_accuracy"] for s in subjects]
    row = dict(dataset="Kumar2024", key="kumar2024", candidate_count=5.,
               retained_count=methods["pair_ref"]["set_size"],
               accuracy_ref=methods["pair_ref"]["balanced_accuracy"],
               accuracy_all=methods["all_sources"]["balanced_accuracy"],
               inference=paired_inference(differences))
    rows.insert(2, row)
    return rows


def eeg_overview_data():
    return pd.DataFrame([dict(dataset=r["dataset"], candidate_count=r["candidate_count"],
        retained_count=r["retained_count"], retained_fraction=r["retained_count"]/r["candidate_count"],
        mean_difference=r["inference"]["mean_diff"], ci95_low=r["inference"]["ci95_low"],
        ci95_high=r["inference"]["ci95_high"], one_sided_lcb95=r["inference"]["one_sided_lcb95"],
        two_pp_noninferiority="established" if r["inference"]["noninferior_at_delta"] else "inconclusive")
        for r in current_cohorts()])


def write_table(name, label, caption, columns, header, body, *, wide=False, out=TABLES):
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join([r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}",
        rf"\label{{{label}}}", r"\WideTableBody" if wide else r"\TableBody",
        rf"\begin{{tabular}}{{{columns}}}", r"\toprule", header+r" \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out / name).write_text(text)


def overview_table(out=TABLES):
    body = []
    for r in current_cohorts():
        q = r["inference"]
        name = r["dataset"].replace("_", r"\_")
        if r["key"] == "kumar2024":
            name += r"\tnote{b}"
        elif r["key"] == "bnci2014_004":
            name += r"\tnote{a}"
        n = q.get("n_subjects", q.get("n"))
        body.append(f"{name} & {n} & {r['candidate_count']:.2f} & {r['retained_count']:.2f} & "
                    f"{r['accuracy_ref']:.3f} & {r['accuracy_all']:.3f} & "
                    rf"\({q['mean_diff']:+.4f}\;[{q['ci95_low']:.3f},\,{q['ci95_high']:.3f}]\) & "
                    + ("Established" if q["noninferior_at_delta"] else "Inconclusive") + r" \\")
    text = r"""\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{Latest-session motor-imagery EEG results.}
\label{tab:eeg-summary}
\WideTableBody
\begin{tabular}{@{}lrrrrrrl@{}}
\toprule
& & \multicolumn{2}{c}{Mean source count} & \multicolumn{2}{c}{Balanced accuracy} & \multicolumn{2}{c}{Paired comparison} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}
Dataset & Subjects & Candidate & Retained & Ref. & All & Difference (95\% CI) & 2-pp NI \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\TableNotes
\item[] Candidate and retained counts are averaged over subjects and downstream target splits where applicable. Difference is the refinement-gate minus all-source subject-level balanced-accuracy difference with a two-sided 95\% confidence interval. The two-percentage-point non-inferiority (NI) decision uses a one-sided paired \(t\)-test at \(\alpha=0.05\), equivalently a 95\% one-sided lower confidence bound.
\item[a] Secondary three-channel check.
\item[b] Exploratory cohort added after the original analyses; the primary analysis protocol was fixed before classification and coverage results were inspected.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_eeg_summary.tex").write_text(text)


def noninferiority_table(out=TABLES):
    margins = pd.read_csv(RESULTS / "eeg/noninferiority/margin_sensitivity.csv")
    body = []
    for r in current_cohorts():
        q = r["inference"]
        name = r["dataset"].replace("_", r"\_")
        if r["key"] == "kumar2024":
            pw = "--"
            sensitivity = "/".join("yes" if q["one_sided_lcb95"] > -margin else "no"
                                   for margin in (.01, .02)) + "/--"
            pt, lcb = q["p_t"], f"{q['one_sided_lcb95']:+.4f}"
        else:
            pw = f"{q['p_noninferiority_wilcoxon']:.3f}"
            pt, lcb = q["p_noninferiority_ttest"], f"{q['one_sided_lcb95']:+.3f}"
            part = margins[margins.dataset == r["key"]].sort_values("delta_NI")
            sensitivity = "/".join("yes" if bool(v) else "no" for v in part.noninferior_at_delta)
        n = q.get("n_subjects", q.get("n"))
        body.append(f"{name} & {n} & {q['mean_diff']:+.4f} & "
            f"$[{q['ci95_low']:+.3f},{q['ci95_high']:+.3f}]$ & {lcb} & "
            + ("yes" if q["noninferior_at_delta"] else "no")
            + f" & {pt:.3f} & {pw} & {sensitivity}" + r" \\")
    write_table("table_noninferiority.tex", "tab:noninferiority",
        r"Subject-level paired non-inferiority of refinement versus all-source balanced accuracy. "
        r"Splits are averaged within subject. The primary one-sided paired $t$ test uses "
        r"$H_0:\mu_{\mathrm{Ref}}-\mu_{\mathrm{All}}\le-\delta_{NI}$, $\alpha=0.05$, "
        r"and $\delta_{NI}=0.02$. Non-inferiority holds when the 95\% one-sided lower "
        r"confidence bound (LCB) exceeds $-\delta_{NI}$. The two-sided 95\% CI describes "
        r"effect size. $p_W$ is the Wilcoxon sensitivity result. The final column varies "
        r"the $t$-test margin. Kumar2024 is exploratory; its fixed plan included only "
        r"the $t$ tests at 1 and 2 pp, so unplanned Wilcoxon and 3-pp results are not reported. "
        r"BNCI2014\_004 is a secondary nine-subject, three-channel check.",
        "lcccccccc", r"Dataset & $n$ & \makecell{Mean\\diff.} & $95\%$ CI & \makecell{$95\%$\\LCB} & \makecell{Non-\\inf.?} & $p_t$ & $p_W$ & \makecell{NI at\\1/2/3 pp}",
        body, wide=True, out=out)


def kumar_table(out=TABLES):
    subjects, summary = kumar_results()
    methods = summary["overall"]["methods"]
    full_count = np.mean([s["screening"]["methods"]["pair_ref"]["set_size"] for s in subjects])
    mcs_difference = paired_inference([
        s["subject_means"]["methods"]["pair_ref"]["balanced_accuracy"]
        - s["subject_means"]["methods"]["mcs_stepdown"]["balanced_accuracy"] for s in subjects])
    all_acc, target_acc = (methods[m]["balanced_accuracy"] for m in ("all_sources", "target_only"))
    body = []
    for key, name in [("all_sources", "All sources"), ("top1", "Top-1"), ("top3", "Top-3"),
                      ("mcs_stepdown", "MCS-style stepdown"), ("pair_ref", "Ref."), ("target_only", "Target only")]:
        r = methods[key]
        below = np.mean([s["subject_means"]["methods"][key]["balanced_accuracy"] <
                         s["subject_means"]["methods"]["target_only"]["balanced_accuracy"] for s in subjects])
        body.append(f"{name} & {r['set_size']:.2f} & {r['training_trials']:.1f} & "
            f"{r['balanced_accuracy']:.3f} & {r['balanced_accuracy']-all_acc:+.3f} & "
            f"{r['balanced_accuracy']-target_acc:+.3f} & {below:.2f}" + r" \\")
    write_table("table_kumar2024_riemann.tex", "tab:kumar2024-riemann",
        r"Exploratory Kumar2024 longitudinal evaluation. Session six is the target; "
        r"sessions one to five are candidate sources for all 18 subjects. Full-target "
        f"refinement screening retains {full_count:.2f} sources on average. Downstream values first "
        r"average 30 stratified half-splits within subject and then subjects. Train trials "
        r"count selected historical trials, except for target only, which uses labeled "
        r"screening-half trials and is not label-budget matched. Below target is the "
        r"fraction of subject-level accuracies below that reference. Refinement minus "
        f"MCS-style accuracy is ${mcs_difference['mean_diff']:+.4f}$ "
        rf"(paired 95\% CI $[{mcs_difference['ci95_low']:.4f},{mcs_difference['ci95_high']:.4f}]$).",
        "lrrrrrr", r"Method & \makecell{Set\\size} & \makecell{Train\\trials} & \makecell{Bal.\\acc.} & \makecell{Diff. vs\\All} & \makecell{Diff. vs\\target} & \makecell{Below\\target}",
        body, out=out)


def empirical_table(out=TABLES, audit_dir=None):
    from verify_empirical_coverage import compute_empirical_coverage
    folder = Path(audit_dir) if audit_dir is not None else RESULTS / "eeg/empirical_reference"
    bnci = compute_empirical_coverage(folder).set_index("dataset").loc["BNCI2014_004"].to_dict()
    bnci["jaccard"] = float(pd.read_csv(folder/"bnci_subject_stability.csv").mean_pairwise_jaccard.mean())
    bnci["subjects"], bnci["outer_total"] = int(bnci["subjects"]), int(bnci["outer_total"])
    _, summary = kumar_results()
    c = summary["overall"]["coverage"]
    kumar = dict(c, subjects=18, outer_total=900, jaccard=c["mean_pairwise_jaccard"])
    body = []
    metrics = ["coverage_c", "coverage_d", "coverage_joint", "oracle_retained", "exact_recovery"]
    for name, count, r in [("Kumar2024", 50, kumar), (r"BNCI2014\_004", 200, bnci)]:
        body.append(f"{name} & {r['subjects']} & {r['outer_total']} ({count}) & "
            + " & ".join(f"{r[m]:.3f}" for m in metrics)
            + f" & {r['set_size']:.2f} & {r['jaccard']:.3f}" + r" \\")
    write_table("table_eeg_empirical_audit.tex", "tab:eeg-empirical-audit",
        r"Empirical-reference coverage and retained-set stability. Full-session empirical "
        r"distributions define pseudo-population discrepancies. Outer draws use 100\% of "
        r"each source-session trial count and 50\% of the target count, with replacement. "
        r"Totals are followed by per-subject counts. Each draw is recalibrated using 499 "
        r"shared-target bootstrap repetitions. Nominal component, contrast, and split "
        r"joint levels are 97.5\%, 97.5\%, and 95\%. Oracle retention and exact recovery "
        r"refer to the empirical-reference argmin set. Set size averages outer draws; "
        r"Jaccard averages within-subject pairwise set overlap. Kumar2024 is exploratory "
        r"and uses the same eligible one-second trials as its downstream analysis.",
        "@{}lrrrrrrrrr@{}", r"Dataset & Subjects & \makecell{Outer\\samples} & $\Pr(\mathcal E_C)$ & $\Pr(\mathcal E_D)$ & Joint & \makecell{Oracle\\retained} & Exact & \makecell{Set\\size} & Jaccard",
        body, wide=True, out=out)


def budget_tables(out=TABLES):
    verify_supplemental()
    old = RESULTS / "eeg/bspc_revision_20261002/bootstrap_budget"
    summary = pd.read_csv(old/"dataset_summary.csv")
    seeds = pd.read_csv(old/"seed_variability_summary.csv")
    diagnostics = pd.read_csv(old/"se_diagnostics_summary.csv")
    current = read_json(SUPPLEMENTAL/"budget_summary.json")
    raw = current["subject_summary"]
    if len(raw) != 12 or {r["subject"] for r in raw} != {1, 2, 3}:
        raise ValueError("Incomplete fixed Kumar2024 budget design")
    body, se_body = [], []
    for dataset, name in [("ma2020", "Ma2020"), ("stieger2021", "Stieger2021"), ("kumar2024", "Kumar2024")]:
        if dataset == "kumar2024":
            rows = pd.DataFrame(current["rows"]).set_index("budget")
            seed_rows = pd.DataFrame(current["seed_variability_summary"]).set_index("budget")
            diagnostic = pd.DataFrame(current["diagnostics"])
        else:
            rows = summary[summary.dataset == dataset].set_index("budget")
            seed_rows = seeds[seeds.dataset == dataset].set_index("budget")
            diagnostic = diagnostics[diagnostics.dataset == dataset]
        for position, budget in enumerate((99, 199, 499, 999)):
            r, seed = rows.loc[budget], seed_rows.loc[budget]
            body.append(" & ".join([name if position == 0 else "", str(budget),
                f"{r.set_size:.2f}", f"{r.jaccard_vs_999:.3f}", f"{seed.pairwise_seed_jaccard:.3f}",
                f"{r.q_component_abs_change_vs_999:.3f}", f"{r.q_contrast_abs_change_vs_999:.3f}"]) + r" \\")
        body.append(r"\midrule")
        ranges = [f"{diagnostic[f'raw_{kind}_se_min'].min():.3g}--{diagnostic[f'raw_{kind}_se_max'].max():.3g}"
                  for kind in ("component", "contrast")]
        totals = [sum(int(diagnostic[f"raw_{kind}_se_{suffix}"].sum()) for kind in ("component", "contrast"))
                  for suffix in ("nonfinite_count", "floor_count")]
        se_body.append(" & ".join([name, *ranges, *map(str, totals)]) + r" \\")
    write_table("table_bspc_bootstrap_budget.tex", "tab:bspc-budget",
        r"Finite-bootstrap numerical stability. Subjects 1--3 from each dataset use "
        r"the first two primary-protocol half-splits and three fixed bootstrap seeds. "
        r"Each budget uses a prefix of the same 999-draw matrix, with standard errors "
        r"and critical values recalculated. $J_{999}$ compares sets within a seed; "
        r"$J_{\mathrm{seed}}$ averages pairwise comparisons across seeds. "
        r"$|\Delta q_C|$ and $|\Delta q_D|$ are absolute critical-value changes from 999 "
        r"draws. Summaries average within subject first. The Kumar2024 check was planned "
        r"after its primary results and is exploratory. No population coverage is estimated.",
        "@{}lrrrrrr@{}", r"Dataset & $B$ & Sessions & $J_{999}$ & $J_{\mathrm{seed}}$ & $|\Delta q_C|$ & $|\Delta q_D|$",
        body[:-1], out=out)
    write_table("table_bspc_se_diagnostics.tex", "tab:bspc-se-diagnostics",
        r"Raw bootstrap standard-error diagnostics across the samples, seeds, and "
        r"four budgets in Table~\ref{tab:bspc-budget}. Ranges combine all examined "
        r"coordinates. The final column counts coordinates below $10^{-12}$ before "
        r"flooring. These diagnostics identify observed numerical degeneracy; they "
        r"do not prove population nondegeneracy or moment bounds.",
        "@{}lrrrr@{}", r"Dataset & Component SE range & Contrast SE range & Nonfinite & Below floor",
        se_body, out=out)


def block_table(out=TABLES):
    verify_supplemental()
    folder = RESULTS / "eeg/empirical_reference"
    old_screen = pd.read_csv(folder/"block_screening_summary.csv")
    old_down = pd.read_csv(folder/"block_downstream_summary.csv")
    current = read_json(SUPPLEMENTAL/"block_summary.json")
    screening = pd.concat([old_screen[old_screen.dataset == "stieger"], pd.DataFrame(current["screening"])])
    downstream = pd.concat([old_down[old_down.dataset == "stieger"], pd.DataFrame(current["downstream"])])
    downstream = downstream[downstream.method == "pair_ref"].set_index(["dataset", "block_length"])
    body = []
    for dataset, name, n in [("stieger", "Stieger2021", 62), ("kumar2024", "Kumar2024", 18)]:
        rows = screening[screening.dataset == dataset].sort_values("block_length")
        if rows.block_length.tolist() != [1, 4, 8] or not (rows.subjects == n).all():
            raise ValueError(f"Incomplete block summary: {dataset}")
        for position, r in enumerate(rows.itertuples()):
            length = int(r.block_length)
            label = "i.i.d." if length == 1 else f"block-{length}"
            accuracy, difference = "--", "--"
            if (dataset, length) in downstream.index:
                d = downstream.loc[(dataset, length)]
                accuracy, difference = f"{d.mean_balanced_accuracy:.3f}", f"{d.mean_diff_vs_all:+.3f}"
            body.append(" & ".join([name if position == 0 else "", label,
                f"{r.pair_ref_size:.2f}", f"{r.mcs_size:.2f}", f"{r.jaccard_vs_iid:.2f}",
                f"{r.exact_match_iid:.2f}", accuracy, difference]) + r" \\")
        body.append(r"\midrule")
    write_table("table_block_bootstrap.tex", "tab:block-bootstrap",
        r"Shared-target circular moving-block sensitivity. Length one is the i.i.d. "
        r"trial-bootstrap control. Full-target screening uses 499 draws at lengths "
        r"1, 4, and 8. Jaccard and exact-set agreement compare refinement sets with "
        r"the i.i.d. control. Downstream accuracy uses ten fixed held-out target "
        r"splits and 99 draws at lengths 1 and 4. Kumar2024 uses all 18 subjects and "
        r"was planned after the primary results. Its ordered arrays join runs and "
        r"omit short trials; screening halves omit further trials. Blocks preserve "
        r"array order, not uninterrupted physical time, and can cross gaps and run boundaries.",
        "llrrrrrr", r"Dataset & Bootstrap & \makecell{Ref.\\size} & \makecell{MCS\\size} & Jaccard & Exact & \makecell{Split\\acc.} & \makecell{$\Delta$ vs\\All}",
        body[:-1], wide=True, out=out)


def certificate_figure(output):
    import matplotlib.pyplot as plt
    from plot_style import STATUS_COLORS, format_axis, setup_style
    verify_supplemental()
    result = read_json(SUPPLEMENTAL/"certificate.json")
    if result["subject"] != 1 or result["bootstrap"] != 1999 or result["source_sessions"] != [1, 2, 3, 4, 5]:
        raise ValueError("Certificate is not the fixed subject-1, 1999-draw example")
    setup_style()
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    rows = result["sources"]
    low = min(r["lower_component"] for r in rows)
    high = max(r["upper_component"] for r in rows)
    span = high - low
    label_x = high + .04 * span
    for position, row in enumerate(rows):
        retained = row["refinement_retained"]
        color = STATUS_COLORS["Retained" if retained else "Excluded"]
        delta = row["discrepancy"]
        ax.errorbar(delta, position,
            xerr=[[delta-row["lower_component"]], [row["upper_component"]-delta]],
            fmt="o", color=color, ecolor=color, capsize=3, lw=1.4, ms=6)
        if retained:
            label = "retained"
        elif row["rectangle_excluded"]:
            label = (f"excluded: rectangle vs S{row['rectangle_competitor_session']}\n"
                     f"lower minus upper = {row['rectangle_exclusion_margin']:.3f}")
        else:
            witnesses = row["positive_contrast_witnesses"]
            if not witnesses:
                raise ValueError("Excluded source has no positive exclusion witness")
            witness = max(witnesses, key=lambda r: r["lower_bound"])
            label = (f"excluded: contrast vs S{witness['competitor_session']}\n"
                     f"lower bound = {witness['lower_bound']:.3f}")
        ax.text(label_x, position, label, color=color, va="center", fontsize=8.2)
    threshold = rows[0]["rectangle_threshold"]
    ax.axvline(threshold, color=".25", linestyle="--", lw=1.2,
               label="Minimum simultaneous upper limit")
    ax.set_yticks(range(5), [f"Source session {r['source_session']}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(low-.04*span, high+.70*span)
    ax.set_xlabel("Squared AIRM discrepancy to target session")
    ax.set_title("Kumar2024 subject 1: refinement screening")
    format_axis(ax, ygrid=False)
    ax.grid(axis="x", alpha=.35)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), frameon=False, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def export_tiffs(figure_dir=FIGURES, output_dir=ROOT/"build"/"submission_figures"):
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise RuntimeError("TIFF export requires the existing Poppler pdftoppm executable")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("fig_eeg_overview", "fig_kumar2024_exclusion_certificate", "fig_shared_target_evidence"):
        subprocess.run([executable, "-r", "300", "-singlefile", "-tiff",
                        str(figure_dir/f"{name}.pdf"), str(output_dir/name)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("core", "supplemental", "all"), default="all")
    parser.add_argument("--table-dir", type=Path, default=TABLES)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    parser.add_argument("--export-tiff", action="store_true", help="also refresh 300-dpi RGB TIFF alternatives")
    args = parser.parse_args()
    if args.part in {"core", "all"}:
        overview_table(args.table_dir)
        noninferiority_table(args.table_dir)
        kumar_table(args.table_dir)
        empirical_table(args.table_dir)
        from rebuild_figures import plot_eeg_overview
        overview = eeg_overview_data()
        plot_eeg_overview(overview, args.figure_dir/"fig_eeg_overview.pdf")
    if args.part in {"supplemental", "all"}:
        budget_tables(args.table_dir)
        block_table(args.table_dir)
        certificate_figure(args.figure_dir/"fig_kumar2024_exclusion_certificate.pdf")
    if args.export_tiff:
        export_tiffs(args.figure_dir)


if __name__ == "__main__":
    main()
