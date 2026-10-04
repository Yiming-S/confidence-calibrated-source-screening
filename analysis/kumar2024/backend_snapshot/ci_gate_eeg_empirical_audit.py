#!/usr/bin/env python3
"""Empirical-reference coverage, stability, and exclusion-certificate audit.

The full BNCI2014_004 and Zhou2020 session empirical distributions define
finite pseudo-populations.  Outer samples are drawn independently from each
empirical source distribution and once from the empirical target distribution.
The CI-gate is then recalibrated inside every outer sample with a shared-target
bootstrap.  Both datasets supply simultaneous-coverage and retained-set
stability audits; Zhou2020 also supplies the bootstrap-budget audit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ci_gate_ma2020_riemann import airm2, riemann_estimates_and_bootstrap, shrink
from ci_gate_realdata_case_study import build_system, pair_gate, rect_gate
from path_config import resolve_cache_root
from plot_style import STATUS_COLORS, format_axis, setup_style


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BNCI_CACHE = ROOT / "simulation_results" / "bnci004_riemann" / "cov_cache"
DEFAULT_OUT = ROOT / "simulation_results" / "eeg_empirical_audit"


def parse_ints(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(value) for value in part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def load_subject(cache_root: Path, subject: int) -> tuple[list[int], list[np.ndarray], np.ndarray]:
    paths = sorted((cache_root / f"S{subject:02d}").glob("session-*_cov.npz"))
    if len(paths) not in {6, 7}:
        raise FileNotFoundError(f"expected six or seven cached sessions for subject {subject}")
    sessions: dict[int, np.ndarray] = {}
    for path in paths:
        session = int(path.stem.split("-")[1].split("_")[0])
        with np.load(path) as npz:
            sessions[session] = npz["covs"].astype(float)
    target_session = max(sessions)
    source_sessions = [session for session in sorted(sessions) if session < target_session]
    return source_sessions, [sessions[session] for session in source_sessions], sessions[target_session]


def load_bnci_subject(cache_root: Path, subject: int) -> tuple[list[int], list[np.ndarray], np.ndarray]:
    paths = sorted((cache_root / f"B{subject:02d}").glob("session-*_cov.npz"))
    if len(paths) != 5:
        raise FileNotFoundError(f"expected five cached BNCI sessions for subject {subject}")
    sessions: dict[int, np.ndarray] = {}
    for path in paths:
        session = int(path.stem.split("-")[1].split("_")[0])
        with np.load(path) as npz:
            sessions[session] = npz["covs"].astype(float)
    source_sessions = [1, 2, 3, 4]
    return source_sessions, [sessions[session] for session in source_sessions], sessions[5]


def discrepancy_vector(source_covs: list[np.ndarray], target_covs: np.ndarray, lam: float) -> np.ndarray:
    target_mean = shrink(target_covs.mean(axis=0), lam)
    return np.asarray(
        [airm2(shrink(source.mean(axis=0), lam), target_mean) for source in source_covs],
        dtype=float,
    )


def sample_empirical(covs: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    size = max(8, int(round(fraction * covs.shape[0])))
    return covs[rng.integers(0, covs.shape[0], size=size)]


def retained_set(system: dict[str, object], k: int) -> np.ndarray:
    rect = rect_gate(system["lower_c"], system["upper_c"])
    pair = pair_gate(k, system["jj"], system["lower_d"])
    return np.asarray(sorted(set(rect.tolist()) & set(pair.tolist())), dtype=int)


def evaluate_outer(
    reference: np.ndarray,
    source_covs: list[np.ndarray],
    target_covs: np.ndarray,
    rng: np.random.Generator,
    source_fraction: float,
    target_fraction: float,
    boot: int,
    lam: float,
    alpha: float,
    alpha_c: float,
    alpha_d: float,
) -> tuple[dict[str, float | int], np.ndarray]:
    sampled_sources = [sample_empirical(covs, source_fraction, rng) for covs in source_covs]
    sampled_target = sample_empirical(target_covs, target_fraction, rng)
    delta_hat, delta_boot = riemann_estimates_and_bootstrap(
        sampled_sources, sampled_target, rng, boot, lam, True
    )
    system = build_system(delta_hat, delta_boot, alpha_c, alpha_d, alpha)
    selected = retained_set(system, len(source_covs))

    oracle = np.flatnonzero(np.isclose(reference, reference.min(), atol=1e-12, rtol=0.0))
    true_d = reference[system["jj"]] - reference[system["ll"]]
    event_c = bool(np.all((reference >= system["lower_c"]) & (reference <= system["upper_c"])))
    event_d = bool(np.all(true_d >= system["lower_d"]))
    oracle_retained = set(oracle.tolist()).issubset(set(selected.tolist()))
    worst_excess = float(np.max(reference[selected] - reference.min())) if selected.size else np.nan
    return {
        "coverage_c": int(event_c),
        "coverage_d": int(event_d),
        "coverage_joint": int(event_c and event_d),
        "oracle_retained": int(oracle_retained),
        "exact_recovery": int(set(oracle.tolist()) == set(selected.tolist())),
        "set_size": int(selected.size),
        "retained_fraction": float(selected.size / len(source_covs)),
        "worst_reference_excess": worst_excess,
    }, selected


def mean_pairwise_jaccard(sets: list[np.ndarray]) -> float:
    values: list[float] = []
    for left, right in itertools.combinations(sets, 2):
        a, b = set(left.tolist()), set(right.tolist())
        values.append(len(a & b) / len(a | b) if (a | b) else 1.0)
    return float(np.mean(values)) if values else 1.0


def audit_subject(
    subject: int,
    cache_root: str,
    settings: dict[str, float | int],
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]], dict[str, float | int]]:
    source_sessions, source_covs, target_covs = load_subject(Path(cache_root), subject)
    reference = discrepancy_vector(source_covs, target_covs, float(settings["shrinkage"]))
    rows: list[dict[str, float | int]] = []
    inclusion_rows: list[dict[str, float | int]] = []
    selected_sets: list[np.ndarray] = []
    counts = np.zeros(len(source_covs), dtype=int)
    for replicate in range(int(settings["mc"])):
        rng = np.random.default_rng(int(settings["seed"]) + subject * 1_000_000 + replicate)
        metrics, selected = evaluate_outer(
            reference,
            source_covs,
            target_covs,
            rng,
            float(settings["source_fraction"]),
            float(settings["target_fraction"]),
            int(settings["boot"]),
            float(settings["shrinkage"]),
            float(settings["alpha"]),
            float(settings["alpha_c"]),
            float(settings["alpha_d"]),
        )
        metrics.update({"subject": subject, "replicate": replicate, "n_sources": len(source_covs)})
        rows.append(metrics)
        selected_sets.append(selected)
        counts[selected] += 1

    oracle = int(np.argmin(reference))
    for index, session in enumerate(source_sessions):
        inclusion_rows.append(
            {
                "subject": subject,
                "source_session": session,
                "reference_discrepancy": float(reference[index]),
                "reference_oracle": int(index == oracle),
                "inclusion_frequency": float(counts[index] / int(settings["mc"])),
            }
        )
    stability = {
        "subject": subject,
        "n_sources": len(source_covs),
        "mean_pairwise_jaccard": mean_pairwise_jaccard(selected_sets),
    }
    return rows, inclusion_rows, stability


def audit_bnci_subject(
    subject: int,
    cache_root: str,
    settings: dict[str, float | int],
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]], dict[str, float | int]]:
    source_sessions, source_covs, target_covs = load_bnci_subject(Path(cache_root), subject)
    reference = discrepancy_vector(source_covs, target_covs, float(settings["shrinkage"]))
    rows: list[dict[str, float | int]] = []
    inclusion_rows: list[dict[str, float | int]] = []
    selected_sets: list[np.ndarray] = []
    counts = np.zeros(len(source_covs), dtype=int)
    for replicate in range(int(settings["bnci_mc"])):
        rng = np.random.default_rng(int(settings["seed"]) + 700_000_000 + subject * 1_000_000 + replicate)
        metrics, selected = evaluate_outer(
            reference,
            source_covs,
            target_covs,
            rng,
            float(settings["source_fraction"]),
            float(settings["target_fraction"]),
            int(settings["boot"]),
            float(settings["shrinkage"]),
            float(settings["alpha"]),
            float(settings["alpha_c"]),
            float(settings["alpha_d"]),
        )
        metrics.update({"subject": subject, "replicate": replicate, "n_sources": len(source_covs)})
        rows.append(metrics)
        selected_sets.append(selected)
        counts[selected] += 1

    oracle = int(np.argmin(reference))
    for index, session in enumerate(source_sessions):
        inclusion_rows.append(
            {
                "subject": subject,
                "source_session": session,
                "reference_discrepancy": float(reference[index]),
                "reference_oracle": int(index == oracle),
                "inclusion_frequency": float(counts[index] / int(settings["bnci_mc"])),
            }
        )
    stability = {
        "subject": subject,
        "n_sources": len(source_covs),
        "mean_pairwise_jaccard": mean_pairwise_jaccard(selected_sets),
    }
    return rows, inclusion_rows, stability


def b_sensitivity_subject(
    subject: int,
    cache_root: str,
    settings: dict[str, float | int],
    b_grid: list[int],
) -> list[dict[str, float | int]]:
    _, source_covs, target_covs = load_subject(Path(cache_root), subject)
    reference = discrepancy_vector(source_covs, target_covs, float(settings["shrinkage"]))
    rows: list[dict[str, float | int]] = []
    for replicate in range(int(settings["b_mc"])):
        rng = np.random.default_rng(int(settings["seed"]) + 500_000_000 + subject * 100_000 + replicate)
        sampled_sources = [sample_empirical(covs, float(settings["source_fraction"]), rng) for covs in source_covs]
        sampled_target = sample_empirical(target_covs, float(settings["target_fraction"]), rng)
        delta_hat, delta_boot = riemann_estimates_and_bootstrap(
            sampled_sources,
            sampled_target,
            rng,
            max(b_grid),
            float(settings["shrinkage"]),
            True,
        )
        baseline: set[int] | None = None
        for boot in sorted(b_grid, reverse=True):
            system = build_system(
                delta_hat,
                delta_boot[:boot],
                float(settings["alpha_c"]),
                float(settings["alpha_d"]),
                float(settings["alpha"]),
            )
            selected = retained_set(system, len(source_covs))
            selected_as_set = set(selected.tolist())
            if baseline is None:
                baseline = selected_as_set
            oracle = int(np.argmin(reference))
            union = baseline | selected_as_set
            rows.append(
                {
                    "subject": subject,
                    "replicate": replicate,
                    "bootstrap_repetitions": boot,
                    "set_size": int(selected.size),
                    "oracle_retained": int(oracle in selected_as_set),
                    "jaccard_vs_max_b": len(baseline & selected_as_set) / len(union) if union else 1.0,
                }
            )
    return rows


def make_certificate_figure(
    subject: int,
    cache_root: Path,
    output: Path,
    seed: int,
    boot: int,
    lam: float,
    alpha: float,
    alpha_c: float,
    alpha_d: float,
) -> dict[str, object]:
    setup_style()
    source_sessions, source_covs, target_covs = load_subject(cache_root, subject)
    delta_hat, delta_boot = riemann_estimates_and_bootstrap(
        source_covs, target_covs, np.random.default_rng(seed), boot, lam, True
    )
    system = build_system(delta_hat, delta_boot, alpha_c, alpha_d, alpha)
    selected = retained_set(system, len(source_covs))
    selected_set = set(selected.tolist())
    threshold = float(np.min(system["upper_c"]))
    jj, ll = system["jj"], system["ll"]

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    y = np.arange(len(source_sessions))
    for index, session in enumerate(source_sessions):
        retained = index in selected_set
        color = STATUS_COLORS["Retained"] if retained else STATUS_COLORS["Excluded"]
        ax.errorbar(
            delta_hat[index],
            y[index],
            xerr=[[delta_hat[index] - system["lower_c"][index]], [system["upper_c"][index] - delta_hat[index]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            lw=1.4,
            ms=6,
        )
        if retained:
            label = "retained"
        else:
            mask = jj == index
            candidate_lowers = np.asarray(system["lower_d"])[mask]
            candidate_competitors = ll[mask]
            witness_pos = int(np.argmax(candidate_lowers))
            competitor = int(candidate_competitors[witness_pos])
            lower = float(candidate_lowers[witness_pos])
            label = f"excluded vs S{source_sessions[competitor]} (LB={lower:.3f})"
        ax.text(float(system["upper_c"][index]) + 0.025, y[index], label, va="center", fontsize=8.5, color=color)

    ax.axvline(threshold, color="0.25", linestyle="--", lw=1.2, label="minimum simultaneous upper limit")
    ax.set_yticks(y, [f"Source session {session}" for session in source_sessions])
    ax.invert_yaxis()
    ax.set_xlabel("Squared AIRM discrepancy to target session")
    ax.set_title(f"Zhou2020 subject {subject}: refinement exclusion certificates")
    format_axis(ax, ygrid=False)
    ax.grid(axis="x", alpha=0.35)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "subject": subject,
        "target_session": max(source_sessions) + 1,
        "source_sessions": source_sessions,
        "retained_sessions": [source_sessions[index] for index in selected],
        "bootstrap_repetitions": boot,
        "threshold": threshold,
    }


AUDIT_METRICS = (
    "coverage_c", "coverage_d", "coverage_joint", "oracle_retained",
    "exact_recovery", "set_size",
)


def summarize_empirical_audit(
    dataset: str,
    raw: pd.DataFrame,
    by_subject: pd.DataFrame,
    stability: pd.DataFrame,
    subjects: list[int],
    outer_per_subject: int,
) -> dict:
    """Validate released records before pooling equally replicated subjects."""
    required = {"subject", "replicate", "n_sources", *AUDIT_METRICS}
    if not required.issubset(raw.columns):
        raise ValueError(f"{dataset}: raw audit is missing required columns")
    expected_index = pd.MultiIndex.from_product(
        [sorted(subjects), range(outer_per_subject)], names=["subject", "replicate"]
    )
    observed_index = pd.MultiIndex.from_frame(raw[["subject", "replicate"]])
    if observed_index.has_duplicates or not observed_index.sort_values().equals(expected_index):
        raise ValueError(f"{dataset}: subjects or outer replicate IDs do not match the manifest")
    for column in AUDIT_METRICS[:-1]:
        if not raw[column].isin([0, 1]).all():
            raise ValueError(f"{dataset}: {column} must contain binary indicators")
    if not np.array_equal(
        raw["coverage_joint"], raw["coverage_c"] * raw["coverage_d"]
    ):
        raise ValueError(f"{dataset}: joint event differs from the event intersection")
    if not raw["set_size"].between(1, raw["n_sources"]).all():
        raise ValueError(f"{dataset}: retained-set size is outside the source count")
    recomputed = raw.groupby("subject").mean(numeric_only=True).sort_index()
    if "subject" not in by_subject or by_subject["subject"].duplicated().any():
        raise ValueError(f"{dataset}: invalid by-subject summary IDs")
    recorded = by_subject.set_index("subject").sort_index()
    if not recorded.index.equals(recomputed.index) or not set(recomputed).issubset(recorded):
        raise ValueError(f"{dataset}: by-subject summary is incomplete")
    if not np.allclose(
        recomputed.to_numpy(), recorded[recomputed.columns].to_numpy(), rtol=0, atol=1e-12
    ):
        raise ValueError(f"{dataset}: by-subject summary differs from raw records")
    if not {"subject", "n_sources", "mean_pairwise_jaccard"}.issubset(stability):
        raise ValueError(f"{dataset}: stability summary is incomplete")
    stable = stability.set_index("subject").sort_index()
    if stable.index.has_duplicates or not stable.index.equals(recomputed.index):
        raise ValueError(f"{dataset}: stability subject IDs differ from raw records")
    if not np.array_equal(stable["n_sources"], recomputed["n_sources"]):
        raise ValueError(f"{dataset}: stability source counts differ from raw records")
    if not stable["mean_pairwise_jaccard"].between(0, 1).all():
        raise ValueError(f"{dataset}: Jaccard summary is outside [0, 1]")
    return {
        "dataset": dataset,
        "subjects": len(subjects),
        "outer_per_subject": outer_per_subject,
        "outer_total": len(raw),
        **{metric: float(raw[metric].mean()) for metric in AUDIT_METRICS},
        "jaccard": float(stable["mean_pairwise_jaccard"].mean()),
    }


def write_empirical_coverage_table(out_dir: Path, paper_tables: Path) -> None:
    """Rebuild only the coverage table from both saved audits, without resampling."""
    manifest = json.loads((out_dir / "manifest.json").read_text())
    datasets = (
        ("Zhou2020", "", manifest["subjects"], manifest["outer_samples_per_subject"]),
        ("BNCI2014_004", "bnci_", list(range(1, 10)), manifest["bnci_outer_samples_per_subject"]),
    )
    rows = []
    for dataset, prefix, subjects, count in datasets:
        summary = summarize_empirical_audit(
            dataset,
            pd.read_csv(out_dir / f"{prefix}empirical_audit_raw.csv"),
            pd.read_csv(out_dir / f"{prefix}empirical_audit_by_subject.csv"),
            pd.read_csv(out_dir / f"{prefix}subject_stability.csv"),
            subjects,
            count,
        )
        name = dataset.replace("_", r"\_")
        rows.append(
            f"{name} & {summary['subjects']} & {summary['outer_total']} ({count}) & "
            + " & ".join(f"{summary[metric]:.3f}" for metric in AUDIT_METRICS[:-1])
            + f" & {summary['set_size']:.2f} & {summary['jaccard']:.3f} \\\\"
        )
    nominal_c = 100 * (1 - manifest["alpha_component"])
    nominal_d = 100 * (1 - manifest["alpha_contrast"])
    nominal_joint = 100 * (1 - manifest["alpha_component"] - manifest["alpha_contrast"])
    audit_tex = rf"""\begin{{table}}[t]
\centering
\caption{{Empirical-reference coverage and retained-set stability in both EEG cohorts. Full-session empirical distributions define the pseudo-population discrepancies. Outer draws use {100 * manifest['outer_source_sample_fraction']:g}\% of the observed source-session trial counts and {100 * manifest['outer_target_sample_fraction']:g}\% of the target-session trial count, with replacement. Outer-sample totals are followed by per-subject counts in parentheses. Every draw is recalibrated with a {manifest['bootstrap_repetitions']}-repetition shared-target bootstrap. The nominal component, contrast, and split joint coverage levels are {nominal_c:g}\%, {nominal_d:g}\%, and {nominal_joint:g}\%, respectively. Oracle retention and exact recovery refer to the empirical-reference argmin set. Set size is averaged over outer samples; Jaccard is the subject-averaged pairwise overlap of retained sets.}}
\label{{tab:eeg-empirical-audit}}
\WideTableBody
\begin{{tabular}}{{@{{}}lrrrrrrrrr@{{}}}}
\toprule
Dataset & Subjects & \makecell{{Outer\\samples}} & $\Pr(\mathcal E_C)$ & $\Pr(\mathcal E_D)$ & Joint & \makecell{{Oracle\\retained}} & Exact & \makecell{{Set\\size}} & Jaccard \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    paper_tables.mkdir(parents=True, exist_ok=True)
    (paper_tables / "table_eeg_empirical_audit.tex").write_text(audit_tex)


def write_tex_tables(out_dir: Path, paper_tables: Path, b_summary: pd.DataFrame) -> None:
    write_empirical_coverage_table(out_dir, paper_tables)
    rows = []
    for boot, group in b_summary.groupby("bootstrap_repetitions", sort=True):
        rows.append(
            f"{int(boot)} & {group['set_size'].mean():.2f} & {group['oracle_retained'].mean():.3f} & {group['jaccard_vs_max_b'].mean():.3f} \\\\"
        )
    b_tex = """\\begin{table}[t]
\\centering
\\caption{Bootstrap-repetition sensitivity in five Zhou2020 subjects over 20 common outer samples per subject.  Jaccard compares each retained set with the set obtained from the same outer sample using 999 bootstrap repetitions.}
\\label{tab:eeg-bootstrap-sensitivity}
\\TableBody
\\begin{tabular}{cccc}
\\toprule
Bootstrap repetitions & Set size & Empirical-oracle retained & Jaccard vs 999 \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    (paper_tables / "table_eeg_bootstrap_sensitivity.tex").write_text(b_tex)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=None,
                        help="Zhou covariance cache; otherwise resolved below EEG_DATA_ROOT.")
    parser.add_argument("--bnci-cache-root", type=Path, default=None,
                        help="BNCI covariance cache; defaults to the repository result cache.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--tables-only", action="store_true",
        help="Validate saved Zhou and BNCI audits and rebuild only the empirical-coverage table; do not rerun experiments.",
    )
    parser.add_argument("--paper-tables", type=Path, default=ROOT / "paper" / "tables")
    parser.add_argument("--subjects", default="1-20")
    parser.add_argument("--mc", type=int, default=50)
    parser.add_argument("--bnci-mc", type=int, default=200)
    parser.add_argument("--boot", type=int, default=499)
    parser.add_argument("--source-fraction", type=float, default=1.0)
    parser.add_argument("--target-fraction", type=float, default=0.5)
    parser.add_argument("--b-subjects", default="1,5,10,15,20")
    parser.add_argument("--b-mc", type=int, default=20)
    parser.add_argument("--b-grid", default="199,499,999")
    parser.add_argument("--certificate-subject", type=int, default=1)
    parser.add_argument("--certificate-boot", type=int, default=1999)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-c", type=float, default=0.025)
    parser.add_argument("--alpha-d", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.tables_only:
        write_empirical_coverage_table(args.out_dir, args.paper_tables)
        return
    args.cache_root = resolve_cache_root(
        args.cache_root,
        parser,
        env_suffix="CI-gate-cache/zhou2020",
        option_name="--cache-root",
    )
    args.bnci_cache_root = resolve_cache_root(
        args.bnci_cache_root,
        parser,
        env_suffix="CI-gate-cache/bnci2014_004",
        fallback=DEFAULT_BNCI_CACHE,
        option_name="--bnci-cache-root",
    )

    start = time.time()
    subjects = parse_ints(args.subjects)
    b_subjects = parse_ints(args.b_subjects)
    b_grid = parse_ints(args.b_grid)
    settings = {
        "mc": args.mc,
        "bnci_mc": args.bnci_mc,
        "boot": args.boot,
        "source_fraction": args.source_fraction,
        "target_fraction": args.target_fraction,
        "b_mc": args.b_mc,
        "shrinkage": args.shrinkage,
        "alpha": args.alpha,
        "alpha_c": args.alpha_c,
        "alpha_d": args.alpha_d,
        "seed": args.seed,
    }
    results = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(audit_subject)(subject, str(args.cache_root), settings) for subject in subjects
    )
    raw_rows = [row for subject_rows, _, _ in results for row in subject_rows]
    inclusion_rows = [row for _, subject_rows, _ in results for row in subject_rows]
    stability_rows = [row for _, _, row in results]

    bnci_results = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(audit_bnci_subject)(subject, str(args.bnci_cache_root), settings)
        for subject in range(1, 10)
    )
    bnci_raw_rows = [row for subject_rows, _, _ in bnci_results for row in subject_rows]
    bnci_inclusion_rows = [row for _, subject_rows, _ in bnci_results for row in subject_rows]
    bnci_stability_rows = [row for _, _, row in bnci_results]

    b_results = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(b_sensitivity_subject)(subject, str(args.cache_root), settings, b_grid)
        for subject in b_subjects
    )
    b_rows = [row for rows in b_results for row in rows]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(raw_rows)
    inclusion = pd.DataFrame(inclusion_rows)
    stability = pd.DataFrame(stability_rows)
    b_raw = pd.DataFrame(b_rows)
    bnci_raw = pd.DataFrame(bnci_raw_rows)
    bnci_inclusion = pd.DataFrame(bnci_inclusion_rows)
    bnci_stability = pd.DataFrame(bnci_stability_rows)
    raw.to_csv(args.out_dir / "empirical_audit_raw.csv", index=False)
    inclusion.to_csv(args.out_dir / "source_inclusion_frequency.csv", index=False)
    stability.to_csv(args.out_dir / "subject_stability.csv", index=False)
    b_raw.to_csv(args.out_dir / "bootstrap_sensitivity_raw.csv", index=False)
    bnci_raw.to_csv(args.out_dir / "bnci_empirical_audit_raw.csv", index=False)
    bnci_inclusion.to_csv(args.out_dir / "bnci_source_inclusion_frequency.csv", index=False)
    bnci_stability.to_csv(args.out_dir / "bnci_subject_stability.csv", index=False)
    raw.groupby("subject", as_index=False).mean(numeric_only=True).to_csv(
        args.out_dir / "empirical_audit_by_subject.csv", index=False
    )
    b_raw.groupby("bootstrap_repetitions", as_index=False).mean(numeric_only=True).to_csv(
        args.out_dir / "bootstrap_sensitivity_summary.csv", index=False
    )
    bnci_raw.groupby("subject", as_index=False).mean(numeric_only=True).to_csv(
        args.out_dir / "bnci_empirical_audit_by_subject.csv", index=False
    )

    certificate = make_certificate_figure(
        args.certificate_subject,
        args.cache_root,
        ROOT / "paper" / "figures" / "fig_zhou2020_exclusion_certificate.pdf",
        args.seed + 900_000_000,
        args.certificate_boot,
        args.shrinkage,
        args.alpha,
        args.alpha_c,
        args.alpha_d,
    )
    manifest = {
        "script": Path(__file__).name,
        "zhou_cache_root": str(args.cache_root),
        "bnci_cache_root": str(args.bnci_cache_root),
        "out_dir": str(args.out_dir),
        "pseudo_populations": [
            "full-session BNCI2014_004 empirical covariance distributions",
            "full-session Zhou2020 empirical covariance distributions",
        ],
        "subjects": subjects,
        "outer_samples_per_subject": args.mc,
        "bnci_outer_samples_per_subject": args.bnci_mc,
        "outer_source_sample_fraction": args.source_fraction,
        "outer_target_sample_fraction": args.target_fraction,
        "bootstrap_repetitions": args.boot,
        "bootstrap_sensitivity_subjects": b_subjects,
        "bootstrap_sensitivity_outer_samples": args.b_mc,
        "bootstrap_grid": b_grid,
        "alpha": args.alpha,
        "alpha_component": args.alpha_c,
        "alpha_contrast": args.alpha_d,
        "shrinkage": args.shrinkage,
        "seed": args.seed,
        "certificate": certificate,
        "runtime_seconds": round(time.time() - start, 1),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_tex_tables(args.out_dir, args.paper_tables, b_raw)


if __name__ == "__main__":
    main()
