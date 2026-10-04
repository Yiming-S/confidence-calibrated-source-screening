#!/usr/bin/env python3
"""Subject-level paired non-inferiority analysis of the Pair/Ref downstream
accuracy against all-source pooling, from the released subject-level outputs.

For each subject we use the released mean balanced accuracy for Pair/Ref and
all-source pooling, form the paired difference, and test the non-inferiority
null

    H0: mean(Pair/Ref - All) <= -delta_NI    (inferior)

against the one-sided alternative at margin delta_NI.  The two-percentage-point
margin was fixed in the initial Ma2020/Stieger2021 analysis and applied unchanged
to BNCI2014_004 and exploratory Kumar2024. We report the paired mean difference with a
two-sided 95% confidence interval for descriptive effect uncertainty (subject as
the unit; repeated splits are averaged within subject, not treated as
independent).  The primary decision uses the 95% one-sided lower confidence
bound, or equivalently the one-sided paired t-test at alpha=0.05.  The one-sided
Wilcoxon signed-rank test is a sensitivity analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_OUT_DIR = ROOT / "build" / "noninferiority"


def analyze(
    subject_csv: Path,
    delta_ni: float,
    analysis: str,
    decision_alpha: float,
) -> dict:
    df = pd.read_csv(subject_csv)
    required = {"subject", "method", "balanced_accuracy"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{subject_csv} is missing required columns: {', '.join(missing)}")
    if "analysis" in df.columns:
        df = df[df["analysis"] == analysis]
        if df.empty:
            raise ValueError(f"{subject_csv} contains no rows with analysis={analysis!r}")
    per = df.groupby(["subject", "method"])["balanced_accuracy"].mean().unstack("method")
    required_methods = {"pair_ref", "all_sources"}
    missing_methods = sorted(required_methods - set(per.columns))
    if missing_methods:
        raise ValueError(
            f"{subject_csv} is missing required methods: {', '.join(missing_methods)}"
        )
    d = (per["pair_ref"] - per["all_sources"]).dropna().to_numpy()
    n = d.size
    if n < 2:
        raise ValueError(f"{subject_csv} has fewer than two complete subject pairs")
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    tcrit_two_sided = stats.t.ppf(0.975, n - 1)
    ci = (
        mean - tcrit_two_sided * se,
        mean + tcrit_two_sided * se,
    )
    one_sided_lcb95 = float(
        mean - stats.t.ppf(1.0 - decision_alpha, n - 1) * se
    )
    # one-sided non-inferiority: t = (mean - (-delta)) / se, H1 mean > -delta
    t_ni = (mean + delta_ni) / se
    p_ni = float(stats.t.sf(t_ni, n - 1))
    try:
        w_p = float(stats.wilcoxon(d + delta_ni, alternative="greater").pvalue)
    except ValueError:
        w_p = float("nan")
    return {
        "n_subjects": int(n),
        "mean_diff": mean,
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "one_sided_lcb95": one_sided_lcb95,
        "delta_NI": delta_ni,
        "noninferior_at_delta": bool(one_sided_lcb95 > -delta_ni),
        "p_noninferiority_ttest": p_ni,
        "p_noninferiority_wilcoxon": w_p,
        "frac_subjects_within_margin": float((d > -delta_ni).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delta-ni", type=float, default=0.02)
    p.add_argument(
        "--decision-alpha",
        type=float,
        default=0.05,
        help="one-sided type-I error used for the non-inferiority decision",
    )
    p.add_argument(
        "--margin-grid",
        type=str,
        default="0.01,0.02,0.03",
        help="comma-separated non-inferiority margins for sensitivity analysis",
    )
    p.add_argument(
        "--analysis",
        default="primary",
        help="analysis label to use when an input contains an 'analysis' column",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS,
        help="root of the released results tree",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    if not 0.0 < args.decision_alpha < 1.0:
        p.error("--decision-alpha must lie strictly between 0 and 1")
    datasets = {
        name: args.results_root / "eeg" / name / "downstream_by_subject.csv"
        for name in ("ma2020", "stieger2021", "kumar2024", "bnci2014_004")
    }
    missing_inputs = [str(path) for path in datasets.values() if not path.is_file()]
    if missing_inputs:
        p.error("missing subject-level input(s): " + ", ".join(missing_inputs))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    sensitivity_path = args.out_dir / "margin_sensitivity.csv"
    manifest_path = args.out_dir / "manifest.json"
    out = {}
    sensitivity = []
    margins = [float(value.strip()) for value in args.margin_grid.split(",") if value.strip()]
    if args.delta_ni not in margins:
        margins.append(args.delta_ni)
    margins = sorted(set(margins))
    for name, path in datasets.items():
        out[name] = analyze(
            path,
            args.delta_ni,
            args.analysis,
            args.decision_alpha,
        )
        if name == "kumar2024":
            out[name].pop("p_noninferiority_wilcoxon")
            out[name]["analysis_role"] = "exploratory"
        print(f"=== {name} ===")
        print(json.dumps(out[name], indent=2))
        for margin in margins:
            if name == "kumar2024" and margin not in (0.01, 0.02):
                continue
            sensitivity.append(
                {
                    "dataset": name,
                    **analyze(
                        path,
                        margin,
                        args.analysis,
                        args.decision_alpha,
                    ),
                }
            )
            if name == "kumar2024":
                sensitivity[-1].pop("p_noninferiority_wilcoxon")
    summary_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(sensitivity).to_csv(sensitivity_path, index=False)
    manifest = {
        "analysis_unit": "subject",
        "split_handling": "balanced accuracy is averaged over target splits within subject",
        "contrast": "Pair/Ref minus all-source pooling",
        "primary_margin": args.delta_ni,
        "margin_sensitivity": margins,
        "primary_test": "one-sided paired t non-inferiority test",
        "sensitivity_test": "one-sided Wilcoxon signed-rank non-inferiority test",
        "effect_interval": "two-sided 95% t interval for descriptive reporting",
        "decision_bound": "95% one-sided t lower confidence bound for the subject-level mean contrast",
        "noninferior_at_delta_rule": "the one-sided 95% lower confidence bound exceeds minus the margin",
        "decision_alpha": args.decision_alpha,
        "analysis_filter": args.analysis,
        "margin_interpretation": (
            "a 0.02 absolute balanced-accuracy decrease is the largest accepted "
            "performance cost for the source-reduction comparison; it is an "
            "operational tolerance, not a clinical threshold"
        ),
        "primary_margin_timing": (
            "the 0.02 margin was used for the Ma2020/Stieger2021 analysis "
            "and carried forward unchanged to BNCI2014_004 and exploratory Kumar2024"
        ),
        "evidence_roles": {
            "stieger2021": "largest-cohort primary EEG evidence",
            "kumar2024": "exploratory cohort added after the original analyses",
            "bnci2014_004": "secondary nine-subject three-channel check",
            "ma2020": "initial longitudinal analysis; inconclusive at the primary margin",
        },
        "kumar_planned_tests": "paired t tests at 0.01 and 0.02 only; no Wilcoxon or 0.03 test",
        "inputs": {
            name: str(path.relative_to(ROOT)) for name, path in datasets.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print(f"wrote {sensitivity_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
