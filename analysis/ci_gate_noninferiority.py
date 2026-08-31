#!/usr/bin/env python3
"""Subject-level paired non-inferiority analysis of the Pair/Ref downstream
accuracy against all-source pooling, from the released subject-level outputs.

For each subject we use the released mean balanced accuracy for Pair/Ref and
all-source pooling, form the paired difference, and test the non-inferiority
null

    H0: mean(Pair/Ref - All) <= -delta_NI    (inferior)

against the one-sided alternative, at a pre-specified margin delta_NI.  We
report the paired mean difference with a 95% confidence interval (subject as the
unit; repeated splits are averaged within subject, not treated as independent),
the one-sided non-inferiority p-value, and the Wilcoxon signed-rank p-value.
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
DEFAULT_OUT_DIR = RESULTS / "eeg" / "noninferiority"


def analyze(subject_csv: Path, delta_ni: float, analysis: str) -> dict:
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
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (mean - tcrit * se, mean + tcrit * se)
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
        "delta_NI": delta_ni,
        "noninferior_at_delta": bool(ci[0] > -delta_ni),
        "p_noninferiority_ttest": p_ni,
        "p_noninferiority_wilcoxon": w_p,
        "frac_subjects_within_margin": float((d > -delta_ni).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delta-ni", type=float, default=0.02)
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
    datasets = {
        name: args.results_root / "eeg" / name / "downstream_by_subject.csv"
        for name in ("ma2020", "stieger2021", "zhou2020")
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
        out[name] = analyze(path, args.delta_ni, args.analysis)
        print(f"=== {name} ===")
        print(json.dumps(out[name], indent=2))
        for margin in margins:
            sensitivity.append(
                {"dataset": name, **analyze(path, margin, args.analysis)}
            )
    summary_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(sensitivity).to_csv(sensitivity_path, index=False)
    manifest = {
        "analysis_unit": "subject",
        "split_handling": "balanced accuracy is averaged over target splits within subject",
        "contrast": "Pair/Ref minus all-source pooling",
        "primary_margin": args.delta_ni,
        "margin_sensitivity": margins,
        "tests": [
            "one-sided paired t non-inferiority test",
            "one-sided Wilcoxon signed-rank non-inferiority test",
        ],
        "confidence_interval": "two-sided 95% t interval for the subject-level mean contrast",
        "noninferior_at_delta_rule": "the two-sided 95% confidence-interval lower endpoint exceeds minus the margin",
        "decision_alpha": 0.05,
        "analysis_filter": args.analysis,
        "primary_margin_timing": (
            "the 0.02 margin was used for the Ma2020/Stieger2021 analysis "
            "and carried forward unchanged to Zhou2020"
        ),
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
