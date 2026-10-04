"""Render completed BSPC revision summaries as manuscript tables.

The inputs are independent experiment outputs. This renderer does not merge
their estimates into the original primary non-inferiority analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "eeg" / "bspc_revision_20261002"
TABLES = ROOT / "build" / "tables"
DATASETS = {"ma2020": "Ma2020", "stieger2021": "Stieger2021", "kumar2024": "Kumar2024"}
METHODS = {
    "all_sources": "All sources", "refinement": "Refinement",
    "mcs_style": "MCS-style", "mcs_stepdown": "MCS-style",
    "top_m_ref": r"Top-$m$", "range_midpoint": "Range midpoint",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def completed(folder: Path) -> None:
    """Require the path-free metadata packaged with completed experiments."""
    manifest = json.loads((RESULTS / "revision_manifest.json").read_text())
    key = f"mcs_{folder.name}" if folder.parent.name == "mcs" else folder.name
    if key not in manifest["analyses"]:
        raise ValueError(f"Missing compact experiment metadata: {key}")


def table(name: str, label: str, caption: str, columns: str,
          header: str, body: list[str]) -> None:
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", r"\TableBody", f"\\begin{{tabular}}{{{columns}}}",
             r"\toprule", header + r" \\", r"\midrule", *body,
             r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / name).write_text("\n".join(lines))


def mcs_tables() -> None:
    body, intervals = [], []
    for dataset in ("ma2020", "stieger2021"):
        folder = RESULTS / "mcs" / dataset
        completed(folder)
        indexed = {r["method"]: r for r in rows(folder / "method_summary.csv")}
        expected_subjects = 25 if dataset == "ma2020" else 62
        for i, method in enumerate(("all_sources", "refinement", "mcs_style", "top_m_ref", "range_midpoint")):
            row = indexed[method]
            if int(row["subjects"]) != expected_subjects:
                raise ValueError(f"Unexpected subject count: {dataset}/{method}")
            body.append(" & ".join([
                DATASETS[dataset] if i == 0 else "", METHODS[method],
                f"{float(row['mean_set_size']):.2f}",
                f"{float(row['mean_training_trials']):.1f}",
                f"{float(row['mean_balanced_accuracy']):.4f}",
            ]) + r" \\")
        contrasts = {(r["method_a"], r["method_b"]): r
                     for r in rows(folder / "paired_accuracy_ci.csv")}
        for i, method in enumerate(("mcs_style", "top_m_ref", "range_midpoint")):
            row = contrasts[("refinement", method)]
            intervals.append(" & ".join([
                DATASETS[dataset] if i == 0 else "", f"Ref. $-$ {METHODS[method]}",
                f"{100*float(row['mean_difference']):+.2f}",
                f"$[{100*float(row['ci_lower']):+.2f},{100*float(row['ci_upper']):+.2f}]$",
            ]) + r" \\")
        body.append(r"\midrule")
        intervals.append(r"\midrule")
    table("table_bspc_mcs.tex", "tab:bspc-mcs",
          "Shared-draw comparison with the MCS-style source-screening analogue. "
          "Means first average 30 target half-splits within subject, then 25 Ma2020 "
          "or 62 Stieger2021 subjects. Both calibrated screens use the same 199 "
          "bootstrap draws on each split. Top-$m$ matches the refinement count. "
          "Training trials count only retained historical trials. This independent "
          "comparison does not replace the primary non-inferiority runs.",
          "@{}llrrr@{}", r"Dataset & Method & Sessions & Train trials & Bal.\ acc.", body[:-1])
    table("table_bspc_paired_comparisons.tex", "tab:bspc-paired",
          "Refinement minus comparator balanced accuracy, in percentage points. "
          "Intervals are two-sided, pointwise 95\\% paired $t$ intervals across "
          "subject-level split averages. They describe precision; they are not "
          "simultaneous intervals or equivalence tests.",
          "@{}llrr@{}", r"Dataset & Comparison & Difference (pp) & 95\% CI (pp)", intervals[:-1])


def runtime_table() -> None:
    folder = RESULTS / "runtime"
    completed(folder)
    summary = {(r["dataset"], r["method"]): r for r in rows(folder / "runtime_summary.csv")}
    body = []
    for dataset in ("ma2020", "stieger2021"):
        for i, method in enumerate(("all_sources", "refinement", "mcs_stepdown")):
            row = summary[(dataset, method)]
            body.append(" & ".join([
                DATASETS[dataset] if i == 0 else "", METHODS[method],
                f"{100*float(row['mean_training_trial_fraction']):.1f}\\%",
                *[f"{float(row['mean_subject_median_' + field + '_seconds']):.2f}"
                  for field in ("screen", "fit", "screen_fit_predict")],
            ]) + r" \\")
        body.append(r"\midrule")
    table("table_bspc_runtime.tex", "tab:bspc-runtime",
          "Serial timing from cached trial covariances, in seconds. Subjects 1--3 "
          "per dataset use their first target half-split. Values average each "
          "subject's median of three repetitions; training fractions average "
          "within-subject fractions. Total includes screening, model fitting, "
          "and held-out prediction. Raw-EEG preprocessing and cache loading are "
          "excluded. Each method independently recomputes all required screening "
          "and fitting operations with one numerical thread and no fitted-model cache.",
          "@{}llrrrr@{}", r"Dataset & Method & Train fraction & Screen (s) & Fit (s) & Total (s)", body[:-1])


def budget_tables() -> None:
    from make_current_publication_artifacts import budget_tables as current_budget_tables
    current_budget_tables(TABLES)
    return


def main() -> None:
    global RESULTS, TABLES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("mcs", "runtime", "budget", "all"), default="all")
    parser.add_argument("--results-root", type=Path, default=RESULTS,
                        help="Compact revision input directory.")
    parser.add_argument("--out-dir", type=Path, default=TABLES,
                        help="Directory for generated LaTeX tables.")
    args = parser.parse_args()
    RESULTS = args.results_root.expanduser().resolve()
    TABLES = args.out_dir.expanduser().resolve()
    if TABLES == RESULTS or TABLES.is_relative_to(RESULTS):
        parser.error("output tables must remain outside the released input directory")
    TABLES.mkdir(parents=True, exist_ok=True)
    if args.part in {"mcs", "all"}:
        mcs_tables()
    if args.part in {"runtime", "all"}:
        runtime_table()
    if args.part in {"budget", "all"}:
        budget_tables()


if __name__ == "__main__":
    main()
