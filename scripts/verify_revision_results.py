#!/usr/bin/env python3
"""Recompute revision summaries using only the compact public result tables.

The released subject summaries are the inputs to this check. Predictions,
individual bootstrap draws, raw covariance caches, split-level results and
individual timing repetitions are not part of this compact release and are
not validated here. The standard-error diagnostic table receives schema and
internal-count checks, not a reconstruction of its omitted coordinates.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "eeg" / "bspc_revision_20261002"
MCS_METHODS = ("refinement", "mcs_style", "top_m_ref", "range_midpoint", "all_sources")
RUNTIME_METHODS = ("all_sources", "refinement", "mcs_stepdown")
BUDGET_FIELDS = (
    "set_size", "jaccard_vs_999", "same_members_vs_999", "size_change_vs_999",
    "absolute_size_change_vs_999", "members_added_vs_999", "members_removed_vs_999",
    "q_component", "q_contrast", "q_component_change_vs_999",
    "q_contrast_change_vs_999", "q_component_abs_change_vs_999", "q_contrast_abs_change_vs_999",
)
SEED_FIELDS = (
    "pairwise_seed_jaccard", "pairwise_seed_exact", "set_size_seed_sd",
    "set_size_seed_range", "q_component_seed_sd", "q_contrast_seed_sd",
)
TIME_FIELDS = (
    "screen_seconds", "fit_seconds", "predict_seconds", "screen_fit_seconds",
    "screen_fit_predict_seconds",
)
RUNTIME_MEANS = (
    "candidate_sessions", "retained_sessions", "candidate_training_trials",
    "training_trials", "training_trial_fraction", "balanced_accuracy",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: object, expected: object, context: str) -> None:
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(left.shape == right.shape, f"{context}: shape mismatch")
    require(np.isfinite(left).all() and np.isfinite(right).all(), f"{context}: non-finite values")
    require(np.allclose(left, right, rtol=1e-9, atol=1e-11), f"{context}: values do not match recomputation")


def read_table(path: Path, columns: set[str], keys: list[str]) -> pd.DataFrame:
    table = pd.read_csv(path)
    require(columns <= set(table.columns), f"{path.name}: missing columns {sorted(columns - set(table.columns))}")
    require(not table.empty, f"{path.name}: empty table")
    require(not table[list(columns)].isna().any().any(), f"{path.name}: missing values")
    require(not table.duplicated(keys).any(), f"{path.name}: duplicate keys {keys}")
    return table


def finite_fields(table: pd.DataFrame, columns: tuple[str, ...] | list[str], context: str) -> None:
    for column in columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
        require(np.isfinite(values).all(), f"{context}: non-finite {column}")


def probabilities(table: pd.DataFrame, columns: tuple[str, ...], context: str) -> None:
    finite_fields(table, columns, context)
    for column in columns:
        require(table[column].between(0, 1).all(), f"{context}: {column} outside [0, 1]")


def verify_mcs(input_dir: Path, dataset: str, manifest: dict) -> dict:
    folder = input_dir / "mcs" / dataset
    expected_subjects = set(manifest["analyses"][f"mcs_{dataset}"]["subjects"])
    subject_columns = {
        "dataset", "subject", "method", "mean_set_size", "mean_training_trials",
        "mean_balanced_accuracy", "sd_balanced_accuracy_across_splits",
        "accuracy_diff_vs_all", "accuracy_diff_vs_refinement",
    }
    subjects = read_table(folder / "subject_summary.csv", subject_columns, ["subject", "method"])
    require(set(subjects.dataset) == {dataset}, f"{dataset}: wrong dataset name")
    require(set(subjects.method) == set(MCS_METHODS), f"{dataset}: wrong method inventory")
    require(set(subjects.subject) == expected_subjects, f"{dataset}: wrong subject inventory")
    require(len(subjects) == len(expected_subjects) * len(MCS_METHODS), f"{dataset}: incomplete subject-method grid")
    numeric = tuple(subject_columns - {"dataset", "method", "subject"})
    finite_fields(subjects, numeric, dataset)
    probabilities(subjects, ("mean_balanced_accuracy",), dataset)
    require((subjects.mean_set_size >= 1).all() and (subjects.mean_training_trials > 0).all(),
            f"{dataset}: invalid source or trial counts")
    require((subjects.sd_balanced_accuracy_across_splits >= 0).all(), f"{dataset}: negative split SD")
    pivot_accuracy = subjects.pivot(index="subject", columns="method", values="mean_balanced_accuracy")
    for row in subjects.itertuples(index=False):
        accuracy = pivot_accuracy.loc[row.subject]
        close(row.accuracy_diff_vs_all, accuracy[row.method] - accuracy["all_sources"],
              f"{dataset}/{row.subject}/{row.method}: difference vs all")
        close(row.accuracy_diff_vs_refinement, accuracy[row.method] - accuracy["refinement"],
              f"{dataset}/{row.subject}/{row.method}: difference vs refinement")
    sizes = subjects.pivot(index="subject", columns="method", values="mean_set_size")
    close(sizes.top_m_ref, sizes.refinement, f"{dataset}: Top-m mean source counts")

    aggregates = {
        "mean_set_size": ("mean_set_size", "mean"), "sd_set_size": ("mean_set_size", "std"),
        "mean_training_trials": ("mean_training_trials", "mean"),
        "sd_training_trials": ("mean_training_trials", "std"),
        "mean_balanced_accuracy": ("mean_balanced_accuracy", "mean"),
        "sd_balanced_accuracy": ("mean_balanced_accuracy", "std"),
        "mean_accuracy_diff_vs_all": ("accuracy_diff_vs_all", "mean"),
        "mean_accuracy_diff_vs_refinement": ("accuracy_diff_vs_refinement", "mean"),
    }
    methods = read_table(folder / "method_summary.csv", {"dataset", "method", "subjects", *aggregates}, ["method"])
    require(set(methods.method) == set(MCS_METHODS) and set(methods.dataset) == {dataset},
            f"{dataset}: wrong method-summary inventory")
    for row in methods.itertuples(index=False):
        block = subjects[subjects.method == row.method]
        close(row.subjects, len(block), f"{dataset}/{row.method}: subject count")
        for output, (source, operation) in aggregates.items():
            values = block[source].to_numpy(float)
            expected = values.mean() if operation == "mean" else values.std(ddof=1)
            close(getattr(row, output), expected, f"{dataset}/{row.method}: {output}")

    expected_pairs = set(itertools.combinations(MCS_METHODS, 2))
    ci_count = 0
    metrics = (
        ("paired_accuracy_ci.csv", "balanced_accuracy", "mean_balanced_accuracy"),
        ("paired_set_size_ci.csv", "retained_sessions", "mean_set_size"),
        ("paired_training_trials_ci.csv", "training_trials", "mean_training_trials"),
    )
    for filename, metric, source in metrics:
        columns = {"metric", "method_a", "method_b", "contrast", "subjects", "mean_difference",
                   "sd_difference", "standard_error", "ci_level", "ci_lower", "ci_upper"}
        reported = read_table(folder / filename, columns, ["method_a", "method_b"])
        require(set(zip(reported.method_a, reported.method_b)) == expected_pairs,
                f"{dataset}/{filename}: missing or unexpected paired contrasts")
        require(set(reported.metric) == {metric}, f"{dataset}/{filename}: wrong metric")
        values = subjects.pivot(index="subject", columns="method", values=source).sort_index()
        for row in reported.itertuples(index=False):
            context = f"{dataset}/{filename}/{row.method_a}-{row.method_b}"
            require(row.contrast == f"{row.method_a} - {row.method_b}", f"{context}: incorrect label")
            close(row.ci_level, 0.95, f"{context}: confidence level")
            differences = (values[row.method_a] - values[row.method_b]).to_numpy(float)
            n = differences.size
            mean = float(differences.mean())
            sd = float(differences.std(ddof=1))
            se = sd / np.sqrt(n)
            half_width = float(stats.t.ppf(0.975, n - 1)) * se
            expected = (n, mean, sd, se, mean - half_width, mean + half_width)
            actual = (row.subjects, row.mean_difference, row.sd_difference,
                      row.standard_error, row.ci_lower, row.ci_upper)
            close(actual, expected, context)
            ci_count += 1
    return {"subjects": len(expected_subjects), "subject_rows": len(subjects),
            "method_rows": len(methods), "paired_intervals_recomputed": ci_count}


def verify_budget(input_dir: Path, manifest: dict) -> dict:
    folder = input_dir / "bootstrap_budget"
    plan = manifest["analyses"]["bootstrap_budget"]
    expected_grid = {(dataset, subject, budget) for dataset, ids in plan["subjects"].items()
                     for subject in ids for budget in plan["budgets"]}
    keys = ["dataset", "subject", "budget"]
    subjects = read_table(folder / "subject_summary.csv", {*keys, *BUDGET_FIELDS}, keys)
    seeds = read_table(folder / "seed_variability_subjects.csv", {*keys, *SEED_FIELDS}, keys)
    for table, name, fields in ((subjects, "budget subjects", BUDGET_FIELDS),
                                (seeds, "seed-variability subjects", SEED_FIELDS)):
        require(set(table[keys].itertuples(index=False, name=None)) == expected_grid,
                f"{name}: wrong subject-budget inventory")
        finite_fields(table, fields, name)
    probabilities(subjects, ("jaccard_vs_999", "same_members_vs_999"), "budget subjects")
    probabilities(seeds, ("pairwise_seed_jaccard", "pairwise_seed_exact"), "seed subjects")
    close(subjects.size_change_vs_999,
          subjects.members_added_vs_999 - subjects.members_removed_vs_999,
          "budget subjects: signed membership change")
    require((subjects.absolute_size_change_vs_999 + 1e-12 >= subjects.size_change_vs_999.abs()).all(),
            "budget subjects: inconsistent absolute size change")
    for column in ("q_component", "q_contrast", "set_size", "absolute_size_change_vs_999",
                   "members_added_vs_999", "members_removed_vs_999",
                   "q_component_abs_change_vs_999", "q_contrast_abs_change_vs_999"):
        require((subjects[column] >= 0).all(), f"budget subjects: negative {column}")
    for column in SEED_FIELDS[2:]:
        require((seeds[column] >= 0).all(), f"seed subjects: negative {column}")
    reference = subjects[subjects.budget == 999]
    close(reference[["jaccard_vs_999", "same_members_vs_999"]], np.ones((len(reference), 2)),
          "budget 999: self-comparison")
    change_columns = [name for name in BUDGET_FIELDS if "change_vs_999" in name]
    change_columns += ["members_added_vs_999", "members_removed_vs_999"]
    close(reference[change_columns], np.zeros((len(reference), len(change_columns))),
          "budget 999: self-change")

    summary_rows = 0
    summaries = {}
    for lower, fields, filename in (
        (subjects, BUDGET_FIELDS, "dataset_summary.csv"),
        (seeds, SEED_FIELDS, "seed_variability_summary.csv"),
    ):
        reported = read_table(folder / filename, {"dataset", "budget", "n_subjects", *fields},
                              ["dataset", "budget"])
        expected_keys = {(dataset, budget) for dataset in plan["subjects"] for budget in plan["budgets"]}
        require(set(zip(reported.dataset, reported.budget)) == expected_keys,
                f"{filename}: wrong dataset-budget inventory")
        for row in reported.itertuples(index=False):
            block = lower[(lower.dataset == row.dataset) & (lower.budget == row.budget)]
            close(row.n_subjects, len(block), f"{filename}/{row.dataset}/{row.budget}: subjects")
            for column in fields:
                close(getattr(row, column), block[column].to_numpy(float).mean(),
                      f"{filename}/{row.dataset}/{row.budget}: {column}")
        summaries[filename] = reported
        summary_rows += len(reported)

    # Only the aggregated SE diagnostics are released.  Check their schema and
    # count consistency; do not imply reconstruction of the underlying errors.
    count_columns = (
        "n_subjects", "n_seed_sample_cases", "n_component_coordinates", "n_contrast_coordinates",
        "changed_set_cases_vs_999", "maximum_absolute_size_change_vs_999",
        "raw_component_se_nonfinite_count", "raw_component_se_floor_count",
        "raw_contrast_se_nonfinite_count", "raw_contrast_se_floor_count",
    )
    bounds = ("raw_component_se_min", "raw_component_se_max", "raw_contrast_se_min", "raw_contrast_se_max")
    diagnostics = read_table(folder / "se_diagnostics_summary.csv",
                             {"dataset", "budget", "minimum_jaccard_vs_999", *count_columns, *bounds},
                             ["dataset", "budget"])
    require(set(zip(diagnostics.dataset, diagnostics.budget)) == expected_keys,
            "SE diagnostics: wrong dataset-budget inventory")
    finite_fields(diagnostics, (*count_columns, *bounds), "SE diagnostics")
    probabilities(diagnostics, ("minimum_jaccard_vs_999",), "SE diagnostics")
    for column in count_columns:
        require((diagnostics[column] >= 0).all(), f"SE diagnostics: negative {column}")
        close(diagnostics[column], np.round(diagnostics[column]), f"SE diagnostics: integer {column}")
    for prefix in ("raw_component_se", "raw_contrast_se"):
        require(((diagnostics[f"{prefix}_min"] >= 0) &
                 (diagnostics[f"{prefix}_max"] >= diagnostics[f"{prefix}_min"])).all(),
                f"SE diagnostics: invalid {prefix} range")
    dataset_summary = summaries["dataset_summary.csv"].set_index(["dataset", "budget"])
    for row in diagnostics.itertuples(index=False):
        count = len(plan["subjects"][row.dataset])
        cases = count * plan["splits_per_subject"] * len(plan["bootstrap_seeds"])
        close((row.n_subjects, row.n_seed_sample_cases), (count, cases), "SE diagnostic counts")
        same = dataset_summary.loc[(row.dataset, row.budget), "same_members_vs_999"]
        close(row.changed_set_cases_vs_999, (1 - same) * cases, "SE diagnostic changed-set count")
    audit = plan["audit"]
    close(audit["minimum_jaccard_vs_999"], diagnostics.minimum_jaccard_vs_999.min(),
          "budget audit: minimum Jaccard")
    close(audit["maximum_absolute_size_change_vs_999"], diagnostics.maximum_absolute_size_change_vs_999.max(),
          "budget audit: maximum size change")
    return {"subject_rows": len(subjects), "seed_subject_rows": len(seeds),
            "dataset_and_seed_summary_rows_recomputed": summary_rows,
            "se_diagnostic_rows_schema_and_count_checked": len(diagnostics)}


def verify_runtime(input_dir: Path, manifest: dict) -> dict:
    folder = input_dir / "runtime"
    plan = manifest["analyses"]["runtime"]["protocol"]
    ids = plan["subjects_each_dataset"]
    datasets = set(plan["base_seeds"])
    time_columns = tuple(f"{field}_{statistic}" for field in TIME_FIELDS
                         for statistic in ("median", "mean", "min", "max"))
    columns = {"dataset", "subject", "method", "timing_repetitions", "selected_sessions",
               "screen_trials", "test_trials", "channels", "tangent_features", *RUNTIME_MEANS, *time_columns}
    subjects = read_table(folder / "runtime_subject_summary.csv", columns, ["dataset", "subject", "method"])
    expected_grid = {(dataset, subject, method) for dataset in datasets for subject in ids for method in RUNTIME_METHODS}
    require(set(subjects[["dataset", "subject", "method"]].itertuples(index=False, name=None)) == expected_grid,
            "runtime: incomplete subject-method grid")
    finite_fields(subjects, (*RUNTIME_MEANS, *time_columns), "runtime subjects")
    probabilities(subjects, ("training_trial_fraction", "balanced_accuracy"), "runtime subjects")
    close(subjects.timing_repetitions, np.full(len(subjects), plan["repetitions"]), "runtime repetition count")
    require((subjects[list(time_columns)] >= 0).all().all(), "runtime subjects: negative duration")
    require((subjects.training_trials > 0).all() and (subjects.candidate_training_trials > 0).all(),
            "runtime subjects: invalid trial count")
    close(subjects.training_trial_fraction, subjects.training_trials / subjects.candidate_training_trials,
          "runtime subjects: trial fraction")
    close(subjects.tangent_features, subjects.channels * (subjects.channels + 1) / 2,
          "runtime subjects: tangent feature dimension")
    for row in subjects.itertuples(index=False):
        selected = json.loads(row.selected_sessions)
        require(isinstance(selected, list) and len(set(selected)) == len(selected) == row.retained_sessions,
                "runtime subjects: inconsistent selected session list")
        all_row = subjects[(subjects.dataset == row.dataset) & (subjects.subject == row.subject) &
                           (subjects.method == "all_sources")].iloc[0]
        require(set(selected) <= set(json.loads(all_row.selected_sessions)), "runtime subjects: unknown selected session")
        close((row.candidate_sessions, row.candidate_training_trials),
              (all_row.candidate_sessions, all_row.candidate_training_trials), "runtime candidate pool")
        require(row.training_trials <= row.candidate_training_trials, "runtime: selected trials exceed candidate trials")
    for field in TIME_FIELDS:
        require(((subjects[f"{field}_min"] <= subjects[f"{field}_median"]) &
                 (subjects[f"{field}_median"] <= subjects[f"{field}_max"])).all(),
                f"runtime subjects: inconsistent {field} range")
        if plan["repetitions"] == 3:
            close(subjects[f"{field}_mean"],
                  subjects[[f"{field}_min", f"{field}_median", f"{field}_max"]].mean(axis=1),
                  f"runtime subjects: three-repetition {field} mean")
    close(subjects.screen_fit_seconds_mean, subjects.screen_seconds_mean + subjects.fit_seconds_mean,
          "runtime subjects: mean screening + fitting")
    close(subjects.screen_fit_predict_seconds_mean, subjects.screen_fit_seconds_mean + subjects.predict_seconds_mean,
          "runtime subjects: mean screening + fitting + prediction")
    # Medians are not additive: total medians are checked against the released
    # total-median inputs, never against the sum of component medians.
    all_subjects = subjects[subjects.method == "all_sources"]
    close(all_subjects.screen_seconds_max, np.zeros(len(all_subjects)), "runtime All: no screening charge")
    close(all_subjects.training_trials, all_subjects.candidate_training_trials, "runtime All: trial count")

    output_means = {f"mean_{column}": column for column in RUNTIME_MEANS}
    output_times = {f"mean_subject_median_{field}": f"{field}_median" for field in TIME_FIELDS}
    ratio_fields = ("fit_seconds", "screen_fit_seconds", "screen_fit_predict_seconds")
    reported = read_table(folder / "runtime_summary.csv",
                          {"dataset", "method", "subjects", "timing_repetitions_per_subject",
                           *output_means, *output_times, *(f"{field}_ratio_vs_all" for field in ratio_fields)},
                          ["dataset", "method"])
    require(set(zip(reported.dataset, reported.method)) == set(itertools.product(datasets, RUNTIME_METHODS)),
            "runtime summary: wrong dataset-method inventory")
    for row in reported.itertuples(index=False):
        block = subjects[(subjects.dataset == row.dataset) & (subjects.method == row.method)]
        close(row.subjects, len(block), "runtime summary: subject count")
        close(row.timing_repetitions_per_subject, plan["repetitions"], "runtime summary: repetition count")
        for output, source in {**output_means, **output_times}.items():
            close(getattr(row, output), block[source].to_numpy(float).mean(),
                  f"runtime/{row.dataset}/{row.method}: {output}")
        all_block = subjects[(subjects.dataset == row.dataset) & (subjects.method == "all_sources")]
        for field in ratio_fields:
            denominator = all_block[f"{field}_median"].mean()
            require(denominator > 0, "runtime summary: non-positive All timing")
            expected = block[f"{field}_median"].mean() / denominator
            close(getattr(row, f"{field}_ratio_vs_all"), expected,
                  f"runtime/{row.dataset}/{row.method}: {field} ratio")
    return {"subject_summary_rows": len(subjects), "dataset_summary_rows_recomputed": len(reported),
            "ratios_recomputed": len(reported) * len(ratio_fields)}


def verify_revision_results(input_dir: Path = DEFAULT_INPUT) -> dict:
    input_dir = Path(input_dir)
    manifest = json.loads((input_dir / "revision_manifest.json").read_text(encoding="utf-8"))
    report = {
        "status": "passed",
        "mcs": {dataset: verify_mcs(input_dir, dataset, manifest) for dataset in ("ma2020", "stieger2021")},
        "bootstrap_budget": verify_budget(input_dir, manifest),
        "runtime": verify_runtime(input_dir, manifest),
        "scope": "Recomputation of reported aggregate statistics from released subject summaries; schema and count checks for aggregate SE diagnostics.",
        "not_reconstructed": [
            "subject summaries from split-level predictions or selections",
            "bootstrap draws, confidence bounds, or individual standard-error coordinates",
            "runtime subject medians from individual timing repetitions",
            "raw EEG preprocessing or covariance caches",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    print(json.dumps(verify_revision_results(args.input_dir), indent=2))


if __name__ == "__main__":
    main()
