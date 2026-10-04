#!/usr/bin/env python3
"""Verify classifier-matched target-only comparisons from compact subject means.

The check reconstructs group comparisons, not omitted split predictions,
training/test assignments, covariance preprocessing, or classifier fits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_INPUT = DEFAULT_RESULTS / "eeg" / "target_only_reference"
CLASSIFIER = "tangent-space LDA (solver=lsqr, shrinkage=auto)"
REFERENCE = "Riemannian mean of target training half only"
COHORTS = (("ma2020", "ma2020", 25), ("stieger", "stieger2021", 62))
METHODS = ("target_only", "all_sources", "top1", "top3", "pair_ref")
METRICS = (
    "subjects", "mean_balanced_accuracy", "mean_diff_vs_target_only",
    "below_target_only_rate", "subjects_above_target_only",
)
STRATIFIED_METHODS = (*METHODS, "mcs_stepdown")
STRATIFIED_GROUPS = (
    ("kumar2024", "analysis", {"primary": 18}),
    ("bnci2014_004", "analysis", {"primary": 9, "rolling": 9}),
    ("rolling_origin", "dataset", {"ma2020": 25, "stieger": 62}),
)


def verify_target_dependent_aggregates(results_root: Path = DEFAULT_RESULTS) -> dict:
    """Check target references and derived differences in every released stratum."""
    report = {"subject_method_rows_checked": 0, "summary_rows_recomputed": 0, "strata": {}}
    rolling_reconstructed = None
    for folder, stratum_column, expected_subjects in STRATIFIED_GROUPS:
        directory = results_root / "eeg" / folder
        data = pd.read_csv(directory / "downstream_by_subject.csv")
        required = {stratum_column, "subject", "method", "balanced_accuracy", "target_accuracy", "diff_vs_target"}
        if not required.issubset(data):
            raise ValueError(f"{folder}: target-dependent subject summary is incomplete")
        if set(data[stratum_column]) != set(expected_subjects):
            raise ValueError(f"{folder}: missing or unexpected analysis strata")
        rows = []
        for stratum, n_subjects in expected_subjects.items():
            block = data[data[stratum_column] == stratum]
            expected_keys = {(subject, method) for subject in range(1, n_subjects + 1) for method in STRATIFIED_METHODS}
            keys = list(zip(block["subject"], block["method"]))
            label = f"{folder}/{stratum}"
            if len(keys) != len(set(keys)) or set(keys) != expected_keys:
                raise ValueError(f"{label}: incomplete subject/method grid")
            for field in ("balanced_accuracy", "target_accuracy"):
                if not block[field].between(0, 1).all():
                    raise ValueError(f"{label}: {field} is outside [0, 1]")
            targets = block[block["method"] == "target_only"].set_index("subject")["balanced_accuracy"]
            expected_target = block["subject"].map(targets).to_numpy()
            if not np.allclose(block["target_accuracy"], expected_target, rtol=1e-12, atol=1e-12):
                raise ValueError(f"{label}: stale target_accuracy differs from target-only subject rows")
            differences = block["balanced_accuracy"].to_numpy() - expected_target
            if not np.allclose(block["diff_vs_target"], differences, rtol=1e-12, atol=1e-12):
                raise ValueError(f"{label}: stale diff_vs_target differs from subject accuracies")
            for method in STRATIFIED_METHODS:
                part = block[block["method"] == method]
                # Preserve the published strict-zero rule on validated stored
                # differences, including floating-point ties near zero.
                rows.append({
                    stratum_column: stratum, "method": method, "subjects": n_subjects,
                    "mean_balanced_accuracy": float(part["balanced_accuracy"].mean()),
                    "mean_diff_vs_target": float(part["diff_vs_target"].mean()),
                    "below_target_rate": float((part["diff_vs_target"] < 0).mean()),
                })
            report["strata"][label] = {
                "subjects": n_subjects, "methods": len(STRATIFIED_METHODS),
                "target_mean_balanced_accuracy": float(targets.mean()),
            }
        reconstructed = pd.DataFrame(rows).set_index([stratum_column, "method"])
        summary = pd.read_csv(directory / "downstream_summary.csv")
        summary_fields = ["subjects", "mean_balanced_accuracy", "mean_diff_vs_target", "below_target_rate"]
        if not {stratum_column, "method", *summary_fields}.issubset(summary):
            raise ValueError(f"{folder}: target-dependent summary columns are incomplete")
        if summary.duplicated([stratum_column, "method"]).any():
            raise ValueError(f"{folder}: repeated stratum/method summary rows")
        summary = summary.set_index([stratum_column, "method"])
        if set(summary.index) != set(reconstructed.index):
            raise ValueError(f"{folder}: incomplete stratum/method summary grid")
        if not np.allclose(
            summary.loc[reconstructed.index, summary_fields].to_numpy(),
            reconstructed[summary_fields].to_numpy(), rtol=1e-12, atol=1e-12,
        ):
            raise ValueError(f"{folder}: stale target-dependent summary differs from subject rows")
        report["subject_method_rows_checked"] += len(data)
        report["summary_rows_recomputed"] += len(summary)
        if folder == "rolling_origin":
            rolling_reconstructed = reconstructed
    manuscript = pd.read_csv(results_root / "eeg" / "rolling_origin" / "manuscript_summary.csv")
    if not {"dataset", "subjects", "diff_target"}.issubset(manuscript):
        raise ValueError("rolling manuscript summary is missing required columns")
    if manuscript["dataset"].duplicated().any() or set(manuscript["dataset"]) != {"Ma2020", "Stieger2021"}:
        raise ValueError("rolling manuscript summary must contain both cohorts exactly once")
    for label, dataset in (("Ma2020", "ma2020"), ("Stieger2021", "stieger")):
        observed = manuscript.set_index("dataset").loc[label]
        expected = rolling_reconstructed.loc[(dataset, "pair_ref")]
        if int(observed["subjects"]) != int(expected["subjects"]) or not np.isclose(
            observed["diff_target"], expected["mean_diff_vs_target"], rtol=1e-12, atol=1e-12
        ):
            raise ValueError(f"rolling manuscript summary has a stale target difference: {label}")
    report["rolling_manuscript_rows_checked"] = len(manuscript)
    return report


def verify_target_only_reference(
    input_dir: Path = DEFAULT_INPUT, results_root: Path = DEFAULT_RESULTS
) -> dict:
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    config = manifest.get("target_only_classifier", manifest)
    if config.get("classifier") != CLASSIFIER or manifest.get("classifier") != CLASSIFIER:
        raise ValueError("target-only classifier metadata must identify LSQR with automatic shrinkage")
    if config.get("trial_covariance_shrinkage") != 0.05:
        raise ValueError("target-only trial-covariance shrinkage must be 0.05")
    if config.get("reference", config.get("target_reference")) != REFERENCE:
        raise ValueError("target-only reference must use the target training half only")
    if manifest.get("subjects") != {"ma2020": 25, "stieger": 62} or manifest.get("splits") != 30:
        raise ValueError("target-only metadata must describe 87 subjects and 30 splits")
    target = pd.read_csv(input_dir / "target_only_by_subject.csv")
    required = {"dataset", "subject", "method", "balanced_accuracy", "n_train", "n_test"}
    if not required.issubset(target):
        raise ValueError("target-only subject summary is missing required columns")
    if set(target["dataset"]) != {"ma2020", "stieger"} or set(target["method"]) != {"target_only"}:
        raise ValueError("target-only subject summary has an unexpected dataset or method")
    if target.duplicated(["dataset", "subject"]).any():
        raise ValueError("target-only subject summary contains duplicate subjects")
    if not target["balanced_accuracy"].between(0, 1).all():
        raise ValueError("target-only accuracy is outside [0, 1]")
    for field in ("n_train", "n_test"):
        values = target[field].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any() or not np.array_equal(values, np.rint(values)):
            raise ValueError(f"target-only {field} must contain positive integer counts")
    rows = []
    for dataset, folder, n_subjects in COHORTS:
        block = target[target["dataset"] == dataset].set_index("subject").sort_index()
        if list(block.index) != list(range(1, n_subjects + 1)):
            raise ValueError(f"{dataset}: missing or unexpected target-only subjects")
        source = pd.read_csv(results_root / "eeg" / folder / "downstream_by_subject.csv")
        if not {"subject", "method", "balanced_accuracy"}.issubset(source):
            raise ValueError(f"{dataset}: source-trained summary is missing required columns")
        expected_keys = {(subject, method) for subject in block.index for method in METHODS[1:]}
        keys = list(zip(source["subject"], source["method"]))
        if len(keys) != len(set(keys)) or set(keys) != expected_keys:
            raise ValueError(f"{dataset}: source-trained subject/method grid is incomplete")
        if not source["balanced_accuracy"].between(0, 1).all():
            raise ValueError(f"{dataset}: source-trained accuracy is outside [0, 1]")
        for method in METHODS:
            scores = block["balanced_accuracy"] if method == "target_only" else (
                source[source["method"] == method].set_index("subject")["balanced_accuracy"].reindex(block.index)
            )
            differences = scores - block["balanced_accuracy"]
            rows.append({
                "dataset": dataset, "method": method, "subjects": n_subjects,
                "mean_balanced_accuracy": float(scores.mean()),
                "mean_diff_vs_target_only": float(differences.mean()),
                "below_target_only_rate": float((differences < 0).mean()),
                "subjects_above_target_only": int((differences > 0).sum()),
            })
    reconstructed = pd.DataFrame(rows).set_index(["dataset", "method"])
    released = pd.read_csv(input_dir / "target_only_comparison.csv")
    if not {"dataset", "method", *METRICS}.issubset(released):
        raise ValueError("target-only comparison is missing required columns")
    if released.duplicated(["dataset", "method"]).any():
        raise ValueError("target-only comparison contains duplicate cohort/method rows")
    released = released.set_index(["dataset", "method"])
    if set(released.index) != set(reconstructed.index):
        raise ValueError("target-only comparison must contain all ten cohort/method rows")
    if not np.allclose(
        released.loc[reconstructed.index, list(METRICS)].to_numpy(),
        reconstructed[list(METRICS)].to_numpy(), rtol=1e-12, atol=1e-12,
    ):
        raise ValueError("target-only comparison differs from the released subject means")
    return {
        "status": "passed", "target_subject_rows_checked": len(target),
        "comparison_rows_recomputed": len(reconstructed),
        "classifier": CLASSIFIER,
        "target_dependent_aggregates": verify_target_dependent_aggregates(results_root),
        "not_reconstructed": ["split predictions", "training/test assignments", "covariance preprocessing", "classifier fits"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    print(json.dumps(verify_target_only_reference(args.input_dir, args.results_root), indent=2))


if __name__ == "__main__":
    main()
