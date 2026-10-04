#!/usr/bin/env python3
"""Independently verify saved BSPC MCS-comparison outputs without model fitting.

Reads existing results and covariance-cache metadata. It does not import the
experiment's aggregation functions, rerun selection, refit classifiers, or
overwrite results. Reports are written outside the input result directory.
"""

from __future__ import annotations

import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import math
import re
import shlex
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.model_selection import StratifiedShuffleSplit


METHODS = ("all_sources", "refinement", "mcs_style", "top_m_ref", "range_midpoint")
PAIRS = (("refinement", "all_sources"), ("mcs_style", "all_sources"),
         ("top_m_ref", "all_sources"), ("range_midpoint", "all_sources"),
         ("refinement", "mcs_style"), ("refinement", "top_m_ref"),
         ("refinement", "range_midpoint"), ("mcs_style", "top_m_ref"),
         ("mcs_style", "range_midpoint"), ("top_m_ref", "range_midpoint"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures: list[str] = []
        self.max_summary_absolute_error = 0.0

    def check(self, condition: bool, message: str):
        self.checks += 1
        if not condition:
            self.failures.append(message)

    def close(self, actual, expected, label: str):
        if isinstance(expected, str):
            self.check(actual == expected, label)
            return
        error = abs(float(actual) - float(expected))
        if math.isfinite(error):
            self.max_summary_absolute_error = max(self.max_summary_absolute_error, error)
        self.check(math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-12),
                   f"{label}: actual={actual!r}, expected={expected!r}")

    def table(self, actual: pd.DataFrame, expected: list[dict], keys: list[str], label: str):
        self.check(not actual.duplicated(keys).any(), f"{label}: duplicate keys")
        observed = {tuple(row[key] for key in keys): row for row in actual.to_dict("records")}
        reference = {tuple(row[key] for key in keys): row for row in expected}
        self.check(set(observed) == set(reference), f"{label}: row keys differ")
        for key in observed.keys() & reference.keys():
            self.check(set(observed[key]) == set(reference[key]), f"{label} {key}: columns differ")
            for column, value in reference[key].items():
                if column in observed[key]:
                    self.close(observed[key][column], value, f"{label} {key} {column}")


def recompute_subjects(rows: pd.DataFrame, dataset: str, subjects: list[int]) -> list[dict]:
    result = []
    for subject in subjects:
        by_method = {}
        for method in METHODS:
            selected = rows[(rows.subject == subject) & (rows.method == method)]
            accuracy = selected.balanced_accuracy.tolist()
            by_method[method] = {
                "dataset": dataset, "subject": subject, "method": method,
                "mean_set_size": statistics.mean(selected.set_size.tolist()),
                "mean_training_trials": statistics.mean(selected.training_trials.tolist()),
                "mean_balanced_accuracy": statistics.mean(accuracy),
                "sd_balanced_accuracy_across_splits": statistics.stdev(accuracy),
            }
        for method, row in by_method.items():
            row["accuracy_diff_vs_all"] = row["mean_balanced_accuracy"] - by_method["all_sources"]["mean_balanced_accuracy"]
            row["accuracy_diff_vs_refinement"] = row["mean_balanced_accuracy"] - by_method["refinement"]["mean_balanced_accuracy"]
            result.append(row)
    return result


def recompute_methods(subject_rows: list[dict], dataset: str) -> list[dict]:
    result = []
    for method in METHODS:
        selected = [row for row in subject_rows if row["method"] == method]
        output = {"dataset": dataset, "method": method, "subjects": len(selected)}
        for metric in ("set_size", "training_trials", "balanced_accuracy"):
            values = [row[f"mean_{metric}"] for row in selected]
            output[f"mean_{metric}"] = statistics.mean(values)
            output[f"sd_{metric}"] = statistics.stdev(values)
        for contrast in ("all", "refinement"):
            output[f"mean_accuracy_diff_vs_{contrast}"] = statistics.mean(
                row[f"accuracy_diff_vs_{contrast}"] for row in selected)
        result.append(output)
    return result


def recompute_intervals(subject_rows: list[dict], metric: str, output_name: str) -> list[dict]:
    subjects = sorted({row["subject"] for row in subject_rows})
    lookup = {(row["subject"], row["method"]): row[metric] for row in subject_rows}
    result = []
    for left, right in PAIRS:
        differences = [lookup[subject, left] - lookup[subject, right] for subject in subjects]
        n = len(differences)
        mean = statistics.mean(differences)
        sd = statistics.stdev(differences)
        se = sd / math.sqrt(n)
        half_width = float(t.ppf(0.975, n - 1)) * se
        result.append({"metric": output_name, "method_a": left, "method_b": right,
                       "contrast": f"{left} - {right}", "subjects": n,
                       "mean_difference": mean, "sd_difference": sd, "standard_error": se,
                       "ci_level": 0.95, "ci_lower": mean - half_width, "ci_upper": mean + half_width})
    return result


def verify(run_dir: Path, audit: Audit) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    settings = manifest["settings"]
    dataset = manifest["dataset"]
    subjects = manifest["subjects_requested"]
    splits = int(settings["splits"])
    audit.check(manifest["status"] == "complete", "manifest is not complete")
    audit.check(manifest["subjects_completed"] == subjects, "completed subjects differ from requested")
    audit.check(manifest["subjects_failed"] == [], "subjects failed")
    audit.check(json.loads((run_dir / "errors.json").read_text()) == [], "errors.json not empty")
    audit.check(json.loads((run_dir / "anomalies.json").read_text()) == [], "anomalies.json not empty")
    audit.check(splits == 30 and settings["boot"] == 199, "formal comparison budgets differ")
    rows = pd.read_csv(run_dir / "split_results.csv")
    assignments = pd.read_csv(run_dir / "split_assignments.csv")
    expected_keys = {(s, split, method) for s in subjects for split in range(splits) for method in METHODS}
    observed_keys = set(rows[["subject", "split", "method"]].itertuples(index=False, name=None))
    audit.check(len(rows) == len(expected_keys), "split-result row count differs")
    audit.check(observed_keys == expected_keys, "subject/split/method coverage differs")
    audit.check(not rows.duplicated(["subject", "split", "method"]).any(), "duplicate result rows")
    audit.check(len(assignments) == len(subjects) * splits, "split-assignment count differs")
    audit.check(set(assignments[["subject", "split"]].itertuples(index=False, name=None)) ==
                {(s, split) for s in subjects for split in range(splits)}, "assignment keys differ")
    audit.check(not assignments.duplicated(["subject", "split"]).any(), "duplicate assignments")
    audit.check(rows.dataset.eq(dataset).all() and assignments.dataset.eq(dataset).all(), "dataset labels differ")
    audit.check(np.isfinite(rows.balanced_accuracy).all() and rows.balanced_accuracy.between(0, 1).all(),
                "balanced accuracy is nonfinite or out of range")

    input_hashes = pd.read_csv(run_dir / "input_hashes.csv")
    aggregate = hashlib.sha256()
    for record in input_hashes.itertuples(index=False):
        path = Path(record.path)
        audit.check(path.is_file(), f"input missing: {path}")
        if path.is_file():
            audit.check(path.stat().st_size == record.bytes and sha256(path) == record.sha256,
                        f"input changed: {path}")
        aggregate.update(f"{record.role}\0{record.path}\0{record.sha256}\n".encode())
    audit.check(aggregate.hexdigest() == manifest["aggregate_input_sha256"], "input aggregate hash differs")

    cache_root = Path(settings["cache_root"])
    cache_metadata = []
    for subject in subjects:
        subject_dir = cache_root / (f"sub-{subject:03d}" if dataset == "ma2020" else f"S{subject}")
        paths = sorted((int(re.fullmatch(r"ses-(\d+)_cov.npz", p.name).group(1)), p)
                       for p in subject_dir.glob("ses-*_cov.npz"))
        target = int(settings["target_session"] or max(session for session, _ in paths))
        sources = [session for session, _ in paths if session < target]
        trial_counts = {}
        target_labels = None
        for session, path in paths:
            if session > target:
                continue
            with np.load(path, allow_pickle=False) as archive:
                labels = archive["labels"]
                shape = archive["covs"].shape
            audit.check(shape[0] == len(labels), f"subject {subject} session {session}: cache count mismatch")
            trial_counts[session] = len(labels)
            cache_metadata.append({"subject": subject, "session": session, "n_trials": len(labels),
                                   "n_channels": shape[1]})
            if session == target:
                target_labels = labels
        assert target_labels is not None
        original = list(StratifiedShuffleSplit(n_splits=splits, test_size=0.5,
                                               random_state=int(settings["seed"]) + subject).split(
                                                   np.zeros(len(target_labels)), target_labels))
        for assignment in assignments[assignments.subject == subject].itertuples(index=False):
            split = int(assignment.split)
            screen, test = json.loads(assignment.screen_indices), json.loads(assignment.test_indices)
            label = f"subject {subject} split {split}"
            audit.check(len(screen) == len(set(screen)) and len(test) == len(set(test)), f"{label}: duplicate trial indices")
            audit.check(not set(screen) & set(test), f"{label}: screening/test overlap")
            audit.check(set(screen) | set(test) == set(range(len(target_labels))), f"{label}: target trials not partitioned")
            audit.check(screen == original[split][0].tolist() and test == original[split][1].tolist(),
                        f"{label}: differs from recorded stratified split protocol")
            audit.check(assignment.split_random_state == int(settings["seed"]) + subject, f"{label}: split seed differs")
            audit.check(assignment.bootstrap_seed == int(settings["seed"]) + 12345 + subject, f"{label}: bootstrap seed differs")
            delta = np.asarray(json.loads(assignment.delta_hat))
            audit.check(delta.shape == (len(sources),) and np.isfinite(delta).all(), f"{label}: discrepancy dimensions/values")
            split_rows = rows[(rows.subject == subject) & (rows.split == split)].set_index("method")
            ref_size = int(split_rows.loc["refinement", "set_size"])
            for method, record in split_rows.iterrows():
                selected = json.loads(record.selected_source_indices)
                sessions = json.loads(record.selected_session_ids)
                tag = f"{label} {method}"
                audit.check(len(selected) == len(set(selected)), f"{tag}: duplicate selected source")
                audit.check(all(isinstance(i, int) and 0 <= i < len(sources) for i in selected), f"{tag}: invalid source indices")
                audit.check(sessions == [sources[i] for i in selected], f"{tag}: selected index/session mapping")
                audit.check(record.set_size == len(selected), f"{tag}: selected-set count")
                audit.check(record.candidate_sessions == len(sources) and record.target_session == target, f"{tag}: candidate/target count")
                audit.check(record.training_trials == sum(trial_counts[s] for s in sessions), f"{tag}: training-trial count")
                if method == "all_sources":
                    audit.check(selected == list(range(len(sources))), f"{tag}: not all sources")
                if method == "top_m_ref":
                    audit.check(record.set_size == ref_size, f"{tag}: count differs from refinement")
                    audit.check(selected == np.argsort(delta, kind="mergesort")[:max(1, ref_size)].tolist(), f"{tag}: discrepancy ranking differs")
                if method == "range_midpoint":
                    threshold = float(delta.min() + 0.5 * (delta.max() - delta.min()))
                    audit.check(selected == np.flatnonzero(delta <= threshold).tolist(), f"{tag}: midpoint rule differs")

    subject_rows = recompute_subjects(rows, dataset, subjects)
    audit.table(pd.read_csv(run_dir / "subject_summary.csv"), subject_rows,
                ["dataset", "subject", "method"], "subject_summary")
    method_rows = recompute_methods(subject_rows, dataset)
    audit.table(pd.read_csv(run_dir / "method_summary.csv"), method_rows,
                ["dataset", "method"], "method_summary")
    for metric, output_name, filename in (("mean_balanced_accuracy", "balanced_accuracy", "paired_accuracy_ci.csv"),
                                          ("mean_set_size", "retained_sessions", "paired_set_size_ci.csv"),
                                          ("mean_training_trials", "training_trials", "paired_training_trials_ci.csv")):
        audit.table(pd.read_csv(run_dir / filename), recompute_intervals(subject_rows, metric, output_name),
                    ["method_a", "method_b"], filename)
    return {"dataset": dataset, "subjects": len(subjects), "splits_per_subject": splits,
            "methods": len(METHODS), "split_result_rows": len(rows), "split_assignment_rows": len(assignments),
            "subject_summary_rows": len(subject_rows), "method_summary_rows": len(method_rows),
            "paired_intervals_checked": len(PAIRS) * 3, "input_hashes_checked": len(input_hashes),
            "cached_sessions_checked": len(cache_metadata), "cache_metadata": cache_metadata,
            "method_summary_recomputed": method_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report_path = (args.report_path or run_dir.parent.parent / "verification" / f"mcs_{run_dir.name}.json").resolve()
    if report_path.is_relative_to(run_dir):
        parser.error("report must remain outside the experimental result directory")
    source_files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    before = {str(path): sha256(path) for path in source_files}
    audit = Audit()
    summary = {}
    try:
        summary = verify(run_dir, audit)
    except Exception as error:
        audit.check(False, f"verification interrupted: {type(error).__name__}: {error}")
    unchanged = all(path.is_file() and sha256(path) == before[str(path)] for path in source_files)
    audit.check(unchanged, "experimental outputs changed during audit")
    report = {"utc": datetime.now(timezone.utc).isoformat(), "passed": not audit.failures,
              "checks": audit.checks, "failures": audit.failures, "summary": summary,
              "max_summary_absolute_error": audit.max_summary_absolute_error,
              "command": shlex.join([sys.executable, *sys.argv]), "verifier_sha256": sha256(Path(__file__)),
              "input_output_sha256": before, "experimental_outputs_unchanged": unchanged,
              "scope": "Independent aggregation, paired-CI, partition and cache-count audit; no classifier refit or prediction verification."}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": audit.checks, "failures": audit.failures,
                      "subjects": summary.get("subjects"), "split_results": summary.get("split_result_rows"),
                      "max_summary_absolute_error": audit.max_summary_absolute_error, "report": str(report_path)}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
