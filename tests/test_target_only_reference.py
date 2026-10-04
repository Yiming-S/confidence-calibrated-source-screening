"""Classifier parity and compact target-only checks using synthetic records."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import ci_gate_ma2020_riemann as source_pipeline
from ci_gate_target_only_reference import train_eval_target_only, write_latex_table
from verify_target_only_reference import (
    CLASSIFIER, COHORTS, METHODS, REFERENCE, STRATIFIED_GROUPS,
    STRATIFIED_METHODS, verify_target_only_reference,
)


class TargetOnlyClassifierTests(unittest.TestCase):
    @staticmethod
    def arrays(channels: int, train_n: int, test_n: int):
        rng = np.random.default_rng(1871 + channels)
        matrices = rng.normal(size=(train_n + test_n, channels, channels))
        covs = matrices @ matrices.transpose(0, 2, 1) + np.eye(channels)
        return covs[:train_n], np.arange(train_n) % 2, covs[train_n:], np.arange(test_n) % 2

    def test_identical_arrays_match_the_source_classifier(self) -> None:
        for channels, train_n, test_n in ((3, 24, 10), (6, 10, 8)):
            with self.subTest(channels=channels, train_trials=train_n):
                arrays = self.arrays(channels, train_n, test_n)
                self.assertEqual(train_eval_target_only(*arrays), source_pipeline.train_eval_ts(*arrays))

    def test_target_only_uses_lsqr_with_automatic_shrinkage(self) -> None:
        with patch.object(source_pipeline, "LinearDiscriminantAnalysis", wraps=source_pipeline.LinearDiscriminantAnalysis) as classifier:
            train_eval_target_only(*self.arrays(6, 10, 8))
        classifier.assert_called_once_with(solver="lsqr", shrinkage="auto")


class TargetOnlyCompactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.results = Path(self.temporary.name) / "results"
        self.input = self.results / "eeg" / "target_only_reference"
        self.input.mkdir(parents=True)
        manifest = {
            "classifier": CLASSIFIER, "trial_covariance_shrinkage": 0.05,
            "target_reference": REFERENCE, "subjects": {"ma2020": 25, "stieger": 62}, "splits": 30,
        }
        (self.input / "manifest.json").write_text(json.dumps(manifest))
        targets, comparisons = [], []
        for dataset, folder, n_subjects in COHORTS:
            source_rows = []
            for subject in range(1, n_subjects + 1):
                targets.append({"dataset": dataset, "subject": subject, "method": "target_only",
                                "balanced_accuracy": 0.5, "n_train": 20, "n_test": 20})
                for method in METHODS[1:]:
                    source_rows.append({"subject": subject, "method": method, "balanced_accuracy": 0.75})
            directory = self.results / "eeg" / folder
            directory.mkdir()
            pd.DataFrame(source_rows).to_csv(directory / "downstream_by_subject.csv", index=False)
            for method in METHODS:
                target = method == "target_only"
                comparisons.append({"dataset": dataset, "method": method, "subjects": n_subjects,
                                    "mean_balanced_accuracy": 0.5 if target else 0.75,
                                    "mean_diff_vs_target_only": 0.0 if target else 0.25,
                                    "below_target_only_rate": 0.0,
                                    "subjects_above_target_only": 0 if target else n_subjects})
        pd.DataFrame(targets).to_csv(self.input / "target_only_by_subject.csv", index=False)
        pd.DataFrame(comparisons).to_csv(self.input / "target_only_comparison.csv", index=False)
        for folder, stratum_column, groups in STRATIFIED_GROUPS:
            directory = self.results / "eeg" / folder
            directory.mkdir()
            subject_rows, summary_rows = [], []
            for stratum, n_subjects in groups.items():
                for method in STRATIFIED_METHODS:
                    target = method == "target_only"
                    for subject in range(1, n_subjects + 1):
                        subject_rows.append({
                            stratum_column: stratum, "subject": subject, "method": method,
                            "balanced_accuracy": 0.5 if target else 0.75, "target_accuracy": 0.5,
                            "diff_vs_target": 0.0 if target else 0.25,
                        })
                    summary_rows.append({
                        stratum_column: stratum, "method": method, "subjects": n_subjects,
                        "mean_balanced_accuracy": 0.5 if target else 0.75,
                        "mean_diff_vs_target": 0.0 if target else 0.25, "below_target_rate": 0.0,
                    })
            pd.DataFrame(subject_rows).to_csv(directory / "downstream_by_subject.csv", index=False)
            pd.DataFrame(summary_rows).to_csv(directory / "downstream_summary.csv", index=False)
        pd.DataFrame([
            {"dataset": "Ma2020", "subjects": 25, "diff_target": 0.25},
            {"dataset": "Stieger2021", "subjects": 62, "diff_target": 0.25},
        ]).to_csv(self.results / "eeg" / "rolling_origin" / "manuscript_summary.csv", index=False)

    def test_synthetic_compact_records_reconstruct_without_reruns_or_mutation(self) -> None:
        before = {p: p.read_bytes() for p in self.results.rglob("*") if p.is_file()}
        report = verify_target_only_reference(self.input, self.results)
        self.assertEqual(report["target_subject_rows_checked"], 87)
        self.assertEqual(report["comparison_rows_recomputed"], 10)
        extended = report["target_dependent_aggregates"]
        self.assertEqual(extended["subject_method_rows_checked"], 738)
        self.assertEqual(extended["summary_rows_recomputed"], 30)
        self.assertEqual(extended["rolling_manuscript_rows_checked"], 2)
        self.assertEqual(len(extended["strata"]), 5)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_table_caption_identifies_the_matched_classifier(self) -> None:
        table = Path(self.temporary.name) / "comparison.tex"
        write_latex_table(pd.read_csv(self.input / "target_only_comparison.csv"), table)
        text = table.read_text()
        self.assertIn("tangent-space LSQR LDA pipeline with automatic shrinkage", text)
        self.assertIn("trial-covariance shrinkage 0.05", text)
        self.assertIn(r"\WideTableBody", text)
        self.assertNotIn("SVD", text)

    def test_legacy_classifier_metadata_is_rejected(self) -> None:
        path = self.input / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["classifier"] = "tangent-space LDA (SVD solver for p > n)"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "LSQR"):
            verify_target_only_reference(self.input, self.results)

    def test_missing_target_subject_is_rejected(self) -> None:
        path = self.input / "target_only_by_subject.csv"
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "target-only subjects"):
            verify_target_only_reference(self.input, self.results)

    def test_changed_comparison_is_rejected(self) -> None:
        path = self.input / "target_only_comparison.csv"
        data = pd.read_csv(path)
        data.loc[1, "mean_diff_vs_target_only"] = 0.3
        data.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "differs from the released subject means"):
            verify_target_only_reference(self.input, self.results)

    def test_missing_method_comparison_is_rejected(self) -> None:
        path = self.input / "target_only_comparison.csv"
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "all ten"):
            verify_target_only_reference(self.input, self.results)

    def test_missing_source_subject_method_is_rejected(self) -> None:
        path = self.results / "eeg" / "ma2020" / "downstream_by_subject.csv"
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "source-trained subject/method grid"):
            verify_target_only_reference(self.input, self.results)

    def test_stale_kumar_subject_target_reference_is_rejected(self) -> None:
        path = self.results / "eeg" / "kumar2024" / "downstream_by_subject.csv"
        data = pd.read_csv(path)
        data.loc[0, "target_accuracy"] = 0.49
        data.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "kumar2024/primary: stale target_accuracy"):
            verify_target_only_reference(self.input, self.results)

    def test_stale_bnci_target_difference_in_each_stratum_is_rejected(self) -> None:
        path = self.results / "eeg" / "bnci2014_004" / "downstream_summary.csv"
        original = path.read_bytes()
        for stratum in ("primary", "rolling"):
            data = pd.read_csv(path)
            data.loc[(data["analysis"] == stratum) & (data["method"] == "pair_ref"), "mean_diff_vs_target"] = 0.3
            data.to_csv(path, index=False)
            with self.subTest(stratum=stratum), self.assertRaisesRegex(ValueError, "bnci2014_004: stale target-dependent summary"):
                verify_target_only_reference(self.input, self.results)
            path.write_bytes(original)

    def test_stale_rolling_below_target_rate_is_rejected(self) -> None:
        path = self.results / "eeg" / "rolling_origin" / "downstream_summary.csv"
        data = pd.read_csv(path)
        data.loc[(data["dataset"] == "stieger") & (data["method"] == "pair_ref"), "below_target_rate"] = 0.5
        data.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "rolling_origin: stale target-dependent summary"):
            verify_target_only_reference(self.input, self.results)

    def test_stale_rolling_manuscript_difference_is_rejected(self) -> None:
        path = self.results / "eeg" / "rolling_origin" / "manuscript_summary.csv"
        data = pd.read_csv(path)
        data.loc[0, "diff_target"] = 0.3
        data.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "rolling manuscript summary has a stale target difference"):
            verify_target_only_reference(self.input, self.results)

    def test_each_additional_stratum_requires_complete_subjects(self) -> None:
        for folder, stratum_column, groups in STRATIFIED_GROUPS:
            path = self.results / "eeg" / folder / "downstream_by_subject.csv"
            original = path.read_bytes()
            for stratum in groups:
                data = pd.read_csv(path)
                data = data[~((data[stratum_column] == stratum) & (data["subject"] == 1))]
                data.to_csv(path, index=False)
                with self.subTest(folder=folder, stratum=stratum), self.assertRaisesRegex(ValueError, "incomplete subject/method grid"):
                    verify_target_only_reference(self.input, self.results)
                path.write_bytes(original)

    def test_full_rolling_summary_is_required(self) -> None:
        (self.results / "eeg" / "rolling_origin" / "downstream_summary.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            verify_target_only_reference(self.input, self.results)


if __name__ == "__main__":
    unittest.main()
