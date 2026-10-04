"""Public-table reconstruction and deliberate-corruption regression tests."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_revision_results import DEFAULT_INPUT, verify_revision_results


class RevisionResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.results = Path(self.temporary.name) / "revision_results"
        shutil.copytree(DEFAULT_INPUT, self.results)

    def corrupt(self, relative: str, column: str, amount: float = 0.01) -> None:
        path = self.results / relative
        data = pd.read_csv(path)
        data.loc[0, column] += amount
        data.to_csv(path, index=False)

    def assert_rejected(self) -> None:
        with self.assertRaises(ValueError):
            verify_revision_results(self.results)

    def test_public_summaries_reconstruct_without_raw_artifacts(self) -> None:
        report = verify_revision_results(self.results)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["mcs"]["ma2020"]["subjects"], 25)
        self.assertEqual(report["mcs"]["stieger2021"]["subjects"], 62)
        self.assertEqual(sum(block["paired_intervals_recomputed"] for block in report["mcs"].values()), 60)
        self.assertEqual(report["bootstrap_budget"]["dataset_and_seed_summary_rows_recomputed"], 24)
        self.assertEqual(report["runtime"]["dataset_summary_rows_recomputed"], 6)
        self.assertEqual(report["runtime"]["ratios_recomputed"], 18)
        self.assertEqual(len(report["not_reconstructed"]), 4)
        self.assertEqual(len(list(self.results.rglob("*.csv"))), 17)

    def test_tampered_mcs_mean_is_rejected(self) -> None:
        self.corrupt("mcs/ma2020/method_summary.csv", "mean_balanced_accuracy")
        self.assert_rejected()

    def test_tampered_mcs_standard_deviation_is_rejected(self) -> None:
        self.corrupt("mcs/stieger2021/method_summary.csv", "sd_set_size")
        self.assert_rejected()

    def test_tampered_confidence_interval_is_rejected(self) -> None:
        self.corrupt("mcs/ma2020/paired_accuracy_ci.csv", "ci_upper")
        self.assert_rejected()

    def test_tampered_trial_interval_is_rejected(self) -> None:
        self.corrupt("mcs/stieger2021/paired_training_trials_ci.csv", "ci_lower", amount=1.0)
        self.assert_rejected()

    def test_missing_paired_contrast_is_rejected(self) -> None:
        path = self.results / "mcs/ma2020/paired_set_size_ci.csv"
        data = pd.read_csv(path)
        data.iloc[1:].to_csv(path, index=False)
        self.assert_rejected()

    def test_missing_subject_method_is_rejected(self) -> None:
        path = self.results / "mcs/stieger2021/subject_summary.csv"
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        self.assert_rejected()

    def test_tampered_budget_dataset_mean_is_rejected(self) -> None:
        self.corrupt("bootstrap_budget/dataset_summary.csv", "q_component")
        self.assert_rejected()

    def test_tampered_seed_summary_is_rejected(self) -> None:
        self.corrupt("bootstrap_budget/seed_variability_summary.csv", "q_contrast_seed_sd")
        self.assert_rejected()

    def test_tampered_diagnostic_count_is_rejected(self) -> None:
        self.corrupt("bootstrap_budget/se_diagnostics_summary.csv", "n_seed_sample_cases", amount=1.0)
        self.assert_rejected()

    def test_tampered_runtime_mean_is_rejected(self) -> None:
        self.corrupt("runtime/runtime_summary.csv", "mean_subject_median_fit_seconds")
        self.assert_rejected()

    def test_tampered_runtime_ratio_is_rejected(self) -> None:
        self.corrupt("runtime/runtime_summary.csv", "screen_fit_predict_seconds_ratio_vs_all")
        self.assert_rejected()

    def test_tampered_runtime_subject_median_is_rejected(self) -> None:
        self.corrupt("runtime/runtime_subject_summary.csv", "fit_seconds_median")
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main()
