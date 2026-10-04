"""Regression tests for the fixed-design budget check; no EEG data required."""

import argparse
import hashlib
import json
import shlex
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

import ci_gate_bspc_bootstrap_budget as budget
from ci_gate_ma2020_riemann import riemann_estimates_and_bootstrap
from ci_gate_realdata_case_study import build_system, pair_gate, rect_gate


class BudgetCheckTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1807)
        def covariances(n, shift):
            x = rng.normal(size=(n, 3, 6))
            return np.einsum("nct,ndt->ncd", x, x) / 6 + shift * np.eye(3)
        self.sources = [covariances(18, 0.3), covariances(20, 1.1), covariances(23, 2.4)]
        self.target = covariances(16, 0.1)

    def test_discrepancy_draws_match_primary_script(self):
        old_estimate, old_draws = riemann_estimates_and_bootstrap(
            self.sources, self.target, np.random.default_rng(17), 999, 0.1, True)
        estimate, draws = budget.discrepancy_bootstrap(
            self.sources, self.target, np.random.default_rng(17), 999)
        np.testing.assert_allclose(estimate, old_estimate, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(draws, old_draws, rtol=1e-10, atol=1e-11)

    def test_each_prefix_matches_primary_calibration(self):
        estimate, draws = budget.discrepancy_bootstrap(
            self.sources, self.target, np.random.default_rng(27), 999)
        for count in budget.BUDGETS:
            original = build_system(estimate, draws[:count], 0.025, 0.025, 0.05)
            current = budget.calibrate(estimate, draws[:count])
            for key in ("q_c", "q_d", "se_c", "se_d", "lower_c", "upper_c", "lower_d"):
                np.testing.assert_allclose(current[key], original[key], rtol=1e-12, atol=1e-12)
            rect = rect_gate(original["lower_c"], original["upper_c"])
            pair = pair_gate(len(estimate), original["jj"], original["lower_d"])
            np.testing.assert_array_equal(current["selected"], sorted(set(rect) & set(pair)))

    def test_subject_aggregation_does_not_weight_repeated_cases_as_subjects(self):
        data = pd.DataFrame({"dataset": ["x"] * 4, "subject": [1, 1, 1, 2],
                             "budget": [99] * 4, "metric": [1.0, 1.0, 1.0, 3.0]})
        subjects, summary = budget.subject_aggregation(data, ["metric"])
        self.assertEqual(len(subjects), 2)
        self.assertEqual(summary.loc[0, "metric"], 2.0)
        self.assertEqual(summary.loc[0, "n_subjects"], 2)
        self.assertNotEqual(summary.loc[0, "metric"], data.metric.mean())

    def test_first_splits_reproduce_primary_protocol(self):
        labels = np.repeat([0, 1], 31)
        for dataset in budget.SPLIT_SEEDS:
            actual = budget.split_indices(labels, dataset, 3, 2)
            original = list(StratifiedShuffleSplit(n_splits=30, test_size=0.5,
                                                  random_state=budget.SPLIT_SEEDS[dataset] + 3)
                            .split(np.zeros(len(labels)), labels))[:2]
            for (screen, test), (old_screen, old_test) in zip(actual, original):
                np.testing.assert_array_equal(screen, old_screen)
                np.testing.assert_array_equal(test, old_test)
                self.assertFalse(set(screen) & set(test))

    def test_identical_draws_have_identical_budget_result(self):
        estimate, draws = budget.discrepancy_bootstrap(
            self.sources, self.target, np.random.default_rng(18), 999)
        first, second = budget.calibrate(estimate, draws), budget.calibrate(estimate, draws.copy())
        self.assertEqual(first["q_c"], second["q_c"])
        self.assertEqual(first["q_d"], second["q_d"])
        self.assertEqual(budget.jaccard(first["selected"], second["selected"]), 1.0)

    def test_raw_standard_errors_are_preserved_before_numerical_floor(self):
        estimate = np.array([0.0, 0.0, 0.0])
        system = budget.calibrate(estimate, np.zeros((99, 3)))
        np.testing.assert_array_equal(system["raw_se_c"], np.zeros(3))
        np.testing.assert_array_equal(system["raw_se_d"], np.zeros(6))
        np.testing.assert_array_equal(system["se_c"], np.full(3, 1e-12))
        np.testing.assert_array_equal(system["se_d"], np.full(6, 1e-12))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("-v", action="store_true")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BudgetCheckTests)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2 if args.v else 1).run(suite)
    if args.report_path:
        report = {"utc": datetime.now(timezone.utc).isoformat(), "tests_run": result.testsRun,
                  "passed": result.wasSuccessful(), "failures": len(result.failures),
                  "errors": len(result.errors), "command": shlex.join([sys.executable, *sys.argv]),
                  "test_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "experiment_script_sha256": hashlib.sha256(Path(budget.__file__).read_bytes()).hexdigest()}
        args.report_path.write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
