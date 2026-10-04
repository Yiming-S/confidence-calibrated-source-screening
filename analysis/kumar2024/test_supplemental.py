"""Synthetic regression tests for fixed supplementary computations."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

import supplemental as supplement
import verify_supplemental as verification


class SupplementalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).resolve().parent
        frozen = directory / "backend_snapshot" if (directory / "backend_snapshot").is_dir() else directory / "simulations"
        extra = directory.parent
        cls.modules = supplement.backend(frozen, extra)
        rng = np.random.default_rng(212)
        def covariances(offset):
            data = rng.normal(size=(16, 3, 20))
            return np.einsum("nct,ndt->ncd", data, data) / 20 + np.eye(3) * offset
        cls.sources = [covariances(.1 + i * .1) for i in range(5)]
        cls.labels = [np.array([0, 1] * 8) for _ in range(5)]
        cls.target = {"covs": covariances(.2), "labels": np.array([0, 1] * 8), "trial_ids": np.array([f"trial-{i}" for i in range(16)])}

    def test_budget_helper_matches_primary_formula_and_nested_prefixes(self):
        expected, draws = self.modules["ci_gate_ma2020_riemann"].riemann_estimates_and_bootstrap(
            self.sources, self.target["covs"], np.random.default_rng(99), 19, .1, True)
        actual, optimized = self.modules["ci_gate_bspc_bootstrap_budget"].discrepancy_bootstrap(
            self.sources, self.target["covs"], np.random.default_rng(99), 19, .1)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(optimized, draws, atol=1e-12, rtol=1e-10)
        for count in (9, 19):
            system = self.modules["ci_gate_bspc_bootstrap_budget"].calibrate(actual, optimized[:count])
            selected = self.modules["ci_gate_bnci004_riemann"].selected_sets(actual, optimized[:count], .05, .025, .025)
            np.testing.assert_array_equal(system["selected"], selected["pair_ref"])

    def test_block_helper_matches_explicit_weighted_means(self):
        helper = self.modules["ci_gate_block_bootstrap_sensitivity"]
        ordinary = self.modules["ci_gate_ma2020_riemann"]
        estimate, draws = helper.riemann_estimates_and_block_bootstrap(self.sources, self.target["covs"], np.random.default_rng(81), 9, .1, 4)
        rng = np.random.default_rng(81)
        target_counts = helper.circular_block_counts(16, 9, 4, rng)
        np.testing.assert_array_equal(target_counts.sum(axis=1), np.full(9, 16))
        target_means = np.einsum("bn,ncd->bcd", target_counts, self.target["covs"]) / 16
        manual = np.zeros_like(draws)
        for j, source in enumerate(self.sources):
            counts = helper.circular_block_counts(16, 9, 4, rng)
            source_means = np.einsum("bn,ncd->bcd", counts, source) / 16
            manual[:, j] = [ordinary.airm2(ordinary.shrink(a, .1), ordinary.shrink(b, .1)) for a, b in zip(source_means, target_means)]
        np.testing.assert_allclose(draws, manual, atol=1e-11, rtol=1e-9)

    def test_certificate_distinguishes_component_and_contrast_certificates(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = supplement.certificate(self.sources, self.target, {"seed": 2026100200, "bootstrap": 19}, self.modules, Path(temporary), "test")
            self.assertEqual(len(result["sources"]), 5)
            self.assertEqual(len(result["system"]["lower_d"]), 20)
            for row in result["sources"]:
                self.assertEqual(row["rectangle_excluded"], row["rectangle_exclusion_margin"] > 0)
                self.assertEqual(row["direct_contrast_excluded"], bool(row["positive_contrast_witnesses"]))
                self.assertEqual(row["refinement_retained"], not (row["rectangle_excluded"] or row["direct_contrast_excluded"]))

    def test_fixed_budget_case_has_all_budgets_and_replayable_draws(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            result = supplement.budget_case(1, 0, 2026100201, self.sources, self.target, self.modules, out, "test")
            self.assertEqual([r["budget"] for r in result["rows"]], [99, 199, 499, 999])
            self.assertEqual(result["rows"][-1]["same_members_vs_999"], 1)
            with np.load(out / "draws" / result["archive"]) as saved:
                self.assertEqual(saved["draws"].shape, (999, 5))
            self.assertFalse(set(result["screen_indices"]) & set(result["test_indices"]))

    def test_block_subject_has_fixed_case_counts_and_ordered_screening(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = supplement.block_subject(1, self.sources, self.labels, self.target, self.modules, Path(temporary), "test")
            self.assertEqual(len(result["screening"]), 3)
            self.assertEqual(len(result["downstream"]), 50)
            self.assertEqual(len(result["draws"]), 23)
            for row in result["splits"]:
                self.assertEqual(row["ordered_screen_indices"], sorted(row["screen_indices"]))
                self.assertFalse(set(row["screen_indices"]) & set(row["test_indices"]))
            self.assertEqual(sum(row["method"] == "all_sources" for row in result["downstream"]), 10)

    def test_independent_record_and_summary_verification_on_synthetic_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            result = supplement.certificate(self.sources, self.target, {"seed": 2026100200, "bootstrap": 1999}, self.modules, out, "test")
            result["kind"] = "certificate"
            results = [result]
            for subject in (1, 2, 3):
                for split in (0, 1):
                    for seed in (2026100201, 2026100202, 2026100203):
                        case = supplement.budget_case(subject, split, seed, self.sources, self.target, self.modules, out, "test")
                        case["kind"] = "budget"
                        results.append(case)
            for subject in range(1, 19):
                case = supplement.block_subject(subject, self.sources, self.labels, self.target, self.modules, out, "test")
                case["kind"] = "block"
                results.append(case)
            supplement.aggregate(results, out, self.modules)
            verification.verify_certificate(out, "test", self.modules)
            verification.verify_budget(out, "test", self.modules)
            verification.verify_block(out, "test", self.modules)
            changed = supplement.read(out / "block_summary.json")
            changed["all_source_accuracy"] += .01
            supplement.save(out / "block_summary.json", changed)
            with self.assertRaisesRegex(ValueError, "all-source mean"):
                verification.verify_block(out, "test", self.modules)


if __name__ == "__main__":
    unittest.main()
