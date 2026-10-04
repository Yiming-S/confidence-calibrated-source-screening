"""Focused synthetic regression tests; no Kumar data or full experiment required."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from scipy import stats

SPEC = importlib.util.spec_from_file_location("kumar_analysis", Path(__file__).with_name("analyze.py"))
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    channels = [f"E{i}" for i in range(22)]

    def cache(self, root, subject=1, sessions=range(1, 7), corrupt=None):
        for session in sessions:
            path = analysis.session_path(root, subject, session)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {"covs": np.stack([np.eye(22) * (i + session + 1) for i in range(16)]),
                    "labels": np.array([0, 1] * 8),
                    "trial_ids": np.array([f"s{subject}/session{session}/trial{i}" for i in range(16)]),
                    "protocol_sha256": "p", "unit": "microvolt_squared", "channels": np.array(self.channels)}
            if corrupt:
                corrupt(data)
            np.savez(path, **data)

    def test_protocol_is_frozen_and_hash_covers_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.json"
            protocol = {"analysis": copy.deepcopy(analysis.FIXED_ANALYSIS), "name": "a"}
            analysis.write_json(path, protocol)
            _, first = analysis.read_protocol(path)
            protocol["name"] = "b"
            analysis.write_json(path, protocol)
            self.assertNotEqual(first, analysis.read_protocol(path)[1])
            protocol["analysis"]["downstream_boot"] = 200
            analysis.write_json(path, protocol)
            with self.assertRaisesRegex(ValueError, "Frozen protocol mismatch"):
                analysis.read_protocol(path)

    def test_complete_cache_required_and_class_counts_not_silently_excluded(self):
        settings = {**analysis.FIXED_ANALYSIS, "subjects": [1]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.cache(root, sessions=range(1, 6))
            with self.assertRaises(FileNotFoundError):
                analysis.preflight(root, settings, "p", self.channels)
            self.cache(root, sessions=[6], corrupt=lambda data: data.update(labels=np.array([0] * 13 + [1] * 3)))
            with self.assertRaisesRegex(ValueError, "fewer than 4"):
                analysis.preflight(root, settings, "p", self.channels)

    def test_cache_rejects_alignment_and_duplicate_ids(self):
        settings = analysis.FIXED_ANALYSIS
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.cache(root, sessions=[1], corrupt=lambda data: data.update(labels=data["labels"][:-1]))
            with self.assertRaisesRegex(ValueError, "not aligned"):
                analysis.load_session(analysis.session_path(root, 1, 1), settings, "p", self.channels)
            self.cache(root, sessions=[1], corrupt=lambda data: data.update(trial_ids=np.array(["same"] * 16)))
            with self.assertRaisesRegex(ValueError, "unique strings"):
                analysis.load_session(analysis.session_path(root, 1, 1), settings, "p", self.channels)

    def test_cache_protocol_unit_channels_and_global_trial_ids_are_checked(self):
        settings = {**analysis.FIXED_ANALYSIS, "subjects": [1]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.cache(root)
            with self.assertRaisesRegex(ValueError, "protocol hash mismatch"):
                analysis.preflight(root, settings, "different", self.channels)
            with self.assertRaisesRegex(ValueError, "channel names or order"):
                analysis.preflight(root, settings, "p", list(reversed(self.channels)))
            self.cache(root, sessions=[1], corrupt=lambda data: data.update(unit="volt_squared"))
            with self.assertRaisesRegex(ValueError, "unit microvolt_squared"):
                analysis.preflight(root, settings, "p", self.channels)
            self.cache(root, corrupt=lambda data: data.update(trial_ids=np.array([f"reused{i}" for i in range(16)])))
            with self.assertRaisesRegex(ValueError, "overlap a previous session or subject"):
                analysis.preflight(root, settings, "p", self.channels)

    def test_subject_mapping_skips_raw_subject_ten(self):
        self.assertEqual(analysis.subject_info(9), {"subject": 9, "raw_subject": 9, "group": "GR"})
        self.assertEqual(analysis.subject_info(10), {"subject": 10, "raw_subject": 11, "group": "PAR"})
        self.assertEqual(analysis.subject_info(18)["raw_subject"], 19)

    def test_actual_reused_backend_on_synthetic_covariances(self):
        root = Path(os.environ.get("CI_GATE_SIMULATIONS_ROOT", Path(__file__).resolve().parent / "backend_snapshot"))
        if not root.is_dir():
            self.skipTest("Canonical simulations directory not present in this test layout")
        backend = analysis.load_backend(root)
        rng = np.random.default_rng(12)
        def covariances(offset):
            x = rng.normal(size=(16, 3, 24))
            return np.einsum("nct,ndt->ncd", x, x) / 24 + np.eye(3) * offset
        sources = [covariances(.2 + i * .2) for i in range(5)]
        target = covariances(.3)
        hat, boot = backend.bootstrap(sources, target[:8], np.random.default_rng(42), 19, .1, True)
        settings = analysis.FIXED_ANALYSIS
        selections = analysis.normalized_sets(hat, boot, settings, backend)
        self.assertEqual(set(selections), set(analysis.SOURCE_METHODS))
        self.assertEqual(boot.shape, (19, 5))
        labels = np.array([0, 1] * 8)
        x, y = backend.pool(selections["pair_ref"], sources, [labels] * 5)
        value = backend.train_eval_ts(x, y, target[8:], labels[8:])
        self.assertTrue(0 <= value <= 1)
        reference = backend.discrepancy_vector(sources, target, .1)
        metrics, chosen = backend.evaluate_outer(reference, sources, target, np.random.default_rng(43),
                                                   1., .5, 19, .1, .05, .025, .025)
        self.assertEqual(set(analysis.COVERAGE_METRICS), set(metrics))
        self.assertEqual(metrics["coverage_joint"], metrics["coverage_c"] * metrics["coverage_d"])
        self.assertGreater(len(chosen), 0)

    def test_paired_inference_uses_subject_n_and_correct_one_sided_boundary(self):
        values = np.array([-0.005, 0.003, -0.002, 0.001])
        result = analysis.paired_inference(values.tolist())
        se = np.std(values, ddof=1) / np.sqrt(4)
        self.assertEqual(result["n_subjects"], 4)
        self.assertAlmostEqual(result["two_sided_95_ci"][0], np.mean(values) - stats.t.ppf(.975, 3) * se)
        self.assertAlmostEqual(result["noninferiority"]["0.02"]["one_sided_p"], stats.t.sf((np.mean(values) + .02) / se, 3))
        self.assertFalse(analysis.paired_inference([-.02] * 4)["noninferiority"]["0.02"]["noninferior"])

    def test_final_summary_uses_all_subjects_and_keeps_groups_descriptive(self):
        checkpoints = []
        for subject in range(1, 19):
            methods = {method: {"balanced_accuracy": .6, "set_size": 3, "training_trials": 48}
                       for method in analysis.METHODS}
            methods["pair_ref"]["balanced_accuracy"] += subject / 10000
            metric = {"coverage_c": 1, "coverage_d": 1, "coverage_joint": 1,
                      "oracle_retained": 1, "exact_recovery": 0, "set_size": 2,
                      "retained_fraction": .4, "worst_reference_excess": .1,
                      "source_indices": [0, 1]}
            checkpoints.append({"subject": subject, "complete": True,
                                "source_sessions": list(range(1, 6)),
                                "downstream": [{"methods": methods} for _ in range(30)],
                                "coverage": [metric for _ in range(50)],
                                "screening": {"methods": {m: {"set_size": 3} for m in analysis.SOURCE_METHODS}}})
        summary = analysis.summarize(checkpoints, analysis.FIXED_ANALYSIS)
        primary = summary["primary_refinement_minus_all_sources"]
        self.assertEqual(primary["n_subjects"], 18)
        self.assertAlmostEqual(primary["mean_difference"], np.mean(np.arange(1, 19) / 10000))
        self.assertEqual(summary["overall"]["n_outer"], 900)
        for group in ("GR", "PAR"):
            self.assertEqual(summary["group_descriptive_only"][group]["n_subjects"], 9)
            self.assertNotIn("noninferiority", summary["group_descriptive_only"][group])
        self.assertEqual(summary["refinement_minus_mcs"]["noninferiority"], {})
        with self.assertRaisesRegex(ValueError, "all prespecified subjects"):
            analysis.summarize(checkpoints[:-1], analysis.FIXED_ANALYSIS)

    def test_checkpoint_rejects_changed_code_or_nonsequential_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            identity = {"subject": 1, "protocol_sha256": "p", "code_sha256": "c", "cache_sha256": "d"}
            checkpoint = analysis.load_checkpoint(path, identity, analysis.FIXED_ANALYSIS)
            analysis.write_json(path, checkpoint)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                analysis.load_checkpoint(path, {**identity, "code_sha256": "changed"}, analysis.FIXED_ANALYSIS)
            checkpoint["downstream"] = [{"split": 1}]
            analysis.write_json(path, checkpoint)
            with self.assertRaisesRegex(ValueError, "indices"):
                analysis.load_checkpoint(path, identity, analysis.FIXED_ANALYSIS)

    def test_shared_draws_splits_cache_target_only_and_resume(self):
        # Reduced repetitions exist only in this injected-backend unit test.
        # The production protocol reader rejects these changed settings.
        settings = {**analysis.FIXED_ANALYSIS, "subjects": [1], "n_splits": 2, "audit_outer": 2}
        bootstrap_calls, fit_calls, selected_calls = [], [], []

        def bootstrap(sources, target, rng, budget, shrinkage, shared):
            bootstrap_calls.append((target.copy(), budget, shrinkage, shared))
            return np.arange(5, dtype=float), np.zeros((budget, 5))

        def selected_sets(hat, boot, alpha, alpha_c, alpha_d):
            selected_calls.append(id(boot))
            return {"top1": np.array([0]), "top3": np.array([2, 1, 0]),
                    "mcs_stepdown": np.array([0, 1, 2]), "pair_ref": np.array([0, 1, 2])}

        def fit(covs, labels, test, test_labels):
            fit_calls.append((covs.copy(), labels.copy(), test.copy(), test_labels.copy()))
            return .75

        def evaluate(reference, sources, target, rng, sf, tf, budget, lam, alpha, ac, ad):
            return {"coverage_c": 1, "coverage_d": 1, "coverage_joint": 1,
                    "oracle_retained": 1, "exact_recovery": 1, "set_size": 1,
                    "retained_fraction": .2, "worst_reference_excess": 0.}, np.array([0])

        backend = SimpleNamespace(bootstrap=Mock(side_effect=bootstrap), selected_sets=selected_sets,
                                  pool=lambda idx, x, y: (np.concatenate([x[i] for i in idx]), np.concatenate([y[i] for i in idx])),
                                  train_eval_ts=fit, discrepancy_vector=lambda x, y, lam: np.arange(5.),
                                  evaluate_outer=Mock(side_effect=evaluate))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.cache(root / "cache")
            manifest = analysis.preflight(root / "cache", settings, "p", self.channels)["1"]
            identity = {"subject": 1, "protocol_sha256": "p", "code_sha256": "c", "cache_sha256": manifest["sha256"]}
            args = (1, str(root / "cache"), str(root / "out"), str(root / "unused"), settings, identity, manifest)
            result = analysis.process_subject(*args, backend=backend)
            self.assertTrue(result["complete"])
            self.assertEqual(len(bootstrap_calls), 3)  # one full-target, one shared call per split
            self.assertEqual(len(selected_calls), 3)
            self.assertEqual(len(fit_calls), 8)  # three unique source pools + target-only per split
            self.assertEqual([c[1] for c in bootstrap_calls], [499, 199, 199])
            self.assertTrue(all(c[2:] == (.1, True) for c in bootstrap_calls))
            for row, call in zip(result["downstream"], bootstrap_calls[1:]):
                self.assertFalse(set(row["screen_indices"]) & set(row["test_indices"]))
                self.assertFalse(set(row["screen_trial_ids"]) & set(row["test_trial_ids"]))
                self.assertEqual(len(call[0]), 8)
                self.assertEqual(row["methods"]["top3"], row["methods"]["pair_ref"])
                self.assertEqual(row["seed"], 20260714 + 100000 + row["split"])
            self.assertEqual(result["coverage"][0]["seed"], 20260714 + 1000000)
            backend.bootstrap.side_effect = AssertionError("Completed checkpoint must not rerun")
            resumed = analysis.process_subject(*args, backend=backend)
            self.assertEqual(resumed, json.loads((root / "out/subjects/sub-01.json").read_text()))
            # A partial checkpoint resumes only missing outer samples.
            resumed["coverage"] = resumed["coverage"][:1]
            resumed["complete"] = False
            analysis.write_json(root / "out/subjects/sub-01.json", resumed)
            before = backend.evaluate_outer.call_count
            partial = analysis.process_subject(*args, backend=backend)
            self.assertTrue(partial["complete"])
            self.assertEqual(backend.evaluate_outer.call_count - before, 1)


if __name__ == "__main__":
    unittest.main()
