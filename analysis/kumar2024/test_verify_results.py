"""Synthetic-only checks for the independent post-run verifier."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import t
from sklearn.model_selection import StratifiedShuffleSplit

SPEC = importlib.util.spec_from_file_location("kumar_verifier", Path(__file__).with_name("verify_results.py"))
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def synthetic_records():
    checkpoints, caches = [], {}
    settings = {"seed": 20260714}
    for subject in range(1, 19):
        caches[subject] = {}
        for session in range(1, 7):
            caches[subject][session] = {"covs": np.repeat(np.eye(22)[None], 16, axis=0),
                                       "labels": np.array([0, 1] * 8),
                                       "trial_ids": np.array([f"s{subject}/session{session}/trial{i}" for i in range(16)])}
        choices = {"all_sources": [0, 1, 2, 3, 4], "top1": [0], "top3": [0, 1, 2],
                   "mcs_stepdown": [0, 1], "pair_ref": [0, 1], "target_only": []}
        def selection(method):
            indices = choices[method]
            return {"source_indices": indices[:], "source_sessions": [i + 1 for i in indices],
                    "set_size": len(indices), "training_trials": 8 if method == "target_only" else 16 * len(indices)}
        target = caches[subject][6]
        rows = []
        splitter = StratifiedShuffleSplit(n_splits=30, test_size=.5, random_state=20260714 + subject)
        for split, (screen, test) in enumerate(splitter.split(np.zeros(16), target["labels"])):
            seed = 20260714 + subject * 100000 + split
            boot = np.random.default_rng(seed).normal(size=(199, 5))
            rows.append({"split": split, "splitter_seed": 20260714 + subject, "seed": seed,
                         "bootstrap_repetitions": 199, "screen_indices": screen.tolist(), "test_indices": test.tolist(),
                         "screen_trial_ids": target["trial_ids"][screen].tolist(), "test_trial_ids": target["trial_ids"][test].tolist(),
                         "discrepancy_estimates": [0., 1., 2., 3., 4.],
                         "shared_bootstrap_matrix_sha256": hashlib.sha256(np.asarray(boot, dtype="<f8").tobytes()).hexdigest(),
                         "methods": {m: {**selection(m), "balanced_accuracy": .75} for m in verifier.METHODS}})
        outer = [{"replicate": rep, "seed": 20260714 + subject * 1000000 + rep, "bootstrap_repetitions": 499,
                  "source_indices": [0, 1], "source_sessions": [1, 2], "set_size": 2, "retained_fraction": .4,
                  "coverage_c": 1, "coverage_d": 1, "coverage_joint": 1, "oracle_retained": 1,
                  "exact_recovery": 0, "worst_reference_excess": 1.} for rep in range(50)]
        checkpoint = {"subject": subject, "raw_subject": subject if subject <= 9 else subject + 1,
                      "group": "GR" if subject <= 9 else "PAR", "complete": True,
                      "target_session": 6, "source_sessions": [1, 2, 3, 4, 5], "downstream": rows, "coverage": outer,
                      "screening": {"seed": 20260714 + subject * 10000, "bootstrap_repetitions": 499,
                                    "target_trials": 16, "discrepancy_estimates": [0., 1., 2., 3., 4.],
                                    "shared_bootstrap_matrix_sha256": "a" * 64,
                                    "methods": {m: selection(m) for m in verifier.SOURCE_METHODS}},
                      "empirical_reference": {"discrepancies": [0., 1., 2., 3., 4.], "oracle_source_indices": [0],
                                              "source_fraction": 1., "target_fraction": .5,
                                              "outer_source_sample_sizes": [16] * 5, "outer_target_sample_size": 8}}
        checkpoint["subject_means"] = verifier.independent_subject_mean(checkpoint)
        checkpoints.append(checkpoint)
    return checkpoints, caches, settings


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.caches, cls.settings = synthetic_records()

    def test_all_fixed_record_counts_and_partitions(self):
        self.assertEqual(verifier.verify_records(self.records, self.caches, self.settings),
                         {"subjects": 18, "splits": 540, "method_rows": 3240, "outer_samples": 900})

    def test_cache_class_minimum_comes_from_protocol(self):
        cases = ((4, (6, 7), True), (4, (7, 6), True), (4, (4, 4), True),
                 (4, (3, 7), False), (4, (7, 3), False), (8, (6, 7), False))
        for minimum, counts, should_pass in cases:
            with self.subTest(minimum=minimum, counts=counts), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                channels = [f"EEG{i}" for i in range(22)]
                protocol = {"analysis": {"min_trials_per_class": minimum},
                            "preprocessing": {"eeg_channels": channels}}
                protocol_hash = "synthetic-protocol-hash"
                labels = np.array([0] * counts[0] + [1] * counts[1])
                metadata = {}
                for session in range(1, 7):
                    relative = f"sub-01/session-{session:02d}_cov.npz"
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    ids = np.array([f"subject1/session{session}/trial{i}" for i in range(len(labels))])
                    np.savez(path, covs=np.repeat(np.eye(22)[None], len(labels), axis=0),
                             labels=labels, trial_ids=ids, protocol_sha256=protocol_hash,
                             unit="microvolt_squared", channels=np.array(channels))
                    metadata[str(session)] = {
                        "relative_path": relative, "sha256": verifier.digest_file(path),
                        "n_trials": len(labels), "class_counts": list(counts),
                        "trial_ids": ids.tolist(), "protocol_sha256": protocol_hash,
                        "unit": "microvolt_squared", "channels": channels}
                item = {"sessions": metadata, "sha256": verifier.digest_object(metadata)}
                checkpoint = {"subject": 1, "cache_manifest": item,
                              "identity": {"cache_sha256": item["sha256"]}}
                cache_manifest = {"1": item}
                manifest = {"cache": cache_manifest,
                            "identity": {"cache_sha256": verifier.digest_object(cache_manifest)}}
                arguments = (protocol, protocol_hash, root, [checkpoint], manifest)
                if should_pass:
                    caches = verifier.load_caches(*arguments)
                    self.assertEqual(len(caches[1]), 6)
                    self.assertEqual(len(caches[1][1]["labels"]), sum(counts))
                else:
                    with self.assertRaisesRegex(verifier.VerificationError, f"fewer than {minimum} trials per class"):
                        verifier.load_caches(*arguments)

    def test_detects_missing_subject_split_and_method(self):
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_records(self.records[:-1], self.caches, self.settings)
        records = copy.deepcopy(self.records)
        records[0]["downstream"].pop()
        with self.assertRaisesRegex(verifier.VerificationError, "missing or extra splits"):
            verifier.verify_records(records, self.caches, self.settings)
        records = copy.deepcopy(self.records)
        del records[0]["downstream"][0]["methods"]["target_only"]
        with self.assertRaisesRegex(verifier.VerificationError, "methods"):
            verifier.verify_records(records, self.caches, self.settings)

    def test_detects_trial_leakage_and_wrong_selection_size(self):
        records = copy.deepcopy(self.records)
        records[0]["downstream"][0]["screen_indices"][0] = records[0]["downstream"][0]["test_indices"][0]
        with self.assertRaisesRegex(verifier.VerificationError, "overlap"):
            verifier.verify_records(records, self.caches, self.settings)
        records = copy.deepcopy(self.records)
        records[0]["downstream"][0]["methods"]["pair_ref"]["set_size"] = 3
        with self.assertRaisesRegex(verifier.VerificationError, "set_size"):
            verifier.verify_records(records, self.caches, self.settings)

    def test_detects_inconsistent_joint_event_and_oracle_retention(self):
        for field, value, error in (("coverage_joint", 0, "joint_intersection"), ("oracle_retained", 0, "oracle_retained")):
            records = copy.deepcopy(self.records)
            records[0]["coverage"][0][field] = value
            with self.assertRaisesRegex(verifier.VerificationError, error):
                verifier.verify_records(records, self.caches, self.settings)

    def test_statistics_use_eighteen_subjects_not_540_splits(self):
        differences = np.arange(18) / 1000 - .01
        result = verifier.independent_paired(differences.tolist(), True)
        se = np.std(differences, ddof=1) / np.sqrt(18)
        self.assertEqual(result["n_subjects"], 18)
        self.assertAlmostEqual(result["two_sided_95_ci"][0], differences.mean() - t.ppf(.975, 17) * se)
        self.assertAlmostEqual(result["noninferiority"]["0.02"]["one_sided_p"],
                               t.sf((differences.mean() + .02) / se, 17))
        self.assertEqual(verifier.independent_paired(differences.tolist(), False)["noninferiority"], {})

    def test_recomputes_summary_and_detects_corruption(self):
        means = [record["subject_means"] for record in self.records]
        summary = {"subject_means": copy.deepcopy(means), "overall": verifier.independent_group(means),
                   "primary_refinement_minus_all_sources": verifier.independent_paired([0.] * 18, True),
                   "refinement_minus_mcs": verifier.independent_paired([0.] * 18, False),
                   "group_descriptive_only": {g: verifier.independent_group([s for s in means if s["group"] == g]) for g in ("GR", "PAR")},
                   "status": "complete", "analysis_role": "exploratory"}
        result = verifier.verify_summaries(self.records, summary)
        self.assertEqual(result["overall"]["methods"]["pair_ref"]["balanced_accuracy"], .75)
        summary["overall"]["methods"]["pair_ref"]["balanced_accuracy"] = .76
        with self.assertRaisesRegex(verifier.VerificationError, "balanced_accuracy"):
            verifier.verify_summaries(self.records, summary)

    def test_fixed_replay_uses_exact_draws_and_refits_all_six_methods(self):
        fits = []
        def bootstrap(sources, target, rng, budget, lam, shared):
            self.assertEqual((budget, lam, shared), (199, .1, True))
            self.assertEqual(len(target), 8)
            return np.arange(5.), rng.normal(size=(199, 5))
        def fit(x, y, xt, yt):
            fits.append(len(x))
            return .75
        backend = SimpleNamespace(bootstrap=bootstrap,
                                  selected_sets=lambda *args: {"top1": np.array([0]), "top3": np.array([0, 1, 2]),
                                                               "mcs_stepdown": np.array([0, 1]), "pair_ref": np.array([0, 1])},
                                  pool=lambda idx, x, y: (np.concatenate([x[i] for i in idx]), np.concatenate([y[i] for i in idx])),
                                  train_eval_ts=fit)
        report = verifier.audit_fixed_splits(self.records, self.caches, self.settings, backend)
        self.assertEqual([r["subject"] for r in report], [1, 9, 10, 18])
        self.assertEqual(len(fits), 24)
        records = copy.deepcopy(self.records)
        records[0]["downstream"][0]["shared_bootstrap_matrix_sha256"] = "wrong"
        with self.assertRaisesRegex(verifier.VerificationError, "bootstrap matrix hash"):
            verifier.audit_fixed_splits(records, self.caches, self.settings, backend)
        records = copy.deepcopy(self.records)
        records[0]["downstream"][0]["methods"]["pair_ref"]["balanced_accuracy"] = .5
        with self.assertRaisesRegex(verifier.VerificationError, "accuracy"):
            verifier.audit_fixed_splits(records, self.caches, self.settings, backend)

    def test_failure_report_has_nonzero_exit_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = ["--protocol", str(root / "missing.json"), "--results-root", str(root / "results"),
                         "--cache-root", str(root / "cache"), "--simulations-root", str(root / "simulations")]
            with patch.object(verifier, "verify", side_effect=verifier.VerificationError("synthetic failure")):
                self.assertEqual(verifier.main(arguments), 1)
            report = json.loads((root / "results/verification.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("synthetic failure", report["error"])

    def test_end_to_end_synthetic_files_and_hashes(self):
        records = copy.deepcopy(self.records)
        backend = SimpleNamespace(
            bootstrap=lambda sources, target, rng, budget, lam, shared: (np.arange(5.), rng.normal(size=(199, 5))),
            selected_sets=lambda *args: {"top1": np.array([0]), "top3": np.array([0, 1, 2]),
                                         "mcs_stepdown": np.array([0, 1]), "pair_ref": np.array([0, 1])},
            pool=lambda idx, x, y: (np.concatenate([x[i] for i in idx]), np.concatenate([y[i] for i in idx])),
            train_eval_ts=lambda *args: .75, discrepancy_vector=lambda *args: np.arange(5.))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            protocol_path.write_bytes(Path(__file__).with_name("protocol_minclass4.json").read_bytes())
            protocol = json.loads(protocol_path.read_text())
            protocol_hash = verifier.digest_file(protocol_path)
            simulations = root / "simulations"
            simulations.mkdir()
            code_files = {"analyze.py": verifier.digest_file(Path(__file__).with_name("analyze.py"))}
            for name in ("ci_gate_bnci004_riemann", "ci_gate_ma2020_riemann", "ci_gate_eeg_empirical_audit"):
                path = simulations / f"{name}.py"
                path.write_text("# Synthetic snapshot fixture. Backend is injected in this test.\n")
                code_files[f"simulations/{name}.py"] = verifier.digest_file(path)
            code_hash = verifier.digest_object(code_files)
            cache_manifest = {}
            for checkpoint in records:
                subject = checkpoint["subject"]
                metadata = {}
                for session in range(1, 7):
                    relative = f"sub-{subject:02d}/session-{session:02d}_cov.npz"
                    path = root / "cache" / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    data = self.caches[subject][session]
                    np.savez(path, **data, protocol_sha256=protocol_hash, unit="microvolt_squared",
                             channels=np.array(protocol["preprocessing"]["eeg_channels"]))
                    metadata[str(session)] = {"relative_path": relative, "sha256": verifier.digest_file(path),
                                              "n_trials": 16, "class_counts": [8, 8],
                                              "trial_ids": data["trial_ids"].tolist(), "protocol_sha256": protocol_hash,
                                              "unit": "microvolt_squared", "channels": protocol["preprocessing"]["eeg_channels"]}
                item = {"subject": subject, "raw_subject": checkpoint["raw_subject"], "group": checkpoint["group"],
                        "sessions": metadata, "sha256": verifier.digest_object(metadata)}
                cache_manifest[str(subject)] = item
                checkpoint["cache_manifest"] = item
                checkpoint["schema_version"] = 1
                checkpoint["identity"] = {"subject": subject, "protocol_sha256": protocol_hash,
                                          "code_sha256": code_hash, "cache_sha256": item["sha256"],
                                          "environment": {"python": "synthetic-test"}}
            identity = {"protocol_sha256": protocol_hash, "code_sha256": code_hash,
                        "cache_sha256": verifier.digest_object(cache_manifest), "environment": {"python": "synthetic-test"}}
            manifest = {"protocol": protocol, "identity": identity, "cache": cache_manifest,
                        "code": {"files": code_files, "sha256": code_hash}}
            results = root / "results"
            verifier.write_report(results / "manifest.json", manifest)
            for checkpoint in records:
                verifier.write_report(results / "subjects" / f"sub-{checkpoint['subject']:02d}.json", checkpoint)
            means = [record["subject_means"] for record in records]
            summary = {"identity": identity, "subject_means": means, "overall": verifier.independent_group(means),
                       "primary_refinement_minus_all_sources": verifier.independent_paired([0.] * 18, True),
                       "refinement_minus_mcs": verifier.independent_paired([0.] * 18, False),
                       "group_descriptive_only": {g: verifier.independent_group([s for s in means if s["group"] == g]) for g in ("GR", "PAR")},
                       "status": "complete", "analysis_role": "exploratory"}
            verifier.write_report(results / "summary.json", summary)
            arguments = ["--protocol", str(protocol_path), "--results-root", str(results),
                         "--cache-root", str(root / "cache"), "--simulations-root", str(simulations)]
            with patch.object(verifier, "load_backend", return_value=backend):
                self.assertEqual(verifier.main(arguments), 0)
            report = json.loads((results / "verification.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["counts"]["method_rows"], 3240)
            # Corrupt one cache byte: the verifier must reject it before classification replay.
            path = root / "cache/sub-01/session-01_cov.npz"
            path.write_bytes(path.read_bytes() + b"synthetic-corruption")
            with patch.object(verifier, "load_backend", side_effect=AssertionError("Must fail before replay")):
                self.assertEqual(verifier.main(arguments), 1)
            report = json.loads((results / "verification.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("digest", report["error"])


if __name__ == "__main__":
    unittest.main()
