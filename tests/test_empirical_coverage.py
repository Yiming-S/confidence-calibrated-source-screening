"""Compact-coverage reporting tests independent of EEG and full audit records."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_empirical_coverage import DEFAULT_INPUT, verify_empirical_coverage


class EmpiricalCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.results = Path(self.temporary.name) / "empirical_reference"
        shutil.copytree(DEFAULT_INPUT, self.results)

    def test_both_cohorts_reconstruct_without_raw_records_or_input_changes(self) -> None:
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.results.iterdir()}
        self.assertFalse(list(self.results.glob("*audit_raw.csv")))
        report = verify_empirical_coverage(self.results)
        self.assertEqual(report["datasets"], ["Kumar2024", "BNCI2014_004"])
        self.assertEqual(report["subject_rows_checked"], 27)
        self.assertEqual(report["outer_counts_from_manifest"], [900, 1800])
        self.assertEqual(report["summary_rows_recomputed"], 2)
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.results.iterdir()}
        self.assertEqual(before, after)

    def test_missing_kumar_input_is_rejected(self) -> None:
        (self.results / "empirical_audit_by_subject.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            verify_empirical_coverage(self.results)

    def test_incomplete_subject_set_is_rejected(self) -> None:
        path = self.results / "bnci_empirical_audit_by_subject.csv"
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "unexpected subjects"):
            verify_empirical_coverage(self.results)

    def test_changed_outer_count_is_rejected(self) -> None:
        path = self.results / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["outer_samples_per_subject"] = 100
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "outer-sample count"):
            verify_empirical_coverage(self.results)

    def test_changed_coverage_summary_is_rejected(self) -> None:
        path = self.results / "coverage_summary.csv"
        summary = pd.read_csv(path)
        summary.loc[0, "coverage_joint"] = 0.95
        summary.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "differs from subject means"):
            verify_empirical_coverage(self.results)

    def test_changed_nominal_level_is_rejected(self) -> None:
        path = self.results / "coverage_summary.csv"
        summary = pd.read_csv(path)
        summary.loc[0, "nominal_component"] = 0.95
        summary.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "nominal design levels"):
            verify_empirical_coverage(self.results)

    def test_bnci_only_summary_is_rejected(self) -> None:
        path = self.results / "coverage_summary.csv"
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "both datasets exactly once"):
            verify_empirical_coverage(self.results)

    def test_subject_rate_incompatible_with_outer_count_is_rejected(self) -> None:
        path = self.results / "empirical_audit_by_subject.csv"
        summary = pd.read_csv(path)
        summary.loc[0, "coverage_c"] -= 0.01
        summary.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "incompatible with the outer-sample count"):
            verify_empirical_coverage(self.results)

    def test_full_local_table_mode_uses_portable_synthetic_records_without_cache(self) -> None:
        sys.path.insert(0, str(ROOT / "analysis"))
        import ci_gate_eeg_empirical_audit as audit

        local = Path(self.temporary.name) / "local_audit"
        local.mkdir()
        manifest = json.loads((self.results / "manifest.json").read_text())
        manifest.update(subjects=[1], outer_samples_per_subject=2, bnci_outer_samples_per_subject=2)
        (local / "manifest.json").write_text(json.dumps(manifest))
        for prefix, subjects in (("", [1]), ("bnci_", list(range(1, 10)))):
            raw = pd.DataFrame([
                {"subject": subject, "replicate": replicate, "n_sources": 2,
                 "coverage_c": 1, "coverage_d": replicate, "coverage_joint": replicate,
                 "oracle_retained": 1, "exact_recovery": 0, "set_size": 2}
                for subject in subjects for replicate in range(2)
            ])
            raw.to_csv(local / f"{prefix}empirical_audit_raw.csv", index=False)
            raw.groupby("subject", as_index=False).mean().to_csv(
                local / f"{prefix}empirical_audit_by_subject.csv", index=False
            )
            pd.DataFrame({"subject": subjects, "n_sources": 2, "mean_pairwise_jaccard": 1.0}).to_csv(
                local / f"{prefix}subject_stability.csv", index=False
            )
        before = {p.name: p.read_bytes() for p in local.iterdir()}
        argv = ["audit", "--tables-only", "--out-dir", str(local)]
        with patch.object(sys, "argv", argv), patch.object(
            audit, "resolve_cache_root", side_effect=AssertionError("cache resolution")
        ), patch.object(audit, "evaluate_outer", side_effect=AssertionError("experiment rerun")):
            audit.main()
        self.assertEqual(before, {name: (local / name).read_bytes() for name in before})
        table = (local / "tables" / "table_eeg_empirical_audit.tex").read_text()
        self.assertIn("Zhou2020 & 1 & 2 (2)", table)
        self.assertIn(r"BNCI2014\_004 & 9 & 18 (2)", table)


if __name__ == "__main__":
    unittest.main()
