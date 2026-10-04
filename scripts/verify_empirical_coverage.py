#!/usr/bin/env python3
"""Reconstruct both empirical-reference coverage summaries from released subject means.

This compact check does not recreate outer samples, bootstrap draws, or
pairwise retained-set Jaccard values. Full raw-record validation is provided by
the local analysis entry point after a complete empirical-audit run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "eeg" / "empirical_reference"
METRICS = (
    "coverage_c", "coverage_d", "coverage_joint", "oracle_retained",
    "exact_recovery", "set_size",
)
DESIGNS = (
    ("Kumar2024", "", 18, 50, "outer_samples_per_subject"),
    ("BNCI2014_004", "bnci_", 9, 200, "bnci_outer_samples_per_subject"),
)


def compute_empirical_coverage(input_dir: Path) -> pd.DataFrame:
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["subjects"] != list(range(1, 19)):
        raise ValueError("Kumar2024: manifest must contain the complete 18-subject audit")
    expected_settings = {
        "bootstrap_repetitions": 499,
        "alpha": 0.05,
        "alpha_component": 0.025,
        "alpha_contrast": 0.025,
        "outer_source_sample_fraction": 1.0,
        "outer_target_sample_fraction": 0.5,
    }
    for field, expected in expected_settings.items():
        if manifest.get(field) != expected:
            raise ValueError(f"empirical audit: unexpected released design value for {field}")
    rows = []
    for dataset, prefix, n_subjects, outer_per_subject, count_field in DESIGNS:
        if manifest.get(count_field) != outer_per_subject:
            raise ValueError(f"{dataset}: unexpected outer-sample count in manifest")
        path = input_dir / f"{prefix}empirical_audit_by_subject.csv"
        data = pd.read_csv(path)
        if not {"subject", "replicate", "n_sources", *METRICS}.issubset(data.columns):
            raise ValueError(f"{dataset}: incomplete subject-summary columns")
        if data["subject"].duplicated().any() or sorted(data["subject"]) != list(range(1, n_subjects + 1)):
            raise ValueError(f"{dataset}: missing, repeated, or unexpected subjects")
        if not np.allclose(data["replicate"], (outer_per_subject - 1) / 2, rtol=0, atol=1e-12):
            raise ValueError(f"{dataset}: replicate-ID mean is inconsistent with the manifest")
        for field in METRICS[:-1]:
            values = data[field].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"{dataset}: {field} is outside [0, 1]")
            successes = values * outer_per_subject
            if not np.allclose(successes, np.rint(successes), rtol=0, atol=1e-9):
                raise ValueError(f"{dataset}: {field} is incompatible with the outer-sample count")
        if not data["set_size"].between(1, data["n_sources"]).all():
            raise ValueError(f"{dataset}: retained-set size is outside the source count")
        if (data["coverage_joint"] > data[["coverage_c", "coverage_d"]].min(axis=1) + 1e-12).any():
            raise ValueError(f"{dataset}: joint coverage exceeds a component event frequency")
        rows.append({
            "dataset": dataset,
            "subjects": n_subjects,
            "outer_per_subject": outer_per_subject,
            "outer_total": n_subjects * outer_per_subject,
            "bootstrap_repetitions": manifest["bootstrap_repetitions"],
            "nominal_component": 1 - manifest["alpha_component"],
            "nominal_contrast": 1 - manifest["alpha_contrast"],
            "nominal_joint": 1 - manifest["alpha_component"] - manifest["alpha_contrast"],
            **{field: float(data[field].mean()) for field in METRICS},
        })
    return pd.DataFrame(rows)


def verify_empirical_coverage(input_dir: Path = DEFAULT_INPUT) -> dict:
    reconstructed = compute_empirical_coverage(input_dir)
    released = pd.read_csv(input_dir / "coverage_summary.csv")
    if list(released.columns) != list(reconstructed.columns):
        raise ValueError("empirical coverage summary columns differ from the reconstructed schema")
    if released["dataset"].duplicated().any() or set(released["dataset"]) != set(reconstructed["dataset"]):
        raise ValueError("empirical coverage summary must contain both datasets exactly once")
    released = released.set_index("dataset").loc[reconstructed["dataset"]]
    expected = reconstructed.set_index("dataset")
    if not np.allclose(released.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-12):
        raise ValueError("empirical coverage summary differs from subject means or nominal design levels")
    return {
        "status": "passed",
        "datasets": reconstructed["dataset"].tolist(),
        "subject_rows_checked": int(reconstructed["subjects"].sum()),
        "outer_counts_from_manifest": reconstructed["outer_total"].tolist(),
        "summary_rows_recomputed": len(reconstructed),
        "not_reconstructed": ["outer-sample records", "bootstrap draws", "retained-set Jaccard values"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-summary", type=Path,
        help="write a reconstructed two-row summary to this explicit path instead of verifying the released summary",
    )
    args = parser.parse_args()
    if args.output_summary is not None:
        summary = compute_empirical_coverage(args.input_dir)
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_summary, index=False)
        print(f"[PASS] empirical coverage: reconstructed {len(summary)} dataset rows")
    else:
        print(json.dumps(verify_empirical_coverage(args.input_dir), indent=2))


if __name__ == "__main__":
    main()
