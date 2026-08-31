#!/usr/bin/env python3
"""Verify the compact result release and its reconstructable main figures.

The verification has four layers: SHA-256 and manifest integrity, required CSV
and JSON schemas, independent reconstruction of Figure 1/2 source values, and
subject-level recomputation of the reported non-inferiority statistics.  Figure
3 is explicitly treated as an archival output because the compact release does
not contain its bootstrap interval endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from scipy import stats

from rebuild_figures import (
    OVERVIEW_COLUMNS,
    compute_eeg_overview,
    read_csv,
    verify_reconstructed_overview,
)


ROOT = Path(__file__).resolve().parents[1]
HEX64 = re.compile(r"^[0-9a-f]{64}$")
INTEGRITY_FILES = {"MANIFEST.csv", "SHA256SUMS"}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def relative_result_path(raw: str, input_dir: Path) -> Path:
    normalized = raw.strip().replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe manifest path: {raw!r}")
    parts = list(pure.parts)
    if parts and parts[0] == "results":
        parts = parts[1:]
    if not parts:
        raise ValueError(f"invalid manifest path: {raw!r}")
    candidate = input_dir.joinpath(*parts).resolve()
    try:
        candidate.relative_to(input_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes input directory: {raw!r}") from exc
    return candidate


def verify_integrity(input_dir: Path) -> int:
    manifest_path = input_dir / "MANIFEST.csv"
    checksums_path = input_dir / "SHA256SUMS"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise FileNotFoundError("results/MANIFEST.csv and results/SHA256SUMS are required")

    manifest = pd.read_csv(manifest_path, dtype={"path": str, "sha256": str})
    required = {"path", "bytes", "sha256"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {', '.join(missing)}")
    if manifest["path"].duplicated().any():
        duplicates = manifest.loc[manifest["path"].duplicated(), "path"].tolist()
        raise ValueError(f"duplicate manifest paths: {duplicates}")

    checksum_rows: dict[Path, str] = {}
    for line_number, line in enumerate(
        checksums_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", line)
        if match is None:
            raise ValueError(f"invalid SHA256SUMS line {line_number}: {line!r}")
        digest, raw_path = match.groups()
        path = relative_result_path(raw_path, input_dir)
        if path in checksum_rows:
            raise ValueError(f"duplicate SHA256SUMS path: {raw_path}")
        checksum_rows[path] = digest.lower()

    manifest_rows: dict[Path, tuple[int, str]] = {}
    for row in manifest.itertuples(index=False):
        path = relative_result_path(str(row.path), input_dir)
        digest = str(row.sha256).lower()
        if not HEX64.fullmatch(digest):
            raise ValueError(f"invalid manifest digest for {row.path}: {row.sha256}")
        try:
            size = int(row.bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid byte size for {row.path}: {row.bytes}") from exc
        manifest_rows[path] = (size, digest)

    if set(manifest_rows) != set(checksum_rows):
        only_manifest = sorted(str(path) for path in set(manifest_rows) - set(checksum_rows))
        only_sha = sorted(str(path) for path in set(checksum_rows) - set(manifest_rows))
        raise ValueError(
            f"manifest/checksum inventory mismatch; only manifest={only_manifest}, only SHA={only_sha}"
        )

    actual_files = {
        path.resolve()
        for path in input_dir.rglob("*")
        if path.is_file() and path.name not in INTEGRITY_FILES
    }
    if actual_files != set(manifest_rows):
        unlisted = sorted(str(path) for path in actual_files - set(manifest_rows))
        missing_files = sorted(str(path) for path in set(manifest_rows) - actual_files)
        raise ValueError(
            f"result inventory mismatch; unlisted={unlisted}, missing={missing_files}"
        )

    for path, (expected_size, expected_digest) in manifest_rows.items():
        if not path.is_file():
            raise FileNotFoundError(f"manifest file is missing: {path}")
        actual_size = path.stat().st_size
        actual_digest = sha256(path)
        if actual_size != expected_size:
            raise ValueError(
                f"size mismatch for {path}: expected {expected_size}, found {actual_size}"
            )
        if actual_digest != expected_digest or checksum_rows[path] != expected_digest:
            raise ValueError(f"SHA-256 mismatch for {path}")
    print(f"[PASS] integrity: {len(manifest_rows)} compact result files")
    return len(manifest_rows)


def assert_probability_columns(data: pd.DataFrame, columns: list[str], path: Path) -> None:
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{path}:{column} contains values outside [0, 1]")


def verify_figure_one_sources(input_dir: Path) -> None:
    straddle_path = input_dir / "simulations" / "straddle_summary.csv"
    target_path = input_dir / "simulations" / "target_limited_audit_summary.csv"

    straddle_required = {
        "geometry",
        "n_target",
        "cover_shared",
        "cover_wrong",
        "var_ratio",
        "replicates",
    }
    target_required = {
        "scenario",
        "n_target",
        "n_source",
        "shared_ref_size",
        "wrong_ref_size",
    }
    straddle = read_csv(straddle_path, straddle_required)
    target = read_csv(target_path, target_required)

    assert_probability_columns(straddle, ["cover_shared", "cover_wrong"], straddle_path)
    if set(straddle["geometry"]) != {"same_side", "straddle"}:
        raise ValueError(f"unexpected geometries in {straddle_path}")
    if (target[["n_target", "n_source", "shared_ref_size", "wrong_ref_size"]] <= 0).any().any():
        raise ValueError(f"non-positive sample or set size in {target_path}")
    print("[PASS] Figure 1 data: simulation summaries satisfy the plotting schema")


def subject_differences(path: Path) -> np.ndarray:
    data = read_csv(path, {"subject", "method", "balanced_accuracy"})
    if "analysis" in data.columns:
        data = data[data["analysis"] == "primary"]
    per_subject = (
        data.groupby(["subject", "method"], as_index=False)["balanced_accuracy"]
        .mean()
        .pivot(index="subject", columns="method", values="balanced_accuracy")
    )
    missing_methods = sorted({"all_sources", "pair_ref"} - set(per_subject.columns))
    if missing_methods:
        raise ValueError(f"{path} is missing methods: {', '.join(missing_methods)}")
    differences = (
        per_subject["pair_ref"] - per_subject["all_sources"]
    ).dropna().to_numpy(float)
    if differences.size < 2 or not np.isfinite(differences).all():
        raise ValueError(f"{path} has insufficient or non-finite paired differences")
    return differences


def noninferiority_statistics(differences: np.ndarray, margin: float) -> dict[str, object]:
    n = differences.size
    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / np.sqrt(n))
    if standard_error == 0:
        t_p = 0.0 if mean > -margin else 1.0
    else:
        t_p = float(stats.t.sf((mean + margin) / standard_error, n - 1))
    t_critical = stats.t.ppf(0.975, n - 1)
    ci_low = float(mean - t_critical * standard_error)
    ci_high = float(mean + t_critical * standard_error)
    try:
        wilcoxon_p = float(
            stats.wilcoxon(differences + margin, alternative="greater").pvalue
        )
    except ValueError:
        wilcoxon_p = float("nan")
    return {
        "n_subjects": int(n),
        "mean_diff": mean,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "delta_NI": float(margin),
        "noninferior_at_delta": bool(ci_low > -margin),
        "p_noninferiority_ttest": t_p,
        "p_noninferiority_wilcoxon": wilcoxon_p,
        "frac_subjects_within_margin": float((differences > -margin).mean()),
    }


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot interpret as Boolean: {value!r}")


def compare_statistic(dataset: str, field: str, observed: object, expected: object) -> None:
    if isinstance(expected, bool):
        if as_bool(observed) != expected:
            raise ValueError(f"{dataset}:{field} mismatch: {observed!r} != {expected!r}")
        return
    if isinstance(expected, int):
        if int(observed) != expected:
            raise ValueError(f"{dataset}:{field} mismatch: {observed!r} != {expected!r}")
        return
    observed_float = float(observed)
    expected_float = float(expected)
    if math.isnan(expected_float):
        if not math.isnan(observed_float):
            raise ValueError(f"{dataset}:{field} expected NaN")
    elif not math.isclose(observed_float, expected_float, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(
            f"{dataset}:{field} mismatch: {observed_float:.16g} != {expected_float:.16g}"
        )


def verify_noninferiority(input_dir: Path) -> None:
    ni_dir = input_dir / "eeg" / "noninferiority"
    summary_path = ni_dir / "summary.json"
    sensitivity_path = ni_dir / "margin_sensitivity.csv"
    manifest_path = ni_dir / "manifest.json"
    for path in (summary_path, sensitivity_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required non-inferiority artifact is missing: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ni_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary_margin = float(ni_manifest.get("primary_margin", 0.02))
    if not math.isclose(primary_margin, 0.02, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"unexpected primary non-inferiority margin: {primary_margin}")

    datasets = {
        "ma2020": input_dir / "eeg" / "ma2020" / "downstream_by_subject.csv",
        "stieger2021": input_dir / "eeg" / "stieger2021" / "downstream_by_subject.csv",
        "zhou2020": input_dir / "eeg" / "zhou2020" / "downstream_by_subject.csv",
    }
    differences: dict[str, np.ndarray] = {}
    required_fields = {
        "n_subjects",
        "mean_diff",
        "ci95_low",
        "ci95_high",
        "delta_NI",
        "noninferior_at_delta",
        "p_noninferiority_ttest",
        "p_noninferiority_wilcoxon",
        "frac_subjects_within_margin",
    }
    if set(summary) != set(datasets):
        raise ValueError(
            f"non-inferiority summary datasets {sorted(summary)} != {sorted(datasets)}"
        )
    for dataset, path in datasets.items():
        differences[dataset] = subject_differences(path)
        expected = noninferiority_statistics(differences[dataset], primary_margin)
        observed = summary[dataset]
        missing = sorted(required_fields - set(observed))
        if missing:
            raise ValueError(f"{summary_path}:{dataset} is missing keys: {', '.join(missing)}")
        for field, expected_value in expected.items():
            compare_statistic(dataset, field, observed[field], expected_value)
        logical_status = float(observed["ci95_low"]) > -float(observed["delta_NI"])
        if as_bool(observed["noninferior_at_delta"]) != logical_status:
            raise ValueError(f"{dataset}: non-inferiority decision contradicts its CI and margin")

    sensitivity = read_csv(sensitivity_path, {"dataset", "delta_NI"} | required_fields)
    if sensitivity.duplicated(["dataset", "delta_NI"]).any():
        raise ValueError(f"duplicate dataset/margin rows in {sensitivity_path}")
    for row in sensitivity.to_dict(orient="records"):
        dataset = str(row["dataset"])
        if dataset not in differences:
            raise ValueError(f"unexpected dataset in {sensitivity_path}: {dataset}")
        expected = noninferiority_statistics(differences[dataset], float(row["delta_NI"]))
        for field, expected_value in expected.items():
            compare_statistic(dataset, field, row[field], expected_value)
    print("[PASS] non-inferiority: subject-level statistics, margins, and decisions reproduced")


def verify_figure_two(input_dir: Path) -> None:
    computed = compute_eeg_overview(input_dir)
    verify_reconstructed_overview(input_dir, computed)
    if list(computed.columns) != OVERVIEW_COLUMNS:
        raise ValueError("Figure 2 overview columns are out of order")
    if not np.allclose(
        computed["retained_fraction"],
        computed["retained_count"] / computed["candidate_count"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Figure 2 retained fractions do not equal retained/candidate")
    expected_status = {
        "Ma2020": "not established",
        "Stieger2021": "established",
        "Zhou2020": "established",
        "BNCI2014_004": "not tested",
    }
    observed_status = computed.set_index("dataset")["two_pp_noninferiority"].to_dict()
    if observed_status != expected_status:
        raise ValueError(f"unexpected Figure 2 NI status: {observed_status}")
    print("[PASS] Figure 2 data: EEG summaries and paired inference reproduce plotted values")


def verify_archival_figure_three(input_dir: Path, figure_dir: Path) -> None:
    metadata_path = input_dir / "eeg" / "empirical_reference" / "manifest.json"
    figure_path = figure_dir / "fig_zhou2020_exclusion_certificate.pdf"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"archival Figure 3 metadata is missing: {metadata_path}")
    if not figure_path.is_file():
        raise FileNotFoundError(f"archival Figure 3 PDF is missing: {figure_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_top = {
        "script",
        "bootstrap_repetitions",
        "alpha",
        "alpha_component",
        "alpha_contrast",
        "shrinkage",
        "seed",
        "certificate",
    }
    missing_top = sorted(required_top - set(metadata))
    if missing_top:
        raise ValueError(f"{metadata_path} is missing keys: {', '.join(missing_top)}")
    certificate = metadata["certificate"]
    required_certificate = {
        "subject",
        "target_session",
        "source_sessions",
        "retained_sessions",
        "bootstrap_repetitions",
        "threshold",
    }
    missing_certificate = sorted(required_certificate - set(certificate))
    if missing_certificate:
        raise ValueError(
            f"{metadata_path}:certificate is missing keys: {', '.join(missing_certificate)}"
        )
    sources = set(certificate["source_sessions"])
    retained = set(certificate["retained_sessions"])
    if not sources or not retained.issubset(sources):
        raise ValueError("Figure 3 retained sessions are not a subset of source sessions")
    if not math.isfinite(float(certificate["threshold"])):
        raise ValueError("Figure 3 threshold is not finite")
    pdf = figure_path.read_bytes()
    if len(pdf) < 1024 or not pdf.startswith(b"%PDF-") or b"%%EOF" not in pdf[-2048:]:
        raise ValueError(f"invalid archival PDF: {figure_path}")
    print("[PASS] Figure 3 archival output and metadata verified (not regenerated)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "results",
        help="compact result root (default: results)",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "figures",
        help="figure directory containing the archival Figure 3 PDF (default: figures)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    verify_integrity(input_dir)
    verify_figure_one_sources(input_dir)
    verify_noninferiority(input_dir)
    verify_figure_two(input_dir)
    verify_archival_figure_three(input_dir, figure_dir)
    print("[PASS] compact result verification complete")


if __name__ == "__main__":
    main()
