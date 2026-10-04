#!/usr/bin/env python3
"""BSPC revision: fair downstream comparison with an MCS-style screen.

This analysis uses the cached Ma2020 and Stieger2021 trial covariances.  For
each of the 30 stratified target half-splits, one shared-target bootstrap is
computed and supplied to both the refinement gate and the MCS-style stepdown
analogue.  Every selected source subset is evaluated with the same
subset-specific tangent-space LDA pipeline on the held-out target half.

The MCS-style procedure is the source-screening analogue documented in this
repository.  It is not presented as a complete implementation of the
classical Hansen--Lunde--Nason MCS procedure.
"""

from __future__ import annotations

import os

# Keep this experiment compatible with other concurrent runs on the host.
# These values must be set before NumPy/SciPy are imported.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shlex
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from ci_gate_ma2020_riemann import (
    riemann_estimates_and_bootstrap,
    riemann_mean,
    screen_pair_ref,
    shrink,
    tangent_features,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "simulation_results" / "bspc_revision_20261002" / "mcs"
DEFAULT_CACHE_ROOTS = {
    "ma2020": ROOT / "simulation_results" / "ma2020_riemann_all_s1_s25_b499" / "cov_cache",
    "stieger2021": ROOT / "simulation_results" / "stieger_riemann_all_s1_s62_b499" / "cov_cache",
}
DATASET_DEFAULTS = {
    "ma2020": {"seed": 20260618, "subjects": "1-25", "target_session": 15},
    "stieger2021": {"seed": 20260707, "subjects": "1-62", "target_session": None},
}
METHOD_ORDER = [
    "all_sources",
    "refinement",
    "mcs_style",
    "top_m_ref",
    "range_midpoint",
]
SESSION_FILE_RE = re.compile(r"ses-(\d+)_cov\.npz$")


def parse_subjects(specification: str) -> list[int]:
    """Parse comma-separated subject numbers and inclusive ranges."""
    subjects: list[int] = []
    for part in specification.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lower, upper = (int(value) for value in part.split("-", 1))
            if upper < lower:
                raise argparse.ArgumentTypeError(f"invalid descending range: {part}")
            subjects.extend(range(lower, upper + 1))
        else:
            subjects.append(int(part))
    if not subjects:
        raise argparse.ArgumentTypeError("at least one subject is required")
    return sorted(set(subjects))


def cache_base(path: Path) -> Path:
    legacy = path / "cov_cache"
    return legacy if legacy.is_dir() else path


def subject_cache_files(
    dataset: str,
    subject: int,
    cache_root: Path,
    target_session: int | None,
) -> list[tuple[int, Path]]:
    subject_dir = (
        cache_root / f"sub-{subject:03d}"
        if dataset == "ma2020"
        else cache_root / f"S{subject}"
    )
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"missing subject cache directory: {subject_dir}")
    indexed: list[tuple[int, Path]] = []
    for path in subject_dir.glob("ses-*_cov.npz"):
        match = SESSION_FILE_RE.search(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    if dataset == "ma2020":
        if target_session is None:
            raise ValueError("Ma2020 requires --target-session")
        required = list(range(1, target_session + 1))
        observed = [session for session, _ in indexed if session <= target_session]
        if observed != required:
            raise ValueError(
                f"Ma2020 subject {subject}: expected cached sessions {required}, got {observed}"
            )
        indexed = [(session, path) for session, path in indexed if session <= target_session]
    elif len(indexed) < 6:
        raise ValueError(
            f"Stieger2021 subject {subject}: expected at least 6 cached sessions, got {len(indexed)}"
        )
    return indexed


def load_cached_subject(
    dataset: str,
    subject: int,
    cache_root: Path,
    target_session: int | None,
) -> tuple[list[int], int, list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    indexed = subject_cache_files(dataset, subject, cache_root, target_session)
    session_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for session, path in indexed:
        with np.load(path, allow_pickle=False) as archive:
            if not {"covs", "labels"}.issubset(archive.files):
                raise ValueError(f"cache lacks covs/labels arrays: {path}")
            covariances = archive["covs"].astype(float)
            labels = archive["labels"].astype(int)
        if covariances.ndim != 3 or covariances.shape[1] != covariances.shape[2]:
            raise ValueError(f"invalid covariance shape {covariances.shape}: {path}")
        if covariances.shape[0] != labels.size:
            raise ValueError(
                f"covariance/label length mismatch {covariances.shape[0]} != {labels.size}: {path}"
            )
        if not np.isfinite(covariances).all():
            raise ValueError(f"non-finite cached covariance: {path}")
        if np.unique(labels).size < 2:
            raise ValueError(f"fewer than two target classes in cache: {path}")
        session_data[session] = (covariances, labels)

    target = target_session if target_session is not None else max(session_data)
    sources = [session for session in sorted(session_data) if session < target]
    if not sources or target not in session_data:
        raise ValueError(
            f"subject {subject}: cannot form source/target split from sessions {sorted(session_data)}"
        )
    source_covariances = [session_data[session][0] for session in sources]
    source_labels = [session_data[session][1] for session in sources]
    target_covariances, target_labels = session_data[target]
    return sources, target, source_covariances, source_labels, target_covariances, target_labels


def mcs_style_stepdown(
    delta_hat: np.ndarray,
    delta_boot: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return the repository's max-t MCS-style source-screening analogue.

    One active source is removed at a time when the largest observed
    studentized excess over the active empirical best exceeds the bootstrap
    maximum over ordered active-pair contrasts.  This is not a claim to
    implement every feature of classical MCS.
    """
    active = list(range(delta_hat.size))
    centered = delta_boot - delta_hat[None, :]
    while len(active) > 1:
        active_array = np.asarray(active, dtype=int)
        best = int(active_array[np.argmin(delta_hat[active_array])])
        worst = int(active_array[np.argmax(delta_hat[active_array])])
        competitors = active_array[active_array != best]

        difference_hat = delta_hat[competitors] - delta_hat[best]
        difference_boot = centered[:, competitors] - centered[:, [best]]
        difference_se = np.maximum(np.std(difference_boot, axis=0, ddof=1), 1e-12)

        pair_i, pair_j = np.where(~np.eye(active_array.size, dtype=bool))
        pair_left = active_array[pair_i]
        pair_right = active_array[pair_j]
        pair_boot = centered[:, pair_left] - centered[:, pair_right]
        pair_se = np.maximum(np.std(pair_boot, axis=0, ddof=1), 1e-12)

        observed = float(np.max(difference_hat / difference_se))
        critical = float(
            np.quantile(np.max(pair_boot / pair_se[None, :], axis=1), 1.0 - alpha)
        )
        if observed <= critical:
            break
        active.remove(worst)
    return np.asarray(active, dtype=int)


def range_midpoint_select(delta_hat: np.ndarray) -> np.ndarray:
    lower = float(np.min(delta_hat))
    upper = float(np.max(delta_hat))
    if upper <= lower:
        return np.arange(delta_hat.size, dtype=int)
    selected = np.flatnonzero(delta_hat <= lower + 0.5 * (upper - lower))
    if selected.size == 0:
        selected = np.asarray([int(np.argmin(delta_hat))], dtype=int)
    return selected


def fit_subset_model(
    source_covariances: list[np.ndarray],
    source_labels: list[np.ndarray],
    selected: tuple[int, ...],
) -> tuple[np.ndarray, LinearDiscriminantAnalysis] | None:
    if not selected:
        return None
    train_covariances = np.concatenate(
        [source_covariances[index] for index in selected], axis=0
    )
    train_labels = np.concatenate([source_labels[index] for index in selected], axis=0)
    if np.unique(train_labels).size < 2:
        return None
    reference = riemann_mean(shrink(train_covariances, 0.05))
    train_features = tangent_features(shrink(train_covariances, 0.05), reference)
    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    classifier.fit(train_features, train_labels)
    return reference, classifier


def evaluate_subset(
    fit_cache: dict[tuple[int, ...], tuple[np.ndarray, LinearDiscriminantAnalysis] | None],
    source_covariances: list[np.ndarray],
    source_labels: list[np.ndarray],
    selected_array: np.ndarray,
    test_covariances: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    selected = tuple(sorted(int(index) for index in selected_array))
    if selected not in fit_cache:
        fit_cache[selected] = fit_subset_model(
            source_covariances, source_labels, selected
        )
    fitted = fit_cache[selected]
    if fitted is None:
        return float("nan")
    reference, classifier = fitted
    test_features = tangent_features(shrink(test_covariances, 0.05), reference)
    predictions = classifier.predict(test_features)
    return float(balanced_accuracy_score(test_labels, predictions))


def run_subject(subject: int, settings: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    dataset = str(settings["dataset"])
    cache_root = Path(settings["cache_root"])
    source_sessions, target_session, source_covariances, source_labels, target_covariances, target_labels = (
        load_cached_subject(
            dataset,
            subject,
            cache_root,
            settings.get("target_session"),
        )
    )
    source_count = len(source_covariances)
    splitter = StratifiedShuffleSplit(
        n_splits=int(settings["splits"]),
        test_size=0.5,
        random_state=int(settings["seed"]) + subject,
    )
    # This matches the per-subject bootstrap seed used by the existing simple
    # baseline run while allowing safe subject-level parallelism.
    bootstrap_seed = int(settings["seed"]) + 12345 + subject
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    fit_cache: dict[
        tuple[int, ...], tuple[np.ndarray, LinearDiscriminantAnalysis] | None
    ] = {}
    split_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []

    for split_index, (screen_indices, test_indices) in enumerate(
        splitter.split(target_covariances, target_labels)
    ):
        screen_covariances = target_covariances[screen_indices]
        test_covariances = target_covariances[test_indices]
        test_labels = target_labels[test_indices]
        delta_hat, delta_boot = riemann_estimates_and_bootstrap(
            source_covariances,
            screen_covariances,
            bootstrap_rng,
            int(settings["boot"]),
            float(settings["shrinkage"]),
            True,
        )
        refinement = screen_pair_ref(
            delta_hat,
            delta_boot,
            source_count,
            float(settings["alpha_c"]),
            float(settings["alpha_d"]),
            float(settings["alpha"]),
        )
        mcs_style = mcs_style_stepdown(
            delta_hat, delta_boot, float(settings["alpha"])
        )
        top_m_ref = np.argsort(delta_hat, kind="mergesort")[: max(1, refinement.size)]
        range_midpoint = range_midpoint_select(delta_hat)
        selections = {
            "all_sources": np.arange(source_count, dtype=int),
            "refinement": refinement,
            "mcs_style": mcs_style,
            "top_m_ref": top_m_ref,
            "range_midpoint": range_midpoint,
        }
        assignment_rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "split": split_index,
                "split_random_state": int(settings["seed"]) + subject,
                "bootstrap_seed": bootstrap_seed,
                "screen_indices": json.dumps(screen_indices.tolist(), separators=(",", ":")),
                "test_indices": json.dumps(test_indices.tolist(), separators=(",", ":")),
                "delta_hat": json.dumps(delta_hat.tolist(), separators=(",", ":")),
            }
        )
        for method in METHOD_ORDER:
            selected = np.asarray(selections[method], dtype=int)
            accuracy = evaluate_subset(
                fit_cache,
                source_covariances,
                source_labels,
                selected,
                test_covariances,
                test_labels,
            )
            selected_sessions = [source_sessions[int(index)] for index in selected]
            training_trials = int(
                sum(source_covariances[int(index)].shape[0] for index in selected)
            )
            split_rows.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "target_session": target_session,
                    "split": split_index,
                    "method": method,
                    "candidate_sessions": source_count,
                    "set_size": int(selected.size),
                    "training_trials": training_trials,
                    "balanced_accuracy": accuracy,
                    "selected_source_indices": json.dumps(
                        selected.tolist(), separators=(",", ":")
                    ),
                    "selected_session_ids": json.dumps(
                        selected_sessions, separators=(",", ":")
                    ),
                }
            )
    return {
        "subject": subject,
        "split_rows": split_rows,
        "assignment_rows": assignment_rows,
        "runtime_seconds": time.perf_counter() - start,
        "n_source_sessions": source_count,
        "n_target_trials": int(target_covariances.shape[0]),
        "n_channels": int(target_covariances.shape[1]),
    }


def safe_run_subject(subject: int, settings: dict[str, Any]) -> dict[str, Any]:
    try:
        return {"ok": True, "result": run_subject(subject, settings)}
    except Exception as error:  # preserve every failure in the run artifacts
        return {
            "ok": False,
            "subject": subject,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def subject_summary(split_results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        split_results.groupby(["dataset", "subject", "method"], as_index=False)
        .agg(
            mean_set_size=("set_size", "mean"),
            mean_training_trials=("training_trials", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            sd_balanced_accuracy_across_splits=("balanced_accuracy", "std"),
        )
        .sort_values(["subject", "method"])
    )
    all_accuracy = (
        summary[summary["method"] == "all_sources"]
        .set_index("subject")["mean_balanced_accuracy"]
    )
    ref_accuracy = (
        summary[summary["method"] == "refinement"]
        .set_index("subject")["mean_balanced_accuracy"]
    )
    summary["accuracy_diff_vs_all"] = summary.apply(
        lambda row: row["mean_balanced_accuracy"] - all_accuracy.loc[row["subject"]],
        axis=1,
    )
    summary["accuracy_diff_vs_refinement"] = summary.apply(
        lambda row: row["mean_balanced_accuracy"] - ref_accuracy.loc[row["subject"]],
        axis=1,
    )
    return summary


def method_summary(subject_results: pd.DataFrame) -> pd.DataFrame:
    return (
        subject_results.groupby(["dataset", "method"], as_index=False)
        .agg(
            subjects=("subject", "nunique"),
            mean_set_size=("mean_set_size", "mean"),
            sd_set_size=("mean_set_size", "std"),
            mean_training_trials=("mean_training_trials", "mean"),
            sd_training_trials=("mean_training_trials", "std"),
            mean_balanced_accuracy=("mean_balanced_accuracy", "mean"),
            sd_balanced_accuracy=("mean_balanced_accuracy", "std"),
            mean_accuracy_diff_vs_all=("accuracy_diff_vs_all", "mean"),
            mean_accuracy_diff_vs_refinement=("accuracy_diff_vs_refinement", "mean"),
        )
        .sort_values("method")
    )


def paired_ci_table(
    subject_results: pd.DataFrame,
    value_column: str,
    output_name: str,
) -> pd.DataFrame:
    pivot = subject_results.pivot(
        index="subject", columns="method", values=value_column
    )
    oriented_pairs = [
        ("refinement", "all_sources"),
        ("mcs_style", "all_sources"),
        ("top_m_ref", "all_sources"),
        ("range_midpoint", "all_sources"),
        ("refinement", "mcs_style"),
        ("refinement", "top_m_ref"),
        ("refinement", "range_midpoint"),
        ("mcs_style", "top_m_ref"),
        ("mcs_style", "range_midpoint"),
        ("top_m_ref", "range_midpoint"),
    ]
    rows: list[dict[str, Any]] = []
    for method_a, method_b in oriented_pairs:
        paired = pivot[[method_a, method_b]].dropna()
        differences = paired[method_a] - paired[method_b]
        n_subjects = int(differences.size)
        mean_difference = float(differences.mean()) if n_subjects else float("nan")
        if n_subjects >= 2:
            sd_difference = float(differences.std(ddof=1))
            standard_error = sd_difference / math.sqrt(n_subjects)
            critical = float(student_t.ppf(0.975, n_subjects - 1))
            lower = mean_difference - critical * standard_error
            upper = mean_difference + critical * standard_error
        else:
            sd_difference = standard_error = lower = upper = float("nan")
        rows.append(
            {
                "metric": output_name,
                "method_a": method_a,
                "method_b": method_b,
                "contrast": f"{method_a} - {method_b}",
                "subjects": n_subjects,
                "mean_difference": mean_difference,
                "sd_difference": sd_difference,
                "standard_error": standard_error,
                "ci_level": 0.95,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_input_hashes(
    args: argparse.Namespace,
    cache_root: Path,
    output_directory: Path,
) -> tuple[pd.DataFrame, str]:
    inputs: list[tuple[str, Path]] = []
    for subject in args.subjects:
        for _, path in subject_cache_files(
            args.dataset, subject, cache_root, args.target_session
        ):
            inputs.append(("cached_covariance", path.resolve()))
    for path in (
        Path(__file__).resolve(),
        Path(__file__).with_name("ci_gate_ma2020_riemann.py").resolve(),
        Path(__file__).with_name("ci_gate_realdata_case_study.py").resolve(),
        Path(__file__).with_name("screening_core.py").resolve(),
        Path(__file__).with_name("path_config.py").resolve(),
        (ROOT / "requirements.txt").resolve(),
    ):
        inputs.append(("analysis_code_or_environment", path))
    provenance_manifest = (
        ROOT
        / "simulation_results"
        / ("ma2020_riemann_all_s1_s25_b499" if args.dataset == "ma2020" else "stieger_riemann_all_s1_s62_b499")
        / "manifest.json"
    )
    if provenance_manifest.exists():
        inputs.append(("source_cache_manifest", provenance_manifest.resolve()))

    rows: list[dict[str, Any]] = []
    for role, path in sorted(inputs, key=lambda item: str(item[1])):
        stat = path.stat()
        rows.append(
            {
                "role": role,
                "path": str(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
        )
    hashes = pd.DataFrame(rows)
    hashes.to_csv(output_directory / "input_hashes.csv", index=False)
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update(f"{row['role']}\0{row['path']}\0{row['sha256']}\n".encode("utf-8"))
    return hashes, aggregate.hexdigest()


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "mne"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    parser.add_argument("--subjects", type=parse_subjects, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--target-session", type=int, default=None)
    parser.add_argument("--splits", type=int, default=30)
    parser.add_argument("--boot", type=int, default=199)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-c", type=float, default=0.025)
    parser.add_argument("--alpha-d", type=float, default=0.025)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]
    if args.subjects is None:
        args.subjects = parse_subjects(str(defaults["subjects"]))
    if args.seed is None:
        args.seed = int(defaults["seed"])
    if args.target_session is None:
        args.target_session = defaults["target_session"]
    if args.cache_root is None:
        args.cache_root = DEFAULT_CACHE_ROOTS[args.dataset]
    args.cache_root = cache_base(args.cache_root.expanduser().resolve())
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUTPUT_ROOT / args.dataset
    args.out_dir = args.out_dir.expanduser().resolve()

    if args.workers < 1 or args.workers > 2:
        parser.error("--workers must be 1 or 2 for this shared host")
    if args.splits != 30:
        parser.error("this BSPC comparison requires exactly 30 target splits")
    if args.boot != 199:
        parser.error("this BSPC comparison requires B=199")
    if args.dataset == "ma2020" and args.target_session != 15:
        parser.error("this BSPC comparison requires Ma2020 target session 15")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error(f"refusing to overwrite nonempty output directory: {args.out_dir}")
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat()
    command = shlex.join([sys.executable, *sys.argv])
    shell_command = (
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 " + command
    )
    (args.out_dir / "actual_command.txt").write_text(shell_command + "\n", encoding="utf-8")

    settings: dict[str, Any] = {
        "dataset": args.dataset,
        "cache_root": str(args.cache_root),
        "target_session": args.target_session,
        "splits": args.splits,
        "boot": args.boot,
        "seed": args.seed,
        "shrinkage": args.shrinkage,
        "alpha": args.alpha,
        "alpha_c": args.alpha_c,
        "alpha_d": args.alpha_d,
    }
    manifest: dict[str, Any] = {
        "status": "running",
        "analysis": "BSPC fair downstream MCS-style comparison",
        "mcs_scope": (
            "Shared-target max-t stepdown source-screening analogue; not a complete "
            "implementation of classical Hansen--Lunde--Nason MCS."
        ),
        "started_at": started_at,
        "actual_command": shell_command,
        "script": str(Path(__file__).resolve()),
        "dataset": args.dataset,
        "subjects_requested": args.subjects,
        "settings": settings,
        "workers": args.workers,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "statistical_unit": "subject",
        "paired_ci": "two-sided 95% Student-t interval over subject-level mean differences",
        "selection_target_labels": False,
        "downstream_model": "subset-specific tangent-space LSQR LDA with automatic shrinkage",
    }
    write_json(args.out_dir / "manifest.json", manifest)

    try:
        _, aggregate_input_hash = record_input_hashes(
            args, args.cache_root, args.out_dir
        )
        manifest["aggregate_input_sha256"] = aggregate_input_hash
        write_json(args.out_dir / "manifest.json", manifest)
    except Exception as error:
        manifest.update(
            {
                "status": "failed_preflight",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "finished_at": datetime.now().astimezone().isoformat(),
            }
        )
        write_json(args.out_dir / "manifest.json", manifest)
        raise

    run_start = time.perf_counter()
    if args.workers == 1:
        outcomes = [safe_run_subject(subject, settings) for subject in args.subjects]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            outcomes = list(
                executor.map(
                    safe_run_subject,
                    args.subjects,
                    [settings] * len(args.subjects),
                )
            )

    successful = [outcome["result"] for outcome in outcomes if outcome["ok"]]
    errors = [outcome for outcome in outcomes if not outcome["ok"]]
    write_json(args.out_dir / "errors.json", errors)
    runtime_rows = [
        {
            "dataset": args.dataset,
            "subject": result["subject"],
            "runtime_seconds": result["runtime_seconds"],
            "n_source_sessions": result["n_source_sessions"],
            "n_target_trials": result["n_target_trials"],
            "n_channels": result["n_channels"],
        }
        for result in successful
    ]
    pd.DataFrame(runtime_rows).to_csv(
        args.out_dir / "runtime_by_subject.csv", index=False
    )

    split_rows = [row for result in successful for row in result["split_rows"]]
    assignment_rows = [
        row for result in successful for row in result["assignment_rows"]
    ]
    split_results = pd.DataFrame(split_rows)
    split_assignments = pd.DataFrame(assignment_rows)
    split_results.to_csv(args.out_dir / "split_results.csv", index=False)
    split_assignments.to_csv(args.out_dir / "split_assignments.csv", index=False)

    anomalies: list[dict[str, Any]] = []
    expected_rows_per_subject = args.splits * len(METHOD_ORDER)
    for result in successful:
        if len(result["split_rows"]) != expected_rows_per_subject:
            anomalies.append(
                {
                    "subject": result["subject"],
                    "type": "unexpected_split_row_count",
                    "expected": expected_rows_per_subject,
                    "observed": len(result["split_rows"]),
                }
            )
    if not split_results.empty:
        nonfinite = split_results.loc[
            ~np.isfinite(split_results["balanced_accuracy"]),
            ["subject", "split", "method", "balanced_accuracy"],
        ]
        anomalies.extend(
            {"type": "nonfinite_accuracy", **row}
            for row in nonfinite.to_dict(orient="records")
        )
    write_json(args.out_dir / "anomalies.json", anomalies)

    if not split_results.empty:
        subjects = subject_summary(split_results)
        methods = method_summary(subjects)
        subjects.to_csv(args.out_dir / "subject_summary.csv", index=False)
        methods.to_csv(args.out_dir / "method_summary.csv", index=False)
        for value_column, output_name, filename in (
            ("mean_balanced_accuracy", "balanced_accuracy", "paired_accuracy_ci.csv"),
            ("mean_set_size", "retained_sessions", "paired_set_size_ci.csv"),
            ("mean_training_trials", "training_trials", "paired_training_trials_ci.csv"),
        ):
            paired_ci_table(subjects, value_column, output_name).to_csv(
                args.out_dir / filename, index=False
            )

    finished_at = datetime.now().astimezone().isoformat()
    manifest.update(
        {
            "status": "complete" if not errors and not anomalies else "incomplete",
            "subjects_completed": [result["subject"] for result in successful],
            "subjects_failed": [error["subject"] for error in errors],
            "error_count": len(errors),
            "anomaly_count": len(anomalies),
            "runtime_seconds": time.perf_counter() - run_start,
            "finished_at": finished_at,
            "outputs": sorted(path.name for path in args.out_dir.iterdir()),
        }
    )
    write_json(args.out_dir / "manifest.json", manifest)

    if not split_results.empty:
        print(method_summary(subject_summary(split_results)).to_string(index=False))
    print(
        f"[{args.dataset}] status={manifest['status']} "
        f"subjects={len(successful)}/{len(args.subjects)} "
        f"runtime={manifest['runtime_seconds']:.1f}s -> {args.out_dir}",
        flush=True,
    )
    if errors or anomalies:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
