#!/usr/bin/env python3
"""Limited serial timing benchmark starting from cached trial covariances.

Frozen benchmark: subjects 1--3 in Ma2020 and Stieger2021; the first original
stratified half-target split; All / refinement / MCS-style screening; B=199;
three timing repetitions.  No fitted model, tangent reference, tangent feature,
session mean, or bootstrap draw is reused between methods or repetitions.

This is NOT a raw-EEG pipeline benchmark: cache reads, raw preprocessing,
accuracy scoring, provenance hashing, and output writing are outside timings.
The full screening charge includes every candidate source, even when excluded.
The same per-subject bootstrap seed is reset for refinement and MCS so that
both receive identical resamples, but both recompute and pay for those draws.

Use --mode pilot for the single Ma2020 subject-1 timing trial.  The full run
must be coordinated with other experiments to prevent CPU contention.
"""

from __future__ import annotations

import os

# These are set before importing numerical packages.  The context manager in
# main also enforces the limit for libraries that were initialized elsewhere.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_info, threadpool_limits

from ci_gate_extended_baselines_sim import mcs_stepdown
from ci_gate_ma2020_riemann import (
    riemann_estimates_and_bootstrap,
    riemann_mean,
    screen_pair_ref,
    shrink,
    tangent_features,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "simulation_results" / "bspc_revision_20261002" / "runtime"
DEFAULT_CACHE = {
    "ma2020": ROOT / "simulation_results" / "ma2020_riemann_all_s1_s25_b499" / "cov_cache",
    "stieger2021": ROOT / "simulation_results" / "stieger_riemann_all_s1_s62_b499" / "cov_cache",
}
BASE_SEEDS = {"ma2020": 20260618, "stieger2021": 20260707}
METHODS = ("all_sources", "refinement", "mcs_stepdown")
N_BOOT = 199
ALPHA = 0.05
ALPHA_C = ALPHA_D = 0.025
SCREEN_SHRINKAGE = 0.1
CLASSIFIER_SHRINKAGE = 0.05
TIME_FIELDS = (
    "screen_seconds", "fit_seconds", "predict_seconds",
    "screen_fit_seconds", "screen_fit_predict_seconds",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() or result.stderr.strip()


def process_snapshot() -> list[str]:
    """Record relevant contenders; the caller schedules exclusive experiment use."""
    output = command_output(["ps", "-axo", "pid,pcpu,comm,args"])
    this_pid = str(os.getpid())
    return [line for line in output.splitlines()
            if ("ci_gate_" in line or "python" in line.lower())
            and line.split(maxsplit=1)[0] != this_pid]


def environment_manifest() -> dict:
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    sources = [Path(__file__), Path(__file__).with_name("ci_gate_ma2020_riemann.py"),
               Path(__file__).with_name("ci_gate_extended_baselines_sim.py"),
               Path(__file__).with_name("ci_gate_realdata_case_study.py"),
               Path(__file__).with_name("screening_core.py"),
               Path(__file__).with_name("path_config.py")]
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "python_executable": sys.executable,
        "cpu": command_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]),
        "logical_cpus": os.cpu_count(),
        "physical_memory_bytes": command_output(["/usr/sbin/sysctl", "-n", "hw.memsize"]),
        "versions": {module.__name__: module.__version__ for module in
                     (np, scipy, pd, sklearn, mne, threadpoolctl)},
        "thread_pools": threadpool_info(),
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "numpy_configuration": config.getvalue(),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "command": sys.argv,
    }


def cache_paths(dataset: str, subject: int, cache_root: Path) -> list[tuple[int, Path]]:
    folder = cache_root / (f"sub-{subject:03d}" if dataset == "ma2020" else f"S{subject}")
    if dataset == "ma2020":
        sessions = list(range(1, 16))
    else:
        sessions = sorted(int(path.name.split("_")[0].split("-")[1])
                          for path in folder.glob("ses-*_cov.npz"))
        if len(sessions) < 6 or any(session > 11 for session in sessions):
            raise ValueError(f"Unexpected original Stieger session inventory: {folder}: {sessions}")
    paths = [(session, folder / f"ses-{session:02d}_cov.npz") for session in sessions]
    for _, path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Existing covariance cache required; no extraction fallback: {path}")
    return paths


def load_case(dataset: str, subject: int, cache_root: Path) -> tuple[dict, dict]:
    """Load and validate caches outside timing; never write or recreate a cache."""
    paths = cache_paths(dataset, subject, cache_root)
    covariances, labels, inputs = [], [], []
    channels = None
    for session, path in paths:
        with np.load(path, allow_pickle=False) as archive:
            covs = archive["covs"].astype(float)
            labs = archive["labels"].astype(int)
        if covs.ndim != 3 or covs.shape[1] != covs.shape[2] or len(labs) != len(covs):
            raise ValueError(f"Invalid covariance or label shape: {path}")
        if len(covs) < 4 or not np.isfinite(covs).all() or np.unique(labs).size != 2:
            raise ValueError(f"Non-finite, too small, or nonbinary cache: {path}")
        if channels is not None and covs.shape[1] != channels:
            raise ValueError(f"Channel counts differ between sessions: {path}")
        channels = covs.shape[1]
        covs.setflags(write=False)
        labs.setflags(write=False)
        covariances.append(covs)
        labels.append(labs)
        inputs.append({"session": session, "path": str(path.resolve()),
                       "sha256": sha256(path), "trials": len(covs), "channels": channels,
                       "label_counts": {str(int(k)): int(v) for k, v in
                                        zip(*np.unique(labs, return_counts=True))}})
    seed = BASE_SEEDS[dataset] + subject
    splitter = StratifiedShuffleSplit(n_splits=30, test_size=0.5, random_state=seed)
    screen_idx, test_idx = next(splitter.split(covariances[-1], labels[-1]))
    if np.intersect1d(screen_idx, test_idx).size:
        raise AssertionError("Screening/test overlap")
    metadata = {
        "dataset": dataset, "subject": subject, "cache_root": str(cache_root.resolve()),
        "source_sessions": [session for session, _ in paths[:-1]],
        "target_session": paths[-1][0], "channels": channels,
        "source_trial_counts": [len(x) for x in covariances[:-1]],
        "target_trials": len(covariances[-1]), "screen_trials": len(screen_idx),
        "test_trials": len(test_idx), "split_seed": seed,
        "bootstrap_seed": BASE_SEEDS[dataset] + 12345 + subject,
        "screen_indices": screen_idx.tolist(), "test_indices": test_idx.tolist(),
        "input_files": inputs,
    }
    arrays = {
        "source_covs": covariances[:-1], "source_labels": labels[:-1],
        "screen_covs": covariances[-1][screen_idx],
        "test_covs": covariances[-1][test_idx], "test_labels": labels[-1][test_idx],
    }
    return arrays, metadata


def timed_method(method: str, arrays: dict, metadata: dict) -> dict:
    src, labels = arrays["source_covs"], arrays["source_labels"]
    k = len(src)
    if method == "all_sources":
        selected = np.arange(k)
        screen_seconds = 0.0
    else:
        start = time.perf_counter_ns()
        rng = np.random.default_rng(metadata["bootstrap_seed"])
        estimates, draws = riemann_estimates_and_bootstrap(
            src, arrays["screen_covs"], rng, N_BOOT, SCREEN_SHRINKAGE, True)
        if method == "refinement":
            selected = screen_pair_ref(estimates, draws, k, ALPHA_C, ALPHA_D, ALPHA)
        elif method == "mcs_stepdown":
            selected = mcs_stepdown(estimates, draws, ALPHA)
        else:
            raise ValueError(method)
        screen_seconds = (time.perf_counter_ns() - start) / 1e9
    if not len(selected):
        raise RuntimeError(f"Empty selection in {metadata['dataset']} subject {metadata['subject']}")

    # Pooling, repeated shrinkage, reference estimation, training features and
    # model fitting match the existing train_eval_ts implementation.  No cache.
    start = time.perf_counter_ns()
    train_covs = np.concatenate([src[index] for index in selected], axis=0)
    train_labels = np.concatenate([labels[index] for index in selected], axis=0)
    reference = riemann_mean(shrink(train_covs, CLASSIFIER_SHRINKAGE))
    train_features = tangent_features(shrink(train_covs, CLASSIFIER_SHRINKAGE), reference)
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(train_features, train_labels)
    fit_seconds = (time.perf_counter_ns() - start) / 1e9

    start = time.perf_counter_ns()
    test_features = tangent_features(shrink(arrays["test_covs"], CLASSIFIER_SHRINKAGE), reference)
    prediction = model.predict(test_features)
    predict_seconds = (time.perf_counter_ns() - start) / 1e9

    candidate_trials = sum(len(source) for source in src)
    return {
        "method": method, "candidate_sessions": k, "retained_sessions": len(selected),
        "candidate_training_trials": candidate_trials, "training_trials": len(train_covs),
        "training_trial_fraction": len(train_covs) / candidate_trials,
        "screen_trials": len(arrays["screen_covs"]), "test_trials": len(arrays["test_covs"]),
        "channels": metadata["channels"], "tangent_features": train_features.shape[1],
        "selected_source_indices": json.dumps(selected.tolist()),
        "selected_sessions": json.dumps([metadata["source_sessions"][i] for i in selected]),
        "training_label_counts": json.dumps({str(int(key)): int(value) for key, value in
                                             zip(*np.unique(train_labels, return_counts=True))}),
        "balanced_accuracy": float(balanced_accuracy_score(arrays["test_labels"], prediction)),
        "screen_seconds": screen_seconds, "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "screen_fit_seconds": screen_seconds + fit_seconds,
        "screen_fit_predict_seconds": screen_seconds + fit_seconds + predict_seconds,
    }


def warm_up() -> None:
    """Initialize linear algebra on unrelated small synthetic arrays; no model reuse."""
    rng = np.random.default_rng(20261002)
    data = rng.normal(size=(32, 8, 20))
    covs = np.einsum("nct,ndt->ncd", data, data) / 20 + np.eye(8) * 0.05
    reference = riemann_mean(shrink(covs, CLASSIFIER_SHRINKAGE))
    features = tangent_features(shrink(covs, CLASSIFIER_SHRINKAGE), reference)
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
        features, np.tile([0, 1], 16))
    model.predict(features)
    estimates, draws = riemann_estimates_and_bootstrap(
        [covs[:16], covs[16:]], covs[:16], rng, 9, SCREEN_SHRINKAGE, True)
    screen_pair_ref(estimates, draws, 2, ALPHA_C, ALPHA_D, ALPHA)
    mcs_stepdown(estimates, draws, ALPHA)


def self_test() -> dict:
    """Cheap synthetic checks of timing accounting, matching and no-fit caching."""
    from unittest.mock import patch
    from ci_gate_ma2020_riemann import train_eval_ts

    rng = np.random.default_rng(2026100201)
    sources, labels = [], []
    for j, n in enumerate((12, 16, 20)):
        signal = rng.normal(size=(n, 4, 10)) * (1 + 0.2 * j)
        covs = np.einsum("nct,ndt->ncd", signal, signal) / 10 + np.eye(4) * 0.1
        sources.append(covs)
        labels.append(np.tile([0, 1], n // 2))
    arrays = {"source_covs": sources, "source_labels": labels,
              "screen_covs": sources[0][:6], "test_covs": sources[0][6:],
              "test_labels": labels[0][6:]}
    metadata = {"dataset": "synthetic_self_test", "subject": 0, "bootstrap_seed": 17,
                "channels": 4, "source_sessions": [1, 2, 3]}
    captured = []
    original_bootstrap = riemann_estimates_and_bootstrap

    def capture_bootstrap(*args, **kwargs):
        result = original_bootstrap(*args, **kwargs)
        captured.append(tuple(array.copy() for array in result))
        return result

    rows = []
    with patch(__name__ + ".riemann_estimates_and_bootstrap", side_effect=capture_bootstrap), \
            patch(__name__ + ".riemann_mean", wraps=riemann_mean) as means:
        for _ in range(2):
            for method in METHODS:
                rows.append(timed_method(method, arrays, metadata))
        assert means.call_count == 6, "Each model must refit its own tangent reference"
    assert len(captured) == 4, "Refinement/MCS must independently recompute screening on each repetition"
    for estimates, draws in captured[1:]:
        np.testing.assert_array_equal(estimates, captured[0][0])
        np.testing.assert_array_equal(draws, captured[0][1])
    for first, repeat in zip(rows[:3], rows[3:]):
        for key in ("selected_sessions", "training_trials", "balanced_accuracy"):
            assert first[key] == repeat[key], f"Same-seed repeats changed {key}"
    for row in rows[:3]:
        selected = json.loads(row["selected_source_indices"])
        count = sum(len(sources[index]) for index in selected)
        assert row["training_trials"] == count
        assert row["candidate_training_trials"] == 48
        assert np.isclose(row["training_trial_fraction"], count / 48)
        assert np.isclose(row["screen_fit_seconds"], row["screen_seconds"] + row["fit_seconds"])
        assert np.isclose(row["screen_fit_predict_seconds"],
                          row["screen_fit_seconds"] + row["predict_seconds"])
        assert all(row[key] >= 0 for key in TIME_FIELDS)
        original_accuracy = train_eval_ts(
            np.concatenate([sources[index] for index in selected]),
            np.concatenate([labels[index] for index in selected]),
            arrays["test_covs"], arrays["test_labels"])
        assert row["balanced_accuracy"] == original_accuracy
    assert rows[0]["screen_seconds"] == 0
    return {"status": "passed", "checks": [
        "six independent model-reference fits for six method calls",
        "four independently recomputed but identical shared-target bootstrap arrays",
        "identical selections and accuracy across fixed-seed timing repetitions",
        "actual training trials equal the sum across selected sessions",
        "phase durations add exactly to declared totals",
        "predictions yield the same accuracy as the original train_eval_ts on synthetic arrays",
        "all-source method has zero screening charge",
    ]}


def write_summaries(rows: list[dict], out_dir: Path) -> None:
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "runtime_raw.csv", index=False)
    subject_rows = []
    for (dataset, subject, method), block in raw.groupby(["dataset", "subject", "method"]):
        stable = ["candidate_sessions", "retained_sessions", "candidate_training_trials",
                  "training_trials", "training_trial_fraction", "balanced_accuracy",
                  "selected_sessions", "screen_trials", "test_trials", "channels", "tangent_features"]
        for column in stable:
            if block[column].nunique(dropna=False) != 1:
                raise AssertionError(f"Timing repetitions changed {column}: {dataset}/{subject}/{method}")
        row = {"dataset": dataset, "subject": subject, "method": method,
               "timing_repetitions": len(block), **{column: block.iloc[0][column] for column in stable}}
        for column in TIME_FIELDS:
            for statistic in ("median", "mean", "min", "max"):
                row[f"{column}_{statistic}"] = float(getattr(block[column], statistic)())
        subject_rows.append(row)
    subject_frame = pd.DataFrame(subject_rows)
    subject_frame.to_csv(out_dir / "runtime_subject_summary.csv", index=False)
    summaries = []
    for (dataset, method), block in subject_frame.groupby(["dataset", "method"]):
        row = {"dataset": dataset, "method": method, "subjects": len(block),
               "timing_repetitions_per_subject": int(block["timing_repetitions"].min())}
        for column in ("candidate_sessions", "retained_sessions", "candidate_training_trials",
                       "training_trials", "training_trial_fraction", "balanced_accuracy"):
            row[f"mean_{column}"] = float(block[column].mean())
        for column in TIME_FIELDS:
            row[f"mean_subject_median_{column}"] = float(block[f"{column}_median"].mean())
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    all_rows = summary.loc[summary.method == "all_sources"].set_index("dataset")
    for column in ("fit_seconds", "screen_fit_seconds", "screen_fit_predict_seconds"):
        key = f"mean_subject_median_{column}"
        summary[f"{column}_ratio_vs_all"] = [
            row[key] / all_rows.loc[row["dataset"], key] for row in summaries]
    summary.to_csv(out_dir / "runtime_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "self-test", "pilot", "benchmark"), default="preflight")
    parser.add_argument("--ma-cache-root", type=Path, default=DEFAULT_CACHE["ma2020"])
    parser.add_argument("--stieger-cache-root", type=Path, default=DEFAULT_CACHE["stieger2021"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = {"ma2020": args.ma_cache_root, "stieger2021": args.stieger_cache_root}
    if args.mode == "self-test":
        with threadpool_limits(limits=1):
            print(json.dumps(self_test(), indent=2))
        return
    if args.mode == "preflight":
        for dataset, root in roots.items():
            for subject in (1, 2, 3):
                paths = cache_paths(dataset, subject, root)
                print(json.dumps({"dataset": dataset, "subject": subject,
                                  "sessions": [session for session, _ in paths],
                                  "cache_files": len(paths)}), flush=True)
        with threadpool_limits(limits=1):
            print(json.dumps(environment_manifest(), indent=2))
        return

    pilot = args.mode == "pilot"
    cases = [("ma2020", 1)] if pilot else [(dataset, subject) for dataset in roots for subject in (1, 2, 3)]
    repetitions = 1 if pilot else 3
    out_dir = args.out_dir / "pilot_ma2020_s001" if pilot else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / name).exists() for name in ("runtime_raw.csv", "runtime_raw.jsonl", "manifest.json")):
        raise FileExistsError(f"Use a new output directory; existing results will not be overwritten: {out_dir}")
    manifest = {
        "status": "running", "started_at": utc_now(), "mode": args.mode,
        "inference_scope": "Descriptive timing on prespecified subjects, not an accuracy comparison or population timing estimate.",
        "timing_boundary": "In-memory cached trial covariances through source selection, model fitting and test prediction; not raw-EEG acquisition/preprocessing or cache I/O.",
        "excluded_from_timing": ["module imports", "cache reading and validation", "input hashing",
                                 "target split creation", "warm-up", "accuracy scoring", "output writing"],
        "screening_accounting": "Refinement and MCS each recompute all source means, target mean, all AIRM distances, B=199 shared-target bootstrap draws and their own selector. No cross-method sharing of measured work. All pays no screening charge.",
        "fit_accounting": "Pool concatenation, per-trial shrinkage, five-step subset-specific Riemannian mean, tangent features, LSQR LDA with automatic shrinkage. No model/feature/reference cache.",
        "prediction_accounting": "Held-out covariance shrinkage, tangent transformation at the fitted reference, and model prediction; excludes accuracy scoring.",
        "repetition_policy": "Identical data split and reset bootstrap seed per subject across methods and timing repetitions; method order rotates to balance order effects.",
        "summary_policy": "Median of three timings within each subject/method; arithmetic mean of subject medians within dataset. Ratios are ratios of those dataset means.",
        "protocol": {"subjects_each_dataset": [1] if pilot else [1, 2, 3],
                     "repetitions": repetitions, "n_boot": N_BOOT, "alpha": ALPHA,
                     "alpha_c": ALPHA_C, "alpha_d": ALPHA_D,
                     "screen_shrinkage": SCREEN_SHRINKAGE,
                     "classifier_shrinkage": CLASSIFIER_SHRINKAGE,
                     "split": "first of original 30 StratifiedShuffleSplit draws; test_size=0.5",
                     "base_seeds": BASE_SEEDS,
                     "bootstrap_seed_formula": "dataset base seed + 12345 + subject",
                     "riemann_mean_iterations": 5, "numerical_threads": 1},
        "cases": [], "execution_log": [],
    }
    start_all = time.perf_counter()
    rows: list[dict] = []
    try:
        with threadpool_limits(limits=1):
            manifest["environment"] = environment_manifest()
            manifest["processes_before"] = process_snapshot()
            warm_up()
            for case_index, (dataset, subject) in enumerate(cases):
                arrays, metadata = load_case(dataset, subject, roots[dataset])
                manifest["cases"].append(metadata)
                for repetition in range(repetitions):
                    offset = (case_index + repetition) % len(METHODS)
                    order = METHODS[offset:] + METHODS[:offset]
                    manifest["execution_log"].append({
                        "dataset": dataset, "subject": subject, "repetition": repetition + 1,
                        "time": utc_now(), "load_average": list(os.getloadavg()),
                        "method_order": order, "other_python_processes": process_snapshot()})
                    for order_index, method in enumerate(order):
                        result = timed_method(method, arrays, metadata)
                        row = {"dataset": dataset, "subject": subject,
                               "target_session": metadata["target_session"],
                               "repetition": repetition + 1, "method_order_index": order_index,
                               "split_seed": metadata["split_seed"],
                               "bootstrap_seed": metadata["bootstrap_seed"], **result}
                        rows.append(row)
                        with (out_dir / "runtime_raw.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
                        print(json.dumps(row), flush=True)
                    write_summaries(rows, out_dir)
                del arrays
            manifest["status"] = "complete"
            manifest["processes_after"] = process_snapshot()
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["wall_seconds_including_excluded_work"] = time.perf_counter() - start_all
        manifest["completed_rows"] = len(rows)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    expected = len(cases) * repetitions * len(METHODS)
    if len(rows) != expected:
        raise AssertionError(f"Expected {expected} rows, found {len(rows)}")
    print(f"Completed {args.mode}: {len(rows)} rows -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
