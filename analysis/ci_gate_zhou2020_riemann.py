#!/usr/bin/env python3
"""Seven-session Zhou2020 covariance-space CI-gate replication.

The analysis is fixed before inspecting the Zhou2020 results.  For each of
the 20 participants, the latest available session is the target and all
earlier sessions are candidate sources (six sources for subjects 1--19 and
five for subject 20, whose official archive contains six sessions).  Only
left- and right-hand motor-imagery trials are used.  Continuous
EEG is filtered to 8--30 Hz, every five-second trial is represented by its
spatial covariance, and source--target discrepancy is the squared
affine-invariant Riemannian distance between mean covariances.

Full-target screening uses the shared-target bootstrap.  Downstream evaluation
uses 30 pre-specified stratified half splits of the target session: the
unlabeled half is used for screening and the held-out half only for balanced-accuracy
evaluation with tangent-space LDA.  The protocol matches the other longitudinal
case studies in this repository.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedShuffleSplit

from ci_gate_bnci004_riemann import pool, selected_sets, summarize_downstream
from ci_gate_ma2020_riemann import riemann_estimates_and_bootstrap, train_eval_ts
from ci_gate_target_only_reference import train_eval_target_only
from path_config import resolve_cache_root, resolve_data_root


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "simulation_results" / "zhou2020_riemann"
GDF_LABELS = {769: 0, 770: 1}


def parse_subjects(spec: str) -> list[int]:
    subjects: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(value) for value in part.split("-", 1))
            subjects.extend(range(lo, hi + 1))
        else:
            subjects.append(int(part))
    subjects = sorted(set(subjects))
    invalid = [subject for subject in subjects if subject < 1 or subject > 20]
    if invalid:
        raise ValueError(f"Zhou2020 subjects must be in 1--20, got {invalid}")
    return subjects


def dataset_dir(data_root: Path) -> Path:
    return data_root / "MNE-zhou2020-data"


def download_subject(subject: int, data_root: Path) -> Path:
    """Download and extract one subject through the official MOABB adapter."""
    try:
        from moabb.datasets import Zhou2020
    except ImportError as exc:
        raise RuntimeError(
            "The installed MOABB release does not expose the Zhou2020 adapter. "
            "Provide an existing MNE-zhou2020-data directory and use --skip-download."
        ) from exc
    data_root.mkdir(parents=True, exist_ok=True)
    dataset = Zhou2020(subjects=[subject])
    path = Path(dataset.data_path(subject, path=data_root, force_update=False))
    subject_dir = path / f"S{subject:02d}"
    if not any(subject_dir.rglob("*.npz")):
        raise FileNotFoundError(f"no Zhou2020 NPZ files found under {subject_dir}")
    return subject_dir


def session_dirs(subject_dir: Path) -> list[Path]:
    directories = [
        path for path in subject_dir.iterdir()
        if path.is_dir() and any(path.glob("*.npz"))
    ]
    directories.sort(key=lambda path: path.name)
    if len(directories) not in {6, 7}:
        raise RuntimeError(
            f"{subject_dir.name}: expected six or seven session directories, found "
            f"{len(directories)} ({[path.name for path in directories]})"
        )
    return directories


def extract_session_covs(
    path: Path,
    n_eeg: int,
    tmin: float,
    tmax: float,
    l_freq: float,
    h_freq: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Extract left/right trial covariances from all runs in one session."""
    covs: list[np.ndarray] = []
    labels: list[int] = []
    sampling_rates: set[float] = set()
    run_files = sorted(path.glob("*.npz"))
    if not run_files:
        raise FileNotFoundError(f"no run NPZ files found in {path}")

    for run_path in run_files:
        with np.load(run_path, allow_pickle=True) as npz:
            signal = np.asarray(npz["signal"][:, :n_eeg], dtype=np.float64)
            markers = np.asarray(npz["MarkOnSignal"])
            fs = float(np.ravel(npz["SampleRate"])[0])
        sampling_rates.add(fs)
        sos = butter(4, [l_freq, h_freq], btype="bandpass", fs=fs, output="sos")
        filtered = sosfiltfilt(sos, signal, axis=0)
        start_offset = int(round(tmin * fs))
        stop_offset = int(round(tmax * fs))
        for onset_value, code_value in np.atleast_2d(markers):
            code = int(code_value)
            if code not in GDF_LABELS:
                continue
            start = int(onset_value) + start_offset
            stop = int(onset_value) + stop_offset
            if start < 0 or stop > filtered.shape[0] or stop <= start:
                continue
            epoch = filtered[start:stop].T
            epoch -= epoch.mean(axis=1, keepdims=True)
            covs.append(epoch @ epoch.T / (epoch.shape[1] - 1))
            labels.append(GDF_LABELS[code])

    if len(sampling_rates) != 1:
        raise RuntimeError(f"{path}: inconsistent sampling rates {sampling_rates}")
    labels_array = np.asarray(labels, dtype=int)
    counts = np.bincount(labels_array, minlength=2)
    if counts.min() < 40:
        raise RuntimeError(
            f"{path}: unexpectedly few left/right trials, counts={counts.tolist()}"
        )
    return np.asarray(covs), labels_array, sampling_rates.pop(), len(run_files)


def load_subject(subject: int, data_root: Path, cache_root: Path, args):
    n_eeg = 41 if subject <= 12 else 26
    cache_dir = cache_root / f"S{subject:02d}"
    cached = sorted(cache_dir.glob("session-*_cov.npz"))
    if len(cached) in {6, 7}:
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        cache_valid = True
        for cache_path in cached:
            session = int(cache_path.stem.split("-")[1].split("_")[0])
            with np.load(cache_path) as npz:
                required = {"l_freq", "h_freq", "tmin", "tmax", "n_eeg"}
                if not required.issubset(npz.files):
                    cache_valid = False
                    break
                expected = {
                    "l_freq": args.l_freq,
                    "h_freq": args.h_freq,
                    "tmin": args.tmin,
                    "tmax": args.tmax,
                    "n_eeg": n_eeg,
                }
                if not all(np.isclose(float(npz[key]), float(value)) for key, value in expected.items()):
                    cache_valid = False
                    break
                result[session] = (
                    npz["covs"].astype(float), npz["labels"].astype(int)
                )
        if cache_valid and sorted(result) == list(range(1, len(cached) + 1)):
            return result

    subject_dir = dataset_dir(data_root) / f"S{subject:02d}"
    if not any(subject_dir.rglob("*.npz")):
        raise FileNotFoundError(
            f"{subject_dir} is missing; run without --skip-download first"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for session, directory in enumerate(session_dirs(subject_dir), start=1):
        covs, labels, fs, n_runs = extract_session_covs(
            directory, n_eeg, args.tmin, args.tmax, args.l_freq, args.h_freq
        )
        np.savez_compressed(
            cache_dir / f"session-{session:02d}_cov.npz",
            covs=covs.astype(np.float32),
            labels=labels,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            tmin=args.tmin,
            tmax=args.tmax,
            n_eeg=n_eeg,
            sampling_rate=fs,
            n_runs=n_runs,
            source_directory=directory.name,
        )
        result[session] = (covs, labels)
    return result


def process_subject(
    subject: int,
    data_root_string: str,
    cache_root_string: str,
    settings: dict,
) -> tuple[dict, list[dict]]:
    class Args:
        pass

    args = Args()
    for key, value in settings.items():
        setattr(args, key, value)
    loaded = load_subject(
        subject, Path(data_root_string), Path(cache_root_string), args
    )
    target_session = max(loaded)
    source_sessions = [session for session in sorted(loaded) if session < target_session]
    source_covs = [loaded[session][0] for session in source_sessions]
    source_labels = [loaded[session][1] for session in source_sessions]
    target_covs, target_labels = loaded[target_session]
    k = len(source_sessions)

    full_rng = np.random.default_rng(args.seed + subject * 10_000)
    delta_hat, delta_boot = riemann_estimates_and_bootstrap(
        source_covs,
        target_covs,
        full_rng,
        args.screening_boot,
        args.shrinkage,
        True,
    )
    selections = selected_sets(
        delta_hat, delta_boot, args.alpha, args.alpha_c, args.alpha_d
    )
    wrong_hat, wrong_boot = riemann_estimates_and_bootstrap(
        source_covs,
        target_covs,
        np.random.default_rng(args.seed + subject * 10_000 + 99),
        args.screening_boot,
        args.shrinkage,
        False,
    )
    wrong = selected_sets(
        wrong_hat, wrong_boot, args.alpha, args.alpha_c, args.alpha_d
    )["pair_ref"]
    screening = {
        "subject": subject,
        "target_session": target_session,
        "n_sources": k,
        "n_channels": int(target_covs.shape[1]),
        "target_trials": int(target_covs.shape[0]),
        "left_target_trials": int((target_labels == 0).sum()),
        "right_target_trials": int((target_labels == 1).sum()),
        "wrong_target_pair_ref_size": int(wrong.size),
    }
    for name, indices in selections.items():
        screening[f"{name}_size"] = int(indices.size)
        screening[f"{name}_sessions"] = ",".join(
            str(source_sessions[int(index)]) for index in indices
        )

    downstream: list[dict] = []
    splitter = StratifiedShuffleSplit(
        n_splits=args.splits,
        test_size=0.5,
        random_state=args.seed + subject,
    )
    for split, (screen_idx, test_idx) in enumerate(
        splitter.split(target_covs, target_labels)
    ):
        split_hat, split_boot = riemann_estimates_and_bootstrap(
            source_covs,
            target_covs[screen_idx],
            np.random.default_rng(args.seed + subject * 100_000 + split),
            args.downstream_boot,
            args.shrinkage,
            True,
        )
        split_sets = selected_sets(
            split_hat, split_boot, args.alpha, args.alpha_c, args.alpha_d
        )
        methods = {
            "all_sources": np.arange(k, dtype=int),
            "top1": split_sets["top1"],
            "top3": split_sets["top3"],
            "mcs_stepdown": split_sets["mcs_stepdown"],
            "pair_ref": split_sets["pair_ref"],
        }
        for method, indices in methods.items():
            train_covs, train_labels = pool(indices, source_covs, source_labels)
            accuracy = train_eval_ts(
                train_covs,
                train_labels,
                target_covs[test_idx],
                target_labels[test_idx],
            )
            downstream.append(
                {
                    "analysis": "primary",
                    "subject": subject,
                    "split": split,
                    "method": method,
                    "set_size": int(indices.size),
                    "n_sources": k,
                    "balanced_accuracy": accuracy,
                }
            )
        target_accuracy = train_eval_target_only(
            target_covs[screen_idx],
            target_labels[screen_idx],
            target_covs[test_idx],
            target_labels[test_idx],
        )
        downstream.append(
            {
                "analysis": "primary",
                "subject": subject,
                "split": split,
                "method": "target_only",
                "set_size": 0,
                "n_sources": k,
                "balanced_accuracy": target_accuracy,
            }
        )
    return screening, downstream


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="MOABB download root. Precedence: this option, then EEG_DATA_ROOT.")
    parser.add_argument("--cache-root", type=Path, default=None,
                        help="Covariance cache directory. Defaults under the resolved data root.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subjects", default="1-20")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=5.0)
    parser.add_argument("--l-freq", type=float, default=8.0)
    parser.add_argument("--h-freq", type=float, default=30.0)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--screening-boot", type=int, default=499)
    parser.add_argument("--downstream-boot", type=int, default=199)
    parser.add_argument("--splits", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-c", type=float, default=0.025)
    parser.add_argument("--alpha-d", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    args.data_root = resolve_data_root(args.data_root, parser, env_suffix=None)
    args.cache_root = resolve_cache_root(
        args.cache_root,
        parser,
        env_suffix="CI-gate-cache/zhou2020",
        fallback=args.data_root / "CI-gate-cache" / "zhou2020",
    )
    subjects = parse_subjects(args.subjects)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not args.skip_download:
        for subject in subjects:
            print(f"[zhou2020] preparing subject {subject} ...", flush=True)
            download_subject(subject, args.data_root)
    if args.download_only:
        print(
            f"[zhou2020] downloaded {len(subjects)} subjects in "
            f"{round(time.time() - t0, 1)}s"
        )
        return

    settings = {
        key: value
        for key, value in vars(args).items()
        if key
        not in {
            "data_root",
            "cache_root",
            "out_dir",
            "subjects",
            "jobs",
            "download_only",
            "skip_download",
        }
    }
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(process_subject)(
                subject, str(args.data_root), str(args.cache_root), settings
            )
            for subject in subjects
        )
    screening = pd.DataFrame([result[0] for result in results])
    downstream = pd.DataFrame(
        [row for result in results for row in result[1]]
    )
    screening.to_csv(args.out_dir / "zhou2020_screening_raw.csv", index=False)
    downstream.to_csv(args.out_dir / "zhou2020_downstream_raw.csv", index=False)
    screen_summary = pd.DataFrame(
        [
            {
                "subjects": screening.subject.nunique(),
                "mean_n_sources": screening.n_sources.mean(),
                "mean_n_channels": screening.n_channels.mean(),
                "mean_target_trials": screening.target_trials.mean(),
                "rect_size": screening.rect_size.mean(),
                "pair_ref_size": screening.pair_ref_size.mean(),
                "pair_ref_fraction": (
                    screening.pair_ref_size / screening.n_sources
                ).mean(),
                "mcs_size": screening.mcs_stepdown_size.mean(),
                "wrong_target_pair_ref_size": (
                    screening.wrong_target_pair_ref_size.mean()
                ),
                "pair_ref_smaller_than_rect_subjects": int(
                    (screening.pair_ref_size < screening.rect_size).sum()
                ),
            }
        ]
    )
    screen_summary.to_csv(
        args.out_dir / "zhou2020_screening_summary.csv", index=False
    )
    down_subject, down_summary = summarize_downstream(downstream)
    down_subject.to_csv(
        args.out_dir / "zhou2020_downstream_by_subject.csv", index=False
    )
    down_summary.to_csv(
        args.out_dir / "zhou2020_downstream_summary.csv", index=False
    )
    manifest = {
        "script": Path(__file__).name,
        "data_root": str(args.data_root),
        "dataset_directory": str(dataset_dir(args.data_root)),
        "cache_root": str(args.cache_root),
        "out_dir": str(args.out_dir),
        "subjects": subjects,
        "subject_channel_counts": {"1-12": 41, "13-20": 26},
        "classes": ["left_hand", "right_hand"],
        "primary_target": "latest available session",
        "source_sessions": "all earlier available sessions",
        "minimum_sessions": 6,
        "screening_boot": args.screening_boot,
        "downstream_boot": args.downstream_boot,
        "splits": args.splits,
        "filter_hz": [args.l_freq, args.h_freq],
        "epoch_seconds": [args.tmin, args.tmax],
        "covariance_shrinkage": args.shrinkage,
        "alpha": args.alpha,
        "alpha_component": args.alpha_c,
        "alpha_contrast": args.alpha_d,
        "seed": args.seed,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(screen_summary.to_string(index=False))
    print(down_summary.to_string(index=False))
    print(
        f"[zhou2020] done in {manifest['runtime_seconds']}s -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
