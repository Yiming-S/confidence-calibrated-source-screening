#!/usr/bin/env python3
"""Five-session covariance-space CI-gate analysis for BNCI2014_004.

The five MOABB session/block indices are reconstructed from the three runs in
``BxxT.mat`` followed by the two runs in ``BxxE.mat``.  Only the three EEG
channels (C3, Cz, C4) are used; the remaining three channels are EOG.  Session
5 is the primary target and sessions 1--4 are candidate sources.  Targets
3--5 additionally form a rolling-origin robustness analysis.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedShuffleSplit

from ci_gate_extended_baselines_sim import mcs_stepdown
from ci_gate_ma2020_riemann import (
    riemann_estimates_and_bootstrap,
    screen_pair_ref,
    train_eval_ts,
)
from ci_gate_realdata_case_study import build_system, pair_gate, rect_gate
from ci_gate_target_only_reference import train_eval_target_only
from path_config import resolve_cache_root, resolve_data_root


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "simulation_results" / "bnci004_riemann"


def mat_path(data_root: Path, subject: int, split: str) -> Path:
    return data_root / f"B{subject:02d}{split}.mat"


def extract_file_sessions(
    path: Path,
    tmin: float,
    tmax: float,
    l_freq: float,
    h_freq: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    runs = np.atleast_1d(mat["data"])
    sessions: list[tuple[np.ndarray, np.ndarray]] = []
    for run in runs:
        x = np.asarray(run.X, dtype=float)[:, :3]
        fs = float(run.fs)
        trials = np.asarray(run.trial, dtype=int)
        labels = np.asarray(run.y, dtype=int)
        artifacts = np.asarray(run.artifacts, dtype=int)
        sos = butter(4, [l_freq, h_freq], btype="bandpass", fs=fs, output="sos")
        filtered = sosfiltfilt(sos, x, axis=0)
        start_offset = int(round(tmin * fs))
        stop_offset = int(round(tmax * fs))
        covs: list[np.ndarray] = []
        kept_labels: list[int] = []
        for onset, label, artifact in zip(trials, labels, artifacts):
            if int(artifact) != 0:
                continue
            start = int(onset) - 1 + start_offset
            stop = int(onset) - 1 + stop_offset
            if start < 0 or stop > filtered.shape[0] or stop <= start:
                continue
            epoch = filtered[start:stop].T
            epoch -= epoch.mean(axis=1, keepdims=True)
            covs.append(epoch @ epoch.T / (epoch.shape[1] - 1))
            kept_labels.append(int(label) - 1)
        if not covs:
            raise RuntimeError(f"no artifact-free trials extracted from {path}")
        sessions.append((np.asarray(covs), np.asarray(kept_labels, dtype=int)))
    return sessions


def load_subject(subject: int, data_root: Path, cache_root: Path, args):
    cache_dir = cache_root / f"B{subject:02d}"
    cached = sorted(cache_dir.glob("session-*_cov.npz"))
    if len(cached) == 5:
        result = {}
        cache_valid = True
        for path in cached:
            session = int(path.stem.split("-")[1].split("_")[0])
            z = np.load(path)
            required = {"l_freq", "h_freq", "tmin", "tmax"}
            if not required.issubset(z.files) or not all(
                np.isclose(float(z[key]), float(getattr(args, key))) for key in required
            ):
                cache_valid = False
                break
            result[session] = (z["covs"].astype(float), z["labels"].astype(int))
        if cache_valid:
            return result

    sessions = extract_file_sessions(
        mat_path(data_root, subject, "T"), args.tmin, args.tmax, args.l_freq, args.h_freq
    )
    sessions += extract_file_sessions(
        mat_path(data_root, subject, "E"), args.tmin, args.tmax, args.l_freq, args.h_freq
    )
    if len(sessions) != 5:
        raise RuntimeError(f"B{subject:02d}: expected five sessions, found {len(sessions)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for session, (covs, labels) in enumerate(sessions, start=1):
        np.savez_compressed(
            cache_dir / f"session-{session:02d}_cov.npz",
            covs=covs.astype(np.float32),
            labels=labels,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            tmin=args.tmin,
            tmax=args.tmax,
        )
        result[session] = (covs, labels)
    return result


def selected_sets(delta_hat, delta_boot, alpha, alpha_c, alpha_d):
    """Build screening sets; ``pair_ref`` is the legacy key for the refinement intersection."""
    k = delta_hat.size
    system = build_system(delta_hat, delta_boot, alpha_c, alpha_d, alpha)
    rect = rect_gate(system["lower_c"], system["upper_c"])
    pair = pair_gate(k, system["jj"], system["lower_d"])
    pair_ref = np.asarray(sorted(set(rect.tolist()) & set(pair.tolist())), dtype=int)
    return {
        "rect": rect,
        "pair_ref": pair_ref,
        "mcs_stepdown": mcs_stepdown(delta_hat, delta_boot, alpha),
        "top1": np.argsort(delta_hat, kind="mergesort")[:1],
        "top3": np.argsort(delta_hat, kind="mergesort")[: min(3, k)],
    }


def pool(indices, source_covs, source_labels):
    return (
        np.concatenate([source_covs[int(i)] for i in indices], axis=0),
        np.concatenate([source_labels[int(i)] for i in indices], axis=0),
    )


def process_subject(subject: int, data_root_string: str, cache_root_string: str, settings: dict):
    class Args:
        pass

    args = Args()
    for key, value in settings.items():
        setattr(args, key, value)
    data_root, cache_root = Path(data_root_string), Path(cache_root_string)
    loaded = load_subject(subject, data_root, cache_root, args)
    screen_rows, down_rows = [], []

    for target_session in (3, 4, 5):
        source_sessions = list(range(1, target_session))
        source_covs = [loaded[s][0] for s in source_sessions]
        source_labels = [loaded[s][1] for s in source_sessions]
        target_covs, target_labels = loaded[target_session]
        k = len(source_sessions)
        rng = np.random.default_rng(args.seed + subject * 10_000 + target_session * 100)
        delta_hat, delta_boot = riemann_estimates_and_bootstrap(
            source_covs, target_covs, rng, args.screening_boot, args.shrinkage, True
        )
        selections = selected_sets(delta_hat, delta_boot, args.alpha, args.alpha_c, args.alpha_d)
        row = {
            "subject": subject,
            "target_session": target_session,
            "analysis": "primary" if target_session == 5 else "rolling",
            "n_sources": k,
            "target_trials": int(target_covs.shape[0]),
        }
        for name, indices in selections.items():
            row[f"{name}_size"] = int(indices.size)
            row[f"{name}_sessions"] = ",".join(str(source_sessions[int(i)]) for i in indices)
        if target_session == 5:
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
            row["wrong_target_pair_ref_size"] = int(wrong.size)
        else:
            row["wrong_target_pair_ref_size"] = np.nan
        screen_rows.append(row)

        n_splits = args.primary_splits if target_session == 5 else args.rolling_splits
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=0.5,
            random_state=args.seed + subject * 100 + target_session,
        )
        for split, (screen_idx, test_idx) in enumerate(splitter.split(target_covs, target_labels)):
            split_rng = np.random.default_rng(
                args.seed + subject * 100_000 + target_session * 1_000 + split
            )
            split_hat, split_boot = riemann_estimates_and_bootstrap(
                source_covs,
                target_covs[screen_idx],
                split_rng,
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
                    train_covs, train_labels, target_covs[test_idx], target_labels[test_idx]
                )
                down_rows.append(
                    {
                        "subject": subject,
                        "target_session": target_session,
                        "analysis": "primary" if target_session == 5 else "rolling",
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
            down_rows.append(
                {
                    "subject": subject,
                    "target_session": target_session,
                    "analysis": "primary" if target_session == 5 else "rolling",
                    "split": split,
                    "method": "target_only",
                    "set_size": 0,
                    "n_sources": k,
                    "balanced_accuracy": target_accuracy,
                }
            )
    return screen_rows, down_rows


def summarize_downstream(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_subject = raw.groupby(
        ["analysis", "subject", "method"], as_index=False
    ).agg(balanced_accuracy=("balanced_accuracy", "mean"), set_size=("set_size", "mean"))
    all_ref = by_subject[by_subject.method == "all_sources"][
        ["analysis", "subject", "balanced_accuracy"]
    ].rename(columns={"balanced_accuracy": "all_accuracy"})
    target_ref = by_subject[by_subject.method == "target_only"][
        ["analysis", "subject", "balanced_accuracy"]
    ].rename(columns={"balanced_accuracy": "target_accuracy"})
    by_subject = by_subject.merge(all_ref, on=["analysis", "subject"], how="left")
    by_subject = by_subject.merge(target_ref, on=["analysis", "subject"], how="left")
    by_subject["diff_vs_all"] = by_subject.balanced_accuracy - by_subject.all_accuracy
    by_subject["diff_vs_target"] = by_subject.balanced_accuracy - by_subject.target_accuracy
    summary = by_subject.groupby(["analysis", "method"], as_index=False).agg(
        subjects=("subject", "nunique"),
        mean_set_size=("set_size", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_diff_vs_all=("diff_vs_all", "mean"),
        mean_diff_vs_target=("diff_vs_target", "mean"),
        below_all_rate=("diff_vs_all", lambda x: float((x < 0).mean())),
        below_target_rate=("diff_vs_target", lambda x: float((x < 0).mean())),
    )
    return by_subject, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="BNCI2014_004 directory. Precedence: this option, then EEG_DATA_ROOT.")
    parser.add_argument("--cache-root", type=Path, default=None,
                        help="Covariance cache directory. Defaults under the output directory.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subjects", default="1-9")
    # One trial-onset-relative analysis window is applied across all five
    # sessions for comparability.  Sessions 1--2 use the no-feedback screening
    # protocol, whereas sessions 3--5 use the feedback protocol; this common
    # numerical window is therefore not presented as one universal official
    # imagery interval.
    parser.add_argument("--tmin", type=float, default=3.0)
    parser.add_argument("--tmax", type=float, default=7.5)
    parser.add_argument("--l-freq", type=float, default=8.0)
    parser.add_argument("--h-freq", type=float, default=30.0)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--screening-boot", type=int, default=499)
    parser.add_argument("--downstream-boot", type=int, default=199)
    parser.add_argument("--primary-splits", type=int, default=30)
    parser.add_argument("--rolling-splits", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-c", type=float, default=0.025)
    parser.add_argument("--alpha-d", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    args.data_root = resolve_data_root(
        args.data_root,
        parser,
        env_suffix="BNCI2014_004/MNE-bnci-data/database/data-sets/004-2014",
    )
    args.cache_root = resolve_cache_root(
        args.cache_root,
        parser,
        env_suffix="CI-gate-cache/bnci2014_004",
        fallback=args.out_dir / "cov_cache",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = (args.subjects.split("-") + [args.subjects])[:2]
    subjects = list(range(int(lo), int(hi) + 1))
    settings = {
        key: value for key, value in vars(args).items()
        if key not in {"data_root", "cache_root", "out_dir", "subjects", "jobs"}
    }
    t0 = time.time()
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(process_subject)(subject, str(args.data_root), str(args.cache_root), settings)
            for subject in subjects
        )
    screening = pd.DataFrame([row for result in results for row in result[0]])
    downstream = pd.DataFrame([row for result in results for row in result[1]])
    screening.to_csv(args.out_dir / "bnci004_screening_raw.csv", index=False)
    downstream.to_csv(args.out_dir / "bnci004_downstream_raw.csv", index=False)
    screen_summary = screening.groupby("analysis", as_index=False).agg(
        subjects=("subject", "nunique"),
        target_cases=("target_session", "size"),
        mean_n_sources=("n_sources", "mean"),
        rect_size=("rect_size", "mean"),
        pair_ref_size=("pair_ref_size", "mean"),
        mcs_size=("mcs_stepdown_size", "mean"),
        wrong_target_pair_ref_size=("wrong_target_pair_ref_size", "mean"),
    )
    screen_summary.to_csv(args.out_dir / "bnci004_screening_summary.csv", index=False)
    down_subject, down_summary = summarize_downstream(downstream)
    down_subject.to_csv(args.out_dir / "bnci004_downstream_by_subject.csv", index=False)
    down_summary.to_csv(args.out_dir / "bnci004_downstream_summary.csv", index=False)
    manifest = {
        "script": Path(__file__).name,
        "data_root": str(args.data_root),
        "cache_root": str(args.cache_root),
        "out_dir": str(args.out_dir),
        "subjects": subjects,
        "channels": ["C3", "Cz", "C4"],
        "session_mapping": {"1": "01T", "2": "02T", "3": "03T", "4": "04E", "5": "05E"},
        "primary_target": 5,
        "rolling_targets": [3, 4, 5],
        "screening_boot": args.screening_boot,
        "downstream_boot": args.downstream_boot,
        "primary_splits": args.primary_splits,
        "rolling_splits": args.rolling_splits,
        "filter_hz": [args.l_freq, args.h_freq],
        "epoch_seconds": [args.tmin, args.tmax],
        "covariance_shrinkage": args.shrinkage,
        "alpha": args.alpha,
        "alpha_component": args.alpha_c,
        "alpha_contrast": args.alpha_d,
        "seed": args.seed,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(screen_summary.to_string(index=False))
    print(down_summary.to_string(index=False))
    print(f"done in {manifest['runtime_seconds']}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
