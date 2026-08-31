#!/usr/bin/env python3
"""Shared-target moving-block bootstrap sensitivity for the EEG case studies.

The primary EEG analyses resample trials independently.  This script repeats
the Stieger2021 and Zhou2020 screening with a circular moving-block bootstrap
that preserves short-range ordering within every session.  Within a bootstrap
replicate, one target block-resample is reused across every source; source
sessions receive independent block-resamples.  Block length one is the i.i.d.
bootstrap control.

The full-target analysis compares block lengths 1, 4, and 8.  A smaller,
pre-specified downstream sensitivity repeats ten held-out target splits with
block lengths 1 and 4.  Target screening indices are sorted back into recording
order before block resampling.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import StratifiedShuffleSplit

from ci_gate_extended_baselines_sim import mcs_stepdown
from ci_gate_ma2020_riemann import airm2, screen_pair_ref, shrink, train_eval_ts
from path_config import resolve_cache_root


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STIEGER_CACHE = (
    ROOT / "simulation_results" / "stieger_riemann_all_s1_s62_b499" / "cov_cache"
)
DEFAULT_OUT = ROOT / "simulation_results" / "block_bootstrap_sensitivity"


def circular_block_counts(
    n: int,
    n_boot: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Multinomial-style counts from a circular moving-block bootstrap."""
    if n < 1:
        raise ValueError("cannot bootstrap an empty session")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    n_blocks = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    indices = indices.reshape(n_boot, -1)[:, :n]
    counts = np.zeros((n_boot, n), dtype=np.int16)
    for b in range(n_boot):
        counts[b] = np.bincount(indices[b], minlength=n)
    return counts


def _batch_inverse_sqrt(spd: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (spd + np.swapaxes(spd, -1, -2)))
    values = np.clip(values, 1e-12, None)
    scaled = vectors * (values ** -0.5)[..., None, :]
    return scaled @ np.swapaxes(vectors, -1, -2)


def _batch_airm2_from_target(
    source: np.ndarray,
    target_inverse_sqrt: np.ndarray,
) -> np.ndarray:
    whitened = target_inverse_sqrt @ source @ target_inverse_sqrt
    whitened = 0.5 * (whitened + np.swapaxes(whitened, -1, -2))
    values = np.linalg.eigvalsh(whitened)
    log_values = np.log(np.clip(values, 1e-12, None))
    return np.sum(log_values * log_values, axis=1)


def riemann_estimates_and_block_bootstrap(
    source_covs: list[np.ndarray],
    target_covs: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    lam: float,
    block_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """AIRM discrepancies and a shared-target circular block bootstrap."""
    theta_target = shrink(target_covs.mean(axis=0), lam)
    estimates = np.asarray(
        [airm2(shrink(source.mean(axis=0), lam), theta_target) for source in source_covs]
    )

    target_counts = circular_block_counts(
        target_covs.shape[0], n_boot, block_length, rng
    ).astype(float)
    target_means = np.einsum("bn,ncd->bcd", target_counts, target_covs)
    target_means /= target_covs.shape[0]
    target_inverse_sqrt = _batch_inverse_sqrt(shrink(target_means, lam))

    boot = np.empty((n_boot, len(source_covs)))
    for source_index, source in enumerate(source_covs):
        source_counts = circular_block_counts(
            source.shape[0], n_boot, block_length, rng
        ).astype(float)
        source_means = np.einsum("bn,ncd->bcd", source_counts, source)
        source_means /= source.shape[0]
        boot[:, source_index] = _batch_airm2_from_target(
            shrink(source_means, lam), target_inverse_sqrt
        )
    return estimates, boot


def _load_npz_sessions(directory: Path, pattern: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    covariance_sessions: list[np.ndarray] = []
    label_sessions: list[np.ndarray] = []
    for path in sorted(directory.glob(pattern)):
        with np.load(path) as npz:
            covariance_sessions.append(npz["covs"].astype(float))
            label_sessions.append(npz["labels"].astype(int))
    if len(covariance_sessions) < 2:
        raise ValueError(f"fewer than two cached sessions under {directory}")
    return covariance_sessions, label_sessions


def load_subject(
    dataset: str,
    subject: int,
    stieger_cache: Path,
    zhou_cache: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    if dataset == "stieger":
        covariances, labels = _load_npz_sessions(
            stieger_cache / f"S{subject}", "ses-*_cov.npz"
        )
    elif dataset == "zhou2020":
        covariances, labels = _load_npz_sessions(
            zhou_cache / f"S{subject:02d}", "session-*_cov.npz"
        )
    else:
        raise ValueError(f"unknown dataset {dataset}")
    return covariances[:-1], labels[:-1], covariances[-1], labels[-1]


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_set, right_set = set(left.tolist()), set(right.tolist())
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _selection(
    sources: list[np.ndarray],
    target: np.ndarray,
    block_length: int,
    n_boot: int,
    lam: float,
    alpha: float,
    alpha_c: float,
    alpha_d: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    estimates, boot = riemann_estimates_and_block_bootstrap(
        sources,
        target,
        np.random.default_rng(seed),
        n_boot,
        lam,
        block_length,
    )
    pair_ref = screen_pair_ref(
        estimates, boot, len(sources), alpha_c, alpha_d, alpha
    )
    mcs = mcs_stepdown(estimates, boot, alpha)
    return pair_ref, mcs


def _pool(
    indices: np.ndarray,
    source_covariances: list[np.ndarray],
    source_labels: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([source_covariances[int(i)] for i in indices]),
        np.concatenate([source_labels[int(i)] for i in indices]),
    )


def process_subject(dataset: str, subject: int, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    sources, source_labels, target, target_labels = load_subject(
        dataset, subject, args.stieger_cache, args.zhou_cache
    )
    n_sources = len(sources)
    screening_rows: list[dict] = []
    selections: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for block_length in args.block_lengths:
        pair_ref, mcs = _selection(
            sources,
            target,
            block_length,
            args.screening_boot,
            args.shrinkage,
            args.alpha,
            args.alpha_c,
            args.alpha_d,
            args.seed + subject * 100_000 + block_length * 1_000,
        )
        selections[block_length] = (pair_ref, mcs)

    iid_pair = selections[1][0]
    for block_length in args.block_lengths:
        pair_ref, mcs = selections[block_length]
        screening_rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "block_length": block_length,
                "n_sources": n_sources,
                "target_trials": target.shape[0],
                "pair_ref_size": pair_ref.size,
                "pair_ref_fraction": pair_ref.size / n_sources,
                "mcs_size": mcs.size,
                "jaccard_vs_iid": jaccard(pair_ref, iid_pair),
                "exact_match_iid": int(np.array_equal(pair_ref, iid_pair)),
            }
        )

    downstream_rows: list[dict] = []
    all_indices = np.arange(n_sources, dtype=int)
    all_covariances, all_labels = _pool(all_indices, sources, source_labels)
    splitter = StratifiedShuffleSplit(
        n_splits=args.downstream_splits,
        test_size=0.5,
        random_state=args.seed + subject,
    )
    for split, (screen_indices, test_indices) in enumerate(
        splitter.split(target, target_labels)
    ):
        ordered_screen_indices = np.sort(screen_indices)
        screen_target = target[ordered_screen_indices]
        test_target = target[test_indices]
        test_labels = target_labels[test_indices]
        all_accuracy = train_eval_ts(
            all_covariances, all_labels, test_target, test_labels
        )
        downstream_rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "split": split,
                "block_length": 0,
                "method": "all_sources",
                "set_size": n_sources,
                "balanced_accuracy": all_accuracy,
            }
        )
        for block_length in args.downstream_block_lengths:
            pair_ref, mcs = _selection(
                sources,
                screen_target,
                block_length,
                args.downstream_boot,
                args.shrinkage,
                args.alpha,
                args.alpha_c,
                args.alpha_d,
                args.seed
                + subject * 1_000_000
                + split * 10_000
                + block_length * 100,
            )
            for method, indices in (("pair_ref", pair_ref), ("mcs_stepdown", mcs)):
                train_covariances, train_labels = _pool(
                    indices, sources, source_labels
                )
                accuracy = train_eval_ts(
                    train_covariances,
                    train_labels,
                    test_target,
                    test_labels,
                )
                downstream_rows.append(
                    {
                        "dataset": dataset,
                        "subject": subject,
                        "split": split,
                        "block_length": block_length,
                        "method": method,
                        "set_size": indices.size,
                        "balanced_accuracy": accuracy,
                    }
                )
    return screening_rows, downstream_rows


def summarize_screening(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["dataset", "block_length"], as_index=False)
        .agg(
            subjects=("subject", "nunique"),
            mean_sources=("n_sources", "mean"),
            pair_ref_size=("pair_ref_size", "mean"),
            pair_ref_fraction=("pair_ref_fraction", "mean"),
            mcs_size=("mcs_size", "mean"),
            jaccard_vs_iid=("jaccard_vs_iid", "mean"),
            exact_match_iid=("exact_match_iid", "mean"),
        )
        .sort_values(["dataset", "block_length"])
    )


def summarize_downstream(rows: pd.DataFrame) -> pd.DataFrame:
    per_subject = (
        rows.groupby(
            ["dataset", "subject", "block_length", "method"], as_index=False
        )
        .agg(
            set_size=("set_size", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
        )
    )
    all_accuracy = (
        per_subject[per_subject.method == "all_sources"]
        .set_index(["dataset", "subject"])["balanced_accuracy"]
    )
    output: list[dict] = []
    for (dataset, block_length, method), part in per_subject[
        per_subject.method != "all_sources"
    ].groupby(["dataset", "block_length", "method"]):
        part = part.set_index(["dataset", "subject"])
        difference = part.balanced_accuracy - all_accuracy.loc[part.index]
        output.append(
            {
                "dataset": dataset,
                "block_length": int(block_length),
                "method": method,
                "subjects": part.shape[0],
                "mean_set_size": part.set_size.mean(),
                "mean_balanced_accuracy": part.balanced_accuracy.mean(),
                "mean_diff_vs_all": difference.mean(),
            }
        )
    return pd.DataFrame(output).sort_values(["dataset", "block_length", "method"])


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="stieger,zhou2020")
    parser.add_argument("--stieger-cache", type=Path, default=None,
                        help="Stieger covariance cache; defaults to the repository result cache.")
    parser.add_argument("--zhou-cache", type=Path, default=None,
                        help="Zhou covariance cache; otherwise resolved below EEG_DATA_ROOT.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--block-lengths", type=parse_int_list, default=[1, 4, 8])
    parser.add_argument(
        "--downstream-block-lengths", type=parse_int_list, default=[1, 4]
    )
    parser.add_argument("--screening-boot", type=int, default=499)
    parser.add_argument("--downstream-boot", type=int, default=99)
    parser.add_argument("--downstream-splits", type=int, default=10)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-c", type=float, default=0.025)
    parser.add_argument("--alpha-d", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    args.datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(args.datasets) - {"stieger", "zhou2020"})
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")
    if "stieger" in args.datasets:
        args.stieger_cache = resolve_cache_root(
            args.stieger_cache,
            parser,
            env_suffix="CI-gate-cache/stieger2021",
            fallback=DEFAULT_STIEGER_CACHE,
            option_name="--stieger-cache",
        )
    elif args.stieger_cache is not None:
        args.stieger_cache = args.stieger_cache.expanduser()
    if "zhou2020" in args.datasets:
        args.zhou_cache = resolve_cache_root(
            args.zhou_cache,
            parser,
            env_suffix="CI-gate-cache/zhou2020",
            option_name="--zhou-cache",
        )
    elif args.zhou_cache is not None:
        args.zhou_cache = args.zhou_cache.expanduser()
    if 1 not in args.block_lengths:
        raise ValueError("block-lengths must contain the i.i.d. control 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subject_map = {"stieger": range(1, 63), "zhou2020": range(1, 21)}
    tasks = [
        (dataset, subject)
        for dataset in args.datasets
        for subject in subject_map[dataset]
    ]
    started = time.time()
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(process_subject)(dataset, subject, args)
            for dataset, subject in tasks
        )
    screening_rows = pd.DataFrame(
        [row for screening, _ in results for row in screening]
    )
    downstream_rows = pd.DataFrame(
        [row for _, downstream in results for row in downstream]
    )
    screening_summary = summarize_screening(screening_rows)
    downstream_summary = summarize_downstream(downstream_rows)

    screening_rows.to_csv(args.out_dir / "block_screening_raw.csv", index=False)
    screening_summary.to_csv(
        args.out_dir / "block_screening_summary.csv", index=False
    )
    downstream_rows.to_csv(args.out_dir / "block_downstream_raw.csv", index=False)
    downstream_summary.to_csv(
        args.out_dir / "block_downstream_summary.csv", index=False
    )
    manifest = {
        "script": Path(__file__).name,
        "datasets": args.datasets,
        "stieger_cache": str(args.stieger_cache) if args.stieger_cache else None,
        "zhou_cache": str(args.zhou_cache) if args.zhou_cache else None,
        "out_dir": str(args.out_dir),
        "subjects": {dataset: list(subject_map[dataset]) for dataset in args.datasets},
        "block_lengths": args.block_lengths,
        "downstream_block_lengths": args.downstream_block_lengths,
        "screening_boot": args.screening_boot,
        "downstream_boot": args.downstream_boot,
        "downstream_splits": args.downstream_splits,
        "shared_target_blocks": True,
        "independent_source_block_resamples": True,
        "circular_blocks": True,
        "block_length_one_is_iid_control": True,
        "full_target_trial_order": "native cache order",
        "downstream_screen_indices_sorted_to_recording_order": True,
        "seed": args.seed,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(screening_summary.to_string(index=False))
    print(downstream_summary.to_string(index=False))
    print(f"completed in {time.time() - started:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
