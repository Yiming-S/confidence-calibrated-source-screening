#!/usr/bin/env python3
"""Rolling-origin robustness analysis for the Ma2020 and Stieger2021 cases.

For every subject and every session with at least ``min_sources`` earlier
sessions, the earlier sessions are candidate sources and the current session
is the target.  The full target is used for the descriptive retained-set
analysis.  A pre-specified stratified half split is used for downstream
evaluation: the screening half is label-free for source selection and the
held-out half is used only for balanced-accuracy evaluation.

The repeated target sessions are not treated as independent subjects.  The
script writes subject-level aggregates that are the units used for paired
summaries and confidence intervals in the manuscript.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import StratifiedShuffleSplit

from ci_gate_extended_baselines_sim import mcs_stepdown
from ci_gate_ma2020_riemann import (
    riemann_estimates_and_bootstrap,
    screen_pair_ref,
    train_eval_ts,
)
from ci_gate_target_only_reference import train_eval_target_only
from screening_core import build_system, pair_gate, rect_gate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MA_CACHE = ROOT / "simulation_results" / "ma2020_riemann_all_s1_s25_b499" / "cov_cache"
DEFAULT_STIEGER_CACHE = ROOT / "simulation_results" / "stieger_riemann_all_s1_s62_b499" / "cov_cache"
DEFAULT_OUT = ROOT / "simulation_results" / "rolling_targets"


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["covs"].astype(float), z["labels"].astype(int)


def subject_paths(dataset: str, cache_root: Path, subject: int) -> dict[int, Path]:
    directory = cache_root / (f"sub-{subject:03d}" if dataset == "ma2020" else f"S{subject}")
    paths: dict[int, Path] = {}
    for path in directory.glob("ses-*_cov.npz"):
        session = int(path.stem.split("-")[1].split("_")[0])
        paths[session] = path
    return dict(sorted(paths.items()))


def pair_ref_from_system(system: dict[str, np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    rect = rect_gate(system["lower_c"], system["upper_c"])
    pair = pair_gate(k, system["jj"], system["lower_d"])
    ref = np.asarray(sorted(set(rect.tolist()) & set(pair.tolist())), dtype=int)
    return rect, ref


def pooled(indices: np.ndarray, source_covs: list[np.ndarray], source_labels: list[np.ndarray]):
    if indices.size == 0:
        return None, None
    return (
        np.concatenate([source_covs[int(i)] for i in indices], axis=0),
        np.concatenate([source_labels[int(i)] for i in indices], axis=0),
    )


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def process_subject(
    dataset: str,
    subject: int,
    cache_root_string: str,
    min_sources: int,
    screening_boot: int,
    downstream_boot: int,
    downstream_splits: int,
    shrinkage: float,
    alpha: float,
    alpha_c: float,
    alpha_d: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    cache_root = Path(cache_root_string)
    paths = subject_paths(dataset, cache_root, subject)
    sessions = sorted(paths)
    loaded = {session: load_npz(path) for session, path in paths.items()}
    screening_rows: list[dict] = []
    downstream_rows: list[dict] = []
    selected_by_target: dict[int, dict[str, set[int]]] = {}
    dataset_offset = 0 if dataset == "ma2020" else 10_000_000

    for target_pos, target_session in enumerate(sessions):
        source_sessions = sessions[:target_pos]
        if len(source_sessions) < min_sources:
            continue
        source_covs = [loaded[s][0] for s in source_sessions]
        source_labels = [loaded[s][1] for s in source_sessions]
        target_covs, target_labels = loaded[target_session]
        k = len(source_sessions)

        rng = np.random.default_rng(seed + dataset_offset + subject * 10_000 + target_session * 100)
        delta_hat, delta_boot = riemann_estimates_and_bootstrap(
            source_covs, target_covs, rng, screening_boot, shrinkage, True
        )
        system = build_system(delta_hat, delta_boot, alpha_c, alpha_d, alpha)
        rect, pair_ref = pair_ref_from_system(system, k)
        mcs = mcs_stepdown(delta_hat, delta_boot, alpha)
        top1 = np.argsort(delta_hat, kind="mergesort")[:1]
        top3 = np.argsort(delta_hat, kind="mergesort")[: min(3, k)]
        selected = {
            "rect": rect,
            "pair_ref": pair_ref,
            "mcs_stepdown": mcs,
            "top1": top1,
            "top3": top3,
        }
        selected_by_target[target_session] = {
            name: {source_sessions[int(i)] for i in idx} for name, idx in selected.items()
        }
        row = {
            "dataset": dataset,
            "subject": subject,
            "target_session": target_session,
            "n_sources": k,
            "target_trials": int(target_covs.shape[0]),
        }
        for name, idx in selected.items():
            row[f"{name}_size"] = int(idx.size)
            row[f"{name}_fraction"] = float(idx.size / k)
            row[f"{name}_sessions"] = ",".join(str(source_sessions[int(i)]) for i in idx)
        screening_rows.append(row)

        splitter = StratifiedShuffleSplit(
            n_splits=downstream_splits,
            test_size=0.5,
            random_state=seed + dataset_offset + subject * 1_000 + target_session,
        )
        for split, (screen_idx, test_idx) in enumerate(splitter.split(target_covs, target_labels)):
            split_rng = np.random.default_rng(
                seed + dataset_offset + subject * 100_000 + target_session * 100 + split
            )
            delta_split, boot_split = riemann_estimates_and_bootstrap(
                source_covs,
                target_covs[screen_idx],
                split_rng,
                downstream_boot,
                shrinkage,
                True,
            )
            split_pair = screen_pair_ref(delta_split, boot_split, k, alpha_c, alpha_d, alpha)
            split_mcs = mcs_stepdown(delta_split, boot_split, alpha)
            split_top1 = np.argsort(delta_split, kind="mergesort")[:1]
            split_top3 = np.argsort(delta_split, kind="mergesort")[: min(3, k)]
            sets = {
                "all_sources": np.arange(k, dtype=int),
                "top1": split_top1,
                "top3": split_top3,
                "mcs_stepdown": split_mcs,
                "pair_ref": split_pair,
            }
            for method, indices in sets.items():
                train_covs, train_labels = pooled(indices, source_covs, source_labels)
                accuracy = train_eval_ts(
                    train_covs, train_labels, target_covs[test_idx], target_labels[test_idx]
                )
                downstream_rows.append(
                    {
                        "dataset": dataset,
                        "subject": subject,
                        "target_session": target_session,
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
            downstream_rows.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "target_session": target_session,
                    "split": split,
                    "method": "target_only",
                    "set_size": 0,
                    "n_sources": k,
                    "balanced_accuracy": target_accuracy,
                }
            )

    stability_rows: list[dict] = []
    targets = sorted(selected_by_target)
    for previous, current in zip(targets[:-1], targets[1:]):
        for method in ("rect", "pair_ref", "mcs_stepdown", "top3"):
            stability_rows.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "previous_target": previous,
                    "target_session": current,
                    "method": method,
                    "jaccard": jaccard(
                        selected_by_target[previous][method], selected_by_target[current][method]
                    ),
                }
            )
    return screening_rows, downstream_rows, stability_rows


def summarize(screening: pd.DataFrame, downstream: pd.DataFrame, stability: pd.DataFrame):
    screen_long = []
    for row in screening.itertuples(index=False):
        for method in ("rect", "pair_ref", "mcs_stepdown", "top1", "top3"):
            screen_long.append(
                {
                    "dataset": row.dataset,
                    "subject": row.subject,
                    "target_session": row.target_session,
                    "method": method,
                    "set_size": getattr(row, f"{method}_size"),
                    "retained_fraction": getattr(row, f"{method}_fraction"),
                }
            )
    screen_long = pd.DataFrame(screen_long)
    screen_subject = screen_long.groupby(["dataset", "subject", "method"], as_index=False).agg(
        mean_set_size=("set_size", "mean"),
        mean_retained_fraction=("retained_fraction", "mean"),
    )
    screen_summary = screen_subject.groupby(["dataset", "method"], as_index=False).agg(
        subjects=("subject", "nunique"),
        mean_set_size=("mean_set_size", "mean"),
        sd_set_size=("mean_set_size", "std"),
        mean_retained_fraction=("mean_retained_fraction", "mean"),
    )

    down_case = downstream.groupby(
        ["dataset", "subject", "target_session", "method"], as_index=False
    ).agg(balanced_accuracy=("balanced_accuracy", "mean"), set_size=("set_size", "mean"))
    down_subject = down_case.groupby(["dataset", "subject", "method"], as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean"), set_size=("set_size", "mean")
    )
    all_ref = down_subject[down_subject.method == "all_sources"][
        ["dataset", "subject", "balanced_accuracy"]
    ].rename(columns={"balanced_accuracy": "all_accuracy"})
    target_ref = down_subject[down_subject.method == "target_only"][
        ["dataset", "subject", "balanced_accuracy"]
    ].rename(columns={"balanced_accuracy": "target_accuracy"})
    down_subject = down_subject.merge(all_ref, on=["dataset", "subject"], how="left")
    down_subject = down_subject.merge(target_ref, on=["dataset", "subject"], how="left")
    down_subject["diff_vs_all"] = down_subject["balanced_accuracy"] - down_subject["all_accuracy"]
    down_subject["diff_vs_target"] = down_subject["balanced_accuracy"] - down_subject["target_accuracy"]
    down_summary = down_subject.groupby(["dataset", "method"], as_index=False).agg(
        subjects=("subject", "nunique"),
        mean_set_size=("set_size", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_diff_vs_all=("diff_vs_all", "mean"),
        mean_diff_vs_target=("diff_vs_target", "mean"),
        below_all_rate=("diff_vs_all", lambda x: float((x < 0).mean())),
        below_target_rate=("diff_vs_target", lambda x: float((x < 0).mean())),
    )
    stability_subject = stability.groupby(["dataset", "subject", "method"], as_index=False).agg(
        mean_jaccard=("jaccard", "mean")
    )
    stability_summary = stability_subject.groupby(["dataset", "method"], as_index=False).agg(
        subjects=("subject", "nunique"), mean_jaccard=("mean_jaccard", "mean")
    )
    return screen_long, screen_subject, screen_summary, down_subject, down_summary, stability_summary


def parse_subjects(text: str) -> list[int]:
    lo, hi = (text.split("-") + [text])[:2]
    return list(range(int(lo), int(hi) + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="ma2020,stieger")
    parser.add_argument("--ma-cache", type=Path, default=DEFAULT_MA_CACHE)
    parser.add_argument("--stieger-cache", type=Path, default=DEFAULT_STIEGER_CACHE)
    parser.add_argument("--ma-subjects", default="1-25")
    parser.add_argument("--stieger-subjects", default="1-62")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-sources", type=int, default=4)
    parser.add_argument("--screening-boot", type=int, default=199)
    parser.add_argument("--downstream-boot", type=int, default=99)
    parser.add_argument("--downstream-splits", type=int, default=1)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-c", type=float, default=0.025)
    parser.add_argument("--alpha-d", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    jobs = []
    for dataset in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        cache = args.ma_cache if dataset == "ma2020" else args.stieger_cache
        subjects = parse_subjects(args.ma_subjects if dataset == "ma2020" else args.stieger_subjects)
        for subject in subjects:
            if subject_paths(dataset, cache, subject):
                jobs.append((dataset, subject, str(cache)))

    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(process_subject)(
                dataset,
                subject,
                cache,
                args.min_sources,
                args.screening_boot,
                args.downstream_boot,
                args.downstream_splits,
                args.shrinkage,
                args.alpha,
                args.alpha_c,
                args.alpha_d,
                args.seed,
            )
            for dataset, subject, cache in jobs
        )

    screening = pd.DataFrame([row for result in results for row in result[0]])
    downstream = pd.DataFrame([row for result in results for row in result[1]])
    stability = pd.DataFrame([row for result in results for row in result[2]])
    screening.to_csv(args.out_dir / "rolling_screening_raw.csv", index=False)
    downstream.to_csv(args.out_dir / "rolling_downstream_raw.csv", index=False)
    stability.to_csv(args.out_dir / "rolling_stability_raw.csv", index=False)

    outputs = summarize(screening, downstream, stability)
    names = (
        "rolling_screening_long.csv",
        "rolling_screening_by_subject.csv",
        "rolling_screening_summary.csv",
        "rolling_downstream_by_subject.csv",
        "rolling_downstream_summary.csv",
        "rolling_stability_summary.csv",
    )
    for name, frame in zip(names, outputs):
        frame.to_csv(args.out_dir / name, index=False)

    manifest = {
        "script": Path(__file__).name,
        "datasets": sorted(screening.dataset.unique().tolist()),
        "subjects": screening.groupby("dataset").subject.nunique().to_dict(),
        "target_cases": screening.groupby("dataset").size().to_dict(),
        "min_sources": args.min_sources,
        "screening_boot": args.screening_boot,
        "downstream_boot": args.downstream_boot,
        "downstream_splits": args.downstream_splits,
        "shrinkage": args.shrinkage,
        "alpha": args.alpha,
        "alpha_c": args.alpha_c,
        "alpha_d": args.alpha_d,
        "seed": args.seed,
        "jobs": args.jobs,
        "ma_cache": str(args.ma_cache),
        "stieger_cache": str(args.stieger_cache),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(outputs[2].to_string(index=False))
    print(outputs[4].to_string(index=False))
    print(f"done in {manifest['runtime_seconds']}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
