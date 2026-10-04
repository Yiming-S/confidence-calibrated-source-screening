#!/usr/bin/env python3
"""Classifier-matched supervised target-only references.

The screening methods use target covariates but not target labels.  This script
uses the labelled screening half of each target session to fit the same
tangent-space LDA pipeline and evaluates it on the held-out half.  It therefore
provides a conventional no-transfer reference, while remaining explicitly
distinct from the label-free source-screening task.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ci_gate_ma2020_riemann import train_eval_ts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MA = ROOT / "simulation_results" / "ma2020_riemann_all_s1_s25_b499"
DEFAULT_STIEGER = ROOT / "simulation_results" / "stieger_riemann_all_s1_s62_b499"
DEFAULT_OUT = ROOT / "simulation_results" / "target_only_reference"
DEFAULT_TABLE = ROOT / "paper" / "tables" / "table_target_only_reference.tex"


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["covs"].astype(float), z["labels"].astype(int)


def train_eval_target_only(
    train_covs: np.ndarray,
    train_labels: np.ndarray,
    test_covs: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    """Use the source pipeline unchanged, fitted only to target training trials.

    The reference mean is estimated from the labeled target training half.
    Trial covariances use shrinkage 0.05; LDA uses LSQR and automatic shrinkage.
    Historical SVD results are archived separately and must not be mixed with
    results produced by this classifier-matched helper.
    """
    return train_eval_ts(train_covs, train_labels, test_covs, test_labels)


def target_path(dataset: str, cache_root: Path, subject: int) -> Path:
    if dataset == "ma2020":
        return cache_root / "cov_cache" / f"sub-{subject:03d}" / "ses-15_cov.npz"
    subject_dir = cache_root / "cov_cache" / f"S{subject}"
    paths = sorted(subject_dir.glob("ses-*_cov.npz"))
    if not paths:
        raise FileNotFoundError(subject_dir)
    return paths[-1]


def evaluate_subject(job: tuple[str, str, int, int, int]) -> list[dict[str, object]]:
    dataset, root_string, subject, splits, seed = job
    covs, labels = load_npz(target_path(dataset, Path(root_string), subject))
    splitter = StratifiedShuffleSplit(
        n_splits=splits, test_size=0.5, random_state=seed + subject
    )
    rows: list[dict[str, object]] = []
    for split, (train_idx, test_idx) in enumerate(splitter.split(covs, labels)):
        accuracy = train_eval_target_only(
            covs[train_idx], labels[train_idx], covs[test_idx], labels[test_idx]
        )
        rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "split": split,
                "method": "target_only",
                "balanced_accuracy": accuracy,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
            }
        )
    return rows


def parse_subjects(text: str) -> list[int]:
    lo, hi = (text.split("-") + [text])[:2]
    return list(range(int(lo), int(hi) + 1))


def existing_subjects(dataset: str, root: Path, requested: list[int]) -> list[int]:
    return [s for s in requested if target_path_exists(dataset, root, s)]


def target_path_exists(dataset: str, root: Path, subject: int) -> bool:
    try:
        return target_path(dataset, root, subject).exists()
    except FileNotFoundError:
        return False


def downstream_raw_path(dataset: str, root: Path) -> Path:
    return root / f"{dataset}_riemann_downstream_raw.csv"


def comparison_table(target_by_subject: pd.DataFrame, dataset: str, root: Path) -> pd.DataFrame:
    transfer = pd.read_csv(downstream_raw_path(dataset, root))
    transfer_by_subject = (
        transfer.groupby(["subject", "method"], as_index=False)["balanced_accuracy"].mean()
    )
    target = target_by_subject[target_by_subject["dataset"] == dataset][
        ["subject", "balanced_accuracy"]
    ].rename(columns={"balanced_accuracy": "target_only_accuracy"})
    merged = transfer_by_subject.merge(target, on="subject", how="inner")
    target_rows = target.assign(
        method="target_only", balanced_accuracy=target["target_only_accuracy"]
    )
    merged = pd.concat(
        [
            merged,
            target_rows[["subject", "method", "balanced_accuracy", "target_only_accuracy"]],
        ],
        ignore_index=True,
    )
    merged["diff_vs_target_only"] = merged["balanced_accuracy"] - merged["target_only_accuracy"]
    order = ["target_only", "all_sources", "top1", "top3", "pair_ref"]
    rows = []
    for method in order:
        part = merged[merged["method"] == method]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "subjects": int(part.shape[0]),
                "mean_balanced_accuracy": part["balanced_accuracy"].mean(),
                "mean_diff_vs_target_only": part["diff_vs_target_only"].mean(),
                "below_target_only_rate": float((part["diff_vs_target_only"] < 0).mean()),
                "subjects_above_target_only": int((part["diff_vs_target_only"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    dataset_labels = {"ma2020": "Ma2020", "stieger": "Stieger2021"}
    method_labels = {
        "target_only": "Target only",
        "all_sources": "All sources",
        "top1": "Top-1",
        "top3": "Top-3",
        "pair_ref": "Ref.",
    }
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{Supervised target-only reference.  Target only trains on the "
            r"labelled screening half of the target session and tests on the held-out "
            r"half. Both target-only and source-trained methods use the same "
            r"tangent-space LSQR LDA pipeline with automatic shrinkage and trial-"
            r"covariance shrinkage 0.05. Each reference mean is estimated only "
            r"from that method's training data. Source-screening methods do not "
            r"use target labels, so these are distinct training-data regimes. "
            r"Below-target rate is computed from subject-level mean accuracies.}"
        ),
        r"\label{tab:target-only-reference}",
        r"\WideTableBody",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        (
            r"Dataset & Method & \(n\) & \makecell{Bal.\\acc.} & \makecell{Diff. vs\\target} & "
            r"\makecell{Below-target\\rate} & \makecell{Above\\target} \\"
        ),
        r"\midrule",
    ]
    for dataset in ["ma2020", "stieger"]:
        block = summary[summary["dataset"] == dataset]
        for index, row in enumerate(block.itertuples(index=False)):
            dataset_cell = dataset_labels[dataset] if index == 0 else ""
            lines.append(
                f"{dataset_cell} & {method_labels[row.method]} & {row.subjects:d} & "
                f"{row.mean_balanced_accuracy:.4f} & {row.mean_diff_vs_target_only:+.4f} & "
                f"{row.below_target_only_rate:.2f} & {row.subjects_above_target_only:d} \\\\"
            )
        if dataset == "ma2020":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", "", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ma-dir", type=Path, default=DEFAULT_MA)
    parser.add_argument("--stieger-dir", type=Path, default=DEFAULT_STIEGER)
    parser.add_argument("--ma-subjects", default="1-25")
    parser.add_argument("--stieger-subjects", default="1-62")
    parser.add_argument("--splits", type=int, default=30)
    parser.add_argument("--ma-seed", type=int, default=20260618)
    parser.add_argument("--stieger-seed", type=int, default=20260707)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()

    started = time.time()
    jobs: list[tuple[str, str, int, int, int]] = []
    for dataset, root, requested, seed in [
        ("ma2020", args.ma_dir, parse_subjects(args.ma_subjects), args.ma_seed),
        ("stieger", args.stieger_dir, parse_subjects(args.stieger_subjects), args.stieger_seed),
    ]:
        for subject in existing_subjects(dataset, root, requested):
            jobs.append((dataset, str(root), subject, args.splits, seed))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        nested = list(pool.map(evaluate_subject, jobs))
    raw = pd.DataFrame([row for subject_rows in nested for row in subject_rows])
    raw.to_csv(args.out_dir / "target_only_raw.csv", index=False)
    by_subject = (
        raw.groupby(["dataset", "subject", "method"], as_index=False)
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            n_train=("n_train", "first"),
            n_test=("n_test", "first"),
        )
    )
    by_subject.to_csv(args.out_dir / "target_only_by_subject.csv", index=False)
    summary = pd.concat(
        [
            comparison_table(by_subject, "ma2020", args.ma_dir),
            comparison_table(by_subject, "stieger", args.stieger_dir),
        ],
        ignore_index=True,
    )
    summary.to_csv(args.out_dir / "target_only_comparison.csv", index=False)
    write_latex_table(summary, args.table_path)
    manifest = {
        "script": Path(__file__).name,
        "ma_dir": str(args.ma_dir),
        "stieger_dir": str(args.stieger_dir),
        "splits": args.splits,
        "ma_seed": args.ma_seed,
        "stieger_seed": args.stieger_seed,
        "jobs": args.jobs,
        "classifier": "tangent-space LDA (solver=lsqr, shrinkage=auto)",
        "trial_covariance_shrinkage": 0.05,
        "target_reference": "Riemannian mean of target training half only",
        "table_path": str(args.table_path),
        "subjects": raw.groupby("dataset")["subject"].nunique().to_dict(),
        "runtime_seconds": round(time.time() - started, 1),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
