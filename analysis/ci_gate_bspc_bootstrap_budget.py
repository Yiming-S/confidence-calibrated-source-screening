#!/usr/bin/env python3
"""Small, fixed-design bootstrap-budget stability check for the BSPC revision.

This is a numerical stability check, not a population-coverage experiment.
Only previously extracted covariance caches are read.  No raw preprocessing,
classifier fitting, coverage claim, or outcome-based case selection is added.
For each fixed sample and seed, all budgets use prefixes of one B=999 draw
matrix; standard errors and quantiles are recomputed separately at each budget.
"""

from __future__ import annotations

import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import re
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_info, threadpool_limits

from ci_gate_ma2020_riemann import airm2, shrink
from ci_gate_realdata_case_study import build_system, pair_gate, rect_gate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "simulation_results/bspc_revision_20261002/bootstrap_budget"
SPLIT_SEEDS = {"ma2020": 20260618, "stieger2021": 20260707, "zhou2020": 20260714}
DATASET_IDS = {"ma2020": 1, "stieger2021": 2, "zhou2020": 3}
BOOTSTRAP_SEEDS = (2026100201, 2026100202, 2026100203)
BUDGETS = (99, 199, 499, 999)
REFERENCE_FILES = ("ci_gate_ma2020_riemann.py", "ci_gate_realdata_case_study.py",
                   "ci_gate_stieger_riemann.py", "ci_gate_zhou2020_riemann.py",
                   "screening_core.py", "path_config.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def discrepancy_bootstrap(sources: list[np.ndarray], target: np.ndarray,
                          rng: np.random.Generator, n_boot: int,
                          shrinkage: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Same draws/formulas as the primary script; reuse target means once.

    Multinomial rows generate the ordinary empirical bootstrap.  Matrix
    multiplication is an algebraically identical faster weighted-mean step.
    The regression tests compare this with the original einsum implementation.
    """
    nt, p, _ = target.shape
    target_mean = shrink(target.mean(axis=0), shrinkage)
    estimate = np.array([airm2(shrink(x.mean(axis=0), shrinkage), target_mean)
                         for x in sources])
    target_counts = rng.multinomial(nt, np.full(nt, 1.0 / nt), size=n_boot).astype(float)
    target_means = (target_counts @ target.reshape(nt, -1) / nt).reshape(n_boot, p, p)
    target_means = np.stack([shrink(x, shrinkage) for x in target_means])
    draws = np.empty((n_boot, len(sources)))
    for j, source in enumerate(sources):
        ns = len(source)
        counts = rng.multinomial(ns, np.full(ns, 1.0 / ns), size=n_boot).astype(float)
        means = (counts @ source.reshape(ns, -1) / ns).reshape(n_boot, p, p)
        for b in range(n_boot):
            draws[b, j] = airm2(shrink(means[b], shrinkage), target_means[b])
    return estimate, draws


def calibrate(estimate: np.ndarray, draws: np.ndarray, alpha_c: float = 0.025,
              alpha_d: float = 0.025) -> dict[str, object]:
    """Recompute the primary Bonferroni-split gate for one draw prefix."""
    system = build_system(estimate, draws, alpha_c, alpha_d, 0.05)
    rect = rect_gate(system["lower_c"], system["upper_c"])
    pair = pair_gate(len(estimate), system["jj"], system["lower_d"])
    system["selected"] = np.asarray(sorted(set(rect) & set(pair)), dtype=int)
    system["raw_se_c"] = draws.std(axis=0, ddof=1)
    system["raw_se_d"] = (draws[:, system["jj"]] - draws[:, system["ll"]]).std(axis=0, ddof=1)
    return system


def jaccard(left: list[int] | np.ndarray, right: list[int] | np.ndarray) -> float:
    a, b = set(left), set(right)
    return float(len(a & b) / len(a | b)) if a | b else 1.0


def split_indices(labels: np.ndarray, dataset: str, subject: int, n_splits: int):
    # The first n_splits of the original 30-split sequence, not a new seed.
    splitter = StratifiedShuffleSplit(n_splits=30, test_size=0.5,
                                     random_state=SPLIT_SEEDS[dataset] + subject)
    return list(itertools.islice(splitter.split(np.zeros(len(labels)), labels), n_splits))


def subject_candidates(dataset: str, cache_root: Path) -> tuple[list[dict], list[dict]]:
    prefix = "sub-" if dataset == "ma2020" else "S"
    valid, excluded = [], []
    for directory in sorted(cache_root.glob(f"{prefix}*")):
        match = re.fullmatch(re.escape(prefix) + r"(\d+)", directory.name)
        if not match or not directory.is_dir():
            continue
        subject = int(match.group(1))
        pattern = "session-*_cov.npz" if dataset == "zhou2020" else "ses-*_cov.npz"
        paths = sorted(directory.glob(pattern))
        sessions = [int(re.search(r"(?:ses|session)-(\d+)", p.name).group(1)) for p in paths]
        reason = None
        if dataset == "ma2020" and sessions != list(range(1, 16)):
            reason = "requires sessions 1 through 15"
        if dataset == "stieger2021" and (len(sessions) < 6 or max(sessions, default=0) > 11):
            reason = "requires at least six cached sessions within 1 through 11"
        if dataset == "zhou2020" and (len(sessions) not in (6, 7)
                                      or sessions != list(range(1, len(sessions) + 1))):
            reason = "requires six or seven consecutive cached sessions"
        if reason:
            excluded.append({"subject": subject, "reason": reason})
        else:
            valid.append({"dataset": dataset, "subject": subject, "sessions": sessions,
                          "cache_paths": [str(p.resolve()) for p in paths]})
    return sorted(valid, key=lambda x: x["subject"]), excluded


def load_subject(record: dict) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[dict]]:
    covs, labels, qc = [], [], []
    for session, path_string in zip(record["sessions"], record["cache_paths"]):
        path = Path(path_string)
        with np.load(path, allow_pickle=False) as data:
            x = np.asarray(data["covs"], dtype=float)
            y = np.asarray(data["labels"], dtype=int)
            if record["dataset"] == "zhou2020":
                expected = {"l_freq": 8, "h_freq": 30, "tmin": 0, "tmax": 5,
                            "n_eeg": 41 if record["subject"] <= 12 else 26}
                if any(key not in data or not np.isclose(float(data[key]), value)
                       for key, value in expected.items()):
                    raise ValueError(f"Zhou cache does not match the original protocol: {path}")
        if x.ndim != 3 or x.shape[1] != x.shape[2] or len(x) != len(y):
            raise ValueError(f"invalid covariance/label dimensions: {path}")
        if not np.isfinite(x).all() or np.unique(y).size != 2:
            raise ValueError(f"nonfinite covariances or nonbinary labels: {path}")
        if not np.allclose(x, x.swapaxes(1, 2), rtol=1e-10, atol=1e-15):
            raise ValueError(f"nonsymmetric covariance cache: {path}")
        trace = np.trace(x, axis1=1, axis2=2)
        if np.any(trace <= 0):
            raise ValueError(f"nonpositive covariance trace: {path}")
        covs.append(x)
        labels.append(y)
        qc.append({"dataset": record["dataset"], "subject": record["subject"],
                   "session": session, "n_trials": len(x), "n_channels": x.shape[1],
                   "trace_min": float(trace.min()), "trace_max": float(trace.max()),
                   "frobenius_max": float(np.linalg.norm(x, axis=(1, 2)).max()),
                   "cache_path": str(path), "cache_sha256": sha256_file(path)})
    if len({x.shape[1] for x in covs}) != 1:
        raise ValueError("channel counts differ within a subject")
    return covs[:-1], covs[-1], labels[-1], qc


def make_plan(args) -> tuple[dict, list[dict]]:
    records, availability, qc = [], {}, []
    roots = {"ma2020": args.ma_cache, "stieger2021": args.stieger_cache,
             "zhou2020": args.zhou_cache}
    for dataset, cache_root in roots.items():
        candidates, excluded = subject_candidates(dataset, cache_root)
        chosen = candidates[:args.subjects_per_dataset]
        if not chosen:
            raise FileNotFoundError(f"no valid cached subjects: {dataset}, {cache_root}")
        availability[dataset] = {"cache_root": str(cache_root.resolve()),
                                 "available_subjects": [r["subject"] for r in candidates],
                                 "selected_subjects": [r["subject"] for r in chosen],
                                 "invalid_candidates": excluded,
                                 "shortfall": max(0, args.subjects_per_dataset - len(chosen))}
        for record in chosen:
            _, target, target_labels, stats = load_subject(record)
            splits = split_indices(target_labels, dataset, record["subject"], args.splits)
            record["splits"] = [{"split": i, "screen_indices": screen.tolist(),
                                  "test_indices": test.tolist()}
                                 for i, (screen, test) in enumerate(splits)]
            record["n_target_trials"] = len(target)
            record["input_hashes"] = {x["cache_path"]: x["cache_sha256"] for x in stats}
            qc.extend(stats)
            records.append(record)
    plan = {"design_version": 1, "datasets": records, "availability": availability,
            "subjects_per_dataset_requested": args.subjects_per_dataset,
            "splits_per_subject": args.splits, "split_seed_bases": SPLIT_SEEDS,
            "bootstrap_seeds": list(BOOTSTRAP_SEEDS), "budgets": list(BUDGETS),
            "alpha_component": 0.025, "alpha_contrast": 0.025, "shrinkage": 0.1,
            "source_choice": "all cached sessions before the latest target; Ma target 15",
            "case_selection": "first available subject IDs, selected before computing results",
            "resampling": "ordinary empirical bootstrap; one target draw shared by every source",
            "common_draws": "all budgets are prefixes of one 999-row discrepancy draw matrix per seed",
            "aggregation": "average seeds and splits within subject, then equally average subjects",
            "purpose": "numerical stability only; no coverage or oracle-retention estimate",
            "reference_source_sha256": {name: sha256_file(Path(__file__).resolve().parent / name)
                                         for name in REFERENCE_FILES}}
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan, qc


def case_name(record: dict, split: int, seed: int) -> str:
    return f"{record['dataset']}_s{record['subject']:03d}_split{split:02d}_seed{seed}"


def run_case(task: tuple[dict, int, int, str, str]) -> dict:
    record, split_index, seed, out_string, plan_hash = task
    out = Path(out_string)
    name = case_name(record, split_index, seed)
    archive = out / "draws" / f"{name}.npz"
    if archive.exists():
        with np.load(archive, allow_pickle=False) as old:
            if str(old["plan_sha256"]) != plan_hash:
                raise ValueError(f"existing draws have a different fixed plan: {archive}")
        return {"case": name, "reused": True, "archive": str(archive)}
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        sources, target, labels, _ = load_subject(record)
        entry = record["splits"][split_index]
        screen = np.asarray(entry["screen_indices"], dtype=int)
        test = np.asarray(entry["test_indices"], dtype=int)
        assert not set(screen) & set(test)
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, DATASET_IDS[record["dataset"]], record["subject"], split_index]))
        estimate, draws = discrepancy_bootstrap(sources, target[screen], rng, max(BUDGETS))
    elapsed = time.perf_counter() - started
    np.savez_compressed(archive, estimate=estimate, draws=draws, screen_indices=screen,
                        test_indices=test, source_sessions=np.asarray(record["sessions"][:-1]),
                        plan_sha256=np.asarray(plan_hash), runtime_seconds=np.asarray(elapsed))
    print(json.dumps({"case": name, "runtime_seconds": round(elapsed, 3),
                      "n_channels": target.shape[1], "n_sources": len(sources)}), flush=True)
    return {"case": name, "reused": False, "runtime_seconds": elapsed,
            "n_channels": target.shape[1], "n_sources": len(sources), "archive": str(archive)}


def subject_aggregation(frame: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the subject as the aggregation unit, including unbalanced test fixtures."""
    by_subject = frame.groupby(["dataset", "subject", "budget"], as_index=False)[metrics].mean()
    summary = by_subject.groupby(["dataset", "budget"], as_index=False)[metrics].mean()
    sizes = by_subject.groupby(["dataset", "budget"])["subject"].nunique().rename("n_subjects")
    return by_subject, summary.merge(sizes, on=["dataset", "budget"])


def summarize(plan: dict, out: Path) -> dict:
    rows, memberships = [], []
    for record in plan["datasets"]:
        for split in record["splits"]:
            for seed in BOOTSTRAP_SEEDS:
                path = out / "draws" / f"{case_name(record, split['split'], seed)}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    estimate, draws = saved["estimate"], saved["draws"]
                    if str(saved["plan_sha256"]) != plan["plan_sha256"]:
                        raise ValueError("draw archive does not match the fixed plan")
                systems = {b: calibrate(estimate, draws[:b]) for b in BUDGETS}
                reference = systems[max(BUDGETS)]
                selected_ref = set(reference["selected"].tolist())
                sessions = np.asarray(record["sessions"][:-1])
                for budget, system in systems.items():
                    selected = set(system["selected"].tolist())
                    row = {"dataset": record["dataset"], "subject": record["subject"],
                           "target_session": record["sessions"][-1], "split": split["split"],
                           "bootstrap_seed": seed, "budget": budget, "n_sources": len(sessions),
                           "set_size": len(selected), "members": json.dumps(sessions[sorted(selected)].tolist()),
                           "jaccard_vs_999": jaccard(sorted(selected), sorted(selected_ref)),
                           "same_members_vs_999": int(selected == selected_ref),
                           "size_change_vs_999": len(selected) - len(selected_ref),
                           "absolute_size_change_vs_999": abs(len(selected) - len(selected_ref)),
                           "members_added_vs_999": len(selected - selected_ref),
                           "members_removed_vs_999": len(selected_ref - selected),
                           "q_component": system["q_c"], "q_contrast": system["q_d"],
                           "q_component_change_vs_999": system["q_c"] - reference["q_c"],
                           "q_contrast_change_vs_999": system["q_d"] - reference["q_d"],
                           "q_component_abs_change_vs_999": abs(system["q_c"] - reference["q_c"]),
                           "q_contrast_abs_change_vs_999": abs(system["q_d"] - reference["q_d"])}
                    for label, key in (("component", "raw_se_c"), ("contrast", "raw_se_d")):
                        raw_se = system[key]
                        row[f"raw_{label}_se_min"] = float(raw_se.min())
                        row[f"raw_{label}_se_max"] = float(raw_se.max())
                        row[f"raw_{label}_se_nonfinite_count"] = int(np.sum(~np.isfinite(raw_se)))
                        row[f"raw_{label}_se_floor_count"] = int(np.sum(raw_se < 1e-12))
                    rows.append(row)
                    for j, session in enumerate(sessions):
                        memberships.append({"dataset": record["dataset"], "subject": record["subject"],
                                            "split": split["split"], "bootstrap_seed": seed, "budget": budget,
                                            "source_session": int(session), "selected": int(j in selected),
                                            "selected_at_999": int(j in selected_ref),
                                            "discrepancy": float(estimate[j]), "se_component": float(system["se_c"][j]),
                                            "raw_se_component": float(system["raw_se_c"][j]),
                                            "lower_component": float(system["lower_c"][j]),
                                            "upper_component": float(system["upper_c"][j])})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "case_results.csv", index=False)
    pd.DataFrame(memberships).to_csv(out / "source_membership.csv", index=False)
    metrics = ["set_size", "jaccard_vs_999", "same_members_vs_999", "size_change_vs_999",
               "absolute_size_change_vs_999", "members_added_vs_999", "members_removed_vs_999",
               "q_component", "q_contrast", "q_component_change_vs_999", "q_contrast_change_vs_999",
               "q_component_abs_change_vs_999", "q_contrast_abs_change_vs_999"]
    by_subject, summary = subject_aggregation(frame, metrics)
    by_subject.to_csv(out / "subject_summary.csv", index=False)
    summary.to_csv(out / "dataset_summary.csv", index=False)
    diagnostic_rows = []
    for (dataset, count), group in frame.groupby(["dataset", "budget"]):
        item = {"dataset": dataset, "budget": count,
                "n_subjects": int(group.subject.nunique()), "n_seed_sample_cases": len(group),
                "n_component_coordinates": int(group.n_sources.sum()),
                "n_contrast_coordinates": int((group.n_sources * (group.n_sources - 1)).sum()),
                "minimum_jaccard_vs_999": float(group.jaccard_vs_999.min()),
                "changed_set_cases_vs_999": int((1 - group.same_members_vs_999).sum()),
                "maximum_absolute_size_change_vs_999": int(group.absolute_size_change_vs_999.max())}
        for label in ("component", "contrast"):
            item[f"raw_{label}_se_min"] = float(group[f"raw_{label}_se_min"].min())
            item[f"raw_{label}_se_max"] = float(group[f"raw_{label}_se_max"].max())
            item[f"raw_{label}_se_nonfinite_count"] = int(group[f"raw_{label}_se_nonfinite_count"].sum())
            item[f"raw_{label}_se_floor_count"] = int(group[f"raw_{label}_se_floor_count"].sum())
        diagnostic_rows.append(item)
    pd.DataFrame(diagnostic_rows).to_csv(out / "se_diagnostics_summary.csv", index=False)
    seed_rows = []
    for (dataset, subject, split, budget), group in frame.groupby(["dataset", "subject", "split", "budget"]):
        sets = [json.loads(value) for value in group["members"]]
        pairs = list(itertools.combinations(sets, 2))
        seed_rows.append({"dataset": dataset, "subject": subject, "split": split, "budget": budget,
                          "pairwise_seed_jaccard": float(np.mean([jaccard(a, b) for a, b in pairs])),
                          "pairwise_seed_exact": float(np.mean([set(a) == set(b) for a, b in pairs])),
                          "set_size_seed_sd": float(group["set_size"].std(ddof=1)),
                          "set_size_seed_range": int(group["set_size"].max() - group["set_size"].min()),
                          "q_component_seed_sd": float(group["q_component"].std(ddof=1)),
                          "q_contrast_seed_sd": float(group["q_contrast"].std(ddof=1))})
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(out / "seed_variability_cases.csv", index=False)
    seed_metrics = [c for c in seeds.columns if c not in ("dataset", "subject", "split", "budget")]
    seed_subjects, seed_summary = subject_aggregation(seeds, seed_metrics)
    seed_subjects.to_csv(out / "seed_variability_subjects.csv", index=False)
    seed_summary.to_csv(out / "seed_variability_summary.csv", index=False)
    expected = len(plan["datasets"]) * plan["splits_per_subject"] * len(BOOTSTRAP_SEEDS) * len(BUDGETS)
    assert len(frame) == expected
    assert not frame.duplicated(["dataset", "subject", "split", "bootstrap_seed", "budget"]).any()
    assert frame.loc[frame.budget == max(BUDGETS), "same_members_vs_999"].eq(1).all()
    return {"n_subjects": len(plan["datasets"]), "n_fixed_samples": len(seed_rows) // len(BUDGETS),
            "n_seed_sample_draw_arrays": expected // len(BUDGETS), "n_budget_results": expected,
            "minimum_jaccard_vs_999": float(frame.jaccard_vs_999.min()),
            "maximum_absolute_size_change_vs_999": int(frame.absolute_size_change_vs_999.max()),
            "component_floor_count": int(frame.raw_component_se_floor_count.sum()),
            "contrast_floor_count": int(frame.raw_contrast_se_floor_count.sum()),
            "component_nonfinite_count": int(frame.raw_component_se_nonfinite_count.sum()),
            "contrast_nonfinite_count": int(frame.raw_contrast_se_nonfinite_count.sum())}


def update_hashes(out: Path) -> None:
    paths = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    lines = [f"{sha256_file(p)}  {p.relative_to(out)}" for p in paths]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ma-cache", type=Path, default=ROOT / "simulation_results/ma2020_riemann_all_s1_s25_b499/cov_cache")
    parser.add_argument("--stieger-cache", type=Path, default=ROOT / "simulation_results/stieger_riemann_all_s1_s62_b499/cov_cache")
    parser.add_argument("--zhou-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subjects-per-dataset", type=int, default=3)
    parser.add_argument("--splits", type=int, default=2)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--mode", choices=("prepare", "pilot", "run"), default="run")
    args = parser.parse_args()
    if not 1 <= args.splits <= 30 or args.subjects_per_dataset < 1:
        parser.error("positive subject count and 1 through 30 splits required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "draws").mkdir(exist_ok=True)
    plan, qc = make_plan(args)
    plan_path = args.out_dir / "fixed_plan.json"
    if plan_path.exists():
        prior = json.loads(plan_path.read_text())
        if prior["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("fixed design changed; use a separate output directory")
    else:
        # This is saved before the first random bootstrap draw or result is computed.
        json_dump(plan_path, plan)
    pd.DataFrame(qc).to_csv(args.out_dir / "input_cache_diagnostics.csv", index=False)
    command = shlex.join([sys.executable, *sys.argv])
    commands_path = args.out_dir / "commands.json"
    commands = json.loads(commands_path.read_text()) if commands_path.exists() else []
    commands.append({"utc": datetime.now(timezone.utc).isoformat(), "command": command, "mode": args.mode})
    json_dump(commands_path, commands)
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "availability": plan["availability"]}), flush=True)
    if args.mode == "prepare":
        update_hashes(args.out_dir)
        return
    tasks = [(record, split["split"], seed, str(args.out_dir.resolve()), plan["plan_sha256"])
             for record in plan["datasets"] for split in record["splits"] for seed in BOOTSTRAP_SEEDS]
    if args.mode == "pilot":
        task = next(x for x in tasks if x[0]["dataset"] == "stieger2021")
        result = run_case(task)
        json_dump(args.out_dir / "pilot_timing.json", result)
        update_hashes(args.out_dir)
        return
    started = time.perf_counter()
    if args.workers == 1:
        timings = [run_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            timings = list(executor.map(run_case, tasks))
    audit = summarize(plan, args.out_dir)
    versions = {package: importlib.metadata.version(package)
                for package in ("numpy", "scipy", "pandas", "scikit-learn", "threadpoolctl")}
    manifest = {"completed_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "executable": sys.executable, "platform": platform.platform(), "packages": versions,
                "threadpool_info": threadpool_info(), "workers": args.workers,
                "thread_environment": {key: os.environ[key] for key in (
                    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                "script_sha256": sha256_file(Path(__file__)), "plan_sha256": plan["plan_sha256"],
                "runtime_seconds": time.perf_counter() - started, "audit": audit,
                "timings": timings, "interpretation": plan["purpose"]}
    json_dump(args.out_dir / "manifest.json", manifest)
    update_hashes(args.out_dir)
    print(json.dumps({"completed": True, **audit}), flush=True)


if __name__ == "__main__":
    main()
