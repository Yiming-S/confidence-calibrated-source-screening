#!/usr/bin/env python3
"""Frozen, exploratory Kumar2024 analysis; never edits manuscript or old results.

Input: 108 validated covariance caches, sub-01/session-01_cov.npz through
sub-18/session-06_cov.npz. The protocol JSON must contain the fixed ``analysis``
mapping below. Extra protocol metadata are retained and included in its hash.
All confidence intervals use subjects, not repeated splits, as observations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

# Set before importing numerical libraries, including in spawned workers.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_limits


FIXED_ANALYSIS = {
    "seed": 20260714,
    "subjects": list(range(1, 19)),
    "sessions": list(range(1, 7)),
    "target_session": 6,
    "source_sessions": list(range(1, 6)),
    "n_channels": 22,
    "min_trials_per_class": 4,
    "screening_boot": 499,
    "downstream_boot": 199,
    "n_splits": 30,
    "test_size": 0.5,
    "shrinkage": 0.1,
    "alpha": 0.05,
    "alpha_c": 0.025,
    "alpha_d": 0.025,
    "audit_outer": 50,
    "audit_boot": 499,
    "source_fraction": 1.0,
    "target_fraction": 0.5,
}
SOURCE_METHODS = ("all_sources", "top1", "top3", "mcs_stepdown", "pair_ref")
METHODS = SOURCE_METHODS + ("target_only",)
COVERAGE_METRICS = (
    "coverage_c", "coverage_d", "coverage_joint", "oracle_retained",
    "exact_recovery", "set_size", "retained_fraction", "worst_reference_excess",
)
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value) -> None:
    """Atomic replace: a terminated worker cannot leave a half JSON checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_protocol(path: Path) -> tuple[dict, str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or not isinstance(protocol.get("analysis"), dict):
        raise ValueError("Protocol must contain an analysis object")
    analysis = protocol["analysis"]
    if set(analysis) != set(FIXED_ANALYSIS):
        raise ValueError("Protocol analysis keys differ from the frozen design: "
                         f"missing={sorted(set(FIXED_ANALYSIS) - set(analysis))}, "
                         f"extra={sorted(set(analysis) - set(FIXED_ANALYSIS))}")
    for key, expected in FIXED_ANALYSIS.items():
        if analysis[key] != expected or isinstance(analysis[key], bool):
            raise ValueError(f"Frozen protocol mismatch for {key}: {analysis[key]!r} != {expected!r}")
    return protocol, sha256_file(path)


def load_backend(simulations_root: Path) -> SimpleNamespace:
    """Only import the explicitly requested, frozen simulations directory."""
    root = simulations_root.resolve(strict=True)
    required = ("ci_gate_bnci004_riemann", "ci_gate_ma2020_riemann",
                "ci_gate_eeg_empirical_audit")
    for name in required:
        if not (root / f"{name}.py").is_file():
            raise FileNotFoundError(root / f"{name}.py")
    sys.path.insert(0, str(root))
    modules = [importlib.import_module(name) for name in required]
    for name, module in tuple(sys.modules.items()):
        if (name.startswith("ci_gate_") or name in {"path_config", "plot_style"}) and getattr(module, "__file__", None):
            if Path(module.__file__).resolve().parent != root:
                raise RuntimeError(f"Wrong simulations module loaded: {name}: {module.__file__}")
    bnci, riemann, audit = modules
    return SimpleNamespace(selected_sets=bnci.selected_sets, pool=bnci.pool,
                           bootstrap=riemann.riemann_estimates_and_bootstrap,
                           train_eval_ts=riemann.train_eval_ts,
                           discrepancy_vector=audit.discrepancy_vector,
                           evaluate_outer=audit.evaluate_outer,
                           mean_pairwise_jaccard=audit.mean_pairwise_jaccard)


def code_manifest(simulations_root: Path) -> dict:
    paths = sorted(simulations_root.resolve(strict=True).glob("*.py"))
    if not paths:
        raise ValueError("Empty simulations snapshot")
    files = {f"simulations/{p.name}": sha256_file(p) for p in paths}
    files["analyze.py"] = sha256_file(Path(__file__).resolve())
    return {"files": files, "sha256": canonical_hash(files)}


def session_path(cache_root: Path, subject: int, session: int) -> Path:
    return cache_root / f"sub-{subject:02d}" / f"session-{session:02d}_cov.npz"


def load_session(path: Path, settings: dict, protocol_hash: str, channels: list[str]) -> dict:
    with np.load(path, allow_pickle=False) as data:
        covs, labels, ids = (data[k] for k in ("covs", "labels", "trial_ids"))
        cache_protocol = str(data["protocol_sha256"].item())
        unit = str(data["unit"].item())
        cache_channels = data["channels"].astype(str).tolist()
    c = settings["n_channels"]
    if cache_protocol != protocol_hash:
        raise ValueError(f"{path}: cache preprocessing protocol hash mismatch")
    if unit != "microvolt_squared":
        raise ValueError(f"{path}: expected covariance unit microvolt_squared, found {unit}")
    if len(channels) != c or cache_channels != channels:
        raise ValueError(f"{path}: EEG channel names or order differ from the protocol")
    if covs.dtype != np.float64 or covs.ndim != 3 or covs.shape[1:] != (c, c):
        raise ValueError(f"{path}: covs must be float64 N x {c} x {c}")
    if labels.shape != (len(covs),) or ids.shape != (len(covs),):
        raise ValueError(f"{path}: covariances, labels, and trial IDs are not aligned")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"{path}: labels must contain exactly 0 and 1")
    counts = [int(np.count_nonzero(labels == label)) for label in (0, 1)]
    if min(counts) < settings["min_trials_per_class"]:
        raise ValueError(f"{path}: fewer than {settings['min_trials_per_class']} trials per class: {counts}")
    if ids.dtype.kind not in "US" or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: trial_ids must be unique strings")
    ids = ids.astype(str)
    if np.any(ids == ""):
        raise ValueError(f"{path}: trial_ids must not be empty")
    if not np.isfinite(covs).all() or not np.allclose(covs, covs.swapaxes(1, 2), rtol=1e-10, atol=1e-10):
        raise ValueError(f"{path}: covariance matrices must be finite and symmetric")
    trace = np.trace(covs, axis1=1, axis2=2)
    if np.any(trace <= 0):
        raise ValueError(f"{path}: covariance trace must be positive")
    return {"covs": covs, "labels": labels.astype(np.int64), "trial_ids": ids,
            "n_trials": len(covs), "class_counts": counts,
            "protocol_sha256": cache_protocol, "unit": unit, "channels": cache_channels}


def subject_info(subject: int) -> dict:
    return {"subject": subject, "raw_subject": subject if subject <= 9 else subject + 1,
            "group": "GR" if subject <= 9 else "PAR"}


def preflight(cache_root: Path, settings: dict, protocol_hash: str, channels: list[str]) -> dict:
    """Finish every subject/session check before submitting any analysis worker."""
    result = {}
    seen_ids = set()
    for subject in settings["subjects"]:
        sessions = {}
        for session in settings["sessions"]:
            path = session_path(cache_root, subject, session)
            loaded = load_session(path, settings, protocol_hash, channels)
            current_ids = set(loaded["trial_ids"].tolist())
            if seen_ids.intersection(current_ids):
                raise ValueError(f"{path}: trial IDs overlap a previous session or subject")
            seen_ids.update(current_ids)
            sessions[str(session)] = {
                "relative_path": str(path.relative_to(cache_root)), "sha256": sha256_file(path),
                "n_trials": loaded["n_trials"], "class_counts": loaded["class_counts"],
                "trial_ids": loaded["trial_ids"].tolist(),
                "protocol_sha256": loaded["protocol_sha256"],
                "unit": loaded["unit"], "channels": loaded["channels"],
            }
        result[str(subject)] = {**subject_info(subject), "sessions": sessions,
                                "sha256": canonical_hash(sessions)}
    return result


def normalized_sets(delta_hat, delta_boot, settings: dict, backend) -> dict:
    if not np.isfinite(delta_hat).all() or not np.isfinite(delta_boot).all():
        raise ValueError("Nonfinite discrepancy or bootstrap output")
    selected = backend.selected_sets(delta_hat, delta_boot, settings["alpha"],
                                     settings["alpha_c"], settings["alpha_d"])
    selected["all_sources"] = np.arange(len(settings["source_sessions"]), dtype=int)
    answer = {}
    for method in SOURCE_METHODS:
        indices = np.asarray(sorted(np.asarray(selected[method], dtype=int).tolist()), dtype=int)
        if not len(indices) or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(delta_hat)):
            raise ValueError(f"Invalid/empty selected set for {method}: {indices}")
        answer[method] = indices
    return answer


def selection_record(indices, settings: dict, sources: list[dict]) -> dict:
    return {"source_indices": indices.tolist(),
            "source_sessions": [settings["source_sessions"][int(i)] for i in indices],
            "set_size": int(len(indices)),
            "training_trials": int(sum(sources[int(i)]["n_trials"] for i in indices))}


def bootstrap_record(delta_hat, delta_boot, seed: int, budget: int) -> dict:
    return {"seed": seed, "bootstrap_repetitions": budget,
            "discrepancy_estimates": np.asarray(delta_hat).tolist(),
            "shared_bootstrap_matrix_sha256": hashlib.sha256(
                np.asarray(delta_boot, dtype="<f8").tobytes()).hexdigest()}


def load_checkpoint(path: Path, identity: dict, settings: dict) -> dict:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "identity": identity,
                **subject_info(identity["subject"]), "screening": None,
                "downstream": [], "coverage": [], "complete": False}
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("schema_version") != SCHEMA_VERSION or checkpoint.get("identity") != identity:
        raise ValueError(f"Checkpoint protocol/code/cache identity mismatch: {path}")
    for key, index, maximum in (("downstream", "split", settings["n_splits"]),
                                ("coverage", "replicate", settings["audit_outer"])):
        rows = checkpoint.get(key, [])
        if len(rows) > maximum or [r[index] for r in rows] != list(range(len(rows))):
            raise ValueError(f"Invalid checkpoint {key} indices: {path}")
    if checkpoint.get("complete") and (checkpoint.get("screening") is None
            or len(checkpoint["downstream"]) != settings["n_splits"]
            or len(checkpoint["coverage"]) != settings["audit_outer"]):
        raise ValueError(f"Incomplete checkpoint marked complete: {path}")
    return checkpoint


def process_subject(subject: int, cache_root_string: str, out_dir_string: str,
                    simulations_root_string: str, settings: dict, identity: dict,
                    cache_manifest: dict, backend=None) -> dict:
    started = time.monotonic()
    with threadpool_limits(limits=1):
        result = _process_subject(subject, Path(cache_root_string), Path(out_dir_string),
                                  Path(simulations_root_string), settings, identity,
                                  cache_manifest, backend)
    print(f"Subject {subject:02d} checkpoint complete; elapsed_seconds={time.monotonic() - started:.1f}", flush=True)
    return result


def _process_subject(subject: int, cache_root: Path, out_dir: Path,
                     simulations_root: Path, settings: dict, identity: dict,
                     cache_manifest: dict, backend=None) -> dict:
    path = out_dir / "subjects" / f"sub-{subject:02d}.json"
    checkpoint = load_checkpoint(path, identity, settings)
    if checkpoint["complete"]:
        return checkpoint
    backend = backend or load_backend(simulations_root)
    sessions = {}
    for session in settings["sessions"]:
        cache_path = session_path(cache_root, subject, session)
        if sha256_file(cache_path) != cache_manifest["sessions"][str(session)]["sha256"]:
            raise ValueError(f"Cache changed after preflight: {cache_path}")
        sessions[session] = load_session(cache_path, settings, identity["protocol_sha256"],
                                         cache_manifest["sessions"][str(session)]["channels"])
    sources = [sessions[s] for s in settings["source_sessions"]]
    target = sessions[settings["target_session"]]
    source_covs = [s["covs"] for s in sources]
    source_labels = [s["labels"] for s in sources]
    checkpoint["cache_manifest"] = cache_manifest
    checkpoint["target_session"] = settings["target_session"]
    checkpoint["source_sessions"] = settings["source_sessions"]
    if checkpoint["screening"] is None:
        seed = settings["seed"] + subject * 10_000
        hat, boot = backend.bootstrap(source_covs, target["covs"], np.random.default_rng(seed),
                                      settings["screening_boot"], settings["shrinkage"], True)
        selections = normalized_sets(hat, boot, settings, backend)
        checkpoint["screening"] = {
            **bootstrap_record(hat, boot, seed, settings["screening_boot"]),
            "target_trials": target["n_trials"],
            "methods": {m: selection_record(idx, settings, sources) for m, idx in selections.items()},
        }
        write_json(path, checkpoint)

    splitter = StratifiedShuffleSplit(n_splits=settings["n_splits"], test_size=settings["test_size"],
                                       random_state=settings["seed"] + subject)
    splits = list(splitter.split(target["covs"], target["labels"]))
    for split, (screen_idx, test_idx) in enumerate(splits):
        if split < len(checkpoint["downstream"]):
            saved = checkpoint["downstream"][split]
            if saved["screen_indices"] != screen_idx.tolist() or saved["test_indices"] != test_idx.tolist():
                raise ValueError(f"Checkpoint split allocation changed for subject {subject}, split {split}")
            continue
        seed = settings["seed"] + subject * 100_000 + split
        # Only unlabeled screening covariances enter the shared bootstrap.
        hat, boot = backend.bootstrap(source_covs, target["covs"][screen_idx],
                                      np.random.default_rng(seed), settings["downstream_boot"],
                                      settings["shrinkage"], True)
        selections = normalized_sets(hat, boot, settings, backend)
        row = {"split": split, "splitter_seed": settings["seed"] + subject,
               **bootstrap_record(hat, boot, seed, settings["downstream_boot"]),
               "screen_indices": screen_idx.tolist(), "test_indices": test_idx.tolist(),
               "screen_trial_ids": target["trial_ids"][screen_idx].tolist(),
               "test_trial_ids": target["trial_ids"][test_idx].tolist(), "methods": {}}
        fit_results = {}
        for method, indices in selections.items():
            # All source pools use ascending source-session order. Reuse is confined
            # to one split, where pool, labels, test covariances and labels are identical.
            key = tuple(indices.tolist())
            if key not in fit_results:
                train_covs, train_labels = backend.pool(indices, source_covs, source_labels)
                fit_results[key] = float(backend.train_eval_ts(
                    train_covs, train_labels, target["covs"][test_idx], target["labels"][test_idx]))
            accuracy = fit_results[key]
            if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError(f"Invalid accuracy: subject={subject}, split={split}, method={method}")
            row["methods"][method] = {**selection_record(indices, settings, sources),
                                        "balanced_accuracy": accuracy}
        accuracy = float(backend.train_eval_ts(target["covs"][screen_idx], target["labels"][screen_idx],
                                               target["covs"][test_idx], target["labels"][test_idx]))
        if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"Invalid target-only accuracy for subject {subject}, split {split}")
        row["methods"]["target_only"] = {"source_indices": [], "source_sessions": [],
                                            "set_size": 0, "training_trials": len(screen_idx),
                                            "balanced_accuracy": accuracy}
        checkpoint["downstream"].append(row)
        write_json(path, checkpoint)

    reference = backend.discrepancy_vector(source_covs, target["covs"], settings["shrinkage"])
    if not np.isfinite(reference).all():
        raise ValueError(f"Nonfinite empirical reference for subject {subject}")
    checkpoint["empirical_reference"] = {
        "discrepancies": np.asarray(reference).tolist(),
        "oracle_source_indices": np.flatnonzero(np.isclose(reference, np.min(reference), atol=1e-12, rtol=0)).tolist(),
        "source_fraction": settings["source_fraction"], "target_fraction": settings["target_fraction"],
        "outer_source_sample_sizes": [max(8, round(settings["source_fraction"] * s["n_trials"])) for s in sources],
        "outer_target_sample_size": max(8, round(settings["target_fraction"] * target["n_trials"])),
    }
    for replicate in range(len(checkpoint["coverage"]), settings["audit_outer"]):
        seed = settings["seed"] + subject * 1_000_000 + replicate
        metrics, selected = backend.evaluate_outer(
            reference, source_covs, target["covs"], np.random.default_rng(seed),
            settings["source_fraction"], settings["target_fraction"], settings["audit_boot"],
            settings["shrinkage"], settings["alpha"], settings["alpha_c"], settings["alpha_d"])
        if not all(np.isfinite(metrics[k]) for k in COVERAGE_METRICS):
            raise ValueError(f"Nonfinite audit metric for subject {subject}, replicate {replicate}")
        if int(metrics["coverage_joint"]) != int(bool(metrics["coverage_c"]) and bool(metrics["coverage_d"])):
            raise ValueError("Joint coverage must be the component/contrast intersection")
        checkpoint["coverage"].append({"replicate": replicate, "seed": seed,
                                       "bootstrap_repetitions": settings["audit_boot"],
                                       **plain(metrics), "source_indices": selected.tolist(),
                                       "source_sessions": [settings["source_sessions"][int(i)] for i in selected]})
        write_json(path, checkpoint)
    checkpoint["subject_means"] = subject_summary(checkpoint, backend)
    checkpoint["complete"] = True
    write_json(path, checkpoint)
    return checkpoint


def subject_summary(checkpoint: dict, backend=None) -> dict:
    rows = checkpoint["downstream"]
    methods = {method: {metric: float(np.mean([r["methods"][method][metric] for r in rows]))
                        for metric in ("balanced_accuracy", "set_size", "training_trials")}
               for method in METHODS}
    coverage = {metric: float(np.mean([r[metric] for r in checkpoint["coverage"]]))
                for metric in COVERAGE_METRICS}
    sets = [set(r["source_indices"]) for r in checkpoint["coverage"]]
    overlap = [len(a & b) / len(a | b) if a | b else 1.0
               for i, a in enumerate(sets) for b in sets[i + 1:]]
    coverage["mean_pairwise_jaccard"] = float(np.mean(overlap)) if overlap else 1.0
    coverage["source_inclusion_frequencies"] = {
        str(session): float(np.mean([i in s for s in sets]))
        for i, session in enumerate(checkpoint["source_sessions"])}
    return {**subject_info(checkpoint["subject"]), "n_splits": len(rows),
            "n_outer": len(checkpoint["coverage"]), "methods": methods, "coverage": coverage,
            "full_target_screening": checkpoint["screening"]["methods"]}


def paired_inference(differences: list[float], margins=(0.02, 0.01)) -> dict:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Paired inference requires finite per-subject differences")
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    df = len(values) - 1
    radius = float(stats.t.ppf(0.975, df) * se)
    lower = float(mean - stats.t.ppf(0.95, df) * se)
    ni = {}
    for margin in margins:
        p_value = float(stats.t.sf((mean + margin) / se, df)) if se else (0.0 if mean > -margin else 1.0 if mean < -margin else 0.5)
        ni[f"{margin:.2f}"] = {"margin": margin, "one_sided_p": p_value,
                               "one_sided_95_lower": lower, "noninferior": lower > -margin}
    return {"n_subjects": len(values), "differences": values.tolist(), "mean_difference": mean,
            "standard_error": se, "two_sided_95_ci": [mean - radius, mean + radius],
            "noninferiority": ni, "unit": "balanced_accuracy_fraction"}


def descriptive_summary(subjects: list[dict]) -> dict:
    return {"n_subjects": len(subjects),
            "methods": {method: {metric: float(np.mean([s["methods"][method][metric] for s in subjects]))
                                   for metric in ("balanced_accuracy", "set_size", "training_trials")}
                        for method in METHODS},
            "coverage": {metric: float(np.mean([s["coverage"][metric] for s in subjects]))
                         for metric in COVERAGE_METRICS + ("mean_pairwise_jaccard",)},
            "n_outer": sum(s["n_outer"] for s in subjects)}


def summarize(checkpoints: list[dict], settings: dict) -> dict:
    checkpoints = sorted(checkpoints, key=lambda x: x["subject"])
    if [c["subject"] for c in checkpoints] != settings["subjects"] or not all(c["complete"] for c in checkpoints):
        raise ValueError("Summary requires all prespecified subjects, with complete records")
    subjects = [subject_summary(c) for c in checkpoints]
    differences = lambda comparator: [s["methods"]["pair_ref"]["balanced_accuracy"]
                                     - s["methods"][comparator]["balanced_accuracy"] for s in subjects]
    return {"status": "complete", "analysis_role": "exploratory", "subject_means": subjects,
            "overall": descriptive_summary(subjects),
            "primary_refinement_minus_all_sources": paired_inference(differences("all_sources")),
            "refinement_minus_mcs": paired_inference(differences("mcs_stepdown"), margins=()),
            "group_descriptive_only": {group: descriptive_summary([s for s in subjects if s["group"] == group])
                                       for group in ("GR", "PAR")},
            "statistical_note": "Each subject contributes one mean across 30 target splits. The 95% paired t intervals are pointwise. Noninferiority uses one-sided alpha=0.05 and margins 0.02 (primary) and 0.01 (sensitivity); groups are descriptive only. Target-only uses labeled screening trials; source screens do not. Empirical-reference coverage is not population coverage."}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--simulations-root", type=Path, required=True,
                        help="Explicit frozen simulations snapshot; no implicit repository fallback")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4; every worker uses one numerical thread")
    protocol, protocol_hash = read_protocol(args.protocol)
    settings = protocol["analysis"]
    code = code_manifest(args.simulations_root)
    load_backend(args.simulations_root)
    caches = preflight(args.cache_root, settings, protocol_hash, protocol["preprocessing"]["eeg_channels"])
    import scipy
    import sklearn
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                   "platform": platform.platform(), "numeric_threads_per_worker": 1}
    run_identity = {"protocol_sha256": protocol_hash, "code_sha256": code["sha256"],
                    "cache_sha256": canonical_hash(caches), "environment": environment}
    run_manifest_path = args.out_dir / "manifest.json"
    if run_manifest_path.exists():
        previous = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if previous.get("identity") != run_identity:
            raise ValueError("Output directory belongs to a different protocol, code, cache, or environment")
    else:
        write_json(run_manifest_path, {"schema_version": SCHEMA_VERSION, "identity": run_identity,
                                      "created_at": datetime.now(timezone.utc).isoformat(),
                                      "protocol": protocol, "code": code, "cache": caches,
                                      "jobs": args.jobs, "preflight": "passed_all_108_sessions"})
    task_args = []
    for subject in settings["subjects"]:
        identity = {"subject": subject, "protocol_sha256": protocol_hash,
                    "code_sha256": code["sha256"], "cache_sha256": caches[str(subject)]["sha256"],
                    "environment": environment}
        # Validate all existing checkpoints before any worker is launched.
        load_checkpoint(args.out_dir / "subjects" / f"sub-{subject:02d}.json", identity, settings)
        task_args.append((subject, str(args.cache_root.resolve()), str(args.out_dir.resolve()),
                          str(args.simulations_root.resolve()), settings, identity, caches[str(subject)]))
    checkpoints = []
    if args.jobs == 1:
        for task in task_args:
            checkpoints.append(process_subject(*task))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {executor.submit(process_subject, *task): task[0] for task in task_args}
            for future in as_completed(futures):
                checkpoints.append(future.result())
    summary = summarize(checkpoints, settings)
    summary["identity"] = run_identity
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "analysis_results.json", {
        "schema_version": SCHEMA_VERSION, "identity": run_identity,
        "subjects": sorted(checkpoints, key=lambda x: x["subject"]), "summary": summary})
    print(f"Complete: {args.out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
