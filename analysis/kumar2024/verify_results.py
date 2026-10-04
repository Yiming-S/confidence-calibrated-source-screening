#!/usr/bin/env python3
"""Independently verify completed Kumar2024 results; never alters analysis outputs.

This program does not import analyze.py or reuse its summary/statistical functions.
It recomputes subject-level summaries from split records and repeats the fixed
subject 1/9/10/18, split-0 calculations with the supplied frozen backend.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from scipy.stats import t as student_t
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

METHODS = ("all_sources", "top1", "top3", "mcs_stepdown", "pair_ref", "target_only")
SOURCE_METHODS = METHODS[:-1]
AUDIT_SUBJECTS = (1, 9, 10, 18)
COVERAGE_FIELDS = ("coverage_c", "coverage_d", "coverage_joint", "oracle_retained",
                   "exact_recovery", "set_size", "retained_fraction", "worst_reference_excess")
METHOD_FIELDS = ("balanced_accuracy", "set_size", "training_trials")


class VerificationError(ValueError):
    pass


def require(condition, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest_object(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def equal(actual, expected, where: str, atol=1e-12, rtol=1e-10) -> None:
    """Compare numeric results with a declared tolerance, structure exactly."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{where}: keys differ")
        for key in expected:
            equal(actual[key], expected[key], f"{where}.{key}", atol, rtol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{where}: list length differs")
        for i, value in enumerate(expected):
            equal(actual[i], value, f"{where}[{i}]", atol, rtol)
    elif isinstance(expected, bool):
        require(isinstance(actual, bool) and actual == expected, f"{where}: boolean differs")
    elif isinstance(expected, (int, float, np.integer, np.floating)):
        require(isinstance(actual, (int, float, np.integer, np.floating)) and not isinstance(actual, bool),
                f"{where}: expected a number")
        require(math.isfinite(float(actual)) and math.isfinite(float(expected))
                and math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol),
                f"{where}: numeric mismatch ({actual!r} != {expected!r})")
    else:
        require(actual == expected, f"{where}: value differs")


def fixed_protocol(protocol: dict) -> dict:
    expected = {"seed": 20260714, "subjects": list(range(1, 19)), "sessions": list(range(1, 7)),
                "target_session": 6, "source_sessions": list(range(1, 6)), "n_channels": 22,
                "min_trials_per_class": 4, "screening_boot": 499, "downstream_boot": 199,
                "n_splits": 30, "test_size": .5, "shrinkage": .1, "alpha": .05,
                "alpha_c": .025, "alpha_d": .025, "audit_outer": 50, "audit_boot": 499,
                "source_fraction": 1., "target_fraction": .5}
    equal(protocol["analysis"], expected, "protocol.analysis", atol=0, rtol=0)
    equal(protocol["reporting"]["methods"], list(METHODS), "protocol.reporting.methods")
    require(len(protocol["preprocessing"]["eeg_channels"]) == 22, "Protocol must specify 22 EEG channels")
    return protocol["analysis"]


def verify_code(manifest: dict, simulations_root: Path) -> None:
    files = manifest["code"]["files"]
    required = {"analyze.py", "simulations/ci_gate_bnci004_riemann.py",
                "simulations/ci_gate_ma2020_riemann.py", "simulations/ci_gate_eeg_empirical_audit.py"}
    require(required.issubset(files), "Required analysis/backend code hashes are missing")
    equal(digest_object(files), manifest["code"]["sha256"], "code manifest digest")
    equal(manifest["code"]["sha256"], manifest["identity"]["code_sha256"], "run code digest")
    for name, expected in files.items():
        if name == "analyze.py":
            path = Path(__file__).with_name("analyze.py")
        else:
            require(name.startswith("simulations/") and Path(name).name == name.split("/", 1)[1],
                    f"Unexpected code-manifest path: {name}")
            path = simulations_root / Path(name).name
        equal(digest_file(path), expected, f"code file {name}")
    require("analyze.py" in files, "Analysis code hash is missing")
    snapshot_names = {f"simulations/{p.name}" for p in simulations_root.glob("*.py")}
    equal(sorted(snapshot_names), sorted(name for name in files if name.startswith("simulations/")),
          "frozen simulations file inventory")


def load_backend(simulations_root: Path):
    root = simulations_root.resolve(strict=True)
    sys.path.insert(0, str(root))
    bnci = importlib.import_module("ci_gate_bnci004_riemann")
    riemann = importlib.import_module("ci_gate_ma2020_riemann")
    audit = importlib.import_module("ci_gate_eeg_empirical_audit")
    for name, module in tuple(sys.modules.items()):
        if (name.startswith("ci_gate_") or name in {"path_config", "plot_style"}) and getattr(module, "__file__", None):
            require(Path(module.__file__).resolve().parent == root, f"Backend escaped frozen directory: {name}")
    return SimpleNamespace(bootstrap=riemann.riemann_estimates_and_bootstrap,
                           selected_sets=bnci.selected_sets, pool=bnci.pool,
                           train_eval_ts=riemann.train_eval_ts,
                           discrepancy_vector=audit.discrepancy_vector)


def load_caches(protocol: dict, protocol_hash: str, cache_root: Path,
                checkpoints: list[dict], manifest: dict) -> dict:
    caches, all_ids = {}, set()
    channels = protocol["preprocessing"]["eeg_channels"]
    minimum_per_class = protocol["analysis"]["min_trials_per_class"]
    for checkpoint in checkpoints:
        subject = checkpoint["subject"]
        stored_manifest = checkpoint["cache_manifest"]
        equal(stored_manifest, manifest["cache"][str(subject)], f"subject {subject} cache manifest")
        equal(digest_object(stored_manifest["sessions"]), stored_manifest["sha256"],
              f"subject {subject} cache manifest digest")
        equal(stored_manifest["sha256"], checkpoint["identity"]["cache_sha256"], f"subject {subject} cache identity")
        caches[subject] = {}
        for session in range(1, 7):
            relative = f"sub-{subject:02d}/session-{session:02d}_cov.npz"
            path = cache_root / relative
            metadata = stored_manifest["sessions"][str(session)]
            equal(metadata["relative_path"], relative, f"{relative} path")
            equal(digest_file(path), metadata["sha256"], f"{relative} digest")
            with np.load(path, allow_pickle=False) as data:
                covs, labels, ids = (data[k] for k in ("covs", "labels", "trial_ids"))
                equal(str(data["protocol_sha256"].item()), protocol_hash, f"{relative} protocol")
                equal(str(data["unit"].item()), "microvolt_squared", f"{relative} unit")
                equal(data["channels"].astype(str).tolist(), channels, f"{relative} channels")
            require(covs.dtype == np.float64 and covs.ndim == 3 and covs.shape[1:] == (22, 22),
                    f"{relative}: wrong covariance shape/dtype")
            require(np.isfinite(covs).all(), f"{relative}: nonfinite covariance")
            require(labels.shape == ids.shape == (len(covs),), f"{relative}: unaligned arrays")
            require(set(np.unique(labels).tolist()) == {0, 1}, f"{relative}: invalid labels")
            counts = [int(np.count_nonzero(labels == label)) for label in (0, 1)]
            require(min(counts) >= minimum_per_class,
                    f"{relative}: fewer than {minimum_per_class} trials per class: {counts}")
            require(ids.dtype.kind in "US", f"{relative}: trial IDs are not strings")
            ids = ids.astype(str)
            require(len(set(ids.tolist())) == len(ids) and not np.any(ids == ""), f"{relative}: invalid trial IDs")
            require(not all_ids.intersection(ids.tolist()), f"{relative}: repeated global trial ID")
            all_ids.update(ids.tolist())
            equal(metadata["trial_ids"], ids.tolist(), f"{relative} recorded trial IDs")
            equal(metadata["n_trials"], len(covs), f"{relative} trial count")
            equal(metadata["class_counts"], counts, f"{relative} class counts")
            equal(metadata["protocol_sha256"], protocol_hash, f"{relative} metadata protocol")
            equal(metadata["unit"], "microvolt_squared", f"{relative} metadata unit")
            equal(metadata["channels"], channels, f"{relative} metadata channels")
            caches[subject][session] = {"covs": covs, "labels": labels.astype(int), "trial_ids": ids}
    equal(digest_object(manifest["cache"]), manifest["identity"]["cache_sha256"], "overall cache identity")
    return caches


def check_indices(values, size: int, where: str, allow_empty=False) -> list[int]:
    require(isinstance(values, list) and all(type(v) is int for v in values), f"{where}: indices must be integers")
    require((allow_empty or bool(values)) and len(set(values)) == len(values), f"{where}: empty or repeated indices")
    require(all(0 <= v < size for v in values), f"{where}: out-of-range indices")
    return values


def verify_selection(record: dict, method: str, sessions: dict, screen_count: int, where: str) -> None:
    indices = check_indices(record["source_indices"], 5, where, allow_empty=method == "target_only")
    equal(record["source_sessions"], [i + 1 for i in indices], f"{where}.source_sessions")
    equal(record["set_size"], len(indices), f"{where}.set_size")
    if method == "target_only":
        require(indices == [], f"{where}: target-only must not use source sessions")
        expected_count = screen_count
    else:
        equal(indices, sorted(indices), f"{where}.canonical_source_order")
        expected_count = sum(len(sessions[i + 1]["labels"]) for i in indices)
    equal(record["training_trials"], expected_count, f"{where}.training_trials")
    if method == "all_sources":
        equal(indices, list(range(5)), f"{where}.all_sources")
    elif method in ("top1", "top3"):
        equal(len(indices), {"top1": 1, "top3": 3}[method], f"{where}.fixed_count")


def verify_screen_record(row: dict, where: str) -> None:
    discrepancies = np.asarray(row["discrepancy_estimates"], dtype=float)
    require(discrepancies.shape == (5,) and np.isfinite(discrepancies).all(), f"{where}: invalid discrepancy vector")
    matrix_hash = row["shared_bootstrap_matrix_sha256"]
    require(isinstance(matrix_hash, str) and len(matrix_hash) == 64
            and all(c in "0123456789abcdef" for c in matrix_hash), f"{where}: invalid bootstrap matrix hash")
    ranking = np.argsort(discrepancies, kind="stable").tolist()
    for method, count in (("top1", 1), ("top3", 3)):
        equal(row["methods"][method]["source_indices"], sorted(ranking[:count]), f"{where}.{method}.ranking")


def verify_records(checkpoints: list[dict], caches: dict, settings: dict) -> dict:
    equal([c["subject"] for c in checkpoints], list(range(1, 19)), "subject roster")
    splits_total = methods_total = outer_total = 0
    for checkpoint in checkpoints:
        subject = checkpoint["subject"]
        prefix = f"subject {subject}"
        require(checkpoint["complete"] is True, f"{prefix}: checkpoint is incomplete")
        equal(checkpoint["raw_subject"], subject if subject <= 9 else subject + 1, f"{prefix}.raw_subject")
        equal(checkpoint["group"], "GR" if subject <= 9 else "PAR", f"{prefix}.group")
        equal(checkpoint["target_session"], 6, f"{prefix}.target_session")
        equal(checkpoint["source_sessions"], [1, 2, 3, 4, 5], f"{prefix}.source_sessions")
        sessions = caches[subject]
        target = sessions[6]
        n_target = len(target["labels"])
        full = checkpoint["screening"]
        equal(full["seed"], settings["seed"] + subject * 10000, f"{prefix}.screening.seed")
        equal(full["bootstrap_repetitions"], 499, f"{prefix}.screening.bootstrap_repetitions")
        equal(full["target_trials"], n_target, f"{prefix}.screening.target_trials")
        equal(sorted(full["methods"]), sorted(SOURCE_METHODS), f"{prefix}.screening.methods")
        verify_screen_record(full, f"{prefix}.screening")
        for method in SOURCE_METHODS:
            verify_selection(full["methods"][method], method, sessions, n_target, f"{prefix}.screening.{method}")
        splitter = StratifiedShuffleSplit(n_splits=30, test_size=.5, random_state=settings["seed"] + subject)
        expected_splits = list(splitter.split(np.zeros(n_target), target["labels"]))
        require(len(checkpoint["downstream"]) == 30, f"{prefix}: missing or extra splits")
        for split, row in enumerate(checkpoint["downstream"]):
            where = f"{prefix}.split{split}"
            equal(row["split"], split, f"{where}.split")
            equal(row["splitter_seed"], settings["seed"] + subject, f"{where}.splitter_seed")
            equal(row["seed"], settings["seed"] + subject * 100000 + split, f"{where}.seed")
            equal(row["bootstrap_repetitions"], 199, f"{where}.bootstrap_repetitions")
            screen = check_indices(row["screen_indices"], n_target, f"{where}.screen_indices")
            test = check_indices(row["test_indices"], n_target, f"{where}.test_indices")
            require(not set(screen).intersection(test), f"{where}: screening/test indices overlap")
            require(set(screen).union(test) == set(range(n_target)), f"{where}: incomplete target partition")
            equal(screen, expected_splits[split][0].tolist(), f"{where}.deterministic_screen_indices")
            equal(test, expected_splits[split][1].tolist(), f"{where}.deterministic_test_indices")
            equal(row["screen_trial_ids"], target["trial_ids"][screen].tolist(), f"{where}.screen_trial_ids")
            equal(row["test_trial_ids"], target["trial_ids"][test].tolist(), f"{where}.test_trial_ids")
            require(not set(row["screen_trial_ids"]).intersection(row["test_trial_ids"]), f"{where}: trial ID leakage")
            require(set(row["screen_trial_ids"]).union(row["test_trial_ids"]) == set(target["trial_ids"].tolist()),
                    f"{where}: target trial IDs are incomplete")
            equal(sorted(row["methods"]), sorted(METHODS), f"{where}.methods")
            verify_screen_record(row, where)
            for method in METHODS:
                record = row["methods"][method]
                verify_selection(record, method, sessions, len(screen), f"{where}.{method}")
                accuracy = record["balanced_accuracy"]
                require(isinstance(accuracy, (float, int)) and math.isfinite(accuracy) and 0 <= accuracy <= 1,
                        f"{where}.{method}: invalid balanced accuracy")
                methods_total += 1
            splits_total += 1
        reference = checkpoint["empirical_reference"]
        discrepancies = np.asarray(reference["discrepancies"], dtype=float)
        require(discrepancies.shape == (5,) and np.isfinite(discrepancies).all(), f"{prefix}: invalid reference")
        oracle = np.flatnonzero(np.isclose(discrepancies, discrepancies.min(), atol=1e-12, rtol=0)).tolist()
        equal(reference["oracle_source_indices"], oracle, f"{prefix}.reference.oracle")
        equal(reference["source_fraction"], 1., f"{prefix}.reference.source_fraction")
        equal(reference["target_fraction"], .5, f"{prefix}.reference.target_fraction")
        equal(reference["outer_source_sample_sizes"], [len(sessions[s]["labels"]) for s in range(1, 6)],
              f"{prefix}.reference.source_sizes")
        equal(reference["outer_target_sample_size"], max(8, round(n_target * .5)), f"{prefix}.reference.target_size")
        require(len(checkpoint["coverage"]) == 50, f"{prefix}: missing or extra outer samples")
        for replicate, row in enumerate(checkpoint["coverage"]):
            where = f"{prefix}.outer{replicate}"
            equal(row["replicate"], replicate, f"{where}.replicate")
            equal(row["seed"], settings["seed"] + subject * 1000000 + replicate, f"{where}.seed")
            equal(row["bootstrap_repetitions"], 499, f"{where}.bootstrap_repetitions")
            indices = check_indices(row["source_indices"], 5, where)
            equal(row["source_sessions"], [i + 1 for i in indices], f"{where}.source_sessions")
            equal(row["set_size"], len(indices), f"{where}.set_size")
            equal(row["retained_fraction"], len(indices) / 5, f"{where}.retained_fraction")
            for field in ("coverage_c", "coverage_d", "coverage_joint", "oracle_retained", "exact_recovery"):
                require(type(row[field]) is int and row[field] in (0, 1), f"{where}.{field}: expected binary event")
            equal(row["coverage_joint"], int(row["coverage_c"] == 1 and row["coverage_d"] == 1), f"{where}.joint_intersection")
            equal(row["oracle_retained"], int(set(oracle).issubset(indices)), f"{where}.oracle_retained")
            equal(row["exact_recovery"], int(set(oracle) == set(indices)), f"{where}.exact_recovery")
            equal(row["worst_reference_excess"], float(np.max(discrepancies[indices] - discrepancies.min())),
                  f"{where}.worst_reference_excess")
            outer_total += 1
    equal([splits_total, methods_total, outer_total], [540, 3240, 900], "complete record counts")
    return {"subjects": 18, "splits": splits_total, "method_rows": methods_total, "outer_samples": outer_total}


def independent_subject_mean(checkpoint: dict) -> dict:
    subject = checkpoint["subject"]
    rows, outer = checkpoint["downstream"], checkpoint["coverage"]
    methods = {method: {field: math.fsum(float(row["methods"][method][field]) for row in rows) / len(rows)
                        for field in METHOD_FIELDS} for method in METHODS}
    coverage = {field: math.fsum(float(row[field]) for row in outer) / len(outer) for field in COVERAGE_FIELDS}
    sets = [set(row["source_indices"]) for row in outer]
    pairs = list(itertools.combinations(sets, 2))
    coverage["mean_pairwise_jaccard"] = math.fsum(len(a & b) / len(a | b) if a | b else 1. for a, b in pairs) / len(pairs)
    coverage["source_inclusion_frequencies"] = {str(i + 1): sum(i in chosen for chosen in sets) / len(sets) for i in range(5)}
    return {"subject": subject, "raw_subject": subject if subject <= 9 else subject + 1,
            "group": "GR" if subject <= 9 else "PAR", "n_splits": len(rows), "n_outer": len(outer),
            "methods": methods, "coverage": coverage, "full_target_screening": checkpoint["screening"]["methods"]}


def independent_paired(differences, include_ni: bool) -> dict:
    n = len(differences)
    mean = math.fsum(differences) / n
    variance = math.fsum((difference - mean) ** 2 for difference in differences) / (n - 1)
    se = math.sqrt(variance / n)
    radius = float(student_t.isf(.025, n - 1)) * se
    lower = mean - float(student_t.isf(.05, n - 1)) * se
    ni = {}
    for margin in ((.02, .01) if include_ni else ()):
        p = float(student_t.sf((mean + margin) / se, n - 1)) if se else (0. if mean > -margin else 1. if mean < -margin else .5)
        ni[f"{margin:.2f}"] = {"margin": margin, "one_sided_p": p,
                               "one_sided_95_lower": lower, "noninferior": lower > -margin}
    return {"n_subjects": n, "differences": differences, "mean_difference": mean, "standard_error": se,
            "two_sided_95_ci": [mean - radius, mean + radius], "noninferiority": ni,
            "unit": "balanced_accuracy_fraction"}


def independent_group(subjects: list[dict]) -> dict:
    n = len(subjects)
    return {"n_subjects": n,
            "methods": {method: {field: math.fsum(s["methods"][method][field] for s in subjects) / n
                                   for field in METHOD_FIELDS} for method in METHODS},
            "coverage": {field: math.fsum(s["coverage"][field] for s in subjects) / n
                         for field in COVERAGE_FIELDS + ("mean_pairwise_jaccard",)},
            "n_outer": sum(s["n_outer"] for s in subjects)}


def verify_summaries(checkpoints: list[dict], summary: dict) -> dict:
    recomputed = [independent_subject_mean(c) for c in checkpoints]
    for checkpoint, subject in zip(checkpoints, recomputed):
        equal(checkpoint["subject_means"], subject, f"subject {subject['subject']} stored means")
    equal(summary["subject_means"], recomputed, "summary.subject_means")
    overall = independent_group(recomputed)
    equal(summary["overall"], overall, "summary.overall")
    inferential = {}
    for comparator, key, ni in (("all_sources", "primary_refinement_minus_all_sources", True),
                                ("mcs_stepdown", "refinement_minus_mcs", False)):
        differences = [s["methods"]["pair_ref"]["balanced_accuracy"] - s["methods"][comparator]["balanced_accuracy"] for s in recomputed]
        inferential[key] = independent_paired(differences, ni)
        equal(summary[key], inferential[key], f"summary.{key}")
    grouped = {group: independent_group([s for s in recomputed if s["group"] == group]) for group in ("GR", "PAR")}
    equal(summary["group_descriptive_only"], grouped, "summary.group_descriptive_only")
    equal(summary["status"], "complete", "summary.status")
    equal(summary["analysis_role"], "exploratory", "summary.analysis_role")
    return {"subject_means": recomputed, "overall": overall, **inferential, "group_descriptive_only": grouped}


def audit_fixed_splits(checkpoints: list[dict], caches: dict, settings: dict, backend) -> list[dict]:
    by_subject = {c["subject"]: c for c in checkpoints}
    reports = []
    for subject in AUDIT_SUBJECTS:
        checkpoint, sessions = by_subject[subject], caches[subject]
        row = checkpoint["downstream"][0]
        source_covs = [sessions[s]["covs"] for s in range(1, 6)]
        source_labels = [sessions[s]["labels"] for s in range(1, 6)]
        target = sessions[6]
        screen, test = np.asarray(row["screen_indices"]), np.asarray(row["test_indices"])
        seed = settings["seed"] + subject * 100000
        hat, boot = backend.bootstrap(source_covs, target["covs"][screen], np.random.default_rng(seed), 199, .1, True)
        require(np.isfinite(hat).all() and np.isfinite(boot).all(), f"subject {subject} audit: nonfinite draws")
        equal(np.asarray(hat).tolist(), row["discrepancy_estimates"], f"subject {subject} audit discrepancy")
        matrix_hash = hashlib.sha256(np.asarray(boot, dtype="<f8").tobytes()).hexdigest()
        equal(matrix_hash, row["shared_bootstrap_matrix_sha256"], f"subject {subject} audit bootstrap matrix hash")
        selected = backend.selected_sets(hat, boot, .05, .025, .025)
        selected["all_sources"] = np.arange(5)
        accuracies = {}
        for method in METHODS:
            if method == "target_only":
                x, y = target["covs"][screen], target["labels"][screen]
                indices = []
            else:
                indices = sorted(np.asarray(selected[method], dtype=int).tolist())
                x, y = backend.pool(np.asarray(indices, dtype=int), source_covs, source_labels)
            equal(indices, row["methods"][method]["source_indices"], f"subject {subject} audit {method} source set")
            # Deliberately refit each method independently: no within-split cache.
            accuracy = float(backend.train_eval_ts(x, y, target["covs"][test], target["labels"][test]))
            equal(accuracy, row["methods"][method]["balanced_accuracy"], f"subject {subject} audit {method} accuracy")
            accuracies[method] = accuracy
        reports.append({"subject": subject, "split": 0, "seed": seed, "bootstrap_repetitions": 199,
                        "shared_bootstrap_matrix_sha256": matrix_hash, "balanced_accuracy": accuracies,
                        "status": "passed"})
        print(f"Verification replay complete: subject {subject:02d}, split 0", flush=True)
    return reports


def verify(protocol_path: Path, results_root: Path, cache_root: Path, simulations_root: Path) -> dict:
    protocol = read_json(protocol_path)
    settings = fixed_protocol(protocol)
    protocol_hash = digest_file(protocol_path)
    manifest = read_json(results_root / "manifest.json")
    equal(manifest["protocol"], protocol, "run protocol")
    equal(manifest["identity"]["protocol_sha256"], protocol_hash, "run protocol digest")
    verify_code(manifest, simulations_root)
    expected_paths = [results_root / "subjects" / f"sub-{subject:02d}.json" for subject in range(1, 19)]
    equal(sorted(p.name for p in (results_root / "subjects").glob("sub-*.json")),
          sorted(p.name for p in expected_paths), "subject result files")
    checkpoints = [read_json(path) for path in expected_paths]
    for subject, checkpoint in enumerate(checkpoints, 1):
        equal(checkpoint["schema_version"], 1, f"subject {subject} schema")
        equal(checkpoint["subject"], subject, f"subject file {subject} identity")
        equal(checkpoint["identity"]["subject"], subject, f"subject {subject} identity.subject")
        for field in ("protocol_sha256", "code_sha256", "environment"):
            equal(checkpoint["identity"][field], manifest["identity"][field], f"subject {subject} identity.{field}")
    caches = load_caches(protocol, protocol_hash, cache_root, checkpoints, manifest)
    counts = verify_records(checkpoints, caches, settings)
    summary = read_json(results_root / "summary.json")
    equal(summary["identity"], manifest["identity"], "summary identity")
    recomputed = verify_summaries(checkpoints, summary)
    backend = load_backend(simulations_root)
    with threadpool_limits(limits=1):
        for checkpoint in checkpoints:
            subject = checkpoint["subject"]
            reference = backend.discrepancy_vector([caches[subject][s]["covs"] for s in range(1, 6)],
                                                  caches[subject][6]["covs"], .1)
            equal(np.asarray(reference).tolist(), checkpoint["empirical_reference"]["discrepancies"],
                  f"subject {subject} independently recomputed empirical reference")
        audits = audit_fixed_splits(checkpoints, caches, settings, backend)
    return {"status": "passed", "verified_at": datetime.now(timezone.utc).isoformat(),
            "verifier_sha256": digest_file(Path(__file__).resolve()), "identity": manifest["identity"],
            "counts": counts, "independently_recomputed": recomputed, "fixed_split_replays": audits,
            "numeric_tolerance": {"absolute": 1e-12, "relative": 1e-10},
            "bootstrap_matrix_check": "Exact SHA256 equality of little-endian float64 arrays",
            "scope": "All 18 subjects and all stored records checked. Subject means and inference recomputed without importing analyze.py. All empirical-reference discrepancy vectors recomputed. Bootstrap screening and all six method accuracies independently rerun for prespecified subjects 1,9,10,18 at split 0. The 900 outer bootstrap runs are not rerun; their event/set consistency and summaries are checked."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--simulations-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = verify(args.protocol, args.results_root, args.cache_root, args.simulations_root)
    except Exception as error:
        report = {"status": "failed", "verified_at": datetime.now(timezone.utc).isoformat(),
                  "verifier_sha256": digest_file(Path(__file__).resolve()),
                  "error_type": type(error).__name__, "error": str(error)}
        write_report(args.results_root / "verification.json", report)
        print(f"Verification failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    write_report(args.results_root / "verification.json", report)
    print(f"Verification passed: {args.results_root / 'verification.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
