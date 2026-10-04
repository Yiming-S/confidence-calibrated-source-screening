#!/usr/bin/env python3
"""Fixed post-outcome Kumar supplementary analyses; isolated outputs only."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

import analyze as primary


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    primary.write_json(Path(path), value)


def backend(simulations, supplemental_backend):
    roots = [Path(simulations).resolve(), Path(supplemental_backend).resolve()]
    sys.path[:0] = [str(root) for root in roots]
    modules = {name: importlib.import_module(name) for name in (
        "ci_gate_ma2020_riemann", "ci_gate_realdata_case_study", "ci_gate_bnci004_riemann",
        "ci_gate_bspc_bootstrap_budget", "ci_gate_block_bootstrap_sensitivity")}
    for name, module in modules.items():
        expected = roots[1] if name in ("ci_gate_bspc_bootstrap_budget", "ci_gate_block_bootstrap_sensitivity") else roots[0]
        if Path(module.__file__).resolve().parent != expected:
            raise ValueError(f"Unexpected backend origin: {name}: {module.__file__}")
    return modules


def subject_data(subject, cache_root, protocol, protocol_hash):
    sessions = [primary.load_session(primary.session_path(Path(cache_root), subject, session), protocol["analysis"],
                                    protocol_hash, protocol["preprocessing"]["eeg_channels"]) for session in range(1, 7)]
    return [s["covs"] for s in sessions[:-1]], [s["labels"] for s in sessions[:-1]], sessions[-1]


def splits(target, subject, count):
    iterator = StratifiedShuffleSplit(n_splits=30, test_size=.5, random_state=20260714 + subject)
    return list(itertools.islice(iterator.split(target["covs"], target["labels"]), count))


def split_record(target, screen, test):
    return {"screen_indices": screen.tolist(), "test_indices": test.tolist(),
            "screen_trial_ids": target["trial_ids"][screen].tolist(), "test_trial_ids": target["trial_ids"][test].tolist()}


def archive(path, estimate, draws, plan_hash):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.npz")
    np.savez_compressed(temporary, estimate=estimate, draws=draws, plan_sha256=plan_hash)
    os.replace(temporary, path)
    return {"archive": str(path.name), "archive_sha256": primary.sha256_file(path),
            "bootstrap_matrix_sha256": hashlib.sha256(np.asarray(draws, dtype="<f8").tobytes()).hexdigest()}


def calibrated(estimate, draws, modules):
    system = modules["ci_gate_realdata_case_study"].build_system(estimate, draws, .025, .025, .05)
    chosen = modules["ci_gate_bnci004_riemann"].selected_sets(estimate, draws, .05, .025, .025)
    chosen["pair"] = modules["ci_gate_realdata_case_study"].pair_gate(len(estimate), system["jj"], system["lower_d"])
    system["sets"] = {method: np.asarray(indices).tolist() for method, indices in chosen.items()}
    return primary.plain(system)


def certificate(sources, target, settings, modules, out, plan_hash):
    seed, budget = settings["seed"], settings["bootstrap"]
    estimate, draws = modules["ci_gate_ma2020_riemann"].riemann_estimates_and_bootstrap(
        sources, target["covs"], np.random.default_rng(seed), budget, .1, True)
    system = calibrated(estimate, draws, modules)
    rows = []
    threshold_index = int(np.argmin(system["upper_c"]))
    for index in range(5):
        contrasts = [{"competitor_session": competitor + 1, "lower_bound": lower}
                     for j, competitor, lower in zip(system["jj"], system["ll"], system["lower_d"]) if j == index]
        rows.append({"source_session": index + 1, "discrepancy": float(estimate[index]),
                     "lower_component": system["lower_c"][index], "upper_component": system["upper_c"][index],
                     "rectangle_threshold": system["upper_c"][threshold_index],
                     "rectangle_competitor_session": threshold_index + 1,
                     "rectangle_exclusion_margin": system["lower_c"][index] - system["upper_c"][threshold_index],
                     "rectangle_excluded": system["lower_c"][index] > system["upper_c"][threshold_index],
                     "direct_contrast_excluded": any(r["lower_bound"] > 0 for r in contrasts),
                     "rectangle_retained": index in system["sets"]["rect"],
                     "refinement_retained": index in system["sets"]["pair_ref"],
                     "contrasts": contrasts, "positive_contrast_witnesses": [r for r in contrasts if r["lower_bound"] > 0]})
    return {"dataset": "kumar2024", "subject": 1, "target_session": 6, "source_sessions": [1, 2, 3, 4, 5],
            "seed": seed, "bootstrap": budget, "target_trials": len(target["covs"]), "estimate": estimate.tolist(),
            "system": system, "sources": rows, **archive(out / "draws/certificate.npz", estimate, draws, plan_hash)}


def budget_case(subject, split, seed, sources, target, modules, out, plan_hash):
    screen, test = splits(target, subject, 2)[split]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4, subject, split]))
    helper = modules["ci_gate_bspc_bootstrap_budget"]
    estimate, draws = helper.discrepancy_bootstrap(sources, target["covs"][screen], rng, 999, .1)
    systems = {b: helper.calibrate(estimate, draws[:b], .025, .025) for b in (99, 199, 499, 999)}
    reference, rows, membership = systems[999], [], []
    reference_set = set(reference["selected"].tolist())
    for budget, system in systems.items():
        chosen = set(system["selected"].tolist())
        row = {"dataset": "kumar2024", "subject": subject, "split": split, "bootstrap_seed": seed, "budget": budget,
               "n_sources": 5, "set_size": len(chosen), "members": [i + 1 for i in sorted(chosen)],
               "jaccard_vs_999": helper.jaccard(chosen, reference_set), "same_members_vs_999": int(chosen == reference_set),
               "size_change_vs_999": len(chosen) - len(reference_set), "absolute_size_change_vs_999": abs(len(chosen) - len(reference_set)),
               "members_added_vs_999": len(chosen - reference_set), "members_removed_vs_999": len(reference_set - chosen),
               "q_component": system["q_c"], "q_contrast": system["q_d"],
               "q_component_change_vs_999": system["q_c"] - reference["q_c"],
               "q_contrast_change_vs_999": system["q_d"] - reference["q_d"],
               "q_component_abs_change_vs_999": abs(system["q_c"] - reference["q_c"]),
               "q_contrast_abs_change_vs_999": abs(system["q_d"] - reference["q_d"])}
        for label, key in (("component", "raw_se_c"), ("contrast", "raw_se_d")):
            se = system[key]
            row.update({f"raw_{label}_se_min": float(se.min()), f"raw_{label}_se_max": float(se.max()),
                        f"raw_{label}_se_nonfinite_count": int(np.sum(~np.isfinite(se))),
                        f"raw_{label}_se_floor_count": int(np.sum(se < 1e-12))})
        rows.append(row)
        for i in range(5):
            membership.append({"subject": subject, "split": split, "bootstrap_seed": seed, "budget": budget,
                               "source_session": i + 1, "selected": int(i in chosen), "selected_at_999": int(i in reference_set),
                               "discrepancy": float(estimate[i]), "se_component": float(system["se_c"][i]),
                               "raw_se_component": float(system["raw_se_c"][i]), "lower_component": float(system["lower_c"][i]),
                               "upper_component": float(system["upper_c"][i])})
    name = f"budget_s{subject:02d}_split{split}_seed{seed}.npz"
    return {"subject": subject, "split": split, "bootstrap_seed": seed, **split_record(target, screen, test),
            "rows": rows, "membership": membership, **archive(out / "draws" / name, estimate, draws, plan_hash)}


def block_subject(subject, sources, labels, target, modules, out, plan_hash):
    helper, select, train = modules["ci_gate_block_bootstrap_sensitivity"], modules["ci_gate_bnci004_riemann"], modules["ci_gate_ma2020_riemann"]
    full, downstream, split_rows, draw_rows = [], [], [], []
    sets = {}
    for length in (1, 4, 8):
        seed = 20260715 + subject * 100000 + length * 1000
        estimate, draws = helper.riemann_estimates_and_block_bootstrap(sources, target["covs"], np.random.default_rng(seed), 499, .1, length)
        selected = select.selected_sets(estimate, draws, .05, .025, .025)
        sets[length] = selected
        draw_rows.append({"analysis": "full", "block_length": length, "seed": seed, "bootstrap": 499,
                          "sets": primary.plain(selected), **archive(out / "draws" / f"block_s{subject:02d}_full_l{length}.npz", estimate, draws, plan_hash)})
    for length, selected in sets.items():
        ref, mcs = selected["pair_ref"], selected["mcs_stepdown"]
        full.append({"dataset": "kumar2024", "subject": subject, "block_length": length, "n_sources": 5,
                     "target_trials": len(target["covs"]), "pair_ref_size": len(ref), "pair_ref_fraction": len(ref) / 5,
                     "mcs_size": len(mcs), "pair_ref_sessions": (ref + 1).tolist(), "mcs_sessions": (mcs + 1).tolist(),
                     "jaccard_vs_iid": helper.jaccard(ref, sets[1]["pair_ref"]),
                     "exact_match_iid": int(np.array_equal(ref, sets[1]["pair_ref"]))})
    for split, (screen, test) in enumerate(splits(target, subject, 10)):
        ordered = np.sort(screen)
        split_rows.append({"subject": subject, "split": split, **split_record(target, screen, test),
                           "ordered_screen_indices": ordered.tolist(), "ordered_screen_trial_ids": target["trial_ids"][ordered].tolist()})
        all_covs, all_labels = select.pool(np.arange(5), sources, labels)
        accuracy = train.train_eval_ts(all_covs, all_labels, target["covs"][test], target["labels"][test])
        downstream.append({"dataset": "kumar2024", "subject": subject, "split": split, "block_length": 0,
                           "method": "all_sources", "source_sessions": [1, 2, 3, 4, 5], "set_size": 5, "balanced_accuracy": accuracy})
        for length in (1, 4):
            seed = 20260715 + subject * 1000000 + split * 10000 + length * 100
            estimate, draws = helper.riemann_estimates_and_block_bootstrap(sources, target["covs"][ordered], np.random.default_rng(seed), 99, .1, length)
            selected = select.selected_sets(estimate, draws, .05, .025, .025)
            draw_rows.append({"analysis": "downstream", "split": split, "block_length": length, "seed": seed, "bootstrap": 99,
                              "sets": primary.plain(selected), **archive(out / "draws" / f"block_s{subject:02d}_split{split}_l{length}.npz", estimate, draws, plan_hash)})
            for method in ("pair_ref", "mcs_stepdown"):
                indices = np.asarray(sorted(selected[method].tolist()), dtype=int)
                x, y = select.pool(indices, sources, labels)
                accuracy = train.train_eval_ts(x, y, target["covs"][test], target["labels"][test])
                downstream.append({"dataset": "kumar2024", "subject": subject, "split": split, "block_length": length,
                                   "method": method, "source_sessions": (indices + 1).tolist(), "set_size": len(indices), "balanced_accuracy": accuracy})
    return {"subject": subject, "screening": full, "downstream": downstream, "splits": split_rows, "draws": draw_rows}


def run_task(task):
    kind, subject, split, seed, configuration = task
    out = Path(configuration["out"])
    name = "certificate" if kind == "certificate" else f"budget_s{subject:02d}_split{split}_seed{seed}" if kind == "budget" else f"block_s{subject:02d}"
    path = out / "cases" / f"{name}.json"
    if path.exists():
        result = read(path)
        if result["plan_sha256"] != configuration["plan_hash"]:
            raise ValueError(f"Cannot resume changed plan: {path}")
        return result
    started = time.monotonic()
    modules = backend(configuration["simulations"], configuration["supplemental_backend"])
    sources, labels, target = subject_data(subject, configuration["cache"], configuration["protocol"], configuration["protocol_hash"])
    with threadpool_limits(limits=1):
        if kind == "certificate":
            result = certificate(sources, target, configuration["plan"]["certificate"], modules, out, configuration["plan_hash"])
        elif kind == "budget":
            result = budget_case(subject, split, seed, sources, target, modules, out, configuration["plan_hash"])
        else:
            result = block_subject(subject, sources, labels, target, modules, out, configuration["plan_hash"])
    result.update({"kind": kind, "plan_sha256": configuration["plan_hash"], "elapsed_seconds": time.monotonic() - started})
    save(path, result)
    print(f"Completed {name}: {result['elapsed_seconds']:.1f} seconds", flush=True)
    return result


def budget_summary(cases):
    rows = [row for case in cases for row in case["rows"]]
    frame = pd.DataFrame(rows)
    metrics = ["set_size", "jaccard_vs_999", "same_members_vs_999", "size_change_vs_999", "absolute_size_change_vs_999",
               "members_added_vs_999", "members_removed_vs_999", "q_component", "q_contrast",
               "q_component_change_vs_999", "q_contrast_change_vs_999", "q_component_abs_change_vs_999", "q_contrast_abs_change_vs_999"]
    subject = frame.groupby(["subject", "budget"], as_index=False)[metrics].mean()
    summary = subject.groupby("budget", as_index=False)[metrics].mean()
    diagnostics, seed_cases = [], []
    for budget, group in frame.groupby("budget"):
        record = {"budget": int(budget), "n_subjects": 3, "n_seed_sample_cases": len(group),
                  "changed_set_cases_vs_999": int((group.same_members_vs_999 == 0).sum()),
                  "minimum_jaccard_vs_999": float(group.jaccard_vs_999.min()),
                  "maximum_absolute_size_change_vs_999": int(group.absolute_size_change_vs_999.max()),
                  "n_component_coordinates": int(group.n_sources.sum()), "n_contrast_coordinates": int((group.n_sources * (group.n_sources - 1)).sum())}
        for label in ("component", "contrast"):
            for suffix, operation in (("min", "min"), ("max", "max"), ("nonfinite_count", "sum"), ("floor_count", "sum")):
                key = f"raw_{label}_se_{suffix}"
                record[key] = primary.plain(getattr(group[key], operation)())
        diagnostics.append(record)
    for (subject_id, split, budget), group in frame.groupby(["subject", "split", "budget"]):
        pairs = list(itertools.combinations([set(v) for v in group.members], 2))
        seed_cases.append({"subject": int(subject_id), "split": int(split), "budget": int(budget),
                           "pairwise_seed_jaccard": float(np.mean([len(a & b) / len(a | b) for a, b in pairs])),
                           "pairwise_seed_exact": float(np.mean([a == b for a, b in pairs])),
                           "set_size_seed_sd": float(group.set_size.std(ddof=1)), "set_size_seed_range": int(group.set_size.max() - group.set_size.min()),
                           "q_component_seed_sd": float(group.q_component.std(ddof=1)), "q_contrast_seed_sd": float(group.q_contrast.std(ddof=1))})
    seeds = pd.DataFrame(seed_cases)
    seed_metrics = [c for c in seeds if c not in ("subject", "split", "budget")]
    seed_subject = seeds.groupby(["subject", "budget"], as_index=False)[seed_metrics].mean()
    seed_summary = seed_subject.groupby("budget", as_index=False)[seed_metrics].mean()
    return {"rows": summary.to_dict("records"), "subject_summary": subject.to_dict("records"), "diagnostics": diagnostics,
            "seed_variability_cases": seed_cases, "seed_variability_subjects": seed_subject.to_dict("records"),
            "seed_variability_summary": seed_summary.to_dict("records")}


def aggregate(results, out, modules):
    certificate_result = next(r for r in results if r["kind"] == "certificate")
    budget_cases = sorted((r for r in results if r["kind"] == "budget"), key=lambda r: (r["subject"], r["split"], r["bootstrap_seed"]))
    blocks = sorted((r for r in results if r["kind"] == "block"), key=lambda r: r["subject"])
    if len(budget_cases) != 18 or [r["subject"] for r in blocks] != list(range(1, 19)):
        raise ValueError("Missing fixed supplementary cases")
    save(out / "certificate.json", certificate_result)
    save(out / "budget_results.json", {"cases": budget_cases, "rows": [r for c in budget_cases for r in c["rows"]],
                                       "membership": [r for c in budget_cases for r in c["membership"]]})
    save(out / "budget_summary.json", budget_summary(budget_cases))
    block_result = {"subjects": blocks, "screening": [r for s in blocks for r in s["screening"]],
                    "downstream": [r for s in blocks for r in s["downstream"]]}
    save(out / "block_results.json", block_result)
    helper = modules["ci_gate_block_bootstrap_sensitivity"]
    screening = helper.summarize_screening(pd.DataFrame(block_result["screening"]))
    downstream = helper.summarize_downstream(pd.DataFrame(block_result["downstream"]))
    subject = pd.DataFrame(block_result["downstream"]).groupby(["subject", "block_length", "method"], as_index=False)[["set_size", "balanced_accuracy"]].mean()
    save(out / "block_summary.json", {"screening": screening.to_dict("records"), "downstream": downstream.to_dict("records"),
                                      "subject_downstream": subject.to_dict("records"),
                                      "all_source_accuracy": float(subject[subject.method == "all_sources"].balanced_accuracy.mean())})


def prepare(args):
    plan, protocol = read(args.plan), read(args.protocol)
    protocol_hash = primary.sha256_file(args.protocol)
    if protocol_hash != plan["protocol_sha256"]:
        raise ValueError("Supplemental plan requires the exact final primary protocol")
    verified = read(args.primary_results_root / "verification.json")
    if verified["status"] != "passed" or verified["counts"] != {"subjects": 18, "splits": 540, "method_rows": 3240, "outer_samples": 900}:
        raise ValueError("Complete verified primary results are required")
    caches = primary.preflight(args.cache_root, protocol["analysis"], protocol_hash, protocol["preprocessing"]["eeg_channels"])
    if primary.canonical_hash(caches) != verified["identity"]["cache_sha256"]:
        raise ValueError("Supplementary cache differs from the verified primary cache")
    code = {"supplemental.py": primary.sha256_file(Path(__file__)), "analyze.py": primary.sha256_file(Path(__file__).with_name("analyze.py"))}
    for prefix, root in (("simulations", args.simulations_root), ("supplemental_backend", args.supplemental_backend_root)):
        code.update({f"{prefix}/{p.name}": primary.sha256_file(p) for p in sorted(root.glob("*.py"))})
    original_code = read(args.primary_results_root / "manifest.json")["code"]["files"]
    if any(code.get(name) != digest for name, digest in original_code.items()):
        raise ValueError("Frozen primary code differs from the verified primary run")
    fixed = {"plan": plan, "protocol_sha256": protocol_hash, "cache": caches, "code": code,
             "primary_verification_sha256": primary.sha256_file(args.primary_results_root / "verification.json")}
    plan_hash = primary.canonical_hash(fixed)
    fixed["plan_sha256"] = plan_hash
    path = args.out_dir / "fixed_plan.json"
    if path.exists() and read(path) != fixed:
        raise ValueError("Fixed supplementary plan or inputs changed; do not overwrite existing results")
    if not path.exists():
        save(path, fixed)
    return {"out": str(args.out_dir.resolve()), "simulations": str(args.simulations_root.resolve()),
            "supplemental_backend": str(args.supplemental_backend_root.resolve()), "cache": str(args.cache_root.resolve()),
            "protocol": protocol, "protocol_hash": protocol_hash, "plan": plan, "plan_hash": plan_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--primary-results-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--simulations-root", type=Path, required=True)
    parser.add_argument("--supplemental-backend-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("supplemental_plan.json"))
    parser.add_argument("--mode", choices=("prepare", "run"), default="run")
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    configuration = prepare(args)
    print(f"Fixed supplementary plan: {configuration['plan_hash']}", flush=True)
    if args.mode == "prepare":
        return
    started = time.monotonic()
    tasks = [("certificate", 1, 0, 0, configuration)]
    tasks += [("budget", subject, split, seed, configuration) for subject in (1, 2, 3) for split in (0, 1) for seed in (2026100201, 2026100202, 2026100203)]
    tasks += [("block", subject, 0, 0, configuration) for subject in range(1, 19)]
    if args.jobs == 1:
        results = [run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(run_task, task) for task in tasks]
            results = [f.result() for f in as_completed(futures)]
    modules = backend(args.simulations_root, args.supplemental_backend_root)
    aggregate(results, args.out_dir, modules)
    outputs = {str(p.relative_to(args.out_dir)): primary.sha256_file(p) for p in sorted(args.out_dir.rglob("*"))
               if p.is_file() and p.name not in ("manifest.json", "verification.json")}
    save(args.out_dir / "manifest.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                                         "plan_sha256": configuration["plan_hash"], "elapsed_seconds": time.monotonic() - started,
                                         "case_counts": {"certificate": 1, "budget": 18, "block_subjects": 18}, "outputs": outputs,
                                         "interpretation": configuration["plan"]["analysis_role"]})
    print("Supplementary analyses complete; independent verification required.", flush=True)


if __name__ == "__main__":
    main()
