#!/usr/bin/env python3
"""Independent record/summary checks and fixed numerical replays for supplements."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import supplemental as run
from verify_results import equal, require


def mean(values):
    return math.fsum(float(value) for value in values) / len(values)


def load_draws(out, row, plan_hash):
    path = out / "draws" / row["archive"]
    equal(run.primary.sha256_file(path), row["archive_sha256"], f"archive {path.name}")
    with np.load(path, allow_pickle=False) as saved:
        equal(str(saved["plan_sha256"].item()), plan_hash, f"archive plan {path.name}")
        estimate, draws = saved["estimate"], saved["draws"]
    require(estimate.shape == (5,) and draws.shape[1] == 5 and np.isfinite(draws).all(), f"Invalid draws: {path}")
    equal(run.hashlib.sha256(np.asarray(draws, dtype="<f8").tobytes()).hexdigest(), row["bootstrap_matrix_sha256"], f"matrix {path.name}")
    return estimate, draws


def verify_certificate(out, plan_hash, modules):
    certificate = run.read(out / "certificate.json")
    estimate, draws = load_draws(out, certificate, plan_hash)
    equal(draws.shape[0], 1999, "certificate budget")
    equal(certificate["subject"], 1, "fixed certificate subject")
    equal(certificate["estimate"], estimate.tolist(), "certificate estimates")
    system = run.calibrated(estimate, draws, modules)
    equal(certificate["system"], system, "certificate calibrated bounds")
    for index, row in enumerate(certificate["sources"]):
        equal(row["source_session"], index + 1, "certificate source identity")
        margin = system["lower_c"][index] - min(system["upper_c"])
        witnesses = [{"competitor_session": competitor + 1, "lower_bound": lower}
                     for j, competitor, lower in zip(system["jj"], system["ll"], system["lower_d"]) if j == index and lower > 0]
        equal(row["rectangle_exclusion_margin"], margin, "rectangle exclusion margin")
        equal(row["positive_contrast_witnesses"], witnesses, "direct witnesses")
        equal(row["refinement_retained"], not (margin > 0 or bool(witnesses)), "certificate union of exclusion rules")
    return certificate, estimate, draws


def verify_budget(out, plan_hash, modules):
    result, summary = run.read(out / "budget_results.json"), run.read(out / "budget_summary.json")
    cases, rows = result["cases"], result["rows"]
    equal([(c["subject"], c["split"], c["bootstrap_seed"]) for c in cases],
          [(s, split, seed) for s in (1, 2, 3) for split in (0, 1) for seed in (2026100201, 2026100202, 2026100203)], "budget fixed cases")
    equal(len(rows), 72, "budget row count")
    equal(len(result["membership"]), 360, "budget membership count")
    equal(rows, [r for c in cases for r in c["rows"]], "budget flattened records")
    for case in cases:
        estimate, draws = load_draws(out, case, plan_hash)
        equal(draws.shape[0], 999, "budget max draws")
        require(not set(case["screen_indices"]) & set(case["test_indices"]), "Budget split leakage")
        reference = modules["ci_gate_bspc_bootstrap_budget"].calibrate(estimate, draws)["selected"]
        for row in case["rows"]:
            system = modules["ci_gate_bspc_bootstrap_budget"].calibrate(estimate, draws[:row["budget"]])
            selected = set(system["selected"].tolist())
            reference_set = set(reference.tolist())
            equal(row["members"], [i + 1 for i in sorted(selected)], "budget selected set")
            equal(row["set_size"], len(selected), "budget set size")
            equal(row["jaccard_vs_999"], len(selected & reference_set) / len(selected | reference_set), "budget Jaccard")
            equal(row["same_members_vs_999"], int(selected == reference_set), "budget exact match")
            for field, key in (("q_component", "q_c"), ("q_contrast", "q_d")):
                equal(row[field], system[key], f"budget {field}")
            for label, values in (("component", draws[:row["budget"]]),
                                  ("contrast", draws[:row["budget"], system["jj"]] - draws[:row["budget"], system["ll"]])):
                se = np.std(values, axis=0, ddof=1)
                equal(row[f"raw_{label}_se_min"], float(se.min()), "raw SE minimum")
                equal(row[f"raw_{label}_se_max"], float(se.max()), "raw SE maximum")
                equal(row[f"raw_{label}_se_floor_count"], int(np.sum(se < 1e-12)), "raw SE floor count")
                equal(row[f"raw_{label}_se_nonfinite_count"], int(np.sum(~np.isfinite(se))), "raw SE nonfinite count")
    for stored in summary["rows"]:
        for field in stored:
            if field != "budget":
                within = [mean([row[field] for row in rows if row["subject"] == subject and row["budget"] == stored["budget"]]) for subject in (1, 2, 3)]
                equal(stored[field], mean(within), f"budget subject-averaged summary {field}")
    for stored in summary["subject_summary"]:
        matching = [row for row in rows if row["subject"] == stored["subject"] and row["budget"] == stored["budget"]]
        equal(len(matching), 6, "budget subject sample count")
        for field in stored:
            if field not in ("budget", "subject"):
                equal(stored[field], mean([r[field] for r in matching]), f"budget subject summary {field}")
    for stored in summary["seed_variability_cases"]:
        matching = [r for r in rows if all(r[key] == stored[key] for key in ("subject", "split", "budget"))]
        pairs = list(itertools.combinations([set(r["members"]) for r in matching], 2))
        equal(stored["pairwise_seed_jaccard"], mean([len(a & b) / len(a | b) for a, b in pairs]), "between-seed Jaccard")
        equal(stored["pairwise_seed_exact"], mean([a == b for a, b in pairs]), "between-seed exact")
        equal(stored["set_size_seed_sd"], float(np.std([r["set_size"] for r in matching], ddof=1)), "between-seed size SD")
    return cases


def verify_block(out, plan_hash, modules):
    result, summary = run.read(out / "block_results.json"), run.read(out / "block_summary.json")
    subjects, screening, downstream = result["subjects"], result["screening"], result["downstream"]
    equal([s["subject"] for s in subjects], list(range(1, 19)), "block subject roster")
    equal([len(screening), len(downstream)], [54, 900], "block record counts")
    equal(screening, [r for s in subjects for r in s["screening"]], "full block flattened records")
    equal(downstream, [r for s in subjects for r in s["downstream"]], "downstream block flattened records")
    for subject in subjects:
        equal([len(subject["screening"]), len(subject["downstream"]), len(subject["splits"]), len(subject["draws"])], [3, 50, 10, 23], "per-subject block counts")
        by_case = {}
        for row in subject["draws"]:
            estimate, draws = load_draws(out, row, plan_hash)
            equal(draws.shape[0], 499 if row["analysis"] == "full" else 99, "block budget")
            sets = modules["ci_gate_bnci004_riemann"].selected_sets(estimate, draws, .05, .025, .025)
            equal(row["sets"], run.primary.plain(sets), "block recalibrated selection")
            key = (row["analysis"], row.get("split"), row["block_length"])
            require(key not in by_case, "Repeated block case")
            by_case[key] = sets
        for row in subject["screening"]:
            selected = by_case[("full", None, row["block_length"])]
            ref, iid = set(selected["pair_ref"].tolist()), set(by_case[("full", None, 1)]["pair_ref"].tolist())
            equal(row["pair_ref_sessions"], [i + 1 for i in sorted(ref)], "block full refinement sessions")
            equal(row["mcs_sessions"], (selected["mcs_stepdown"] + 1).tolist(), "block full MCS sessions")
            equal(row["pair_ref_size"], len(ref), "block refinement size")
            equal(row["mcs_size"], len(selected["mcs_stepdown"]), "block MCS size")
            equal(row["jaccard_vs_iid"], len(ref & iid) / len(ref | iid), "block full Jaccard")
            equal(row["exact_match_iid"], int(ref == iid), "block full exact")
        for row in subject["splits"]:
            equal(row["ordered_screen_indices"], sorted(row["screen_indices"]), "block recording order")
            require(not set(row["screen_indices"]) & set(row["test_indices"]), "Block split leakage")
        for row in subject["downstream"]:
            require(math.isfinite(row["balanced_accuracy"]) and 0 <= row["balanced_accuracy"] <= 1, "Invalid block accuracy")
            sessions = [1, 2, 3, 4, 5] if row["method"] == "all_sources" else [i + 1 for i in sorted(by_case[("downstream", row["split"], row["block_length"])][row["method"]].tolist())]
            equal(row["source_sessions"], sessions, "block downstream source set")
            equal(row["set_size"], len(sessions), "block downstream size")
    for stored in summary["screening"]:
        matching = [r for r in screening if r["block_length"] == stored["block_length"]]
        equal(len(matching), 18, "block summary subjects")
        for field in ("pair_ref_size", "pair_ref_fraction", "mcs_size", "jaccard_vs_iid", "exact_match_iid"):
            equal(stored[field], mean([r[field] for r in matching]), f"block summary {field}")
    expected_subjects = {}
    for subject in range(1, 19):
        for length, method in [(0, "all_sources"), (1, "pair_ref"), (1, "mcs_stepdown"), (4, "pair_ref"), (4, "mcs_stepdown")]:
            matching = [r for r in downstream if (r["subject"], r["block_length"], r["method"]) == (subject, length, method)]
            equal(len(matching), 10, "block mean split count")
            expected_subjects[(subject, length, method)] = {field: mean([r[field] for r in matching]) for field in ("set_size", "balanced_accuracy")}
    for stored in summary["subject_downstream"]:
        equal({field: stored[field] for field in ("set_size", "balanced_accuracy")}, expected_subjects[(stored["subject"], stored["block_length"], stored["method"])], "block subject means")
    all_means = [expected_subjects[(s, 0, "all_sources")]["balanced_accuracy"] for s in range(1, 19)]
    equal(summary["all_source_accuracy"], mean(all_means), "block all-source mean")
    for stored in summary["downstream"]:
        selected = [expected_subjects[(s, stored["block_length"], stored["method"])] for s in range(1, 19)]
        equal(stored["mean_set_size"], mean([r["set_size"] for r in selected]), "block downstream mean size")
        equal(stored["mean_balanced_accuracy"], mean([r["balanced_accuracy"] for r in selected]), "block downstream mean accuracy")
        equal(stored["mean_diff_vs_all"], mean([r["balanced_accuracy"] - a for r, a in zip(selected, all_means)]), "block paired difference")
    return subjects


def verify(args):
    out = args.out_dir
    fixed, manifest = run.read(out / "fixed_plan.json"), run.read(out / "manifest.json")
    plan_hash = fixed["plan_sha256"]
    equal(manifest["status"], "complete", "supplementary completion")
    equal(manifest["plan_sha256"], plan_hash, "supplementary manifest plan")
    for name, digest in manifest["outputs"].items():
        equal(run.primary.sha256_file(out / name), digest, f"manifest output {name}")
    modules = run.backend(args.simulations_root, args.supplemental_backend_root)
    certificate, cert_estimate, cert_draws = verify_certificate(out, plan_hash, modules)
    cases = verify_budget(out, plan_hash, modules)
    subjects = verify_block(out, plan_hash, modules)
    protocol = run.read(args.protocol)
    with threadpool_limits(limits=1):
        sources, labels, target = run.subject_data(1, args.cache_root, protocol, fixed["protocol_sha256"])
        estimate, draws = modules["ci_gate_ma2020_riemann"].riemann_estimates_and_bootstrap(sources, target["covs"], np.random.default_rng(2026100200), 1999, .1, True)
        np.testing.assert_array_equal(estimate, cert_estimate)
        np.testing.assert_array_equal(draws, cert_draws)
        case = cases[0]
        screen = np.asarray(case["screen_indices"])
        estimate, draws = modules["ci_gate_bspc_bootstrap_budget"].discrepancy_bootstrap(sources, target["covs"][screen], np.random.default_rng(np.random.SeedSequence([2026100201, 4, 1, 0])), 999, .1)
        stored_estimate, stored_draws = load_draws(out, case, plan_hash)
        np.testing.assert_array_equal(estimate, stored_estimate)
        np.testing.assert_array_equal(draws, stored_draws)
        block = subjects[0]
        split = block["splits"][0]
        screen, test = np.asarray(split["ordered_screen_indices"]), np.asarray(split["test_indices"])
        for length in (1, 4):
            row = next(r for r in block["draws"] if r["analysis"] == "downstream" and r["split"] == 0 and r["block_length"] == length)
            estimate, draws = modules["ci_gate_block_bootstrap_sensitivity"].riemann_estimates_and_block_bootstrap(sources, target["covs"][screen], np.random.default_rng(20260715 + 1000000 + length * 100), 99, .1, length)
            expected_estimate, expected_draws = load_draws(out, row, plan_hash)
            np.testing.assert_array_equal(estimate, expected_estimate)
            np.testing.assert_array_equal(draws, expected_draws)
        for row in [r for r in block["downstream"] if r["split"] == 0]:
            indices = np.asarray(row["source_sessions"]) - 1
            x, y = modules["ci_gate_bnci004_riemann"].pool(indices, sources, labels)
            accuracy = modules["ci_gate_ma2020_riemann"].train_eval_ts(x, y, target["covs"][test], target["labels"][test])
            equal(accuracy, row["balanced_accuracy"], "independent block subject1 split0 classifier replay")
    return {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_hash,
            "verifier_sha256": run.primary.sha256_file(Path(__file__)),
            "counts": {"certificate_sources": 5, "budget_draw_arrays": 18, "budget_rows": 72, "block_subjects": 18,
                       "block_screening_rows": 54, "block_downstream_rows": 900, "block_draw_arrays": 414},
            "checks": "All output hashes; every archived draw recalibrated; source sets, exclusion certificates, raw SE diagnostics, subject-level aggregation and paired means checked. Exact numerical replay: certificate subject1, budget subject1/split0/seed2026100201, block subject1/split0 lengths1/4 and all five downstream model results."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out-dir", "protocol", "cache-root", "simulations-root", "supplemental-backend-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args)
    except Exception as error:
        run.save(args.out_dir / "verification.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
    run.save(args.out_dir / "verification.json", result)
    print("Independent supplementary verification passed.", flush=True)


if __name__ == "__main__":
    main()
