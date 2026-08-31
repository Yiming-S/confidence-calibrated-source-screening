#!/usr/bin/env python3
"""Target-limited analytic-truth simulation for CI-gated source screening.

This block keeps the same Gaussian location model as the main protocol
simulation, but uses unbalanced sample sizes with n_T << n_source.  The purpose
is to stress the shared-target calibration layer in the regime that motivates
multi-source transfer: source sessions can be much larger than the target
sample used for screening.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ci_gate_protocol_sim import (
    as_set,
    bootstrap_delta,
    build_system,
    estimate_delta,
    pair_gate,
    rect_gate,
    set_metrics,
    worst_excess,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "simulation_results" / "target_limited_mc1000_b499"


@dataclass(frozen=True)
class TargetLimitedCondition:
    scenario: str
    delta: tuple[float, ...]
    n_target: int
    n_source: int
    d: int = 5

    @property
    def k(self) -> int:
        return len(self.delta)


def conditions() -> list[TargetLimitedCondition]:
    small = (0.0, 0.02, 0.04, 0.06, 0.20, 0.50, 1.00, 1.80)
    medium = (0.0, 0.05, 0.12, 0.25, 0.50, 0.90, 1.40, 2.00)
    grids = [(20, 100), (20, 500), (50, 200), (50, 500)]
    out: list[TargetLimitedCondition] = []
    for n_t, n_s in grids:
        out.append(TargetLimitedCondition("target_limited_small", small, n_t, n_s))
        out.append(TargetLimitedCondition("target_limited_medium", medium, n_t, n_s))
    return out


def draw_data(condition: TargetLimitedCondition, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(condition.delta, dtype=float)
    mu = np.zeros((condition.k, condition.d))
    mu[:, 0] = np.sqrt(delta)
    sources = rng.normal(size=(condition.k, condition.n_source, condition.d)) + mu[:, None, :]
    target = rng.normal(size=(condition.n_target, condition.d))
    return sources, target


def top_m(delta_hat: np.ndarray, m: int) -> np.ndarray:
    return np.argsort(delta_hat, kind="mergesort")[:m]


def threshold_rule(delta_hat: np.ndarray, radius: float) -> np.ndarray:
    return np.flatnonzero(delta_hat <= np.min(delta_hat) + radius)


def eb_gate(delta_hat: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    best = int(np.argsort(delta_hat, kind="mergesort")[0])
    return np.flatnonzero(lower <= upper[best])


def evaluate_replicate(
    condition: TargetLimitedCondition,
    rng: np.random.Generator,
    n_boot: int,
    alpha: float,
    alpha_c: float,
    alpha_d: float,
    threshold_radius: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    true_delta = np.asarray(condition.delta, dtype=float)
    delta_star = float(np.min(true_delta))
    oracle = np.flatnonzero(np.isclose(true_delta, delta_star, atol=1e-12, rtol=0.0))
    oracle_set = as_set(oracle)
    nonoracle = np.setdiff1d(np.arange(condition.k), oracle)
    gap = float(np.min(true_delta[nonoracle] - delta_star)) if nonoracle.size else math.inf

    sources, target = draw_data(condition, rng)
    delta_hat = estimate_delta(sources, target)

    shared_boot = bootstrap_delta(sources, target, rng, n_boot, shared_target=True)
    shared = build_system(delta_hat, shared_boot, alpha_c, alpha_d, alpha)
    wrong_boot = bootstrap_delta(sources, target, rng, n_boot, shared_target=False)
    wrong = build_system(delta_hat, wrong_boot, alpha_c, alpha_d, alpha)

    jj = shared["pairs_j"]
    ll = shared["pairs_l"]
    true_d = true_delta[jj] - true_delta[ll]
    wrong_true_d = true_delta[wrong["pairs_j"]] - true_delta[wrong["pairs_l"]]

    rect = rect_gate(shared["lower_c"], shared["upper_c"])
    pair = pair_gate(condition.k, jj, shared["lower_d"])
    ref = np.array(sorted(as_set(rect) & as_set(pair)), dtype=int)
    rect_j = rect_gate(shared["lower_cj"], shared["upper_cj"])
    pair_j = pair_gate(condition.k, jj, shared["lower_dj"])
    ref_j = np.array(sorted(as_set(rect_j) & as_set(pair_j)), dtype=int)

    wrong_rect_j = rect_gate(wrong["lower_cj"], wrong["upper_cj"])
    wrong_pair_j = pair_gate(condition.k, wrong["pairs_j"], wrong["lower_dj"])
    wrong_ref_j = np.array(sorted(as_set(wrong_rect_j) & as_set(wrong_pair_j)), dtype=int)

    selections = [
        ("top1", top_m(delta_hat, 1)),
        ("top3", top_m(delta_hat, 3)),
        ("threshold", threshold_rule(delta_hat, threshold_radius)),
        ("eb_gate", eb_gate(delta_hat, shared["lower_c"], shared["upper_c"])),
        ("rect_split", rect),
        ("pair_ref", ref),
        ("ref_combined", ref_j),
        ("ref_combined_wrong_target", wrong_ref_j),
    ]

    e_c = bool(np.all((shared["lower_c"] <= true_delta) & (true_delta <= shared["upper_c"])))
    e_d = bool(np.all(true_d >= shared["lower_d"]))
    e_cj = bool(np.all((shared["lower_cj"] <= true_delta) & (true_delta <= shared["upper_cj"])))
    e_dj = bool(np.all(true_d >= shared["lower_dj"]))
    e_cj_wrong = bool(np.all((wrong["lower_cj"] <= true_delta) & (true_delta <= wrong["upper_cj"])))
    e_dj_wrong = bool(np.all(wrong_true_d >= wrong["lower_dj"]))

    wmax = float(np.max(shared["upper_c"] - shared["lower_c"]))
    epsilon_pair = float(max(0.0, np.max(true_d - shared["lower_d"])))
    ref_excess = worst_excess(ref, true_delta)

    audit = {
        "e_c": e_c,
        "e_d": e_d,
        "e_cd": e_c and e_d,
        "e_cj": e_cj,
        "e_dj": e_dj,
        "e_cdj": e_cj and e_dj,
        "e_cj_wrong": e_cj_wrong,
        "e_dj_wrong": e_dj_wrong,
        "e_cdj_wrong": e_cj_wrong and e_dj_wrong,
        "v_rect": int(e_c and not oracle_set.issubset(as_set(rect))),
        "v_pair": int(e_d and not oracle_set.issubset(as_set(pair))),
        "v_ref_split": int(e_c and e_d and not oracle_set.issubset(as_set(ref))),
        "v_ref_combined": int(e_cj and e_dj and not oracle_set.issubset(as_set(ref_j))),
        "v_ref_combined_wrong_target": int(e_cj_wrong and e_dj_wrong and not oracle_set.issubset(as_set(wrong_ref_j))),
        "v_ref_split_near": int(e_c and e_d and ref_excess > min(2.0 * wmax, epsilon_pair) + 1e-10),
        "v_exact_rect": int(e_c and gap > 2.0 * wmax and as_set(rect) != oracle_set),
        "v_exact_pair": int(e_d and gap > epsilon_pair and as_set(pair) != oracle_set),
        "shared_ref_oracle_retained": int(oracle_set.issubset(as_set(ref_j))),
        "wrong_ref_oracle_retained": int(oracle_set.issubset(as_set(wrong_ref_j))),
        "shared_ref_size": int(ref_j.size),
        "wrong_ref_size": int(wrong_ref_j.size),
        "gap": gap,
        "wmax": wmax,
        "epsilon_pair": epsilon_pair,
    }
    rows = [set_metrics(method, selected, true_delta) for method, selected in selections]
    return rows, audit


def run_condition(
    condition: TargetLimitedCondition,
    rng: np.random.Generator,
    n_rep: int,
    n_boot: int,
    alpha: float,
    alpha_c: float,
    alpha_d: float,
    threshold_radius: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    base = {
        "scenario": condition.scenario,
        "K": condition.k,
        "n_target": condition.n_target,
        "n_source": condition.n_source,
        "d": condition.d,
        "alpha": alpha,
        "alpha_c": alpha_c,
        "alpha_d": alpha_d,
        "n_boot": n_boot,
        "delta": json.dumps(condition.delta),
        "oracle_count": int(np.sum(np.isclose(condition.delta, np.min(condition.delta), atol=1e-12, rtol=0.0))),
    }
    for rep in range(n_rep):
        rows, audit = evaluate_replicate(condition, rng, n_boot, alpha, alpha_c, alpha_d, threshold_radius)
        for row in rows:
            metric_rows.append({**base, "rep": rep, **row})
        audit_rows.append({**base, "rep": rep, **audit})
    return metric_rows, audit_rows


def summarize_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "oracle_retained",
        "exact_recovery",
        "set_size",
        "worst_excess",
        "false_excluded_oracle",
        "false_included_nonoracle",
    ]
    return (
        raw.groupby(["scenario", "K", "n_target", "n_source", "method"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["scenario", "n_target", "n_source", "method"])
    )


def summarize_audit(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "e_c",
        "e_d",
        "e_cd",
        "e_cj",
        "e_dj",
        "e_cdj",
        "e_cj_wrong",
        "e_dj_wrong",
        "e_cdj_wrong",
        "shared_ref_oracle_retained",
        "wrong_ref_oracle_retained",
        "shared_ref_size",
        "wrong_ref_size",
    ]
    cols += sorted(c for c in raw.columns if c.startswith("v_"))
    return (
        raw.groupby(["scenario", "K", "n_target", "n_source"], as_index=False)[cols]
        .mean(numeric_only=True)
        .sort_values(["scenario", "n_target", "n_source"])
    )


def write_report(out_dir: Path, metrics: pd.DataFrame, audit: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Target-Limited Source Screening Simulation",
        "",
        f"- Monte Carlo replicates: `{args.mc}`",
        f"- Bootstrap replicates: `{args.boot}`",
        f"- alpha: `{args.alpha}`",
        f"- alpha split: `({args.alpha_c}, {args.alpha_d})`",
        "",
        "## Theorem Audit and Calibration",
        "",
        audit.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Finite-Sample Metrics",
        "",
        metrics.to_markdown(index=False, floatfmt=".3f"),
    ]
    (out_dir / "target_limited_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", type=int, default=1000)
    parser.add_argument("--boot", type=int, default=499)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-c", type=float, default=0.025)
    parser.add_argument("--alpha-d", type=float, default=0.025)
    parser.add_argument("--threshold-radius", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit-conditions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    conds = conditions()
    if args.limit_conditions is not None:
        conds = conds[: args.limit_conditions]

    manifest = {
        "args": vars(args) | {"out_dir": str(args.out_dir)},
        "conditions": [asdict(c) for c in conds],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metric_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    start = time.time()
    for idx, condition in enumerate(conds, start=1):
        print(
            f"[{idx}/{len(conds)}] {condition.scenario} nT={condition.n_target} nS={condition.n_source} "
            f"mc={args.mc} boot={args.boot} elapsed={time.time() - start:.1f}s",
            flush=True,
        )
        rows, audits = run_condition(
            condition,
            rng,
            args.mc,
            args.boot,
            args.alpha,
            args.alpha_c,
            args.alpha_d,
            args.threshold_radius,
        )
        metric_rows.extend(rows)
        audit_rows.extend(audits)
        pd.DataFrame(metric_rows).to_csv(args.out_dir / "target_limited_raw_metrics.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(args.out_dir / "target_limited_raw_audit.csv", index=False)

    raw_metrics = pd.DataFrame(metric_rows)
    raw_audit = pd.DataFrame(audit_rows)
    metric_summary = summarize_metrics(raw_metrics)
    audit_summary = summarize_audit(raw_audit)
    metric_summary.to_csv(args.out_dir / "target_limited_metric_summary.csv", index=False)
    audit_summary.to_csv(args.out_dir / "target_limited_audit_summary.csv", index=False)
    write_report(args.out_dir, metric_summary, audit_summary, args)
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
