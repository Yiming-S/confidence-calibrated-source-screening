#!/usr/bin/env python3
"""Protocol simulation for CI-gated source-discrepancy screening.

The primary analytic model is

    X_T ~ N(0, I_d),      X_k ~ N(sqrt(Delta_k) e_1, I_d),

so the population discrepancy is Delta_k = ||mu_k - mu_T||_2^2.
The estimator is the bias-corrected squared distance between sample means,

    Delta_hat_k = ||Xbar_k - Xbar_T||_2^2 - d * (1/n_k + 1/n_T).

All bootstrap calibration paths preserve the shared-target dependence unless
the condition explicitly requests the wrong independent-target ablation.
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "simulation_results" / "protocol_v1"


@dataclass(frozen=True)
class Condition:
    scenario: str
    delta: tuple[float, ...]
    n: int
    d: int = 5
    include_wrong_target: bool = False

    @property
    def k(self) -> int:
        return len(self.delta)


def scenario_conditions() -> list[Condition]:
    large = (0.0, 0.25, 0.64, 1.21, 1.96, 2.89, 4.00, 5.29)
    small = (0.0, 0.02, 0.04, 0.06, 0.20, 0.50, 1.00, 1.80)
    tied = (0.0, 0.0, 0.03, 0.30, 0.80, 1.50, 2.40, 3.50)
    cal = (0.0, 0.05, 0.12, 0.25, 0.50, 0.90, 1.40, 2.00)

    out: list[Condition] = []
    for n in (50, 100, 200):
        out.append(Condition("large_margin", large, n))
        out.append(Condition("small_margin", small, n))
        out.append(Condition("tied_oracle", tied, n))
    for k in (5, 10, 20, 40):
        if k == 1:
            delta = (0.0,)
        else:
            delta = tuple(1.2 * j / (k - 1) for j in range(k))
        out.append(Condition("growing_k", delta, 100))
    out.append(Condition("calibration_ablation", cal, 100, include_wrong_target=True))
    return out


def estimate_delta(sources: np.ndarray, target: np.ndarray) -> np.ndarray:
    k, n_source, d = sources.shape
    n_target = target.shape[0]
    diff = sources.mean(axis=1) - target.mean(axis=0)
    return np.sum(diff * diff, axis=1) - d * (1.0 / n_source + 1.0 / n_target)


def draw_data(condition: Condition, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(condition.delta, dtype=float)
    k = delta.size
    d = condition.d
    mu = np.zeros((k, d))
    mu[:, 0] = np.sqrt(delta)
    sources = rng.normal(size=(k, condition.n, d)) + mu[:, None, :]
    target = rng.normal(size=(condition.n, d))
    return sources, target


def bootstrap_delta(
    sources: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    shared_target: bool,
) -> np.ndarray:
    k, n_source, d = sources.shape
    n_target = target.shape[0]
    source_means = np.empty((n_boot, k, d))

    for j in range(k):
        idx = rng.integers(0, n_source, size=(n_boot, n_source))
        source_means[:, j, :] = sources[j, idx, :].mean(axis=1)

    if shared_target:
        idx_t = rng.integers(0, n_target, size=(n_boot, n_target))
        target_means = target[idx_t, :].mean(axis=1)[:, None, :]
    else:
        idx_t = rng.integers(0, n_target, size=(n_boot, k, n_target))
        target_means = target[idx_t, :].mean(axis=2)

    diff = source_means - target_means
    return np.sum(diff * diff, axis=2) - d * (1.0 / n_source + 1.0 / n_target)


def ordered_pairs(k: int) -> tuple[np.ndarray, np.ndarray]:
    jj, ll = np.where(~np.eye(k, dtype=bool))
    return jj.astype(int), ll.astype(int)


def quantile(x: np.ndarray, alpha: float) -> float:
    return float(np.quantile(x, 1.0 - alpha))


def build_system(
    delta_hat: np.ndarray,
    delta_boot: np.ndarray,
    alpha_c: float,
    alpha_d: float,
    alpha_joint: float,
) -> dict[str, object]:
    k = delta_hat.size
    jj, ll = ordered_pairs(k)
    d_hat = delta_hat[jj] - delta_hat[ll]
    d_boot = delta_boot[:, jj] - delta_boot[:, ll]

    se_c = np.maximum(delta_boot.std(axis=0, ddof=1), 1e-12)
    se_d = np.maximum(d_boot.std(axis=0, ddof=1), 1e-12)

    t_c = np.max(np.abs((delta_boot - delta_hat[None, :]) / se_c[None, :]), axis=1)
    t_d = np.max(np.abs((d_boot - d_hat[None, :]) / se_d[None, :]), axis=1)
    t_j = np.maximum(t_c, t_d)

    q_c = quantile(t_c, alpha_c)
    q_d = quantile(t_d, alpha_d)
    q_j = quantile(t_j, alpha_joint)

    return {
        "pairs_j": jj,
        "pairs_l": ll,
        "d_hat": d_hat,
        "se_c": se_c,
        "se_d": se_d,
        "q_c": q_c,
        "q_d": q_d,
        "q_j": q_j,
        "lower_c": delta_hat - q_c * se_c,
        "upper_c": delta_hat + q_c * se_c,
        "lower_d": d_hat - q_d * se_d,
        "lower_cj": delta_hat - q_j * se_c,
        "upper_cj": delta_hat + q_j * se_c,
        "lower_dj": d_hat - q_j * se_d,
    }


def rect_gate(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.flatnonzero(lower <= np.min(upper))


def pair_gate(k: int, jj: np.ndarray, lower_d: np.ndarray) -> np.ndarray:
    keep = np.ones(k, dtype=bool)
    for source, lower in zip(jj, lower_d):
        if lower > 0:
            keep[source] = False
    return np.flatnonzero(keep)


def as_set(x: np.ndarray) -> set[int]:
    return set(int(v) for v in x.tolist())


def worst_excess(selected: np.ndarray, true_delta: np.ndarray) -> float:
    if selected.size == 0:
        return -math.inf
    delta_star = float(np.min(true_delta))
    return float(np.max(true_delta[selected] - delta_star))


def set_metrics(method: str, selected: np.ndarray, true_delta: np.ndarray) -> dict[str, float | int | str]:
    delta_star = float(np.min(true_delta))
    oracle = np.flatnonzero(np.isclose(true_delta, delta_star, atol=1e-12, rtol=0.0))
    selected_set = as_set(selected)
    oracle_set = as_set(oracle)
    return {
        "method": method,
        "oracle_retained": int(oracle_set.issubset(selected_set)),
        "exact_recovery": int(selected_set == oracle_set),
        "set_size": int(selected.size),
        "worst_excess": worst_excess(selected, true_delta) if selected.size else math.nan,
        "false_excluded_oracle": int(len(oracle_set - selected_set)),
        "false_included_nonoracle": int(len(selected_set - oracle_set)),
    }


def evaluate_replicate(
    condition: Condition,
    rng: np.random.Generator,
    n_boot: int,
    alpha: float,
    alpha_c: float,
    alpha_d: float,
    wrong_target: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    true_delta = np.asarray(condition.delta, dtype=float)
    k = true_delta.size
    delta_star = float(np.min(true_delta))
    oracle = np.flatnonzero(np.isclose(true_delta, delta_star, atol=1e-12, rtol=0.0))
    oracle_set = as_set(oracle)
    nonoracle = np.setdiff1d(np.arange(k), oracle)
    gap = float(np.min(true_delta[nonoracle] - delta_star)) if nonoracle.size else math.inf

    sources, target = draw_data(condition, rng)
    delta_hat = estimate_delta(sources, target)
    delta_boot = bootstrap_delta(sources, target, rng, n_boot, shared_target=True)
    system = build_system(delta_hat, delta_boot, alpha_c, alpha_d, alpha)

    jj = system["pairs_j"]
    ll = system["pairs_l"]
    true_d = true_delta[jj] - true_delta[ll]

    rect = rect_gate(system["lower_c"], system["upper_c"])
    pair = pair_gate(k, jj, system["lower_d"])
    ref = np.array(sorted(as_set(rect) & as_set(pair)), dtype=int)
    rect_j = rect_gate(system["lower_cj"], system["upper_cj"])
    pair_j = pair_gate(k, jj, system["lower_dj"])
    ref_j = np.array(sorted(as_set(rect_j) & as_set(pair_j)), dtype=int)
    top1 = np.array([int(np.flatnonzero(delta_hat == np.min(delta_hat))[0])])

    methods = [
        ("top1", top1),
        ("rect_split", rect),
        ("pair_split", pair),
        ("ref_split", ref),
        ("rect_combined", rect_j),
        ("pair_combined", pair_j),
        ("ref_combined", ref_j),
    ]

    wrong_events: dict[str, object] = {}
    if wrong_target:
        wrong_boot = bootstrap_delta(sources, target, rng, n_boot, shared_target=False)
        wrong_system = build_system(delta_hat, wrong_boot, alpha_c, alpha_d, alpha)
        wrong_jj = wrong_system["pairs_j"]
        wrong_ll = wrong_system["pairs_l"]
        wrong_rect_j = rect_gate(wrong_system["lower_cj"], wrong_system["upper_cj"])
        wrong_pair_j = pair_gate(k, wrong_jj, wrong_system["lower_dj"])
        wrong_ref_j = np.array(sorted(as_set(wrong_rect_j) & as_set(wrong_pair_j)), dtype=int)
        methods.append(("ref_combined_wrong_target", wrong_ref_j))
        wrong_true_d = true_delta[wrong_jj] - true_delta[wrong_ll]
        wrong_events = {
            "e_cj_wrong": bool(np.all((wrong_system["lower_cj"] <= true_delta) & (true_delta <= wrong_system["upper_cj"]))),
            "e_dj_wrong": bool(np.all(wrong_true_d >= wrong_system["lower_dj"])),
            "wrong_ref_oracle_retained": int(oracle_set.issubset(as_set(wrong_ref_j))),
            "wrong_ref_size": int(wrong_ref_j.size),
        }

    e_c = bool(np.all((system["lower_c"] <= true_delta) & (true_delta <= system["upper_c"])))
    e_d = bool(np.all(true_d >= system["lower_d"]))
    e_cj = bool(np.all((system["lower_cj"] <= true_delta) & (true_delta <= system["upper_cj"])))
    e_dj = bool(np.all(true_d >= system["lower_dj"]))

    wmax = float(np.max(system["upper_c"] - system["lower_c"]))
    wmax_j = float(np.max(system["upper_cj"] - system["lower_cj"]))
    epsilon_pair = float(max(0.0, np.max(true_d - system["lower_d"])))
    epsilon_pair_j = float(max(0.0, np.max(true_d - system["lower_dj"])))

    rect_set = as_set(rect)
    pair_set = as_set(pair)
    ref_set = as_set(ref)
    rect_j_set = as_set(rect_j)
    pair_j_set = as_set(pair_j)
    ref_j_set = as_set(ref_j)

    tol = 1e-10
    l_rect_pair = system["lower_c"][jj] - system["upper_c"][ll]
    domination_lb_all = bool(np.all(system["lower_d"] >= l_rect_pair - tol))
    domination_se_all = bool(
        np.all(system["q_d"] * system["se_d"] <= system["q_c"] * (system["se_c"][jj] + system["se_c"][ll]) + 1e-12)
    )
    pair_subset_rect = bool(pair_set.issubset(rect_set))

    rect_excess = worst_excess(rect, true_delta)
    pair_excess = worst_excess(pair, true_delta)
    ref_excess = worst_excess(ref, true_delta)
    rect_j_excess = worst_excess(rect_j, true_delta)
    pair_j_excess = worst_excess(pair_j, true_delta)
    ref_j_excess = worst_excess(ref_j, true_delta)

    audit = {
        "e_c": e_c,
        "e_d": e_d,
        "e_cj": e_cj,
        "e_dj": e_dj,
        "v_rect": int(e_c and not oracle_set.issubset(rect_set)),
        "v_pair": int(e_d and not oracle_set.issubset(pair_set)),
        "v_ref_split": int(e_c and e_d and not oracle_set.issubset(ref_set)),
        "v_ref_combined": int(e_cj and e_dj and not oracle_set.issubset(ref_j_set)),
        "v_rect_near": int(e_c and rect_excess > 2.0 * wmax + tol),
        "v_pair_near": int(e_d and pair_excess > epsilon_pair + tol),
        "v_ref_split_near": int(e_c and e_d and ref_excess > min(2.0 * wmax, epsilon_pair) + tol),
        "v_rect_combined_near": int(e_cj and rect_j_excess > 2.0 * wmax_j + tol),
        "v_pair_combined_near": int(e_dj and pair_j_excess > epsilon_pair_j + tol),
        "v_ref_combined_near": int(e_cj and e_dj and ref_j_excess > min(2.0 * wmax_j, epsilon_pair_j) + tol),
        "v_exact_rect": int(e_c and gap > 2.0 * wmax and rect_set != oracle_set),
        "v_exact_pair": int(e_d and gap > epsilon_pair and pair_set != oracle_set),
        "v_exact_ref_split": int(e_c and e_d and gap > min(2.0 * wmax, epsilon_pair) and ref_set != oracle_set),
        "v_exact_rect_combined": int(e_cj and gap > 2.0 * wmax_j and rect_j_set != oracle_set),
        "v_exact_pair_combined": int(e_dj and gap > epsilon_pair_j and pair_j_set != oracle_set),
        "v_exact_ref_combined": int(e_cj and e_dj and gap > min(2.0 * wmax_j, epsilon_pair_j) and ref_j_set != oracle_set),
        "v_ref_subset_rect": int(not ref_set.issubset(rect_set)),
        "v_ref_combined_subset_rect": int(not ref_j_set.issubset(rect_j_set)),
        "v_rect_empty": int(rect.size == 0),
        "v_pair_empty": int(pair.size == 0),
        "v_rect_combined_empty": int(rect_j.size == 0),
        "v_pair_combined_empty": int(pair_j.size == 0),
        "gap": gap,
        "wmax": wmax,
        "wmax_j": wmax_j,
        "epsilon_pair": epsilon_pair,
        "epsilon_pair_j": epsilon_pair_j,
        "rect_excess": rect_excess,
        "pair_excess": pair_excess,
        "ref_excess": ref_excess,
        "rect_combined_excess": rect_j_excess,
        "pair_combined_excess": pair_j_excess,
        "ref_combined_excess": ref_j_excess,
        "domination_all": int(domination_se_all),
        "domination_se_all": int(domination_se_all),
        "domination_lb_all": int(domination_lb_all),
        "pair_subset_rect": int(pair_subset_rect),
        "domination_violation": int(domination_lb_all and not pair_subset_rect),
        "v_domination_equiv": int(domination_lb_all != domination_se_all),
        "v_domination_subset": int(domination_lb_all and not pair_subset_rect),
        "top1_not_oracle": int(int(top1[0]) not in oracle_set),
        "top1_choice": int(top1[0]),
        **wrong_events,
    }

    rows = [set_metrics(method, selected, true_delta) for method, selected in methods]
    return rows, audit


def run_condition(
    condition: Condition,
    rng: np.random.Generator,
    n_rep: int,
    n_boot: int,
    alpha: float,
    alpha_c: float,
    alpha_d: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for rep in range(n_rep):
        rows, audit = evaluate_replicate(
            condition=condition,
            rng=rng,
            n_boot=n_boot,
            alpha=alpha,
            alpha_c=alpha_c,
            alpha_d=alpha_d,
            wrong_target=condition.include_wrong_target,
        )
        base = {
            "scenario": condition.scenario,
            "rep": rep,
            "K": condition.k,
            "n": condition.n,
            "d": condition.d,
            "alpha": alpha,
            "alpha_c": alpha_c,
            "alpha_d": alpha_d,
            "n_boot": n_boot,
            "delta": json.dumps(condition.delta),
            "oracle_count": int(np.sum(np.isclose(condition.delta, np.min(condition.delta), atol=1e-12, rtol=0.0))),
        }
        for row in rows:
            metric_rows.append({**base, **row})
        audit_rows.append({**base, **audit})
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
        raw.groupby(["scenario", "K", "n", "method"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["scenario", "K", "n", "method"])
    )


def summarize_audit(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "e_c",
        "e_d",
        "e_cj",
        "e_dj",
        "domination_all",
        "domination_se_all",
        "domination_lb_all",
        "pair_subset_rect",
        "top1_not_oracle",
    ]
    cols += sorted(c for c in raw.columns if c.startswith("v_"))
    optional = [c for c in ("e_cj_wrong", "e_dj_wrong", "wrong_ref_oracle_retained", "wrong_ref_size") if c in raw.columns]
    agg_cols = list(dict.fromkeys(cols + optional))
    return (
        raw.groupby(["scenario", "K", "n"], as_index=False)[agg_cols]
        .mean()
        .sort_values(["scenario", "K", "n"])
    )


def summarize_top1(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scenario", "K", "n"]
    counts = (
        raw.groupby(group_cols + ["top1_choice"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = raw.groupby(group_cols, as_index=False).size().rename(columns={"size": "total"})
    out = counts.merge(totals, on=group_cols)
    out["selection_frequency"] = out["count"] / out["total"]

    entropy = (
        out.assign(term=lambda x: -x["selection_frequency"] * np.log(x["selection_frequency"]))
        .groupby(group_cols, as_index=False)["term"]
        .sum()
        .rename(columns={"term": "selection_entropy"})
    )
    return out.merge(entropy, on=group_cols).sort_values(group_cols + ["top1_choice"])


def write_report(out_dir: Path, metrics: pd.DataFrame, audit: pd.DataFrame, args: argparse.Namespace) -> None:
    top1_path = out_dir / "protocol_top1_summary.csv"
    top1_block = ""
    if top1_path.exists():
        top1 = pd.read_csv(top1_path)
        top1_block = "\n## Top-1 Selection Frequencies\n\n" + top1.to_markdown(index=False, floatfmt=".3f") + "\n"

    lines = [
        "# CI-Gated Source Screening Protocol Simulation",
        "",
        f"- Monte Carlo replicates: `{args.mc}`",
        f"- Bootstrap replicates: `{args.boot}`",
        f"- alpha: `{args.alpha}`",
        f"- alpha split: `({args.alpha_c}, {args.alpha_d})`",
        "",
        "## Theorem Audit",
        "",
        audit.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Finite-Sample Metrics",
        "",
        metrics.to_markdown(index=False, floatfmt=".3f"),
        top1_block,
        "",
        "The violation columns should be zero up to numerical tolerance.  The wrong-target columns are reported only for conditions that request the independent-target bootstrap ablation.",
    ]
    (out_dir / "protocol_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", type=int, default=2000)
    parser.add_argument("--boot", type=int, default=1999)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-c", type=float, default=0.025)
    parser.add_argument("--alpha-d", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit-conditions", type=int, default=None)
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated scenario names to run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = scenario_conditions()
    if args.scenarios:
        wanted = {x.strip() for x in args.scenarios.split(",") if x.strip()}
        conditions = [c for c in conditions if c.scenario in wanted]
    if args.limit_conditions is not None:
        conditions = conditions[: args.limit_conditions]

    rng = np.random.default_rng(args.seed)
    metric_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    start = time.time()

    manifest = {
        "args": vars(args) | {"out_dir": str(out_dir)},
        "conditions": [asdict(c) for c in conditions],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for idx, condition in enumerate(conditions, start=1):
        elapsed = time.time() - start
        print(
            f"[{idx}/{len(conditions)}] {condition.scenario} K={condition.k} n={condition.n} "
            f"mc={args.mc} boot={args.boot} elapsed={elapsed:.1f}s",
            flush=True,
        )
        rows, audits = run_condition(condition, rng, args.mc, args.boot, args.alpha, args.alpha_c, args.alpha_d)
        metric_rows.extend(rows)
        audit_rows.extend(audits)

        pd.DataFrame(metric_rows).to_csv(out_dir / "protocol_raw_metrics.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(out_dir / "protocol_raw_audit.csv", index=False)

    raw_metrics = pd.DataFrame(metric_rows)
    raw_audit = pd.DataFrame(audit_rows)
    metric_summary = summarize_metrics(raw_metrics)
    audit_summary = summarize_audit(raw_audit)
    top1_summary = summarize_top1(raw_audit)
    metric_summary.to_csv(out_dir / "protocol_metric_summary.csv", index=False)
    audit_summary.to_csv(out_dir / "protocol_audit_summary.csv", index=False)
    top1_summary.to_csv(out_dir / "protocol_top1_summary.csv", index=False)
    write_report(out_dir, metric_summary, audit_summary, args)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
