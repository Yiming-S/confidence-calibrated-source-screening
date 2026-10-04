#!/usr/bin/env python3
"""Low-cost analytic-truth baseline comparison for CI-gated screening.

The main protocol table focuses on Top-1, rectangle, and refinement screening.  This
secondary table adds Top-3, an empirical point-threshold rule, the legacy
empirical-best interval gate, and a max-t MCS-style stepdown rule in the same
known-discrepancy setting.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ci_gate_protocol_sim import (
    Condition,
    as_set,
    bootstrap_delta,
    build_system,
    draw_data,
    estimate_delta,
    pair_gate,
    rect_gate,
    set_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "simulation_results" / "extended_baselines_mc1000_b499"


@dataclass(frozen=True)
class BaselineCondition:
    scenario: str
    delta: tuple[float, ...]
    n: int
    d: int = 5

    @property
    def k(self) -> int:
        return len(self.delta)

    def as_protocol_condition(self) -> Condition:
        return Condition(self.scenario, self.delta, self.n, self.d)


def conditions() -> list[BaselineCondition]:
    large = (0.0, 0.25, 0.64, 1.21, 1.96, 2.89, 4.00, 5.29)
    small = (0.0, 0.02, 0.04, 0.06, 0.20, 0.50, 1.00, 1.80)
    tied = (0.0, 0.0, 0.03, 0.30, 0.80, 1.50, 2.40, 3.50)
    return [
        BaselineCondition("large_margin", large, 200),
        BaselineCondition("small_margin", small, 200),
        BaselineCondition("tied_oracle", tied, 200),
    ]


def top_m(delta_hat: np.ndarray, m: int) -> np.ndarray:
    return np.argsort(delta_hat, kind="mergesort")[:m]


def threshold_rule(delta_hat: np.ndarray, radius: float) -> np.ndarray:
    return np.flatnonzero(delta_hat <= np.min(delta_hat) + radius)


def eb_gate(delta_hat: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    best = int(np.argsort(delta_hat, kind="mergesort")[0])
    return np.flatnonzero(lower <= upper[best])


def mcs_stepdown(delta_hat: np.ndarray, delta_boot: np.ndarray, alpha: float) -> np.ndarray:
    """Return a max-t stepdown model-confidence-set analogue.

    At each step, all active-source excesses over the active empirical best are
    studentized.  If their maximum exceeds a bootstrap critical value formed
    from all ordered active-pair contrasts, the empirically worst active source
    is removed; otherwise, the procedure stops.  This is a
    source-screening analogue of MCS, not a claim to reproduce the complete
    Hansen--Lunde--Nason implementation.
    """
    active = list(range(delta_hat.size))
    centered = delta_boot - delta_hat[None, :]

    while len(active) > 1:
        active_arr = np.asarray(active, dtype=int)
        best = int(active_arr[np.argmin(delta_hat[active_arr])])
        worst = int(active_arr[np.argmax(delta_hat[active_arr])])
        competitors = active_arr[active_arr != best]

        diff_hat = delta_hat[competitors] - delta_hat[best]
        diff_boot = centered[:, competitors] - centered[:, [best]]
        diff_se = np.maximum(np.std(diff_boot, axis=0, ddof=1), 1e-12)

        pair_i, pair_j = np.where(~np.eye(active_arr.size, dtype=bool))
        pair_left = active_arr[pair_i]
        pair_right = active_arr[pair_j]
        pair_boot = centered[:, pair_left] - centered[:, pair_right]
        pair_se = np.maximum(np.std(pair_boot, axis=0, ddof=1), 1e-12)

        observed = np.max(diff_hat / diff_se)
        critical = np.quantile(
            np.max(pair_boot / pair_se[None, :], axis=1), 1 - alpha
        )
        if observed <= critical:
            break
        active.remove(worst)

    return np.asarray(active, dtype=int)


def evaluate_condition(
    condition: BaselineCondition,
    rng: np.random.Generator,
    mc: int,
    boot: int,
    alpha: float,
    alpha_c: float,
    alpha_d: float,
    threshold_radius: float,
) -> list[dict[str, object]]:
    true_delta = np.asarray(condition.delta, dtype=float)
    protocol_condition = condition.as_protocol_condition()
    rows: list[dict[str, object]] = []
    base = {
        "scenario": condition.scenario,
        "K": condition.k,
        "n": condition.n,
        "d": condition.d,
        "alpha": alpha,
        "alpha_c": alpha_c,
        "alpha_d": alpha_d,
        "n_boot": boot,
        "delta": json.dumps(condition.delta),
        "oracle_count": int(np.sum(np.isclose(condition.delta, np.min(condition.delta), atol=1e-12, rtol=0.0))),
    }
    for rep in range(mc):
        sources, target = draw_data(protocol_condition, rng)
        delta_hat = estimate_delta(sources, target)
        delta_boot = bootstrap_delta(sources, target, rng, boot, shared_target=True)
        system = build_system(delta_hat, delta_boot, alpha_c, alpha_d, alpha)

        rect = rect_gate(system["lower_c"], system["upper_c"])
        pair = pair_gate(condition.k, system["pairs_j"], system["lower_d"])
        ref = np.array(sorted(as_set(rect) & as_set(pair)), dtype=int)

        selections = {
            "top1": top_m(delta_hat, 1),
            "top3": top_m(delta_hat, 3),
            "threshold": threshold_rule(delta_hat, threshold_radius),
            "eb_gate": eb_gate(delta_hat, system["lower_c"], system["upper_c"]),
            "mcs_stepdown": mcs_stepdown(delta_hat, delta_boot, alpha),
            "rect_split": rect,
            "pair_ref": ref,
        }
        for method, selected in selections.items():
            rows.append({**base, "rep": rep, **set_metrics(method, selected, true_delta)})
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "oracle_retained",
        "exact_recovery",
        "set_size",
        "worst_excess",
        "false_excluded_oracle",
        "false_included_nonoracle",
    ]
    return (
        raw.groupby(["scenario", "K", "n", "method"], as_index=False)[cols]
        .mean(numeric_only=True)
        .sort_values(["scenario", "n", "method"])
    )


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

    rows: list[dict[str, object]] = []
    start = time.time()
    for idx, condition in enumerate(conds, start=1):
        print(
            f"[{idx}/{len(conds)}] {condition.scenario} n={condition.n} "
            f"mc={args.mc} boot={args.boot} elapsed={time.time() - start:.1f}s",
            flush=True,
        )
        rows.extend(
            evaluate_condition(
                condition,
                rng,
                args.mc,
                args.boot,
                args.alpha,
                args.alpha_c,
                args.alpha_d,
                args.threshold_radius,
            )
        )
        pd.DataFrame(rows).to_csv(args.out_dir / "extended_baselines_raw_metrics.csv", index=False)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    summary.to_csv(args.out_dir / "extended_baselines_summary.csv", index=False)
    report = "\n".join(
        [
            "# Extended Analytic-Truth Baselines",
            "",
            f"- Monte Carlo replicates: `{args.mc}`",
            f"- Bootstrap replicates: `{args.boot}`",
            f"- threshold radius: `{args.threshold_radius}`",
            "",
            summary.to_markdown(index=False, floatfmt=".3f"),
            "",
        ]
    )
    (args.out_dir / "extended_baselines_report.md").write_text(report, encoding="utf-8")
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
