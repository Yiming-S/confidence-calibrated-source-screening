"""Straddling under-coverage simulation for CI-gated source screening.

Empirical companion to Proposition prop:anticonservative: with two sources
straddling the target (mu_1 = -a, mu_2 = +a), the target influence functions
are perfectly negatively correlated, so independent target resampling
understates the contrast variance by a factor of two in the target-limited
regime and its nominal 95% contrast interval covers only about 83.4%.
With both sources on the same side (mu_1 = mu_2 = +a) the correlation is
perfectly positive and independent target resampling over-covers.
Shared-target resampling is calibrated correctly in both geometries.

Model: d = 1, K = 2, sources N(mu_k, 1) with n_source observations each,
target N(0, 1) with n_target observations, squared mean discrepancy with
the bias-corrected plug-in estimator.  True contrast D_12 = a^2 - a^2 = 0
in both geometries (a tie), and the tie is first-order nondegenerate
because the source influence variances 4 a^2 v_s stay positive.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ci_gate_protocol_sim import estimate_delta, bootstrap_delta

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "simulation_results" / "straddle_undercoverage_mc2000_b999"


@dataclass(frozen=True)
class StraddleCondition:
    label: str
    geometry: str  # "straddle" or "same_side"
    a: float
    n_source: int
    n_target: int

    @property
    def mu(self) -> tuple[float, float]:
        if self.geometry == "straddle":
            return (-self.a, self.a)
        return (self.a, self.a)


def conditions(a: float, n_source: int, n_targets: tuple[int, ...]) -> list[StraddleCondition]:
    out: list[StraddleCondition] = []
    for geometry in ("straddle", "same_side"):
        for n_target in n_targets:
            out.append(
                StraddleCondition(
                    label=f"{geometry}_nT{n_target}",
                    geometry=geometry,
                    a=a,
                    n_source=n_source,
                    n_target=n_target,
                )
            )
    return out


def draw_data(cond: StraddleCondition, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(cond.mu, dtype=float).reshape(2, 1)
    sources = rng.normal(size=(2, cond.n_source, 1)) + mu[:, None, :]
    target = rng.normal(size=(cond.n_target, 1))
    return sources, target


def contrast_interval_cover(
    d_hat: float,
    d_true: float,
    boot_dev: np.ndarray,
    alpha: float,
) -> tuple[bool, bool, float]:
    """Two-sided contrast coverage for studentized and percentile intervals.

    boot_dev holds the bootstrap contrast deviations (D*_12 - Dhat_12); the
    studentized interval uses the bootstrap standard error and the quantile
    of the absolute studentized deviations, mirroring build_system.
    """
    se = float(max(np.std(boot_dev, ddof=1), 1e-12))
    q_stud = float(np.quantile(np.abs(boot_dev / se), 1.0 - alpha))
    q_abs = float(np.quantile(np.abs(boot_dev), 1.0 - alpha))
    err = abs(d_hat - d_true)
    return bool(err <= q_stud * se), bool(err <= q_abs), se


def run_condition(
    cond: StraddleCondition,
    n_mc: int,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    d_true = 0.0  # a^2 - a^2 in both geometries
    rows: list[dict[str, object]] = []
    for rep in range(n_mc):
        sources, target = draw_data(cond, rng)
        delta_hat = estimate_delta(sources, target)
        d_hat = float(delta_hat[0] - delta_hat[1])

        boot_shared = bootstrap_delta(sources, target, rng, n_boot, shared_target=True)
        boot_wrong = bootstrap_delta(sources, target, rng, n_boot, shared_target=False)
        dev_shared = (boot_shared[:, 0] - boot_shared[:, 1]) - d_hat
        dev_wrong = (boot_wrong[:, 0] - boot_wrong[:, 1]) - d_hat

        cover_sh_stud, cover_sh_abs, se_sh = contrast_interval_cover(d_hat, d_true, dev_shared, alpha)
        cover_wr_stud, cover_wr_abs, se_wr = contrast_interval_cover(d_hat, d_true, dev_wrong, alpha)

        rows.append(
            {
                "condition": cond.label,
                "geometry": cond.geometry,
                "n_source": cond.n_source,
                "n_target": cond.n_target,
                "replicate": rep,
                "d_hat": d_hat,
                "cover_shared_stud": cover_sh_stud,
                "cover_wrong_stud": cover_wr_stud,
                "cover_shared_abs": cover_sh_abs,
                "cover_wrong_abs": cover_wr_abs,
                "se_shared": se_sh,
                "se_wrong": se_wr,
                "var_ratio_shared_over_wrong": (se_sh / se_wr) ** 2,
            }
        )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        raw.groupby(["geometry", "n_target"], as_index=False)
        .agg(
            cover_shared=("cover_shared_stud", "mean"),
            cover_wrong=("cover_wrong_stud", "mean"),
            cover_shared_abs=("cover_shared_abs", "mean"),
            cover_wrong_abs=("cover_wrong_abs", "mean"),
            var_ratio=("var_ratio_shared_over_wrong", "mean"),
            replicates=("replicate", "count"),
        )
        .sort_values(["geometry", "n_target"])
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc", type=int, default=2000)
    parser.add_argument("--boot", type=int, default=999)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--n-source", type=int, default=2000)
    parser.add_argument("--n-targets", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    all_rows: list[dict[str, object]] = []
    conds = conditions(args.a, args.n_source, tuple(args.n_targets))
    for cond in conds:
        print(f"[straddle] running {cond.label} ...", flush=True)
        all_rows.extend(run_condition(cond, args.mc, args.boot, args.alpha, rng))

    raw = pd.DataFrame(all_rows)
    raw.to_csv(args.out / "straddle_raw.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(args.out / "straddle_summary.csv", index=False)

    manifest = {
        "script": Path(__file__).name,
        "mc": args.mc,
        "boot": args.boot,
        "alpha": args.alpha,
        "a": args.a,
        "n_source": args.n_source,
        "n_targets": list(args.n_targets),
        "seed": args.seed,
        "conditions": [asdict(c) for c in conds],
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))
    print(f"[straddle] done in {manifest['runtime_seconds']}s -> {args.out}")


if __name__ == "__main__":
    main()
