"""Core simultaneous source-screening operations.

The functions in this module are data-agnostic. They operate on estimated
source discrepancies and bootstrap replicates and are shared by the simulation
and EEG entry points.
"""

from __future__ import annotations

import numpy as np


def ordered_pairs(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the indices of all ordered pairs among ``k`` sources."""

    jj, ll = np.where(~np.eye(k, dtype=bool))
    return jj.astype(int), ll.astype(int)


def build_system(
    delta_hat: np.ndarray,
    delta_boot: np.ndarray,
    alpha_c: float,
    alpha_d: float,
    alpha: float,
) -> dict[str, object]:
    """Build simultaneous component, contrast, and joint confidence bounds."""

    k = delta_hat.size
    jj, ll = ordered_pairs(k)
    d_hat = delta_hat[jj] - delta_hat[ll]
    d_boot = delta_boot[:, jj] - delta_boot[:, ll]
    se_c = np.maximum(delta_boot.std(axis=0, ddof=1), 1e-12)
    se_d = np.maximum(d_boot.std(axis=0, ddof=1), 1e-12)
    t_c = np.max(
        np.abs((delta_boot - delta_hat[None, :]) / se_c[None, :]), axis=1
    )
    t_d = np.max(np.abs((d_boot - d_hat[None, :]) / se_d[None, :]), axis=1)
    t_j = np.maximum(t_c, t_d)
    q_c = float(np.quantile(t_c, 1.0 - alpha_c))
    q_d = float(np.quantile(t_d, 1.0 - alpha_d))
    q_j = float(np.quantile(t_j, 1.0 - alpha))
    return {
        "jj": jj,
        "ll": ll,
        "q_c": q_c,
        "q_d": q_d,
        "q_j": q_j,
        "se_c": se_c,
        "se_d": se_d,
        "lower_c": delta_hat - q_c * se_c,
        "upper_c": delta_hat + q_c * se_c,
        "lower_d": d_hat - q_d * se_d,
        "lower_cj": delta_hat - q_j * se_c,
        "upper_cj": delta_hat + q_j * se_c,
        "lower_dj": d_hat - q_j * se_d,
    }


def rect_gate(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Retain sources whose lower limit does not exceed the best upper limit."""

    return np.flatnonzero(lower <= np.min(upper))


def pair_gate(k: int, jj: np.ndarray, lower_d: np.ndarray) -> np.ndarray:
    """Retain sources that are not excluded by any lower contrast bound."""

    keep = np.ones(k, dtype=bool)
    for source, lower in zip(jj, lower_d):
        if lower > 0:
            keep[source] = False
    return np.flatnonzero(keep)
