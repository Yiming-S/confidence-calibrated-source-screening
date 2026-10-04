"""Regression checks for the BSPC MCS-style comparison entry point."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from ci_gate_bspc_mcs_comparison import mcs_style_stepdown
from ci_gate_extended_baselines_sim import mcs_stepdown as established_mcs_stepdown


class MCSStyleParityTest(unittest.TestCase):
    def test_matches_established_implementation_on_fixed_inputs(self) -> None:
        configurations = [
            (4, 199, 1123),
            (8, 199, 9917),
            (14, 199, 42001),
        ]
        for source_count, bootstrap_count, seed in configurations:
            with self.subTest(source_count=source_count, seed=seed):
                rng = np.random.default_rng(seed)
                delta_hat = np.sort(rng.uniform(0.0, 1.5, size=source_count))
                common_target = rng.normal(0.0, 0.08, size=(bootstrap_count, 1))
                source_noise = rng.normal(
                    0.0, 0.12, size=(bootstrap_count, source_count)
                )
                delta_boot = delta_hat[None, :] + common_target + source_noise
                expected = established_mcs_stepdown(delta_hat, delta_boot, 0.05)
                observed = mcs_style_stepdown(delta_hat, delta_boot, 0.05)
                np.testing.assert_array_equal(observed, expected)


if __name__ == "__main__":
    unittest.main()
