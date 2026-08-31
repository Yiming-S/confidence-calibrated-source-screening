"""Deterministic tests for the data-agnostic screening operations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from screening_core import build_system, ordered_pairs, pair_gate, rect_gate


class ScreeningCoreTests(unittest.TestCase):
    def test_ordered_pairs_are_complete_without_self_pairs(self) -> None:
        left, right = ordered_pairs(4)
        pairs = set(zip(left.tolist(), right.tolist()))
        self.assertEqual(len(pairs), 12)
        self.assertTrue(all(i != j for i, j in pairs))

    def test_rectangle_gate_uses_the_best_upper_limit(self) -> None:
        retained = rect_gate(
            np.array([0.10, 0.31, 0.80]),
            np.array([0.25, 0.45, 1.00]),
        )
        np.testing.assert_array_equal(retained, np.array([0]))

    def test_pair_gate_excludes_any_source_with_positive_lower_contrast(self) -> None:
        left, _ = ordered_pairs(3)
        lower = np.full(left.shape, -0.2)
        lower[np.flatnonzero(left == 2)[0]] = 0.1
        retained = pair_gate(3, left, lower)
        np.testing.assert_array_equal(retained, np.array([0, 1]))

    def test_system_shapes_and_joint_critical_value(self) -> None:
        rng = np.random.default_rng(20260831)
        estimate = np.array([0.1, 0.2, 0.4])
        bootstrap = estimate + rng.normal(scale=0.03, size=(500, 3))
        system = build_system(estimate, bootstrap, 0.025, 0.025, 0.05)
        self.assertEqual(np.asarray(system["lower_c"]).shape, (3,))
        self.assertEqual(np.asarray(system["lower_d"]).shape, (6,))
        self.assertGreaterEqual(float(system["q_j"]), 0.0)
        self.assertTrue(
            np.all(np.asarray(system["lower_cj"]) <= np.asarray(system["upper_cj"]))
        )


if __name__ == "__main__":
    unittest.main()
