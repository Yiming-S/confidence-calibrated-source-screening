"""Figure reconstruction must validate, not replace, its released inputs."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rebuild_figures  # noqa: E402


class FigureRebuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.inputs = self.root / "results"
        shutil.copytree(ROOT / "results", self.inputs)
        self.args = argparse.Namespace(
            input_dir=self.inputs, output_dir=self.root / "rebuilt"
        )

    def run_without_rendering(self) -> None:
        with (
            mock.patch.object(rebuild_figures, "parse_args", return_value=self.args),
            mock.patch.object(rebuild_figures, "plot_shared_target_evidence"),
            mock.patch.object(rebuild_figures, "plot_eeg_overview"),
        ):
            rebuild_figures.main()

    def test_reconstruction_preserves_every_released_input(self) -> None:
        before = {
            path.relative_to(self.inputs): path.read_bytes()
            for path in self.inputs.rglob("*") if path.is_file()
        }
        self.run_without_rendering()
        after = {
            path.relative_to(self.inputs): path.read_bytes()
            for path in self.inputs.rglob("*") if path.is_file()
        }
        self.assertEqual(before, after)

    def test_corrupted_overview_is_rejected_without_overwriting_it(self) -> None:
        path = self.inputs / "figure_data" / "eeg_overview.csv"
        data = rebuild_figures.pd.read_csv(path)
        data.loc[0, "retained_count"] += 1.0
        data.to_csv(path, index=False)
        corrupted = path.read_bytes()
        with self.assertRaises(AssertionError):
            self.run_without_rendering()
        self.assertEqual(path.read_bytes(), corrupted)

    def test_all_intervals_and_reference_lines_fit_inside_the_axis(self) -> None:
        data = rebuild_figures.compute_eeg_overview(self.inputs)
        for extension in (0.0, 0.1):
            with self.subTest(interval_extension=extension):
                displayed = data.copy()
                displayed.loc[0, "ci95_low"] -= extension
                displayed.loc[2, "ci95_high"] += extension
                with mock.patch.object(rebuild_figures.plt, "close"):
                    rebuild_figures.plot_eeg_overview(displayed, self.root / "overview.pdf")
                    figure = rebuild_figures.plt.gcf()
                low, high = figure.axes[1].get_xlim()
                self.assertLess(low, min(-0.02, float(displayed.ci95_low.min())))
                self.assertGreater(high, max(0.0, float(displayed.ci95_high.max())))
                rebuild_figures.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
