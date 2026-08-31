#!/usr/bin/env python3
"""Shared plotting style for all CI-gate manuscript figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


METHOD_COLORS = {
    "Top-1": "#333333",
    "Top-3": "#6B7280",
    "Threshold": "#8B5CF6",
    "EB rule": "#A16207",
    "MCS stepdown": "#CC79A7",
    "MCB single-step": "#CC79A7",
    "All sources": "#71717A",
    "Target only": "#9CA3AF",
    "Rect.": "#0072B2",
    "Pair/Ref": "#D55E00",
    "Pair": "#D55E00",
    "Ref.": "#009E73",
    "Rect.-J": "#0072B2",
    "Pair-J": "#D55E00",
    "Ref.-J": "#009E73",
    "Wrong target": "#BDBDBD",
}

METHOD_MARKERS = {
    "Top-1": "^",
    "Top-3": "v",
    "Threshold": "D",
    "EB rule": "P",
    "MCS stepdown": "^",
    "MCB single-step": "^",
    "All sources": "X",
    "Target only": "P",
    "Rect.": "o",
    "Pair/Ref": "s",
    "Pair": "s",
    "Ref.": "D",
    "Rect.-J": "o",
    "Pair-J": "s",
    "Ref.-J": "D",
    "Wrong target": "X",
}

METHOD_DASHES = {
    "Top-1": (2, 2),
    "Top-3": (1.5, 1.5),
    "Threshold": (4, 1.5),
    "EB rule": (6, 1.5),
    "MCS stepdown": (5, 1.5, 1, 1.5),
    "MCB single-step": (5, 1.5, 1, 1.5),
    "All sources": "",
    "Target only": (1, 1.2),
    "Rect.": "",
    "Pair/Ref": (4, 1.5),
    "Pair": (4, 1.5),
    "Ref.": (1.5, 1),
    "Rect.-J": "",
    "Pair-J": (4, 1.5),
    "Ref.-J": (1.5, 1),
    "Wrong target": (1, 1.2),
}

STATUS_COLORS = {
    "Retained": "#009E73",
    "Excluded": "#6B7280",
}

BAR_HATCHES = ["", "///", "\\\\\\", "...", "xx", "--"]


def setup_style() -> None:
    """Apply the manuscript-wide font, line, grid, and export settings."""
    sns.set_theme(style="ticks", context="paper", font="DejaVu Sans", font_scale=1.0)
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "axes.titleweight": "regular",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#3F3F46",
            "axes.linewidth": 0.8,
            "xtick.color": "#27272A",
            "ytick.color": "#27272A",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.9,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "legend.title_fontsize": 8.0,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.4,
            "patch.edgecolor": "#2F3437",
            "figure.dpi": 150,
            "savefig.facecolor": "white",
        }
    )


def format_axis(ax: plt.Axes, *, ygrid: bool = True) -> None:
    """Apply the shared axis treatment without changing data limits."""
    if ygrid:
        ax.grid(True, axis="y")
    else:
        ax.grid(False, axis="y")
    ax.grid(False, axis="x")
    ax.spines["left"].set_color("#3F3F46")
    ax.spines["bottom"].set_color("#3F3F46")
    ax.tick_params(length=3, pad=2)
