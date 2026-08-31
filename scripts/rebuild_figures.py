#!/usr/bin/env python3
"""Rebuild the two data-reconstructable main figures from compact results.

Figure 3 is an archival output: its compact metadata does not contain the
bootstrap interval endpoints needed to redraw it.  ``verify_results.py`` checks
that the archived PDF and its run metadata are present and internally valid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "confidence-calibrated-mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import sem, t


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from plot_style import (  # noqa: E402
    METHOD_COLORS,
    METHOD_DASHES,
    METHOD_MARKERS,
    format_axis,
    setup_style,
)


OVERVIEW_COLUMNS = [
    "dataset",
    "candidate_count",
    "retained_count",
    "retained_fraction",
    "mean_difference",
    "ci95_low",
    "ci95_high",
    "two_pp_noninferiority",
]


def read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required compact result is missing: {path}")
    data = pd.read_csv(path)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return data


def one_method_row(data: pd.DataFrame, method: str, path: Path) -> pd.Series:
    row = data[data["method"] == method]
    if len(row) != 1:
        raise ValueError(f"expected one {method!r} row in {path}, found {len(row)}")
    return row.iloc[0]


def subject_differences(path: Path) -> np.ndarray:
    data = read_csv(path, {"subject", "method", "balanced_accuracy"})
    if "analysis" in data.columns:
        data = data[data["analysis"] == "primary"]
    per_subject = (
        data.groupby(["subject", "method"], as_index=False)["balanced_accuracy"]
        .mean()
        .pivot(index="subject", columns="method", values="balanced_accuracy")
    )
    missing = sorted({"all_sources", "pair_ref"} - set(per_subject.columns))
    if missing:
        raise ValueError(f"{path} is missing methods: {', '.join(missing)}")
    values = (per_subject["pair_ref"] - per_subject["all_sources"]).dropna().to_numpy(float)
    if values.size < 2:
        raise ValueError(f"{path} contains fewer than two paired subjects")
    return values


def _dataset_summary(
    results: Path,
    relative_path: str,
    *,
    analysis: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    path = results / relative_path
    data = read_csv(path, {"method", "mean_set_size"})
    if analysis is not None:
        if "analysis" not in data.columns:
            raise ValueError(f"{path} has no analysis column")
        data = data[data["analysis"] == analysis]
    return one_method_row(data, "all_sources", path), one_method_row(data, "pair_ref", path)


def compute_eeg_overview(results: Path) -> pd.DataFrame:
    ni_path = results / "eeg" / "noninferiority" / "summary.json"
    if not ni_path.is_file():
        raise FileNotFoundError(f"required compact result is missing: {ni_path}")
    noninferiority = json.loads(ni_path.read_text(encoding="utf-8"))

    specifications = [
        (
            "Ma2020",
            "eeg/ma2020/downstream_summary.csv",
            None,
            "ma2020",
        ),
        (
            "Stieger2021",
            "eeg/stieger2021/downstream_summary.csv",
            None,
            "stieger2021",
        ),
        (
            "Zhou2020",
            "eeg/zhou2020/downstream_summary.csv",
            "primary",
            "zhou2020",
        ),
        (
            "BNCI2014_004",
            "eeg/bnci2014_004/downstream_summary.csv",
            "primary",
            None,
        ),
    ]

    rows: list[dict[str, object]] = []
    for dataset, summary_file, analysis, ni_key in specifications:
        all_sources, pair_ref = _dataset_summary(
            results, summary_file, analysis=analysis
        )
        if ni_key is not None:
            if ni_key not in noninferiority:
                raise KeyError(f"{ni_path} has no {ni_key!r} result")
            inference = noninferiority[ni_key]
            required = {
                "mean_diff",
                "ci95_low",
                "ci95_high",
                "noninferior_at_delta",
            }
            missing = sorted(required - set(inference))
            if missing:
                raise ValueError(f"{ni_path}:{ni_key} is missing keys: {', '.join(missing)}")
            mean_difference = float(inference["mean_diff"])
            ci_low = float(inference["ci95_low"])
            ci_high = float(inference["ci95_high"])
            ni_status = (
                "established"
                if bool(inference["noninferior_at_delta"])
                else "not established"
            )
        else:
            values = subject_differences(
                results / "eeg" / "bnci2014_004" / "downstream_by_subject.csv"
            )
            mean_difference = float(values.mean())
            half_width = float(t.ppf(0.975, values.size - 1) * sem(values))
            ci_low = mean_difference - half_width
            ci_high = mean_difference + half_width
            ni_status = "not tested"

        candidate_count = float(all_sources["mean_set_size"])
        retained_count = float(pair_ref["mean_set_size"])
        rows.append(
            {
                "dataset": dataset,
                "candidate_count": candidate_count,
                "retained_count": retained_count,
                "retained_fraction": retained_count / candidate_count,
                "mean_difference": mean_difference,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "two_pp_noninferiority": ni_status,
            }
        )
    return pd.DataFrame(rows, columns=OVERVIEW_COLUMNS)


def verify_reconstructed_overview(results: Path, computed: pd.DataFrame) -> None:
    expected_path = results / "figure_data" / "eeg_overview.csv"
    expected = read_csv(expected_path, set(OVERVIEW_COLUMNS))[OVERVIEW_COLUMNS]
    pd.testing.assert_frame_equal(
        computed.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def plot_eeg_overview(data: pd.DataFrame, output: Path) -> None:
    setup_style()
    y = np.arange(len(data))
    pair_color = METHOD_COLORS["Pair/Ref"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.8, 3.15),
        sharey=True,
        gridspec_kw={"width_ratios": [0.9, 1.15]},
    )

    axes[0].barh(
        y, np.ones(len(data)), color="#F3F4F6", edgecolor="#D1D5DB", height=0.58
    )
    axes[0].barh(
        y, data["retained_fraction"], color=pair_color, edgecolor="#2F3437", height=0.58
    )
    for ypos, row in zip(y, data.itertuples(index=False)):
        axes[0].text(
            min(row.retained_fraction + 0.025, 0.86),
            ypos,
            f"{row.retained_count:.2f}/{row.candidate_count:.2f}",
            va="center",
            fontsize=7.5,
            color="#27272A",
        )
    axes[0].set_yticks(y, data["dataset"])
    axes[0].set_xlim(0, 1.02)
    axes[0].set_xlabel("Pair/Ref retained fraction")
    axes[0].set_title("(a) Source reduction")
    format_axis(axes[0], ygrid=False)
    axes[0].grid(True, axis="x")

    mean = data["mean_difference"].to_numpy(float)
    lower = data["ci95_low"].to_numpy(float)
    upper = data["ci95_high"].to_numpy(float)
    axes[1].errorbar(
        mean,
        y,
        xerr=np.vstack([mean - lower, upper - mean]),
        fmt=METHOD_MARKERS["Pair/Ref"],
        color=pair_color,
        ecolor=pair_color,
        markeredgecolor="#2F3437",
        markeredgewidth=0.6,
        capsize=3,
        linewidth=1.35,
        markersize=5.2,
    )
    axes[1].axvline(0.0, color="#333333", linewidth=1.0)
    axes[1].axvline(-0.02, color="#9CA3AF", linewidth=1.0, linestyle="--")
    axes[1].set_xlim(-0.041, 0.041)
    axes[1].set_xlabel("Balanced-accuracy difference vs all sources")
    axes[1].set_title("(b) Downstream performance")
    axes[1].tick_params(labelleft=False)
    format_axis(axes[1], ygrid=False)
    axes[1].grid(True, axis="x")
    axes[1].legend(
        handles=[
            Line2D([0], [0], color="#333333", lw=1.0, label="No difference"),
            Line2D(
                [0], [0], color="#9CA3AF", lw=1.0, ls="--", label="-2 pp NI margin"
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=7.2,
    )
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=1.1, rect=(0, 0.05, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_shared_target_evidence(results: Path, output: Path) -> None:
    setup_style()
    straddle = read_csv(
        results / "simulations" / "straddle_summary.csv",
        {"geometry", "n_target", "cover_shared", "cover_wrong"},
    )
    straddle = straddle[straddle["geometry"] == "straddle"].sort_values("n_target")
    target = read_csv(
        results / "simulations" / "target_limited_audit_summary.csv",
        {"scenario", "n_target", "n_source", "shared_ref_size", "wrong_ref_size"},
    )

    shared_label = "Shared-target Ref.-J"
    wrong_label = "Independent-target control"
    shared_color = METHOD_COLORS["Ref.-J"]
    wrong_color = METHOD_COLORS["Wrong target"]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(8.1, 2.75),
        gridspec_kw={"width_ratios": [0.9, 1.08, 1.08]},
    )

    (shared_line,) = axes[0].plot(
        straddle["n_target"],
        straddle["cover_shared"],
        marker=METHOD_MARKERS["Ref.-J"],
        color=shared_color,
        label=shared_label,
    )
    (wrong_line,) = axes[0].plot(
        straddle["n_target"],
        straddle["cover_wrong"],
        marker=METHOD_MARKERS["Wrong target"],
        color=wrong_color,
        label=wrong_label,
    )
    shared_line.set_dashes(METHOD_DASHES["Ref.-J"])
    wrong_line.set_dashes(METHOD_DASHES["Wrong target"])
    from scipy.stats import norm

    analytic_limit = 2 * norm.cdf(norm.ppf(0.975) / np.sqrt(2)) - 1
    axes[0].axhline(0.95, color="#333333", linewidth=0.9, linestyle="--")
    axes[0].axhline(analytic_limit, color="#9CA3AF", linewidth=0.9, linestyle=":")
    axes[0].text(
        0.98,
        0.956,
        "95% nominal",
        transform=axes[0].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6.9,
    )
    axes[0].text(
        0.98,
        analytic_limit + 0.005,
        "analytic limit 0.834",
        transform=axes[0].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6.9,
        color="#6B7280",
    )
    axes[0].set_xticks([50, 200])
    axes[0].set_ylim(0.80, 1.005)
    axes[0].set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    axes[0].set_xlabel(r"Target sample size $n_T$")
    axes[0].set_ylabel("Contrast coverage")
    axes[0].set_title("(a) Straddling coverage")
    format_axis(axes[0])

    ratio_order = [(20, 100), (20, 500), (50, 200), (50, 500)]
    x = np.arange(len(ratio_order))
    for axis, scenario, title in zip(
        axes[1:],
        ("target_limited_small", "target_limited_medium"),
        ("(b) Target-limited small", "(c) Target-limited medium"),
    ):
        subset = target[target["scenario"] == scenario].set_index(["n_target", "n_source"])
        missing_rows = [key for key in ratio_order if key not in subset.index]
        if missing_rows:
            raise ValueError(f"target-limited figure data lacks {scenario} rows: {missing_rows}")
        shared_values = [float(subset.loc[key, "shared_ref_size"]) for key in ratio_order]
        wrong_values = [float(subset.loc[key, "wrong_ref_size"]) for key in ratio_order]
        (line_shared,) = axis.plot(
            x,
            shared_values,
            marker=METHOD_MARKERS["Ref.-J"],
            color=shared_color,
            label=shared_label,
        )
        (line_wrong,) = axis.plot(
            x,
            wrong_values,
            marker=METHOD_MARKERS["Wrong target"],
            color=wrong_color,
            label=wrong_label,
        )
        line_shared.set_dashes(METHOD_DASHES["Ref.-J"])
        line_wrong.set_dashes(METHOD_DASHES["Wrong target"])
        axis.set_xticks(x, [f"{n_t}/{n_s}" for n_t, n_s in ratio_order])
        axis.set_xlabel(r"$n_T/n_S$")
        axis.set_title(title)
        axis.set_ylim(4.6, 8.15)
        format_axis(axis)
    axes[1].set_ylabel("Retained set size")
    axes[2].set_ylabel("")
    fig.legend(
        handles=[shared_line, wrong_line],
        labels=[shared_label, wrong_label],
        loc="upper center",
        bbox_to_anchor=(0.55, 1.05),
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "results",
        help="compact result root (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build" / "figures",
        help="figure output directory (default: build/figures)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = args.input_dir.resolve()
    output = args.output_dir.resolve()
    overview = compute_eeg_overview(results)
    verify_reconstructed_overview(results, overview)
    plot_shared_target_evidence(results, output / "fig_shared_target_evidence.pdf")
    plot_eeg_overview(overview, output / "fig_eeg_overview.pdf")
    print(f"[PASS] rebuilt Figure 1: {output / 'fig_shared_target_evidence.pdf'}")
    print(f"[PASS] rebuilt Figure 2: {output / 'fig_eeg_overview.pdf'}")
    print("[INFO] Figure 3 is an archival output and is not regenerated from compact data")


if __name__ == "__main__":
    main()
