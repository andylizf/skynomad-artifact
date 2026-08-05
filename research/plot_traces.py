#!/usr/bin/env python3
"""
Grouped bar plot comparing strategy costs across V100 and H100.

Reads:
- research/data/v100.csv (V100 data)
- research/data/deadline_sensitivity.csv (H100 data, filtered to deadline_ratio=1.5)

Usage:
    # Default: single combined figure with legend and hatch patterns
    python research/plot_traces.py [--output OUTPUT_PATH]

    # Legacy mode: two separate figures with method names as x-axis labels
    python research/plot_traces.py --legacy [--output OUTPUT_PATH]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
import pandas as pd

# The strategy the V100 sweep registers for SkyNomad.
V100_ARM = "unified_cost_model_rate_ratio"

# Import base_plot styling
from sky_spot.figure_values import warn_missing_strategies, write_values
from base_plot import InitMatplotlib, OUR_METHOD_NAME, OUR_METHOD_NAME_DL

NAME_PARTS = OUR_METHOD_NAME_DL.split("\n")
OUR_METHOD_NAME_DL_OPT = NAME_PARTS[0] + "    \n" + NAME_PARTS[1]  # Add spaces for alignment

# ============================================================================
# PLOT CONFIGURATION
# ============================================================================

# Error bar configuration (consistent for all strategies)
ERRORBAR_LINEWIDTH = 2.0
ERRORBAR_CAPSIZE = 5.0

# ============================================================================
# STRATEGY STYLE CONFIGURATION (from plot_deadline_sensitivity.py)
# ============================================================================


@dataclass
class StrategyStyle:
    """Unified style configuration for a strategy."""

    label: str  # Multi-line label for legacy mode (x-axis)
    label_legend: str  # Single-line label for legend
    color: str
    color_light: str
    color_dark: str
    hatch: str = ""
    alpha: float = 1.0


# color ref: https://harmonizer.evilmartians.com/#W1siMjAwIiwyMCwwLjEyLCIzMDAiLDQ1LDAuMTI1LCI0MDAiLDgwLDAuMV0sWyJyZWQiLDI3LCJ5ZWxsb3ciLDEwMCwiZ3JlZW4iLDE2MCwiYmx1ZSIsMjM1LCJwdXJwbGUiLDI5MCwicGluayIsMzUwXSxbImFwY2EiLCJmZ1RvQmciLCJldmVuIiwiI2ZmZiIsIiMwMDAiLDAsInAzIl1d

STRATEGY_STYLES: dict[str, StrategyStyle] = {
    "unified_cost_model_risk": StrategyStyle(
        label=OUR_METHOD_NAME_DL,
        label_legend=OUR_METHOD_NAME,
        color="#00ADF7",
        color_light="#50BFF9",
        color_dark="#0F6992",
    ),
    "unified_cost_model_oracle": StrategyStyle(
        label=f"\n{OUR_METHOD_NAME_DL_OPT}(o)",
        label_legend=f"{OUR_METHOD_NAME}(o)",
        color="#F58F84",
        color_light="#FFCCC4",
        color_dark="#974D46",
        hatch="///",
    ),
    "multi_region_oracle_dp": StrategyStyle(
        label="\nOptimal",
        label_legend="Optimal",
        color="#54BF8B",
        color_light="#B3E4C8",
        color_dark="#19704B",
        hatch="\\\\\\",
    ),
    "rc_cr_threshold": StrategyStyle(
        label="UP",
        label_legend="UP",
        color="#ADA0F8",
        color_light="#D9D4FF",
        color_dark="#655A9A",
        hatch="--",
    ),
    "multi_region_rc_cr_threshold_eager_failover": StrategyStyle(
        label="UP(S)",
        label_legend="UP(S)",
        color="#a0a0a0",
        color_light="#D5D5D5",
        color_dark="#4D4D4D",
        hatch="..",
    ),
    "multi_region_availability_probe_simple": StrategyStyle(
        label="UP(A)",
        label_legend="UP(A)",
        color="#BDAC44",
        color_light="#E2DAAA",
        color_dark="#70630A",
        hatch="xxx",
    ),
    "multi_region_probe_cost_ratio": StrategyStyle(
        label="UP(AP)",
        label_legend="UP(AP)",
        color="#EC8EBC",
        color_light="#FECBE1",
        color_dark="#904D6E",
        hatch="++",
    ),
}

# Strategies to plot (in order)
STRATEGIES_TO_PLOT = [
    "unified_cost_model_risk",
    "unified_cost_model_oracle",
    "multi_region_oracle_dp",
    "rc_cr_threshold",
    "multi_region_rc_cr_threshold_eager_failover",
    "multi_region_availability_probe_simple",
    "multi_region_probe_cost_ratio",
]


def get_style(strategy: str) -> StrategyStyle:
    """Get style for a strategy, with fallback for unknown strategies."""
    return STRATEGY_STYLES.get(
        strategy,
        StrategyStyle(
            label=strategy,
            label_legend=strategy,
            color="#333333",
            color_light="#999999",
            color_dark="#000000",
        ),
    )


def load_data(v100_path: Path, h100_path: Path) -> pd.DataFrame:
    """Load and combine V100 and H100 data."""
    # Load V100 data
    df_v100 = pd.read_csv(v100_path, low_memory=False)
    df_v100["dataset"] = "V100"
    # Both panels of Figure 9 label their SkyNomad arm the same way, so the V100
    # arm borrows the H100 arm's style key for drawing. V100_ARM restores the
    # registered name in the sidecar, which records what was actually run.
    df_v100["strategy"] = df_v100["strategy"].replace(
        V100_ARM, "unified_cost_model_risk"
    )
    print(f"V100: {len(df_v100)} rows, deadline_ratios: {df_v100['deadline_ratio'].unique()}")

    # Load H100 data and filter to deadline_ratio=1.5
    df_h100 = pd.read_csv(h100_path, low_memory=False)
    df_h100 = df_h100[df_h100["deadline_ratio"] == 1.5]
    df_h100["dataset"] = "H100"
    print(f"H100 (deadline=1.5): {len(df_h100)} rows")

    # Combine
    df = pd.concat([df_v100, df_h100], ignore_index=True)
    df["cost_numeric"] = pd.to_numeric(df["cost"], errors="coerce")

    return df


def filter_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to relevant strategies."""
    available_strategies = [
        s for s in STRATEGIES_TO_PLOT if s in df["strategy"].unique()
    ]
    warn_missing_strategies(
        df["strategy"].unique(), STRATEGIES_TO_PLOT, "plot_traces.py"
    )

    parts = []
    for strat in available_strategies:
        if strat == "rc_cr_threshold":
            # Use single_region data
            part = df[(df["strategy"] == strat) & (df["task_type"] == "single_region")]
        else:
            # Multi-region strategies
            part = df[(df["strategy"] == strat) & (df["task_type"] == "multi_region")]
        if not part.empty:
            parts.append(part)

    if parts:
        return pd.concat(parts, ignore_index=True)
    return df[df["strategy"].isin(available_strategies)]


def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cost by strategy and dataset."""
    g = df.groupby(["strategy", "dataset"])

    agg_mean = g["cost_numeric"].mean().reset_index(name="mean")
    agg_std = g["cost_numeric"].std(ddof=1).reset_index(name="std")
    agg_cnt = g["cost_numeric"].count().reset_index(name="count")

    agg = agg_mean.merge(agg_std, on=["strategy", "dataset"]).merge(
        agg_cnt, on=["strategy", "dataset"]
    )

    # Compute standard error
    agg["se"] = agg["std"] / np.sqrt(agg["count"])

    return agg


def plot_grouped_bars_combined(agg: pd.DataFrame, output_path: Path):
    """Create a single figure with V100 and H100 as groups (like plot_region_selection.py)."""
    InitMatplotlib(font_size=12, title_size=14)
    plt.rcParams['hatch.linewidth'] = 1.5

    # Get strategies in order
    available_strategies = [
        s for s in STRATEGIES_TO_PLOT if s in agg["strategy"].unique()
    ]

    datasets = ["V100", "H100"]
    n_datasets = len(datasets)
    n_strategies = len(available_strategies)

    # Bar positioning (like plot_region_selection.py)
    bar_width = 0.116
    bar_spacing = 0.0125
    x = np.arange(n_datasets)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    # Plot bars for each strategy
    for i, strat in enumerate(available_strategies):
        offset = (i - n_strategies / 2 + 0.5) * (bar_width + bar_spacing)
        strat_data = agg[agg["strategy"] == strat]
        style = get_style(strat)

        means = []
        errors = []
        for dataset in datasets:
            row = strat_data[strat_data["dataset"] == dataset]
            if not row.empty:
                means.append(row["mean"].iloc[0])
                errors.append(row["se"].iloc[0])
            else:
                means.append(0)
                errors.append(0)

        ax.bar(
            x + offset,
            means,
            bar_width,
            label=style.label_legend,
            color=style.color_light,
            alpha=style.alpha,
            hatch=style.hatch,
            yerr=errors,
            error_kw={
                "elinewidth": ERRORBAR_LINEWIDTH,
                "capsize": ERRORBAR_CAPSIZE,
                "capthick": ERRORBAR_LINEWIDTH,
                "ecolor": style.color_dark,
                "alpha": 0.8,
            },
            edgecolor=style.color,
            linewidth=0.0,
        )

    # X-axis labels (datasets)
    ax.set_xlabel("Trace Dataset", fontsize=14, labelpad=5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_xlim([-0.5, n_datasets - 0.5])
    ax.tick_params(axis="x", pad=7)

    # Y-axis
    ax.set_ylabel("Cost ($)", fontsize=13)
    ax.set_yscale("log")
    ax.set_ylim(bottom=10, top=1900)

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1.0, 5.0], numticks=15))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    # Minor ticks at remaining subdivisions
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=[2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0], numticks=15)
    )
    ax.grid(True, alpha=0.5, linewidth=1.0, axis="y", which="major", zorder=0)
    ax.grid(True, alpha=0.3, linewidth=0.5, axis="y", which="minor", zorder=0)

    # Legend
    ax.legend(
        loc="lower right",
        frameon=False,
        edgecolor="none",
        framealpha=0.5,
        fontsize=11,
        ncol=3,
        bbox_to_anchor=(1.0, 0.9),
    )

    plt.subplots_adjust(left=0.17, right=0.99, top=0.81, bottom=0.18)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def plot_grouped_bars_legacy(agg: pd.DataFrame, output_path: Path):
    """Create two separate figures - one for V100, one for H100 (legacy mode)."""
    InitMatplotlib(font_size=12, title_size=14)

    # Get strategies in order
    available_strategies = [
        s for s in STRATEGIES_TO_PLOT if s in agg["strategy"].unique()
    ]

    # Strategies that will have bars (exclude Optimal - only shown as horizontal line)
    skip_bar_strategies = {"multi_region_oracle_dp"}
    bar_strategies = [s for s in available_strategies if s not in skip_bar_strategies]

    datasets = ["V100", "H100"]
    n_bar_strategies = len(bar_strategies)
    bar_width = 0.7
    # Add extra spacing between first and second bar
    x = np.array([0.2, 1.35, 2.6, 3.7, 4.7, 5.7][:n_bar_strategies])
    label_x = np.array([0, 1.2, 2.65, 3.75, 4.8, 5.8][:n_bar_strategies])

    # Fixed figure size for both
    fig_width = 3.8
    fig_height = 2.8

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        data = agg[agg["dataset"] == dataset]

        # Plot bars for each bar strategy
        for i, strat in enumerate(bar_strategies):
            style = get_style(strat)
            row = data[data["strategy"] == strat]

            if not row.empty:
                mean_val = row["mean"].values[0]
                se_val = row["se"].values[0]
            else:
                mean_val = 0
                se_val = 0

            ax.bar(
                x[i],
                mean_val,
                bar_width,
                color=style.color,
                alpha=style.alpha,
                yerr=se_val,
                error_kw={
                    "elinewidth": ERRORBAR_LINEWIDTH,
                    "capsize": ERRORBAR_CAPSIZE,
                    "capthick": ERRORBAR_LINEWIDTH,
                    "ecolor": style.color_dark,
                    "alpha": 1.0,
                },
                edgecolor="white",
                linewidth=0.5,
            )

        # X-axis labels (only for bar strategies)
        ax.set_xticks(label_x)
        ax.set_xticklabels(
            [get_style(s).label for s in bar_strategies],
            fontsize=12,
            rotation=45,
            ha="right",
            rotation_mode='anchor',
        )
        # Offset adjustments: SkyNomad=0, SkyNomad(o)=10, UP=-0.5, rest=0
        xtick_yoff = [0, 10, -0.5, 0, 0, 0]
        for tick, yoff in zip(ax.get_xticklabels(), xtick_yoff):
            tick.set_y(tick.get_position()[1] + (yoff + 0.5) / 72)

        # # Fixed x-axis range
        # ax.set_xlim(-0.5, 7.0)

        # Y-axis (linear scale for both)
        if dataset == "V100":
            ax.set_ylim(0, 80 * 1.01)
        else:
            ax.set_ylabel("Cost ($)", fontsize=14)
            ax.set_ylim(0, 1750 * 1.01)
            ax.set_yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750])

        # Grid
        ax.yaxis.grid(True, alpha=0.4, linewidth=0.8, which="major")
        ax.yaxis.grid(True, alpha=0.2, linewidth=0.5, which="minor")
        ax.set_axisbelow(True)

        # Draw horizontal line for Optimal - dashed
        optimal_row = data[data["strategy"] == "multi_region_oracle_dp"]
        if not optimal_row.empty:
            optimal_val = optimal_row["mean"].values[0]
            optimal_style = get_style("multi_region_oracle_dp")
            ax.axhline(
                y=optimal_val,
                color=optimal_style.color_dark,
                linestyle="--",
                linewidth=1.5,
                zorder=5,
                alpha=0.9,
                label="Optimal",
            )
            ax.legend(loc="upper right", frameon=False, fontsize=11, bbox_to_anchor=(1.0, 1.05))

        if dataset == "V100":
            plt.subplots_adjust(left=0.12, right=0.92, top=0.96, bottom=0.30)
        else:
            plt.subplots_adjust(left=0.20, right=1.00, top=0.96, bottom=0.30)

        # Save each figure separately
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path = output_path.parent / f"{output_path.stem}_{dataset.lower()}.png"
        fig.savefig(dataset_path, dpi=300)
        fig.savefig(dataset_path.with_suffix(".pdf"))
        plt.close(fig)

        print(f"Saved: {dataset_path}")
        print(f"Saved: {dataset_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(
        description="Grouped bar plot comparing V100 and H100 costs"
    )
    parser.add_argument(
        "--v100",
        type=Path,
        default=Path("research/data/v100.csv"),
        help="V100 CSV file path",
    )
    parser.add_argument(
        "--h100",
        type=Path,
        default=Path("research/data/deadline_sensitivity.csv"),
        help="H100 CSV file path (will filter to deadline_ratio=1.5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ablations/traces_comparison.png"),
        help="Output plot path",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy mode: two separate figures with method names as x-axis labels",
    )
    args = parser.parse_args()

    print(f"Loading V100 data from: {args.v100}")
    print(f"Loading H100 data from: {args.h100}")

    df = load_data(args.v100, args.h100)
    df = filter_strategies(df)
    print(f"Filtered to {len(df)} rows")
    print(f"Strategies: {df['strategy'].unique().tolist()}")

    agg = aggregate_data(df)
    print(f"Aggregated to {len(agg)} data points")
    print(agg[["strategy", "dataset", "mean", "std", "count"]])

    # Sidecars of the exact series drawn below, so artifact/verify_figs.py can
    # compare numbers rather than only the PDF's text layer. Legacy mode emits
    # one figure per dataset, so it gets one sidecar per dataset.
    value_columns = ["strategy", "dataset", "mean", "std", "count", "se"]
    agg_values = agg.copy()
    v100_arm = (agg_values["dataset"] == "V100") & (
        agg_values["strategy"] == "unified_cost_model_risk"
    )
    agg_values.loc[v100_arm, "strategy"] = V100_ARM
    if args.legacy:
        for dataset in ("V100", "H100"):
            part = agg_values[agg_values["dataset"] == dataset]
            if part.empty:
                continue
            per_dataset = args.output.parent / (
                f"{args.output.stem}_{dataset.lower()}{args.output.suffix}"
            )
            write_values(per_dataset, part, value_columns)
    else:
        write_values(args.output, agg_values, value_columns)

    if args.legacy:
        print("Using legacy mode (two separate figures)")
        plot_grouped_bars_legacy(agg, args.output)
    else:
        print("Using combined mode (single figure with legend)")
        plot_grouped_bars_combined(agg, args.output)


if __name__ == "__main__":
    main()
