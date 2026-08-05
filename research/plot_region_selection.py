#!/usr/bin/env python3
"""
Plot region selection analysis with grouped bar plot.

Usage:
    python research/plot_region_selection.py [--input CSV_PATH] [--output OUTPUT_PATH]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, ScalarFormatter
import numpy as np
import pandas as pd

# Import base_plot styling
from sky_spot.figure_values import warn_missing_strategies, write_values
from base_plot import InitMatplotlib, OUR_METHOD_NAME


# ============================================================================
# PLOT CONFIGURATION
# ============================================================================

# Error bar configuration (consistent for all strategies)
ERRORBAR_LINEWIDTH = 1.8
ERRORBAR_CAPSIZE = 4.0

# ============================================================================
# STRATEGY STYLE CONFIGURATION
# ============================================================================


@dataclass
class StrategyStyle:
    """Unified style configuration for a strategy."""

    label: str
    color: str
    color_light: str
    color_dark: str
    hatch: str = ""
    alpha: float = 1.0


# color ref: https://harmonizer.evilmartians.com/#W1siMjAwIiwyMCwwLjEyLCIzMDAiLDUwLG51bGwsIjQwMCIsODAsMC4xXSxbInJlZCIsMjcsInllbGxvdyIsMTAwLCJncmVlbiIsMTYwLCJibHVlIiwyMzUsInB1cnBsZSIsMjkwLCJwaW5rIiwzNTBdLFsiYXBjYSIsImZnVG9CZyIsImV2ZW4iLCIjZmZmIiwiIzAwMCIsMCwicDMiXV0

# All strategy styles in plot order
STRATEGY_STYLES: dict[str, StrategyStyle] = {
    # Our method
    "unified_cost_model_risk": StrategyStyle(
        label=OUR_METHOD_NAME,
        color="#00ADF7",
        color_light="#50BFF9",
        color_dark="#0F6992",
    ),
    # Oracle baselines
    "unified_cost_model_oracle": StrategyStyle(
        label=f"{OUR_METHOD_NAME}(o)",
        color="#FF786C",
        color_light="#FFCCC4",
        color_dark="#974D46",
        hatch="///",
    ),
    "multi_region_oracle_dp": StrategyStyle(
        label="Optimal",
        color="#00B976",
        color_light="#B3E4C8",
        color_dark="#19704B",
        hatch="\\\\\\",
    ),
    # Single-region baseline
    "rc_cr_threshold": StrategyStyle(
        label="UP",
        color="#A691FF",
        color_light="#D9D4FF",
        color_dark="#655A9A",
        hatch="--",
    ),
    # Multi-region baselines
    "multi_region_rc_cr_threshold_eager_failover": StrategyStyle(
        label="UP(S)",
        color="#808080",
        color_light="#D5D5D5",
        color_dark="#4D4D4D",
        hatch="..",
    ),
    "multi_region_availability_probe_simple": StrategyStyle(
        label="UP(A)",
        color="#BAA200",
        color_light="#E2DAAA",
        color_dark="#70630A",
        hatch="xxx",
    ),
    "multi_region_probe_cost_ratio": StrategyStyle(
        label="UP(AP)",
        color="#F377B7",
        color_light="#FECBE1",
        color_dark="#904D6E",
        hatch="++",
    ),
}

# Strategies to plot (in definition order)
STRATEGIES_TO_PLOT = list(STRATEGY_STYLES.keys())

# Scenario order
SCENARIO_ORDER = [
    "US Only (8)",
    "Europe Only (2)",
    "Asia Only (3)",
    "Global (13)",
]

SCENARIO_DISPLAY_MAP = {
    "US Only (8)": "US\n(8 regions)",
    "Europe Only (2)": "Europe\n(2 regions)",
    "Asia Only (3)": "Asia\n(3 regions)",
    "Global (13)": "Global\n(13 regions)",
}


def get_style(strategy: str) -> StrategyStyle:
    """Get style for a strategy, with fallback for unknown strategies."""
    return STRATEGY_STYLES.get(
        strategy,
        StrategyStyle(
            label=strategy, color="#333333", color_light="#999999", color_dark="#000000"
        ),
    )


def get_scenario_display(scenario: str) -> str:
    return SCENARIO_DISPLAY_MAP.get(scenario, scenario)


def load_and_filter_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV and filter to relevant strategies."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Convert cost to numeric
    df["cost_numeric"] = pd.to_numeric(df["cost"], errors="coerce")

    warn_missing_strategies(
        df["strategy"].unique(), STRATEGIES_TO_PLOT, "plot_region_selection.py"
    )

    # Build filtered dataframe
    parts = []
    for strat in STRATEGIES_TO_PLOT:
        if strat == "rc_cr_threshold":
            # Use single_region data, aggregate by scenario (average across regions)
            part = df[(df["strategy"] == strat) & (df["task_type"] == "single_region")]
        else:
            # Multi-region strategies
            part = df[(df["strategy"] == strat) & (df["task_type"] == "multi_region")]
        if not part.empty:
            parts.append(part)

    if parts:
        return pd.concat(parts, ignore_index=True)
    return df[df["strategy"].isin(STRATEGIES_TO_PLOT)]


def aggregate_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cost by strategy and scenario."""
    g = df.groupby(["strategy", "scenario_name"])

    agg_mean = g["cost_numeric"].mean().reset_index(name="mean")
    agg_std = g["cost_numeric"].std(ddof=1).reset_index(name="std")
    agg_cnt = g["cost_numeric"].count().reset_index(name="count")

    agg = agg_mean.merge(agg_std, on=["strategy", "scenario_name"]).merge(
        agg_cnt, on=["strategy", "scenario_name"]
    )

    # Compute 95% CI
    agg["ci95"] = agg.apply(
        lambda r: (
            1.96 * r["std"] / np.sqrt(r["count"])
            if r["count"] > 1 and not pd.isna(r["std"])
            else 0.0
        ),
        axis=1,
    )

    return agg


def plot_grouped_bars(
    df: pd.DataFrame,
    output_path: Path,
    title: str = None,
):
    """Generate grouped bar plot."""
    # Initialize matplotlib with paper styling
    InitMatplotlib(font_size=14, title_size=16)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.rcParams['hatch.linewidth'] = 1.5  # Hatch line width

    # Get scenarios and strategies in order
    scenarios = [s for s in SCENARIO_ORDER if s in df["scenario_name"].unique()]
    strategies = [s for s in STRATEGIES_TO_PLOT if s in df["strategy"].unique()]

    n_scenarios = len(scenarios)
    n_strategies = len(strategies)

    # Bar positioning
    bar_width = 0.112
    bar_spacing = 0.016
    x = np.arange(n_scenarios)

    # Skip Optimal bar - only show as horizontal line
    skip_bar_strategies = {"multi_region_oracle_dp"}
    bar_strategies = [s for s in strategies if s not in skip_bar_strategies]
    n_bar_strategies = len(bar_strategies)

    # Plot bars for each strategy (except Optimal)
    for i, strat in enumerate(bar_strategies):
        offset = (i - n_bar_strategies / 2 + 0.5) * (bar_width + bar_spacing)
        strat_data = df[df["strategy"] == strat]
        style = get_style(strat)

        means = []
        errors = []
        for scenario in scenarios:
            row = strat_data[strat_data["scenario_name"] == scenario]
            if not row.empty:
                means.append(row["mean"].iloc[0])
                errors.append(row["ci95"].iloc[0])
            else:
                means.append(0)
                errors.append(0)

        # All bars use same style (solid fill)
        ax.bar(
            x + offset,
            means,
            bar_width,
            label=style.label,
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

    # Customize axes
    ax.set_xlabel("Geographic Region Configuration", fontsize=16, labelpad=5)
    ax.set_ylabel("Cost ($)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([get_scenario_display(s) for s in scenarios], fontsize=13)
    ax.set_xlim([-0.5, n_scenarios - 0.5])

    # Log scale for y-axis
    ax.set_yscale("log")
    ax.set_ylim(bottom=500, top=3500)

    # Configure y-axis ticks with LogLocator
    # Major ticks at 1, 2, 5 series (standard log scale subdivisions)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1.0, 5.0], numticks=15))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    # Minor ticks at remaining subdivisions (3, 4, 6, 7, 8, 9)
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=[2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0], numticks=15)
    )
    # Only show labels for 3000, 4000 (not 600, 700, 800, 900)
    ax.yaxis.set_minor_formatter(
        FuncFormatter(lambda v, _: f"{int(v)}" if v >= 2000 and v <= 3000 else "")
    )

    ax.grid(True, alpha=0.5, linewidth=1.0, axis="y", which="major", zorder=0)
    ax.grid(True, alpha=0.3, linewidth=0.5, axis="y", which="minor", zorder=0)

    # Draw horizontal dashed line for optimal (multi_region_oracle_dp)
    optimal_data = df[df["strategy"] == "multi_region_oracle_dp"]
    optimal_style = get_style("multi_region_oracle_dp")
    for idx, scenario in enumerate(scenarios):
        row = optimal_data[optimal_data["scenario_name"] == scenario]
        if row.empty:
            continue
        optimal_val = row["mean"].iloc[0]
        line = ax.axhline(
            xmin=(idx + 0.02) / n_scenarios,
            xmax=(idx + 0.98) / n_scenarios,
            y=optimal_val,
            color=optimal_style.color_dark,
            linestyle="--",
            linewidth=1.5,
            zorder=5,
            alpha=0.9,
            label="Optimal" if idx == 0 else None,  # Only label once for legend
        )

        

    # Legend - put Optimal under SkyNomad(o)
    handles, labels = ax.get_legend_handles_labels()
    # Want layout (3 cols):
    # row1: SkyNomad, SkyNomad(o), Optimal
    # row2: UP, UP(S), UP(AP)
    # row3: UP(A)
    # But matplotlib fills row by row with ncol, so order should be:
    # [SkyNomad, SkyNomad(o), Optimal, UP, UP(S), UP(AP), UP(A)]
    target_order = [OUR_METHOD_NAME, f"{OUR_METHOD_NAME}(o)", "Optimal", "UP", "UP(S)", "UP(AP)", "UP(A)"]
    new_handles = []
    new_labels = []
    for name in target_order:
        if name in labels:
            idx = labels.index(name)
            new_handles.append(handles[idx])
            new_labels.append(labels[idx])
    ax.legend(
        new_handles,
        new_labels,
        loc="lower right",
        frameon=False,
        edgecolor="none",
        framealpha=0.5,
        fontsize=13,
        ncol=3,
        bbox_to_anchor=(1.0, 0.92),
    )

    # Fixed margins to ensure consistent plot area with checkpoint_sensitivity figure
    plt.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.18)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description="Plot region selection analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("research/data/region_selection.csv"),
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ablations/region_selection.png"),
        help="Output plot path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot",
    )
    args = parser.parse_args()

    print(f"Loading data from: {args.input}")
    df = load_and_filter_data(args.input)
    print(f"Loaded {len(df)} rows")
    print(f"Strategies: {df['strategy'].unique().tolist()}")
    print(f"Scenarios: {df['scenario_name'].unique().tolist()}")

    agg = aggregate_by_scenario(df)
    print(f"Aggregated to {len(agg)} data points")

    # Print summary table
    print("\nSummary (Mean Cost):")
    pivot = agg.pivot(index="scenario_name", columns="strategy", values="mean")
    pivot = pivot.reindex(index=SCENARIO_ORDER, columns=STRATEGIES_TO_PLOT)
    pivot.columns = [get_style(s).label for s in pivot.columns]
    print(pivot.round(0).to_string())

    # Sidecar of the exact series drawn below, so artifact/verify_figs.py can
    # compare numbers rather than only the PDF's text layer.
    write_values(
        args.output,
        agg,
        ["strategy", "scenario_name", "mean", "std", "count", "ci95"],
    )

    plot_grouped_bars(agg, args.output, title=args.title)


if __name__ == "__main__":
    main()
