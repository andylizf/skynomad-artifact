#!/usr/bin/env python3
"""
Plot deadline sensitivity analysis using base_plot styling.

Usage:
    python research/plot_deadline_sensitivity.py [--input CSV_PATH] [--output OUTPUT_PATH]
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import pandas as pd

# Import base_plot styling
from sky_spot.figure_values import warn_missing_strategies, write_values
from base_plot import InitMatplotlib, OUR_METHOD_NAME


# ============================================================================
# PLOT CONFIGURATION
# ============================================================================

PLOT_EBAR = False  # True: show error bars, False: show lines only

# ============================================================================
# STRATEGY STYLE CONFIGURATION
# ============================================================================


@dataclass
class StrategyStyle:
    """Unified style configuration for a strategy."""

    label: str
    color: str
    jitter: float  # X-axis offset for decluttering
    linestyle: str = "-"
    linewidth: float = 2.0
    marker: str = "o"
    markersize: float = 7.5
    zorder: int = 1
    elinewidth: float = 0.3
    capsize: float = 0.0
    alpha: float = 0.6
    stroke_color: str | None = None
    stroke_width: float = 4.0


# color ref: https://harmonizer.evilmartians.com/#W1siMjAwIiwzMCwwLjEyLCIzMDAiLDUwLG51bGxdLFsicmVkIiwyNywieWVsbG93IiwxMDAsImdyZWVuIiwxNjAsImJsdWUiLDIzNSwicHVycGxlIiwyOTAsInBpbmsiLDM1MF0sWyJhcGNhIiwiZmdUb0JnIiwiZXZlbiIsIiNmZmYiLCIjMDAwIiwwLCJwMyJdXQ

# All strategy styles in plot order
STRATEGY_STYLES: dict[str, StrategyStyle] = {
    # Our method
    "unified_cost_model_risk": StrategyStyle(
        label=OUR_METHOD_NAME,
        color="#00ADF7",
        jitter=-0.075,
        linestyle="-",
        linewidth=2.5,
        markersize=8.0,
        zorder=5,
        elinewidth=1.0,
        capsize=3.0,
        alpha=1.0,
        # stroke_color="#FF786C",
        # stroke_width=2.0,
    ),
    # Oracle baselines
    "unified_cost_model_oracle": StrategyStyle(
        label=f"{OUR_METHOD_NAME}(o)",
        color="#FF786C",
        jitter=-0.15,
        linestyle=":",
        linewidth=1.5,
        marker="v",
        zorder=10,
        elinewidth=0.7,
        capsize=2.5,
        alpha=0.7,
        stroke_color="#FFB3A9",
    ),
    "multi_region_oracle_dp": StrategyStyle(
        label="Optimal",
        color="#00B976",
        jitter=-0.20,
        linestyle=":",
        linewidth=1.5,
        marker="^",
        zorder=8,
        elinewidth=0.7,
        capsize=2.5,
        alpha=0.8,
        stroke_color="#8BD7AE",
    ),
    # Single-region baseline
    "rc_cr_threshold": StrategyStyle(
        label="UP",
        color="#A691FF",
        jitter=0.15,
        marker=">",
        zorder=-1,
    ),
    # Multi-region baselines
    "multi_region_rc_cr_threshold_eager_failover": StrategyStyle(
        label="UP(S)",
        color="#808080",
        jitter=0.0,
        marker="<",
        zorder=3,
        elinewidth=0.7,
        capsize=2.5,
        alpha=0.7,
    ),
    "multi_region_availability_probe_simple": StrategyStyle(
        label="UP(A)",
        color="#BAA200",
        jitter=0.05,
        marker="X",
    ),
    "multi_region_probe_cost_ratio": StrategyStyle(
        label="UP(AP)",
        color="#F377B7",
        jitter=0.10,
        marker="d",
    ),
}

# Strategies to plot (in order)
STRATEGIES_TO_PLOT = list(STRATEGY_STYLES.keys())


def get_style(strategy: str) -> StrategyStyle:
    """Get style for a strategy, with fallback for unknown strategies."""
    return STRATEGY_STYLES.get(
        strategy,
        StrategyStyle(label=strategy, color="#333333", jitter=0.0),
    )


def load_and_filter_data(
    csv_path: Path,
    checkpoint_size: float = None,
    restart_overhead: float = None,
) -> pd.DataFrame:
    """Load CSV and filter to relevant strategies and parameters."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Convert cost to numeric
    df["cost_numeric"] = pd.to_numeric(df["cost"], errors="coerce")

    # Filter by checkpoint_size if specified
    if checkpoint_size is not None:
        df = df[df["checkpoint_size"] == checkpoint_size]
        print(f"Filtered to checkpoint_size={checkpoint_size}: {len(df)} rows")

    # Filter by restart_overhead if specified
    if restart_overhead is not None:
        df = df[df["restart_overhead"] == restart_overhead]
        print(f"Filtered to restart_overhead={restart_overhead}: {len(df)} rows")

    # Filter to strategies we want
    available_strategies = [
        s for s in STRATEGIES_TO_PLOT if s in df["strategy"].unique()
    ]
    warn_missing_strategies(
        df["strategy"].unique(), STRATEGIES_TO_PLOT, "plot_deadline_sensitivity.py"
    )

    # Build filtered dataframe
    parts = []
    for strat in available_strategies:
        if strat == "rc_cr_threshold":
            # Use single_region data (runs on each region separately)
            part = df[(df["strategy"] == strat) & (df["task_type"] == "single_region")]
        else:
            # Multi-region strategies
            part = df[(df["strategy"] == strat) & (df["task_type"] == "multi_region")]
        if not part.empty:
            parts.append(part)

    if parts:
        return pd.concat(parts, ignore_index=True)
    return df[df["strategy"].isin(available_strategies)]


def aggregate_by_deadline(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cost by strategy and deadline_ratio."""
    g = df.groupby(["strategy", "deadline_ratio"])

    agg_mean = g["cost_numeric"].mean().reset_index(name="mean")
    agg_std = g["cost_numeric"].std(ddof=1).reset_index(name="std")
    agg_cnt = g["cost_numeric"].count().reset_index(name="count")
    agg_p5 = g["cost_numeric"].quantile(0.05).reset_index(name="p5")
    agg_p95 = g["cost_numeric"].quantile(0.95).reset_index(name="p95")

    agg = (
        agg_mean.merge(agg_std, on=["strategy", "deadline_ratio"])
        .merge(agg_cnt, on=["strategy", "deadline_ratio"])
        .merge(agg_p5, on=["strategy", "deadline_ratio"])
        .merge(agg_p95, on=["strategy", "deadline_ratio"])
    )

    # Compute 95% CI
    agg["ci95"] = agg.apply(
        lambda r: (
            1.96 * (r["std"] / math.sqrt(r["count"]))
            if r["count"] > 1 and not pd.isna(r["std"])
            else 0.0
        ),
        axis=1,
    )

    # Compute P5-P95 range for error bars
    agg["err_low"] = agg["mean"] - agg["p5"]
    agg["err_high"] = agg["p95"] - agg["mean"]

    return agg


def plot_deadline_sensitivity(
    df: pd.DataFrame,
    output_path: Path,
    use_p5_p95: bool = False,
    title: str = None,
    checkpoint_size: float = None,
    restart_overhead: float = None,
):
    """Generate deadline sensitivity plot."""
    # Initialize matplotlib with paper styling
    InitMatplotlib(font_size=14, title_size=16)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))

    # Get strategies in desired order
    available_strategies = [
        s for s in STRATEGIES_TO_PLOT if s in df["strategy"].unique()
    ]

    for strat in available_strategies:
        d = df[df["strategy"] == strat].sort_values("deadline_ratio")
        style = get_style(strat)

        if use_p5_p95:
            yerr = [d["err_low"].values, d["err_high"].values]
        else:
            yerr = d["ci95"].values

        # Apply jitter to x-coordinates for decluttering
        # x_values = d["deadline_ratio"].values + style.jitter * 0.01  # Scale jitter for deadline ratio
        x_values = d["deadline_ratio"].values * (
            1 + style.jitter * 0.05
        )  # log scale adjustment
        y_values = d["mean"].values

        # Draw continuous stroke line first (for dotted lines)
        if style.stroke_color:
            ax.plot(
                x_values,
                y_values,
                linestyle="-",  # Solid line as background
                linewidth=style.stroke_width,
                color=style.stroke_color,
                alpha=style.alpha * 0.5,
                zorder=style.zorder - 10,
            )

        # Draw error bars or line based on PLOT_EBAR
        if PLOT_EBAR:
            ax.errorbar(
                x_values,
                y_values,
                yerr=yerr,
                linestyle=style.linestyle,
                linewidth=style.linewidth,
                marker=style.marker,
                markersize=style.markersize,
                color=style.color,
                markeredgecolor="white",
                markeredgewidth=style.linewidth * 0.5,
                alpha=style.alpha,
                zorder=style.zorder + 0.1,
                elinewidth=style.elinewidth,
                capsize=style.capsize,
                label=style.label,
            )
        else:
            ax.plot(
                x_values,
                y_values,
                linestyle=style.linestyle,
                linewidth=style.linewidth,
                marker=style.marker,
                markersize=style.markersize,
                color=style.color,
                markeredgecolor="white",
                markeredgewidth=style.linewidth * 0.5,
                alpha=style.alpha,
                zorder=style.zorder + 0.1,
                label=style.label,
            )

        # Draw markers on top
        ax.scatter(
            x_values,
            y_values,
            marker=style.marker,
            s=style.markersize**2,
            color=style.color,
            edgecolors="white",
            linewidths=style.linewidth * 0.5,
            alpha=style.alpha * 0.75,
            zorder=style.zorder + 10,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.95, 2.15)
    ax.set_xlabel("Deadline Ratio", fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(350, 5000)
    ax.set_ylabel("Cost ($)", fontsize=16)
    # Configure y-axis ticks with LogLocator
    # Major ticks at 1, 2, 5 series (standard log scale subdivisions)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1.0, 2.0, 5.0], numticks=15))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    # Minor ticks at remaining subdivisions (3, 4, 6, 7, 8, 9)
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=[3.0, 4.0, 6.0, 7.0, 8.0, 9.0], numticks=15)
    )
    ax.grid(True, alpha=0.4, linewidth=1.0, axis="y", which="major")
    ax.grid(True, alpha=0.2, linewidth=0.5, axis="y", which="minor")

    # X-axis ticks at every data point
    xticks = list(sorted(df["deadline_ratio"].unique().tolist()))
    xticks_l = [f"{x:.2f}" for x in xticks]
    ax.set_xticks(xticks, labels=xticks_l)
    ax.set_xticks([], minor=True)  # Disable minor ticks on log scale
    # rotate x-tick labels for better visibility
    plt.setp(ax.get_xticklabels(), rotation=45, ha="center")
    ax.tick_params(axis="x", pad=2)

    # Legend
    ax.legend(
        loc="lower right",
        frameon=True,
        edgecolor="none",
        framealpha=0.5,
        fontsize=13,
        ncol=3,
        bbox_to_anchor=(1.0, 0.9),
    )

    # # Title
    # if title:
    #     fig.suptitle(title, fontsize=14, fontweight="bold")
    # else:
    #     ckpt_str = f"{checkpoint_size:.0f}" if checkpoint_size is not None else "?"
    #     ro_str = f"{restart_overhead}" if restart_overhead is not None else "?"
    #     err_str = "P5-P95" if use_p5_p95 else "95% CI"
    #     subtitle = f"(checkpoint={ckpt_str}GB, restart={ro_str}h, mean ± {err_str})"
    #     fig.suptitle(subtitle, fontsize=12, y=0.92)

    # Fixed margins to ensure consistent plot area with region_scaling figure
    plt.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.18)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description="Plot deadline sensitivity analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("research/data/deadline_sensitivity.csv"),
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ablations/deadline_sensitivity.png"),
        help="Output plot path",
    )
    parser.add_argument(
        "--use-p5p95",
        action="store_true",
        help="Use P5-P95 instead of 95%% CI for error bars",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot",
    )
    parser.add_argument(
        "--max-deadline-ratio",
        type=lambda s: float(s) if s not in ("", "none", "None") else None,
        default=3.0,
        help="Drop rows at or above this deadline ratio. Default 3.0 reproduces "
             "the published figure's range; pass '' to plot every ratio in the CSV",
    )
    args = parser.parse_args()

    print(f"Loading data from: {args.input}")
    df = load_and_filter_data(args.input)
    # The published figure stops below ratio 3.0. Keep that as the default so the
    # figure reproduces, but make the cut explicit and announce what it removes.
    if args.max_deadline_ratio is not None:
        dropped = df[df["deadline_ratio"] >= args.max_deadline_ratio]
        if len(dropped):
            per_strategy = (
                dropped.groupby("strategy")["cost"].mean().round(2).to_dict()
            )
            print(
                f"NOTE: dropping {len(dropped)} row(s) with deadline_ratio >= "
                f"{args.max_deadline_ratio} (published figure's range). "
                f"Mean cost in the dropped range: {per_strategy}. "
                f"Pass --max-deadline-ratio '' to keep them."
            )
        df = df[df["deadline_ratio"] < args.max_deadline_ratio]
    print(f"Loaded {len(df)} rows")
    print(f"Strategies: {df['strategy'].unique().tolist()}")
    print(f"Deadline ratios: {sorted(df['deadline_ratio'].unique().tolist())}")

    # Extract metadata before aggregation
    checkpoint_size = (
        df["checkpoint_size"].iloc[0] if "checkpoint_size" in df.columns else None
    )
    restart_overhead = (
        df["restart_overhead"].iloc[0] if "restart_overhead" in df.columns else None
    )

    agg = aggregate_by_deadline(df)
    print(f"Aggregated to {len(agg)} data points")

    # Sidecar of the exact series drawn below, so artifact/verify_figs.py can
    # compare numbers rather than only the PDF's text layer.
    write_values(
        args.output,
        agg,
        ["strategy", "deadline_ratio", "mean", "std", "count", "ci95", "p5", "p95"],
    )

    plot_deadline_sensitivity(
        agg,
        args.output,
        use_p5_p95=args.use_p5p95,
        title=args.title,
        checkpoint_size=checkpoint_size,
        restart_overhead=restart_overhead,
    )


if __name__ == "__main__":
    main()
