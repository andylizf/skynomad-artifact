from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from base_plot import InitMatplotlib, OUR_METHOD_NAME_DL

InitMatplotlib(font_size=12, title_size=14)

# Read CSV data
df = pd.read_csv("research/real.csv", skipinitialspace=True)
df = df.replace("xx", np.nan)

# Convert numeric columns
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill NaN with 0 for breakdown columns
breakdown_cols = [
    c for c in df.columns if "-probe" in c or "-egress" in c or "-OD" in c
]
for col in breakdown_cols:
    df[col] = df[col].fillna(0)

# ============================================================================
# STRATEGY STYLE CONFIGURATION
# ============================================================================


@dataclass
class StrategyStyle:
    """Unified style configuration for a strategy."""

    label: str  # Short label for x-axis (e.g., "US", "EU", "AP")
    color: str
    color_light: str
    color_mid: str
    color_dark: str
    group: Optional[str] = None  # Group name (e.g., "UP", "ASM"), None for standalone
    alpha: float = 1.0
    monospace: bool = False  # Use monospace font for label


# color ref: https://harmonizer.evilmartians.com/#W1siMjAwIiwzMCwwLjA4LCIzMDAiLDUwLDAuMTIsIjM1MCIsNjUsMC4xMiwiNDAwIiw4MCwwLjA4XSxbInJlZCIsMTAsIm9yYW5nZSIsOTAsInllbGxvdyIsMTIwLCJncmVlbiIsMTUwLCJibHVlIiwyMzUsInB1cnBsZSIsMzEwLCJwaW5rIiwzNDBdLFsiYXBjYSIsImZnVG9CZyIsImV2ZW4iLCIjZmZmIiwiIzAwMCIsMCwicDMiXV0

# All strategy styles in plot order
STRATEGY_STYLES: dict[str, StrategyStyle] = {
    "ours": StrategyStyle(
        label=OUR_METHOD_NAME_DL,
        color="#0298D2",
        color_light= "#92CFF3",
        color_mid="#47ABE0",
        color_dark="#2D6988",
        alpha=1.0,
    ),
    "multibase": StrategyStyle(
        label="UP\n(S)",
        color="#808080",
        color_light= "#cccccc",
        color_mid="#a0a0a0",
        color_dark="#666666",
    ),
    "single-us": StrategyStyle(
        label="US",
        color="#9E72BE",
        color_light= "#DABCF1",
        color_mid="#BD90DE",
        color_dark="#745888",
        group="UP",
    ),
    "single-eu": StrategyStyle(
        label="EU",
        color="#B86A9F",
        color_light= "#EFB7DB",
        color_mid="#D988BE",
        color_dark="#855374",
        group="UP",
    ),
    "single-ap": StrategyStyle(
        label="AP",
        color="#C56777",
        color_light= "#F9B5BE",
        color_mid="#E78595",
        color_dark="#8D525B",
        group="UP",
    ),
    "sage-us": StrategyStyle(
        label="US",
        color="#439358",
        color_light= "#A0D4A9",
        color_mid="#62B375",
        color_dark="#3D6D48",
        group="ASM",
    ),
    "sage-eu": StrategyStyle(
        label="EU",
        color="#798C2F",
        color_light= "#BFCE93",
        color_mid="#96AB4F",
        color_dark="#5C6831",
        group="ASM",
    ),
    "sage-ap": StrategyStyle(
        label="AP",
        color="#A08013",
        color_light= "#DBC68B",
        color_mid="#BF9E3E",
        color_dark="#756127",
        group="ASM",
    ),
}


def draw_brace(ax, x_start, x_end, y, height, text, fontsize=14):
    """Draw a horizontal curly brace with text below."""
    x_mid = (x_start + x_end) / 2

    # Draw the brace using a bezier curve
    # Left part
    brace_points_left = np.array(
        [
            [x_start, y],
            [x_start, y - height * 0.3],
            [x_mid - 0.05, y - height * 0.7],
            [x_mid, y - height],
        ]
    )
    # Right part
    brace_points_right = np.array(
        [
            [x_mid, y - height],
            [x_mid + 0.05, y - height * 0.7],
            [x_end, y - height * 0.3],
            [x_end, y],
        ]
    )

    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    # Create bezier curve codes
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]

    # Left curve
    path_left = MplPath(brace_points_left, codes)
    patch_left = PathPatch(
        path_left,
        facecolor="none",
        edgecolor="#333333",
        lw=1.2,
        clip_on=False,
        transform=ax.get_xaxis_transform(),
    )
    ax.add_patch(patch_left)

    # Right curve
    path_right = MplPath(brace_points_right, codes)
    patch_right = PathPatch(
        path_right,
        facecolor="none",
        edgecolor="#333333",
        lw=1.2,
        clip_on=False,
        transform=ax.get_xaxis_transform(),
    )
    ax.add_patch(patch_right)

    # Add text below brace
    ax.text(
        x_mid,
        y - height - 0.02,
        text,
        ha="center",
        va="top",
        fontsize=fontsize,
        transform=ax.get_xaxis_transform(),
    )


def lighten_color(color_hex: str, amount: float = 0.5) -> str:
    """Lighten a hex color by a given amount (0 to 1)."""
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color_hex]
    except KeyError:
        c = color_hex
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    if amount > 0:
        lightened = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    else:
        lightened = colorsys.hls_to_rgb(c[0], -amount * c[1], c[2])
    return mc.to_hex(lightened)


def get_style(strategy: str) -> StrategyStyle:
    """Get style for a strategy, with fallback for unknown strategies."""
    return STRATEGY_STYLES.get(
        strategy,
        StrategyStyle(
            label=strategy, color="#333333", color_light="#cccccc", color_mid="#999999", color_dark="#000000"
        ),
    )


# ============================================================================
# COST BREAKDOWN CONFIGURATION
# ============================================================================

# Hatch patterns for different cost components
HATCH_SPOT = ""  # Solid fill for spot cost
HATCH_OD = "///"  # Diagonal lines for on-demand cost
HATCH_EGRESS = "..."  # Dots for egress cost (denser)
HATCH_PROBE = "xx"  # Cross pattern for probe cost


def get_cost_breakdown(df_row: pd.Series, strategy: str) -> dict[str, float]:
    """
    Get cost breakdown for a strategy.
    Returns dict with keys: 'spot', 'od', 'egress', 'probe'
    """
    total = df_row[strategy]
    if pd.isna(total):
        return {"spot": 0, "od": 0, "egress": 0, "probe": 0}

    breakdown = {"spot": 0, "od": 0, "egress": 0, "probe": 0}

    # Get On-Demand cost if available
    od_col = f"{strategy}-OD"
    if od_col in df_row.index:
        breakdown["od"] = df_row[od_col]

    # Get egress cost if available
    egress_col = f"{strategy}-egress"
    if egress_col in df_row.index:
        breakdown["egress"] = df_row[egress_col]

    # Get probe cost if available (only for 'ours')
    probe_col = f"{strategy}-probe"
    if probe_col in df_row.index:
        breakdown["probe"] = df_row[probe_col]

    # Spot cost = total - other components
    breakdown["spot"] = (
        total - breakdown["od"] - breakdown["egress"] - breakdown["probe"]
    )

    return breakdown


# GPU types (A100 in the middle)
gpu_types = ["L4", "A100", "A10G"]

plot_ylims = {
    "L4": (0, 250 * 1.01),
    "A100": (0, 1200 * 1.01),
    "A10G": (0, 350 * 1.01),
}

# Strategies in STRATEGY_STYLES order (only those present in CSV)
csv_columns = set(df.columns) - {"gpu"}
strategies = [s for s in STRATEGY_STYLES.keys() if s in csv_columns]

# ============================================================================
# BAR POSITIONING WITH GROUPS
# ============================================================================

# Bar layout parameters
bar_width = 0.8
intra_group_gap = 0.15  # Gap between bars within a group (smaller)
inter_group_gap = 0.6  # Gap between groups or standalone bars (larger)


def compute_bar_positions(
    strategies: list[str],
) -> tuple[list[float], dict[str, tuple[float, float]]]:
    """
    Compute x positions for bars considering grouping.
    Returns: (positions, group_spans) where group_spans maps group_name -> (x_start, x_end)
    """
    positions = []
    group_spans: dict[str, tuple[float, float]] = {}
    current_x = 0.0
    current_group = None
    group_start_x = None

    for i, strat in enumerate(strategies):
        style = get_style(strat)
        group = style.group

        if i == 0:
            # First bar
            positions.append(current_x)
            if group:
                current_group = group
                group_start_x = current_x - bar_width / 2
        else:
            prev_group = get_style(strategies[i - 1]).group

            if group and group == prev_group:
                # Same group: use smaller gap
                current_x += bar_width + intra_group_gap
            else:
                # Different group or standalone: use larger gap
                # First, close the previous group if any
                if prev_group and prev_group in [
                    get_style(s).group for s in strategies[:i]
                ]:
                    if prev_group not in group_spans:
                        prev_pos = positions[-1]
                        group_spans[prev_group] = (
                            group_start_x,
                            prev_pos + bar_width / 2,
                        )

                current_x += bar_width + inter_group_gap

                if group:
                    current_group = group
                    group_start_x = current_x - bar_width / 2

            positions.append(current_x)

    # Close the last group if any
    if current_group and current_group not in group_spans:
        group_spans[current_group] = (group_start_x, positions[-1] + bar_width / 2)

    return positions, group_spans


# Plot each GPU in its own figure
for gpu_idx, gpu in enumerate(gpu_types):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    plt.rcParams['hatch.linewidth'] = 2.0  # Hatch line width

    gpu_row = df[df["gpu"] == gpu].iloc[0]
    values = gpu_row[strategies].values

    # Compute bar positions with grouping
    x_positions, group_spans = compute_bar_positions(strategies)
    x = np.array(x_positions)

    # Create stacked bars with breakdown
    for i, (x_pos, strategy) in enumerate(zip(x, strategies)):
        style = get_style(strategy)
        breakdown = get_cost_breakdown(gpu_row, strategy)

        bottom = 0
        # Stack order: spot (bottom), od, egress (top)
        # Note: probe cost is merged into spot (too small to visualize)
        components = [
            (
                "spot",
                breakdown["spot"] + breakdown["probe"],
                HATCH_SPOT,
                style.color_mid,
            ),
            ("od", breakdown["od"], HATCH_OD, style.color),
            ("egress", breakdown["egress"], HATCH_EGRESS, style.color),
        ]

        for comp_name, comp_value, hatch, color in components:
            if comp_value > 0:
                bar = ax.bar(
                    x_pos,
                    comp_value,
                    bar_width,
                    bottom=bottom,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    hatch=hatch,
                    alpha=style.alpha,
                )
                bottom += comp_value

    # Add value labels on top of bars
    for x_pos, val, s in zip(x, values, strategies):
        if not np.isnan(val):
            ax.annotate(
                f"{val:.0f}",
                xy=(x_pos, val),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
                color=get_style(s).color_dark,
            )

    # X-axis labels (short labels for grouped items)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [get_style(s).label for s in strategies], rotation=0, ha="center"
    )
    ax.tick_params(axis="x", pad=5)

    # Style tick labels: gray and smaller for grouped items, black and larger for standalone
    for tick_label, s in zip(ax.get_xticklabels(), strategies):
        style = get_style(s)
        if style.group:
            # Grouped item: gray, smaller font
            tick_label.set_color("#666666")
            tick_label.set_fontsize(12)
        else:
            # Standalone item: black, larger font
            tick_label.set_color("#000000")
            tick_label.set_fontsize(14)
        # Apply monospace font if specified
        if style.monospace:
            tick_label.set_fontfamily("monospace")

    # Draw braces and group labels
    brace_y = -0.075  # Y position in axes coordinates (below x-axis)
    brace_height = 0.02
    for group_name, (x_start, x_end) in group_spans.items():
        draw_brace(ax, x_start, x_end, brace_y, brace_height, group_name, fontsize=14)

    # Y-axis
    if gpu_idx == 0:
        ax.set_ylabel("Cost ($)", fontsize=14)
    ax.set_ylim(plot_ylims[gpu])

    # X-axis limits
    ax.set_xlim(x[0] - bar_width, x[-1] + bar_width)

    # Grid
    ax.yaxis.grid(True, alpha=0.3, linewidth=1.0, which="major")
    ax.set_axisbelow(True)

    # Fixed margins to ensure consistent bar width and figure height across all subplots
    # Leave space at top for legend (consistent across all plots)
    plt.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.13)

    # Add legend for A100 (middle plot) at the top
    if gpu == "A100":
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#bbbbbb", edgecolor="white", label="Spot"),
            Patch(
                facecolor="#999999", edgecolor="white", hatch="///", label="On-Demand"
            ),
            Patch(facecolor="#999999", edgecolor="white", hatch="...", label="Egress"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
            fontsize=14,
            handlelength=1.5,
            handleheight=1.5,
        )

    out_path = Path("outputs/cost_comparison/")
    out_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path / f"cost_comparison_bar_{gpu}.pdf")
    plt.savefig(out_path / f"cost_comparison_bar_{gpu}.png", dpi=150)
    print(
        f"Saved to {out_path / f'cost_comparison_bar_{gpu}.pdf'} and {out_path / f'cost_comparison_bar_{gpu}.png'}"
    )
    plt.close()
