#!/usr/bin/env python3
"""
Plot spot instance availability curves per zone with filled areas.

Uses the format, regions, colors from plot_availability.py,
but draws line charts with filled areas under the curves.
"""

import json
import pathlib
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

from sky_spot.figure_values import write_values

###############

RESEARCH = 1  # For research presentation.
# RESEARCH = 0  # For paper pdf.

font_size = 12  # Manual fix for 0.5-full-col figures.
TEXT_USETEX = bool(1 - RESEARCH)

# Get from LaTeX using "The column width is: \the\columnwidth"
vldb_col_width_pt = 240
icml_col_width_pt = 234.8775
sigmod20_col_width_pt = 241.14749
vldb21_col_width_pt = 241.14749
nsdi23_col_width_pt = 241.02039

fig_width_pt = nsdi23_col_width_pt

inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio


def FigWidth(pt):
    return pt * inches_per_pt  # width in inches


fig_width = FigWidth(fig_width_pt)  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]


def InitMatplotlib(font_size, title_size=9):
    print("use_tex", TEXT_USETEX, "\nfont_size", font_size, "\ntitle_size", title_size)
    params = {
        "backend": "ps",
        "figure.figsize": fig_size,
        "text.usetex": TEXT_USETEX,
        "text.latex.preamble": "\n".join(
            [
                r"\usepackage{siunitx}",
                r"\sisetup{detect-all}",
                r"\usepackage{helvet}",
                r"\usepackage{sansmath}",
                r"\sansmath",
            ]
        ),
        "axes.titlesize": title_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "font.size": font_size,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "0.1",
        "legend.shadow": False,
        "legend.frameon": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "pdf.fonttype": 42,
        "lines.linewidth": 1,
        "xtick.bottom": False,
        "xtick.top": False,
        "ytick.left": False,
        "ytick.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.axisbelow": True,
    }

    plt.style.use("seaborn-v0_8-colorblind")
    plt.rcParams.update(params)


# Regions to plot (selected 8 regions from H100-16 collection)
REGIONS = [
    "asia-south2-b",
    "asia-southeast1-b",
    "asia-southeast1-c",
    "europe-west1-c",
    "us-central1-a",
    "us-east4-b",
    "us-west1-a",
    "us-west1-b",
]

# Data directory
DATA_DIR = pathlib.Path("data/h100_16_runs")

# Availability threshold: instances_created < THRESHOLD → unavailable
AVAILABILITY_THRESHOLD = 16

# Colors for each region
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def with_suffix(output_path: str, suffix: str) -> str:
    path = pathlib.Path(output_path)
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def lighten_color(color: str, amount: float = 0.5) -> str:
    """
    Lighten a color by blending it with white.

    Args:
        color: Hex color string (e.g., '#ff0000')
        amount: Amount to lighten (0=original, 1=white)

    Returns:
        Lightened hex color string
    """
    rgb = mcolors.to_rgb(color)
    # Blend with white
    lightened = tuple(c + (1 - c) * amount for c in rgb)
    return mcolors.to_hex(lightened)


def load_runs_data(
    data_dir: pathlib.Path,
    regions: list[str] | None = None,
    threshold: int = AVAILABILITY_THRESHOLD,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """
    Load data from runs/*.json files.

    Returns:
        Dictionary mapping region to {'timestamps': [...], 'instances_created': [...]}
    """
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    # Filter by date range if specified
    if start_date or end_date:
        filtered_files = []
        for f in json_files:
            file_date = f.name[:8]
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            filtered_files.append(f)
        json_files = filtered_files
        print(f'  Date filter: {start_date or "start"} to {end_date or "end"} -> {len(json_files)} files')

    # Collect all runs
    runs = []
    for f in json_files:
        with open(f) as fp:
            runs.append(json.load(fp))

    # Sort by scheduled_at
    runs = [r for r in runs if r.get("status") == "success"]
    runs.sort(key=lambda r: r.get("scheduled_at", ""))

    if not runs:
        raise ValueError("No successful runs found")

    if regions is None:
        regions = sorted({region for run in runs for region in run.get("zone_results", {}).keys()})

    # Extract data per region
    region_data = {}
    for region in regions:
        timestamps = []
        instances_created = []

        for run in runs:
            try:
                timestamp = datetime.fromisoformat(run["scheduled_at"].replace("Z", "+00:00"))
                timestamp = timestamp.replace(tzinfo=None)
            except (KeyError, ValueError):
                continue

            zone_results = run.get("zone_results", {})
            if region not in zone_results:
                continue

            result = zone_results[region]
            created = result.get("instances_created", 0)

            timestamps.append(timestamp)
            instances_created.append(created)

        if timestamps:
            region_data[region] = {
                "timestamps": timestamps,
                "instances_created": instances_created,
            }

    return region_data


def load_avg_price_data(data_dir: pathlib.Path, regions: list) -> dict:
    data = {}
    for region in regions:
        price_file = data_dir / f"{region}_h100_16" / "full.json"
        if not price_file.exists():
            raise FileNotFoundError(f"Price file not found: {price_file}")
        with open(price_file) as fp:
            price_data = json.load(fp)
            price = price_data["metadata"]["price_info"]["price"]
            data[region] = price
    return data


def plot_availability_filled(
    output_path: str | None = None,
    show_plot: bool = True,
    data_dir: pathlib.Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    price_data_dir: pathlib.Path | None = None,
    all_regions: bool = False,
) -> None:
    """
    Create the availability plot with filled areas for all specified regions.
    Single subplot with all regions stacked vertically (one row per region).
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # Initialize matplotlib with research-grade settings
    InitMatplotlib(font_size=12, title_size=14)

    # Load data
    display_regions = None if all_regions else REGIONS

    # Load all regions from runs data
    print(f"Loading runs data from {data_dir}...")
    try:
        all_region_data = load_runs_data(data_dir, display_regions, AVAILABILITY_THRESHOLD, start_date, end_date)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading runs data: {e}")
        return

    plot_regions = list(all_region_data.keys()) if all_regions else REGIONS
    num_regions = len(plot_regions)
    row_height = 0.2  # Height of each region row
    ylim = row_height * num_regions
    offset = row_height / 2

    fig_height = max(3, num_regions * 0.35)
    fig, ax = plt.subplots(1, 1, figsize=(10.5, fig_height), dpi=300)
    fig.set_size_inches(10.5, fig_height)

    print(f"Loading average price data from {price_data_dir}...")
    try:
        avg_price_data = load_avg_price_data(price_data_dir, plot_regions)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading average price data: {e}")
        return

    # Find global max for normalization
    global_max = 16  # instances_requested is always 16

    value_rows: list[dict] = []
    for i, region in enumerate(plot_regions):
        if region not in all_region_data:
            print(f"  Warning: No data for {region}")
            continue

        data = all_region_data[region]
        color = COLORS[i % len(COLORS)]

        timestamps = data["timestamps"]
        instances = data["instances_created"]

        # Calculate y-position: region 0 at top, region N-1 at bottom
        center = ylim - row_height * i - offset
        half_height = row_height * 0.3  # Leave some gap between rows

        # Normalize instances to fit within the row height
        # instances range: 0 to global_max -> map to center-half_height to center+half_height
        normalized = [center - half_height + (v / global_max) * (2 * half_height) for v in instances]
        baseline = center - half_height

        # Fill the area under the curve (always)
        # Use separate facecolor (lighten=0.7) and edgecolor (lighten=0.3)
        face_color = lighten_color(color, amount=0.7)
        edge_color = lighten_color(color, amount=0.3)
        ax.fill_between(timestamps, baseline, normalized, facecolor=face_color, edgecolor=color, linewidth=0.75)

        # Only draw line segments where instances == 16 (full availability)
        # Group consecutive full-availability points
        df = pd.DataFrame({"time": timestamps, "instances": instances, "normalized": normalized})
        df["is_full"] = df["instances"] >= global_max
        df["group"] = (~df["is_full"]).cumsum()

        # Plot line segments only for full availability groups
        full_df = df[df["is_full"]]
        for group_id in full_df["group"].unique():
            group_data = full_df[full_df["group"] == group_id]
            ax.plot(group_data["time"], group_data["normalized"], linewidth=1.4, color=color, solid_capstyle="butt")

        # # draw a gray top line at full availability
        # full_avail_y = center + half_height
        # ax.hlines(full_avail_y, timestamps[0], timestamps[-1], colors='gray', linewidth=0.75, alpha=0.1)

        # draw a baseline for the region
        ax.hlines(baseline, timestamps[0], timestamps[-1], colors=color, linewidth=1.25)

        avail_pct = 100 * sum(1 for x in instances if x >= AVAILABILITY_THRESHOLD) / len(instances)
        print(f"  {region}: {len(instances)} points, {avail_pct:.1f}% available")

        # Everything this row draws, as numbers. The curve is vector geometry, so
        # a text-layer diff of the PDF cannot see a changed availability series --
        # only the y-axis label's price is text. mean/max instances summarise the
        # filled area, available_points the segments drawn at full availability.
        value_rows.append(
            {
                "region": region,
                "points": len(instances),
                "available_points": sum(1 for x in instances if x >= AVAILABILITY_THRESHOLD),
                "avail_pct": avail_pct,
                "mean_instances": float(np.mean(instances)),
                "max_instances": float(np.max(instances)),
                "avg_price_usd": float(avg_price_data[region]),
            }
        )

    # Set y-axis with region labels
    ax.set_ylim(0, ylim)
    # label_names = list(reversed(REGIONS))
    label_names = [f"{region}  (${avg_price_data[region]:.2f})" for region in reversed(plot_regions)]
    y_pos = np.arange(row_height - offset, row_height * num_regions, row_height)
    ax.set_yticks(y_pos, labels=label_names, horizontalalignment="right")
    ax.tick_params(axis="y", pad=10)

    # Format x-axis
    from matplotlib.dates import DateFormatter

    date_format = DateFormatter("%m/%d")
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(mdates.DayLocator())

    ax.set_xlabel("Timestamp", fontsize=15)

    # Set x-limits based on data
    all_timestamps = []
    for data in all_region_data.values():
        all_timestamps.extend(data["timestamps"])
    if all_timestamps:
        ax.set_xlim(min(all_timestamps), max(all_timestamps))

    # ax.grid(True, alpha=0.5, axis='y', linestyle='-', linewidth=1.25)

    plt.tight_layout()

    # Save and/or show
    if output_path:
        if all_regions:
            output_path = with_suffix(output_path, "_all")
        out_path = pathlib.Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        print(f"Figure saved to: {out_path}")
        if value_rows:
            write_values(
                out_path.with_suffix(".pdf"),
                pd.DataFrame(value_rows),
                ["region", "points", "available_points", "avail_pct",
                 "mean_instances", "max_instances", "avg_price_usd"],
            )

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot spot instance availability with filled areas for multiple regions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o", default="outputs/spot_availability/spot_availability_filled.png", help="Output file path"
    )
    parser.add_argument("--data", "-d", default=str(DATA_DIR), help=f"Path to data directory (default: {DATA_DIR})")
    parser.add_argument("--no-show", action="store_true", help="Don't show the plot interactively")
    parser.add_argument(
        "--start-date", default=None, help="Filter runs starting from this date (YYYYMMDD format, inclusive)"
    )
    parser.add_argument(
        "--end-date",
        default="20251010",
        help="Filter runs up to this date (YYYYMMDD format, inclusive, default: 20251010)",
    )
    parser.add_argument(
        "--price-data",
        default="data/converted_multi_region_aligned_h100_16_merged",
        help="Path to price data directory",
    )
    parser.add_argument(
        "--all-regions", action="store_true", help="Plot all regions discovered in the data, sorted alphabetically"
    )

    args = parser.parse_args()

    plot_availability_filled(
        output_path=args.output,
        show_plot=not args.no_show,
        data_dir=pathlib.Path(args.data),
        start_date=args.start_date,
        end_date=args.end_date,
        price_data_dir=pathlib.Path(args.price_data),
        all_regions=args.all_regions,
    )


if __name__ == "__main__":
    main()
