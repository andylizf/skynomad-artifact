#!/usr/bin/env python3
"""
Plot boxplot of spot instance availability fractions for multiple regions.

This script creates a horizontal boxplot showing the distribution of availability
fractions (calculated over sliding windows) for each region.

Uses the same data format and regions as plot_availability.py.
"""

import json
import pathlib
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

###############

RESEARCH = 1  # For research presentation.
# RESEARCH = 0  # For paper pdf.

font_size = 12  # Manual fix for 0.5-full-col figures.
TEXT_USETEX = bool(1 - RESEARCH)

# Get from LaTeX using "The column width is: \the\columnwidth"
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


# Regions to plot (same as plot_availability.py)
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

# Data directory (relative to project root, not research directory)
DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "h100_16_runs"

# Availability threshold: instances_created < THRESHOLD → unavailable
AVAILABILITY_THRESHOLD = 16

# Window size for calculating spot fraction (in number of data points)
WINDOW_SIZE = 120

# Base colors for each major region prefix
BASE_COLORS = {
    "us": "#acb902",  # Yellow-green
    "europe": "#00b252",  # Green
    "asia": "#af74e4",  # Purple
}

# Classic matplotlib colors (previous style, same as plot_availability.py)
CLASSIC_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def with_suffix(output_path: str, suffix: str) -> str:
    path = pathlib.Path(output_path)
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def get_region_prefix(region: str) -> str:
    """Extract region prefix (us, europe, asia) from region name."""
    if region.startswith("us-"):
        return "us"
    elif region.startswith("europe-"):
        return "europe"
    elif region.startswith("asia-"):
        return "asia"
    return "us"  # Default


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
    if amount < 0:
        # Darken by blending with black
        lightened = tuple(c * (1 + amount) for c in rgb)
    return mcolors.to_hex(lightened)


def load_runs_data(
    data_dir: pathlib.Path,
    regions: list[str] | None = None,
    threshold: int = AVAILABILITY_THRESHOLD,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """
    Load data from runs/*.json files and convert to binary availability.

    Args:
        data_dir: Path to directory containing run JSON files
        regions: List of regions to extract
        threshold: instances_created < threshold → unavailable (default: 16)
        start_date: Filter runs starting from this date (YYYYMMDD format, inclusive)
        end_date: Filter runs up to this date (YYYYMMDD format, inclusive)

    Returns:
        Dictionary mapping region to {'data': [...], 'start_time': datetime, 'gap_seconds': int}
        where data is list of 0/1 (0=available, 1=preempted)
    """
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    # Filter by date range if specified
    if start_date or end_date:
        filtered_files = []
        for f in json_files:
            file_date = f.name[:8]  # Extract YYYYMMDD
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
        availability = []  # 0=available, 1=preempted (matching trace format)

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
            instances_created = result.get("instances_created", 0)

            timestamps.append(timestamp)
            availability.append(0 if instances_created >= threshold else 1)

        if timestamps:
            if len(timestamps) > 1:
                gaps = [(timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)]
                gap_seconds = int(np.median(gaps))
            else:
                gap_seconds = 600

            region_data[region] = {
                "data": availability,
                "start_time": timestamps[0],
                "gap_seconds": gap_seconds,
                "timestamps": timestamps,
            }

    return region_data


def get_spot_frac(avail_list: list, window_size: int) -> list:
    """
    Calculate spot availability fraction for sliding windows.

    Args:
        avail_list: List of 0/1 values (1=available, 0=unavailable)
        window_size: Size of the sliding window

    Returns:
        List of availability fractions (as percentages)
    """
    ret = []
    for i in range(0, len(avail_list), window_size):
        avail_list_sub = avail_list[i : i + window_size]
        if len(avail_list_sub) > 0:
            ret.append(sum(avail_list_sub) / len(avail_list_sub) * 100)
    return ret


def get_availability_lifetimes(avail_list: list, gap_seconds: int) -> list:
    """
    Calculate the duration of each continuous available period.

    Args:
        avail_list: List of 0/1 values (1=available, 0=unavailable)
        gap_seconds: Time interval between data points in seconds

    Returns:
        List of availability lifetimes in hours
    """
    lifetimes = []
    current_length = 0

    for val in avail_list:
        if val == 1:  # Available
            current_length += 1
        else:  # Unavailable
            if current_length > 0:
                # Convert from number of intervals to hours
                lifetime_hours = current_length * gap_seconds / 3600
                lifetimes.append(lifetime_hours)
                current_length = 0

    # Handle the last segment if it ends with availability
    if current_length > 0:
        lifetime_hours = current_length * gap_seconds / 3600
        lifetimes.append(lifetime_hours)

    return lifetimes


def plot_boxplot_figure(
    output_path: str | None = None,
    show_plot: bool = True,
    data_dir: pathlib.Path | None = None,
    threshold: int = AVAILABILITY_THRESHOLD,
    window_size: int = WINDOW_SIZE,
    start_date: str | None = None,
    end_date: str | None = None,
    color_scheme: str = "classic",
    all_regions: bool = False,
) -> None:
    """
    Create a horizontal boxplot of availability fractions for all regions.

    Args:
        output_path: Path to save the figure (optional)
        show_plot: Whether to display the plot interactively
        data_dir: Path to data directory (default: DATA_DIR)
        threshold: Availability threshold for runs data
        window_size: Window size for calculating spot fraction
        start_date: Filter runs starting from this date (YYYYMMDD format, inclusive)
        end_date: Filter runs up to this date (YYYYMMDD format, inclusive)
        color_scheme: 'classic' for matplotlib colors, 'prefix' for region-prefix colors
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # Initialize matplotlib
    InitMatplotlib(font_size=12, title_size=14)

    # Load data
    print(f"Loading runs data from {data_dir} (threshold={threshold})...")
    try:
        all_region_data = load_runs_data(data_dir, None if all_regions else REGIONS, threshold, start_date, end_date)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading runs data: {e}")
        return

    display_regions = list(all_region_data.keys()) if all_regions else REGIONS

    # Calculate availability lifetimes for each region (reversed order for bottom-to-top display)
    lifetime_all = []
    valid_regions = []

    for region in reversed(display_regions):
        if region not in all_region_data:
            print(f"  Warning: No data for {region}")
            continue

        trace_data = all_region_data[region]
        # Convert: 0=available, 1=preempted -> 1=available, 0=unavailable
        avail_list = [1 - x for x in trace_data["data"]]
        gap_seconds = trace_data.get("gap_seconds", 600)

        lifetimes = get_availability_lifetimes(avail_list, gap_seconds)
        lifetime_all.append(lifetimes)
        valid_regions.append(region)

        avg_lifetime = np.mean(lifetimes) if lifetimes else 0
        print(f"  {region}: {len(lifetimes)} periods, avg lifetime: {avg_lifetime:.1f} hours")

    if not lifetime_all:
        print("No valid data to plot")
        return

    # Create figure
    num_regions = len(valid_regions)
    fig_height_adjusted = max(3, num_regions * 0.35)
    fig, ax = plt.subplots(1, 1, figsize=(5.5, fig_height_adjusted), constrained_layout=True, dpi=300)

    # Create boxplot (patch_artist=True to fill boxes with color)
    bp = ax.boxplot(lifetime_all, vert=False, widths=0.5, patch_artist=True, flierprops=dict(marker=".", markersize=7))

    # Get colors for each region based on selected scheme
    # Note: valid_regions is in reversed order (bottom-to-top in y-axis)
    # We want top region to get first color, so reverse the color index
    num_valid = len(valid_regions)
    region_colors = []
    if color_scheme == "prefix":
        for region in valid_regions:
            prefix = get_region_prefix(region)
            region_colors.append(BASE_COLORS.get(prefix, BASE_COLORS["us"]))
    else:  # 'classic' - assign colors top-to-bottom (matching plot_availability.py)
        for i in range(num_valid):
            # valid_regions[0] is at bottom (y=1), valid_regions[-1] is at top (y=n)
            # Top region should get CLASSIC_COLORS[0]
            color_idx = (num_valid - 1 - i) % len(CLASSIC_COLORS)
            region_colors.append(CLASSIC_COLORS[color_idx])

    linewidth = 1.25

    # Style each boxplot element with region-specific colors
    for i, region in enumerate(valid_regions):
        color = region_colors[i]
        light_color = lighten_color(color, amount=0.8)
        darken_color = lighten_color(color, amount=-0.5)

        # Box: lighter fill, original color edge
        bp["boxes"][i].set_facecolor(light_color)
        bp["boxes"][i].set_edgecolor(color)
        bp["boxes"][i].set_linewidth(linewidth)

        # Whiskers: original color (2 whiskers per box)
        bp["whiskers"][i * 2].set_color(color)
        bp["whiskers"][i * 2].set_linewidth(linewidth * 1.2)
        bp["whiskers"][i * 2 + 1].set_color(color)
        bp["whiskers"][i * 2 + 1].set_linewidth(linewidth * 1.2)

        # Caps: original color (2 caps per box)
        bp["caps"][i * 2].set_color(color)
        bp["caps"][i * 2].set_linewidth(linewidth * 1.2)
        bp["caps"][i * 2 + 1].set_color(color)
        bp["caps"][i * 2 + 1].set_linewidth(linewidth * 1.2)

        # Medians: darkened color
        bp["medians"][i].set_color(darken_color)
        bp["medians"][i].set_linewidth(linewidth * 1.2)

        # Fliers (outliers): original color
        bp["fliers"][i].set_markeredgecolor(color)
        bp["fliers"][i].set_markeredgewidth(linewidth * 0.8)
        bp["fliers"][i].set_markerfacecolor(light_color)

    # Set y-axis labels (no region names, just position numbers)
    y_pos = np.arange(1, num_regions + 1)
    ax.set_yticks(y_pos, labels=["" for _ in valid_regions])
    ax.tick_params(axis="y", pad=5)

    # Set x-axis for lifetime (hours) with log scale
    ax.set_xscale("log")
    ax.set_xlim(left=0.15)

    # Add grid
    ax.grid(True, which="major", alpha=0.3, axis="x", linestyle="-", linewidth=1.0)
    ax.grid(True, which="minor", alpha=0.1, axis="x", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Spot Lifetime (hours)", fontsize=15)

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

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot boxplot of spot instance availability fractions for multiple regions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o", default="outputs/spot_availability/spot_availability_boxplot.png", help="Output file path"
    )
    parser.add_argument("--data", "-d", default=str(DATA_DIR), help=f"Path to data directory (default: {DATA_DIR})")
    parser.add_argument(
        "--threshold",
        type=int,
        default=AVAILABILITY_THRESHOLD,
        help=f"Availability threshold (default: {AVAILABILITY_THRESHOLD})",
    )
    parser.add_argument(
        "--window-size",
        "-w",
        type=int,
        default=WINDOW_SIZE,
        help=f"Window size for calculating spot fraction (default: {WINDOW_SIZE})",
    )
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
        "--color-scheme",
        "-c",
        choices=["classic", "prefix"],
        default="classic",
        help="Color scheme: 'classic' (matplotlib colors) or 'prefix' (region-prefix colors). Default: classic",
    )
    parser.add_argument(
        "--all-regions", action="store_true", help="Plot all regions discovered in the data, sorted alphabetically"
    )

    args = parser.parse_args()

    plot_boxplot_figure(
        output_path=args.output,
        show_plot=not args.no_show,
        data_dir=pathlib.Path(args.data),
        threshold=args.threshold,
        window_size=args.window_size,
        start_date=args.start_date,
        end_date=args.end_date,
        color_scheme=args.color_scheme,
        all_regions=args.all_regions,
    )


if __name__ == "__main__":
    main()
