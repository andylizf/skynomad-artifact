#!/usr/bin/env python3
"""
Plot spot instance availability status for multiple regions.

This script visualizes availability patterns for specified GCP regions,
showing when spot instances were available vs preempted over time.
Each region is displayed as a horizontal line with availability segments.

Supports two data formats:
1. Raw runs data (runs/*.json) - applies threshold to determine binary availability
2. Converted trace data (data/converted_*) - uses existing binary availability
"""

import json
import pathlib
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator

###############

RESEARCH = 1  # For research presentation.
# RESEARCH = 0  # For paper pdf.

# font_size = {0: 10, 1: 8}[RESEARCH]
# font_size = 6.5  # Manual fix for 1-full-col figures.
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
    # https://matplotlib.org/3.2.1/tutorials/introductory/customizing.html
    params = {
        "backend": "ps",
        "figure.figsize": fig_size,
        "text.usetex": TEXT_USETEX,
        # 'font.family': 'serif',
        # 'font.serif': ['Times'],
        # 'font.family': 'sans-serif',
        # "font.sans-serif": [
        #     # 'Lato',
        #     # 'DejaVu Sans', 'Bitstream Vera Sans',
        #     # 'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva',
        #     # 'Lucid',
        #     # 'Arial',
        #     "Helvetica",
        #     "Avant Garde",
        #     "sans-serif",
        # ],
        # Make math fonts (e.g., tick labels) sans-serif.
        # https://stackoverflow.com/a/20709149/1165051
        "text.latex.preamble": "\n".join(
            [
                r"\usepackage{siunitx}",  # i need upright \micro symbols, but you need...
                r"\sisetup{detect-all}",  # ...this to force siunitx to actually use your fonts
                r"\usepackage{helvet}",  # set the normal font here
                r"\usepackage{sansmath}",  # load up the sansmath so that math -> helvet
                r"\sansmath",  # <- tricky! -- gotta actually tell tex to use!
            ]
        ),
        # axes.titlesize      : large   # fontsize of the axes title
        # 'axes.titlesize': font_size,
        "axes.titlesize": title_size,  # For plt.title().
        # 'axes.labelsize': 7,
        # 'legend.fontsize': 7,
        # 'font.size': 7,
        # 'xtick.labelsize': 7,
        # 'ytick.labelsize': 7,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "font.size": font_size,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "0.1",  # ~black border.
        "legend.shadow": False,
        "legend.frameon": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # http://phyletica.org/matplotlib-fonts/
        # Important for cam-ready (otherwise some fonts are not embedded):
        "pdf.fonttype": 42,
        "lines.linewidth": 1,
        # Styling.
        # 'grid.color': '#dedddd',
        # 'grid.linewidth': .5,
        # 'axes.grid.axis': 'y',
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
    'asia-south2-b',      # 1
    'asia-southeast1-b',  # 2
    'asia-southeast1-c',  # 3
    'europe-west1-c',     # 4
    'us-central1-a',      # 6
    'us-east4-b',         # 10
    'us-west1-a',         # 12
    'us-west1-b',         # 13
]

# Data directory (default: raw runs data)
DATA_DIR = pathlib.Path('data/h100_16_runs')

# Trace file index to use (8-15 available) - only for trace format
TRACE_INDEX = 8

# Availability threshold: instances_created < THRESHOLD → unavailable
AVAILABILITY_THRESHOLD = 16

# Colors for each region
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def load_runs_data(data_dir: pathlib.Path, regions: list, threshold: int = AVAILABILITY_THRESHOLD,
                   start_date: str = None, end_date: str = None) -> dict:
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
    json_files = sorted(data_dir.glob('*.json'))
    if not json_files:
        raise FileNotFoundError(f'No JSON files found in {data_dir}')

    # Filter by date range if specified (filename format: YYYYMMDD-HHMMSS-*.json)
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
    runs = [r for r in runs if r.get('status') == 'success']
    runs.sort(key=lambda r: r.get('scheduled_at', ''))

    if not runs:
        raise ValueError('No successful runs found')

    # Extract data per region
    region_data = {}
    for region in regions:
        timestamps = []
        availability = []  # 0=available, 1=preempted (matching trace format)

        for run in runs:
            try:
                timestamp = datetime.fromisoformat(run['scheduled_at'].replace('Z', '+00:00'))
                timestamp = timestamp.replace(tzinfo=None)  # Convert to naive datetime
            except (KeyError, ValueError):
                continue

            zone_results = run.get('zone_results', {})
            if region not in zone_results:
                continue

            result = zone_results[region]
            instances_created = result.get('instances_created', 0)

            timestamps.append(timestamp)
            # Binary availability: instances_created >= threshold → available (0)
            availability.append(0 if instances_created >= threshold else 1)

        if timestamps:
            # Calculate gap_seconds from data
            if len(timestamps) > 1:
                gaps = [(timestamps[i+1] - timestamps[i]).total_seconds()
                        for i in range(len(timestamps)-1)]
                gap_seconds = int(np.median(gaps))
            else:
                gap_seconds = 600  # Default

            region_data[region] = {
                'data': availability,
                'start_time': timestamps[0],
                'gap_seconds': gap_seconds,
                'timestamps': timestamps,  # Keep original timestamps for accurate plotting
            }

    return region_data


def load_trace_data(region: str, trace_index: int = TRACE_INDEX) -> dict:
    """
    Load trace data for a specific region.

    Args:
        region: Region name (e.g., 'us-central1-a')
        trace_index: Index of trace file to load (default: 8)

    Returns:
        Dictionary with 'data', 'start_time', 'gap_seconds' keys
    """
    region_dir = DATA_DIR / f'{region}_h100_16'
    trace_file = region_dir / f'{trace_index}.json'

    if not trace_file.exists():
        raise FileNotFoundError(f'Trace file not found: {trace_file}')

    with open(trace_file) as f:
        trace = json.load(f)

    metadata = trace.get('metadata', {})
    data = trace.get('data', [])

    start_time_str = metadata.get('start_time', '')
    try:
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        # Convert to naive datetime for matplotlib
        start_time = start_time.replace(tzinfo=None)
    except (ValueError, AttributeError):
        start_time = datetime.now()

    gap_seconds = metadata.get('gap_seconds', 600)

    return {
        'data': data,
        'start_time': start_time,
        'gap_seconds': gap_seconds,
    }


def plot_avail(data: list, start_time: datetime, gap_seconds: int, ax,
               center: float, color: str, verlen: float = 0.05,
               timestamps: list = None) -> None:
    """
    Plot availability as horizontal segments with vertical bars at preemption points.

    This mimics the plot_avail function from availability_price.ipynb.

    Args:
        data: List of 0/1 values (0=available, 1=preempted)
        start_time: Start datetime of the trace
        gap_seconds: Time interval between data points in seconds
        ax: Matplotlib axes
        center: Y-position for this region's line
        color: Color for this region
        verlen: Half-height of vertical bars at preemption points
        timestamps: Optional list of actual timestamps (if provided, ignores start_time/gap_seconds)
    """
    # Convert data: 0=available (becomes 1), 1=preempted (becomes 0)
    avail_list = [1 - x for x in data]

    # Create time series
    if timestamps is not None:
        cur_time = timestamps
    else:
        time_delta = timedelta(seconds=gap_seconds)
        cur_time = [start_time + i * time_delta for i in range(len(avail_list))]

    df = pd.DataFrame({'time': cur_time, 'avail': avail_list})

    # Add group column for continuous available segments
    df['group'] = (df['avail'] == 0).cumsum()

    # Filter to only available periods
    available_df = df[df['avail'] == 1]

    # Get unique groups
    groups = available_df['group'].unique()

    ymax = center + verlen
    ymin = center - verlen

    for i, group in enumerate(groups):
        group_data = available_df[available_df['group'] == group].copy()
        group_data['avail'] = center  # Set y-value to center

        # Plot the horizontal line for this availability segment
        ax.plot(group_data['time'],
                group_data['avail'],
                linestyle='-',
                color=color,
                linewidth=2,
                solid_capstyle='butt')

        # Add vertical line at the end of each segment
        # For the last segment, only add if it ended before data collection ended (was preempted)
        end_time = group_data['time'].max()
        is_last_group = (i == len(groups) - 1)
        if not is_last_group or end_time < df['time'].max():
            ax.vlines(x=end_time, ymin=ymin, ymax=ymax, linewidth=1.25, color=color)


def plot_availability_figure(output_path: str = None, show_plot: bool = True,
                              data_dir: pathlib.Path = None, use_runs: bool = True,
                              threshold: int = AVAILABILITY_THRESHOLD,
                              start_date: str = None, end_date: str = None) -> None:
    """
    Create the availability plot for all specified regions.

    Args:
        output_path: Path to save the figure (optional)
        show_plot: Whether to display the plot interactively
        data_dir: Path to data directory (default: DATA_DIR)
        use_runs: If True, load from runs/*.json format; if False, use trace format
        threshold: Availability threshold for runs data (instances_created < threshold → unavailable)
        start_date: Filter runs starting from this date (YYYYMMDD format, inclusive)
        end_date: Filter runs up to this date (YYYYMMDD format, inclusive)
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # Initialize matplotlib with research-grade settings
    InitMatplotlib(font_size=12, title_size=14)

    # Load data
    num_regions = len(REGIONS)
    ylim = 0.2 * num_regions

    # Create figure - adjust height based on number of regions
    fig_height = max(3, num_regions * 0.35)
    fig, ax = plt.subplots(1, 1, figsize=(16, fig_height), dpi=300)

    if use_runs:
        # Load all regions at once from runs data
        print(f'Loading runs data from {data_dir} (threshold={threshold})...')
        try:
            all_region_data = load_runs_data(data_dir, REGIONS, threshold, start_date, end_date)
        except (FileNotFoundError, ValueError) as e:
            print(f'Error loading runs data: {e}')
            return

        for i, region in enumerate(REGIONS):
            if region not in all_region_data:
                print(f'  Warning: No data for {region}')
                continue

            trace_data = all_region_data[region]
            color = COLORS[i % len(COLORS)]
            center = ylim - 0.2 * i

            plot_avail(
                data=trace_data['data'],
                start_time=trace_data['start_time'],
                gap_seconds=trace_data['gap_seconds'],
                ax=ax,
                center=center,
                color=color,
                verlen=0.075,
                timestamps=trace_data.get('timestamps')
            )
            avail_pct = 100 * (1 - sum(trace_data['data']) / len(trace_data['data']))
            print(f'  {region}: {len(trace_data["data"])} points, {avail_pct:.1f}% available')
    else:
        # Load each region separately from trace format
        for i, region in enumerate(REGIONS):
            print(f'Loading trace data for {region}...')
            try:
                trace_data = load_trace_data(region)
            except FileNotFoundError as e:
                print(f'  Warning: {e}')
                continue

            color = COLORS[i % len(COLORS)]
            center = ylim - 0.2 * i

            plot_avail(
                data=trace_data['data'],
                start_time=trace_data['start_time'],
                gap_seconds=trace_data['gap_seconds'],
                ax=ax,
                center=center,
                color=color,
                verlen=0.075
            )

    # Set y-axis
    ax.set_ylim(0, ylim * 1.08)
    label_names = list(reversed(REGIONS))
    y_pos = np.arange(0.2, 0.2 * num_regions + 0.01, 0.2)  # +0.01 to avoid float rounding
    ax.set_yticks(y_pos, labels=label_names, horizontalalignment='right')
    ax.tick_params(axis='y', pad=10)

    # Format x-axis - show tick for each day
    date_format = DateFormatter('%m/%d')
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(mdates.DayLocator())

    ax.set_xlabel('Date', fontweight='bold')
    # Set x-limits based on plotted data extents (use a small padding)
    x_vals_min = []
    x_vals_max = []
    # Collect from plotted lines
    for line in ax.get_lines():
        xd = line.get_xdata()
        if len(xd):
            try:
                # Normalize x-data to matplotlib date float numbers for consistent min/max.
                # If the x-data are already numeric (floats/ints), use them directly.
                # Otherwise convert datetime-like objects (e.g., datetime, np.datetime64, pd.Timestamp)
                # to python datetimes via pandas and then to matplotlib floats via date2num.
                first = xd[0]
                if isinstance(first, (float, np.floating, int, np.integer)):
                    xs = np.array(xd, dtype=float)
                else:
                    xs = mdates.date2num(pd.to_datetime(xd).to_pydatetime())
                x_vals_min.append(float(np.min(xs)))
                x_vals_max.append(float(np.max(xs)))
            except Exception:
                pass
                pass

    # Collect from collections (e.g., vlines -> LineCollection segments)
    for coll in getattr(ax, "collections", []):
        try:
            segments = coll.get_segments()
            for seg in segments:
                if len(seg):
                    xs = [p[0] for p in seg]
                    x_vals_min.append(min(xs))
                    x_vals_max.append(max(xs))
        except Exception:
            # Some collections may not support get_segments()
            continue

    if x_vals_min and x_vals_max:
        xmin = min(x_vals_min)
        xmax = max(x_vals_max)
        # delta = xmax - xmin

        # # Compute padding (works for datetime timedelta or numeric)
        # if isinstance(delta, timedelta):
        #     pad = delta * 0.01 if delta > timedelta(0) else timedelta(hours=12)
        # else:
        #     pad = delta * 0.01 if delta > 0 else 3600.0  # seconds fallback

        # Apply padding and set limits
        try:
            ax.set_xlim(xmin, xmax)
        except Exception:
            # Fallback: set to automatic if something unexpected occurs
            ax.set_xlim(auto=True)

    ax.grid(True, alpha=0.5, axis='y', linestyle='-', linewidth=1.25)

    plt.tight_layout()

    # Save and/or show
    if output_path:
        out_path = pathlib.Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
        print(f'Figure saved to: {out_path}')

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot spot instance availability for multiple regions.\n\n'
                    'Supports two data formats:\n'
                    '  1. Raw runs: runs/*.json (default, from spot-trace-prober)\n'
                    '  2. Converted trace: data/converted_* directories (--use-trace)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--output', '-o',
        default='outputs/spot_availability/spot_availability.png',
        help='Output file path'
    )
    parser.add_argument(
        '--data', '-d',
        default=str(DATA_DIR),
        help=f'Path to data directory (default: {DATA_DIR})'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=AVAILABILITY_THRESHOLD,
        help=f'Availability threshold: instances_created < threshold → unavailable (default: {AVAILABILITY_THRESHOLD})'
    )
    parser.add_argument(
        '--use-trace',
        action='store_true',
        help='Use converted trace format instead of runs format'
    )
    parser.add_argument(
        '--trace-index', '-t',
        type=int,
        default=8,
        help='Trace file index to use when --use-trace (default: 8)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help="Don't show the plot interactively"
    )
    parser.add_argument(
        '--start-date',
        default=None,
        help='Filter runs starting from this date (YYYYMMDD format, inclusive)'
    )
    parser.add_argument(
        '--end-date',
        default='20251010',
        help='Filter runs up to this date (YYYYMMDD format, inclusive, default: 20251010)'
    )

    args = parser.parse_args()

    # Update global trace index for trace format
    global TRACE_INDEX
    TRACE_INDEX = args.trace_index

    plot_availability_figure(
        output_path=args.output,
        show_plot=not args.no_show,
        data_dir=pathlib.Path(args.data),
        use_runs=not args.use_trace,
        threshold=args.threshold,
        start_date=args.start_date,
        end_date=args.end_date
    )


if __name__ == '__main__':
    main()
