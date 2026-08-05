#!/usr/bin/env python3
"""
Analyze spot duration distributions across all regions.
Generates log-log plots showing duration frequency for each region.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Color scheme from base_plot.py
colors = sns.color_palette("ch:2.5,+.2,dark=.3")[:2]
# Darken the base purple slightly
base_purple = colors[-1]
dark_factor = 0.8
darker_purple = tuple(max(0.0, min(1.0, c * dark_factor)) for c in base_purple)
light_purple = colors[0]
light_blue = "#a2d4ec"
set2 = sns.color_palette("Set2")
paired = sns.color_palette("Paired")

# Plot configuration
RESEARCH = 1  # For research presentation
font_size = 12
TEXT_USETEX = bool(1 - RESEARCH)

# Figure dimensions
nsdi23_col_width_pt = 241.02039
inches_per_pt = 1.0 / 72.27
golden_mean = (np.sqrt(5) - 1.0) / 2.0


def FigWidth(pt):
    return pt * inches_per_pt


fig_width_pt = nsdi23_col_width_pt
fig_width = FigWidth(fig_width_pt)
fig_height = fig_width * golden_mean
fig_size = [fig_width, fig_height]


def InitMatplotlib(font_size, title_size=9):
    """Initialize matplotlib with standardized settings."""
    print("use_tex", TEXT_USETEX, "\nfont_size", font_size, "\ntitle_size", title_size)

    params = {
        "backend": "ps",
        "figure.figsize": fig_size,
        "text.usetex": TEXT_USETEX,
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

    # Only configure specific fonts for LaTeX mode (for publication)
    if TEXT_USETEX:
        params["font.sans-serif"] = ["Helvetica", "Avant Garde", "sans-serif"]
        params["text.latex.preamble"] = "\n".join([
            r"\usepackage{siunitx}",
            r"\sisetup{detect-all}",
            r"\usepackage{helvet}",
            r"\usepackage{sansmath}",
            r"\sansmath",
        ])

    plt.style.use("seaborn-v0_8-colorblind")
    plt.rcParams.update(params)


def extract_spot_durations(trace_file: Path) -> List[float]:
    """Extract all spot durations from a trace file.

    Args:
        trace_file: Path to trace JSON file

    Returns:
        List of spot durations in hours
    """
    try:
        with open(trace_file, 'r') as f:
            trace = json.load(f)

        gap_seconds = trace['metadata']['gap_seconds']
        data = trace['data']

        # Extract continuous available periods
        spot_durations_hours = []
        current_duration = 0

        for status in data:
            if status == 0:  # Available
                current_duration += 1
            else:  # Preempted
                if current_duration > 0:
                    duration_hours = current_duration * gap_seconds / 3600
                    spot_durations_hours.append(duration_hours)
                    current_duration = 0

        # Don't forget the last run if it ends with availability
        if current_duration > 0:
            duration_hours = current_duration * gap_seconds / 3600
            spot_durations_hours.append(duration_hours)

        return spot_durations_hours

    except Exception as e:
        logger.warning(f"Failed to process {trace_file}: {e}")
        return []


def quantize_durations(durations: List[float], quantum: float = 0.5) -> List[float]:
    """Quantize durations to the nearest quantum (rounding up).

    Args:
        durations: List of durations in hours
        quantum: Quantization unit (default 0.5 hours)

    Returns:
        List of quantized durations
    """
    return [float(np.ceil(d / quantum) * quantum) for d in durations]


def fit_loglog_line(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Fit a line in log-log space (power law: y = A * x^B).

    Args:
        x: X values (must be positive)
        y: Y values (must be positive)

    Returns:
        Tuple of (slope, intercept, r_value, r_squared) or None if fit fails
        - slope (B): Power law exponent
        - intercept (log_A): Log of the coefficient
        - r_value: Pearson correlation coefficient
        - r_squared: R² value (coefficient of determination)
    """
    if len(x) < 2 or len(y) < 2:
        return None

    # Take logarithms
    log_x = np.log10(x)
    log_y = np.log10(y)

    # Perform linear regression in log-log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

    return slope, intercept, r_value, r_value**2 # type: ignore


def analyze_region_durations(data_dir: Path, use_full_trace: bool = True,
                            quantize: bool = True, quantum: float = 0.5) -> Dict[str, List[float]]:
    """Analyze spot durations for all regions.

    Args:
        data_dir: Directory containing region subdirectories
        use_full_trace: If True, use full.json; otherwise use all numbered traces
        quantize: If True, quantize durations to nearest quantum (rounding up)
        quantum: Quantization unit in hours (default 0.5)

    Returns:
        Dictionary mapping region names to lists of durations
    """
    region_durations = defaultdict(list)

    # Find all region directories
    region_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    logger.info(f"Found {len(region_dirs)} regions to analyze")
    if quantize:
        logger.info(f"Quantizing durations to {quantum}h units (rounding up)")

    for region_dir in region_dirs:
        region_name = region_dir.name
        logger.info(f"Processing region: {region_name}")

        if use_full_trace:
            # Use only full.json
            trace_files = [region_dir / "full.json"]
        else:
            # Use all numbered trace files
            trace_files = sorted(region_dir.glob("[0-9]*.json"))

        trace_files = [f for f in trace_files if f.exists()]

        if not trace_files:
            logger.warning(f"No trace files found for {region_name}")
            continue

        # Extract durations from all traces
        for trace_file in trace_files:
            durations = extract_spot_durations(trace_file)
            if quantize:
                durations = quantize_durations(durations, quantum)
            region_durations[region_name].extend(durations)

        logger.info(f"  → Found {len(region_durations[region_name])} spot instances")

    return dict(region_durations)


def plot_duration_distributions(region_durations: Dict[str, List[float]],
                                output_file: Path,
                                bins_per_decade: int = 20):
    """Plot log-log duration-frequency distributions for all regions.

    Args:
        region_durations: Dictionary mapping region names to duration lists
        output_file: Path to save the plot
        bins_per_decade: Number of bins per decade in log scale
    """
    num_regions = len(region_durations)

    # Create subplot grid
    ncols = 3
    nrows = (num_regions + ncols - 1) // ncols

    # Initialize matplotlib with standard settings
    InitMatplotlib(font_size=10, title_size=11)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    if num_regions == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 else axes

    # Plot each region
    for idx, (region_name, durations) in enumerate(sorted(region_durations.items())):
        ax = axes[idx]

        if not durations:
            ax.text(0.5, 0.5, f"{region_name}\nNo data",
                   ha='center', va='center', transform=ax.transAxes)
            continue

        durations_array = np.array(durations)

        # Calculate statistics
        mean_duration = np.mean(durations_array)
        median_duration = np.median(durations_array)
        total_instances = len(durations_array)

        # Create log-spaced bins
        min_duration = max(durations_array.min(), 0.01)  # Avoid log(0)
        max_duration = durations_array.max()

        num_bins = int(bins_per_decade * (np.log10(max_duration) - np.log10(min_duration)))
        num_bins = max(num_bins, 20)  # At least 20 bins

        bins = np.logspace(np.log10(min_duration), np.log10(max_duration), num_bins)

        # Calculate histogram
        counts, bin_edges = np.histogram(durations_array, bins=bins)

        # Plot on log-log scale
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Remove zero counts for log scale
        nonzero_mask = counts > 0
        x_data = bin_centers[nonzero_mask]
        y_data = counts[nonzero_mask]

        ax.loglog(x_data, y_data,
                 'o', markersize=5, alpha=0.7, color=darker_purple,
                 label='Data')

        # Fit power law in log-log space
        fit_result = fit_loglog_line(x_data, y_data)
        if fit_result is not None:
            slope, intercept, r_value, r_squared = fit_result

            # Generate fitted line
            x_fit = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
            y_fit = 10**intercept * x_fit**slope

            ax.loglog(x_fit, y_fit, '--', linewidth=2, alpha=0.8, color=set2[0],
                     label=f'Fit: $y = {10**intercept:.1f} \\cdot x^{{{slope:.2f}}}$\n$R^2 = {r_squared:.3f}$')

        # Add reference lines
        ax.axvline(median_duration, color=set2[1], linestyle='--',
                  linewidth=1, alpha=0.6, label=f'Median: {median_duration:.2f}h')
        ax.axvline(mean_duration, color=set2[2], linestyle='--',
                  linewidth=1, alpha=0.6, label=f'Mean: {mean_duration:.2f}h')

        # Labels and title
        ax.set_xlabel('Duration (hours)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{region_name}\n({total_instances} instances)', fontweight='bold')
        ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
        ax.legend(loc='best', fontsize=8)

    # Hide unused subplots
    for idx in range(num_regions, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")

    return fig


def plot_aggregated_distribution(region_durations: Dict[str, List[float]],
                                 output_file: Path,
                                 bins_per_decade: int = 20):
    """Plot aggregated log-log duration-frequency distribution across all regions.

    Args:
        region_durations: Dictionary mapping region names to duration lists
        output_file: Path to save the plot
        bins_per_decade: Number of bins per decade in log scale
    """
    # Aggregate all durations across regions
    all_durations = []
    for durations in region_durations.values():
        all_durations.extend(durations)

    if not all_durations:
        logger.warning("No durations to plot in aggregated view")
        return None

    durations_array = np.array(all_durations)

    # Calculate statistics
    mean_duration = np.mean(durations_array)
    median_duration = np.median(durations_array)
    total_instances = len(durations_array)

    # Initialize matplotlib with standard settings
    InitMatplotlib(font_size=12, title_size=14)

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 4))

    # Create log-spaced bins
    min_duration = max(durations_array.min(), 0.01)
    max_duration = durations_array.max()

    num_bins = int(bins_per_decade * (np.log10(max_duration) - np.log10(min_duration)))
    num_bins = max(num_bins, 20)

    bins = np.logspace(np.log10(min_duration), np.log10(max_duration), num_bins)

    # Calculate histogram
    counts, bin_edges = np.histogram(durations_array, bins=bins)

    # Plot on log-log scale
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Remove zero counts for log scale
    nonzero_mask = counts > 0
    x_data = bin_centers[nonzero_mask]
    y_data = counts[nonzero_mask]

    ax.loglog(x_data, y_data,
             'o', markersize=6, alpha=0.8, color=darker_purple)

    # Fit power law in log-log space
    fit_result = fit_loglog_line(x_data, y_data)
    if fit_result is not None:
        slope, intercept, r_value, r_squared = fit_result

        # Generate fitted line
        x_fit = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
        y_fit = 10**intercept * x_fit**slope

        ax.loglog(x_fit, y_fit, '-', linewidth=2.5, alpha=0.9, color=set2[0],
                 label=f'slope={slope:.2f}\n$R^2$={r_squared:.3f}')

    # # Add reference lines
    # ax.axvline(median_duration, color=set2[1], linestyle='--', # type: ignore
    #           linewidth=2, alpha=0.7, label=f'Median: {median_duration:.2f}h')
    # ax.axvline(mean_duration, color=set2[2], linestyle='--', # type: ignore
    #           linewidth=2, alpha=0.7, label=f'Mean: {mean_duration:.2f}h')

    # Add text with statistics
    stats_text = f'Total Instances: {total_instances}\n'
    stats_text += f'Regions: {len(region_durations)}\n'
    stats_text += f'Min: {durations_array.min():.2f}h\n'
    stats_text += f'Max: {durations_array.max():.2f}h\n'
    stats_text += f'Std: {np.std(durations_array):.2f}h'

    # ax.text(0.02, 0.98, stats_text,
    #        transform=ax.transAxes,
    #        verticalalignment='top',
    #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Labels and title
    ax.set_xlabel('Spot Lifetime (hours)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14, labelpad=5)
    # ax.set_title('Aggregated Spot Duration Distribution Across All Regions',
    #             fontweight='bold', pad=20)
    # ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=1.0)
    ax.grid(True, which='minor', alpha=0.15, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.5, edgecolor='none')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
    logger.info(f"Saved aggregated plot to {output_file}")

    return fig


def print_summary_statistics(region_durations: Dict[str, List[float]]):
    """Print summary statistics for all regions."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY REGION")
    print("="*80)

    # Table header
    print(f"{'Region':<30} {'Count':>8} {'Mean(h)':>10} {'Median(h)':>10} {'Min(h)':>10} {'Max(h)':>10}")
    print("-"*80)

    for region_name in sorted(region_durations.keys()):
        durations = region_durations[region_name]

        if not durations:
            print(f"{region_name:<30} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue

        durations_array = np.array(durations)
        count = len(durations_array)
        mean_dur = np.mean(durations_array)
        median_dur = np.median(durations_array)
        min_dur = np.min(durations_array)
        max_dur = np.max(durations_array)

        print(f"{region_name:<30} {count:>8} {mean_dur:>10.2f} {median_dur:>10.2f} {min_dur:>10.2f} {max_dur:>10.2f}")

    print("="*80 + "\n")


def save_raw_data(region_durations: Dict[str, List[float]], output_file: Path):
    """Save raw duration data to JSON file."""
    # Convert to serializable format
    data_to_save = {
        region: {
            'durations': durations,
            'count': len(durations),
            'mean': float(np.mean(durations)) if durations else None,
            'median': float(np.median(durations)) if durations else None,
            'min': float(np.min(durations)) if durations else None,
            'max': float(np.max(durations)) if durations else None,
            'std': float(np.std(durations)) if durations else None,
        }
        for region, durations in region_durations.items()
    }

    with open(output_file, 'w') as f:
        json.dump(data_to_save, f, indent=2)

    logger.info(f"Saved raw data to {output_file}")


def main():
    """Main analysis pipeline."""
    # Configuration
    data_dir = Path("data/converted_multi_region_aligned_h100_16_merged")
    output_dir = Path("outputs/duration_distribution_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    use_full_trace = True  # Set to False to use all numbered traces instead
    quantize = True  # Quantize durations to 0.5h units
    quantum = 1/6  # Quantization unit in hours

    logger.info(f"Starting duration distribution analysis")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using full traces: {use_full_trace}")

    # Analyze all regions
    region_durations = analyze_region_durations(
        data_dir,
        use_full_trace=use_full_trace,
        quantize=quantize,
        quantum=quantum
    )

    if not region_durations:
        logger.error("No regions found or all regions failed to process")
        return

    # Print summary statistics
    print_summary_statistics(region_durations)

    # Save raw data
    suffix = "_quantized" if quantize else ""
    raw_data_file = output_dir / f"region_durations{suffix}.json"
    save_raw_data(region_durations, raw_data_file)

    # # Plot individual region distributions
    # plot_file = output_dir / f"duration_distributions_loglog{suffix}.png"
    # plot_duration_distributions(region_durations, plot_file)

    # Plot aggregated distribution across all regions
    aggregated_plot_file = output_dir / f"duration_distribution_aggregated{suffix}.png"
    plot_aggregated_distribution(region_durations, aggregated_plot_file)

    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
