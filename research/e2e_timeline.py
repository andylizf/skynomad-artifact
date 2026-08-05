#!/usr/bin/env python
"""
Improved timeline visualization for spot instance simulation.

Features:
- Clean instance segment visualization
- Probe success markers
- Preemption and termination markers
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Style Setup (from base_plot.py)
# ============================================================================
def init_matplotlib(font_size=10):
    """Initialize matplotlib with research-style settings."""
    plt.style.use("seaborn-v0_8-colorblind")
    params = {
        "axes.titlesize": font_size + 2,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "font.size": font_size,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "0.8",
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
    plt.rcParams.update(params)


# ============================================================================
# Color Palette - Using seaborn colorblind palette
# ============================================================================
_palette = sns.color_palette("colorblind")
COLORS = {
    # Background
    "background": "#F0F0F0",

    # Instance segments - lighter green
    "spot_instance": "#7FC97F",        # Light green
    "ondemand_instance": _palette[1],  # Orange from colorblind palette

    # Events
    "probe_success": _palette[0],      # Blue
    "probe_failure": "#E41A1C",        # Red for probe failure
    "preemption": "#E41A1C",           # Red
    "termination": "#333333",          # Dark gray
    "completion": "#4DAF4A",           # Green for completion marker
}

# ============================================================================
# Core Functions
# ============================================================================

def load_history(path: str) -> List[Dict[str, Any]]:
    """Load JSONL history file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"History file is empty: {path}")
    return records


def find_instance_segments(history: List[Dict[str, Any]], completion_tick: Optional[int] = None) -> Dict[int, List[Tuple]]:
    """Extract instance run segments from history.

    Args:
        history: List of tick data dictionaries
        completion_tick: If provided, segments that are "still running" at the end
                        will extend to this tick (for task completion visualization)
    """
    segments: Dict[int, List[Tuple]] = {}

    # Track active instances per region
    active_instances: Dict[int, Dict] = {}  # region -> {start_tick, type}

    # Determine effective end tick for still-running instances
    effective_end_tick = len(history) - 1
    if completion_tick is not None:
        effective_end_tick = completion_tick

    for tick_idx, tick_data in enumerate(history):
        current_active = tick_data.get("ActiveInstances", {})

        # Check for new instances
        for region_str, inst_type in current_active.items():
            try:
                region = int(region_str)
            except ValueError:
                continue

            if region not in active_instances:
                # New instance started
                active_instances[region] = {
                    "start_tick": tick_idx,
                    "type": inst_type,
                }

        # Check for terminated/preempted instances
        regions_to_remove = []
        for region, info in active_instances.items():
            if str(region) not in current_active:
                # Instance ended
                end_tick = tick_idx - 1
                if end_tick >= info["start_tick"]:
                    # Determine end reason from events (events are list of dicts)
                    preempt_events = tick_data.get("PreemptionEvents", [])
                    term_events = tick_data.get("TerminateEvents", [])

                    end_reason = "unknown"
                    # Check preemption events
                    for evt in preempt_events:
                        if isinstance(evt, dict) and evt.get("region") == region:
                            end_reason = "preempted"
                            break
                    # Check termination events
                    if end_reason == "unknown":
                        for evt in term_events:
                            if isinstance(evt, dict) and evt.get("region") == region:
                                end_reason = "terminated"
                                break

                    if region not in segments:
                        segments[region] = []
                    segments[region].append((
                        info["start_tick"],
                        end_tick,
                        info["type"],
                        end_reason,
                    ))
                regions_to_remove.append(region)

        for region in regions_to_remove:
            del active_instances[region]

    # Handle still-running instances at end - extend to completion_tick
    for region, info in active_instances.items():
        if region not in segments:
            segments[region] = []
        segments[region].append((
            info["start_tick"],
            effective_end_tick,
            info["type"],
            "running",
        ))

    return segments


def draw_timeline(
    ax: plt.Axes,
    history: List[Dict[str, Any]],
    *,
    region_names: List[str],
    deadline_hours: float,
    gap_seconds: float = 60.0,
    show_legend: bool = True,
    progress_pct: float = 0.0,
) -> None:
    """Draw the timeline visualization."""

    if not history:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    gap_h = gap_seconds / 3600.0
    deadline_ticks = int(round(deadline_hours / gap_h))
    num_regions = len(region_names)
    history_len = min(len(history), deadline_ticks)

    # Find completion tick
    completion_tick = None
    for i, tick in enumerate(history):
        if tick.get("Task/task_is_done"):
            completion_tick = i
            break

    # Layout parameters - reduced bar height with space for probe line above
    region_spacing = 0.55          # Increased to accommodate probe line
    bar_height = 0.25              # Reduced height for instance bars
    probe_line_offset = 0.33       # Offset above the bar for probe line

    # Extract segments - pass completion_tick so running instances extend to completion
    segments = find_instance_segments(history, completion_tick=completion_tick)

    # Find the last active segment across all regions and extend it to history_len if needed
    # This handles cases where an instance was running at the end but ActiveInstances became empty
    last_active_seg = None
    last_active_region = None
    last_active_idx = None
    for region_idx, segs in segments.items():
        for seg_idx, seg in enumerate(segs):
            start_tick, end_tick, inst_type, end_reason = seg
            if last_active_seg is None or end_tick > last_active_seg[1]:
                last_active_seg = seg
                last_active_region = region_idx
                last_active_idx = seg_idx

    # Extend the last segment to history_len - 1 if it didn't end with preemption/termination
    if last_active_seg is not None:
        start_tick, end_tick, inst_type, end_reason = last_active_seg
        if end_reason in ("unknown", "running"):
            new_end = history_len - 1
            segments[last_active_region][last_active_idx] = (start_tick, new_end, inst_type, "running")

    # Collect all probe results from history (both success and failure)
    probe_results_by_region: Dict[int, List[Dict]] = {r: [] for r in range(num_regions)}
    for tick_idx in range(history_len):
        tick_data = history[tick_idx]
        probe_results = tick_data.get("LatestProbeResults", {})
        for region_name, success in probe_results.items():
            if region_name in region_names:
                region_idx = region_names.index(region_name)
                probe_results_by_region[region_idx].append({
                    "tick": tick_idx,
                    "success": success,
                })

    y_positions = []

    # Draw each region
    for region_idx in range(num_regions):
        y_base = (num_regions - region_idx - 1) * region_spacing
        y_positions.append(y_base + bar_height / 2)

        # Draw simple background bar (subtle)
        rect = mpatches.Rectangle(
            (0, y_base), deadline_hours, bar_height,
            facecolor=COLORS["background"], edgecolor="none",
            zorder=1,
        )
        ax.add_patch(rect)

        # Draw instance segments
        for seg in segments.get(region_idx, []):
            start_tick, end_tick, inst_type, end_reason = seg

            # Skip very short segments (less than 0.5 hour) - they're just visual noise
            width_h = (end_tick - start_tick + 1) * gap_h
            if width_h < 0.5:
                continue

            color = COLORS["spot_instance"] if inst_type == "SPOT" else COLORS["ondemand_instance"]

            x = start_tick * gap_h
            width = width_h

            # Main segment rectangle (no border for cleaner look)
            rect = mpatches.Rectangle(
                (x, y_base + bar_height * 0.05),
                width,
                bar_height * 0.9,
                facecolor=color,
                edgecolor="none",
                zorder=5,
            )
            ax.add_patch(rect)

            # End markers
            marker_x = (end_tick + 1) * gap_h
            marker_y = y_base + bar_height / 2

            if end_reason == "preempted":
                # Preemption: red X
                ax.plot(marker_x, marker_y, 'x',
                       color=COLORS["preemption"],
                       markersize=6, markeredgewidth=2, zorder=15)
            elif end_reason == "terminated":
                # Voluntary termination: black square
                ax.plot(marker_x, marker_y, 's',
                       color=COLORS["termination"],
                       markersize=5, markerfacecolor="white",
                       markeredgewidth=1, zorder=15)

    # Draw probe status line above each region's track
    # Only draw blue segments for available periods
    probe_line_width = 2.5
    for region_idx in range(num_regions):
        y_base = (num_regions - region_idx - 1) * region_spacing
        probe_y = y_base + probe_line_offset  # Position above the bar

        # Find available intervals and draw blue segments
        events = probe_results_by_region[region_idx]
        avail_start = None
        for evt in events:
            tick = evt["tick"]
            if completion_tick is not None and tick >= completion_tick:
                break
            success = evt["success"]
            time_h = tick * gap_h

            if success and avail_start is None:
                # Start of available interval
                avail_start = time_h
            elif not success and avail_start is not None:
                # End of available interval
                ax.plot([avail_start, time_h], [probe_y, probe_y],
                       color=COLORS["probe_success"], linewidth=probe_line_width,
                       solid_capstyle='butt', zorder=10)
                avail_start = None

        # Handle trailing available interval
        if avail_start is not None:
            end_tick = completion_tick if completion_tick is not None else history_len
            end_time_h = end_tick * gap_h
            ax.plot([avail_start, end_time_h], [probe_y, probe_y],
                   color=COLORS["probe_success"], linewidth=probe_line_width,
                   solid_capstyle='butt', zorder=10)

    # Set axes
    ax.set_xlim(0, deadline_hours)
    ax.set_ylim(-0.2, num_regions * region_spacing)

    # Draw deadline line at right edge (red vertical line)
    ax.axvline(x=deadline_hours - 0.3, color="#E41A1C", linestyle="-", linewidth=2.0, alpha=0.9, zorder=20)

    # Y-axis labels (region names only, shorter)
    first_tick = history[0] if history else {}
    price_by_region = first_tick.get("PriceByRegion", {})

    y_labels = []
    for region_idx, name in enumerate(region_names):
        prices = price_by_region.get(str(region_idx), {})
        spot_price = prices.get("SPOT", 0)
        # Shorter label: region name + price on same line
        label = f"{name}  (${spot_price:.2f})"
        y_labels.append(label)

    y_tick_offset = 0.05
    ax.set_yticks([y + y_tick_offset for y in y_positions])
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.tick_params(axis='y', pad=8)

    # X-axis
    ax.set_xlabel("Time (hours)", fontsize=14)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

    # # Grid - horizontal lines only
    # ax.yaxis.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    # ax.xaxis.grid(False)
    # ax.set_axisbelow(True)

    # Draw progress/completion marker at right side
    if history_len > 0:
        end_time_h = history_len * gap_h
        # Draw vertical line at current progress point
        ax.axvline(x=end_time_h, color="#666666", linestyle="--", linewidth=1.5, alpha=0.7, zorder=2)

    time_highleghts = [
        {
            "type": "point",
            "time": 5.58,  # us-west-2c preemption at tick 335
            "text": "$t_1$=5.6",
            "label_offset": 0,
        },
        {
            "type": "point",
            "time": 13.0,  # migration to ap-northeast-1c after us-east-2 volatility
            "text": "$t_2$=13.0",
            "label_offset": -1.0,  # slightly left
        },
        {
            "type": "range",
            "start": 16.33,  # ap-northeast-1c preemption at tick 980
            "end": 20.0,     # idle ends, launch in eu-central-1a at tick 1200
            "text": "$t_3$=16.3-20.0",
            "label_offset": 0.7,  # slightly right
        }
    ]
    # Position labels at top of figure (above all region rows)
    label_y_top = num_regions * region_spacing + 0.1
    for th in time_highleghts:
        offset = th.get("label_offset", 0)
        if th["type"] == "point":
            ax.axvline(x=th["time"], color="#666666", linestyle=":", linewidth=1.2, alpha=0.7, zorder=1)
            # Center text on the vertical line with optional offset
            ax.text(th["time"] + offset, label_y_top, f"{th['text']}", zorder=12,
                    color="#333333", fontsize=10, ha="center", va="bottom")
        elif th["type"] == "range":
            ax.axvspan(th["start"], th["end"], color="#cccccc", alpha=0.3, zorder=2)
            # Center text on the middle of the range with optional offset
            range_center = (th["start"] + th["end"]) / 2
            ax.text(range_center + offset, label_y_top, f"{th['text']}", zorder=12,
                    color="#333333", fontsize=10, ha="center", va="bottom")

    # Legend - vertical column on right side with grouped sections
    if show_legend:
        legend_fontsize = 11.8

        # Create section title (invisible handle + bold label)
        def make_title(label):
            return mpatches.Patch(facecolor="none", edgecolor="none", label=label)

        # Spacer for vertical separation
        def make_spacer():
            return mpatches.Patch(facecolor="none", edgecolor="none", label=" ")

        legend_elements = [
            # Section: Instance
            make_title(r"$\bf{Instance}$"),
            mpatches.Patch(facecolor=COLORS["spot_instance"], edgecolor="none",
                          label="Spot"),
            mpatches.Patch(facecolor=COLORS["ondemand_instance"], edgecolor="none",
                          label="On-Demand"),
            # Spacer + Section: End Reason
            make_spacer(),
            make_title(r"$\bf{End\ Reason}$"),
            mlines.Line2D([], [], color=COLORS["preemption"], marker='x',
                         linestyle='None', markersize=6, markeredgewidth=1.5,
                         label="Preemption"),
            mlines.Line2D([], [], color=COLORS["termination"], marker='s',
                         linestyle='None', markersize=5, markerfacecolor="white",
                         markeredgewidth=1, label="Proactive Mig."),
            # Spacer + Section: Probe
            make_spacer(),
            make_title(r"$\bf{Probe}$"),
            mlines.Line2D([], [], color=COLORS["probe_success"],
                         linestyle='-', linewidth=2, label="Avail"),
            # Spacer + Section: Timeline
            make_spacer(),
            mlines.Line2D([], [], color="#E41A1C",
                         linestyle='-', linewidth=2, label="Deadline"),
            mlines.Line2D([], [], color="#666666",
                         linestyle='--', linewidth=1.5, label="Completion"),
        ]
        # Legend position based on actual line positions
        completion_x = (history_len * gap_h) / deadline_hours  # e.g., 32/45 ≈ 0.71
        deadline_x = 1.0

        # Place legend centered between completion and deadline
        legend_center_x = (completion_x + deadline_x) / 2
        legend_y = 0.51

        # Sizing parameters
        legend_labelspacing = 0.22
        legend_handlelength = 1.2
        legend_handleheight = 1.0

        ax.legend(handles=legend_elements, loc="center", ncol=1,
                 fontsize=legend_fontsize, bbox_to_anchor=(legend_center_x, legend_y),
                 handlelength=legend_handlelength, handleheight=legend_handleheight,
                 labelspacing=legend_labelspacing, handletextpad=0.5,
                 frameon=True, edgecolor="none", framealpha=0.75)


def plot_timeline(history_path: str, output_path: str = None) -> str:
    """Main function to create timeline plot."""

    # Initialize matplotlib style
    init_matplotlib(font_size=12)

    logger.info(f"Loading: {history_path}")
    history = load_history(history_path)

    # Extract config
    first = history[0]
    config = first.get("Config", {})

    region_names = config.get("RegionNames", [f"R{i}" for i in range(10)])
    task_duration_hours = config.get("TaskDurationHours", 30.0)
    deadline_hours = config.get("DeadlineHours", 45.0)
    gap_seconds = config.get("GapSeconds", 60.0)
    strategy_name = first.get("Strategy", "unknown")

    # Calculate stats
    last_tick = history[-1]
    done_seconds = last_tick.get("Task/Done(seconds)", 0)
    target_seconds = last_tick.get("Task/Target(seconds)", task_duration_hours * 3600)
    progress_pct = done_seconds / target_seconds * 100 if target_seconds > 0 else 0
    total_cost = last_tick.get("Cost", 0)

    # Time range
    start_time = datetime.datetime.fromisoformat(first.get("WallTime", ""))
    end_time = datetime.datetime.fromisoformat(last_tick.get("WallTime", ""))
    duration = end_time - start_time

    # Create figure - with space for probe lines above bars
    num_regions = len(region_names)
    fig_height = max(3.0, num_regions * 0.25 + 1.0)
    fig, ax = plt.subplots(figsize=(7, fig_height))

    # Draw timeline
    draw_timeline(
        ax, history,
        region_names=region_names,
        deadline_hours=deadline_hours,
        gap_seconds=gap_seconds,
        progress_pct=progress_pct,
    )

    # No title

    # Output
    if output_path is None:
        output_path = Path(history_path).parent / "timeline_improved.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate improved timeline visualization")
    parser.add_argument("history", nargs="?", default="research/history.jsonl",
                       help="Path to history.jsonl file (default: research/history.jsonl)")
    parser.add_argument("-o", "--output", default="outputs/e2e_timeline/timeline_improved.png", help="Output path for the plot")
    args = parser.parse_args()

    plot_timeline(args.history, args.output)


if __name__ == "__main__":
    main()
