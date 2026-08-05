"""Render per-scenario segment grids comparing UCM vs Oracle-DP using existing config.

No external YAML is required. Scenarios come from scenario_config.EXPERIMENT_SCENARIOS,
and which scenarios / how many traces are controlled by CLI flags in modular.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import contextlib
import io
import sys
import matplotlib.patches as patches

import matplotlib.pyplot as plt
import numpy as np

from .trace_utils import get_trace_paths_for_task
from scripts_multi.visualize_timeline_segments import (
    extract_trace_availability,
    find_instance_segments,
    COLORS,
)
from .scenario_config import EXPERIMENT_SCENARIOS, get_segment_viz_strategies

logger = logging.getLogger(__name__)


def _draw_progress_plot(
    ax: plt.Axes,
    history: List[Dict[str, Any]],
    *,
    deadline_hours: float,
    gap_seconds: float,
    task_duration_hours: float,
    restart_overhead_hours: float,
    show_xlabel: bool = True,
) -> None:
    """Draw progress-time plot below the segment visualization.

    Shows task progress over time, marking safety net trigger point.
    """
    if not history:
        ax.text(
            0.5,
            0.5,
            "No history data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
        )
        return

    gap_h = gap_seconds / 3600.0
    deadline_ticks = max(1, int(round(deadline_hours / gap_h)))
    restart_overhead_seconds = restart_overhead_hours * 3600.0

    # Extract progress data from history
    timestamps = []
    progress_values = []
    cluster_types = []  # To track when running on SPOT vs ON_DEMAND
    safety_net_tick = None

    # Get task target from first entry for normalization
    task_target_seconds = None
    if history and isinstance(history[0], dict):
        task_target_seconds = history[0].get("Task/Target(seconds)")
    if task_target_seconds is None or task_target_seconds <= 0:
        task_target_seconds = task_duration_hours * 3600.0

    for tick_idx, tick_data in enumerate(history):
        if tick_idx >= deadline_ticks:
            break

        timestamps.append(tick_idx)

        # Get progress (normalize to 0-1) - use Task/Done(seconds)
        done_seconds = tick_data.get("Task/Done(seconds)", 0.0)
        if isinstance(done_seconds, (int, float)):
            progress_pct = (
                float(done_seconds) / task_target_seconds
                if task_target_seconds > 0
                else 0.0
            )
        else:
            progress_pct = 0.0
        progress_values.append(min(progress_pct, 1.0))  # Cap at 100%

        # Get cluster type for coloring from ActiveInstances
        active_instances = tick_data.get("ActiveInstances", {})
        if active_instances:
            types = list(active_instances.values())
            if "ON_DEMAND" in types:
                cluster_types.append("ON_DEMAND")
            elif "SPOT" in types:
                cluster_types.append("SPOT")
            else:
                cluster_types.append("NONE")
        else:
            cluster_types.append("NONE")

        # Detect safety net trigger
        # Safety net triggers when: needed_ticks >= remaining_ticks
        remaining_seconds = tick_data.get("Task/Remaining(seconds)", 0.0)
        if isinstance(remaining_seconds, (int, float)):
            remaining_task = float(remaining_seconds)
        else:
            remaining_task = (
                task_target_seconds - float(done_seconds)
                if isinstance(done_seconds, (int, float))
                else 0.0
            )

        remaining_ticks = deadline_ticks - tick_idx
        if remaining_task > 0 and safety_net_tick is None:
            import math

            needed_ticks = math.ceil(
                (remaining_task + restart_overhead_seconds) / gap_seconds
            )
            if needed_ticks >= remaining_ticks:
                safety_net_tick = tick_idx

    if not timestamps:
        return

    # Convert to numpy arrays
    timestamps_arr = np.array(timestamps)
    progress_arr = np.array(progress_values)
    time_hours = timestamps_arr * gap_h

    # Draw progress line: SPOT=green, ON_DEMAND=red, NONE=skip
    # Use step plot for staircase effect (vertical edges)
    if len(time_hours) > 1:
        i = 0
        while i < len(cluster_types):
            ct = cluster_types[i]
            if ct == "NONE":
                i += 1
                continue
            # Find end of consecutive same-type segment
            j = i + 1
            while j < len(cluster_types) and cluster_types[j] == ct:
                j += 1
            # Draw segment [i, j]
            end_idx = min(j, len(time_hours))
            if end_idx > i:
                color = COLORS["spot_instance"] if ct == "SPOT" else COLORS["ondemand_instance"]
                ax.step(time_hours[i:end_idx], progress_arr[i:end_idx],
                        where="post", color=color, linewidth=1.5, zorder=2)
            i = j

    # Draw safety net line: progress = (t - (D - T)) / T
    # This is the minimum progress required at each time to finish on time
    T = task_duration_hours
    D = deadline_hours
    slack = D - T  # Time buffer before safety net kicks in

    # Safety net line: starts at t=slack with progress=0, ends at t=D with progress=1
    if slack >= 0:
        sn_t_start = slack
        sn_t_end = D
        sn_p_start = 0.0
        sn_p_end = 1.0
    else:
        # Task is longer than deadline (shouldn't happen normally)
        sn_t_start = 0
        sn_t_end = D
        sn_p_start = -slack / T
        sn_p_end = 1.0

    ax.plot(
        [sn_t_start, sn_t_end],
        [sn_p_start, sn_p_end],
        color="red",
        linestyle="--",
        linewidth=2.0,
        zorder=5,
        label="Safety Net",
    )

    # Add target progress line (100%)
    ax.axhline(y=1.0, color="green", linestyle=":", linewidth=1, alpha=0.5)

    # Set axis limits and labels
    ax.set_xlim(0, deadline_hours)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Progress", fontsize=7)
    if show_xlabel:
        ax.set_xlabel("Time (hours)", fontsize=7)

    # Set x-ticks to match segment plot (use hours)
    if deadline_ticks <= 1:
        major_indices = np.array([0])
    else:
        label_step = max(1, int(np.ceil(deadline_ticks / 8)))
        major_indices = np.arange(0, deadline_ticks, label_step)
        if major_indices.size == 0 or major_indices[0] != 0:
            major_indices = np.insert(major_indices, 0, 0)
        if major_indices[-1] != deadline_ticks - 1:
            major_indices = np.append(major_indices, deadline_ticks - 1)
    major_pos = np.round(major_indices * gap_h, 12)
    ax.set_xticks(major_pos)
    ax.set_xticklabels([f"{pos:.0f}" for pos in major_pos], fontsize=6)

    # Y-axis ticks
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0%", "50%", "100%"], fontsize=6)

    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)


@contextlib.contextmanager
def _silence_logs():
    """Temporarily silence stdout/stderr and lower log levels for segment sims.

    Enabled by default; set SEG_VIZ_VERBOSE=1 to disable silencing.
    """
    import os as _os

    root_logger = logging.getLogger()
    desired_verbose = _os.environ.get("SEG_VIZ_VERBOSE", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )
    # If user sets root logger to DEBUG (e.g. LOG_LEVEL=DEBUG), don't force downgrade
    if desired_verbose or root_logger.level <= logging.DEBUG:
        # Verbose mode: do nothing
        yield
        return
    # Capture stdout/stderr
    _stdout, _stderr = sys.stdout, sys.stderr
    buf_out, buf_err = io.StringIO(), io.StringIO()
    # Lower logging levels
    old_level = root_logger.level
    root_logger.setLevel(logging.WARNING)
    # Also try to quiet common module loggers
    for name in ("sky_spot", "scripts_multi", "openevolve_multi_region_strategy"):
        try:
            logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:
            pass
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            yield
    finally:
        root_logger.setLevel(old_level)
        sys.stdout, sys.stderr = _stdout, _stderr


def _find_scenario_config(name: str) -> Optional[Dict[str, Any]]:
    for s in EXPERIMENT_SCENARIOS:
        if s.get("name") == name:
            return s
    return None


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def _slug_num_trim(v: float) -> str:
    s = f"{v:.2f}"
    if s.endswith("00"):
        s = s[:-3]  # drop .00
    s = s.replace(".", "p").replace("-", "m")
    return s


def _scenario_slug(name: str) -> str:
    import re

    nm = name
    m = re.search(r"\((\d+)\s*Regions?\)", nm)
    suffix = f"_{m.group(1)}" if m else ""
    nm = re.sub(r"\((\d+)\s*Regions?\)", "", nm)
    nm = nm.strip().lower().replace(" ", "_").replace("-", "_")
    nm = "".join(c for c in nm if c.isalnum() or c == "_")
    return nm + suffix


def _short_name(strategy: str) -> str:
    mapping = {
        "unified_cost_model": "UCM",
        "unified_cost_model_v2": "UCMv2",
        "unified_cost_model_oracle": "UCM-Oracle",
        "unified_cost_model_risk": "UCM-Risk",
        "multi_region_oracle_dp": "OracleDP",
        "multi_region_rc_cr_threshold": "RC-CR-TH",
        "multi_region_single_rc_cr_threshold": "RC-CR-TH-SingleRC",
        "multi_region_rc_cr_threshold_egress": "RC-CR-TH-Egress",
        "rc_cr_threshold": "RC-CR-TH-Single",  # Single-region baseline
    }
    return mapping.get(strategy, strategy)


def _compute_region_meta(
    trace_paths: List[str], env_start_hours: float, deadline_hours: float
) -> List[Dict[str, Any]]:
    meta_list: List[Dict[str, Any]] = []
    if not trace_paths:
        return meta_list
    for tp in trace_paths:
        try:
            from pathlib import Path as _P

            with open(_P(tp), "r") as f:
                data = json.load(f)
            gap_s = float(data["metadata"]["gap_seconds"])
            start_idx = max(0, int(round(env_start_hours * 3600.0 / gap_s)))
            window_ticks = max(1, int(round(deadline_hours * 3600.0 / gap_s)))
            avail_seq = data.get("data", [])
            slice_vals = avail_seq[start_idx : start_idx + window_ticks]
            total = len(slice_vals)
            avail = sum(1 for v in slice_vals if v == 0)
            prices = data.get("price") or data.get("prices")
            avg_price = None
            if prices:
                price_slice = prices[start_idx : start_idx + total]
                vals = [p for p in price_slice if p is not None]
                if vals:
                    avg_price = float(sum(vals) / len(vals))
            meta_list.append({"avg_price": avg_price, "avail": avail, "total": total})
        except Exception as e:
            meta_list.append({"avg_price": None, "avail": None, "total": None})
    return meta_list


def _find_history_file(
    hist_root: Path,
    scenario_name: str,
    dr: float,
    ckpt: float,
    ro: float,
    strategy: str,
    trace_id: int,
    env_start_h: Optional[float],
) -> Optional[Path]:
    import glob

    scen_slug = _scenario_slug(scenario_name)
    base_dir = f"dr{_slug_num_trim(dr)}_ck{_slug_num_trim(ckpt)}_ro{_slug_num_trim(ro)}"
    dirs_to_try = []
    if env_start_h is not None:
        dirs_to_try.append(f"{base_dir}_es{_slug_num_trim(env_start_h)}")
    dirs_to_try.append(base_dir)
    strat_short = _short_name(strategy)
    for param_dir in dirs_to_try:
        pattern_gz = str(
            hist_root
            / scen_slug
            / param_dir
            / strat_short
            / f"trace_{trace_id}.json.gz"
        )
        logger.debug("Segment viz history lookup: %s", pattern_gz)
        matches = glob.glob(pattern_gz)
        if matches:
            return Path(matches[0])
        pattern = str(
            hist_root / scen_slug / param_dir / strat_short / f"trace_{trace_id}.json"
        )
        logger.debug("Segment viz history lookup: %s", pattern)
        matches = glob.glob(pattern)
        if matches:
            return Path(matches[0])
    return None


def _draw_segments(
    ax: plt.Axes,
    stats: Dict[str, Any],
    trace_paths: List[str],
    *,
    title: str,
    deadline_hours: float,
    cost_value: Optional[float] = None,
    show_region_labels: bool = True,
    region_meta: Optional[List[Dict[str, Any]]] = None,
    region_names: Optional[List[str]] = None,
    region_indices: Optional[List[int]] = None,
    segments_override: Optional[Dict[int, List[Tuple]]] = None,
    background_grid: Optional[List[List[Optional[bool]]]] = None,
) -> Dict[str, Any]:
    """Draw timeline segments for a single strategy panel."""
    history = _extract_history_list(stats)
    if not history:
        logger.error(
            "Segment viz missing history data: title=%s strategy=%s trace_files=%s deadline=%.2f",
            title,
            stats.get("strategy", {}).get("name")
            if isinstance(stats.get("strategy"), dict)
            else stats.get("strategy"),
            trace_paths,
            deadline_hours,
        )
        raise RuntimeError(
            "Segment visualization requested but no history data was found for this strategy panel. "
            "Regenerate histories (e.g., clear caches or re-run simulations) before plotting."
        )

    env_info = stats.get("env", {})
    gap_seconds = float(env_info.get("gap_seconds", 600))
    gap_h = max(gap_seconds / 3600.0, 1.0 / 3600.0)
    deadline_ticks = max(1, int(round(deadline_hours / gap_h)))

    if background_grid is not None:
        availability_grid = [row[:] for row in background_grid]
        num_regions = len(availability_grid)
    else:
        num_regions = 1
        for tick in history:
            sa = tick.get("SpotAvailability") or {}
            if isinstance(sa, dict) and sa:
                try:
                    num_regions = max(num_regions, 1 + max(int(k) for k in sa.keys()))
                except Exception:
                    pass
                break
        num_regions = max(num_regions, len(trace_paths))

        availability_grid = [[None] * deadline_ticks for _ in range(num_regions)]
        max_tick = min(deadline_ticks, len(history))
        for tick_idx in range(max_tick):
            sa = history[tick_idx].get("SpotAvailability") or {}
            if not isinstance(sa, dict):
                continue
            for region_str, val in sa.items():
                try:
                    region = int(region_str)
                except Exception:
                    continue
                if 0 <= region < num_regions:
                    availability_grid[region][tick_idx] = bool(val)

    if region_names:
        num_regions = max(num_regions, len(region_names))
    if len(availability_grid) < num_regions:
        availability_grid.extend(
            [
                [False] * deadline_ticks
                for _ in range(num_regions - len(availability_grid))
            ]
        )
    elif len(availability_grid) > num_regions:
        availability_grid = availability_grid[:num_regions]

    region_labels: List[str] = list(region_names) if region_names else []
    try:
        from pathlib import Path as _P

        for idx, tp in enumerate(trace_paths[:num_regions]):
            label = _P(tp).parent.name
            if idx < len(region_labels):
                if not region_labels[idx]:
                    region_labels[idx] = label
            else:
                region_labels.append(label)
    except Exception:
        pass
    while len(region_labels) < num_regions:
        region_labels.append(f"R{len(region_labels)}")

    segments = (
        segments_override
        if segments_override is not None
        else find_instance_segments(history)
    )
    if region_indices is not None:
        region_order = [r for r in region_indices if 0 <= r < num_regions]
        if not region_order:
            region_order = list(range(num_regions))
    else:
        # Always include all regions to ensure background is drawn for all
        region_order = list(range(num_regions))

    # Ensure each region renders on its own visually separated band.
    plot_region_count = max(1, len(region_order))
    region_spacing = max(1.1, min(1.6, 1.0 + 0.05 * plot_region_count))
    bar_height = min(0.9, region_spacing - 0.3)
    if bar_height <= 0.5:
        bar_height = 0.5
    plot_height = max(1.0, plot_region_count * region_spacing)

    y_centers: List[float] = []
    y_labels: List[str] = []
    total_drawn = 0

    for idx, region in enumerate(region_order):
        y_base = (plot_region_count - idx - 1) * region_spacing
        raw_label = (
            region_labels[region] if region < len(region_labels) else f"R{region}"
        )
        label_root = raw_label.split("_")[0] if "_" in raw_label else raw_label
        price_txt = "$ -"
        avail_txt = "-/-"
        if region_meta and region < len(region_meta):
            meta = region_meta[region]
            avg_price = meta.get("avg_price")
            if isinstance(avg_price, (int, float)) and np.isfinite(avg_price):
                price_txt = f"$ {avg_price:.2f}"
            avail = meta.get("avail")
            total = meta.get("total")
            if isinstance(avail, int) and isinstance(total, int) and total > 0:
                avail_txt = f"{avail}/{total}"
        y_centers.append(y_base + bar_height / 2)
        y_labels.append(f"{label_root}   {price_txt} | {avail_txt}")

        for tick_idx in range(deadline_ticks):
            val = availability_grid[region][tick_idx]
            x = tick_idx * gap_h
            if val is None:
                # Handle missing data with gray color
                color = "#E0E0E0"  # Light gray for missing data
                alpha = 0.7
            else:
                color = COLORS["spot_available"] if val else COLORS["spot_unavailable"]
                alpha = 0.75  # Further increased for better visibility
            base_rect = patches.Rectangle(
                (x, y_base),
                gap_h,
                bar_height,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=0.3,
            )
            ax.add_patch(base_rect)

        for segment in segments.get(region, []):
            if len(segment) == 5:
                start_tick, end_tick, inst_type, _, _ = segment
            else:
                start_tick, end_tick, inst_type, _ = segment
            color = (
                COLORS["spot_instance"]
                if inst_type == "SPOT"
                else COLORS["ondemand_instance"]
            )
            alpha = 1.0
            vertical_padding = bar_height * 0.1  # 10% padding on each side
            border_width_base = 0.4

            def _draw_rect(span_start: int, span_end: int) -> None:
                nonlocal total_drawn
                if span_end < span_start:
                    return
                start_x = span_start * gap_h
                duration = (span_end - span_start + 1) * gap_h
                if duration <= 0:
                    return
                border_width = 0.3 if duration <= gap_h else border_width_base
                rect = patches.Rectangle(
                    (start_x, y_base + vertical_padding),
                    duration,
                    bar_height - 2 * vertical_padding,
                    facecolor=color,
                    edgecolor="#000000",
                    linewidth=border_width,
                    zorder=4,
                    alpha=alpha,
                    joinstyle="miter",
                    capstyle="projecting",
                    antialiased=False,
                )
                ax.add_patch(rect)
                total_drawn += 1

            if (
                inst_type == "SPOT"
                and background_grid is not None
                and 0 <= region < len(background_grid)
            ):
                availability_row = background_grid[region]
                span_start = None
                for tick_idx in range(start_tick, end_tick + 1):
                    avail = True
                    if 0 <= tick_idx < len(availability_row):
                        avail = bool(availability_row[tick_idx])
                    if avail:
                        if span_start is None:
                            span_start = tick_idx
                    else:
                        if span_start is not None:
                            _draw_rect(span_start, tick_idx - 1)
                            span_start = None
                if span_start is not None:
                    _draw_rect(span_start, end_tick)
            else:
                _draw_rect(start_tick, end_tick)

    ax.set_xlim(0, deadline_hours)
    ax.set_ylim(0, plot_height)
    if show_region_labels:
        ax.set_yticks(y_centers)
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.tick_params(axis="y", labelsize=7, pad=8)
        for tick in ax.yaxis.get_ticklabels():
            tick.set_horizontalalignment("right")
    else:
        ax.set_yticks(y_centers)
        ax.set_yticklabels([""] * len(y_centers))
        ax.tick_params(axis="y", length=0)

    if deadline_ticks <= 1:
        major_indices = np.array([0])
    else:
        label_step = max(1, int(np.ceil(deadline_ticks / 8)))
        major_indices = np.arange(0, deadline_ticks, label_step)
        if major_indices.size == 0 or major_indices[0] != 0:
            major_indices = np.insert(major_indices, 0, 0)
        if major_indices[-1] != deadline_ticks - 1:
            major_indices = np.append(major_indices, deadline_ticks - 1)
    major_pos = np.round(major_indices * gap_h, 12)
    ax.set_xticks(major_pos)
    ax.set_xticklabels([str(int(idx)) for idx in major_indices])

    minor_indices = np.arange(0, deadline_ticks + 1)
    minor_pos = np.round(minor_indices * gap_h, 12)
    ax.set_xticks(minor_pos, minor=True)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.tick_params(axis="x", which="major", length=3)
    ax.set_axisbelow(True)

    title_lines = [title]
    if cost_value is not None and np.isfinite(cost_value):
        title_lines.append(f"Total=${cost_value:.2f}")
    ax.set_title("\n".join(title_lines), fontsize=8, loc="left")

    if total_drawn == 0:
        ax.text(
            0.5,
            0.5,
            "No instances",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
        )

    return {
        "total_segments": total_drawn,
        "segments_per_region": {
            region: len(seg_list) for region, seg_list in segments.items() if seg_list
        },
    }


def _extract_history_list(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    history_obj = stats.get("history", [])
    if isinstance(history_obj, list) and history_obj:
        first = history_obj[0]
        if isinstance(first, dict):
            return history_obj  # type: ignore[return-value]
        if isinstance(first, list) and first and isinstance(first[0], dict):
            return first  # type: ignore[return-value]
    return []


def _first_float(stats_dict: Dict[str, Any], key: str) -> Optional[float]:
    arr = stats_dict.get(key, [])
    if isinstance(arr, list) and arr:
        val = arr[0]
        if isinstance(val, (int, float)) and np.isfinite(val):
            try:
                return float(val)
            except Exception:
                return None
    return None


def _first_int(stats_dict: Dict[str, Any], key: str) -> Optional[int]:
    arr = stats_dict.get(key, [])
    if isinstance(arr, list) and arr:
        val = arr[0]
        if isinstance(val, (int, float)) and np.isfinite(val):
            try:
                return int(val)
            except Exception:
                return None
    return None


def _build_background_grid(
    histories: List[List[Dict[str, Any]]],
    trace_paths: List[str],
    deadline_ticks: int,
    env_start_ticks: int,
    num_regions: int,
) -> List[List[Optional[bool]]]:
    """Combine history availability with trace fallback into a shared background grid."""
    grid: List[List[Optional[bool]]] = [
        [None] * deadline_ticks for _ in range(num_regions)
    ]

    for history in histories:
        max_tick = min(deadline_ticks, len(history))
        for tick_idx in range(max_tick):
            sa = history[tick_idx].get("SpotAvailability") or {}
            if not isinstance(sa, dict):
                continue
            for region_str, val in sa.items():
                try:
                    region = int(region_str)
                except Exception:
                    continue
                if 0 <= region < num_regions:
                    grid[region][tick_idx] = bool(val)

    trace_availability = extract_trace_availability(trace_paths) if trace_paths else {}
    if trace_availability:
        for region, seq in trace_availability.items():
            if region >= num_regions:
                continue
            for tick_idx in range(deadline_ticks):
                if grid[region][tick_idx] is not None:
                    continue
                idx = env_start_ticks + tick_idx
                if 0 <= idx < len(seq):
                    grid[region][tick_idx] = bool(seq[idx])

    # Fill any remaining gaps with False (treated as unavailable)
    for region in range(num_regions):
        row = grid[region]
        for tick_idx in range(deadline_ticks):
            if row[tick_idx] is None:
                row[tick_idx] = False

    return grid


def render_segment_grids_scenarios(
    *,
    scenario_names: List[str],
    num_traces: int,
    params: Dict[str, Any],
    results_df,
    output_dir: Path,
    show_progress_plot: bool = False,
    strategies_override: List[str] | None = None,
) -> None:
    # Optional per-scenario data roots override; fall back to global DATA_PATH
    data_path_map = params.get("DATA_PATH_BY_SCENARIO", {}) or {}
    default_data_spec = params.get("DATA_PATH", [])
    for name in scenario_names:
        scen_conf = _find_scenario_config(name)
        if scen_conf is None:
            logger.warning(
                f"segment_viz: scenario not found in config table: {name}; skip"
            )
            continue
        regions = scen_conf.get("regions", [])
        # Use override strategies if provided, otherwise fall back to scenario/global default
        strategies = strategies_override if strategies_override else get_segment_viz_strategies(name)
        if not strategies:
            logger.warning(
                f"segment_viz: no strategies configured for scenario {name}; skip"
            )
            continue

        scenario_data_spec = data_path_map.get(name, default_data_spec)

        # Determine trace selections from results_df (keep alignment with use_full_trace/env_start)
        df_sc = results_df[results_df["scenario_name"] == name]
        if df_sc.empty:
            logger.warning(f"segment_viz: no results for scenario {name}; skip")
            continue

        base_strategy = strategies[0]
        # Prefer rows for base strategy to derive selections (usually present for each param combo)
        df_sel = df_sc[df_sc["strategy"] == base_strategy].copy()
        if df_sel.empty:
            df_sel = df_sc[df_sc["strategy"].isin(strategies)].copy()
        if df_sel.empty:
            # fallback to any strategy if none of the preferred ones are available
            df_sel = df_sc.copy()
        # Group by parameter combos and emit one figure per combo
        group_cols = [
            c
            for c in [
                "deadline_hours",
                "task_duration",
                "restart_overhead",
                "checkpoint_size",
                "deadline_ratio",
            ]
            if c in df_sel.columns
        ]
        if not group_cols:
            groups = [(None, df_sel)]
        else:
            groups = list(df_sel.groupby(group_cols))

        for _, df_grp in groups:
            # Build rows: in full-trace mode, draw by env_start windows; else by trace_index
            rows_to_draw: List[tuple[int, float]] = []  # (trace_id, env_start_hours)
            is_full = (
                bool(df_grp["use_full_trace"].iloc[0])
                if "use_full_trace" in df_grp.columns and len(df_grp) > 0
                else False
            )
            if is_full:
                if "env_start_hours" in df_grp.columns:
                    env_starts = sorted(
                        df_grp["env_start_hours"].dropna().unique().tolist()
                    )
                else:
                    env_starts = [0.0]
                env_starts = env_starts[: max(1, num_traces)]
                if not env_starts:
                    logger.warning(f"segment_viz: scenario {name}: no windows; skip")
                    continue
                for es in env_starts:
                    rows_to_draw.append((0, float(es)))  # full-trace uses trace_index=0
            else:
                # Legacy indexed
                if "trace_index" in df_grp.columns:
                    tids = sorted(df_grp["trace_index"].dropna().unique().tolist())
                else:
                    tids = [0]
                tids = tids[: max(1, num_traces)]
                for tid in tids:
                    rows_to_draw.append((int(tid), 0.0))

            if not rows_to_draw:
                logger.warning(f"segment_viz: scenario {name}: no rows to draw; skip")
                continue

            # Build grid figure: rows=len(rows_to_draw), columns=len(strategies)
            # When show_progress_plot=True, add a progress subplot below each segment
            n_trace_rows = len(rows_to_draw)
            n_cols = max(1, len(strategies))
            approx_region_count = (
                max(1, len(regions)) if isinstance(regions, (list, tuple)) else 1
            )
            if show_progress_plot:
                # Heights based on content
                segment_height = float(np.clip(approx_region_count * 0.20, 0.5, 1.2))
                progress_height = 0.8

                # Figure dimensions
                fig_height = (segment_height + progress_height) * n_trace_rows + 1.5
                fig_width = max(16.0, 2.5 * n_cols)

                fig, all_axes = plt.subplots(
                    n_trace_rows * 2, n_cols,
                    figsize=(fig_width, fig_height),
                    squeeze=False,
                    constrained_layout=True,
                    gridspec_kw={
                        "height_ratios": [segment_height, progress_height] * n_trace_rows,
                    },
                )

                # Split axes into segment and progress rows
                segment_axes = []
                progress_axes = []
                for r in range(n_trace_rows):
                    segment_axes.append(list(all_axes[r * 2]))
                    progress_axes.append(list(all_axes[r * 2 + 1]))
                axes = np.array(segment_axes)
            else:
                # Normal height when no progress plots
                per_row_height = float(np.clip(approx_region_count * 0.30, 2.4, 5.5))
                fig_height = per_row_height * n_trace_rows
                fig_width = max(16.0, 7.0 * n_cols)
                fig, axes = plt.subplots(
                    n_trace_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False
                )
                progress_axes = None

            # Use per-trace parameter rows when available; fall back to a representative row for this group
            rep_row = df_grp.iloc[0]
            for r, (tid, es) in enumerate(rows_to_draw):
                # Match by (trace_index, env_start_hours) when possible
                row_match = None
                if (
                    "trace_index" in df_grp.columns
                    and "env_start_hours" in df_grp.columns
                ):
                    row_match = df_grp[
                        (df_grp["trace_index"] == tid)
                        & (df_grp["env_start_hours"] == es)
                    ]
                elif "trace_index" in df_grp.columns:
                    row_match = df_grp[df_grp["trace_index"] == tid]
                try:
                    row = (
                        row_match.iloc[0]
                        if row_match is not None and len(row_match) > 0
                        else rep_row
                    )
                except Exception:
                    row = rep_row
                deadline_h = float(
                    row.get("deadline_hours", params.get("DEADLINE_HOURS") or 0.0)
                )
                env_start_h = float(es)

                # Resolve trace metadata for labeling
                task_stub = {
                    "task_type": "multi_region",
                    "regions_in_scenario": regions,
                    "trace_index": int(tid),
                    "use_full_trace": is_full,
                }
                _, trace_paths = get_trace_paths_for_task(task_stub, scenario_data_spec)
                region_meta = _compute_region_meta(trace_paths, env_start_h, deadline_h)

                # History lookup parameters
                hist_root = Path(params.get("OUTPUT_DIR", "outputs")) / "histories"
                dr = float(row.get("deadline_ratio", 0.0) or 0.0)
                ck = float(row.get("checkpoint_size", 0.0) or 0.0)
                ro = float(row.get("restart_overhead", 0.0) or 0.0)
                es = float(env_start_h)

                def _load_history(strat_name: str) -> dict:
                    import gzip as _gz, json as _json

                    path = _find_history_file(
                        hist_root, name, dr, ck, ro, strat_name, int(tid), es
                    )
                    if path and path.exists():
                        try:
                            with _gz.open(path, "rt", encoding="utf-8") as f:
                                return _json.load(f)
                        except Exception:
                            try:
                                with open(path, "r") as f:
                                    return _json.load(f)
                            except Exception:
                                return {}
                    return {}

                task_val_raw = row.get(
                    "task_duration", row.get("task_duration_hours", 0.0)
                )
                try:
                    row_task_hours = float(task_val_raw)
                except Exception:
                    row_task_hours = float(row.get("task_duration", 0.0) or 0.0)
                row_ckpt = float(row.get("checkpoint_size", 0.0) or 0.0)
                row_restart = float(row.get("restart_overhead", 0.0) or 0.0)

                axes_row = axes[r]
                stats_by_strategy: Dict[str, dict] = {}
                history_by_strategy: Dict[str, List[Dict[str, Any]]] = {}
                segments_by_strategy: Dict[str, Dict[int, List[Tuple]]] = {}
                regions_union: set[int] = set()
                costs_row: List[Optional[float]] = []
                strategy_info: List[Dict[str, Any]] = []

                for strat in strategies:
                    stats = _load_history(strat)
                    args_dict = stats.setdefault("args", {})
                    args_dict.setdefault("deadline_hours", deadline_h)
                    args_dict.setdefault("task_duration_hours", [row_task_hours])
                    args_dict.setdefault("checkpoint_size_gb", row_ckpt)
                    args_dict.setdefault("restart_overhead_hours", [row_restart])
                    args_dict.setdefault("env_start_hours", float(env_start_h))
                    args_dict.setdefault("trace_files", [str(p) for p in trace_paths])
                    args_dict.setdefault("use_full_trace", is_full)
                    args_dict.setdefault("trace_index", int(tid))

                    stats_by_strategy[strat] = stats
                    history_for_segments = _extract_history_list(stats)
                    history_by_strategy[strat] = history_for_segments
                    segs = (
                        find_instance_segments(history_for_segments)
                        if history_for_segments
                        else {}
                    )
                    segments_by_strategy[strat] = segs
                    regions_union.update(
                        region for region, seg_list in segs.items() if seg_list
                    )

                    cv = None
                    arr = stats.get("costs", [])
                    if isinstance(arr, list) and arr:
                        try:
                            cv = float(arr[0])
                        except Exception:
                            cv = None
                    costs_row.append(cv)

                gap_seconds = None
                for stats in stats_by_strategy.values():
                    env_gap = stats.get("env", {}).get("gap_seconds")
                    if env_gap is not None:
                        try:
                            gap_seconds = float(env_gap)
                            break
                        except Exception:
                            gap_seconds = None
                if gap_seconds is None:
                    gap_seconds = 600.0
                gap_h = max(gap_seconds / 3600.0, 1.0 / 3600.0)
                deadline_ticks = max(1, int(round(deadline_h / gap_h)))
                env_start_ticks = int(round(env_start_h / gap_h)) if env_start_h else 0

                histories_for_bg = [
                    history_by_strategy.get(s, [])
                    for s in strategies
                    if history_by_strategy.get(s)
                ]
                max_region_idx = -1
                for hist in histories_for_bg:
                    for tick_data in hist:
                        sa = tick_data.get("SpotAvailability") or {}
                        if isinstance(sa, dict):
                            for key in sa.keys():
                                try:
                                    max_region_idx = max(max_region_idx, int(key))
                                except Exception:
                                    continue
                num_regions_bg = max(
                    len(trace_paths), max_region_idx + 1 if max_region_idx >= 0 else 0
                )
                if num_regions_bg <= 0:
                    num_regions_bg = len(trace_paths) or 1

                background_grid = _build_background_grid(
                    histories_for_bg,
                    trace_paths,
                    deadline_ticks,
                    env_start_ticks,
                    num_regions_bg,
                )

                if regions:
                    regions_to_draw = list(range(len(regions)))
                elif regions_union:
                    regions_to_draw = sorted(regions_union)
                else:
                    regions_to_draw = list(range(num_regions_bg))

                for c, strat in enumerate(strategies):
                    stats = stats_by_strategy.get(strat, {})
                    cv = costs_row[c] if c < len(costs_row) else None
                    title = (
                        f"Win {r} – {_short_name(strat)}"
                        if is_full
                        else f"Trace {tid} – {_short_name(strat)}"
                    )
                    draw_meta = _draw_segments(
                        axes_row[c],
                        stats,
                        trace_paths,
                        title=title,
                        deadline_hours=deadline_h,
                        cost_value=cv,
                        show_region_labels=(c == 0),
                        region_meta=region_meta,
                        region_names=regions,
                        region_indices=regions_to_draw,
                        segments_override=segments_by_strategy.get(strat),
                        background_grid=background_grid,
                    )

                    # Draw progress plot if enabled
                    if show_progress_plot and progress_axes is not None:
                        history_for_progress = history_by_strategy.get(strat, [])
                        _draw_progress_plot(
                            progress_axes[r][c],
                            history_for_progress,
                            deadline_hours=deadline_h,
                            gap_seconds=gap_seconds,
                            task_duration_hours=row_task_hours,
                            restart_overhead_hours=row_restart,
                            show_xlabel=(
                                r == n_trace_rows - 1
                            ),  # Only show xlabel on last row
                        )
                        # Hide x-axis label on segment plot when progress plot is shown
                        axes_row[c].set_xlabel("")

                    history_for_metrics = history_by_strategy.get(strat, [])
                    final_snapshot = None
                    if history_for_metrics:
                        last_entry = history_for_metrics[-1]
                        if isinstance(last_entry, dict):
                            final_snapshot = last_entry
                    probe_cost_val = (
                        final_snapshot.get("ProbeCost") if final_snapshot else None
                    )
                    egress_cost_val = (
                        final_snapshot.get("TransferCost") if final_snapshot else None
                    )
                    probe_ticks_val = (
                        final_snapshot.get("ProbeTickCount") if final_snapshot else None
                    )
                    if probe_cost_val is None:
                        probe_cost_val = _first_float(stats, "probe_costs")
                    if egress_cost_val is None:
                        egress_cost_val = _first_float(stats, "transfer_costs")
                    if probe_ticks_val is None:
                        probe_ticks_val = _first_int(stats, "probe_tick_counts")

                    strategy_info.append(
                        {
                            "ax": axes_row[c],
                            "strategy": strat,
                            "title": title,
                            "cost": cv,
                            "probe_cost": probe_cost_val,
                            "egress_cost": egress_cost_val,
                            "probe_ticks": probe_ticks_val,
                            "segments": draw_meta.get("total_segments", 0),
                        }
                    )

                baseline_cost: Optional[float] = None
                if strategy_info:
                    base_cost = strategy_info[0].get("cost")
                    if isinstance(base_cost, (int, float)):
                        try:
                            baseline_cost = float(base_cost)
                        except Exception:
                            baseline_cost = None
                baseline_short = _short_name(strategies[0]) if strategies else ""

                for c, info in enumerate(strategy_info):
                    ax = info["ax"]
                    line_two_parts: List[str] = []
                    if info["cost"] is not None:
                        line_two_parts.append(f"Total=${info['cost']:.2f}")
                    if info["egress_cost"] is not None:
                        try:
                            line_two_parts.append(
                                f"Egress=${float(info['egress_cost']):.2f}"
                            )
                        except Exception:
                            pass
                    line_two_parts.append(f"segs={int(info['segments'])}")

                    lines: List[str] = [info["title"], " | ".join(line_two_parts)]

                    probe_lines: List[str] = []
                    probe_cost = info["probe_cost"]
                    probe_ticks = info["probe_ticks"]
                    strat_name = info.get("strategy", "")
                    has_probe_signal = any(
                        [
                            isinstance(strat_name, str)
                            and "probe" in strat_name.lower(),
                            isinstance(probe_cost, (int, float))
                            and abs(float(probe_cost)) > 1e-9,
                            isinstance(probe_ticks, (int, float))
                            and int(probe_ticks) > 0,
                        ]
                    )
                    if has_probe_signal:
                        if probe_cost is not None:
                            try:
                                pc_val = float(probe_cost)
                                if abs(pc_val) > 1e-9:
                                    probe_lines.append(f"ProbeCost=${pc_val:.2f}")
                            except Exception:
                                pass
                        if probe_ticks:
                            probe_lines.append(f"ProbeTicks={int(probe_ticks)}")
                    if c > 0 and baseline_cost is not None and info["cost"] is not None:
                        try:
                            delta_val = baseline_cost - float(info["cost"])
                            probe_lines.append(
                                f"Δ vs {baseline_short}=$ {delta_val:.2f}"
                            )
                        except Exception:
                            pass
                    if probe_lines:
                        lines.append(" | ".join(probe_lines))

                    ax.set_title(
                        "\n".join(lines), fontsize=8, loc="left", linespacing=1.35
                    )

            # Super title with parameter info (from representative row rep_row) and strategy list
            try:
                rep_deadline_h = float(
                    rep_row.get("deadline_hours", params.get("DEADLINE_HOURS") or 0.0)
                )
                rep_dr = float(rep_row.get("deadline_ratio", 0.0) or 0.0)
                rep_td_h = float(
                    rep_row.get("task_duration", 0.0)
                    or (rep_deadline_h / (rep_dr if rep_dr else 1.0))
                )
                rep_ro_h = float(rep_row.get("restart_overhead", 0.0) or 0.0)
                rep_ckpt = float(rep_row.get("checkpoint_size", 0.0) or 0.0)
            except Exception:
                rep_deadline_h = float(params.get("DEADLINE_HOURS") or 0.0)
                rep_dr = 0.0
                rep_td_h = 0.0
                rep_ro_h = 0.0
                rep_ckpt = 0.0

            subtitle = f"DL={rep_deadline_h:.2f}h  |  Task={rep_td_h:.2f}h  |  RO={rep_ro_h:.2f}h  |  CKPT={rep_ckpt:.0f}GB  |  DR={rep_dr:.2f}"
            strategy_titles = " | ".join(_short_name(s) for s in strategies)
            fig.suptitle(
                f"{name} – Strategies: {strategy_titles}  (n={n_trace_rows})\n{subtitle}",
                fontsize=12,
                fontweight="bold",
            )
            # Layout adjustment (only for non-constrained_layout figures)
            if not show_progress_plot:
                fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.0, w_pad=0.5)

            # New filename: put parameters first, like segments_grid_dr2_ck0_ro0_gcp_a3_h100_5.png
            scen_slug = _scenario_slug(name)
            dr = _slug_num_trim(rep_dr)
            ck = _slug_num_trim(rep_ckpt)
            ro = _slug_num_trim(rep_ro_h)
            filename = f"segments_grid_dr{dr}_ck{ck}_ro{ro}_{scen_slug}.png"
            out = Path(output_dir) / filename
            fig.savefig(out, dpi=300)
            plt.close(fig)
            logger.info(f"🖼️  Segment grid saved: {out}")
