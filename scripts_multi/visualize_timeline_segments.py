#!/usr/bin/env python3
"""
Timeline visualization with continuous segments for running instances.
"""

import json
import argparse
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Color scheme
COLORS = {
    # Background - light colors for availability
    'spot_available': '#C8E6C9',      # Light green
    'spot_unavailable': '#FFCDD2',    # Light red

    # Foreground - strong colors for instances
    'spot_instance': '#1B5E20',       # Dark green
    'ondemand_instance': '#E65100',   # Dark orange
}


def _cluster_type_to_name(value):
    """Map assorted cluster type encodings to canonical names."""
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit():
            value = int(raw)
        else:
            upper = raw.upper()
            if upper in ('SPOT',):
                return 'SPOT'
            if upper in ('ON_DEMAND', 'ONDEMAND', 'OD'):
                return 'ON_DEMAND'
            if upper in ('NONE', '0', 'NOPE'):
                return None
            # Fall through for other representations like "ClusterType.SPOT"
            if 'SPOT' in upper:
                return 'SPOT'
            if 'ON_DEMAND' in upper or 'ONDEMAND' in upper:
                return 'ON_DEMAND'
            if 'NONE' in upper:
                return None
            return None
    if isinstance(value, (int, float)):
        iv = int(value)
        if iv == 2:
            return 'SPOT'
        if iv == 3:
            return 'ON_DEMAND'
        return None
    return None


def load_data(json_path: Path) -> Dict:
    """Load simulation results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_trace_availability(trace_files: List[str]) -> Dict[int, List[bool]]:
    """Extract spot availability from trace files."""
    mount_rewrites = [
        ('/tmp/data/converted_multi_region_aligned', Path('data') / 'converted_multi_region_aligned'),
        ('/root/data/converted_multi_region_aligned', Path('data') / 'converted_multi_region_aligned'),
    ]
    availability = {}
    for idx, trace_file in enumerate(trace_files):
        path = Path(trace_file)
        if not path.exists():
            for prefix, local_base in mount_rewrites:
                if trace_file.startswith(prefix):
                    suffix = trace_file[len(prefix):].lstrip('/')
                    candidate = local_base / suffix
                    if candidate.exists():
                        path = candidate
                        break
            if not path.exists():
                raise FileNotFoundError(
                    f"Trace file not found for availability background: {trace_file}."
                    " Provide a local copy or update the history to point to accessible files."
                )
        with path.open('r') as f:
            data = json.load(f)
            availability[idx] = [val == 0 for val in data['data']]
    return availability

def availability_to_segments(avail: List[bool]) -> List[Tuple[int, int]]:
    """Convert availability booleans (True=available) into (start_tick, end_tick) segments inclusive."""
    segs: List[Tuple[int, int]] = []
    if not avail:
        return segs
    in_run = False
    start = 0
    for i, a in enumerate(avail):
        if a and not in_run:
            in_run = True
            start = i
        elif not a and in_run:
            segs.append((start, i - 1))
            in_run = False
    if in_run:
        segs.append((start, len(avail) - 1))
    return segs


def find_instance_segments(history: List[Dict]) -> Dict[int, List[Tuple]]:
    """Find continuous segments where instances are running.
    Returns: {region: [(start_tick, end_tick, instance_type, final_progress, had_overhead), ...]}
    """
    segments: Dict[int, List[Tuple]] = defaultdict(list)
    current_segment = {}

    for tick_idx, tick_data in enumerate(history):
        # Get progress
        task_done = tick_data.get('Task/Done(seconds)', 0)
        task_target = tick_data.get('Task/Target(seconds)', 1)
        progress = (task_done / task_target * 100) if task_target > 0 else 0
        
        # Check if there's restart overhead
        restart_overhead_remaining = tick_data.get('Strategy/RemainingRestartOverhead(seconds)', 0)
        
        # Find active instances - use ActiveInstances field, not cost!
        raw_active = tick_data.get('ActiveInstances') or {}
        active_instances = raw_active if isinstance(raw_active, dict) else {}

        # Single-region fallback: use cluster type fields when ActiveInstances missing
        if not active_instances:
            fallback = (
                _cluster_type_to_name(tick_data.get('ClusterType')) or
                _cluster_type_to_name(tick_data.get('RequestType')) or
                _cluster_type_to_name(tick_data.get('Strategy/ClusterType'))
            )
            if fallback in ('SPOT', 'ON_DEMAND'):
                active_instances = {'0': fallback}

        active_regions = set()

        for region_str, inst_type_str in active_instances.items():
            region = int(region_str)
            active_regions.add(region)

            # Check if this is a new segment or continuation
            if region not in current_segment:
                # Start new segment
                current_segment[region] = {
                    'start': tick_idx,
                    'type': inst_type_str,
                    'progress': progress,
                    'had_overhead': restart_overhead_remaining > 0
                }
            else:
                seg = current_segment[region]
                if seg['type'] != inst_type_str:
                    # Type changed (e.g., SPOT -> ON_DEMAND); close previous segment first
                    segments[region].append((
                        seg['start'],
                        max(seg['start'], tick_idx - 1),
                        seg['type'],
                        seg['progress'],
                        seg.get('had_overhead', False)
                    ))
                    # Start a fresh segment with the new type beginning this tick
                    current_segment[region] = {
                        'start': tick_idx,
                        'type': inst_type_str,
                        'progress': progress,
                        'had_overhead': restart_overhead_remaining > 0
                    }
                else:
                    # Update progress and overhead status
                    seg['progress'] = progress
                    if restart_overhead_remaining > 0:
                        seg['had_overhead'] = True
        
        # Check for ended segments
        ended_regions = set(current_segment.keys()) - active_regions
        for region in ended_regions:
            seg = current_segment[region]
            segments[region].append((
                seg['start'], 
                tick_idx - 1,  # End at previous tick
                seg['type'], 
                seg['progress'],
                seg.get('had_overhead', False)
            ))
            del current_segment[region]
    
    # Close any remaining segments
    for region, seg in current_segment.items():
        segments[region].append((
            seg['start'], 
            len(history) - 1,
            seg['type'], 
            seg['progress'],
            seg.get('had_overhead', False)
        ))
    
    return dict(segments)


def plot_cost_curve(ax, history: List[Dict], gap_hours: float, deadline_hours: float):
    """Plot cost accumulation curve over time."""
    # Extract costs from history
    times = []
    costs = []
    
    for i, tick_data in enumerate(history):
        time_hours = i * gap_hours
        cost = tick_data.get('Cost', 0)
        times.append(time_hours)
        costs.append(cost)
    
    # If we have data and it doesn't reach deadline, extend it
    if times and times[-1] < deadline_hours:
        times.append(deadline_hours)
        costs.append(costs[-1])  # Keep the last cost value
    
    # Plot the cost curve
    ax.plot(times, costs, color='#1f77b4', linewidth=2, marker='', markersize=4)
    ax.fill_between(times, costs, alpha=0.3, color='#1f77b4')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Labels and formatting
    ax.set_ylabel('Accumulated Cost ($)', fontsize=12)
    ax.set_xlabel('Time (hours)', fontsize=12)
    
    # Add deadline line
    ax.axvline(x=deadline_hours, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Set x-axis to stop at deadline
    ax.set_xlim(0, deadline_hours)
    
    # Add annotations for key points
    if costs:
        # Cost at deadline
        deadline_idx = int(deadline_hours / gap_hours)
        if deadline_idx < len(costs):
            deadline_cost = costs[deadline_idx]
            ax.plot(deadline_hours, deadline_cost, 'ro', markersize=8)
            ax.annotate(f'${deadline_cost:.2f}', 
                       xy=(deadline_hours, deadline_cost), 
                       xytext=(deadline_hours - 5, deadline_cost),
                       ha='right', va='center',
                       fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_progress_curve(ax, history: List[Dict], gap_hours: float, deadline_hours: float, task_hours: float):
    """Plot task progress by colored line segments:
    - SPOT progress: green line
    - ON_DEMAND progress: red line
    - Restart overhead tick: black horizontal line
    - Flat (no progress) and not in restart: no line
    """
    # Precompute times and progress
    times = []
    progress_percent = []
    for i, tick_data in enumerate(history):
        time_hours = i * gap_hours
        task_done = tick_data.get('Task/Done(seconds)', 0) / 3600.0
        pct = (task_done / task_hours * 100.0) if task_hours > 0 else 0.0
        progress_percent.append(min(pct, 100.0))
        times.append(time_hours)

    # Helper to get instance type for a tick
    def inst_type_at(i: int) -> str:
        if i >= len(history):
            return 'NONE'
        tick = history[i]
        act = tick.get('ActiveInstances')
        if isinstance(act, dict) and act:
            return next(iter(act.values()))
        fallback = (
            _cluster_type_to_name(tick.get('ClusterType')) or
            _cluster_type_to_name(tick.get('RequestType')) or
            _cluster_type_to_name(tick.get('Strategy/ClusterType'))
        )
        return fallback if fallback else 'NONE'

    # Draw line segments per tick interval
    for i in range(1, len(times)):
        t1, t2 = times[i - 1], times[i]
        p1, p2 = progress_percent[i - 1], progress_percent[i]
        dp = p2 - p1

        # Detect restart for this interval: remaining overhead on previous tick OR current tick
        prev_over = history[i - 1].get('Strategy/RemainingRestartOverhead(seconds)', 0) or 0.0
        curr_over = history[i].get('Strategy/RemainingRestartOverhead(seconds)', 0) or 0.0
        is_restart = (prev_over > 1e-6) or (curr_over > 1e-6)

        if dp <= 1e-9:
            # Flat progress: only draw if in restart -> black horizontal line
            if is_restart:
                ax.plot([t1, t2], [p1, p2], color='black', linewidth=2)
            continue

        # Choose color by instance type
        ctype = inst_type_at(i)
        color = '#2ca02c' if ctype == 'SPOT' else ('#E53935' if ctype == 'ON_DEMAND' else 'gray')
        # If interval started with restart overhead, show as black; else colored by ctype
        if is_restart:
            ax.plot([t1, t2], [p1, p2], color='black', linewidth=2)
        else:
            ax.plot([t1, t2], [p1, p2], color=color, linewidth=2)

    # Ideal progress line
    ax.plot([0, deadline_hours], [0, 100], 'k--', linewidth=1.2, alpha=0.6, label='Ideal')

    # Axes setup
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylabel('Task Progress (%)', fontsize=12)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.axvline(x=deadline_hours, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, deadline_hours)
    # Legend entries (patchless lines)
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color='#2ca02c', lw=2, label='SPOT progress'),
        Line2D([0], [0], color='#E53935', lw=2, label='ON_DEMAND progress'),
        Line2D([0], [0], color='black', lw=2, label='Restart overhead'),
        Line2D([0], [0], color='k', lw=1.2, ls='--', alpha=0.6, label='Ideal'),
    ]
    ax.legend(handles=legend_lines, loc='upper left', fontsize=9, framealpha=0.9)
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def print_structured_summary(data: Dict, segments: Dict, num_regions: int, gap_hours: float, deadline_hours: float, output_path: Path,
                             *, print_region_segments: bool=False, min_window_hours: float=0.0, max_list: int=20):
    """Print structured summary for LLM parsing."""
    # Basic info
    strategy = data.get('strategy', {}).get('name', 'Unknown')
    cost = data.get('costs', [0])[0] if data.get('costs') else 0
    args = data.get('args', {})
    task_hours = args.get('task_duration_hours', [48])[0]
    history = data.get('history', [[]])[0]
    
    # Calculate final progress
    final_progress = 0
    if history:
        final_tick = history[-1]
        task_done = final_tick.get('Task/Done(seconds)', 0) / 3600
        final_progress = (task_done / task_hours * 100) if task_hours > 0 else 0
        final_progress = min(final_progress, 100.0)
    
    # Count instance types and migrations
    spot_count = 0
    ondemand_count = 0
    migration_count = 0
    restart_count = 0
    total_runtime = 0
    
    for region in range(num_regions):
        for seg_idx, seg in enumerate(segments[region]):
            start, end, inst_type, _, _ = seg if len(seg) == 5 else (*seg, False)
            duration = (end - start + 1) * gap_hours
            total_runtime += duration
            
            if inst_type == 'SPOT':
                spot_count += 1
            else:
                ondemand_count += 1
            
            # Check if migration
            is_migration = False
            for other_region in range(num_regions):
                if other_region != region:
                    for other_seg in segments[other_region]:
                        if other_seg[1] < start and start - other_seg[1] <= 5:
                            is_migration = True
                            migration_count += 1
                            break
                if is_migration:
                    break
            
            # Count restarts (non-first segments that aren't migrations)
            if seg_idx > 0 and not is_migration:
                restart_count += 1
    
    # Compact output format
    print(f"\n=== VISUALIZATION SUMMARY ===")
    print(f"Strategy: {strategy} | Cost: ${cost:.2f} | Progress: {final_progress:.1f}%")
    print(f"Instances: {spot_count}S+{ondemand_count}OD | Migrations: {migration_count} | Restarts: {restart_count}")
    
    # Timeline summary - ultra compact
    print(f"\nTimeline (h:type@region):")
    timeline_events = []
    for region in range(num_regions):
        for seg in segments[region]:
            start, end, inst_type, progress, _ = seg if len(seg) == 5 else (*seg, False)
            start_h = start * gap_hours
            end_h = (end + 1) * gap_hours
            type_abbr = 'S' if inst_type == 'SPOT' else 'OD'
            timeline_events.append((start_h, f"{start_h:.1f}-{end_h:.1f}:{type_abbr}@R{region}[{progress:.0f}%]"))
    
    # Sort by time and print
    timeline_events.sort(key=lambda x: x[0])
    for _, event_str in timeline_events:
        print(f"  {event_str}")
    
    # Key metrics
    print(f"\nKey Metrics:")
    print(f"  Total runtime: {total_runtime:.1f}h across all instances")
    print(f"  Deadline: {deadline_hours}h | Task: {task_hours}h")
    print(f"  Checkpoint size: {args.get('checkpoint_size_gb', 50)}GB")
    
    # Output location
    abs_path = os.path.abspath(output_path)
    print(f"\nVisualization saved: {abs_path}")

    if print_region_segments:
        # Build availability segments from trace files
        trace_files = args.get('trace_files', [])
        if isinstance(trace_files, str):
            trace_files = [p for p in trace_files.split(',') if p]
        elif not isinstance(trace_files, list):
            trace_files = list(trace_files) if trace_files else []

        single_trace = args.get('trace_file')
        if single_trace:
            if not trace_files:
                trace_files = [single_trace]
            elif single_trace not in trace_files:
                trace_files.append(single_trace)
        avail_map = extract_trace_availability(trace_files)
        print("\n=== PER-REGION SEGMENTS ===")
        for region in range(num_regions):
            print(f"\nRegion {region}")
            # Availability windows
            av = avail_map.get(region, [])
            av_segs = availability_to_segments(av)
            av_lines = []
            for s, e in av_segs:
                dur_h = (e - s + 1) * gap_hours
                if dur_h + 1e-9 < min_window_hours:
                    continue
                av_lines.append((s * gap_hours, (e + 1) * gap_hours, dur_h))
            av_lines = av_lines[:max_list]
            if av_lines:
                print("  Available windows (>=%.2fh):" % min_window_hours)
                for s_h, e_h, d_h in av_lines:
                    print(f"    {s_h:.1f}-{e_h:.1f} ({d_h:.1f}h)")
            else:
                print("  Available windows: none (or below threshold)")

            # Instance segments used
            used = segments.get(region, [])
            if used:
                print("  Used instances:")
                cnt = 0
                for seg in used:
                    s, e, inst_type, prog, _ = seg if len(seg) == 5 else (*seg, False)
                    s_h, e_h, d_h = s * gap_hours, (e + 1) * gap_hours, (e - s + 1) * gap_hours
                    print(f"    {s_h:.1f}-{e_h:.1f} {inst_type} ({d_h:.1f}h) end_progress={prog:.0f}%")
                    cnt += 1
                    if cnt >= max_list:
                        break
            else:
                print("  Used instances: none")


def create_timeline_segments(data: Dict, output_path: Path, *, print_region_segments: bool=False, min_window_hours: float=0.0, max_list: int=20):
    """Create timeline visualization with continuous segments."""
    history = data.get('history', [[]])[0]
    if not history:
        print("No history data found")
        return
    
    num_ticks = len(history)
    args = data.get('args', {})
    
    # Get trace files for spot availability
    trace_files = args.get('trace_files', [])
    if isinstance(trace_files, str):
        trace_files = [p for p in trace_files.split(',') if p]
    elif not isinstance(trace_files, list):
        trace_files = list(trace_files) if trace_files else []

    single_trace = args.get('trace_file')
    if single_trace:
        if not trace_files:
            trace_files = [single_trace]
        elif single_trace not in trace_files:
            trace_files.append(single_trace)

    spot_availability = extract_trace_availability(trace_files)
    num_regions = len(spot_availability) if spot_availability else 1
    
    # Figure setup - now with three subplots
    fig, (ax, ax_cost, ax_progress) = plt.subplots(3, 1, figsize=(16, num_regions * 1.5 + 7), 
                                       gridspec_kw={'height_ratios': [num_regions * 1.5, 3, 3]},
                                       sharex=True)
    
    # Time setup
    gap_seconds = data.get('env', {}).get('gap_seconds', 600)
    gap_hours = gap_seconds / 3600
    deadline_hours = args.get('deadline_hours', 52)
    
    # Calculate how many ticks we need to show up to deadline
    deadline_ticks = int(deadline_hours / gap_hours)
    
    # Track heights
    bar_height = 0.8
    region_spacing = 1.2
    
    # Draw each region
    for region in range(num_regions):
        y_base = (num_regions - region - 1) * region_spacing
        
        # Region label
        ax.text(-0.5, y_base + bar_height/2, f'Region {region}', 
                ha='right', va='center', fontsize=12, fontweight='bold')
        
        # Draw availability background up to deadline
        if region in spot_availability:
            availability = spot_availability[region]
            # Show availability up to deadline or available data, whichever is smaller
            max_tick = min(deadline_ticks, len(availability))
            for tick in range(max_tick):
                x = tick * gap_hours
                color = COLORS['spot_available'] if availability[tick] else COLORS['spot_unavailable']
                rect = patches.Rectangle(
                    (x, y_base), gap_hours, bar_height,
                    facecolor=color, edgecolor='none', alpha=0.5
                )
                ax.add_patch(rect)
    
    # Find and draw instance segments
    segments = find_instance_segments(history)
    for region in range(num_regions):
        segments.setdefault(region, [])
    
    for region in range(num_regions):
        y_base = (num_regions - region - 1) * region_spacing
        
        for seg_idx, segment_data in enumerate(segments[region]):
            # Handle both old format (4 elements) and new format (5 elements)
            if len(segment_data) == 5:
                start_tick, end_tick, inst_type, final_progress, had_overhead = segment_data
            else:
                start_tick, end_tick, inst_type, final_progress = segment_data
                had_overhead = False
                
            start_x = start_tick * gap_hours
            duration = (end_tick - start_tick + 1) * gap_hours
            
            color = COLORS['spot_instance'] if inst_type == 'SPOT' else COLORS['ondemand_instance']
            
            # Draw instance bar as one continuous segment
            rect = patches.Rectangle(
                (start_x, y_base + 0.1), duration, bar_height - 0.2,
                facecolor=color, edgecolor='black', linewidth=1.5
            )
            ax.add_patch(rect)
            
            # Add overhead bar at the beginning if there was restart overhead
            if had_overhead or seg_idx > 0 or (seg_idx == 0 and start_tick == 0):
                # Check if this is a migration (different region active before)
                is_migration = False
                for other_region in range(num_regions):
                    if other_region != region:
                        for other_seg in segments[other_region]:
                            # Check if other region was active just before this segment
                            if other_seg[1] < start_tick and start_tick - other_seg[1] <= 5:
                                is_migration = True
                                break
                    if is_migration:
                        break
                
                # Calculate overhead duration
                restart_overhead_hours = args.get('restart_overhead_hours', [0.2])[0]
                
                if is_migration:
                    # For migration, calculate actual migration time
                    from sky_spot.migration_model import get_transfer_time_hours
                    
                    # Find the source region (where we migrated from)
                    from_region = None
                    for other_region in range(num_regions):
                        if other_region != region:
                            for other_seg in segments[other_region]:
                                if other_seg[1] < start_tick and start_tick - other_seg[1] <= 5:
                                    from_region = other_region
                                    break
                            if from_region is not None:
                                break
                    
                    if from_region is not None:
                        # Get region names from trace files
                        from_region_name = os.path.basename(os.path.dirname(trace_files[from_region]))
                        to_region_name = os.path.basename(os.path.dirname(trace_files[region]))
                        
                        # Get checkpoint size
                        checkpoint_size_gb = args.get('checkpoint_size_gb', 50.0)
                        
                        # Calculate transfer time
                        transfer_hours = get_transfer_time_hours(
                            from_region_name, to_region_name, checkpoint_size_gb
                        )
                        
                        # Total migration time = startup + transfer
                        total_migration_hours = restart_overhead_hours + transfer_hours
                        
                        # Draw startup overhead (black) first
                        startup_width = min(restart_overhead_hours, duration)
                        startup_rect = patches.Rectangle(
                            (start_x, y_base + 0.1), startup_width, bar_height - 0.2,
                            facecolor='black', alpha=0.8, edgecolor='black', linewidth=1.5
                        )
                        ax.add_patch(startup_rect)
                        
                        # Draw transfer overhead (purple) after startup
                        if transfer_hours > 0 and duration > restart_overhead_hours:
                            transfer_width = min(transfer_hours, duration - restart_overhead_hours)
                            transfer_rect = patches.Rectangle(
                                (start_x + startup_width, y_base + 0.1), transfer_width, bar_height - 0.2,
                                facecolor='#8B008B', alpha=0.8, edgecolor='black', linewidth=1.5
                            )
                            ax.add_patch(transfer_rect)
                    else:
                        # Fallback: just draw startup overhead
                        overhead_width = min(restart_overhead_hours, duration)
                        overhead_rect = patches.Rectangle(
                            (start_x, y_base + 0.1), overhead_width, bar_height - 0.2,
                            facecolor='black', alpha=0.8, edgecolor='black', linewidth=1.5
                        )
                        ax.add_patch(overhead_rect)
                else:
                    # Regular restart overhead (just startup)
                    overhead_width = min(restart_overhead_hours, duration)
                    overhead_rect = patches.Rectangle(
                        (start_x, y_base + 0.1), overhead_width, bar_height - 0.2,
                        facecolor='black', alpha=0.8, edgecolor='black', linewidth=1.5
                    )
                    ax.add_patch(overhead_rect)
            
            # Add progress text at the right edge of the segment
            text_x = start_x + duration - 0.2
            text_y = y_base + bar_height/2
            ax.text(text_x, text_y, f'{final_progress:.0f}%', 
                   ha='right', va='center', fontsize=9, 
                   color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            
            # Add markers for cold start, restarts, and migrations
            # Cold start: first segment overall
            is_first_overall = start_tick == 0
            if is_first_overall:
                ax.text(start_x + 0.1, y_base + bar_height + 0.05, 'COLD START', 
                       ha='left', va='bottom', fontsize=8, 
                       color='blue', fontweight='bold')
            else:
                # Check if this is a migration (different region active before)
                is_migration = False
                for other_region in range(num_regions):
                    if other_region != region:
                        for other_seg in segments[other_region]:
                            # Check if other region was active just before this segment
                            if other_seg[1] < start_tick and start_tick - other_seg[1] <= 5:
                                is_migration = True
                                break
                    if is_migration:
                        break
                
                if is_migration:
                    ax.text(start_x + 0.1, y_base + bar_height + 0.05, 'MIGRATION', 
                           ha='left', va='bottom', fontsize=8, 
                           color='purple', fontweight='bold')
                elif seg_idx > 0:  # Not first segment in this region, so it's a restart
                    ax.text(start_x + 0.1, y_base + bar_height + 0.05, 'RESTART', 
                           ha='left', va='bottom', fontsize=8, 
                           color='red', fontweight='bold')
    
    # Configure axes
    ax.set_xlim(0, deadline_hours)
    ax.set_ylim(-0.2, num_regions * region_spacing)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_yticks([])
    ax.grid(True, axis='x', alpha=0.3)
    
    # Remove spines
    for spine in ['left', 'right', 'top']:
        ax.spines[spine].set_visible(False)
    
    # Add deadline
    ax.axvline(x=deadline_hours, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(deadline_hours, num_regions * region_spacing - 0.1, 'Deadline', 
            ha='center', va='bottom', color='red', fontsize=10, fontweight='bold')
    
    # Legend - place it in upper right corner, outside the plot area
    legend_elements = [
        patches.Patch(facecolor=COLORS['spot_available'], alpha=0.5, label='Spot Available'),
        patches.Patch(facecolor=COLORS['spot_unavailable'], alpha=0.5, label='Spot Unavailable'),
        patches.Patch(facecolor=COLORS['spot_instance'], label='SPOT Instance'),
        patches.Patch(facecolor=COLORS['ondemand_instance'], label='ON_DEMAND Instance'),
        patches.Patch(facecolor='black', alpha=0.8, label='Startup Overhead'),
        patches.Patch(facecolor='#8B008B', alpha=0.8, label='Transfer Overhead'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9,
              bbox_to_anchor=(1.12, 1.0))
    
    # Title
    strategy = data.get('strategy', {}).get('name', 'Unknown')
    cost = data.get('costs', [0])[0] if data.get('costs') else 0
    task_hours = args.get('task_duration_hours', [48])[0]
    title = f'Strategy: {strategy} | Cost: ${cost:.2f} | Task: {task_hours}h'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Plot cost curve
    plot_cost_curve(ax_cost, history, gap_hours, deadline_hours)
    
    # Plot progress curve
    plot_progress_curve(ax_progress, history, gap_hours, deadline_hours, task_hours)
    
    plt.tight_layout()
    
    # Always save to file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate structured console output for LLM parsing
    print_structured_summary(data, segments, num_regions, gap_hours, deadline_hours, output_path,
                             print_region_segments=print_region_segments,
                             min_window_hours=min_window_hours,
                             max_list=max_list)


def main():
    parser = argparse.ArgumentParser(description="Timeline visualization with segments")
    parser.add_argument("json_file", help="Path to simulation output JSON file")
    parser.add_argument("--output", "-o", default='output.png', help="Output image path (default: output.png)")
    parser.add_argument("--print-segments", action='store_true', help="Also print per-region availability and used instance segments")
    parser.add_argument("--min-window-hours", type=float, default=0.0, help="Only print availability windows with duration >= this (hours)")
    parser.add_argument("--max-list", type=int, default=20, help="Max lines to list per region for availability/instances")
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File {json_path} not found")
        return
    
    data = load_data(json_path)
    output_path = Path(args.output)
    create_timeline_segments(data, output_path,
                             print_region_segments=args.print_segments,
                             min_window_hours=args.min_window_hours,
                             max_list=args.max_list)


if __name__ == "__main__":
    main()
