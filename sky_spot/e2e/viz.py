"""Visualization module for E2E testing.

This module provides functions for generating timeline visualizations
of simulation results.
"""

import logging
import os
from typing import List, Dict, Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# Colors for different cluster types
CLUSTER_COLORS = {
    "SPOT": "#22c55e",      # Green
    "ON_DEMAND": "#3b82f6",  # Blue
}


def generate_timeline_plot(
    history: List[Dict[str, Any]],
    region_names: List[str],
    task_duration_hours: float,
    strategy_name: str = "unified_cost_model_risk",
    output_path: str = "output/timeline.png",
):
    """Generate a timeline visualization showing instance segments across regions.

    Args:
        history: List of history dictionaries from the simulation
        region_names: List of region names (trace_files)
        task_duration_hours: Task duration in hours
        strategy_name: Name of the strategy being visualized
        output_path: Path to save the plot
    """
    if not history:
        logger.info("No history data to plot")
        return

    from sky_spot.e2e import config

    # Get deadline from Config if available
    if history[0].get("Config") and history[0]["Config"].get("DeadlineHours"):
        deadline_hours = history[0]["Config"]["DeadlineHours"]
    else:
        deadline_hours = task_duration_hours * config.DEADLINE_RATIO

    # Validate that data does not exceed deadline
    first_wall = history[0].get("WallTime")
    last_wall = history[-1].get("WallTime")
    if first_wall and last_wall:
        try:
            import datetime as dt_mod

            start_dt = dt_mod.datetime.fromisoformat(first_wall)
            end_dt = dt_mod.datetime.fromisoformat(last_wall)
            actual_duration_hours = (end_dt - start_dt).total_seconds() / 3600
            gap_hours = config.GAP_SECONDS / 3600
            if history[0].get("Config"):
                gap_hours = history[0]["Config"].get("GapSeconds", config.GAP_SECONDS) / 3600
            if actual_duration_hours > deadline_hours + gap_hours:
                raise ValueError(
                    f"Data exceeds deadline! Duration: {actual_duration_hours:.2f}h > "
                    f"Deadline: {deadline_hours:.2f}h (tolerance: {gap_hours:.4f}h)"
                )
        except ValueError:
            raise
        except Exception as e:
            logger.warning("Could not validate duration: %s", e)

    # Extract segments from history
    segments = []
    for entry in history:
        active = entry.get("ActiveInstances", {})
        wall_time = entry.get("WallTime")
        if not wall_time or not active:
            continue

        for region_idx, cluster_type in active.items():
            # region_idx is str from JSON, cluster_type is the value (e.g., "SPOT")
            segments.append({
                "region": int(region_idx),
                "cluster_type": cluster_type,
                "start": wall_time,  # Simplified: use wall_time as start
                "end": wall_time,
            })

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, max(4, len(region_names) * 0.6)))
    
    _draw_segments(
        ax=ax,
        segments=segments,
        region_names=region_names,
        task_duration_hours=task_duration_hours,
        deadline_hours=deadline_hours,
        strategy_name=strategy_name,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Timeline plot saved to {output_path}")


def _load_history_from_file(path: str) -> List[Dict[str, Any]]:
    """Load history from a JSON file."""
    import json
    
    with open(path, 'r') as f:
        return json.load(f)


def _infer_region_names_from_history(history: List[Dict[str, Any]]) -> List[str]:
    """Infer region names from history data."""
    if not history:
        return []
    
    # Get region names from Config if available
    if history[0].get("Config") and history[0]["Config"].get("RegionNames"):
        return history[0]["Config"]["RegionNames"]
    
    # Fallback: extract from active instances
    regions = set()
    for entry in history:
        active = entry.get("ActiveInstances", {})
        for region_idx in active.keys():
            # region_idx is str from JSON
            regions.add(int(region_idx))

    # Return generic names
    max_region = max(regions) if regions else 0
    return [f"R{i}" for i in range(max_region + 1)]
