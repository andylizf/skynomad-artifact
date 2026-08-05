"""Wandb visualization module for E2E simulation.

Provides real-time timeline visualization using Plotly charts logged to wandb.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Global state
_wandb = None
_segments: List[Dict[str, Any]] = []
_start_time: Optional[str] = None
_region_names: List[str] = []


def init(project: str, name: str, region_names: List[str], config: Optional[Dict] = None):
    """Initialize wandb run."""
    global _wandb, _segments, _start_time, _region_names

    try:
        import wandb
        _wandb = wandb
        _wandb.init(project=project, name=name, config=config)
        _segments = []
        _start_time = None
        _region_names = region_names
        logger.info("[wandb] Initialized: %s/%s", project, name)
    except ImportError:
        logger.warning("[wandb] wandb not installed, skipping visualization")
        _wandb = None
    except Exception as e:
        logger.warning("[wandb] Failed to initialize: %s", e)
        _wandb = None


def log_tick(
    tick: int,
    progress_pct: float,
    cost: float,
    active_instances: Dict[int, str],  # {region_idx: cluster_type_name}
    wall_time: str,
    elapsed_hours: float,
    deadline_hours: float,
    extra_metrics: Optional[Dict] = None,
):
    """Log metrics and update timeline for a tick."""
    global _segments, _start_time

    if _wandb is None:
        return

    if _start_time is None:
        _start_time = wall_time

    # Update segments based on active instances
    elapsed_h = elapsed_hours
    for region_idx, cluster_type in active_instances.items():
        # Find or create segment for this region
        existing = None
        for seg in _segments:
            if seg["region"] == region_idx and seg.get("end") is None:
                existing = seg
                break

        if existing is None:
            # Start new segment
            _segments.append({
                "region": region_idx,
                "type": cluster_type,
                "start": elapsed_h,
                "end": None,
            })

    # Close segments for regions that are no longer active
    active_regions = set(active_instances.keys())
    for seg in _segments:
        if seg.get("end") is None and seg["region"] not in active_regions:
            seg["end"] = elapsed_h

    # Log basic metrics
    log_data = {
        "tick": tick,
        "progress": progress_pct,
        "cost": cost,
        "elapsed_hours": elapsed_h,
        "active_regions": len(active_instances),
    }

    # Log region status (1=active, 0=idle)
    for i, region_name in enumerate(_region_names):
        status = 1 if i in active_instances else 0
        log_data[f"region/{region_name}"] = status

    if extra_metrics:
        log_data.update(extra_metrics)

    # Generate and log timeline chart
    try:
        fig = _create_timeline_figure(elapsed_h, deadline_hours)
        if fig:
            log_data["charts/region_timeline"] = _wandb.Plotly(fig)
    except Exception as e:
        logger.debug("[wandb] Failed to create timeline: %s", e)

    _wandb.log(log_data)


def _create_timeline_figure(current_hours: float, deadline_hours: float):
    """Create Plotly timeline figure."""
    import plotly.graph_objects as go

    if not _segments:
        return None

    fig = go.Figure()

    # Better color scheme
    colors = {"SPOT": "#22c55e", "ON_DEMAND": "#3b82f6"}  # Green, Blue

    # Track which types we've added (for legend)
    legend_added = set()

    for seg in _segments:
        region_idx = seg["region"]
        region_name = _region_names[region_idx] if region_idx < len(_region_names) else f"R{region_idx}"
        start = seg["start"]
        end = seg["end"] if seg["end"] is not None else current_hours
        cluster_type = seg["type"]

        # Show legend only once per type
        show_legend = cluster_type not in legend_added
        if show_legend:
            legend_added.add(cluster_type)

        fig.add_trace(go.Bar(
            x=[end - start],
            y=[region_name],
            base=[start],
            orientation="h",
            marker=dict(
                color=colors.get(cluster_type, "#888888"),
                line=dict(width=0),
            ),
            name=cluster_type,
            showlegend=show_legend,
            legendgroup=cluster_type,
            hovertemplate=f"<b>{region_name}</b><br>{cluster_type}<br>{start:.2f}h → {end:.2f}h<extra></extra>",
        ))

    # Add deadline line
    fig.add_vline(
        x=deadline_hours,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text="Deadline",
        annotation_position="top right",
        annotation_font_size=10,
    )

    # Add current time line
    fig.add_vline(
        x=current_hours,
        line_dash="dot",
        line_color="#f97316",
        line_width=2,
        annotation_text="Now",
        annotation_position="bottom right",
        annotation_font_size=10,
    )

    # Calculate x-axis range
    x_max = max(deadline_hours * 1.05, current_hours * 1.1)

    fig.update_layout(
        title=dict(
            text="Region Activity Timeline",
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Time (hours)",
            range=[0, x_max],
            gridcolor="#e5e7eb",
            showgrid=True,
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=list(reversed(_region_names)),  # Top to bottom
        ),
        height=max(400, len(_region_names) * 60 + 100),
        barmode="overlay",
        bargap=0.3,
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=100, r=40, t=60, b=40),
    )

    return fig


def finish():
    """Finish wandb run."""
    if _wandb is None:
        return

    try:
        _wandb.finish()
        logger.info("[wandb] Run finished")
    except Exception as e:
        logger.warning("[wandb] Failed to finish: %s", e)
