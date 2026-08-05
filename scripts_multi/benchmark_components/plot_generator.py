"""Plot generation module - self-contained."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence
import json

from benchmark_components.scenario_config import DEFAULT_PARAMS
from benchmark_components.trace_utils import (
    find_region_base_path,
    normalize_data_roots,
)

logger = logging.getLogger(__name__)


_DEFAULT_TRACE_DATA_ROOTS = normalize_data_roots(
    DEFAULT_PARAMS.get("DATA_PATH", "data/converted_multi_region_aligned")
)

def get_strategy_display_name(strategy: str) -> str:
    """Get display name for strategy using heuristic naming with rc_cr special handling."""

    # Explicit mappings for specific strategies
    explicit_names = {
        "multi_region_rc_cr_threshold_eager_failover": "SkyPilot Eager Failover",
        "multi_region_availability_probe_simple": "Probe (Avail)",
        "multi_region_probe_cost_ratio": "Probe (Avail/Cost)",
        "unified_cost_model_oracle": "Next Spot Oracle",
        "multi_region_single_rc_cr_threshold": "Single UP",
        "multi_region_oracle_dp": "Oracle (DP)",
    }
    if strategy in explicit_names:
        return explicit_names[strategy]

    # Special handling for rc_cr patterns -> "Uniform Progress"
    if "rc_cr" in strategy:
        # Replace rc_cr with "Uniform Progress" in the name
        display_name = strategy.replace("rc_cr_threshold", "uniform_progress")
        display_name = strategy.replace("rc_cr", "uniform_progress")
        # Convert to title case
        words = display_name.split('_')
        display_name = ' '.join(word.capitalize() for word in words)
        return display_name

    # Default heuristic naming: remove underscores, capitalize words
    words = strategy.split('_')
    display_name = ' '.join(word.capitalize() for word in words)
    return display_name

# Global registry for sequential color assignment
_strategy_color_registry = {}
_strategy_color_index = 0

def get_strategy_color(strategy: str, mode_tag: str = "") -> str:
    """Get color for strategy using sequential assignment for maximum contrast.

    Args:
        strategy: Strategy name
        mode_tag: Mode tag like [Single/Avg], [Single/Best], [Multi] etc.
    """
    global _strategy_color_registry, _strategy_color_index

    # Maximally distinct color palette (ordered for contrast)
    colors = [
        "#e41a1c",  # Red
        "#377eb8",  # Blue
        "#4daf4a",  # Green
        "#ff7f00",  # Orange
        "#984ea3",  # Purple
        "#00ced1",  # Dark turquoise
        "#f781bf",  # Pink
        "#a65628",  # Brown
        "#ffff33",  # Yellow
        "#666666",  # Gray
        "#1b9e77",  # Teal
        "#d95f02",  # Dark orange
        "#7570b3",  # Slate blue
        "#e7298a",  # Magenta
        "#66a61e",  # Lime green
        "#e6ab02",  # Gold
        "#a6761d",  # Sienna
        "#222222",  # Black
    ]

    # Combine strategy and mode_tag for unique identification
    combined_key = f"{strategy}{mode_tag}"

    # Sequential assignment - each new strategy gets next color
    if combined_key not in _strategy_color_registry:
        _strategy_color_registry[combined_key] = colors[_strategy_color_index % len(colors)]
        _strategy_color_index += 1

    return _strategy_color_registry[combined_key]


def reset_strategy_colors():
    """Reset color registry for fresh assignment."""
    global _strategy_color_registry, _strategy_color_index
    _strategy_color_registry = {}
    _strategy_color_index = 0


def _mode_tag_from_row(task_type: str, trace_mode: str) -> str:
    if task_type == 'multi_region':
        return ''  # Multi-region strategies don't need tags (they're already explicitly multi-region)
    if task_type == 'union_pool':
        return ' [Union Pool]'
    if task_type == 'single_region':
        return ' [Single]'
    if task_type == 'trace_mode_baseline':
        if trace_mode == 'best_single':
            return ' [Best Single]'
        if trace_mode == 'average_single':
            return ' [Average Single]'
        return ' [Baseline]'
    return ''


def get_strategy_mode_tag(strategy_data_df: pd.DataFrame, strategy: str = None) -> str:
    """Derive a short mode tag from df rows for this strategy using task_type/trace_mode only.
    
    Args:
        strategy_data_df: DataFrame already filtered for a specific strategy/task_type combination
        strategy: Strategy name (not used)
    """
    if strategy_data_df.empty:
        return ''
    # Use the first row's task_type and trace_mode
    sample = strategy_data_df.iloc[0]
    task_type = sample['task_type']
    trace_mode = sample['trace_mode'] if 'trace_mode' in sample else None
    return _mode_tag_from_row(task_type, trace_mode)


def create_restart_overhead_plot(
    df: pd.DataFrame, 
    num_traces: int, 
    restart_overheads: List[float],
    output_dir: Path
) -> Path:
    """Generate restart overhead impact analysis plot."""
    logger.info(f"📊 Creating restart overhead analysis plot with {len(restart_overheads)} values")
    
    # Select up to 4 scenarios with the most data points
    scenario_counts = df['scenario_name'].value_counts()
    available_scenarios = list(scenario_counts.head(4).index)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(available_scenarios[:4]):
        ax = axes[idx]
        scenario_df = df[df['scenario_name'] == scenario]
        
        # Plot each strategy-task_type-trace_mode combination
        # Group by strategy, task_type, AND trace_mode to avoid mixing different uses of the same strategy
        grouping_cols = ['strategy', 'task_type', 'trace_mode']
        strategy_task_combos = scenario_df[grouping_cols].drop_duplicates()
        
        for _, row in strategy_task_combos.iterrows():
            strategy = row['strategy']
            task_type = row['task_type']
            trace_mode = row['trace_mode']
            
            # Filter for this specific strategy-task_type-trace_mode combination
            filter_mask = ((scenario_df['strategy'] == strategy) & 
                          (scenario_df['task_type'] == task_type) & 
                          (scenario_df['trace_mode'] == trace_mode))
            strategy_data = scenario_df[filter_mask]
            
            # Skip single_region entries (we have trace_mode_baseline for those)
            if task_type == 'single_region':
                continue
            
            # Group by restart_overhead
            overhead_costs = []
            for ro in restart_overheads:
                ro_data = strategy_data[strategy_data['restart_overhead'] == ro]
                if not ro_data.empty:
                    overhead_costs.append(ro_data['cost'].mean())
                else:
                    overhead_costs.append(np.nan)
            
            # Plot if we have data
            if any(not np.isnan(c) for c in overhead_costs):
                mode_tag = get_strategy_mode_tag(strategy_data, strategy)
                # Use dashed lines for baseline strategies and union pool
                is_baseline = task_type in ['trace_mode_baseline', 'union_pool']
                marker = 's' if is_baseline else 'o'
                linestyle = '--' if is_baseline else '-'
                ax.plot(restart_overheads, overhead_costs,
                       marker=marker, linewidth=2, markersize=6, linestyle=linestyle,
                       color=get_strategy_color(strategy, mode_tag),
                       label=f"{get_strategy_display_name(strategy)}{mode_tag}")
        
        ax.set_title(scenario, fontsize=12, fontweight='bold')
        ax.set_xlabel('Restart Overhead (hours)', fontsize=11)
        ax.set_ylabel('Mean Execution Cost ($)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(restart_overheads)
        
        # Adjust Y-axis
        _adjust_y_axis(ax, scenario_df)
    
    # Hide unused subplots
    for i in range(len(available_scenarios), 4):
        axes[i].set_visible(False)
    
    # Add a single legend for all subplots
    if available_scenarios:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    plt.suptitle(f'Restart Overhead Impact Analysis (n={num_traces})', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for legend
    
    plot_path = output_dir / f"restart_overhead_analysis_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Restart overhead analysis plot saved to {plot_path}")
    return plot_path


def create_checkpoint_size_plot(
    df: pd.DataFrame,
    num_traces: int,
    checkpoint_sizes: List[float],
    output_dir: Path
) -> Path:
    """Generate checkpoint size impact analysis plot."""
    logger.info(f"📊 Creating checkpoint size analysis plot with {len(checkpoint_sizes)} sizes")
    
    # Select up to 4 scenarios with the most data points
    scenario_counts = df['scenario_name'].value_counts()
    available_scenarios = list(scenario_counts.head(4).index)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(available_scenarios[:4]):
        ax = axes[idx]
        scenario_df = df[df['scenario_name'] == scenario]
        
        # Plot each strategy-task_type-trace_mode combination
        # Group by strategy, task_type, AND trace_mode to avoid mixing different uses of the same strategy
        grouping_cols = ['strategy', 'task_type', 'trace_mode']
        strategy_task_combos = scenario_df[grouping_cols].drop_duplicates()
        
        for _, row in strategy_task_combos.iterrows():
            strategy = row['strategy']
            task_type = row['task_type']
            trace_mode = row['trace_mode']
            
            # Filter for this specific strategy-task_type-trace_mode combination
            filter_mask = ((scenario_df['strategy'] == strategy) & 
                          (scenario_df['task_type'] == task_type) & 
                          (scenario_df['trace_mode'] == trace_mode))
            strategy_data = scenario_df[filter_mask]
            
            # Skip single_region entries (we have trace_mode_baseline for those)
            if task_type == 'single_region':
                continue
            
            # Group by checkpoint_size
            size_costs = []
            for cs in checkpoint_sizes:
                cs_data = strategy_data[strategy_data['checkpoint_size'] == cs]
                if not cs_data.empty:
                    size_costs.append(cs_data['cost'].mean())
                else:
                    size_costs.append(np.nan)
            
            # Plot if we have data
            if any(not np.isnan(c) for c in size_costs):
                mode_tag = get_strategy_mode_tag(strategy_data, strategy)
                # Use dashed lines for baseline strategies and union pool
                is_baseline = task_type in ['trace_mode_baseline', 'union_pool']
                marker = 's' if is_baseline else 'o'
                linestyle = '--' if is_baseline else '-'
                ax.plot(checkpoint_sizes, size_costs,
                       marker=marker, linewidth=2, markersize=6, linestyle=linestyle,
                       color=get_strategy_color(strategy, mode_tag),
                       label=f"{get_strategy_display_name(strategy)}{mode_tag}")
        
        ax.set_title(scenario, fontsize=12, fontweight='bold')
        ax.set_xlabel('Checkpoint Size (GB)', fontsize=11)
        ax.set_ylabel('Mean Execution Cost ($)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(checkpoint_sizes)
        
        # Adjust Y-axis
        _adjust_y_axis(ax, scenario_df)
    
    # Hide unused subplots
    for i in range(len(available_scenarios), 4):
        axes[i].set_visible(False)
    
    # Add a single legend for all subplots
    if available_scenarios:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    plt.suptitle(f'Checkpoint Size Impact Analysis (n={num_traces})', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for legend
    
    plot_path = output_dir / f"checkpoint_size_analysis_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Checkpoint size analysis plot saved to {plot_path}")
    return plot_path


def create_scenario_bar_plot(
    df: pd.DataFrame,
    num_traces: int,
    output_dir: Path,
    scenario_configs: List[Dict] = None,
    data_path: Optional[Path | str] = None,
) -> Path:
    """Generate main comparison bar plot across scenarios with trace statistics."""
    logger.info("📊 Creating scenario comparison bar plot with trace statistics")
    
    # Parse cost column (convert from string list to numeric)
    def parse_cost(cost_str):
        try:
            if isinstance(cost_str, str) and cost_str.startswith('['):
                # Parse string representation of list and take first element
                import ast
                cost_list = ast.literal_eval(cost_str)
                return float(cost_list[0]) if isinstance(cost_list, list) and len(cost_list) > 0 else float('nan')
            else:
                return float(cost_str)
        except:
            return float('nan')
    
    df['cost_numeric'] = df['cost'].apply(parse_cost)

    # Exclude raw single-region runs from comparison bars to avoid redundancy
    df_bar = df[df['task_type'] != 'single_region'].copy()

    # Group by scenario, strategy, task_type and trace_mode to keep variants separate
    grouped = (
        df_bar
        .groupby(['scenario_name', 'strategy', 'task_type', 'trace_mode'])['cost_numeric']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    grouped.rename(columns={'mean': 'cost', 'std': 'std', 'count': 'count'}, inplace=True)
    grouped['std'] = grouped.apply(
        lambda row: float(row['std']) if row['count'] > 1 and not np.isnan(row['std']) else 0.0,
        axis=1
    )
    
    # Get unique scenarios and strategy combinations
    scenarios = grouped['scenario_name'].unique()
    strategy_combos = grouped[['strategy', 'task_type', 'trace_mode']].drop_duplicates().values.tolist()
    
    # Create figure with cleaner style similar to availability stats plot
    try:
        plt.style.use('ggplot')
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(22, 8))  # Wider figure
    ax.set_facecolor('white')
    
    # Set up bar positions - use spacing > 1.0 to prevent overlap between scenario groups
    bar_group_width = 1.5  # Total width of all bars in one scenario group
    scenario_spacing = bar_group_width + 0.3  # Add gap between groups
    x = np.arange(len(scenarios)) * scenario_spacing
    width = bar_group_width / len(strategy_combos)  # Thicker bars
    
    # Plot bars for each strategy combination
    for i, combo in enumerate(strategy_combos):
        strategy, task_type, trace_mode = combo
        
        # Filter data for this combination, handling NaN in trace_mode
        if pd.isna(trace_mode):
            filter_mask = ((grouped['strategy'] == strategy) & 
                          (grouped['task_type'] == task_type) & 
                          (grouped['trace_mode'].isna()))
        else:
            filter_mask = ((grouped['strategy'] == strategy) & 
                          (grouped['task_type'] == task_type) & 
                          (grouped['trace_mode'] == trace_mode))
        strategy_data = grouped[filter_mask]
        costs = []
        errors = []
        for scenario in scenarios:
            scenario_rows = strategy_data[strategy_data['scenario_name'] == scenario]
            if len(scenario_rows) > 0:
                row = scenario_rows.iloc[0]
                costs.append(row['cost'])
                errors.append(row['std'])
            else:
                costs.append(np.nan)
                errors.append(0.0)

        offset = (i - len(strategy_combos)/2 + 0.5) * width
        mode_tag = _mode_tag_from_row(task_type, trace_mode)
        label = f"{get_strategy_display_name(strategy)}{mode_tag}"
        bars = ax.bar(
            x + offset,
            costs,
            width,
            label=label,
            color=get_strategy_color(strategy, mode_tag),
            alpha=0.9,
            edgecolor='white',
            linewidth=0.6,
            yerr=errors,
            capsize=5,
            error_kw={'elinewidth': 0.8, 'alpha': 0.7}
        )
        
        # Cost labels on bars disabled - too cluttered with many strategies
        # for bar, cost in zip(bars, costs):
        #     if not np.isnan(cost):
        #         height = bar.get_height()
        #         ax.text(bar.get_x() + bar.get_width()/2., height,
        #                f'${cost:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Scenarios', fontsize=12)
    ax.set_ylabel('Mean Execution Cost ($)', fontsize=12)
    ax.set_title(f'Strategy Performance Across Different Region Configurations (n={num_traces})', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis='y')
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_alpha(0.2)
    
    # Add trace availability statistics for each scenario
    if isinstance(data_path, dict):
        scenario_data_roots = {
            name: normalize_data_roots(paths)
            for name, paths in data_path.items()
        }
        default_trace_roots = list(_DEFAULT_TRACE_DATA_ROOTS)
    else:
        default_trace_roots = normalize_data_roots(
            data_path if data_path is not None else _DEFAULT_TRACE_DATA_ROOTS
        )
        scenario_data_roots = {}

    if scenario_configs:
        # Find y position for text (below the x-axis)
        y_min = ax.get_ylim()[0]
        y_text = y_min - (ax.get_ylim()[1] - y_min) * 0.15

        for i, scenario_name in enumerate(scenarios):
            # Find matching scenario config
            config = next((s for s in scenario_configs if s['name'] == scenario_name), None)
            if config:
                roots = scenario_data_roots.get(scenario_name, default_trace_roots)
                stats = calculate_scenario_trace_statistics(
                    scenario_name,
                    config['regions'],
                    data_path=roots,
                )
                if stats['overall']:
                    text = f"Avg Avail: {stats['overall']['mean']:.1%}\n({len(config['regions'])} regions)"
                    ax.text(i, y_text, text, ha='center', va='top', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    plot_path = output_dir / f"scenario_comparison_bar_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Scenario comparison bar plot saved to {plot_path}")
    return plot_path


def create_deadline_sensitivity_plot(
    df: pd.DataFrame,
    num_traces: int,
    deadline_ratios: List[float],
    output_dir: Path
) -> Path:
    """Generate deadline sensitivity analysis with dynamic subplots."""
    logger.info("📊 Creating deadline sensitivity analysis plot")
    
    # Get unique scenarios
    scenarios = sorted(df['scenario_name'].unique())
    n_scenarios = len(scenarios)
    
    # Determine subplot layout
    if n_scenarios == 1:
        rows, cols = 1, 1
    elif n_scenarios <= 4:
        rows, cols = 2, 2
    elif n_scenarios <= 6:
        rows, cols = 2, 3
    elif n_scenarios <= 9:
        rows, cols = 3, 3
    else:
        rows = int(np.ceil(np.sqrt(n_scenarios)))
        cols = int(np.ceil(n_scenarios / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 6*rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        scenario_df = df[df['scenario_name'] == scenario]
        
        # Define markers and linestyles for different strategies
        markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        
        # Plot each strategy-task_type-trace_mode combination to avoid mixing different uses
        grouping_cols = ['strategy', 'task_type', 'trace_mode']
        strategy_task_combos = scenario_df[grouping_cols].drop_duplicates()
        
        plot_idx = 0
        for _, row in strategy_task_combos.iterrows():
            strategy = row['strategy']
            task_type = row['task_type']
            trace_mode = row['trace_mode']
            
            # Filter for this specific strategy-task_type-trace_mode combination
            filter_mask = ((scenario_df['strategy'] == strategy) & 
                          (scenario_df['task_type'] == task_type) & 
                          (scenario_df['trace_mode'] == trace_mode))
            strategy_data = scenario_df[filter_mask]
            
            # Skip single_region entries (we have trace_mode_baseline for those)
            if task_type == 'single_region':
                continue
            
            # Group by deadline_ratio
            grouped = strategy_data.groupby('deadline_ratio')['cost'].agg(['mean', 'std', 'count'])
            
            if grouped.empty:
                continue
                
            x_values = grouped.index
            y_values = grouped['mean']
            
            # Add small y-offset to avoid complete overlap
            # Group strategies by similar cost ranges
            if strategy in ['multi_region_rc_cr_threshold', 'multi_region_rc_cr_no_cond2', 
                          'multi_region_rc_cr_randomized', 'multi_region_rc_cr_reactive']:
                y_offset = plot_idx * 0.3  # Small offset for overlapping multi-region strategies
            else:
                y_offset = 0
            
            mode_tag = get_strategy_mode_tag(strategy_data, strategy)
            ax.plot(x_values, y_values + y_offset,
                   marker=markers[plot_idx % len(markers)], 
                   linewidth=2.5, 
                   markersize=8, 
                   linestyle=linestyles[plot_idx % len(linestyles)],
                   color=get_strategy_color(strategy, mode_tag),
                   label=f"{get_strategy_display_name(strategy)}{mode_tag}",
                   alpha=0.85,  # Slight transparency
                   markeredgecolor='white',  # White edge on markers for visibility
                   markeredgewidth=0.5)
            
            plot_idx += 1
        
        ax.set_title(scenario, fontsize=11, fontweight='bold')
        ax.set_xlabel('Deadline Ratio', fontsize=10)
        ax.set_ylabel('Mean Cost ($)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at ratio=1.0
        ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.5)
        
        # Adjust Y-axis
        _adjust_y_axis(ax, scenario_df)
    
    # Hide unused subplots
    for i in range(n_scenarios, len(axes)):
        axes[i].set_visible(False)
    
    # Add a single legend for all subplots
    # Get handles and labels from the first subplot that has data
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

    plt.suptitle(f'Deadline Sensitivity Analysis (n={num_traces})', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.82, 0.96])  # Leave space for legend on right
    
    plot_path = output_dir / f"deadline_sensitivity_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Deadline sensitivity plot saved to {plot_path}")
    return plot_path


def create_cost_heatmap(
    df: pd.DataFrame,
    num_traces: int,
    deadline_ratios: List[float],
    checkpoint_sizes: List[float],
    output_dir: Path
) -> Path:
    """Generate cost heatmap for deadline × checkpoint interactions."""
    logger.info("📊 Creating cost heatmap (deadline × checkpoint)")
    
    # Get unique scenarios and strategies
    scenarios = sorted(df['scenario_name'].unique())
    strategies = sorted(df['strategy'].unique())
    
    # Calculate subplot layout
    total_subplots = len(scenarios) * len(strategies)
    cols = min(4, len(strategies))  # Max 4 columns
    rows = int(np.ceil(total_subplots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    subplot_idx = 0
    for scenario in scenarios:
        for strategy in strategies:
            if subplot_idx >= len(axes):
                break
                
            ax = axes[subplot_idx]
            
            # Filter data
            subset = df[(df['scenario_name'] == scenario) & (df['strategy'] == strategy)]
            
            # Create pivot table for heatmap
            pivot = subset.pivot_table(
                values='cost',
                index='checkpoint_size',
                columns='deadline_ratio',
                aggfunc='mean'
            )
            
            # Create heatmap
            if not pivot.empty:
                im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
                
                # Set ticks
                ax.set_xticks(np.arange(len(pivot.columns)))
                ax.set_yticks(np.arange(len(pivot.index)))
                ax.set_xticklabels([f'{x:.2f}' for x in pivot.columns], fontsize=8)
                ax.set_yticklabels([f'{int(y)}GB' for y in pivot.index], fontsize=8)
                
                # Add text annotations
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        value = pivot.values[i, j]
                        if not np.isnan(value):
                            text = ax.text(j, i, f'${value:.0f}',
                                         ha="center", va="center", color="black", fontsize=7)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
                
                ax.set_title(f'{scenario}\n{get_strategy_display_name(strategy)}', 
                           fontsize=9, fontweight='bold')
                ax.set_xlabel('Deadline Ratio', fontsize=8)
                ax.set_ylabel('Checkpoint Size', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{scenario}\n{get_strategy_display_name(strategy)}', 
                           fontsize=9, fontweight='bold')
            
            subplot_idx += 1
    
    # Hide unused subplots
    for i in range(subplot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Cost Heatmap: Deadline × Checkpoint Interaction (n={num_traces})', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = output_dir / f"cost_heatmap_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Cost heatmap saved to {plot_path}")
    return plot_path


def create_region_scaling_plot(
    df: pd.DataFrame,
    num_traces: int,
    output_dir: Path
) -> Path:
    """Generate region scaling benefit analysis."""
    logger.info("📊 Creating region scaling analysis plot")
    
    # Use num_regions field directly
    df = df.copy()
    df = df[df['num_regions'] > 0]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot for each strategy combination
    plotted_strategies = []
    strategy_combos = df[['strategy', 'task_type', 'trace_mode']].drop_duplicates().values.tolist()
    
    for combo in strategy_combos:
        strategy, task_type, trace_mode = combo
        
        # Filter for this specific combination
        filter_mask = ((df['strategy'] == strategy) & 
                      (df['task_type'] == task_type) & 
                      (df['trace_mode'] == trace_mode))
        strategy_data = df[filter_mask]
        
        # Skip if no valid data for this combination
        if strategy_data.empty or strategy_data['cost'].isna().all():
            logger.warning(f"No valid data for {strategy}/{task_type}/{trace_mode}, skipping")
            continue
        
        # Group by number of regions
        grouped = strategy_data.groupby('num_regions')['cost'].agg(['mean', 'std', 'count'])
        
        # Skip if no groups formed
        if grouped.empty:
            logger.warning(f"No region groups formed for {strategy}/{task_type}/{trace_mode}, skipping")
            continue
        
        x_values = grouped.index
        y_values = grouped['mean']
        yerr = grouped['std'] / np.sqrt(grouped['count'])  # Standard error
        
        # Visual styling based on task_type
        is_baseline = task_type in ['trace_mode_baseline', 'union_pool']
        marker = 's' if is_baseline else 'o'
        linestyle = '--' if is_baseline else '-'
        
        mode_tag = _mode_tag_from_row(task_type, trace_mode)
        ax.errorbar(x_values, y_values, yerr=yerr,
                   marker=marker, linewidth=2, markersize=8, linestyle=linestyle,
                   color=get_strategy_color(strategy, mode_tag),
                   label=f"{get_strategy_display_name(strategy)}{mode_tag}",
                   capsize=5, alpha=0.8)
        plotted_strategies.append(strategy)
    
    ax.set_xlabel('Number of Regions', fontsize=12)
    ax.set_ylabel('Mean Execution Cost ($)', fontsize=12)
    ax.set_title(f'Region Scaling Benefits Analysis (n={num_traces})', 
                fontsize=14, fontweight='bold')
    ax.set_xticks([2, 3, 5])
    ax.grid(True, alpha=0.3)
    
    # Only add legend if we actually plotted something
    if plotted_strategies:
        ax.legend(fontsize=10, loc='best')
    else:
        # Add text indicating no data available
        ax.text(0.5, 0.5, 'No valid data available for region scaling analysis', 
                transform=ax.transAxes, ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Add percentage improvement annotations
    # Calculate improvement from 2 to 5 regions for each strategy combination
    for combo in strategy_combos:
        strategy, task_type, trace_mode = combo
        
        # Filter for this specific combination
        filter_mask = ((df['strategy'] == strategy) & 
                      (df['task_type'] == task_type) & 
                      (df['trace_mode'] == trace_mode))
        strategy_data = df[filter_mask]
        
        costs_by_region = strategy_data.groupby('num_regions')['cost'].mean()
        
        if 2 in costs_by_region.index and 5 in costs_by_region.index:
            cost_2r = costs_by_region[2]
            cost_5r = costs_by_region[5]
            improvement = (cost_2r - cost_5r) / cost_2r * 100
            
            # Add text annotation
            mode_tag = _mode_tag_from_row(task_type, trace_mode)
            combo_idx = strategy_combos.index(combo)
            ax.text(0.02, 0.98 - 0.05 * combo_idx, 
                   f'{get_strategy_display_name(strategy)}{mode_tag}: {improvement:.1f}% improvement (2→5 regions)',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    plot_path = output_dir / f"region_scaling_analysis_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Region scaling analysis plot saved to {plot_path}")
    return plot_path


def create_detailed_parameter_comparison_grouped(
    df: pd.DataFrame,
    num_traces: int,
    checkpoint_sizes: List[float],
    deadline_ratios: List[float],
    restart_overheads: List[float],
    output_dir: Path
) -> List[Path]:
    """Generate detailed comparison plots with grouped bars for each task_duration × restart_overhead.

    Creates multiple files, one for each task_duration × restart_overhead combination.
    Each file shows deadline_ratio (rows) × checkpoint_size (cols) grid.
    Each subplot has grouped bars for different scenarios and strategies with error bars.
    """
    logger.info("📊 Creating detailed parameter comparison plots with grouped bars")

    # Reset color registry for fresh sequential assignment
    reset_strategy_colors()
    
    # Parse cost column (same as other functions)
    def parse_cost(cost_str):
        try:
            if isinstance(cost_str, str) and cost_str.startswith('['):
                import ast
                cost_list = ast.literal_eval(cost_str)
                return float(cost_list[0]) if isinstance(cost_list, list) and len(cost_list) > 0 else float('nan')
            else:
                return float(cost_str)
        except:
            return float('nan')
    
    df['cost_numeric'] = df['cost'].apply(parse_cost)
    
    # Exclude raw single-region runs
    df = df[df['task_type'] != 'single_region']
    
    # Get unique scenarios
    scenarios = sorted(df['scenario_name'].unique())
    n_scenarios = len(scenarios)
    
    # Setup style
    try:
        plt.style.use('ggplot')
    except Exception:
        pass
    
    generated_plots = []
    
    # Create a plot for each restart_overhead
    for restart_overhead in restart_overheads:
        # Filter data for this restart_overhead
        filtered_df = df[df['restart_overhead'] == restart_overhead]
        
        if filtered_df.empty:
            continue
        
        # Create figure with grid layout
        n_rows = len(deadline_ratios)
        n_cols = len(checkpoint_sizes)

        # Wider figure to accommodate many strategies
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14 * n_cols, 5 * n_rows))
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        # Plot each cell in the grid
        for row_idx, deadline_ratio in enumerate(deadline_ratios):
            for col_idx, checkpoint_size in enumerate(checkpoint_sizes):
                ax = axes[row_idx][col_idx]
                ax.set_facecolor('white')
                
                # Filter for this specific parameter combination
                cell_mask = (
                    (filtered_df['deadline_ratio'] == deadline_ratio) &
                    (filtered_df['checkpoint_size'] == checkpoint_size)
                )
                cell_df = filtered_df[cell_mask]
                
                if cell_df.empty:
                    ax.set_visible(False)
                    continue
                    
                # Prepare data for grouped bar chart
                # Group by scenario, strategy, task_type, and trace_mode to keep variants separate
                grouping_cols = ['scenario_name', 'strategy', 'task_type', 'trace_mode']
                grouped = cell_df.groupby(grouping_cols)['cost_numeric'].agg(['mean', 'std', 'count']).reset_index()

                # Split strategy combos into 'single' (trace_mode_baseline) and 'multi' (multi_region/union_pool)
                combos_all = grouped[grouping_cols[1:]].drop_duplicates().copy()
                # Remove oracle DP from bars (will be shown as dashed lower bound instead)
                if 'strategy' in combos_all.columns:
                    combos_all = combos_all[combos_all['strategy'] != 'multi_region_oracle_dp']
                def _bucket(tt: str) -> str:
                    if tt == 'trace_mode_baseline':
                        return 'single'
                    return 'multi'
                combos_all['bucket'] = combos_all['task_type'].apply(_bucket)
                single_combos = combos_all[combos_all['bucket'] == 'single'][['strategy','task_type','trace_mode']].values.tolist()
                multi_combos = combos_all[combos_all['bucket'] == 'multi'][['strategy','task_type','trace_mode']].values.tolist()
                n_singles, n_multis = len(single_combos), len(multi_combos)
                n_strategies = n_singles + n_multis

                if n_strategies == 0:
                    ax.set_visible(False)
                    continue

                # Setup bar positions with a gap between single and multi groups
                # Wider bars for better visibility
                total_width = 0.92
                group_gap = 0.04 if (n_singles > 0 and n_multis > 0) else 0.0
                # Ensure minimum bar width for visibility
                min_bar_width = 0.055
                width_single = max(min_bar_width, ((total_width - group_gap) * (n_singles / max(1, n_strategies)) / max(1, n_singles))) if n_singles > 0 else 0.0
                width_multi  = max(min_bar_width, ((total_width - group_gap) * (n_multis  / max(1, n_strategies)) / max(1, n_multis ))) if n_multis  > 0 else 0.0
                x_base = np.arange(n_scenarios)

                def _plot_bucket(combos, start_offset, bar_w):
                    for i, combo in enumerate(combos):
                        strategy, task_type = combo[0], combo[1]
                        trace_mode = combo[2] if len(combo) > 2 else None
                        mask = ((grouped['strategy'] == strategy) & (grouped['task_type'] == task_type) & (grouped['trace_mode'] == trace_mode))
                        strategy_data = grouped[mask]

                        means, stds = [], []
                        for scenario in scenarios:
                            srow = strategy_data[strategy_data['scenario_name'] == scenario]
                            if not srow.empty:
                                means.append(float(srow['mean'].values[0]))
                                sv = srow['std'].values[0]
                                stds.append(float(sv) if not pd.isna(sv) else 0.0)
                            else:
                                means.append(0.0)
                                stds.append(0.0)

                        x_pos = x_base + start_offset + i * bar_w
                        mode_tag = _mode_tag_from_row(task_type, trace_mode)
                        color = get_strategy_color(strategy, mode_tag)
                        ax.bar(x_pos, means, bar_w,
                               label=f"{get_strategy_display_name(strategy)}{mode_tag}",
                               color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
                        if any(std > 0 for std in stds):
                            ax.errorbar(x_pos, means, yerr=stds,
                                        fmt='none', ecolor='black', capsize=2, alpha=0.6, linewidth=0.9)

                left_edge = -total_width / 2.0
                single_offset = left_edge
                multi_offset  = left_edge + width_single * max(0, n_singles) + (group_gap if n_singles and n_multis else 0.0)
                if n_singles:
                    _plot_bucket(single_combos, single_offset + width_single / 2.0, width_single)
                if n_multis:
                    _plot_bucket(multi_combos, multi_offset + width_multi / 2.0, width_multi)
                    
                # Customize axes
                ax.set_xticks(x_base)
                ax.set_xticklabels([s[:15] for s in scenarios], rotation=45, ha='right', fontsize=7)

                # Add labels
                if col_idx == 0:
                    ax.set_ylabel(f'DR={deadline_ratio:.2f}\nCost ($)', fontsize=9)
                else:
                    ax.set_ylabel('Cost ($)', fontsize=8)
                
                # Add title for top row
                if row_idx == 0:
                    ax.set_title(f'Checkpoint={checkpoint_size:.0f}GB', fontsize=10, fontweight='bold')
                
                # Add grid
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add legend to first subplot only
                # Legend moved to figure-level (right sidebar); do not draw per-axis legend
                
                # Draw per-scenario lower-bound dashed line for multi_region_oracle_dp if present
                try:
                    oracle_df = grouped[(grouped['strategy'] == 'multi_region_oracle_dp') & (grouped['task_type'] == 'multi_region')]
                    if not oracle_df.empty:
                        oracle_means = {row['scenario_name']: float(row['mean']) for _, row in oracle_df.iterrows()}
                        for si, scen in enumerate(scenarios):
                            if scen in oracle_means:
                                y = oracle_means[scen]
                                xmin = x_base[si] - total_width / 2.0
                                xmax = x_base[si] + total_width / 2.0
                                ax.hlines(y, xmin, xmax, colors='black', linestyles='dashed', linewidth=1.0, alpha=0.7)
                                # Smaller font to fit bar width
                                ax.text(xmin - 0.01, y, f"O={y:.1f}", va='center', ha='right', fontsize=5, color='black')
                except Exception:
                    pass
            
        # Add overall title
        title = f'Restart Overhead={restart_overhead:.1f}h (n={num_traces})'
        plt.suptitle(title, fontsize=14, fontweight='bold')
        
        # Build a figure-level legend in a dedicated right area
        try:
            # Collect handles/labels from the first visible axis
            first_ax = None
            for r in range(n_rows):
                for c in range(n_cols):
                    if hasattr(axes[r][c], 'get_legend_handles_labels'):
                        first_ax = axes[r][c]
                        break
                if first_ax is not None:
                    break
            if first_ax is not None:
                handles, labels = first_ax.get_legend_handles_labels()
                if handles and labels:
                    # Place legend in a dedicated right panel
                    fig.legend(
                        handles, labels,
                        loc='center left',
                        bbox_to_anchor=(0.88, 0.5),
                        fontsize=9,
                        frameon=True,
                        fancybox=True,
                        shadow=False,
                        borderpad=1,
                        labelspacing=0.8,
                        handlelength=2.5,
                        handleheight=1.2,
                        facecolor='white',
                        edgecolor='lightgray',
                    )
        except Exception:
            pass

        # Adjust layout: reserve right margin for legend panel
        plt.tight_layout(rect=[0.0, 0.02, 0.85, 0.96])
        
        # Save figure
        filename = f"detailed_comparison_ro{restart_overhead:.2f}.png"
        plot_path = output_dir / filename
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {plot_path}")
        generated_plots.append(plot_path)
    
    logger.info(f"📊 Created {len(generated_plots)} detailed comparison plots")
    return generated_plots



def calculate_scenario_trace_statistics(
    scenario_name: str,
    regions: List[str],
    data_path: Optional[Path | str | Sequence[Path | str]] = None,
) -> Dict:
    """Calculate trace availability statistics for a scenario."""
    stats = {
        'scenario': scenario_name,
        'regions': {},
        'overall': {}
    }
    
    all_availabilities = []
    
    data_roots = normalize_data_roots(data_path if data_path is not None else _DEFAULT_TRACE_DATA_ROOTS)
    if not data_roots:
        logger.warning("No data paths provided for trace statistics.")
        return stats

    for region in regions:
        region_path = find_region_base_path(region, data_roots)
        if region_path is None:
            roots_str = ", ".join(str(p) for p in data_roots) or "<none>"
            logger.warning(
                f"Region '{region}' not found in provided data paths ({roots_str})."
            )
            continue
            
        region_availabilities = []
        
        # Analyze first 10 traces (or all if less)
        trace_files = sorted(region_path.glob("*.json"))[:10]
        
        for trace_file in trace_files:
            try:
                with open(trace_file, 'r') as f:
                    trace_data = json.load(f)
                # Convert from preempted format (0=available, 1=preempted)
                availability = 1 - np.mean(trace_data['data'])
                region_availabilities.append(availability)
                all_availabilities.append(availability)
            except:
                continue
        
        if region_availabilities:
            stats['regions'][region] = {
                'mean': np.mean(region_availabilities),
                'std': np.std(region_availabilities),
                'min': np.min(region_availabilities),
                'max': np.max(region_availabilities)
            }
    
    if all_availabilities:
        stats['overall'] = {
            'mean': np.mean(all_availabilities),
            'std': np.std(all_availabilities),
            'min': np.min(all_availabilities),
            'max': np.max(all_availabilities),
            'num_traces': len(all_availabilities)
        }
    
    return stats


def create_scenario_availability_plot(
    scenario_configs: List[Dict],
    output_dir: Path,
    data_path: Optional[Path | str | Sequence[Path | str] | Dict[str, Sequence[Path | str]]] = None,
) -> Path:
    """Create a detailed plot showing trace availability statistics for all scenarios."""
    logger.info("📊 Creating scenario availability statistics plot")
    
    # Calculate statistics for each scenario
    all_stats = []

    if isinstance(data_path, dict):
        scenario_data_roots = {
            name: normalize_data_roots(paths)
            for name, paths in data_path.items()
        }
        default_roots = list(_DEFAULT_TRACE_DATA_ROOTS)
    else:
        default_roots = normalize_data_roots(
            data_path if data_path is not None else _DEFAULT_TRACE_DATA_ROOTS
        )
        scenario_data_roots = {}

    if not default_roots and not scenario_data_roots:
        logger.error("No data paths provided for scenario availability plot.")
        return None

    for config in scenario_configs:
        roots = scenario_data_roots.get(config['name'], default_roots)
        stats = calculate_scenario_trace_statistics(
            config['name'],
            config['regions'],
            data_path=roots,
        )
        if stats['overall']:
            all_stats.append(stats)

    if not all_stats:
        roots_for_log: Dict[str, List[str]] = {}
        for cfg in scenario_configs:
            cfg_roots = scenario_data_roots.get(cfg['name'], default_roots)
            roots_for_log[cfg['name']] = [str(p) for p in cfg_roots]
        roots_str = ", ".join(f"{name}:{paths}" for name, paths in roots_for_log.items()) or "<none>"
        logger.warning(
            f"No availability statistics available - checked data paths: {roots_str}"
        )
        unique_roots = set()
        sources = list(scenario_data_roots.values()) or [default_roots]
        for roots in sources:
            for root in roots:
                if root in unique_roots:
                    continue
                unique_roots.add(root)
                if root.exists():
                    available_regions = [d.name for d in root.iterdir() if d.is_dir()]
                    logger.info(f"Available regions in data path {root}: {available_regions}")
                else:
                    logger.error(f"Data path does not exist: {root.absolute()}")
        return None
    
    # Create figure with subplots for each scenario
    n_scenarios = len(all_stats)
    rows = int(np.ceil(np.sqrt(n_scenarios)))
    cols = int(np.ceil(n_scenarios / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_scenarios == 1:
        axes = [axes]
    elif n_scenarios > 1:
        axes = axes.flatten()
    
    for idx, stats in enumerate(all_stats):
        ax = axes[idx]
        
        # Prepare data for plotting
        regions = list(stats['regions'].keys())
        region_means = [stats['regions'][r]['mean'] for r in regions]
        region_stds = [stats['regions'][r]['std'] for r in regions]
        
        # Short region names for x-axis
        short_regions = [r.split('_')[0] for r in regions]
        
        # Create bar plot with error bars
        x = np.arange(len(regions))
        bars = ax.bar(x, region_means, yerr=region_stds, capsize=5, 
                      alpha=0.8, color='skyblue', edgecolor='black')
        
        # Add value labels on bars
        for bar, mean in zip(bars, region_means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{mean:.1%}', ha='center', va='bottom', fontsize=8)
        
        # Add overall average line
        ax.axhline(stats['overall']['mean'], color='red', linestyle='--', 
                  label=f"Overall: {stats['overall']['mean']:.1%}")
        
        ax.set_xlabel('Region', fontsize=10)
        ax.set_ylabel('Availability Rate', fontsize=10)
        ax.set_title(f"{stats['scenario']}\n(n={stats['overall']['num_traces']} traces)", 
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(short_regions, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_scenarios, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Trace Availability Statistics by Scenario', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = output_dir / "scenario_availability_statistics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Scenario availability statistics plot saved to {plot_path}")
    return plot_path


def _adjust_y_axis(ax, scenario_df: pd.DataFrame) -> None:
    """Helper to adjust Y-axis range for better visibility."""
    if not scenario_df.empty:
        all_costs = scenario_df['cost'].dropna()
        if len(all_costs) > 0:
            cost_min, cost_max = all_costs.min(), all_costs.max()
            cost_range = cost_max - cost_min
            if cost_range > 0:
                margin = cost_range * 0.1
                ax.set_ylim(cost_min - margin, cost_max + margin)
