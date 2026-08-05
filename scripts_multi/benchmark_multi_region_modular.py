#!/usr/bin/env python3
"""
Modular multi-region benchmark script following Bacterial programming principles.
Each component is self-contained and can be easily copied/modified.
"""

import argparse
import logging
import os
import random
import shlex
import signal
import sys
import atexit
import time
import math
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional, Sequence

from benchmark_components.cache_manager import generate_cache_filename, load_from_cache, save_to_cache
from benchmark_components.simulation_runner import run_single_simulation
from benchmark_components.plot_generator import (
    create_restart_overhead_plot, 
    create_checkpoint_size_plot,
    create_scenario_bar_plot,
    create_deadline_sensitivity_plot,
    create_cost_heatmap,
    create_region_scaling_plot,
    create_scenario_availability_plot,
    create_detailed_parameter_comparison_grouped,
)
from benchmark_components.error_reporter import print_error_summary, log_simulation_failure
from benchmark_components.trace_utils import (
    get_trace_paths_for_task,
    get_min_full_trace_hours,
    normalize_data_roots,
)
from benchmark_components.scenario_config import (
    EXPERIMENT_SCENARIOS,
    DEFAULT_PARAMS,
    MULTI_REGION_STRATEGIES,
    UNION_POOL_STRATEGIES,
    SINGLE_REGION_STRATEGIES,
    get_segment_viz_strategies,
    get_data_roots_for_scenario,
)
from benchmark_components.configs import list_configs, load_config
from benchmark_components.skypilot_executor import execute_tasks_with_skypilot, get_active_clusters
from benchmark_components.batch_worker import (
    _history_file_exists,
    _run_simulation_with_history,
)

# Setup logging
_env_log_level = os.environ.get('LOG_LEVEL')
if _env_log_level:
    _env_log_level = _env_log_level.upper()
    _resolved_level = getattr(logging, _env_log_level, logging.INFO)
else:
    _resolved_level = logging.DEBUG if os.environ.get('DEBUG') else logging.INFO

logging.basicConfig(
    level=_resolved_level,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# If logging.basicConfig has no effect due to existing handlers (e.g. batch_worker initialized as INFO first),
# main script still needs to elevate root logger and its handlers to desired level so LOG_LEVEL=DEBUG can propagate.
root_logger = logging.getLogger()
root_logger.setLevel(_resolved_level)
for handler in root_logger.handlers:
    handler.setLevel(_resolved_level)

# Matplotlib font finding logs are too noisy even under DEBUG, force to INFO level.
logging.getLogger("matplotlib").setLevel(max(logging.INFO, _resolved_level))
logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

DEFAULT_INDEXED_TRACES = 5

_CANCEL_TRIGGERED = False


def _cancel_remote_clusters(reason: str) -> None:
    global _CANCEL_TRIGGERED
    clusters = get_active_clusters()
    if not clusters or _CANCEL_TRIGGERED:
        return
    _CANCEL_TRIGGERED = True
    cluster_list = ', '.join(clusters)
    logger.warning(f"{reason}; cancelling {len(clusters)} cluster(s): {cluster_list}")
    import sky

    for name in clusters:
        try:
            sky.cancel(name)
        except Exception as e:
            logger.error(f"Failed to cancel {name}: {e}")
    for name in clusters:
        try:
            sky.down(name)
        except Exception as e:
            logger.error(f"Failed to tear down {name}: {e}")


def _sigint_handler(signum, frame):
    _cancel_remote_clusters("Received Ctrl-C")
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _sigint_handler)
atexit.register(lambda: _cancel_remote_clusters("Process exiting"))


def create_simulation_task(
    scenario: Dict,
    strategy: str,
    trace_index: int,
    task_type: str,
    task_duration: float,
    checkpoint_size: float,
    restart_overhead: float,
    deadline_ratio: float,
    deadline_hours: float,
    env_start_hours: float = 0.0,
    use_full_trace: bool = False,
    enforce_window_bound: bool = False,
    region: Optional[str] = None,
    strategy_file: Optional[str] = None,
    data_paths: Optional[Sequence[str]] = None,
    window_index: Optional[int] = None,
) -> Dict:
    """Create a single simulation task dictionary."""
    serialized_data_paths = None
    if data_paths is not None:
        serialized_data_paths = [str(Path(p)) for p in data_paths]
    return {
        "scenario_name": scenario["name"],
        "regions_in_scenario": scenario["regions"],
        "num_regions": len(scenario["regions"]),
        "strategy": strategy,
        "trace_index": trace_index,
        "task_type": task_type,
        "trace_mode": "",  # Always include trace_mode to prevent float column in pandas
        "region": region,
        "strategy_file": strategy_file,
        "task_duration": task_duration,
        "checkpoint_size": checkpoint_size,
        "restart_overhead": restart_overhead,
        "deadline_ratio": deadline_ratio,
        "deadline_hours": deadline_hours,
        "env_start_hours": env_start_hours,
        "use_full_trace": use_full_trace,
        "enforce_window_bound": enforce_window_bound,
        "data_paths": serialized_data_paths,
        "window_index": window_index,
    }


def add_trace_mode_baselines(df: pd.DataFrame, single_region_strategies=None) -> pd.DataFrame:
    """Add trace mode baselines: run strategies on union/best_single/average_single trace modes.

    For single-region strategies, checkpoint_size doesn't affect the cost (no migration),
    so we calculate cost only for checkpoint_size=0 and replicate for other sizes.

    Args:
        df: Results dataframe
        single_region_strategies: List of (strategy_name, mode) tuples. Defaults to SINGLE_REGION_STRATEGIES.
    """
    if single_region_strategies is None:
        single_region_strategies = SINGLE_REGION_STRATEGIES

    # Filter single-region results for baseline calculations
    single_region_df = df[df["task_type"] == "single_region"].copy()

    if single_region_df.empty:
        logger.info("No single-region results found, skipping trace mode baseline calculations")
        return df

    # Calculate trace mode baselines for each scenario and strategy combination
    baseline_rows = []

    for scenario_name in single_region_df["scenario_name"].unique():
        scenario_data = single_region_df[single_region_df["scenario_name"] == scenario_name]

        # Get unique strategies tested in single regions
        strategies_tested = scenario_data["strategy"].unique()

        for strategy in strategies_tested:
            strategy_data = scenario_data[scenario_data["strategy"] == strategy]

            # Get all unique checkpoint sizes
            checkpoint_sizes = sorted(strategy_data['checkpoint_size'].unique()) if 'checkpoint_size' in strategy_data.columns else [0.0]

            # Group by parameters EXCEPT checkpoint_size (since it doesn't affect single-region cost)
            param_cols = ['restart_overhead', 'deadline_ratio', 'task_duration', 'deadline_hours']
            existing_params = [col for col in param_cols if col in strategy_data.columns]

            # Group by all parameters except checkpoint_size
            for _params, group_data in strategy_data.groupby(existing_params):
                # Group by region and calculate mean cost for each region
                # Use data from ANY checkpoint_size (they should all be the same for single-region)
                region_means = group_data.groupby("region")["cost"].mean().reset_index()

                if not region_means.empty and not region_means["cost"].isna().all():
                    # Find best and average single region costs
                    best_cost = region_means["cost"].min()
                    avg_cost = region_means["cost"].mean()

                    # Get a representative row for copying metadata
                    sample_row = group_data.iloc[0].copy()

                    # Find all configurations for this strategy
                    strategy_configs = [(name, mode) for name, mode in single_region_strategies if name == strategy]
                    
                    # Create baseline entries for EACH checkpoint_size (replicating the same cost)
                    for checkpoint_size in checkpoint_sizes:
                        for _, baseline_mode in strategy_configs:
                            if baseline_mode == "best":
                                # Create best single region entry for this strategy
                                best_row = sample_row.copy()
                                best_row.update({
                                    "strategy": strategy,
                                    "task_type": "trace_mode_baseline",
                                    "region": None,
                                    "cost": best_cost,
                                    "checkpoint_size": checkpoint_size,  # Set the checkpoint_size
                                    "migrations": 0,
                                    "trace_mode": "best_single",
                                    "base_strategy": strategy
                                })
                                baseline_rows.append(best_row)
                            
                            elif baseline_mode == "average":
                                # Create average single region entry for this strategy
                                avg_row = sample_row.copy()
                                avg_row.update({
                                    "strategy": strategy,
                                    "task_type": "trace_mode_baseline",
                                    "region": None,
                                    "cost": avg_cost,
                                    "checkpoint_size": checkpoint_size,  # Set the checkpoint_size
                                    "migrations": 0,
                                    "trace_mode": "average_single", 
                                    "base_strategy": strategy
                                })
                                baseline_rows.append(avg_row)
                    
                    # Log with parameter information
                    logger.debug(f"Added {strategy} baselines for {scenario_name} (all checkpoint sizes): best=${best_cost:.2f}, avg=${avg_cost:.2f}")
    
    # Add baseline rows to the original dataframe
    if baseline_rows:
        baseline_df = pd.DataFrame(baseline_rows)
        df = pd.concat([df, baseline_df], ignore_index=True)
        logger.info(f"Added {len(baseline_rows)} trace mode baseline entries to results")
    
    return df


def execute_simulation_with_cache(task: Dict, cache_dir: Path, params: Dict) -> Dict:
    """Execute simulation with caching support."""
    try:
        # Generate cache key - include task_type and trace_mode to differentiate variants
        base_strategy = task.get("strategy_file") or task["strategy"]
        task_type = task.get("task_type", "")
        trace_mode = task.get("trace_mode", "")

        # Create a unique cache key that includes variant information
        if task_type and task_type != "multi_region":
            if trace_mode:
                cache_key_strategy = f"{base_strategy}_{task_type}_{trace_mode}"
            else:
                cache_key_strategy = f"{base_strategy}_{task_type}"
        else:
            cache_key_strategy = base_strategy

        # Strategy flags change the result, so they have to change the key.
        strategy_argv = params.get("STRATEGY_ARGV")
        if strategy_argv:
            cache_key_strategy = f"{cache_key_strategy}[{'_'.join(strategy_argv)}]"

        requires_history = bool(params.get("SEGMENT_VIZ_ENABLED")) and (
            task.get("strategy") in get_segment_viz_strategies(task.get("scenario_name"))
        )

        # Generate environment type and trace paths
        data_path_spec = (
            task.get("data_paths")
            or params.get("DATA_PATH_BY_SCENARIO", {}).get(task.get("scenario_name"))
            or params["DATA_PATH"]
        )
        env_type, trace_paths = get_trace_paths_for_task(task, data_path_spec)

        # Use the deadline_hours that was calculated in the main function
        deadline_hours = task["deadline_hours"]
        env_start_hours = task.get("env_start_hours", params.get("ENV_START_HOURS", 0.0))

        cache_filename = generate_cache_filename(
            cache_key_strategy,
            env_type,
            trace_paths,
            task["checkpoint_size"],
            task["restart_overhead"],
            deadline_hours,
            task["task_duration"],
            env_start_hours,
            count_cross_region_migration_time=params.get("COUNT_CROSS_REGION_MIGRATION_TIME", False),
        )
        cache_file = cache_dir / cache_filename

        history_exists = _history_file_exists(task, params, env_start_hours)
        logger.debug(
            "Cache lookup: strategy=%s key=%s file=%s requires_history=%s history_exists=%s env_start=%.2f",
            task.get("strategy"),
            cache_key_strategy,
            cache_file,
            requires_history,
            history_exists,
            env_start_hours,
        )

        win_idx = task.get('window_index')
        win_tag = f"[W{win_idx}]" if win_idx is not None else ""

        cost = load_from_cache(cache_file)
        if cost is not None:
            logger.info(f"{win_tag} Cache HIT: {task.get('strategy')} -> ${cost:.2f}")
            if requires_history and not history_exists:
                logger.error(
                    "%s Cache HIT without history: scenario=%s strategy=%s env_start=%.2f",
                    win_tag,
                    task.get('scenario_name'),
                    task.get('strategy'),
                    env_start_hours,
                )
                raise RuntimeError(
                    "Segment visualization requires history data, but cache entry exists without a corresponding "
                    "history file. Delete the cache (or rerun without cache) so histories can be regenerated."
                )
            return {**task, "cost": cost, "migrations": -1}

        logger.info(f"{win_tag} Cache MISS: {task.get('strategy')}")

        if requires_history:
            logger.info(
                "%s Regenerating with history: scenario=%s strategy=%s env_start=%.2f",
                win_tag,
                task.get('scenario_name'),
                task.get('strategy'),
                env_start_hours,
            )
            result_record = _run_simulation_with_history(
                task=task,
                params=params,
                env_type=env_type,
                trace_paths=trace_paths,
                task_duration=task["task_duration"],
                deadline_hours=deadline_hours,
                task_env_start_h=env_start_hours,
            )
        else:
            cost, migrations, downtime, transfer, probe = run_single_simulation(
                task["strategy"],
                env_type,
                trace_paths,
                task["task_duration"],
                deadline_hours,
                task["restart_overhead"],
                task["checkpoint_size"],
                Path(params["OUTPUT_DIR"]) / "sim_temp",
                env_start_hours,
                task.get("strategy_file"),
                enforce_window_bound=params["ENFORCE_WINDOW_BOUND"],
                count_cross_region_migration_time=params.get("COUNT_CROSS_REGION_MIGRATION_TIME", False),
                gang_threshold=params.get("GANG_THRESHOLD"),
                strategy_argv=params.get("STRATEGY_ARGV"),
                dump_history=params.get("DUMP_HISTORY", False),
            )
            result_record = {"cost": cost, "migrations": migrations, "downtime_cost": downtime,
                             "transfer_cost": transfer, "probe_cost": probe}

        if not pd.isna(result_record.get("cost", float("nan"))):
            save_to_cache(cache_file, result_record["cost"])

        return {**task, **result_record}
        
    except ValueError as e:
        if "INFEASIBLE:" in str(e):
            # Task is infeasible - extract the detailed message
            error_msg = str(e).replace("INFEASIBLE: ", "")
            log_simulation_failure(task, e)
            return {**task, "cost": float('nan'), "migrations": 0, "error_type": "Task Infeasible", "error_details": error_msg}
        elif "TRACE_INSUFFICIENT:" in str(e):
            # Trace data is insufficient - extract the detailed message
            error_msg = str(e).replace("TRACE_INSUFFICIENT: ", "")
            log_simulation_failure(task, e)
            return {**task, "cost": float('nan'), "migrations": 0, "error_type": "Trace Insufficient", "error_details": error_msg}
        else:
            log_simulation_failure(task, e)
            return {**task, "cost": float('nan'), "migrations": 0, "error_type": "ValueError", "error_details": str(e)}
    except Exception as e:
        log_simulation_failure(task, e)
        return {**task, "cost": float('nan'), "migrations": 0, "error_type": type(e).__name__, "error_details": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Modular multi-region benchmark")

    # Config selection (NEW)
    parser.add_argument("--config", type=str, default=None,
                        help=f"Load experiment config. Available: {', '.join(list_configs())}. "
                             "Use --config help to see descriptions.")

    parser.add_argument("--num-traces", type=int, default=None,
                        help="When using full-trace auto-slicing, omit to use all K windows; otherwise, number of indexed traces (default 5)")
    parser.add_argument("--checkpoint-sizes", nargs='+', type=float, default=[50.0])
    parser.add_argument("--restart-overhead-hours", nargs='+', type=float, default=[0.2])
    parser.add_argument("--deadline-ratios", nargs='+', type=float, default=[1.083], 
                        help="Deadline as ratio of task duration (e.g., 1.083 = 52h/48h)")
    parser.add_argument("--deadline-hours", type=float, default=None,
                        help="Fixed deadline in hours (if not set, auto-calculated from traces)")
    parser.add_argument("--task-duration-hours", type=float, default=None,
                        help="Fixed task duration in hours (mutually exclusive with --deadline-ratios)")
    parser.add_argument("--scenarios", nargs='*', help="Specific scenarios to run")
    parser.add_argument("--output-dir", default=DEFAULT_PARAMS["OUTPUT_DIR"])
    parser.add_argument(
        "--data-path",
        nargs='+',
        default=None,
        help="Override data path(s) (directories containing <region> folders)."
             " Provide multiple paths to search in fallback order.")
    parser.add_argument("--skip-infeasible", action='store_true',
                        help="Skip tasks that are known to be infeasible (deadline < task_duration + overhead)")
    parser.add_argument("--use-full-trace", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable auto-slicing on full.json traces (default: enabled). Use --no-use-full-trace to disable.")
    parser.add_argument("--enforce-window-bound", action=argparse.BooleanOptionalAction, default=True,
                        help="Enforce runtime window bound equal to deadline (disable with --no-enforce-window-bound)")
    parser.add_argument("--cross-region-migration-time", action='store_true',
                        help="Add extra downtime for cross-region migrations (default overlaps the transfer).")

    
    # SkyPilot option
    parser.add_argument("--use-skypilot", action='store_true',
                        help="Use SkyPilot for parallel execution on cloud (auto-selects best resources)")
    parser.add_argument("--sky-max-clusters", type=int, default=None, help="Override SkyPilot cluster count (default auto)")
    parser.add_argument("--sky-tasks-per-cluster", type=int, default=None, help="Override number of tasks per SkyPilot cluster")
    # Segment grid visualization (optional, config-less)
    parser.add_argument("--segment-viz", action='store_true',
                        help="Render per-scenario segment grids comparing UCM vs Oracle-DP")
    parser.add_argument("--segment-viz-traces", type=int, default=30,
                        help="Number of traces per scenario to visualize (default: 30)")
    parser.add_argument("--segment-viz-scenarios", nargs='*', default=None,
                        help="Which scenario names to visualize (default: all present in results)")
    parser.add_argument("--progress-plot", action='store_true',
                        help="Show progress-time plots below each segment visualization")
    parser.add_argument("--only-strategies", nargs='*', default=None,
                        help="Restrict simulations to the specified strategy names (across multi-region/union/single).")
    parser.add_argument("--strategy-args", default=None, metavar="'FLAGS'",
                        help="Extra flags handed to every strategy's own parser, as one "
                             "shell-quoted string: --strategy-args='--probe-revalidate'. "
                             "Written with '=' because the value starts with a dash and "
                             "argparse would otherwise read it as a flag of this script. "
                             "Without this the runner clears sys.argv before building a "
                             "strategy, so policy knobs are stuck at their defaults. The "
                             "flags are part of the cache key, so two values do not collide "
                             "in one output directory.")
    parser.add_argument("--full-trace-windows", nargs='*', type=int, default=None,
                        help="When using --use-full-trace, restrict to these window indices (0-based).")
    parser.add_argument(
        "--full-trace-start-mode",
        choices=("sequential", "random", "stratified"),
        default="sequential",
        help="When using --use-full-trace without --full-trace-windows, choose how window starts are selected. "
             "'stratified' divides the trace into N equal strata and picks one random start per stratum, "
             "ensuring windows are spread across the entire trace with minimum gaps.")
    parser.add_argument(
        "--full-trace-random-seed",
        type=int,
        default=42,
        help="Seed for random start selection (default: 42 for reproducibility).",
    )
    # Local parallelism option
    parser.add_argument("--max-workers", type=int, default=0,
                        help="Local parallel worker count (0 = auto: min(os.cpu_count(), #tasks))")
    
    args = parser.parse_args()

    # =========================================================================
    # Handle --config: load experiment configuration
    # =========================================================================
    # These will be overridden if --config is specified
    active_multi_region_strategies = MULTI_REGION_STRATEGIES
    active_single_region_strategies = SINGLE_REGION_STRATEGIES
    active_scenarios = EXPERIMENT_SCENARIOS
    gang_threshold_from_config = None  # Gang-scheduling: None = disabled

    cfg_params: Dict = {}

    if args.config:
        if args.config == "help":
            print("\nAvailable experiment configurations:\n")
            for cfg_name in list_configs():
                try:
                    cfg = load_config(cfg_name)
                    desc = cfg.get("DESCRIPTION", "(no description)")
                    n_strategies = len(cfg.get("STRATEGIES", []))
                    n_scenarios = len(cfg.get("SCENARIOS", []))
                    print(f"  {cfg_name:20s} - {desc}")
                    print(f"                       {n_strategies} strategies, {n_scenarios} scenarios")
                except Exception as e:
                    print(f"  {cfg_name:20s} - (error loading: {e})")
            print("\nUsage: python benchmark_multi_region_modular.py --config <name> [--num-traces N]\n")
            return

        logger.info(f"Loading experiment config: {args.config}")
        try:
            config = load_config(args.config)
        except ValueError as e:
            parser.error(str(e))

        # Override strategies
        if config["STRATEGIES"]:
            active_multi_region_strategies = config["STRATEGIES"]
            logger.info(f"  Strategies: {len(active_multi_region_strategies)} "
                       f"({', '.join(active_multi_region_strategies[:3])}{'...' if len(active_multi_region_strategies) > 3 else ''})")

        # Override single-region strategies
        if config["SINGLE_REGION_STRATEGIES"]:
            active_single_region_strategies = config["SINGLE_REGION_STRATEGIES"]

        # Override scenarios
        if config["SCENARIOS"]:
            active_scenarios = config["SCENARIOS"]
            logger.info(f"  Scenarios: {len(active_scenarios)} "
                       f"({', '.join(s['name'] for s in active_scenarios[:2])}{'...' if len(active_scenarios) > 2 else ''})")

        # Override parameters (only if not explicitly set on CLI)
        cfg_params = config.get("PARAMS", {})
        if cfg_params.get("deadline_hours") and args.deadline_hours is None:
            args.deadline_hours = cfg_params["deadline_hours"]
            logger.info(f"  Deadline hours: {args.deadline_hours}")
        if cfg_params.get("deadline_ratios") and args.deadline_ratios == [1.083]:
            args.deadline_ratios = cfg_params["deadline_ratios"]
            logger.info(f"  Deadline ratios: {args.deadline_ratios}")
        if cfg_params.get("checkpoint_sizes") and args.checkpoint_sizes == [50.0]:
            args.checkpoint_sizes = cfg_params["checkpoint_sizes"]
        if cfg_params.get("restart_overhead_hours") and args.restart_overhead_hours == [0.2]:
            args.restart_overhead_hours = cfg_params["restart_overhead_hours"]
        if cfg_params.get("data_path") and args.data_path is None:
            args.data_path = [cfg_params["data_path"]] if isinstance(cfg_params["data_path"], str) else cfg_params["data_path"]

        # Gang-scheduling threshold (for multi-node experiments)
        gang_threshold_from_config = cfg_params.get("gang_threshold")

        # Sampling configuration (num_traces, start_mode)
        if cfg_params.get("num_traces") and args.num_traces is None:
            args.num_traces = cfg_params["num_traces"]
            logger.info(f"  Num traces: {args.num_traces}")
        if cfg_params.get("start_mode") and args.full_trace_start_mode == "sequential":
            args.full_trace_start_mode = cfg_params["start_mode"]
            logger.info(f"  Start mode: {args.full_trace_start_mode}")
        if cfg_params.get("random_seed") and args.full_trace_random_seed == 42:
            args.full_trace_random_seed = cfg_params["random_seed"]
        # A config may name the window indices its panel was measured on, so the
        # documented command stays `--config <name>`. Any window selection given on
        # the command line wins: explicit indices, a window count, or a start mode.
        # Without this guard the pinned indices take the explicit-window branch
        # below, which ignores --num-traces and --full-trace-start-mode outright.
        cli_chose_windows = (
            bool(args.full_trace_windows)
            or args.num_traces is not None
            or args.full_trace_start_mode != "sequential"
        )
        if cfg_params.get("full_trace_windows") and not cli_chose_windows:
            args.full_trace_windows = list(cfg_params["full_trace_windows"])
            logger.info(f"  Windows: {args.full_trace_windows}")

        logger.info(f"  Description: {config.get('DESCRIPTION', 'N/A')}")

    if args.full_trace_windows and not args.use_full_trace:
        parser.error("--full-trace-windows can only be used together with --use-full-trace")
    if args.full_trace_start_mode in ("random", "stratified") and not args.use_full_trace:
        parser.error(f"--full-trace-start-mode={args.full_trace_start_mode} requires --use-full-trace")

    if args.only_strategies:
        # Validate against the full strategy registry (auto-populated by
        # sky_spot.strategies on import) rather than the preset's whitelist, so
        # any registered strategy can be requested without adding a config for
        # it. This is what makes the V-variant sweeps (artifact/v_ablation.py)
        # runnable against the same scenarios as the main figures.
        from sky_spot.strategies.strategy import Strategy, MultiRegionStrategy
        multi_region_registry = {
            name for name, cls in Strategy.SUBCLASSES.items()
            if issubclass(cls, MultiRegionStrategy)
        }
        single_region_registry = {name for name, _ in active_single_region_strategies}
        available_strategy_names = (
            multi_region_registry
            | set(UNION_POOL_STRATEGIES)
            | single_region_registry
        )
        missing_strategies = sorted(set(args.only_strategies) - available_strategy_names)
        if missing_strategies:
            parser.error(f"Unknown strategy names in --only-strategies: {', '.join(missing_strategies)}")
        # Inject requested multi-region strategies into the runner list so the
        # task-building loops iterate over exactly what was asked for.
        requested_multi = [s for s in args.only_strategies if s in multi_region_registry]
        if requested_multi:
            active_multi_region_strategies = requested_multi
        allowed_strategies: Optional[set[str]] = set(args.only_strategies)
        logger.info("Restricting strategies to: %s", ", ".join(sorted(allowed_strategies)))
    else:
        allowed_strategies = None

    # Derive a human-friendly trace count label for outputs/plots
    if args.num_traces is not None:
        num_traces_label = args.num_traces
    elif args.use_full_trace:
        num_traces_label = "full"
    else:
        num_traces_label = DEFAULT_INDEXED_TRACES
    num_traces_label_str = str(num_traces_label)
    
    # Validate mutually exclusive arguments
    if args.task_duration_hours is not None and args.deadline_ratios != [1.083]:
        parser.error("Cannot specify both --task-duration-hours and --deadline-ratios")
    
    # Setup directories - include config name or timestamp as subdirectory
    base_output_dir = Path(args.output_dir)
    if args.config:
        # Use config name as subdirectory
        output_subdir = args.config
    else:
        # Use timestamp as subdirectory to avoid overwriting
        output_subdir = time.strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Cache is scoped to the config, not shared across configs. The cache key
    # already distinguishes every input that changes a simulation, but scoping
    # the directory as well means a bug in the key can only ever corrupt one
    # config's results instead of silently leaking across all five sweeps.
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Normalize data root specification early for deadline computation and downstream use
    data_path_spec = args.data_path if args.data_path is not None else DEFAULT_PARAMS["DATA_PATH"]
    data_roots = normalize_data_roots(data_path_spec)
    if not data_roots:
        parser.error("No valid data paths provided. Specify at least one directory containing trace data.")

    # Resolve the scenario list BEFORE using it anywhere below
    scenarios = active_scenarios
    if args.scenarios:
        scenarios = [s for s in scenarios if s["name"] in args.scenarios]

    # Scenario-specific data root mapping (per-scenario override keeps fallback order)
    scenario_data_roots: Dict[str, list[Path]] = {}
    scenario_data_paths_serialized: Dict[str, list[str]] = {}
    for scenario in scenarios:
        scen_name = scenario.get("name")
        roots = get_data_roots_for_scenario(scenario, data_path_spec)
        if not roots:
            parser.error(f"No data roots resolved for scenario '{scen_name}'. Provide valid data directories in scenario_config.")
        scenario_data_roots[scen_name] = roots
        scenario_data_paths_serialized[scen_name] = [str(p) for p in roots]

    # Build task list
    t0_build = time.time()
    all_tasks = []
    skipped_tasks = 0

    # Calculate global deadline if not specified
    if args.deadline_hours is None:
        from benchmark_components.scenario_config import calculate_global_deadline
        deadline_hours = calculate_global_deadline(
            scenarios,
            args.num_traces,
            scenario_data_roots_map=scenario_data_roots,
            default_spec=data_path_spec,
        )
        logger.info(f"🎯 Auto-calculated deadline: {deadline_hours:.2f}h (shortest trace across all scenarios)")
    else:
        deadline_hours = args.deadline_hours
        logger.info(f"🎯 Using fixed deadline: {deadline_hours:.2f}h")
    
    # Determine if we're using fixed task duration or ratio-based
    if args.task_duration_hours is not None:
        # Fixed task duration mode
        use_fixed_task_duration = True
        fixed_task_duration = args.task_duration_hours
        effective_ratios = [deadline_hours / fixed_task_duration]  # Calculate effective ratio for display
        logger.info(f"📋 Using fixed task duration: {fixed_task_duration:.2f}h (effective ratio: {effective_ratios[0]:.3f})")
    else:
        # Ratio-based mode (default)
        use_fixed_task_duration = False
        fixed_task_duration = None
        effective_ratios = args.deadline_ratios
        logger.info(f"📋 Using deadline ratios: {effective_ratios}")
    
    for scenario in scenarios:
        # Compute default K for full-trace mode per deadline later
        for checkpoint_size in args.checkpoint_sizes:
            for restart_overhead in args.restart_overhead_hours:
                for deadline_ratio in effective_ratios:
                    # Calculate task duration
                    if use_fixed_task_duration and fixed_task_duration is not None:
                        task_duration = fixed_task_duration
                    else:
                        task_duration = deadline_hours / deadline_ratio
                        logger.debug(f"Task duration {task_duration:.2f}h (deadline {deadline_hours:.2f}h ÷ ratio {deadline_ratio})")
                    
                    min_time_needed = task_duration + restart_overhead
                    
                    if args.skip_infeasible and min_time_needed > deadline_hours:
                        logger.info(f"Skipping infeasible configuration: task={task_duration}h + overhead={restart_overhead}h > deadline={deadline_hours}h")
                        # Estimate skipped count conservatively (cannot know K yet)
                        total_strategies_this_scenario = (
                            len(active_multi_region_strategies)
                            + len(UNION_POOL_STRATEGIES)
                            + len(active_single_region_strategies) * len(scenario["regions"])
                        )
                        # Use 0 for full-trace unknown K; or fallback num_traces
                        est_traces = 0 if args.use_full_trace else (args.num_traces or 5)
                        skipped_tasks += total_strategies_this_scenario * est_traces
                        continue

                    if args.use_full_trace:
                        # Full-trace auto-slicing: compute K and window starts
                        try:
                            min_hours, gap_seconds = get_min_full_trace_hours(
                                scenario["regions"], scenario_data_roots[scenario["name"]]
                            )
                        except FileNotFoundError as e:
                            logger.error(f"Full trace missing for scenario '{scenario['name']}': {e}")
                            continue
                        deadline_ticks = int(math.ceil(deadline_hours * 3600 / gap_seconds))
                        min_len_ticks = int(math.floor(min_hours * 3600 / gap_seconds))
                        if deadline_ticks <= 0:
                            logger.warning("Deadline too small; skipping this configuration")
                            continue
                        K = min_len_ticks // deadline_ticks
                        if K <= 0:
                            logger.info(
                                f"Scenario '{scenario['name']}' insufficient span for deadline={deadline_hours}h (min={min_hours:.2f}h). Skipping."
                            )
                            continue
                        start_ticks_list: list[int]
                        if args.full_trace_windows:
                            if args.full_trace_start_mode == "random":
                                logger.info(
                                    "Random start mode ignored for scenario '%s' because --full-trace-windows was provided.",
                                    scenario["name"],
                                )
                            requested_windows = sorted(set(args.full_trace_windows))
                            window_indices = [w for w in requested_windows if 0 <= w < K]
                            skipped_windows = [w for w in requested_windows if w < 0 or w >= K]
                            if skipped_windows:
                                logger.warning(
                                    "Requested window indices out of range for scenario '%s' (K=%d): %s",
                                    scenario["name"],
                                    K,
                                    ", ".join(str(w) for w in skipped_windows),
                                )
                            if not window_indices:
                                logger.warning(
                                    "No valid window indices remain for scenario '%s'; skipping configuration.",
                                    scenario["name"],
                                )
                                continue
                            logger.info(
                                "Auto-slicing full traces: K=%d windows of %.2fh; using explicit windows %s.",
                                K,
                                deadline_hours,
                                ", ".join(str(w) for w in window_indices),
                            )
                            start_ticks_list = [w * deadline_ticks for w in window_indices]
                        else:
                            if args.full_trace_start_mode in ("random", "stratified"):
                                max_start_tick_index = min_len_ticks - deadline_ticks
                                if max_start_tick_index < 0:
                                    logger.warning(
                                        "Auto-slicing full traces: scenario '%s' has insufficient ticks for random slicing; skipping.",
                                        scenario["name"],
                                    )
                                    continue
                                available_positions = max_start_tick_index + 1
                                if args.full_trace_random_seed is not None:
                                    # NOTE: scenario['name'] is intentionally excluded from seed
                                    # so that all scenarios use the same random window positions,
                                    # enabling fair cost comparison across different region sets.
                                    seed_material = (
                                        f"{args.full_trace_random_seed}:"
                                        f"{checkpoint_size}:"
                                        f"{restart_overhead}:"
                                        f"{deadline_ratio}:"
                                        f"{deadline_hours}"
                                    )
                                    rng = random.Random(seed_material)
                                else:
                                    seed_material = None
                                    rng = random.Random()
                                desired_windows = (
                                    args.num_traces if args.num_traces is not None else min(K, available_positions)
                                )
                                if desired_windows <= 0:
                                    logger.warning(
                                        "Auto-slicing full traces: scenario '%s' requested 0 windows; skipping configuration.",
                                        scenario["name"],
                                    )
                                    continue
                                if desired_windows > available_positions:
                                    logger.warning(
                                        "Auto-slicing full traces: scenario '%s' requested %d windows but only %d random start positions exist; clipping.",
                                        scenario["name"],
                                        desired_windows,
                                        available_positions,
                                    )
                                    desired_windows = available_positions

                                if args.full_trace_start_mode == "stratified":
                                    # Stratified sampling: divide available range into N equal strata,
                                    # pick one random position per stratum.
                                    # This ensures windows are spread across the entire trace with minimum gaps.
                                    stratum_width = available_positions // desired_windows
                                    if stratum_width < 1:
                                        stratum_width = 1
                                    start_ticks_list = []
                                    for i in range(desired_windows):
                                        stratum_start = i * stratum_width
                                        stratum_end = min(stratum_start + stratum_width, available_positions)
                                        if stratum_start >= available_positions:
                                            break
                                        pos = rng.randint(stratum_start, stratum_end - 1)
                                        start_ticks_list.append(pos)
                                    mode_desc = "stratified"
                                else:
                                    # Pure random sampling
                                    start_ticks_list = sorted(rng.sample(range(available_positions), k=desired_windows))
                                    mode_desc = "random"

                                seed_note = (
                                    f" seed_scope='{seed_material}'" if seed_material is not None else ""
                                )
                                logger.info(
                                    "Auto-slicing full traces: %.2fh window with %d/%d %s start positions%s.",
                                    deadline_hours,
                                    len(start_ticks_list),
                                    available_positions,
                                    mode_desc,
                                    seed_note,
                                )
                            else:
                                max_windows = K if (args.num_traces is None) else min(args.num_traces, K)
                                if max_windows <= 0:
                                    logger.warning(
                                        "Auto-slicing full traces: scenario '%s' requested 0 windows; skipping configuration.",
                                        scenario["name"],
                                    )
                                    continue
                                window_indices = list(range(max_windows))
                                logger.info(
                                    "Auto-slicing full traces: K=%d windows of %.2fh; using first %d sequential window(s).",
                                    K,
                                    deadline_hours,
                                    len(window_indices),
                                )
                                start_ticks_list = [w * deadline_ticks for w in window_indices]

                        # For SkyPilot remote execution, do not embed local data_paths into tasks
                        # to avoid remote path mismatches; rely on params["DATA_PATH"] mount instead.
                        _task_data_paths = None if args.use_skypilot else scenario_data_paths_serialized[scenario["name"]]

                        for win_idx, start_ticks in enumerate(start_ticks_list):
                            env_start_hours = start_ticks * gap_seconds / 3600.0

                            # Multi-region strategies
                            for strategy in active_multi_region_strategies:
                                if allowed_strategies is not None and strategy not in allowed_strategies:
                                    continue
                                all_tasks.append(
                                    create_simulation_task(
                                        scenario,
                                        strategy,
                                        trace_index=0,
                                        task_type="multi_region",
                                        task_duration=task_duration,
                                        checkpoint_size=checkpoint_size,
                                        restart_overhead=restart_overhead,
                                        deadline_ratio=deadline_ratio,
                                        deadline_hours=deadline_hours,
                                        env_start_hours=env_start_hours,
                                        use_full_trace=True,
                                        enforce_window_bound=args.enforce_window_bound,
                                        data_paths=_task_data_paths,
                                        window_index=win_idx,
                                    )
                                )

                            # Union pool strategies
                            for strategy in UNION_POOL_STRATEGIES:
                                task = create_simulation_task(
                                    scenario,
                                    strategy,
                                    trace_index=0,
                                    task_type="union_pool",
                                    task_duration=task_duration,
                                    checkpoint_size=checkpoint_size,
                                    restart_overhead=restart_overhead,
                                    deadline_ratio=deadline_ratio,
                                    deadline_hours=deadline_hours,
                                    env_start_hours=env_start_hours,
                                    use_full_trace=True,
                                    enforce_window_bound=args.enforce_window_bound,
                                    data_paths=_task_data_paths,
                                    window_index=win_idx,
                                )
                                task["trace_mode"] = "union"
                                # Provide window ticks so union can be sliced identically
                                task["window_start_ticks"] = start_ticks
                                task["window_length_ticks"] = deadline_ticks
                                all_tasks.append(task)

                            # Single-region comparison (use same K for comparability)
                            # Dedupe by strategy name to avoid generating duplicate single-region tasks.
                            for strategy_name in sorted({name for name, _ in active_single_region_strategies}):
                                if allowed_strategies is not None and strategy_name not in allowed_strategies:
                                    continue
                                for region in scenario["regions"]:
                                    all_tasks.append(
                                        create_simulation_task(
                                            scenario,
                                            strategy_name,
                                            trace_index=0,
                                            task_type="single_region",
                                            task_duration=task_duration,
                                            checkpoint_size=checkpoint_size,
                                            restart_overhead=restart_overhead,
                                            deadline_ratio=deadline_ratio,
                                            deadline_hours=deadline_hours,
                                            env_start_hours=env_start_hours,
                                            use_full_trace=True,
                                            enforce_window_bound=args.enforce_window_bound,
                                            region=region,
                                            data_paths=_task_data_paths,
                                            window_index=win_idx,
                                        )
                                    )
                    else:
                        # Legacy indexed traces flow
                        legacy_num_traces = args.num_traces or 5
                        # Multi-region strategies
                        for strategy in active_multi_region_strategies:
                            if allowed_strategies is not None and strategy not in allowed_strategies:
                                continue
                            for i in range(legacy_num_traces):
                                all_tasks.append(create_simulation_task(
                                    scenario, strategy, i, "multi_region",
                                    task_duration, checkpoint_size, restart_overhead, deadline_ratio, deadline_hours,
                                    data_paths=(None if args.use_skypilot else scenario_data_paths_serialized[scenario["name"]])
                                ))
                        # Union pool strategies
                        for strategy in UNION_POOL_STRATEGIES:
                            if allowed_strategies is not None and strategy not in allowed_strategies:
                                continue
                            for i in range(legacy_num_traces):
                                task = create_simulation_task(
                                    scenario, strategy, i, "union_pool",
                                    task_duration, checkpoint_size, restart_overhead, deadline_ratio, deadline_hours,
                                    data_paths=(None if args.use_skypilot else scenario_data_paths_serialized[scenario["name"]])
                                )
                                task["trace_mode"] = "union"  # Force union trace
                                all_tasks.append(task)
                        # Single-region comparison (deduped by strategy name)
                        for strategy_name in sorted({name for name, _ in active_single_region_strategies}):
                            if allowed_strategies is not None and strategy_name not in allowed_strategies:
                                continue
                            for region in scenario["regions"]:
                                for i in range(legacy_num_traces):
                                    all_tasks.append(create_simulation_task(
                                        scenario, strategy_name, i, "single_region",
                                        task_duration, checkpoint_size, restart_overhead, deadline_ratio, deadline_hours,
                                        region=region,
                                        data_paths=(None if args.use_skypilot else scenario_data_paths_serialized[scenario["name"]])
                                    ))
    t1_build = time.time()
    logger.info(f"🧮 Built {len(all_tasks)} simulation tasks in {t1_build - t0_build:.2f}s")
    if not all_tasks:
        # Without this the run dies further down inside ProcessPoolExecutor with
        # "max_workers must be greater than 0", which says nothing about the
        # actual cause. The usual cause is a scenario whose `data_paths` point
        # at a directory that does not hold the regions it asks for, so every
        # trace lookup logged "Required trace file not found" and no task was
        # built.
        sys.exit(
            "error: 0 simulation tasks were built.\n"
            "  Check that the scenario's data_paths contain a directory per "
            "region with a full.json inside.\n"
            f"  Data roots used: {[str(r) for r in data_roots]}\n"
            "  Run `bash artifact/prepare_data.sh` if the traces have not been "
            "unpacked yet."
        )
    logger.info(f"🚀 Running {len(all_tasks)} simulation tasks")
    if skipped_tasks > 0:
        logger.info(f"📋 Skipped {skipped_tasks} infeasible tasks (use --skip-infeasible to filter these)")
    
    # Create params dict with actual task duration
    params = DEFAULT_PARAMS.copy()
    if args.use_skypilot:
        # SkyPilot path expects a single directory to mount; choose the first available root
        preferred_root = None
        # Prefer the first scenario's first root if available, else global data_roots[0]
        try:
            first_scen = scenarios[0]["name"] if scenarios else None
            if first_scen and scenario_data_roots.get(first_scen):
                preferred_root = scenario_data_roots[first_scen][0]
        except Exception:
            preferred_root = None
        if preferred_root is None:
            preferred_root = data_roots[0]
        params["DATA_PATH"] = str(preferred_root)
        # Do not pass per-scenario lists to remote; they point to local host paths
        params.pop("DATA_PATH_BY_SCENARIO", None)
    else:
        params["DATA_PATH"] = [str(p) for p in data_roots]
        params["DATA_PATH_BY_SCENARIO"] = {
            name: paths for name, paths in scenario_data_paths_serialized.items()
        }
    params["ENFORCE_WINDOW_BOUND"] = args.enforce_window_bound
    params["COUNT_CROSS_REGION_MIGRATION_TIME"] = args.cross_region_migration_time
    # A config can carry its own strategy flags. --strategy-args adds to them and
    # overrides the ones it names, rather than discarding the lot: replacing them
    # wholesale means adding one unrelated flag silently drops every setting the
    # config relies on, and the run still looks like the config's.
    cli_argv = shlex.split(args.strategy_args) if args.strategy_args else []
    cfg_argv = shlex.split(cfg_params.get("strategy_args") or "")
    if cfg_argv:
        logger.info("  Strategy args from config: %s", cfg_params["strategy_args"])
    named_on_cli = {tok.split("=", 1)[0] for tok in cli_argv if tok.startswith("-")}
    merged, i = [], 0
    while i < len(cfg_argv):
        flag = cfg_argv[i]
        end = i + 1
        while end < len(cfg_argv) and not cfg_argv[end].startswith("-"):
            end += 1
        if flag.split("=", 1)[0] not in named_on_cli:
            merged.extend(cfg_argv[i:end])
        else:
            logger.info("  --strategy-args overrides config's %s", flag)
        i = end
    merged.extend(cli_argv)
    params["STRATEGY_ARGV"] = merged or None
    # Whether the cost includes the tick that completes the task. See
    # simulation_runner.run_single_simulation -- this is a reporting difference of
    # about half a percent, not a debug switch. Only the appendix's ablation
    # configs set it.
    params["DUMP_HISTORY"] = bool(cfg_params.get("dump_history", False))
    # Gang-scheduling threshold (16 = require all 16 nodes, None = disabled)
    params["GANG_THRESHOLD"] = gang_threshold_from_config
    # Segment viz controls for workers/viz step
    params["SEGMENT_VIZ_ENABLED"] = bool(args.segment_viz)
    params["SEGMENT_VIZ_TRACES"] = int(args.segment_viz_traces or 0)
    # Override OUTPUT_DIR from command-line args
    params["OUTPUT_DIR"] = args.output_dir
    # TASK_DURATION_HOURS is now stored in each task, not globally in params
    
    # Execute tasks based on execution mode
    if args.use_skypilot:
        total_tasks = len(all_tasks)
        env_clusters = os.environ.get("SKY_MAX_CLUSTERS")
        if args.sky_max_clusters is not None:
            cluster_count = max(1, args.sky_max_clusters)
        elif env_clusters:
            try:
                cluster_count = max(1, int(env_clusters))
            except ValueError:
                cluster_count = 1
        else:
            threshold = int(os.environ.get("SKY_AUTO_CLUSTER_THRESHOLD", "20000"))
            cap = int(os.environ.get("SKY_AUTO_CLUSTER_CAP", "8"))
            cluster_count = max(1, min(cap, math.ceil(total_tasks / threshold))) if total_tasks > threshold else 1
        os.environ["SKY_MAX_CLUSTERS"] = str(cluster_count)

        env_tpc = os.environ.get("SKY_TASKS_PER_CLUSTER")
        if args.sky_tasks_per_cluster is not None:
            tasks_per_cluster = max(1, args.sky_tasks_per_cluster)
        elif env_tpc:
            try:
                tasks_per_cluster = max(1, int(env_tpc))
            except ValueError:
                tasks_per_cluster = max(1, math.ceil(total_tasks / cluster_count))
        else:
            tasks_per_cluster = max(1, math.ceil(total_tasks / cluster_count))
        os.environ["SKY_TASKS_PER_CLUSTER"] = str(tasks_per_cluster)

        if "SKY_CPUS" in os.environ:
            cpus_setting = os.environ["SKY_CPUS"]
        else:
            # 32+ CPUs is sufficient for simulation workloads (~1.5GB/worker)
            cpus_setting = "32+"
            os.environ["SKY_CPUS"] = cpus_setting

        if "SKY_MEMORY" not in os.environ:
            # 64GB is sufficient for 32 workers at 1.5GB each + OS overhead
            os.environ["SKY_MEMORY"] = "64+"

        # Use N2 instance type by default (N4 has very low quota on most projects)
        if "SKY_INSTANCE_TYPE" not in os.environ:
            os.environ["SKY_INSTANCE_TYPE"] = "n2-standard-32"

        def _parse_cpus(value: str) -> int:
            digits = ''.join(ch for ch in value if ch.isdigit())
            return int(digits) if digits else 64

        if "WORKERS" not in os.environ:
            cpus_numeric = _parse_cpus(cpus_setting)
            workers = max(24, min(128, cpus_numeric))
            os.environ["WORKERS"] = str(workers)
        else:
            workers = int(os.environ.get("WORKERS", "64") or 64)

        if "WORKER_PER_GB" not in os.environ:
            os.environ["WORKER_PER_GB"] = os.environ.get("SKY_WORKER_PER_GB_DEFAULT", "1.0")

        memory_setting = os.environ.get("SKY_MEMORY", "128+")
        logger.info(
            f"🌩️  Using SkyPilot for cloud parallel execution (clusters={cluster_count}, tasks_per_cluster={tasks_per_cluster}, cpus={cpus_setting}, memory={memory_setting}, workers={os.environ.get('WORKERS')})"
        )
        results = execute_tasks_with_skypilot(
            all_tasks,
            cache_dir,
            params,
            auto_down=True  # Auto-terminate unless failures occur on a cluster
        )
    else:
        # Execute tasks in parallel locally (process-based to avoid GIL and maximize CPU)
        results: list[dict] = []
        total_tasks = len(all_tasks)
        # Allow env override; fallback to CLI; then auto
        env_workers = os.getenv("MULTI_BENCH_MAX_WORKERS")
        auto_workers = os.cpu_count() or 4
        max_workers = (
            int(env_workers) if env_workers and env_workers.isdigit() and int(env_workers) > 0
            else (args.max_workers if args.max_workers and args.max_workers > 0 else auto_workers)
        )
        max_workers = min(max_workers, total_tasks)
        logger.info(f"🧵 Local execution with {max_workers} worker processes (auto={auto_workers})")
        t0_exec = time.time()
        # ProcessPool gives independent interpreters; our function remains pickleable
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(execute_simulation_with_cache, task, cache_dir, params) for task in all_tasks]
            for fut in futures:
                results.append(fut.result())
        t1_exec = time.time()
        logger.info(f"✅ Completed {len(results)} tasks in {t1_exec - t0_exec:.2f}s")
    
    # Save results
    df = pd.DataFrame(results)
    
    # Add trace mode baseline calculations
    df = add_trace_mode_baselines(df, single_region_strategies=active_single_region_strategies)

    csv_path = output_dir / f"scenario_results_t{num_traces_label_str}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("=" * 60)
    logger.info(f"📊 RESULTS SUMMARY")
    logger.info(f"   Config: {args.config or 'custom'}")
    logger.info(f"   Total runs: {len(df)}")
    logger.info(f"   Strategies: {df['strategy'].nunique()}")
    logger.info(f"   Scenarios: {df['scenario_name'].nunique()}")
    logger.info(f"📁 Output file: {csv_path}")
    logger.info("=" * 60)

    # Generate plots
    logger.info("📊 Generating visualizations...")

    # 1. Main scenario comparison bar plot (always generate)
    # Get scenario configurations (reuse already-filtered scenarios from above)
    # Note: 'scenarios' variable was already set earlier in main() based on active_scenarios and args.scenarios
    create_scenario_bar_plot(
        df,
        num_traces_label,
        output_dir,
        scenarios,
        data_path=scenario_data_paths_serialized,
    )
    
    # 2. Restart overhead analysis (if multiple values)
    if len(args.restart_overhead_hours) > 1:
        create_restart_overhead_plot(df, num_traces_label, args.restart_overhead_hours, output_dir)
    
    # 3. Checkpoint size analysis (if multiple values)
    if len(args.checkpoint_sizes) > 1:
        create_checkpoint_size_plot(df, num_traces_label, args.checkpoint_sizes, output_dir)
    
    # 4. Deadline sensitivity analysis (if multiple deadline ratios)
    if len(args.deadline_ratios) > 1:
        create_deadline_sensitivity_plot(df, num_traces_label, args.deadline_ratios, output_dir)
    
    # 5. Cost heatmap (if both deadline and checkpoint have multiple values)
    if len(args.deadline_ratios) > 1 and len(args.checkpoint_sizes) > 1:
        create_cost_heatmap(df, num_traces_label, args.deadline_ratios, args.checkpoint_sizes, output_dir)
    
    # 6. Region scaling analysis (if there are multiple scenarios with different region counts)
    unique_region_counts = df['num_regions'].unique()
    if len(unique_region_counts) > 1:
        create_region_scaling_plot(df, num_traces_label, output_dir)
    
    # 7. Scenario availability statistics (always generate)
    create_scenario_availability_plot(
        scenarios,
        output_dir,
        data_path=scenario_data_paths_serialized,
    )

    # 8. Migration statistics (if migration data is available)
    if 'migrations' in df.columns:
        pass  # Migration visualization not implemented yet

    # 9. Detailed grouped comparison (always generate by default)
    try:
        chk_sizes = sorted(df['checkpoint_size'].dropna().unique().tolist()) if 'checkpoint_size' in df.columns else []
        ddl_ratios = sorted(df['deadline_ratio'].dropna().unique().tolist()) if 'deadline_ratio' in df.columns else []
        ro_hours = sorted(df['restart_overhead'].dropna().unique().tolist()) if 'restart_overhead' in df.columns else []
        if chk_sizes and ddl_ratios and ro_hours:
            create_detailed_parameter_comparison_grouped(
                df, num_traces_label, chk_sizes, ddl_ratios, ro_hours, output_dir
            )
    except Exception as e:
        logger.warning(f"Failed to generate detailed grouped plots: {e}")
    
    logger.info("📊 All visualizations generated successfully")
    
    # Print error summary
    print_error_summary(results)

    # Optional: segment grid visualizations per scenario (UCM vs Oracle-DP)
    if args.segment_viz:
        # Histories were saved by workers (for selected strategies) and downloaded above.
        # Rendering locally is lightweight (no simulation) and avoids extra clusters.
        scen_list = args.segment_viz_scenarios or sorted(df['scenario_name'].dropna().unique().tolist())
        # Filter strategies for segment viz to only those actually run
        segment_viz_strategies = active_multi_region_strategies
        if allowed_strategies is not None:
            segment_viz_strategies = [s for s in active_multi_region_strategies if s in allowed_strategies]
        try:
            from benchmark_components.segment_viz import render_segment_grids_scenarios
            render_segment_grids_scenarios(
                scenario_names=scen_list,
                num_traces=args.segment_viz_traces,
                params=params,
                results_df=df,
                output_dir=output_dir,
                show_progress_plot=args.progress_plot,
                strategies_override=segment_viz_strategies,
            )
        except Exception as e:
            # Explicitly fail: segment visualization is required in current workflow
            raise RuntimeError(f"Segment grid rendering failed: {e}") from e


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user.")
        _cancel_remote_clusters("KeyboardInterrupt")
        raise
