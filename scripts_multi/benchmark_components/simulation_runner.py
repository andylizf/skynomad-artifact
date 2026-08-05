"""Simulation execution module - in-proc implementation.

Runs simulations by importing project modules directly (no subprocess),
which removes per-task process startup and parsing overhead while keeping
the same results and output artifacts.
"""

import json
import logging
import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Tuple
import configargparse
import sys

# Ensure project root is importable when running as a script (not as a module)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sky_spot.env import TraceEnv, MultiTraceEnv
from sky_spot.strategies import strategy as strategy_lib
from sky_spot.task import SingleTask
from sky_spot import simulate as simulate_lib

logger = logging.getLogger(__name__)


def _make_base_parser(deadline_hours, restart_overhead, output_dir, env_start_hours,
                      checkpoint_size) -> "configargparse.ArgumentParser":
    """The global fields every strategy parser starts from."""
    parser = configargparse.ArgumentParser(add_help=False)
    parser.add_argument('--deadline-hours', type=float, default=deadline_hours)
    parser.add_argument('--restart-overhead-hours', type=float, nargs='+', default=[restart_overhead])
    parser.add_argument('--output-dir', type=str, default=str(output_dir))
    parser.add_argument('--env-start-hours', type=float, default=env_start_hours)
    parser.add_argument('--checkpoint-size-gb', type=float, default=checkpoint_size)
    return parser


def _accepted_strategy_argv(StrategyClass, base_args, argv) -> List[str]:
    """Narrow `argv` to the flags this strategy's own parser declares.

    A sweep passes one --strategy-args list for every strategy it runs, but the
    strategies do not share a flag set: --probe-revalidate exists on the unified
    cost model and not on the DP oracle. Each _from_args ends in parse_args(),
    which exits the process on an unknown flag, so an unfiltered list would kill
    every run of every strategy that happens not to declare it.

    So: build a throwaway parser, let the strategy register its flags on it, and
    keep only what it declared. Anything dropped is logged -- silently ignoring a
    flag the caller asked for would misreport which configuration was measured.
    """
    if not argv:
        return []
    probe = _make_base_parser(*base_args)
    saved = sys.argv
    try:
        sys.argv = [saved[0]]
        StrategyClass._from_args(probe)
    except SystemExit:
        return []
    finally:
        sys.argv = saved

    declared = set()
    for action in probe._actions:
        declared.update(action.option_strings)

    kept: List[str] = []
    dropped: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("-"):
            i += 1
            continue
        name = token.split("=", 1)[0]
        j = i + 1
        while j < len(argv) and not argv[j].startswith("-"):
            j += 1
        (kept if name in declared else dropped).extend(argv[i:j])
        i = j

    if dropped:
        logger.info(
            "strategy %s does not declare %s; running it without those flags",
            StrategyClass.NAME, " ".join(dropped),
        )
    return kept


def run_single_simulation(
    strategy: str,
    env_type: str,
    trace_paths: List[str],
    task_duration_hours: float,
    deadline_hours: float,
    restart_overhead: float,
    checkpoint_size: float,
    output_dir: Path,
    env_start_hours: float = 0,
    strategy_file: Optional[str] = None,
    enforce_window_bound: bool = False,
    count_cross_region_migration_time: bool = False,
    gang_threshold: Optional[int] = None,
    strategy_argv: Optional[List[str]] = None,
    dump_history: bool = False,
) -> Tuple[float, int, float, float, float]:
    """Run one simulation in-proc.

    Returns (cost, migrations, downtime, transfer, probe). simulate() bills those
    last three separately from `cost`, and a policy pays all of them, so anything
    comparing policies against the optimum has to add them back.

    `dump_history` selects which snapshot the cost is read from, so it changes the
    reported figure and not just whether a history file is written. With it on, the
    final snapshot is rebuilt after the loop, so the cost runs through the tick that
    completes the task; with it off, the last snapshot taken inside the loop is
    used. See sky_spot/simulate.py:_simulate_one.

    Off is the default, which is the setting the figure sweeps use; the appendix's
    ablation configs turn it on. Compare results that were produced with the same
    setting.
    """

    # Build environment(s)
    window_hours: Optional[float] = deadline_hours if enforce_window_bound else None
    if env_type == "trace":
        envs = [TraceEnv(trace_paths[0], env_start_hours, window_hours, gang_threshold=gang_threshold)]
    else:
        envs = [
            MultiTraceEnv(
                trace_paths,
                env_start_hours,
                window_hours,
                count_cross_region_migration_time=count_cross_region_migration_time,
                gang_threshold=gang_threshold,
            )
        ]

    # Build task
    task = SingleTask(config={
        'duration': task_duration_hours,
        'checkpoint_size_gb': checkpoint_size,
    })

    # Build strategy via its parser defaults to avoid hand-maintaining args
    # We inject only the global fields we control as defaults.
    base_args = (deadline_hours, restart_overhead, output_dir, env_start_hours, checkpoint_size)
    parser = _make_base_parser(*base_args)

    StrategyClass = strategy_lib.Strategy.get(strategy)
    accepted = _accepted_strategy_argv(StrategyClass, base_args, list(strategy_argv or []))
    _argv = sys.argv
    try:
        # Avoid leaking outer CLI args into strategy parsers. `strategy_argv`
        # is the one exception: it is the caller's explicit list of strategy
        # flags (from --strategy-args), so a sweep can vary a policy knob
        # without a config per value.
        sys.argv = [_argv[0]] + accepted
        strat = StrategyClass._from_args(parser)
    finally:
        sys.argv = _argv

    # Run simulation
    stats = simulate_lib.simulate(
        envs=envs,
        strategy=strat,
        task=task,
        trace_file=trace_paths[0] if trace_paths else env_type,
        deadline_hours=deadline_hours,
        restart_overhead_hours=[restart_overhead],
        env_start_hours=env_start_hours,
        output_dir=str(output_dir),
        kwargs=vars(strat.args),
        dump_history=dump_history,
        silent=True,
    )

    # Extract scalar metrics
    costs = stats.get('costs', [])
    migs = stats.get('migrations', [])
    if not costs:
        raise ValueError("NO_COSTS: simulate() returned no costs")
    cost = float(costs[0])
    migrations = int(migs[0]) if migs else 0
    # Downtime is billed separately from Cost by env.py (Cost = compute +
    # transfer), but a policy that idles through a preemption is paying for it,
    # so anything comparing policies against the optimum needs it back.
    downtime = float((stats.get("downtime_costs") or [0.0])[0])
    transfer = float((stats.get("transfer_costs") or [0.0])[0])
    probe = float((stats.get("probe_costs") or [0.0])[0])
    return cost, migrations, downtime, transfer, probe


def run_single_simulation_with_history(
    *,
    strategy: str,
    env_type: str,
    trace_paths: List[str],
    task_duration_hours: float,
    deadline_hours: float,
    restart_overhead: float,
    checkpoint_size: float,
    output_dir: Path,
    env_start_hours: float = 0.0,
    strategy_file: Optional[str] = None,
    enforce_window_bound: bool = True,
    count_cross_region_migration_time: bool = False,
    gang_threshold: Optional[int] = None,
    strategy_argv: Optional[List[str]] = None,
) -> dict:
    """Run a single simulation and return the full stats dict (with history)."""
    # Build environment(s)
    window_hours: Optional[float] = deadline_hours if enforce_window_bound else None
    if env_type == "trace":
        envs = [TraceEnv(trace_paths[0], env_start_hours, window_hours, gang_threshold=gang_threshold)]
    else:
        envs = [
            MultiTraceEnv(
                trace_paths,
                env_start_hours,
                window_hours,
                count_cross_region_migration_time=count_cross_region_migration_time,
                gang_threshold=gang_threshold,
            )
        ]

    task = SingleTask(config={'duration': task_duration_hours, 'checkpoint_size_gb': checkpoint_size})

    # Build strategy using the provided deadline; window enforcement happens via envs
    base_args = (deadline_hours, restart_overhead, output_dir, env_start_hours, checkpoint_size)
    parser = _make_base_parser(*base_args)

    StrategyClass = strategy_lib.Strategy.get(strategy)
    accepted = _accepted_strategy_argv(StrategyClass, base_args, list(strategy_argv or []))
    _argv = sys.argv
    try:
        sys.argv = [_argv[0]] + accepted
        strat = StrategyClass._from_args(parser)
    finally:
        sys.argv = _argv

    stats = simulate_lib.simulate(
        envs=envs,
        strategy=strat,
        task=task,
        trace_file=trace_paths[0] if trace_paths else env_type,
        deadline_hours=deadline_hours,
        restart_overhead_hours=[restart_overhead],
        env_start_hours=env_start_hours,
        output_dir=str(output_dir),
        kwargs=vars(strat.args),
        dump_history=True,
        silent=True,
    )
    return stats


def check_simulation_errors(results: List[dict]) -> dict:
    """Analyze simulation results and report errors."""
    total_tasks = len(results)
    failed_tasks = sum(1 for r in results if np.isnan(r.get('cost', np.nan)))
    
    errors_by_strategy = {}
    for result in results:
        if np.isnan(result.get('cost', np.nan)):
            strategy = result.get('strategy', 'unknown')
            if strategy not in errors_by_strategy:
                errors_by_strategy[strategy] = 0
            errors_by_strategy[strategy] += 1
    
    return {
        'total_tasks': total_tasks,
        'failed_tasks': failed_tasks,
        'success_rate': (total_tasks - failed_tasks) / total_tasks * 100,
        'errors_by_strategy': errors_by_strategy
    }
