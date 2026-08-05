import json
import logging
import os
import typing
from typing import Sequence, Optional

import tqdm

try:
    import wandb  # type: ignore
except Exception:  # Keep simulate importable without wandb installed

    class _WandbStub:
        run = None

        def init(self, *args, **kwargs):  # noqa: D401
            return None

    wandb = _WandbStub()  # type: ignore
import numpy as np

from sky_spot import env as env_lib
from sky_spot.strategies import strategy as strategy_lib
from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType, wandb_log
from sky_spot import task as task_lib

logger = logging.getLogger(__name__)


def _simulate_one(
    env: env_lib.Env, strategy: strategy_lib.Strategy, record_history: bool = True
):
    # When record_history=False, we only retain the last snapshot to minimize memory.
    history = [] if record_history else None
    last_snapshot = None
    last_request_type = ClusterType.NONE

    # Check if this is a multi-region setup
    is_multi_region = isinstance(strategy, MultiRegionStrategy) and isinstance(
        env, env_lib.MultiTraceEnv
    )
    multi_env = typing.cast(env_lib.MultiTraceEnv, env)
    multi_strategy = typing.cast(MultiRegionStrategy, strategy)

    # Initialize for type checker (will be set in multi-region branch before use)
    active_instances_snapshot: dict[int, str] = {}

    while not strategy.task_done:
        if is_multi_region:
            multi_env.observe()
            # Snapshot active instances BEFORE preemption handling for history recording
            # This represents what was running during [tick-1, tick), consistent with
            # single-region's cluster_type_histroy which is appended before preemption check
            active_instances_snapshot = {
                k: v.name for k, v in multi_env.get_active_instances().items()
            }
            multi_env.update_strategy_progress(
                multi_strategy
            )  # Update task progress based on PREVIOUS tick (includes preemption handling)

            # Check if task became done after progress update
            if strategy.task_done:
                break
            # Enforce deadline AFTER progress update to respect tick semantics
            # See docs/multi_region_progress_model.md (Billing vs Progress Timing)
            # Reach or exceeds the deadline and the task is not done, then fail (equal to timeout)
            if env.elapsed_seconds >= strategy.deadline:
                raise ValueError(
                    f"Deadline exceeded: elapsed={env.elapsed_seconds:.3f}s >= deadline={strategy.deadline:.3f}s"
                )

            # Check safety net BEFORE strategy execution
            # Safety net will skip strategy actions if it has already latched
            multi_env.maybe_trigger_safety_net(multi_strategy)
            multi_env.execute_multi_strategy(multi_strategy)

            multi_env.tick += 1

            # Finalize costs for this tick (they will be recorded in next observe())
            # This ensures costs are properly tracked even for the last tick

            # For history tracking, use a representative request type
            # (could be enhanced to track all regions)
            active = multi_env.get_active_instances()
            if active:
                # Use the type of the first active instance as representative
                request_type = next(iter(active.values()))
            else:
                request_type = ClusterType.NONE
        else:
            # Single-region execution path with in-loop finalization to avoid OOB
            # Align timing with docs: first finalize previous tick's progress via step(),
            # then enforce deadline using boundary-aware check.
            request_type = strategy.step()
            # If task not yet done after progress update, enforce deadline at boundary
            # Same as multi-region: if the task is not done after progress update, enforce deadline at boundary
            if not strategy.task_done and env.elapsed_seconds >= strategy.deadline:
                raise ValueError(
                    f"Deadline exceeded: elapsed={env.elapsed_seconds:.3f}s >= deadline={strategy.deadline:.3f}s"
                )
            # If the task just became done after realizing the previous gap,
            # finalize this tick without probing next availability.
            if strategy.task_done:
                # Advance one tick with NONE to satisfy env.info() invariant
                # and avoid calling spot_available() at tick == len(trace).
                env.step(ClusterType.NONE)
            else:
                env.step(request_type)

        info = {
            "RequestType": last_request_type.value
            if not is_multi_region
            else request_type.value,
            **env.info(),
            **strategy.info(),
        }

        # Add multi-region specific info if applicable
        if is_multi_region:
            cost_breakdown = multi_env.get_cost_breakdown()
            # Convert ClusterType keys to strings for JSON serialization
            cost_by_type_str = {k.name: v for k, v in cost_breakdown["by_type"].items()}

            # Add spot availability snapshot at last completed tick to avoid OOB and
            # align with env.info() Timestamp semantics (tick-1)
            spot_availability = multi_env.get_spot_availability_snapshot()

            info.update(
                {
                    "ActiveRegions": len(active_instances_snapshot),
                    "CostByRegion": cost_breakdown["by_region"],
                    "CostByType": cost_by_type_str,
                    "SpotAvailability": spot_availability,
                    # Use snapshot from BEFORE preemption handling to represent what was
                    # running during [tick-1, tick), consistent with single-region
                    "ActiveInstances": active_instances_snapshot,
                    "MigrationCount": multi_env.migration_count,
                }
            )

        last_request_type = request_type
        if record_history:
            history.append(info)  # type: ignore[union-attr]
        else:
            # Keep only the last snapshot
            last_snapshot = info
        wandb_log(info)
        if env.tick % 100 == 0:
            logger.debug(f"==> Timestamp: {env.tick}")

    # Final step after task is done
    if is_multi_region:
        # Advance tick to satisfy env.info() invariant (tick == observed_tick + 1)
        # When we break after task_done, tick == observed_tick (both are T)
        # We need tick = T+1 for info() to work correctly
        # NOTE: We do NOT finalize costs here because no work was done during [T*gap, (T+1)*gap]
        # The work that completed the task was during [(T-1)*gap, T*gap], already billed in observe()
        multi_env.tick += 1
        multi_env.observed_tick = multi_env.tick - 1
    else:
        # Single-region: do not perform an extra step here.
        # The last tick was already realized inside the loop, and advancing again
        # would call observe()/spot_available() at the boundary.
        pass

    info = {
        "RequestType": ClusterType.NONE.value,
        **env.info(),
        **strategy.info(),
    }

    if is_multi_region:
        cost_breakdown = multi_env.get_cost_breakdown()
        info.update(
            {
                "ActiveRegions": 0,
                "CostByRegion": cost_breakdown["by_region"],
                "CostByType": cost_breakdown["by_type"],
            }
        )

    # Ensure we always return a list with at least the final snapshot at index -1
    if record_history:
        # Append the final info entry for multi-region to match single-region behavior
        # This records the tick where task_done=True with the final cost
        if is_multi_region:
            history.append(info)  # type: ignore[union-attr]
        final_history = history
    else:
        final_history = [last_snapshot if last_snapshot is not None else info]
    return final_history, env.tick


# Single-region strategies that have a drop-in multi-region counterpart. Used
# only to make the error message below actionable.
_MULTI_REGION_EQUIVALENT = {
    "rc_cr_threshold": "multi_region_single_rc_cr_threshold",
}


def _check_env_strategy_compatibility(
    envs: Sequence[env_lib.Env], strategy: strategy_lib.Strategy
) -> None:
    """Reject a single-region strategy paired with a multi-region environment."""
    if isinstance(strategy, MultiRegionStrategy):
        return
    if not any(isinstance(e, env_lib.MultiTraceEnv) for e in envs):
        return

    name = getattr(strategy, "NAME", type(strategy).__name__)
    equivalent = _MULTI_REGION_EQUIVALENT.get(name)
    hint = (
        f"Use --env=trace --trace-file <one region>/full.json to run it as the "
        f"single-region baseline it is"
    )
    if equivalent:
        hint += (
            f", or --strategy={equivalent} to run its multi-region-API "
            f"equivalent under --env=multi_trace"
        )
    raise ValueError(
        f"Strategy '{name}' is a single-region strategy and cannot run against a "
        f"multi-region environment (--env=multi_trace): it never sees the other "
        f"regions, so failover and the on-demand safety net never fire. {hint}."
    )


def simulate(
    envs: Sequence[env_lib.Env],
    strategy: strategy_lib.Strategy,
    task: task_lib.Task,
    trace_file: str,
    deadline_hours: float,
    restart_overhead_hours: list[float],
    env_start_hours: float,
    output_dir: str,
    kwargs: dict,
    output_filename: Optional[str] = None,
    silent: bool = False,
    dump_history: bool = True,
):
    # Pre-check: the strategy and the environment must agree on region count.
    # A single-region strategy handed a MultiTraceEnv silently falls through to
    # the single-region loop below, where the multi-region bookkeeping (failover,
    # safety net) never runs, and the job then blows its deadline several hundred
    # ticks later with an opaque "Deadline exceeded". Fail here instead, with the
    # invocation that actually works.
    _check_env_strategy_compatibility(envs, strategy)

    # Pre-check: Ensure task is feasible within deadline
    task_duration_hours = task.get_total_duration_hours()
    assert len(restart_overhead_hours) == 1, "Only one restart overhead is supported"
    max_restart_overhead = max(restart_overhead_hours) if restart_overhead_hours else 0
    min_time_needed = task_duration_hours + max_restart_overhead

    if min_time_needed > deadline_hours:
        # Task is infeasible with any strategy - return ON_DEMAND fallback cost
        # This represents the minimum cost to complete the task (pure ON_DEMAND)
        logger.warning(
            f"Task infeasible: minimum time needed ({min_time_needed:.2f}h = "
            f"task {task_duration_hours:.2f}h + max overhead {max_restart_overhead:.2f}h) "
            f"exceeds deadline ({deadline_hours:.2f}h). Returning ON_DEMAND fallback cost."
        )

        # Get the lowest ON_DEMAND price from all envs
        from sky_spot.utils import ClusterType
        min_od_price = float('inf')
        for e in envs:
            if hasattr(e, 'envs'):  # MultiTraceEnv
                for sub_e in e.envs:
                    od_price = sub_e.get_price()[ClusterType.ON_DEMAND]
                    min_od_price = min(min_od_price, od_price)
            else:
                od_price = e.get_price()[ClusterType.ON_DEMAND]
                min_od_price = min(min_od_price, od_price)

        # ON_DEMAND cost = price * (task_duration + cold_start)
        od_cost = min_od_price * min_time_needed
        logger.warning(f"  ON_DEMAND fallback cost: ${od_cost:.2f} ({min_od_price:.2f}/h * {min_time_needed:.2f}h)")

        # Return synthetic result matching the normal return format
        fallback_history = [{
            "Cost": od_cost,
            "MigrationCount": 0,
            "TransferCost": 0.0,
            "ComputeCost": od_cost,
            "ProbeCost": 0.0,
            "CostByType": {str(ClusterType.ON_DEMAND): od_cost},
            "task_done": True,
            "tick": 0,
            "infeasible_fallback": True,  # Flag to indicate this was infeasible
        }]

        # Return one result per env (to match normal stats format)
        return {
            "args": kwargs,
            "costs": [od_cost for _ in envs],
            "migrations": [0 for _ in envs],
            "transfer_costs": [0.0 for _ in envs],
            "probe_costs": [0.0 for _ in envs],
            "probe_cost_by_region": [{} for _ in envs],
            "probe_tick_counts": [0 for _ in envs],
            "migration_hours": [0.0 for _ in envs],
            "migrations_cross_az": [0 for _ in envs],
            "migrations_cross_region": [0 for _ in envs],
            "migrations_cross_continent": [0 for _ in envs],
            "downtime_costs": [0.0 for _ in envs],
            "strategy": strategy.config if hasattr(strategy, 'config') else {},
            "env": {},
            "task": task.get_config() if hasattr(task, 'get_config') else {},
            "history": [fallback_history for _ in envs] if dump_history else None,
            "ticks": [0 for _ in envs] if dump_history else None,
            "infeasible_fallback": True,  # Flag to indicate this was infeasible
        }

    # Pre-check: Ensure trace data has enough discrete ticks for task execution
    # This prevents out-of-bounds errors by checking discrete tick feasibility
    import math

    task_duration_seconds = task_duration_hours * 3600

    for i, env in enumerate(envs):
        if hasattr(env, "trace") and hasattr(env, "_start_index"):
            # Single-region environment
            if hasattr(env, "get_available_ticks"):
                available_ticks = env.get_available_ticks()  # type: ignore
            else:
                available_ticks = len(env.trace) - env._start_index
            gap_seconds = env.gap_seconds

            # Calculate required ticks for task completion (discrete tick constraint)
            required_ticks = math.ceil(task_duration_seconds / gap_seconds)

            if required_ticks > available_ticks:
                error_msg = (
                    f"Trace data insufficient: trace {i} has {available_ticks} ticks "
                    f"but task needs {required_ticks} ticks. "
                    f"Task duration {task_duration_hours:.2f}h = {task_duration_seconds}s "
                    f"with gap {gap_seconds}s requires {required_ticks} discrete ticks, "
                    f"but trace only provides {available_ticks} ticks."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        elif hasattr(env, "envs"):
            # Multi-region environment
            for region_idx, region_env in enumerate(env.envs):
                if hasattr(region_env, "trace") and hasattr(region_env, "_start_index"):
                    if hasattr(region_env, "get_available_ticks"):
                        available_ticks = region_env.get_available_ticks()  # type: ignore
                    else:
                        available_ticks = (
                            len(region_env.trace) - region_env._start_index
                        )
                    gap_seconds = region_env.gap_seconds

                    # Calculate required ticks for task completion (discrete tick constraint)
                    required_ticks = math.ceil(task_duration_seconds / gap_seconds)

                    if required_ticks > available_ticks:
                        error_msg = (
                            f"Trace data insufficient: region {region_idx} has {available_ticks} ticks "
                            f"but task needs {required_ticks} ticks. "
                            f"Task duration {task_duration_hours:.2f}h = {task_duration_seconds}s "
                            f"with gap {gap_seconds}s requires {required_ticks} discrete ticks, "
                            f"but region trace only provides {available_ticks} ticks."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

    histories = []
    costs = []
    ticks = []
    probe_costs = []
    probe_cost_by_region = []
    probe_tick_counts = []
    migrations = []
    migrations_cross_az = []
    migrations_cross_region = []
    migrations_cross_continent = []
    transfer_costs = []
    migration_hours = []
    downtime_costs = []

    # Keep incoming trace_file label as-is (human-readable). It's not used for filenames.
    env_name = envs[0].NAME
    env_config = envs[0].config
    # RESETTING STRATEGY IS VERY IMPORTANT!
    strategy.reset(envs[0], task)

    restart_overhead_str = "_".join(map(str, restart_overhead_hours))
    # Use compact, length-safe run name with stable hash (avoids filename-too-long on multi-region)
    import hashlib

    key_src = f"{env_name}|{trace_file}|{deadline_hours}|{task}|{restart_overhead_str}"
    short_id = hashlib.md5(key_src.encode("utf-8")).hexdigest()[:12]
    run_name = f"{strategy.name}-{env_name}-t={short_id}-ddl={deadline_hours:.4f}-over={restart_overhead_str}"
    logger.debug(run_name)
    if not silent and wandb.run is not None:
        wandb.run.name = run_name
        wandb.config.update(
            {
                "trace_file": trace_file,
                "deadline_hours": deadline_hours,
                "restart_overhead_hours": restart_overhead_hours,
                "env_start_hours": env_start_hours,
                "task_config": task.get_config(),
                "other_args": {
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "deadline_hours",
                        "task_duration_hours",
                        "task_duration_hours_2",
                        "restart_overhead_hours",
                        "env_start_hours",
                    ]
                },
            }
        )
        wandb.config.update({"env_metadata": env_config})
        wandb.config.update({"strategy_metadata": strategy.config})

    if silent:
        pbar = envs
    else:
        pbar = tqdm.tqdm(envs)
    for env in pbar:
        # pbar.set_description(f'env: {env}')
        # ! Must reset env and strategy at the first time!
        # ! reset is their init method
        env.reset()
        strategy.reset(env, task)

        logger.debug(kwargs)
        logger.debug(env)
        logger.debug(strategy)

        try:
            history, tick = _simulate_one(env, strategy, record_history=dump_history)
        except ValueError as e:
            # Emit a ready-to-run main.py reproduction command for this failed env
            try:
                strategy_name = getattr(strategy, "NAME", strategy.__class__.__name__)
                task_hours = task.get_total_duration_hours()
                over_hrs = restart_overhead_hours[0] if restart_overhead_hours else 0.0
                if isinstance(env, env_lib.TraceEnv):
                    trace_arg = env.config.get("trace_file", "(unknown)")
                    repro = (
                        f"python main.py --strategy={strategy_name} --env=trace "
                        f"--trace-file '{trace_arg}' --task-duration-hours={task_hours} "
                        f"--deadline-hours={deadline_hours} --restart-overhead-hours={over_hrs}"
                    )
                elif isinstance(env, env_lib.MultiTraceEnv):
                    # Build multi-trace repro from child envs
                    traces = [
                        subenv.config.get("trace_file", "(unknown)")
                        for subenv in env.envs
                    ]
                    tf_args = " ".join(f"'{t}'" for t in traces)
                    repro = (
                        f"python main.py --strategy={strategy_name} --env=multi_trace "
                        f"--trace-files {tf_args} --task-duration-hours={task_hours} "
                        f"--deadline-hours={deadline_hours} --restart-overhead-hours={over_hrs}"
                    )
                else:
                    repro = "(unrecognized env type; cannot form repro command)"
                logger.error(
                    "The run failed. Reproduction command, for reference: %s", repro
                )
            except Exception:
                pass
            raise
        histories.append(history)
        costs.append(history[-1]["Cost"])
        ticks.append(tick)

        # Extract migration count if available (multi-region)
        migration_count = history[-1].get("MigrationCount", 0)
        migrations.append(migration_count)
        # Extract transfer cost if available (multi-region)
        transfer_costs.append(float(getattr(env, "transfer_cost_total", 0.0)))
        probe_costs.append(float(getattr(env, "probe_cost_total", 0.0)))
        probe_cost_by_region.append(
            {
                str(k): float(v)
                for k, v in getattr(env, "probe_cost_by_region", {}).items()
            }
        )
        probe_tick_counts.append(int(getattr(env, "probe_tick_count", 0)))
        # Extract migration hours if available (multi-region)
        migration_hours.append(float(getattr(env, "migration_hours_total", 0.0)))
        # Extract migration breakdown counts from final snapshot
        final = history[-1]
        migrations_cross_az.append(int(final.get("CrossAZMigrations", 0)))
        migrations_cross_region.append(int(final.get("CrossRegionMigrations", 0)))
        migrations_cross_continent.append(int(final.get("CrossContinentMigrations", 0)))
        # Extract exact downtime billed cost from final snapshot (computed in env.info())
        downtime_costs.append(float(history[-1].get("DowntimeCost", 0.0)))

        # if len(envs) > 1:
        #     env.reset()
        #     new_args = copy.deepcopy(args)
        #     new_args.deadline_hours = 1000
        #     spot_strategy = sky_spot.strategies.only_spot.OnlySpotStrategy(new_args)
        #     spot_costs.append(simulate(env, spot_strategy))
        #     cost_ratio.append(costs[-1] / spot_costs[-1])

        # mean_strategy_cost = np.mean(costs)
        # std_strategy_cost = np.std(costs)
        # mean_spot_cost = np.mean(spot_costs)
        # std_spot_cost = np.std(spot_costs)
        # mean_cost_ratio = np.mean(cost_ratio)
        # std_cost_ratio = np.std(cost_ratio)
        # msg = f'cost: {mean_strategy_cost:.2f}±{std_strategy_cost:.2f}; spot cost: {mean_spot_cost:.2f}±{std_spot_cost:.2f}; cost ratio: {mean_cost_ratio:.2f}±{std_cost_ratio:.2f}'
        # logger.debug('=== ' + msg + ' ===')
        # pbar.set_description(msg)
        # wandb.log({'MeanCost': mean_strategy_cost, 'StdCost': std_strategy_cost, 'MeanSpotCost': mean_spot_cost, 'StdSpotCost': std_spot_cost, 'MeanCostRatio': mean_cost_ratio, 'StdCostRatio': std_cost_ratio})

    os.makedirs(output_dir, exist_ok=True)
    if output_filename is not None:
        run_name = output_filename
    stats = {
        "args": kwargs,
        "costs": costs,
        "migrations": migrations,
        "transfer_costs": transfer_costs,
        "probe_costs": probe_costs,
        "probe_cost_by_region": probe_cost_by_region,
        "probe_tick_counts": probe_tick_counts,
        "migration_hours": migration_hours,
        "migrations_cross_az": migrations_cross_az,
        "migrations_cross_region": migrations_cross_region,
        "migrations_cross_continent": migrations_cross_continent,
        "downtime_costs": downtime_costs,
        "strategy": strategy.config,
        "env": env_config,
        "task": task.get_config(),
    }
    if dump_history:
        stats.update(
            {
                "history": histories,
                "ticks": ticks,
            }
        )
    output_path = os.path.join(output_dir, f"{run_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f)
    # Emit canonical path for strict runner parsing
    # Use logger to avoid BrokenPipeError when stdout closes under ProcessPoolExecutor
    logger.info(f"Saved to {output_path}")
    mean_cost = np.mean(costs)
    std_cost = np.std(costs)
    p99_cost = np.percentile(costs, 99)
    p90_cost = np.percentile(costs, 90)
    logger.info(
        f"mean: {mean_cost}; std: {std_cost}; worst 1%: {p99_cost}; worst 10%: {p90_cost}"
    )
    return stats
