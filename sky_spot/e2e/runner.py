"""Runner module for E2E simulation.

This module contains the main run_simulation function that orchestrates
the end-to-end simulation of multi-region spot instance scheduling.
"""

import argparse
import datetime
import io
import json
import logging
import os
import threading
import time
import traceback
from typing import Dict, Any, List

import configargparse

from sky_spot.env import MultiTraceEnv
from sky_spot.strategies import strategy as strategy_lib
from sky_spot.task import SingleTask
from sky_spot.utils import ClusterType

from sky_spot.e2e import config
from sky_spot.e2e.console import SimulationConsole
from sky_spot.e2e.cluster import (
    sim_console,
    probe_event,
    _create_bucket,
    _get_cluster_name,
    _get_cluster_record_with_refresh,
    _actual_launch_internal,
    _actual_terminate_internal,
    _actual_probe_internal,
    _actual_check_is_preempted_internal,
    _actual_check_ondemand_health_internal,
    _patch_restart_overheads,
    _compute_total_progress,
    _probe_thread,
    _cleanup_launched_clusters,
)
from sky_spot.e2e.transfer import _transfer_s3_bucket
from sky_spot.e2e.viz import generate_timeline_plot
from sky_spot.e2e import wandb_viz

import sky
from sky.client import sdk

logger = logging.getLogger(__name__)


def run_simulation(strategy_name: str = config.DEFAULT_STRATEGY, region_id: int = 0):
    """Run an end-to-end simulation with the specified strategy.
    
    Args:
        strategy_name: Name of the strategy to use (e.g., "risk", "single", "baseline")
        region_id: Region index for single-region strategies
    """
    # Clear events from previous run
    config._preemption_events.clear()
    config._terminate_events.clear()
    config._preemption_displayed_regions.clear()

    # Create output directory
    if strategy_name == "single":
        zone_name = config.trace_files[region_id]
        config.current_output_dir = f"output/{config.task_name}-{strategy_name}-{zone_name}"
    else:
        config.current_output_dir = f"output/{config.task_name}-{strategy_name}"
    os.makedirs(config.current_output_dir)
    print(f"Output directory: {config.current_output_dir}")

    # Setup per-tick logging
    strategy_log_dir = f"{config.current_output_dir}/strategy"
    os.makedirs(strategy_log_dir, exist_ok=True)
    strategy_logger = logging.getLogger("sky_spot.strategies")
    strategy_logger.handlers.clear()
    strategy_logger.setLevel(logging.DEBUG)
    strategy_logger.propagate = False
    _strategy_file_handler = None

    def _setup_tick_logging(tick: int):
        nonlocal _strategy_file_handler
        try:
            if _strategy_file_handler:
                strategy_logger.removeHandler(_strategy_file_handler)
                _strategy_file_handler.close()
            os.makedirs(strategy_log_dir, exist_ok=True)
            _strategy_file_handler = logging.FileHandler(
                f"{strategy_log_dir}/tick-{tick:04d}.log", mode="w"
            )
            _strategy_file_handler.setFormatter(
                logging.Formatter("%(name)s %(levelname)s: %(message)s")
            )
            strategy_logger.addHandler(_strategy_file_handler)
            from sky_spot.strategies.ucm_decision_logger import set_log_file
            set_log_file(f"{strategy_log_dir}/tick-{tick:04d}.log")
        except Exception as e:
            logger.warning("[TICK_LOG] Failed to setup tick logging: %s", e)

    # Create buckets (one per base region)
    seen_region_prefixes = set()
    for region in range(len(config.trace_files)):
        region_prefix = config.trace_files[region][:-1]
        if region_prefix in seen_region_prefixes:
            continue
        seen_region_prefixes.add(region_prefix)
        _create_bucket(region)

    # Create environment
    env = MultiTraceEnv(
        trace_files=config.trace_files,
        env_start_hours=0.0,
        window_hours=None,
        gap_seconds=config.GAP_SECONDS,
        instance_type=config.instance_type,
    )
    config._current_env = env

    # Inject real cloud operations
    env._actual_launch_internal = _actual_launch_internal
    env._actual_terminate_internal = _actual_terminate_internal
    env._actual_probe_internal = _actual_probe_internal
    env._actual_check_is_preempted_internal = _actual_check_is_preempted_internal
    env._actual_check_ondemand_health_internal = _actual_check_ondemand_health_internal
    env._patch_restart_overheads = _patch_restart_overheads
    env._compute_total_progress = _compute_total_progress
    env._transfer_s3_bucket = _transfer_s3_bucket

    # Log fetching thread
    def _fetch_logs():
        train_log_dir = f"{config.current_output_dir}/training"
        os.makedirs(train_log_dir, exist_ok=True)
        st = time.time()
        while not probe_event.is_set():
            if env.active_instances:
                latest_log = ""
                try:
                    region, ct = list(env.active_instances.keys())[0]
                    cluster_name = _get_cluster_name(region, ct)
                    record = _get_cluster_record_with_refresh(cluster_name)
                    if record is None or record.status != sky.ClusterStatus.UP:
                        status_str = record.status if record else "NOT_FOUND"
                        latest_log = f"[fetch_logs] Skipping: cluster {cluster_name} status={status_str}"
                    else:
                        output_stream = io.StringIO()
                        sdk.tail_logs(cluster_name, job_id=None, follow=False, output_stream=output_stream)
                        latest_log = output_stream.getvalue()
                except Exception as e:
                    latest_log = f"error: {type(e)} {e}\ntraceback: {traceback.format_exc()}"
                elapsed = int(time.time() - st)
                with open(f"{train_log_dir}/elapsed-{elapsed}-tick-{env.tick}.txt", "w") as f:
                    f.write(latest_log)
            time.sleep(30)

    # Setup strategy
    scaled_duration = config.actual_job_duration_seconds * config.time_scale / 3600.0
    parser = configargparse.ArgumentParser(add_help=False)
    namespace = argparse.Namespace(
        deadline_hours=scaled_duration * config.DEADLINE_RATIO,
        restart_overhead_hours=[config.actual_cold_start_time_seconds * config.time_scale / 3600.0],
        risk_probe_interval_ticks=1,
        region_id=region_id,
    )

    # Start background threads
    probe_thread_obj = None
    if strategy_name == "risk":
        probe_thread_obj = threading.Thread(
            target=_probe_thread,
            args=(config.PROBE_INTERVAL_TICKS * config.GAP_SECONDS / config.time_scale,),
            daemon=True,
        )
        probe_thread_obj.start()
    
    fetch_thread = threading.Thread(target=_fetch_logs, daemon=True)
    fetch_thread.start()

    # Initialize strategy
    full_strategy_name = config._resolve_strategy(strategy_name)
    StrategyClass = strategy_lib.Strategy.get(full_strategy_name)
    logger.debug("[run_simulation] Using strategy: %s (%s)", strategy_name, full_strategy_name)
    strategy = StrategyClass._from_args(parser, namespace)
    assert isinstance(strategy, strategy_lib.MultiRegionStrategy)

    task = SingleTask(config={
        "duration": scaled_duration,
        "checkpoint_size_gb": config.CHECKPOINT_SIZE_GB,
    })

    env.reset()
    strategy.reset(env, task)
    log_file = f"{config.current_output_dir}/history.jsonl"
    history: List[Dict[str, Any]] = []

    # Initialize console
    sim_console.__init__()
    task_id = config.current_output_dir.replace("output/", "")
    sim_console.start(
        strategy_name=full_strategy_name,
        regions=config.trace_files,
        task_duration_h=scaled_duration,
        deadline_h=scaled_duration * config.DEADLINE_RATIO,
        task_id=task_id,
    )

    # Initialize wandb
    wandb_viz.init(
        project="skynomad",
        name=task_id,
        region_names=config.trace_files,
        config={
            "strategy": full_strategy_name,
            "task_duration_hours": scaled_duration,
            "deadline_hours": scaled_duration * config.DEADLINE_RATIO,
            "regions": config.trace_files,
            "instance_type": config.instance_type,
        },
    )

    total_preemption_count = 0
    progress_warning_shown = False

    # Main simulation loop
    while not strategy.task_done:
        st = time.time()
        _setup_tick_logging(env.tick)
        logger.debug("===tick=== %s timestamp %s", env.tick, time.time())
        
        env.observe()
        for sub_env in env.envs:
            sub_env._update_realtime_price(env.tick)
        env.update_strategy_progress(strategy)

        # Check if progress reporting is working correctly (warn once)
        # Compare actual progress with expected based on elapsed time
        # Account for cold start time (restart_overhead) before expecting progress
        actual_progress_sec = sum(strategy.task_done_time)
        cold_start_sec = config.actual_cold_start_time_seconds
        time_after_coldstart = env.elapsed_seconds - cold_start_sec
        if time_after_coldstart > config.GAP_SECONDS * 10 and not progress_warning_shown:
            # After cold start + 10 ticks, expect at least some progress
            # Allow 50% slack for restarts, preemptions, additional overhead
            expected_min_progress = time_after_coldstart * 0.5
            if actual_progress_sec < expected_min_progress * 0.1:  # < 5% of time after cold start
                logger.warning(
                    "[PROGRESS WARNING] Actual %.1fs << Expected ~%.1fs (after %.0fs cold start). "
                    "Is PROGRESS reporting working? Use: echo 'PROGRESS: step/total'",
                    actual_progress_sec, expected_min_progress, cold_start_sec
                )
                progress_warning_shown = True

        if strategy.task_done:
            break
        if env.elapsed_seconds >= strategy.deadline:
            sim_console.error(f"Deadline exceeded: {env.elapsed_seconds:.0f}s >= {strategy.deadline:.0f}s")
            raise ValueError(f"Deadline exceeded")

        safety_latched = env.maybe_trigger_safety_net(strategy, abort_on_trigger=True)
        if safety_latched:
            active = env.get_active_instances()
            region_name = config.trace_files[list(active.keys())[0]] if active else "unknown"
            sim_console.safety_net(region_name)
            break

        env.execute_multi_strategy(strategy)
        env.tick += 1

        # Record history
        info: Dict[str, Any] = {
            **env.info(),
            **strategy.info(),
            "Strategy": full_strategy_name,
            "WallTime": datetime.datetime.now().isoformat(),
            "SafetyNetTriggered": safety_latched,
            "LeaderRegion": getattr(env, "_current_leader_region", None),
        }
        if env.tick == 1:
            info["Config"] = {
                "RegionNames": config.trace_files,
                "TaskDurationHours": scaled_duration,
                "DeadlineHours": scaled_duration * config.DEADLINE_RATIO,
                "GapSeconds": config.GAP_SECONDS,
                "InstanceType": config.instance_type,
                "RestartOverheadHours": config.actual_cold_start_time_seconds * config.time_scale / 3600.0,
            }

        cost_breakdown = env.get_cost_breakdown()
        cost_by_type_str = {k.name: v for k, v in cost_breakdown["by_type"].items()}
        
        # Get price snapshot
        price_by_region: Dict[str, Dict[str, float]] = {}
        for region_idx, sub_env in enumerate(env.envs):
            try:
                price_map = sub_env.get_price(env.tick - 1)
            except Exception:
                price_map = sub_env._update_realtime_price(env.tick - 1)
            price_by_region[str(region_idx)] = {
                "ON_DEMAND": float(price_map.get(ClusterType.ON_DEMAND, 0.0)),
                "SPOT": float(price_map.get(ClusterType.SPOT, 0.0)),
            }

        # Process events
        preemption_events = config._preemption_events.copy()
        config._preemption_events.clear()
        terminate_events = config._terminate_events.copy()
        config._terminate_events.clear()

        for pe in preemption_events:
            region = pe["region"]
            if region in config._preemption_displayed_regions:
                config._preemption_displayed_regions.discard(region)
                total_preemption_count += 1
                continue
            event_type = pe.get("type", "spot")
            sim_console.preemption(pe["region"], pe["region_name"], event_type)
            total_preemption_count += 1

        info.update({
            "ActiveRegions": len(env.get_active_instances()),
            "CostByRegion": cost_breakdown["by_region"],
            "CostByType": cost_by_type_str,
            "PriceByRegion": price_by_region,
            "ActiveInstances": {k: v.name for k, v in env.get_active_instances().items()},
            "MigrationCount": env.migration_count,
            "LatestProbeResults": config.latest_probe_result,
            "PreemptionEvents": preemption_events,
            "TerminateEvents": terminate_events,
        })

        # Print progress
        task_done_sec = float(info.get("Task/Done(seconds)", 0.0) or 0.0)
        task_target_sec = float(info.get("Task/Target(seconds)", 1.0) or 1.0)
        progress_pct = (task_done_sec / task_target_sec) * 100.0 if task_target_sec > 0 else 0.0
        current_cost = float(info.get("Cost", 0.0) or 0.0)
        sim_console.tick(
            env.tick, progress_pct, current_cost,
            env.get_active_instances_with_age(),
            config.trace_files, config.GAP_SECONDS,
        )

        # Log to wandb with Plotly timeline
        wandb_viz.log_tick(
            tick=env.tick,
            progress_pct=progress_pct,
            cost=current_cost,
            active_instances={int(k): v for k, v in info.get("ActiveInstances", {}).items()},
            wall_time=info.get("WallTime", ""),
            elapsed_hours=env.elapsed_seconds / 3600.0,
            deadline_hours=scaled_duration * config.DEADLINE_RATIO,
            extra_metrics={
                "migrations": info.get("MigrationCount", 0),
                "compute_cost": info.get("ComputeCost", 0),
                "transfer_cost": info.get("TransferCost", 0),
            },
        )

        history.append(info)
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(info) + "\n")
        except Exception as e:
            logger.warning("[HISTORY] Failed to write: %s", e)

        time_to_sleep = max(0, (env.gap_seconds / config.time_scale - (time.time() - st)))
        time.sleep(time_to_sleep)

        if safety_latched:
            break

    # Cleanup
    probe_event.set()
    env.cleanup_after_simulation()
    if probe_thread_obj:
        probe_thread_obj.join()
    fetch_thread.join()

    # Final summary
    final_cost = history[-1].get("Cost", 0) if history else 0
    compute_cost = history[-1].get("ComputeCost", 0) if history else 0
    transfer_cost = history[-1].get("TransferCost", 0) if history else 0
    migrations = history[-1].get("MigrationCount", 0) if history else 0
    final_progress_sec = float(history[-1].get("Task/Done(seconds)", 0.0) or 0.0) if history else 0.0
    final_target_sec = float(history[-1].get("Task/Target(seconds)", 1.0) or 1.0) if history else 1.0
    final_progress_pct = (final_progress_sec / final_target_sec) * 100.0 if final_target_sec > 0 else 0.0

    sim_console.done(
        total_cost=final_cost,
        migrations=migrations,
        preemptions=total_preemption_count,
        final_progress=final_progress_pct,
        compute_cost=compute_cost,
        transfer_cost=transfer_cost,
    )

    # Generate static visualization (optional, wandb has real-time viz)
    try:
        generate_timeline_plot(
            history, config.trace_files, scaled_duration,
            strategy_name=strategy.NAME if hasattr(strategy, "NAME") else "unified_cost_model_risk",
            output_path=f"{config.current_output_dir}/timeline.png",
        )
    except Exception as e:
        logger.warning("[viz] Static timeline generation failed: %s", e)

    # Finish wandb run
    wandb_viz.finish()

    return history
