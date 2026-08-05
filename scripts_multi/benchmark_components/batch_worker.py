"""Batch worker script that processes multiple tasks on a single SkyPilot cluster."""

import argparse
import json
import gzip
import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .simulation_runner import run_single_simulation, run_single_simulation_with_history
from .trace_utils import get_trace_paths_for_task
from .cache_manager import (
    generate_cache_filename,
    load_result_entry,
    save_result_entry,
)

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil not always present on workers
    psutil = None  # type: ignore

# Respect LOG_LEVEL environment variable
_log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
_log_level = getattr(logging, _log_level_str, logging.INFO)

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=_log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
else:
    # If handlers exist, still set the root logger level
    logging.getLogger().setLevel(_log_level)
logger = logging.getLogger(__name__)


def _parse_cacheable_strategies() -> Set[str]:
    raw = os.environ.get("CACHEABLE_STRATEGIES", "multi_region_oracle_dp")
    return {s.strip() for s in raw.split(',') if s.strip()}


CACHEABLE_STRATEGIES = _parse_cacheable_strategies()

def _short_strategy_name(name: str) -> str:
    mapping = {
        'unified_cost_model': 'UCM',
        'unified_cost_model_v2': 'UCMv2',
        'unified_cost_model_oracle': 'UCM-Oracle',
        'unified_cost_model_risk': 'UCM-Risk',
        'multi_region_oracle_dp': 'OracleDP',
        'multi_region_rc_cr_threshold': 'RC-CR-TH',
        'multi_region_single_rc_cr_threshold': 'RC-CR-TH-SingleRC',
        'multi_region_rc_cr_threshold_egress': 'RC-CR-TH-Egress',
        'rc_cr_threshold': 'RC-CR-TH-Single',  # Single-region baseline
    }
    return mapping.get(name, name)

def _slug_num_trim(v: float) -> str:
    s = f"{float(v):.2f}"
    if s.endswith('00'):
        s = s[:-3]
    return s.replace('.', 'p').replace('-', 'm')

def _scenario_slug(name: str) -> str:
    import re
    nm = name
    m = re.search(r"\((\d+)\s*Regions?\)", nm)
    suffix = f"_{m.group(1)}" if m else ""
    nm = re.sub(r"\((\d+)\s*Regions?\)", "", nm)
    nm = nm.strip().lower().replace(' ', '_').replace('-', '_')
    nm = ''.join(c for c in nm if c.isalnum() or c == '_')
    return nm + suffix

def _history_path_components(task: Dict, env_start_h: float):
    scen_slug = _scenario_slug(task.get('scenario_name', 'scenario'))
    dr = _slug_num_trim(task.get('deadline_ratio', 0.0))
    ck = _slug_num_trim(task.get('checkpoint_size', 0.0))
    ro = _slug_num_trim(task.get('restart_overhead', 0.0))
    es = _slug_num_trim(env_start_h)
    strat_short = _short_strategy_name(task.get('strategy', 'strategy'))
    trace_id = int(task.get('trace_index', 0))
    return scen_slug, dr, ck, ro, es, strat_short, trace_id

def _history_file_exists(task: Dict, params: Dict, env_start_h: float) -> bool:
    scen_slug, dr, ck, ro, es, strat_short, trace_id = _history_path_components(task, env_start_h)
    hist_root = Path(params["OUTPUT_DIR"]) / 'histories'
    logger.debug(
        "History check: scenario=%s strategy=%s trace=%s env_start=%.2f root=%s",
        task.get('scenario_name'),
        task.get('strategy'),
        trace_id,
        env_start_h,
        hist_root,
    )
    param_dirs = [
        f"dr{dr}_ck{ck}_ro{ro}_es{es}",
        f"dr{dr}_ck{ck}_ro{ro}",
    ]
    filenames = [
        f"trace_{trace_id}.json.gz",
        f"trace_{trace_id}.json",
    ]
    for param_dir in param_dirs:
        for fname in filenames:
            candidate = hist_root / scen_slug / param_dir / strat_short / fname
            logger.debug("History candidate: %s", candidate)
            if candidate.exists():
                logger.debug("History found: %s", candidate)
                return True
    logger.debug("History missing for scenario=%s strategy=%s trace=%s env_start=%.2f",
                 task.get('scenario_name'), task.get('strategy'), trace_id, env_start_h)
    return False

def _run_simulation_with_history(
    *,
    task: Dict,
    params: Dict,
    env_type: str,
    trace_paths,
    task_duration: float,
    deadline_hours: float,
    task_env_start_h: float,
) -> Dict:
    stats = run_single_simulation_with_history(
        strategy=task["strategy"],
        env_type=env_type,
        trace_paths=trace_paths,
        task_duration_hours=task_duration,
        deadline_hours=deadline_hours,
        restart_overhead=task["restart_overhead"],
        checkpoint_size=task["checkpoint_size"],
        output_dir=Path(params["OUTPUT_DIR"]) / "sim_temp",
        env_start_hours=task_env_start_h,
        enforce_window_bound=True,
        count_cross_region_migration_time=params.get('COUNT_CROSS_REGION_MIGRATION_TIME', False),
        gang_threshold=params.get('GANG_THRESHOLD'),
    )
    try:
        cost = float(stats.get('costs', [float('nan')])[0])
    except Exception:
        cost = float('nan')
    migrations = int(stats.get('migrations', [0])[0]) if isinstance(stats.get('migrations'), list) else 0

    scen_slug, dr, ck, ro, es, strat_short, trace_id = _history_path_components(task, task_env_start_h)
    param_dir = f"dr{dr}_ck{ck}_ro{ro}_es{es}"
    out_file = Path(params["OUTPUT_DIR"]) / 'histories' / scen_slug / param_dir / strat_short / f'trace_{trace_id}.json.gz'
    logger.info(
        "Saving history: scenario=%s strategy=%s trace=%s env_start=%.2f path=%s",
        task.get('scenario_name'),
        task.get('strategy'),
        trace_id,
        task_env_start_h,
        out_file,
    )
    _save_history_minimal(
        stats,
        out_file,
        context={
            'deadline_hours': deadline_hours,
            'task_duration_hours': task_duration,
            'checkpoint_size_gb': task.get('checkpoint_size', 0.0),
            'restart_overhead_hours': task.get('restart_overhead', 0.0),
            'trace_files': [str(p) for p in trace_paths],
            'env_start_hours': task_env_start_h,
            'use_full_trace': task.get('use_full_trace', False),
            'trace_index': trace_id,
        },
    )
    return {"cost": cost, "migrations": migrations}

def _should_dump_history(task: Dict, params: Dict) -> bool:
    # Generate history for ALL strategies when segment viz is enabled
    # This avoids needing to re-run benchmarks when adding new strategies to visualize
    return bool(params.get('SEGMENT_VIZ_ENABLED'))

def _save_history_minimal(stats: Dict, out_file: Path, *, context: Dict[str, object]) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    keep = {
        'history': stats.get('history', []),
        'env': stats.get('env', {}),
        'strategy': stats.get('strategy', {}),
        'costs': stats.get('costs', []),
        'args': {
            'deadline_hours': context.get('deadline_hours'),
            'task_duration_hours': context.get('task_duration_hours'),
            'checkpoint_size_gb': context.get('checkpoint_size_gb'),
            'restart_overhead_hours': [context.get('restart_overhead_hours')],
            'trace_files': context.get('trace_files', []),
            'env_start_hours': context.get('env_start_hours'),
            'use_full_trace': context.get('use_full_trace'),
            'trace_index': context.get('trace_index'),
        }
    }
    with gzip.open(out_file, 'wt', encoding='utf-8') as f:
        json.dump(keep, f)

def execute_task(task: Dict, params: Dict) -> Dict:
    """Execute a single simulation task."""
    try:
        # Generate environment type and trace paths
        data_path_spec = (
            task.get("data_paths")
            or params.get("DATA_PATH_BY_SCENARIO", {}).get(task.get("scenario_name"))
            or params["DATA_PATH"]
        )
        env_type, trace_paths = get_trace_paths_for_task(task, data_path_spec)
        
        # Task duration is pre-calculated during task generation
        task_duration = float(task["task_duration"])
        deadline_hours = float(task["deadline_hours"])
        
        # Effective env_start_hours: prefer task-specific (full-trace slicing), fallback to global
        task_env_start_h = float(task.get("env_start_hours", params.get("ENV_START_HOURS", 0.0)))

        cache_file: Optional[Path] = None
        requires_history = _should_dump_history(task, params)

        if task.get("strategy") in CACHEABLE_STRATEGIES:
            cache_dir = Path(params["OUTPUT_DIR"]) / "cache"
            _argv = params.get("STRATEGY_ARGV")
            # Strategy flags change the result, so they have to change the key.
            _cache_strategy = task.get("strategy")
            if _argv:
                _cache_strategy = f"{_cache_strategy}[{'_'.join(_argv)}]"
            cache_filename = generate_cache_filename(
                _cache_strategy,
                env_type,
                trace_paths,
                task.get("checkpoint_size", 0.0),
                task.get("restart_overhead", 0.0),
                deadline_hours,
                task_duration,
                task_env_start_h,
                count_cross_region_migration_time=params.get('COUNT_CROSS_REGION_MIGRATION_TIME', False),
            )
            cache_file = cache_dir / cache_filename
            cached = load_result_entry(cache_file)
            if cached is not None:
                logger.info(
                    "Cache HIT %s trace=%s ddl=%.2f ckpt=%.1f ro=%.2f",
                    task.get("strategy"),
                    task.get("trace_index"),
                    deadline_hours,
                    task.get("checkpoint_size", 0.0),
                    task.get("restart_overhead", 0.0),
                )
                if requires_history and not _history_file_exists(task, params, task_env_start_h):
                    logger.error(
                        "Cache had no history: scenario=%s strategy=%s trace=%s env_start=%.2f cache=%s",
                        task.get('scenario_name'),
                        task.get('strategy'),
                        task.get('trace_index'),
                        task_env_start_h,
                        cache_file,
                    )
                    raise RuntimeError(
                        "Segment visualization requires history data, but cache entry exists without a "
                        "corresponding history file. Delete the cache (or rerun without cache) so histories "
                        "can be regenerated."
                    )
                return {**task, **cached}

        if requires_history:
            result_record = _run_simulation_with_history(
                task=task,
                params=params,
                env_type=env_type,
                trace_paths=trace_paths,
                task_duration=task_duration,
                deadline_hours=deadline_hours,
                task_env_start_h=task_env_start_h,
            )
        else:
            # Run simulation (no history dump)
            cost, migrations, downtime, transfer, probe = run_single_simulation(
                task["strategy"],
                env_type,
                trace_paths,
                task_duration,
                deadline_hours,
                task["restart_overhead"],
                task["checkpoint_size"],
                Path(params["OUTPUT_DIR"]) / "sim_temp",
                task_env_start_h,
                task.get("strategy_file"),
                enforce_window_bound=True,
                count_cross_region_migration_time=params.get('COUNT_CROSS_REGION_MIGRATION_TIME', False),
                gang_threshold=params.get('GANG_THRESHOLD'),
                strategy_argv=params.get('STRATEGY_ARGV'),
                dump_history=params.get('DUMP_HISTORY', False),
            )
            result_record = {"cost": cost, "migrations": migrations, "downtime_cost": downtime,
                             "transfer_cost": transfer, "probe_cost": probe}

        final_result = {**task, **result_record}

        if cache_file is not None and not math.isnan(final_result.get("cost", float('nan'))):
            save_result_entry(cache_file, final_result)

        return final_result

    except ValueError as e:
        if "INFEASIBLE:" in str(e):
            error_msg = str(e).replace("INFEASIBLE: ", "")
            return {**task, "cost": float('nan'), "error_type": "Task Infeasible", "error_details": error_msg}
        elif "TRACE_INSUFFICIENT:" in str(e):
            error_msg = str(e).replace("TRACE_INSUFFICIENT: ", "")
            return {**task, "cost": float('nan'), "error_type": "Trace Insufficient", "error_details": error_msg}
        else:
            return {**task, "cost": float('nan'), "error_type": "ValueError", "error_details": str(e)}
    except Exception as e:
        import traceback
        # Log the error with context and full traceback
        error_trace = traceback.format_exc()
        logger.error(f"[{task.get('strategy')}] {task.get('scenario_name')} trace_{task.get('trace_index')} failed: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{error_trace}")
        return {**task, "cost": float('nan'), "error_type": type(e).__name__, "error_details": str(e), "error_traceback": error_trace}


def process_single_task(args):
    """Process a single task (for multiprocessing)."""
    task, params = args
    return execute_task(task, params)


def main():
    parser = argparse.ArgumentParser(description="Batch worker for SkyPilot execution")
    parser.add_argument("--tasks-file", type=str, required=True,
                        help="JSON file containing list of tasks")
    parser.add_argument("--params-json", type=str, required=True,
                        help="JSON string or file containing parameters")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    args = parser.parse_args()
    
    # Parse inputs
    with open(args.tasks_file, 'r') as f:
        tasks = json.load(f)
    
    # Handle params - can be JSON string or file path
    if os.path.exists(args.params_json):
        with open(args.params_json, 'r') as f:
            params = json.load(f)
    else:
        params = json.loads(args.params_json)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # CRITICAL: ensure all worker-side artifacts (including histories) are written
    # under the mounted output_dir, so they sync to GCS and can be downloaded.
    params["OUTPUT_DIR"] = str(output_dir)
    
    logger.info(f"Processing {len(tasks)} tasks in batch mode")

    def _meminfo_available_gb() -> float:
        """Best-effort read of available memory in GiB."""
        # Prefer psutil if present.
        try:
            return psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            pass
        # Fallback to /proc/meminfo MemAvailable if present, else 0.0
        try:
            with open("/proc/meminfo", "r") as f:
                mem_total_kb = None
                mem_avail_kb = None
                for line in f:
                    if line.startswith("MemAvailable:"):
                        mem_avail_kb = float(line.split()[1])
                    elif line.startswith("MemTotal:"):
                        mem_total_kb = float(line.split()[1])
                if mem_avail_kb is not None:
                    return mem_avail_kb / 1_048_576.0
                if mem_total_kb is not None:
                    # As a last resort, assume 50% of total is available
                    return 0.5 * mem_total_kb / 1_048_576.0
        except Exception:
            pass
        return 0.0

    # Sizing parameters (env-overridable) with robust parsing
    def _safe_float(var: str, default: float) -> float:
        val = os.environ.get(var)
        try:
            if val is None or val.strip() == '':
                return default
            return float(val)
        except Exception:
            return default

    # Reserve memory for OS, leave headroom to avoid swapping
    os_reserved_gb = _safe_float('OS_RESERVED_GB', 4.0)
    # Per-worker memory estimate; can be tuned down if workloads are light
    per_worker_gb = _safe_float('WORKER_PER_GB', 1.5)
    mem_util = _safe_float('MEM_UTIL', 0.80)
    cpu_cap_default = 64  # allow high parallelism on big VMs
    cpu_cap = min(multiprocessing.cpu_count(), cpu_cap_default)

    avail_gb = _meminfo_available_gb()
    safe_avail_gb = max(0.0, avail_gb - os_reserved_gb)
    by_mem = int(max(1, (mem_util * safe_avail_gb) // max(per_worker_gb, 0.1))) if safe_avail_gb > 0 else 1

    # Allow hard override via env (WORKERS)
    env_workers = os.environ.get('WORKERS')
    if env_workers and env_workers.strip().isdigit() and int(env_workers) > 0:
        num_workers = min(int(env_workers.strip()), len(tasks))
    else:
        num_workers = min(cpu_cap, by_mem, len(tasks))
        num_workers = max(1, num_workers)

    # Use spawn to avoid copy-on-write bloat when parent holds memory.
    mp_ctx = multiprocessing.get_context("spawn")

    logger.info(
        f"Using {num_workers} parallel workers (cpu_cap={cpu_cap}, avail_gb={avail_gb:.1f}, "
        f"safe_avail_gb={safe_avail_gb:.1f}, per_worker_gb={per_worker_gb}, mem_util={mem_util}, "
        f"os_reserved_gb={os_reserved_gb})"
    )
    
    # Process tasks in parallel - using JSONL for streaming output
    results_file = output_dir / "results.jsonl"
    summary_file = output_dir / "batch_summary.json"
    
    # Create executor with spawn when supported; fall back gracefully if not.
    try:
        _executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx)
    except TypeError:
        logger.warning("mp_context not supported on this Python; falling back to default start method.")
        _executor = ProcessPoolExecutor(max_workers=num_workers)

    completed = 0
    failed = 0
    
    # Open JSONL file for streaming writes
    with open(results_file, 'w') as jsonl_file, _executor as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_task, (task, params)): task
            for task in tasks
        }
        
        # Process results as they complete - streaming to JSONL
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            try:
                result = future.result()
                # Stream write to JSONL - one line per result
                json.dump(result, jsonl_file)
                jsonl_file.write('\n')
                jsonl_file.flush()  # Ensure it's written immediately
                
                # Also save individual file for backwards compatibility
                output_file = output_dir / f"result_{task.get('_task_index', completed)}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                if math.isnan(result.get('cost', float('nan'))):
                    failed += 1
                    
                logger.info(f"✅ [{completed}/{len(tasks)}] Completed: {task['scenario_name']} - {task['strategy']} - trace {task['trace_index']}")
            except Exception as e:
                logger.error(f"❌ [{completed}/{len(tasks)}] Failed: {task['scenario_name']} - {task['strategy']} - {e}")
                result = {
                    **task,
                    "cost": float('nan'),
                    "error_type": "ProcessingError",
                    "error_details": str(e)
                }
                # Stream write error result too
                json.dump(result, jsonl_file)
                jsonl_file.write('\n')
                jsonl_file.flush()
                failed += 1
    
    # Write lightweight summary (no results array)
    with open(summary_file, 'w') as f:
        json.dump({
            "total_tasks": len(tasks),
            "completed_tasks": completed,
            "successful_tasks": completed - failed,
            "failed_tasks": failed,
            "results_file": "results.jsonl"
        }, f, indent=2)
    
    logger.info(f"Batch processing completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
