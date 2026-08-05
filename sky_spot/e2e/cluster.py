"""Cluster operations module for E2E testing.

This module handles all cluster lifecycle operations including launch,
terminate, status checking, and progress tracking.
"""

import copy
import io
import json
import logging
import os
import re
import time
import traceback
import threading
from typing import Optional

import sky
from sky.client import sdk
from sky.utils import common

from sky_spot.utils import ClusterType
from sky_spot.e2e import config
from sky_spot.e2e.console import SimulationConsole

logger = logging.getLogger(__name__)

# Global simulation console instance
sim_console = SimulationConsole()


def _write_alert(alert_type: str, region: str, message: str) -> None:
    """Write an alert to the alerts.jsonl file for monitor.py to pick up."""
    if not config.current_output_dir:
        return
    try:
        os.makedirs(config.current_output_dir, exist_ok=True)
    except Exception:
        return
    alerts_file = f"{config.current_output_dir}/alerts.jsonl"
    alert = {
        "timestamp": time.time(),
        "type": alert_type,
        "region": region,
        "message": message,
    }
    try:
        with open(alerts_file, "a") as f:
            f.write(json.dumps(alert) + "\n")
    except Exception as e:
        logger.warning("[ALERT] Failed to write alert: %s", e)


def _register_cluster(cluster_name: str) -> None:
    with config._launched_clusters_lock:
        config._launched_clusters.add(cluster_name)


def _get_launched_clusters():
    with config._launched_clusters_lock:
        return list(config._launched_clusters)


def _get_cluster_name(
    region: int, cluster_type: ClusterType, probe: bool = False
) -> str:
    cn = f"{config.task_name}-{region}-{config.trace_files[region]}-{cluster_type.name}"
    if probe:
        cn += "-probe"
    return cn


def _get_bucket_name(region: int) -> str:
    """Return a valid S3 bucket name for the given region."""
    import hashlib

    region_prefix = config.trace_files[region][:-1]
    name = f"{config.task_name}-{region_prefix}-bucket"
    if len(name) > 63:
        hash_suffix = hashlib.sha1(config.task_name.encode("utf-8")).hexdigest()[:8]
        name = f"cbl-{hash_suffix}-{region_prefix}-bucket"
    return name


def _create_bucket(region: int):
    """Create an S3 bucket in the target region."""
    import boto3
    import botocore

    bucket_name = _get_bucket_name(region)
    location_constraint = config.trace_files[region][:-1]

    regional_s3 = boto3.client("s3", region_name=location_constraint)

    try:
        if location_constraint == "us-east-1":
            response = regional_s3.create_bucket(Bucket=bucket_name)
        else:
            response = regional_s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": location_constraint},
            )
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code == "BucketAlreadyOwnedByYou":
            logger.info("Bucket already owned by us: %s", bucket_name)
            return
        if code == "BucketAlreadyExists":
            raise RuntimeError(
                f"Bucket name '{bucket_name}' is already taken by another AWS account. "
                f"Choose a different task_name to generate a unique bucket name."
            ) from e
        logger.error(
            "FATAL: unexpected error creating bucket %s: %s %s (%s)",
            bucket_name,
            type(e),
            e,
            code,
        )
        raise
    else:
        logger.info("Created bucket: %s in %s", bucket_name, location_constraint)

    try:
        regional_s3.head_bucket(Bucket=bucket_name)
        logger.info("Verified bucket accessible: %s", bucket_name)
    except Exception as e:
        raise RuntimeError(
            f"Bucket '{bucket_name}' was created but is not accessible: {e}"
        ) from e


def _get_task(
    region: int, cluster_type: ClusterType, is_probe: bool = False
) -> sky.Task:
    # Priority 1: Use YAML task config if provided (from run.py run_from_yaml)
    if config._yaml_task_config is not None:
        task_config = copy.deepcopy(config._yaml_task_config)
        if is_probe:
            # For probe, only use resources
            task_config = {"resources": task_config.get("resources", {})}
        task = sky.Task.from_yaml_config(task_config)
        zone = config.trace_files[region]
        region_name = zone[:-1]
        return task.set_resources_override(
            {
                "region": region_name,
                "zone": zone,
                "use_spot": cluster_type == ClusterType.SPOT,
            }
        )

    # Priority 2: Built-in training workload
    if config.USE_TRAINING_WORKLOAD:
        task_config = copy.deepcopy(config._cached_task_yaml_config)
        if is_probe:
            task_config = {"resources": task_config["resources"].copy()}
        task = sky.Task.from_yaml_config(task_config)
        zone = config.trace_files[region]
        region_name = zone[:-1]

        if not is_probe:
            if config.USE_GPU in ("L4", "A10G", "V100"):
                extra_envs = {
                    "EPOCHS": str(config.get_cfg("EPOCHS")),
                    "MAX_TRAIN_SAMPLES": str(config.get_cfg("MAX_TRAIN_SAMPLES")),
                    "GRAD_ACCUM_STEPS": str(config.get_cfg("GRAD_ACCUM_STEPS")),
                    "BATCH_SIZE": str(config.get_cfg("BATCH_SIZE")),
                    "SAVE_STEPS": str(config.get_cfg("SAVE_STEPS")),
                    "SAVE_TOTAL_LIMIT": str(config.get_cfg("SAVE_TOTAL_LIMIT")),
                    "MAX_SEQ_LEN": str(config.get_cfg("MAX_SEQ_LEN")),
                    "MODELSIZE": str(config.get_cfg("MODELSIZE")),
                }
            else:
                extra_envs = config.train_cfg
        else:
            extra_envs = {}

        extra_envs["BUCKET_NAME"] = _get_bucket_name(region)

        return task.update_envs(extra_envs).set_resources_override(
            {
                "region": region_name,
                "zone": zone,
                "use_spot": cluster_type == ClusterType.SPOT,
            }
        )

    if config.USE_FAKE_WORKLOAD:
        zone = config.trace_files[region]
        region_name = zone[:-1]
        bucket_name = _get_bucket_name(region)

        task = sky.Task()
        task.set_resources(
            sky.Resources(
                instance_type=config.instance_type,
                cloud=sky.AWS(),
                zone=zone,
                use_spot=cluster_type == ClusterType.SPOT,
            )
        )

        if is_probe:
            task.run = "sleep 60"
        else:
            task.workdir = "eval/fake_workload"
            task.set_storage_mounts(
                {
                    "/opt/ml/checkpoints": sky.Storage(
                        name=bucket_name,
                        source=f"s3://{bucket_name}",
                        mode=sky.StorageMode.MOUNT,
                    ),
                }
            )

            task.run = " ".join(
                [
                    f"TOTAL_STEPS={config.total_steps}",
                    f"STEP_SECONDS={config.step_seconds}",
                    f"SAVE_STEPS={config.save_steps}",
                    f"SAVE_TOTAL_LIMIT={config.save_total_limit}",
                    f"CHECKPOINT_SIZE_MB={config.checkpoint_size_mb}",
                    "OUTPUT_DIR=/opt/ml/checkpoints",
                    "python fake_train.py",
                ]
            )
        return task

    # Simple test mode
    task = sky.Task()
    task.set_resources(
        sky.Resources(
            instance_type=config.instance_type,
            cloud=sky.AWS(),
            zone=config.trace_files[region],
            use_spot=cluster_type == ClusterType.SPOT,
        )
    )
    return task


def _actual_launch_internal(region: int, cluster_type: ClusterType) -> bool:
    task = _get_task(region, cluster_type)
    # Only use simple test mode if NOT using YAML config, training workload, or fake workload
    if (not config.USE_TRAINING_WORKLOAD and
        not config.USE_FAKE_WORKLOAD and
        config._yaml_task_config is None):
        task.setup = "\n".join(
            [
                f"sleep {config.actual_cold_start_time_seconds}",
                f'echo "{config.setup_finished}"',
            ]
        )
        task.run = "\n".join(
            [
                f"for i in {{1..{int(config.actual_job_duration_seconds / 10)}}}; do",
                f"    echo progress $((i * 10))/{config.actual_job_duration_seconds}",
                "    sleep 10",
                "done",
            ]
        )
    cluster_name = _get_cluster_name(region, cluster_type)
    logger.debug(
        "[LAUNCH] Starting launch: region=%d (%s), type=%s, cluster=%s",
        region,
        config.trace_files[region],
        cluster_type.name,
        cluster_name,
    )
    try:
        request_id = sdk.launch(task, cluster_name=cluster_name)
    except Exception as e:
        err_msg = f"sdk.launch failed for {cluster_name}: {type(e).__name__}: {e}"
        logger.error("[LAUNCH] %s", err_msg)
        sim_console.error(err_msg)
        _write_alert("sdk_launch_error", config.trace_files[region], err_msg)
        return False
    _register_cluster(cluster_name)

    log_dir = f"{config.current_output_dir}/launch"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/{time.time()}-{config.trace_files[region]}-{cluster_type.name}.txt"
    with open(log_path, "w") as f:
        try:
            sdk.stream_until_output_contains(
                request_id, contains="CBL-INSTANCE-PENDING", output_stream=f
            )
            logger.debug("[LAUNCH] Success: %s is pending", cluster_name)
            # Record launch timestamp for cold start measurement
            config._launch_timestamps[cluster_name] = time.time()
            sim_console.launch(
                region,
                config.trace_files[region],
                cluster_type.name,
                cluster_type == ClusterType.SPOT,
            )
            return True
        except Exception as e:
            f.write("\n=== LAUNCH ERROR ===\n")
            f.write(f"{type(e)} {e}\n")
            f.write("TRACEBACK:\n")
            f.write(traceback.format_exc())
            f.write("\n")
            f.flush()
            try:
                sdk.stream_and_get(request_id, output_stream=f)
            except Exception:
                pass

    try:
        with open(log_path, "r") as f:
            log_content = f.read()
    except Exception:
        log_content = ""

    error_type = config._classify_launch_error(log_content)
    if error_type == "quota":
        sim_console.error(
            f"QUOTA EXCEEDED in {config.trace_files[region]}: Check {log_path}. "
            f"Request increase at: https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas"
        )
        _write_alert("quota_exceeded", config.trace_files[region], log_path)
    elif error_type == "capacity":
        logger.debug("[launch] No capacity for %s (expected, may retry)", cluster_name)
    else:
        logger.error(
            "[launch] Unexpected error launching %s. See %s", cluster_name, log_path
        )
    return False


def _actual_terminate_internal(region: int, cluster_type: ClusterType) -> bool:
    cluster_name = _get_cluster_name(region, cluster_type)
    logger.debug(
        "[TERMINATE] Starting terminate: region=%d (%s), type=%s, cluster=%s",
        region,
        config.trace_files[region],
        cluster_type.name,
        cluster_name,
    )
    try:
        request_id = sdk.down(cluster_name=cluster_name)
    except Exception as e:
        err_msg = f"sdk.down failed for {cluster_name}: {type(e).__name__}: {e}"
        logger.error("[TERMINATE] %s", err_msg)
        sim_console.error(err_msg)
        _write_alert("sdk_down_error", config.trace_files[region], err_msg)
        return False
    try:
        sdk.stream_and_get(request_id)
    except Exception as e:
        logger.error(
            "[TERMINATE] Error waiting for %s: %s %s", cluster_name, type(e), e
        )
        return False
    logger.debug("[TERMINATE] Success: %s terminated", cluster_name)
    sim_console.terminate(region, config.trace_files[region], cluster_type.name)
    config._terminate_events.append(
        {
            "region": region,
            "region_name": config.trace_files[region],
            "cluster_type": cluster_type.name,
            "timestamp": time.time(),
        }
    )
    return True


def _actual_probe_internal(region: int) -> bool:
    result = config.latest_probe_result.get(config.trace_files[region], True)
    logger.debug("===probe=== %s %s", region, result)
    return result


def _get_cluster_record_with_refresh(cluster_name: str) -> Optional[object]:
    """Get full cluster record with refresh."""
    request_id = sdk.status(
        cluster_names=[cluster_name], refresh=common.StatusRefreshMode.FORCE
    )
    try:
        res = sdk.stream_and_get(request_id, output_stream=io.StringIO())
    except Exception as e:
        logger.debug("Status error %s %s", type(e), e)
        return None
    logger.debug("Status result %s %s", type(res), res)
    if res is not None and len(res) > 0:
        return res[0]
    return None


def _get_cluster_status_with_refresh(cluster_name: str) -> Optional[sky.ClusterStatus]:
    record = _get_cluster_record_with_refresh(cluster_name)
    if record is not None:
        return record.status
    return None


def _actual_check_is_preempted_internal(region: int) -> bool:
    """Check if spot instance is available (not preempted)."""
    cluster_name = _get_cluster_name(region, ClusterType.SPOT)
    record = _get_cluster_record_with_refresh(cluster_name)
    if record is None:
        logger.debug(
            "[PREEMPTION] Cluster %s not found (record=None)",
            cluster_name,
        )
        config._preemption_events.append(
            {
                "region": region,
                "region_name": config.trace_files[region],
                "cluster_name": cluster_name,
                "timestamp": time.time(),
            }
        )
        return False

    if record.status == sky.ClusterStatus.UP:
        if _check_job_failed_via_logs(cluster_name):
            logger.warning(
                "[JOB FAILED] Cluster %s job FAILED (not preempted - job error/timeout). "
                "Terminating and will retry.",
                cluster_name,
            )
            sim_console.preemption(region, config.trace_files[region], event_type="job_failed")
            config._preemption_events.append(
                {
                    "region": region,
                    "region_name": config.trace_files[region],
                    "cluster_name": cluster_name,
                    "timestamp": time.time(),
                    "reason": "job_failed",
                }
            )
            try:
                sdk.down(cluster_name)
                logger.info("[JOB FAILED] Terminated cluster %s", cluster_name)
            except Exception as e:
                logger.warning(
                    "[JOB FAILED] Failed to terminate %s: %s", cluster_name, e
                )
            return False
        return True

    if not record.cluster_ever_up and record.status == sky.ClusterStatus.INIT:
        logger.debug(
            "Cluster %s still launching (ever_up=False, status=INIT)", cluster_name
        )
        return True

    logger.debug(
        "[PREEMPTION] Cluster %s was preempted (ever_up=%s, status=%s)",
        cluster_name,
        record.cluster_ever_up,
        record.status,
    )
    config._preemption_events.append(
        {
            "region": region,
            "region_name": config.trace_files[region],
            "cluster_name": cluster_name,
            "timestamp": time.time(),
        }
    )
    return False


def _actual_check_ondemand_health_internal(region: int) -> bool:
    """Check if ON_DEMAND instance is healthy."""
    cluster_name = _get_cluster_name(region, ClusterType.ON_DEMAND)
    record = _get_cluster_record_with_refresh(cluster_name)
    if record is None:
        return False

    if record.status == sky.ClusterStatus.UP:
        if _check_job_failed_via_logs(cluster_name):
            logger.warning(
                "[JOB FAILED] ON_DEMAND cluster %s job FAILED (job error/timeout)",
                cluster_name,
            )
            sim_console.preemption(region, config.trace_files[region], event_type="job_failed")
            config._preemption_events.append(
                {
                    "region": region,
                    "region_name": config.trace_files[region],
                    "cluster_name": cluster_name,
                    "timestamp": time.time(),
                    "reason": "job_failed",
                    "type": "ondemand_failure",
                }
            )
            try:
                sdk.down(cluster_name)
                logger.info("[JOB FAILED] Terminated ON_DEMAND cluster %s", cluster_name)
            except Exception as e:
                logger.warning(
                    "[JOB FAILED] Failed to terminate %s: %s", cluster_name, e
                )
            return False
        return True

    if not record.cluster_ever_up and record.status == sky.ClusterStatus.INIT:
        logger.debug(
            "Cluster %s still launching (ever_up=False, status=INIT)", cluster_name
        )
        return True

    logger.warning(
        "[ON_DEMAND FAILURE] Cluster %s failed (ever_up=%s, status=%s) - treating like preemption",
        cluster_name,
        record.cluster_ever_up,
        record.status,
    )
    config._preemption_events.append(
        {
            "region": region,
            "region_name": config.trace_files[region],
            "cluster_name": cluster_name,
            "timestamp": time.time(),
            "type": "ondemand_failure",
        }
    )
    return False


def _get_logs(cluster_name: str) -> Optional[str]:
    status = _get_cluster_status_with_refresh(cluster_name)
    if status == sky.ClusterStatus.UP:
        output_stream = io.StringIO()
        try:
            sdk.tail_logs(
                cluster_name, job_id=None, follow=False, output_stream=output_stream
            )
        except sky.exceptions.ClusterDoesNotExist:
            return None
        return output_stream.getvalue()
    return None


def _check_job_failed_via_logs(cluster_name: str) -> bool:
    """Check if job failed by calling tail_logs and checking exit code."""
    status = _get_cluster_status_with_refresh(cluster_name)
    if status != sky.ClusterStatus.UP:
        return False

    try:
        exit_code = sdk.tail_logs(
            cluster_name, job_id=None, follow=False, output_stream=io.StringIO()
        )
        if exit_code == sky.exceptions.JobExitCode.FAILED:
            logger.warning(
                "[JOB FAILED] Job on %s failed (exit_code=%s)",
                cluster_name, exit_code
            )
            return True
    except Exception as e:
        logger.debug("[JOB] Failed to check job status for %s: %s", cluster_name, e)
    return False


def _patch_restart_overheads(
    region: int, is_spot: bool, remaining_restart_overhead: float, gap_seconds: float
) -> float:
    cluster_name = _get_cluster_name(
        region, ClusterType.SPOT if is_spot else ClusterType.ON_DEMAND
    )
    log = _get_logs(cluster_name)
    restart_finished = False
    if log is not None:
        for line in log.split("\n"):
            if config.setup_finished in line:
                restart_finished = True
                break

    if remaining_restart_overhead > gap_seconds:
        return remaining_restart_overhead
    if not restart_finished and remaining_restart_overhead <= gap_seconds:
        return remaining_restart_overhead + gap_seconds
    if restart_finished:
        return 0.0
    return remaining_restart_overhead


def _compute_total_progress(
    strategy, region: int, is_spot: bool, per_overheads, total_before
) -> float:
    del strategy, per_overheads, total_before
    cluster_name = _get_cluster_name(
        region, ClusterType.SPOT if is_spot else ClusterType.ON_DEMAND
    )
    log = _get_logs(cluster_name)
    if log is None:
        logger.debug("[PROGRESS] No logs found for %s", cluster_name)
        return 0.0
    for line in reversed(log.split("\n")):
        # Priority 1: Official PROGRESS format (recommended for users)
        # Usage: echo "PROGRESS: 50/100"
        progress_match = re.search(r"PROGRESS:\s*(\d+)\s*/\s*(\d+)", line)
        if progress_match:
            current = int(progress_match.group(1))
            total = int(progress_match.group(2))
            if total > 0:
                # Measure cold start on first progress
                if cluster_name not in config._first_progress_seen:
                    config._first_progress_seen[cluster_name] = True
                    if cluster_name in config._launch_timestamps:
                        cold_start = time.time() - config._launch_timestamps[cluster_name]
                        config._cold_start_measured[region] = cold_start
                        logger.info("[COLD_START] Region %d (%s): %.1f seconds",
                                    region, config.trace_files[region], cold_start)

                ratio = current / total
                progress_sec = ratio * config.actual_job_duration_seconds * config.time_scale
                logger.debug("[PROGRESS] From PROGRESS format: %d/%d = %.1f%%", current, total, ratio * 100)
                return progress_sec

        # Priority 2: Training workload JSON events
        if config.USE_TRAINING_WORKLOAD or config.USE_FAKE_WORKLOAD:
            if '"event": "dcp_async_save_completed"' in line:
                match = re.search(r'"step": (\d+)', line)
                if match:
                    step = int(match.group(1))
                    logger.debug(
                        "[PROGRESS] From dcp_async_save_completed: step %s / %s",
                        step,
                        config.total_steps,
                    )
                    return step / config.total_steps * config.actual_job_duration_seconds * config.time_scale
            if '"event": "progress"' in line:
                match = re.search(r'"step": (\d+)', line)
                if match:
                    step = int(match.group(1))
                    logger.debug(
                        "[PROGRESS] From JSON progress event: step %s / %s",
                        step,
                        config.total_steps,
                    )
                    return step / config.total_steps * config.actual_job_duration_seconds * config.time_scale
        if "progress" in line and "/" in line:
            try:
                prog = line.split("progress")[1].split("/")[0].strip()
                prog_float = float(prog)
                logger.debug("[PROGRESS] From simple text format: %s", prog_float)
                return prog_float * config.time_scale
            except (ValueError, IndexError):
                continue
        # Support "Step X/Y" format (case insensitive)
        step_match = re.search(r"[Ss]tep\s+(\d+)\s*/\s*(\d+)", line)
        if step_match:
            current_step = int(step_match.group(1))
            total_steps = int(step_match.group(2))
            if total_steps > 0:
                progress_ratio = current_step / total_steps
                progress_seconds = progress_ratio * config.actual_job_duration_seconds * config.time_scale
                logger.debug(
                    "[PROGRESS] From Step format: %d/%d = %.1f%%",
                    current_step, total_steps, progress_ratio * 100
                )
                return progress_seconds
    logger.debug("[PROGRESS] No progress found in logs for %s", cluster_name)
    return 0.0


probe_event = threading.Event()


def _probe_thread(interval_seconds: float):
    """Background thread that probes regions for spot availability.

    Each region is probed once per interval_seconds, but probes are staggered
    so only one region is probed at a time.
    """
    probe_log_dir = f"{config.current_output_dir}/probe"
    os.makedirs(probe_log_dir, exist_ok=True)
    from rich.console import Console as RichConsole

    probe_console = RichConsole()
    num_regions = len(config.trace_files)

    # Track last probe time per region
    last_probe_time: dict[int, float] = {r: 0.0 for r in range(num_regions)}

    # Sleep interval between probes (stagger across regions)
    sleep_interval = max(5.0, interval_seconds / num_regions)

    while not probe_event.is_set():
        try:
            now = time.time()

            # Find the region that hasn't been probed for the longest time
            region_idx = min(range(num_regions), key=lambda r: last_probe_time[r])

            # Check if it's time to probe this region
            if now - last_probe_time[region_idx] < interval_seconds:
                time.sleep(sleep_interval)
                continue

            # Skip if this region already has an active SPOT instance
            if config._current_env is not None:
                has_active_spot = any(
                    r == region_idx and ct == ClusterType.SPOT
                    for (r, ct) in config._current_env.active_instances.keys()
                )
                if has_active_spot:
                    last_probe_time[region_idx] = now
                    config.latest_probe_result[config.trace_files[region_idx]] = True
                    logger.debug("Probe region %d: skipped (has active SPOT)", region_idx)
                    continue

            # Probe this single region
            region_name = config.trace_files[region_idx]
            cn = _get_cluster_name(region_idx, ClusterType.SPOT, probe=True)
            ts = time.strftime("%H:%M:%S")
            probe_console.print(f"[dim]{ts}[/dim] [cyan]Probe[/cyan] {region_name}...")

            success = False
            try:
                task = _get_task(region_idx, ClusterType.SPOT, is_probe=True)
                request_id = sdk.launch(task, cluster_name=cn)
                _register_cluster(cn)

                log_path = f"{probe_log_dir}/{now}-{region_name}.txt"
                with open(log_path, "w") as f:
                    sdk.stream_until_output_contains(
                        request_id, contains="CBL-INSTANCE-PENDING", output_stream=f
                    )
                success = True

            except Exception as e:
                log_path = f"{probe_log_dir}/{now}-{region_name}.txt"
                try:
                    with open(log_path, "r") as f:
                        log_content = f.read()
                except Exception:
                    log_content = ""

                error_type = config._classify_launch_error(log_content)
                if error_type == "quota":
                    probe_console.print(f"[bold red]⚠ QUOTA[/bold red] {region_name}")
                    _write_alert("quota_exceeded", region_name, log_path)
                elif error_type == "capacity":
                    logger.debug("Probe %s: no capacity", region_name)
                else:
                    err_msg = f"Probe failed for {cn}: {type(e).__name__}: {e}"
                    logger.error("[PROBE] %s", err_msg)
                    _write_alert("probe_launch_error", region_name, err_msg)

            # Update result
            config.latest_probe_result[region_name] = success
            last_probe_time[region_idx] = time.time()

            # Print result
            ts = time.strftime("%H:%M:%S")
            if success:
                probe_console.print(f"[dim]{ts}[/dim] [cyan]Probe[/cyan] {region_name}: [green]✓[/green]")
            else:
                probe_console.print(f"[dim]{ts}[/dim] [cyan]Probe[/cyan] {region_name}: [red]✗[/red]")

            # Terminate probe cluster
            try:
                sdk.down(cluster_name=cn)
            except Exception as e:
                logger.debug("Probe sdk.down error for %s: %s", cn, e)

            time.sleep(sleep_interval)

        except Exception as e:
            err_msg = f"Probe cycle failed: {type(e).__name__}: {e}"
            logger.error("[PROBE] %s", err_msg)
            _write_alert("probe_cycle_error", "all", err_msg)
            time.sleep(10)


def _cleanup_launched_clusters() -> None:
    """Cleanup all clusters launched during this simulation."""
    from rich.console import Console as RichConsole
    cleanup_console = RichConsole()
    
    clusters = _get_launched_clusters()
    if not clusters:
        logger.info("[CLEANUP] No clusters to clean up")
        return
    
    cleanup_console.print(f"\n[yellow]Cleaning up {len(clusters)} clusters...[/yellow]")
    for cluster_name in clusters:
        try:
            cleanup_console.print(f"  Terminating [cyan]{cluster_name}[/cyan]...")
            sdk.down(cluster_name)
            cleanup_console.print(f"  [green]✓[/green] {cluster_name}")
        except Exception as e:
            cleanup_console.print(
                f"  [red]✗[/red] Failed to terminate {cluster_name}: {e}"
            )
    cleanup_console.print("[green]Cleanup complete[/green]\n")
