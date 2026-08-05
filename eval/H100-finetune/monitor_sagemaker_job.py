#!/usr/bin/env python
"""Monitor SageMaker training jobs with human-readable streaming output."""

import argparse
import datetime as dt
import json
import math
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Set

import boto3
from botocore.config import Config
from tabulate import tabulate
from pathlib import Path
import os
from rich.console import Console

console = Console()

PRICING_REGION = "us-east-1"
REGION_TO_LOCATION = {
    "us-east-1": "US East (N. Virginia)",
    "us-west-2": "US West (Oregon)",
    "eu-west-1": "EU (Ireland)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
}

STATE_MAPPING = {
    "Starting": "COLD_START",
    "Downloading": "COLD_START",
    "Training": "MAKE_PROGRESS",
    "Uploading": "POST_RUN",
    "Stopped": "STOPPED",
    "Stopping": "STOPPING",
    "Interrupted": "INTERRUPTED",
    "Completed": "COMPLETED",
    "Failed": "FAILED",
}

TERMINAL_TRAINING_STATUSES = {"Completed", "Stopped", "Failed"}
DEADLINE_SECONDARY_STATUSES = {"MaxRuntimeExceeded", "MaxWaitTimeExceeded"}


def map_status(status: str, message: Optional[str]) -> str:
    status = status or "UNKNOWN"
    msg = (message or "").lower()
    if "insufficient capacity" in msg:
        return "WAITING_FOR_CAPACITY"
    if status == "Downloading":
        return "COLD_START"
    return STATE_MAPPING.get(status, status)


MONITORED_EVENTS = {"progress", "train_start", "train_end", "resume_from_checkpoint", "train_plan", "train_begin"}


def isoformat(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_ec2_spot_price(region: str, instance: str) -> Optional[float]:
    """Fetch current EC2 Spot price for the given instance type.

    SageMaker Managed Spot Training uses EC2 Spot instances, so the actual
    Spot price comes from EC2, not SageMaker Pricing API.

    BillableTime = TrainingTime × (SpotPrice / OnDemandPrice)
    ActualCost = BillableTime × OnDemandPrice = TrainingTime × SpotPrice
    """
    try:
        ec2 = boto3.client("ec2", region_name=region)
        # Convert ml.xxx to xxx (SageMaker instance to EC2 instance)
        ec2_instance = instance.replace("ml.", "") if instance.startswith("ml.") else instance

        response = ec2.describe_spot_price_history(
            InstanceTypes=[ec2_instance],
            ProductDescriptions=["Linux/UNIX"],
            MaxResults=10,
        )

        prices = [float(item["SpotPrice"]) for item in response.get("SpotPriceHistory", [])]
        if prices:
            return sum(prices) / len(prices)  # Average across AZs
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to fetch EC2 Spot price: {e}[/yellow]")
    return None


def fetch_sagemaker_ondemand_price(region: str, instance: str) -> Optional[float]:
    """Fetch SageMaker Training On-Demand price."""
    pricing = boto3.client("pricing", region_name=PRICING_REGION)
    next_token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {
            "ServiceCode": "AmazonSageMaker",
            "MaxResults": 100,
        }
        if next_token:
            kwargs["NextToken"] = next_token
        resp = pricing.get_products(**kwargs)
        for price_str in resp["PriceList"]:
            price = json.loads(price_str)
            attrs = price["product"].get("attributes", {})
            if attrs.get("regionCode") != region:
                continue
            attr_instance = attrs.get("instanceType")
            valid_instance = {
                instance,
                f"{instance}-Training",
                f"{instance}-training",
            }
            if attr_instance not in valid_instance:
                continue
            usage = attrs.get("usagetype", "").lower()
            component = attrs.get("component", "").lower()
            if "train" not in usage and "training" not in component:
                continue
            terms = price.get("terms", {}).get("OnDemand", {})
            for term in terms.values():
                for dim in term.get("priceDimensions", {}).values():
                    usd = dim.get("pricePerUnit", {}).get("USD")
                    if usd is not None:
                        try:
                            return float(usd)
                        except (TypeError, ValueError):
                            continue
        next_token = resp.get("NextToken")
        if not next_token:
            break
    return None


def fetch_price(region: str, instance: str, spot: bool) -> Optional[float]:
    """Fetch price for SageMaker instance.

    Always returns SageMaker On-Demand Training price.
    For Spot instances, AWS bills: BillableTimeInSeconds × OnDemandPrice
    where BillableTimeInSeconds already has the spot discount baked in.
    """
    return fetch_sagemaker_ondemand_price(region, instance)


def fetch_progress_events(
    log_client, job_name: str, region: str
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    paginator = log_client.get_paginator("describe_log_streams")
    for page in paginator.paginate(
        logGroupName="/aws/sagemaker/TrainingJobs",
        logStreamNamePrefix=job_name,
    ):
        for stream in page.get("logStreams", []):
            name = stream["logStreamName"]
            kwargs = {
                "logGroupName": "/aws/sagemaker/TrainingJobs",
                "logStreamName": name,
                "startFromHead": True,
            }
            next_token = None
            while True:
                if next_token:
                    kwargs["nextToken"] = next_token
                resp = log_client.get_log_events(**kwargs)
                for event in resp.get("events", []):
                    message = event.get("message", "").strip()
                    if not message.startswith("{"):
                        continue
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("event") not in MONITORED_EVENTS:
                        continue
                    payload["timestamp"] = event["timestamp"] / 1000.0
                    events.append(payload)
                next_token = resp.get("nextForwardToken")
                if not resp.get("events") or not next_token:
                    break
    events.sort(key=lambda e: e["timestamp"])
    return events


def latest_step_and_rate(
    events: List[Dict[str, Any]], window: int = 10
) -> Tuple[Optional[int], Optional[float]]:
    """Return (latest_step, seconds_per_step) computed from recent progress events.

    seconds_per_step is estimated using the slope over the last `window` events.
    """
    prog = [
        e
        for e in events
        if e.get("event") == "progress" and isinstance(e.get("step"), int)
    ]
    if len(prog) < 2:
        return (prog[-1]["step"] if prog else None, None)
    data = prog[-window:]
    t0, s0 = data[0]["timestamp"], data[0]["step"]
    t1, s1 = data[-1]["timestamp"], data[-1]["step"]
    if s1 <= s0:
        return (s1, None)
    sec_per_step = (t1 - t0) / float(s1 - s0)
    return (s1, sec_per_step)


def build_timeline(
    job_name: str,
    region: str,
    deadline: Optional[dt.datetime] = None,
    stop_if_impossible: bool = False,
) -> Dict[str, Any]:
    sm = boto3.client("sagemaker", region_name=region)
    logs = boto3.client(
        "logs", region_name=region, config=Config(retries={"max_attempts": 5})
    )

    desc = sm.describe_training_job(TrainingJobName=job_name)
    transitions = desc.get("SecondaryStatusTransitions", [])
    start = desc.get("TrainingStartTime") or desc.get("CreationTime")
    end = desc.get("TrainingEndTime") or dt.datetime.now(dt.timezone.utc)
    instance_type = desc["ResourceConfig"]["InstanceType"]
    is_spot = desc.get("EnableManagedSpotTraining", False)

    progress_events = fetch_progress_events(logs, job_name, region)
    # Extract planned max steps if available
    planned = next((e for e in progress_events if e.get("event") == "train_plan"), None)
    planned_max_steps = planned.get("planned_max_steps") if planned else None
    train_end_seen = any(e.get("event") == "train_end" for e in progress_events)
    latest_step, sec_per_step = latest_step_and_rate(progress_events)

    progress_complete: Optional[bool] = None
    if train_end_seen:
        progress_complete = True
    elif planned_max_steps is not None:
        progress_complete = bool(latest_step is not None and latest_step >= planned_max_steps)

    segments: List[Dict[str, Any]] = []
    if not transitions:
        status = desc.get("SecondaryStatus") or desc.get("TrainingJobStatus")
        segments.append(
            {
                "status": map_status(status, None),
                "start": isoformat(start),
                "end": isoformat(end),
                "made_progress": False,
            }
        )
    else:
        for idx, transition in enumerate(transitions):
            status = transition.get("Status")
            seg_start = transition.get("StartTime", start)
            if idx + 1 < len(transitions):
                seg_end = transitions[idx + 1].get("StartTime", end)
            else:
                seg_end = end
            message = transition.get("StatusMessage")
            seg_status = map_status(status, message)
            seg_start_ts = seg_start.timestamp()
            seg_end_ts = seg_end.timestamp()
            made_progress = any(
                seg_start_ts <= ev["timestamp"] < seg_end_ts
                for ev in progress_events
                if ev.get("event") == "progress"
            )
            segments.append(
                {
                    "status": seg_status,
                    "start": isoformat(seg_start),
                    "end": isoformat(seg_end),
                    "made_progress": made_progress,
                    "message": message,
                }
            )

    price = fetch_price(region, instance_type, is_spot)

    training_status = str(desc.get("TrainingJobStatus") or "")
    job_finished = training_status in TERMINAL_TRAINING_STATUSES
    last_segment = segments[-1] if segments else None
    last_status = (last_segment or {}).get("status") or map_status(
        desc.get("SecondaryStatus"), None
    )
    last_message = ((last_segment or {}).get("message") or "").lower()
    deadline_forced_stop = (
        last_status in DEADLINE_SECONDARY_STATUSES
        or "maxruntime" in last_message
        or "maxwaittime" in last_message
    )

    if progress_complete is False and (job_finished or deadline_forced_stop):
        raise RuntimeError(
            (
                "Training job %s@%s terminated (status=%s) before completing progress: "
                "latest_step=%s planned_max_steps=%s"
            )
            % (job_name, region, last_status or training_status, latest_step, planned_max_steps)
        )

    decision: Optional[Dict[str, Any]] = None
    if deadline and planned_max_steps and latest_step is not None and sec_per_step:
        now = dt.datetime.now(dt.timezone.utc)
        remaining_time = max((deadline - now).total_seconds(), 0.0)
        remaining_steps = max(planned_max_steps - latest_step, 0)
        # Estimate cold start cost as the last COLD_START segment duration (if any), else 300s
        cold_durations = []
        for seg in segments:
            if seg["status"] == "COLD_START":
                try:
                    st = dt.datetime.fromisoformat(seg["start"].replace("Z", "+00:00"))
                    en = dt.datetime.fromisoformat(seg["end"].replace("Z", "+00:00"))
                    cold_durations.append((en - st).total_seconds())
                except Exception:
                    pass
        expected_cold = cold_durations[-1] if cold_durations else 300.0
        remaining_progress_sec = remaining_steps * sec_per_step
        will_fail = remaining_time < (remaining_progress_sec + expected_cold)
        decision = {
            "remaining_time_sec": remaining_time,
            "remaining_steps": remaining_steps,
            "sec_per_step": sec_per_step,
            "expected_coldstart_sec": expected_cold,
            "condition": "remaining_time < remaining_progress + coldstart",
            "will_fail": will_fail,
        }
        if will_fail and stop_if_impossible:
            try:
                sm.stop_training_job(TrainingJobName=job_name)
                decision["action"] = "StopTrainingJob"
            except Exception as exc:  # noqa: BLE001
                decision["action_error"] = str(exc)

    return {
        "job_name": job_name,
        "region": region,
        "instance_type": instance_type,
        "spot": is_spot,
        "price_usd_per_hour": price,
        "segments": segments,
        "progress_events": progress_events,
        "plan_max_steps": planned_max_steps,
        "latest_step": latest_step,
        "sec_per_step": sec_per_step,
        "progress_complete": progress_complete,
        "train_end_seen": train_end_seen,
        "training_status": training_status,
        "safety_decision": decision,
    }


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 0:
        return "-"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m"
    else:
        return f"{int(seconds)}s"


def stream_monitor(
    job_name: str,
    region: str,
    interval: float = 30.0,
    output_file: Optional[str] = None,
    deadline_hours: Optional[float] = None,
    write_history: bool = True,
    task_duration_hours: Optional[float] = None,
    restart_overhead_hours: float = 0.0,
) -> None:
    """Stream human-readable monitoring output to console.

    If write_history=True, writes a history.jsonl compatible with plot_timeline.py
    to output/{job_name}-sagemaker/history.jsonl
    """
    import os
    sm = boto3.client("sagemaker", region_name=region)
    logs = boto3.client(
        "logs", region_name=region, config=Config(retries={"max_attempts": 3})
    )

    # Create output directory structure
    output_dir = f"output/{job_name}-sagemaker"

    seen_events: Set[float] = set()  # Track seen event timestamps
    last_status: Optional[str] = None
    last_step: Optional[int] = None
    job_start_time: Optional[dt.datetime] = None
    job_creation_time: Optional[dt.datetime] = None
    total_steps: Optional[int] = None
    deadline_dt: Optional[dt.datetime] = None
    tick_count: int = 0
    accumulated_cost: float = 0.0
    price_per_hour: Optional[float] = None
    instance_type: Optional[str] = None
    history_first_tick_written: bool = False
    # AZ tracking for multi-region timeline visualization
    # Pre-defined AZs per region (SageMaker typically uses these)
    REGION_AZS: Dict[str, List[str]] = {
        "us-west-2": ["us-west-2a", "us-west-2b", "us-west-2c", "us-west-2d"],
        "us-east-1": ["us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d", "us-east-1e", "us-east-1f"],
        "us-east-2": ["us-east-2a", "us-east-2b", "us-east-2c"],
        "eu-west-1": ["eu-west-1a", "eu-west-1b", "eu-west-1c"],
        "ap-northeast-1": ["ap-northeast-1a", "ap-northeast-1c", "ap-northeast-1d"],
        "ap-southeast-1": ["ap-southeast-1a", "ap-southeast-1b", "ap-southeast-1c"],
    }
    region_azs = REGION_AZS.get(region, [f"{region}a", f"{region}b", f"{region}c"])
    az_to_region_idx: Dict[str, int] = {az: i for i, az in enumerate(region_azs)}
    current_az: Optional[str] = None  # Current AZ where instance is running
    instance_start_time: Optional[dt.datetime] = None  # When current instance started (for age tracking)
    # For progress tracking in history
    task_target_seconds: Optional[float] = None
    if task_duration_hours:
        task_target_seconds = task_duration_hours * 3600.0
    gap_seconds: float = interval  # Use polling interval as gap_seconds

    # Check if we can write history (need task duration)
    if write_history and not task_target_seconds:
        console.print("[yellow]⚠️  History writing disabled (--task-duration-hours not provided)[/yellow]")
        write_history = False

    if write_history:
        os.makedirs(output_dir, exist_ok=True)
        history_path = f"{output_dir}/history.jsonl"
        history_handle = open(history_path, "w")
    else:
        history_handle = None

    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold]🔍 Monitoring: {job_name}[/bold]")
    console.print(f"   Region: [cyan]{region}[/cyan]")
    console.print(f"   Output: {output_dir}")
    if deadline_hours:
        console.print(f"   Deadline: [cyan]{deadline_hours:.1f}h[/cyan]")
    if write_history:
        console.print(f"   History: {output_dir}/history.jsonl")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

    output_handle = open(output_file, "a") if output_file else None

    def write_history_tick(
        done_seconds: float,
        is_running: bool,
        wall_time: dt.datetime,
        first_tick: bool = False,
    ) -> None:
        """Write a history tick compatible with plot_timeline.py."""
        nonlocal history_first_tick_written, tick_count, accumulated_cost
        if not history_handle or task_target_seconds is None:
            return

        tick_count += 1

        # Build history record
        record: Dict[str, Any] = {
            "Task/Target(seconds)": task_target_seconds,
            "Task/Done(seconds)": done_seconds,
            "Task/Remaining(seconds)": max(0, task_target_seconds - done_seconds),
            "Cost": accumulated_cost,
            "WallTime": wall_time.isoformat(),
            "Strategy": "sagemaker_spot",
        }

        # ActiveInstances - use correct region index based on current AZ
        if is_running and current_az and current_az in az_to_region_idx:
            region_idx = az_to_region_idx[current_az]
            record["ActiveInstances"] = {str(region_idx): "SPOT"}
        elif is_running:
            # Fallback to region 0 if no AZ info
            record["ActiveInstances"] = {"0": "SPOT"}
        else:
            record["ActiveInstances"] = {}

        # Add Config on first tick (RegionNames is fixed based on region's AZs)
        if first_tick and not history_first_tick_written:
            history_first_tick_written = True
            record["Config"] = {
                "RegionNames": region_azs,
                "TaskDurationHours": task_target_seconds / 3600.0 if task_target_seconds else 0,
                "DeadlineHours": deadline_hours or (task_target_seconds / 3600.0 * 1.5 if task_target_seconds else 0),
                "GapSeconds": gap_seconds,
                "InstanceType": instance_type or "unknown",
                "RestartOverheadHours": 0,  # SageMaker handles restarts internally
            }

        history_handle.write(json.dumps(record) + "\n")
        history_handle.flush()

    try:
        while True:
            loop_start = time.time()
            now_str = dt.datetime.now().strftime('%H:%M:%S')
            try:
                desc = sm.describe_training_job(TrainingJobName=job_name)
            except Exception as e:
                console.print(f"[dim]{now_str}[/dim] [red]❌ Error: {e}[/red]")
                # Sleep for remaining interval time
                elapsed = time.time() - loop_start
                time.sleep(max(0, interval - elapsed))
                continue

            # Get job times
            creation_time = desc.get("CreationTime")
            start_time = desc.get("TrainingStartTime")
            end_time = desc.get("TrainingEndTime")
            status = desc.get("TrainingJobStatus", "Unknown")
            secondary_status = desc.get("SecondaryStatus", "")
            instance_type = desc["ResourceConfig"]["InstanceType"]
            is_spot = desc.get("EnableManagedSpotTraining", False)

            if job_creation_time is None and creation_time:
                job_creation_time = creation_time
                if deadline_hours and deadline_dt is None:
                    deadline_dt = creation_time + dt.timedelta(hours=deadline_hours)

            if job_start_time is None and start_time:
                job_start_time = start_time
                spot_str = "[green]SPOT[/green]" if is_spot else "[blue]ON_DEMAND[/blue]"
                console.print(f"[dim]{now_str}[/dim] 🚀 Job started on {instance_type} ({spot_str})")
                # Fetch price for cost calculation
                if price_per_hour is None:
                    try:
                        price_per_hour = fetch_price(region, instance_type, is_spot)
                        if price_per_hour:
                            console.print(f"[dim]{now_str}[/dim] 💵 Price: [yellow]${price_per_hour:.2f}/hour[/yellow]")
                    except Exception:
                        pass

            # Check for status changes
            current_status = f"{status}/{secondary_status}"
            if current_status != last_status:
                if secondary_status == "Starting":
                    console.print(f"[dim]{now_str}[/dim] ⏳ Starting up...")
                elif secondary_status == "Downloading":
                    console.print(f"[dim]{now_str}[/dim] 📥 Downloading data/model...")
                elif secondary_status == "Training":
                    console.print(f"[dim]{now_str}[/dim] 🏃 Training in progress...")
                elif secondary_status == "Uploading":
                    console.print(f"[dim]{now_str}[/dim] 📤 Uploading model/checkpoints...")
                elif secondary_status == "Interrupted":
                    console.print(f"[dim]{now_str}[/dim] [bold red]⚠️  PREEMPTED[/bold red] - spot instance reclaimed")
                    instance_start_time = None  # Reset age tracking
                elif secondary_status == "MaxWaitTimeExceeded":
                    console.print(f"[dim]{now_str}[/dim] ⏰ Max wait time exceeded")
                elif secondary_status == "MaxRuntimeExceeded":
                    console.print(f"[dim]{now_str}[/dim] ⏰ Max runtime exceeded")
                elif status == "Completed":
                    console.print(f"[dim]{now_str}[/dim] [bold green]✅ Job completed successfully[/bold green]")
                elif status == "Failed":
                    reason = desc.get("FailureReason", "Unknown")
                    console.print(f"[dim]{now_str}[/dim] [bold red]❌ Job failed: {reason}[/bold red]")
                elif status == "Stopped":
                    console.print(f"[dim]{now_str}[/dim] 🛑 Job stopped")
                last_status = current_status

            # Fetch progress events from CloudWatch
            try:
                progress_events = fetch_progress_events(logs, job_name, region)
            except Exception:
                progress_events = []

            # Process new events
            for ev in progress_events:
                ts = ev.get("timestamp", 0)
                if ts in seen_events:
                    continue
                seen_events.add(ts)

                event_type = ev.get("event", "")
                event_time = dt.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                if event_type == "train_plan":
                    # Extract total steps from train_plan event
                    planned = ev.get("planned_max_steps")
                    if planned and total_steps is None:
                        total_steps = int(planned)
                        console.print(f"[dim]{event_time}[/dim] 📋 Training plan: {total_steps} steps")

                elif event_type == "train_begin" or event_type == "train_start":
                    step = ev.get("step", 0)
                    # Record instance start time for age tracking
                    instance_start_time = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
                    # Extract availability zone if present
                    az = ev.get("availability_zone")
                    if az:
                        current_az = az
                        console.print(f"[dim]{event_time}[/dim] 🟢 Launched [green]SPOT[/green] in [cyan]{az}[/cyan]")
                    else:
                        console.print(f"[dim]{event_time}[/dim] 🟢 Launched [green]SPOT[/green] in [cyan]{region}[/cyan]")

                elif event_type == "resume_from_checkpoint":
                    ckpt = ev.get("checkpoint", "")
                    step = ev.get("restored_step", ev.get("step", "?"))
                    console.print(f"[dim]{event_time}[/dim] 🔄 Resumed from checkpoint at step {step}")

                elif event_type == "progress":
                    step = ev.get("step", 0)
                    total = ev.get("total", total_steps)
                    if total:
                        total_steps = total
                    pct = ev.get("pct", (step / total * 100) if total else 0)

                    # Only print if step changed significantly
                    if last_step is None or step >= last_step + 10 or step == total:
                        now = dt.datetime.now(dt.timezone.utc)

                        # Instance age
                        age_str = ""
                        if instance_start_time:
                            age_sec = (now - instance_start_time).total_seconds()
                            age_h = int(age_sec // 3600)
                            age_m = int((age_sec % 3600) // 60)
                            age_str = f"{age_h}h{age_m:02d}m"

                        # Task progress: done and remaining
                        if task_target_seconds:
                            done_sec = (pct / 100.0) * task_target_seconds
                            remain_sec = task_target_seconds - done_sec
                            done_h = int(done_sec // 3600)
                            done_m = int((done_sec % 3600) // 60)
                            remain_h = int(remain_sec // 3600)
                            remain_m = int((remain_sec % 3600) // 60)
                            progress_str = f"{done_h}h{done_m:02d}m/{remain_h}h{remain_m:02d}m {pct:.0f}%"
                        else:
                            progress_str = f"{step}/{total} ({pct:.1f}%)"

                        # Deadline remaining
                        dl_str = ""
                        if job_creation_time and deadline_dt:
                            elapsed_sec = (now - job_creation_time).total_seconds()
                            deadline_remaining_sec = max((deadline_dt - now).total_seconds(), 0)
                            el_h = int(elapsed_sec // 3600)
                            el_m = int((elapsed_sec % 3600) // 60)
                            dl_h = int(deadline_remaining_sec // 3600)
                            dl_m = int((deadline_remaining_sec % 3600) // 60)
                            dl_str = f"{el_h}h{el_m:02d}m/{dl_h}h{dl_m:02d}m"

                        # Build status: 🟢AZ age | progress | cost | deadline
                        az_display = current_az or region
                        status = f"🟢{az_display}"
                        if age_str:
                            status += f" {age_str}"

                        console.print(
                            f"[dim]{event_time}[/dim] {status} | "
                            f"[green]{done_h}h{done_m:02d}m[/green]/[yellow]{remain_h}h{remain_m:02d}m[/yellow] "
                            f"[bold green]{pct:.0f}%[/bold green] | "
                            f"[yellow]${accumulated_cost:.2f}[/yellow] | "
                            f"{el_h}h{el_m:02d}m/[cyan]{dl_h}h{dl_m:02d}m[/cyan]"
                        )
                        last_step = step

                elif event_type == "train_end":
                    step = ev.get("step", "?")
                    console.print(f"[dim]{event_time}[/dim] [bold green]🏁 Training complete[/bold green] at step {step}")

                elif event_type == "dcp_async_save_completed":
                    step = ev.get("step", "?")
                    console.print(f"[dim]{event_time}[/dim] 💾 Checkpoint saved at step {step}")

            # Write full snapshot to file if specified
            if output_handle:
                snapshot = {
                    "job_name": job_name,
                    "region": region,
                    "status": status,
                    "secondary_status": secondary_status,
                    "instance_type": instance_type,
                    "spot": is_spot,
                    "latest_step": last_step,
                    "total_steps": total_steps,
                    "poll_ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
                output_handle.write(json.dumps(snapshot) + "\n")
                output_handle.flush()

            # Calculate progress in seconds based on step/total_steps
            done_seconds = 0.0
            if task_target_seconds and last_step is not None and total_steps and total_steps > 0:
                done_seconds = (last_step / total_steps) * task_target_seconds

            # Write history tick for timeline plotting
            if write_history and task_target_seconds:
                now = dt.datetime.now(dt.timezone.utc)
                is_training = secondary_status == "Training"

                # Update accumulated cost from BillableTimeInSeconds
                # AWS bills: BillableTimeInSeconds × OnDemandPrice
                # BillableTimeInSeconds already has spot discount baked in
                billable_seconds = desc.get("BillableTimeInSeconds", 0)
                if billable_seconds and price_per_hour:
                    accumulated_cost = (billable_seconds / 3600.0) * price_per_hour

                # Write history tick
                is_first_tick = not history_first_tick_written
                write_history_tick(
                    done_seconds=done_seconds,
                    is_running=is_training,
                    wall_time=now,
                    first_tick=is_first_tick,
                )

            # Safety net check: cancel job if we can't finish before deadline
            # Logic matches test.py: needed >= remaining_time (with restart overhead, rounded to gap)
            if task_target_seconds and deadline_dt and status not in {"Completed", "Failed", "Stopped"}:
                now_check = dt.datetime.now(dt.timezone.utc)
                remaining_task_seconds = task_target_seconds - done_seconds
                restart_overhead_seconds = restart_overhead_hours * 3600.0

                # Round remaining_time down to whole intervals (like test.py)
                raw_remaining_time = (deadline_dt - now_check).total_seconds()
                remaining_time_seconds = math.floor(raw_remaining_time / gap_seconds) * gap_seconds

                # Round needed time up to whole intervals (like test.py)
                needed_seconds = math.ceil((remaining_task_seconds + restart_overhead_seconds) / gap_seconds) * gap_seconds

                # Trigger if needed >= remaining (same as test.py)
                if needed_seconds >= remaining_time_seconds and remaining_task_seconds > 0:
                    console.print(f"\n[bold red]{'=' * 60}[/bold red]")
                    console.print(f"[bold red]⚠️  SAFETY NET TRIGGERED[/bold red]")
                    console.print(f"   Remaining task: {remaining_task_seconds/3600:.2f}h")
                    console.print(f"   Restart overhead: {restart_overhead_seconds/3600:.2f}h")
                    console.print(f"   Needed (ceil): {needed_seconds/3600:.2f}h")
                    console.print(f"   Remaining time (floor): {remaining_time_seconds/3600:.2f}h")
                    console.print(f"   Cannot finish before deadline - cancelling job")
                    console.print(f"[bold red]{'=' * 60}[/bold red]\n")

                    try:
                        sm.stop_training_job(TrainingJobName=job_name)
                        console.print(f"[dim]{now_str}[/dim] 🛑 Job cancelled by safety net")
                    except Exception as e:
                        console.print(f"[dim]{now_str}[/dim] [red]Failed to cancel job: {e}[/red]")
                    break

            # Check if job is done
            if status in {"Completed", "Failed", "Stopped"}:
                # Calculate cost if possible
                if start_time and end_time:
                    duration = (end_time - start_time).total_seconds()
                    billable = desc.get("BillableTimeInSeconds", duration)
                    console.print(f"[dim]{now_str}[/dim] ⏱️  Duration: {format_duration(duration)}, Billed: {format_duration(billable)}")
                    if billable < duration:
                        savings = (1 - billable / duration) * 100
                        console.print(f"[dim]{now_str}[/dim] 💰 Spot savings: [green]{savings:.1f}%[/green]")
                console.print(f"[bold cyan]{'-' * 60}[/bold cyan]")
                break

            # Sleep for remaining interval time to maintain precise timing
            elapsed = time.time() - loop_start
            time.sleep(max(0, interval - elapsed))

    except KeyboardInterrupt:
        console.print(f"\n[dim]{dt.datetime.now().strftime('%H:%M:%S')}[/dim] Monitoring stopped by user")
    finally:
        if output_handle:
            output_handle.close()
        if history_handle:
            history_handle.close()
            console.print(f"History saved to: {output_dir}/history.jsonl")
            console.print(f"Generate timeline: [cyan]python plot_timeline.py {job_name}[/cyan]")


def _load_jobs_from_file(path: str) -> List[Tuple[str, str]]:
    """Load job@region tuples from JSON or JSONL.

    Accepts JSONL with objects containing {job_name, region}, or a JSON array of
    such objects or ["name@region", ...].
    """
    import json

    pairs: List[Tuple[str, str]] = []
    with open(path, "r") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith("["):
            payload = json.load(f)
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, str) and "@" in item:
                        name, region = item.split("@", 1)
                        pairs.append((name.strip(), region.strip()))
                    elif isinstance(item, dict):
                        name = item.get("job_name") or item.get("name")
                        region = item.get("region")
                        if name and region:
                            pairs.append((str(name), str(region)))
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                name = obj.get("job_name") or obj.get("name")
                region = obj.get("region")
                if name and region:
                    pairs.append((str(name), str(region)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor SageMaker training jobs")
    parser.add_argument("jobs", nargs="*", help="Training job names or name@region")
    parser.add_argument("--region", default="us-west-2", help="Default AWS region")
    parser.add_argument("--jobs-file", help="JSONL/JSON file with jobs and regions")
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Polling interval in seconds (default: 30).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to JSONL file for appending snapshots (one per line).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of human-readable streaming logs.",
    )
    parser.add_argument(
        "--delta-only",
        action="store_true",
        help=(
            "Write only incremental changes (new progress events or new status segments) to --output."
            " Requires a cursor; see --cursor-file."
        ),
    )
    parser.add_argument(
        "--cursor-file",
        type=str,
        help=(
            "Path to persist last-seen state for --delta-only. If omitted, uses <output>.cursor.json "
            "when --output is set, else ~/.cache/sm_monitor_cursor.json."
        ),
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print a compact table summary instead of raw JSON.",
    )
    parser.add_argument(
        "--deadline-iso",
        type=str,
        help="Deadline in ISO8601 (e.g., 2025-11-10T08:00:00Z).",
    )
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        help="Deadline as seconds from now (e.g., 108000 for 30h).",
    )
    parser.add_argument(
        "--stop-if-impossible",
        action="store_true",
        help="If true, stop the job when remaining_time < remaining_progress + coldstart.",
    )
    parser.add_argument(
        "--task-duration-hours",
        type=float,
        help="Expected task duration in hours (required for history writing).",
    )
    parser.add_argument(
        "--deadline-hours",
        type=float,
        help="Deadline in hours from job creation (for timeline plotting).",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable writing history.jsonl for timeline plotting.",
    )
    args = parser.parse_args()

    # Build (job_name, region) pairs from CLI and/or file
    job_pairs: List[Tuple[str, str]] = []
    if args.jobs_file:
        job_pairs.extend(_load_jobs_from_file(args.jobs_file))
    for token in args.jobs:
        if "@" in token:
            name, region = token.split("@", 1)
            job_pairs.append((name.strip(), region.strip()))
        else:
            job_pairs.append((token, args.region))

    if not job_pairs:
        parser.error("No jobs provided (use positional args or --jobs-file)")

    # Default: use streaming monitor (human-readable output)
    if not args.json and not args.table and len(job_pairs) == 1:
        job_name, job_region = job_pairs[0]
        stream_monitor(
            job_name=job_name,
            region=job_region,
            interval=args.interval,
            output_file=args.output,
            deadline_hours=args.deadline_hours,
            write_history=not args.no_history,
            task_duration_hours=args.task_duration_hours,
        )
        return

    interval = max(args.interval, 0.0)

    # Cursor helpers for delta mode
    def _default_cursor_path() -> Optional[str]:
        if args.cursor_file:
            return args.cursor_file
        if args.output:
            return str(Path(args.output).with_suffix(Path(args.output).suffix + ".cursor.json"))
        cache = os.path.expanduser("~/.cache")
        try:
            os.makedirs(cache, exist_ok=True)
        except Exception:
            pass
        return os.path.join(cache, "sm_monitor_cursor.json")

    cursor_path: Optional[str] = _default_cursor_path() if args.delta_only else None

    def _load_cursor() -> Dict[str, Any]:
        if not cursor_path:
            return {}
        try:
            with open(cursor_path, "r") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _save_cursor(state: Dict[str, Any]) -> None:
        if not cursor_path:
            return
        try:
            with open(cursor_path, "w") as fh:
                json.dump(state, fh)
        except Exception:
            pass

    cursor_state: Dict[str, Any] = _load_cursor()

    output_handle = open(args.output, "a", buffering=1) if args.output else None

    try:
        while True:
            loop_start = time.time()
            snapshots: List[Dict[str, Any]] = []
            poll_timestamp = isoformat(dt.datetime.now(dt.timezone.utc))
            # Resolve deadline (once per poll so --deadline-seconds counts down)
            deadline_dt: Optional[dt.datetime] = None
            if args.deadline_iso:
                try:
                    s = args.deadline_iso.replace("Z", "+00:00")
                    deadline_dt = dt.datetime.fromisoformat(s)
                except Exception:
                    deadline_dt = None
            elif args.deadline_seconds:
                deadline_dt = dt.datetime.now(dt.timezone.utc) + dt.timedelta(
                    seconds=float(args.deadline_seconds)
                )
            for (job, job_region) in job_pairs:
                try:
                    snapshot = build_timeline(
                        job,
                        job_region,
                        deadline=deadline_dt,
                        stop_if_impossible=args.stop_if_impossible,
                    )
                except Exception as exc:  # noqa: BLE001
                    snapshot = {
                        "job_name": job,
                        "region": job_region,
                        "error": str(exc),
                    }
                snapshot["poll_ts"] = poll_timestamp
                snapshots.append(snapshot)

                # Write output: full or delta-only
                if output_handle:
                    if not args.delta_only:
                        output_handle.write(json.dumps(snapshot) + "\n")
                        output_handle.flush()
                    else:
                        # Compute delta against cursor
                        job_key = f"{snapshot.get('job_name','')}@{snapshot.get('region','')}"
                        prev = cursor_state.get(job_key, {}) if isinstance(cursor_state, dict) else {}
                        segs = snapshot.get("segments") or []
                        # New segments based on length increase or status change
                        prev_len = int(prev.get("segments_len", 0)) if isinstance(prev, dict) else 0
                        segments_added: List[Dict[str, Any]] = []
                        if len(segs) > prev_len:
                            segments_added = segs[prev_len:]
                            prev_len = len(segs)
                        else:
                            # If last status changed since last tick, include the last segment once
                            last_status = segs[-1].get("status") if segs else None
                            last_status_prev = prev.get("last_status") if isinstance(prev, dict) else None
                            if last_status and last_status != last_status_prev and segs:
                                segments_added = [segs[-1]]
                        # New progress events by timestamp
                        last_ev_ts = float(prev.get("last_event_ts", 0.0)) if isinstance(prev, dict) else 0.0
                        new_prog: List[Dict[str, Any]] = []
                        for ev in snapshot.get("progress_events") or []:
                            ts = ev.get("timestamp") or ev.get("ts")
                            try:
                                tsf = float(ts)
                            except Exception:
                                tsf = 0.0
                            if tsf > last_ev_ts:
                                new_prog.append(ev)
                                if tsf > last_ev_ts:
                                    last_ev_ts = tsf
                        # Only emit a line if there is something new
                        if segments_added or new_prog:
                            delta = {
                                "job_name": snapshot.get("job_name"),
                                "region": snapshot.get("region"),
                                "instance_type": snapshot.get("instance_type"),
                                "spot": snapshot.get("spot"),
                                "price_usd_per_hour": snapshot.get("price_usd_per_hour"),
                                "latest_step": snapshot.get("latest_step"),
                                "sec_per_step": snapshot.get("sec_per_step"),
                                "delta": {
                                    "segments_added": segments_added,
                                    "progress_events": new_prog,
                                    "status": (segs[-1].get("status") if segs else None),
                                },
                                "poll_ts": poll_timestamp,
                            }
                            output_handle.write(json.dumps(delta) + "\n")
                            output_handle.flush()
                        # Update cursor regardless to avoid re-emitting on next tick
                        cursor_state[job_key] = {
                            "segments_len": prev_len,
                            "last_status": (segs[-1].get("status") if segs else None),
                            "last_event_ts": last_ev_ts,
                        }
                        _save_cursor(cursor_state)
            if not output_handle:
                if args.table:
                    # Compact table: region, instance, spot, job, status, last_ts, error
                    table_rows: List[List[str]] = []
                    for s in snapshots:
                        last_status = "?"
                        last_ts = ""
                        error = s.get("error") or ""
                        if not error:
                            segs = s.get("segments") or []
                            if segs:
                                last = segs[-1]
                                last_status = last.get("state") or last.get("status") or "?"
                                last_ts = last.get("endTime") or last.get("startTime") or ""
                        table_rows.append(
                            [
                                s.get("region", ""),
                                s.get("instance_type", ""),
                                "spot" if s.get("spot") else "on-demand",
                                s.get("job_name", ""),
                                last_status,
                                last_ts,
                                (s.get("FailureReason") or "")[:60] if not error else error[:60],
                            ]
                        )
                    print(
                        tabulate(
                            table_rows,
                            headers=[
                                "Region",
                                "Instance",
                                "Type",
                                "Job",
                                "LatestStatus",
                                "LastTS",
                                "Error/Reason",
                            ],
                            tablefmt="github",
                        )
                    )
                else:
                    json.dump(
                        snapshots if len(snapshots) > 1 else snapshots[0],
                        sys.stdout,
                        indent=2,
                    )
                    sys.stdout.write("\n")
                    sys.stdout.flush()

            if interval <= 0:
                break

            # Sleep for remaining interval time to maintain precise timing
            elapsed = time.time() - loop_start
            time.sleep(max(0, interval - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        if output_handle:
            output_handle.close()


if __name__ == "__main__":
    main()
