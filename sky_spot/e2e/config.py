"""Configuration module for E2E testing.

This module contains all global configuration variables, GPU configurations,
zone mappings, and shared state used across E2E simulation components.
"""

import os
import sys
import time
import threading
import logging
import yaml
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


def _parse_gpu_arg() -> str:
    """Parse --gpu/-g argument early (before full argparse) since it affects module-level config."""
    default = "A10G"
    valid = {"A100", "L4", "A10G", "A10", "V100"}
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("--gpu", "-g") and i < len(sys.argv):
            val = sys.argv[i + 1].upper()
            if val == "A10":
                val = "A10G"
            if val not in valid:
                print(f"Error: --gpu must be one of {valid}, got '{val}'")
                sys.exit(1)
            return val
        if arg.startswith("--gpu="):
            val = arg.split("=", 1)[1].upper()
            if val == "A10":
                val = "A10G"
            if val not in valid:
                print(f"Error: --gpu must be one of {valid}, got '{val}'")
                sys.exit(1)
            return val
    return default


# ============================================================================
# WORKLOAD CONFIGURATION
# ============================================================================

# Workload modes:
# - USE_TRAINING_WORKLOAD=True: Real GPU training (expensive)
# - USE_FAKE_WORKLOAD=True: Fake CPU workload with same log format (cheap, for testing)
# - Both False: Simple echo workload (original test mode)
USE_TRAINING_WORKLOAD = True
USE_FAKE_WORKLOAD = False

# GPU Configuration:
# - "A100": A100:8 (p4d.24xlarge) - default, good availability
# - "L4": L4:4 (g6.12xlarge) - poor availability, good for arbitrage testing
# - "A10G": A10G:4 (g5.12xlarge) - moderate availability
# - "V100": V100:4 (p3.8xlarge) - older GPU, good availability
#
# Can be set via CLI: --gpu A10G or -g L4
# Accepts: A100, L4, A10G (or A10), V100
USE_GPU = _parse_gpu_arg()

# Backwards compatibility
USE_L4 = USE_GPU == "L4"

debug = False

# ============================================================================
# ZONE CONFIGURATION
# ============================================================================

# A100:8 zones (p4d.24xlarge) - good availability
A100_MULTI_REGION_ZONES = [
    # EU
    "eu-central-1a",
    "eu-central-1b",
    "eu-central-1c",
    # US West
    "us-west-2a",
    "us-west-2b",
    "us-west-2c",
    "us-west-2d",
    # Asia Pacific
    "ap-northeast-1a",
    "ap-northeast-1c",
    # US East
    "us-east-1a",
    "us-east-1b",
    "us-east-1c",
    "us-east-1d",
    "us-east-2b",
]

# L4:4 zones (g6.12xlarge) - poor availability, good for testing arbitrage
L4_MULTI_REGION_ZONES = [
    # US East - occasionally has capacity
    "us-east-2b",
    "us-east-2c",
    # US West - rarely has capacity, more expensive when available
    "us-west-2a",
    "us-west-2b",
    "us-west-2c",
    "us-west-2d",
    # EU Central
    "eu-central-1a",
    "eu-central-1b",
    # Asia Pacific
    "ap-northeast-1a",
    "ap-northeast-1c",
]

# A10G:4 zones (g5.12xlarge) - moderate availability
A10G_MULTI_REGION_ZONES = [
    # US East
    "us-east-1a",
    "us-east-1b",
    "us-east-1c",
    "us-east-1d",
    "us-east-1f",
    # US West
    "us-west-2a",
    "us-west-2b",
    "us-west-2c",
    # EU Central
    "eu-central-1a",
    "eu-central-1b",
    "eu-central-1c",
    # Asia Pacific
    "ap-northeast-1a",
    "ap-northeast-1c",
]

# V100:4 zones (p3.8xlarge) - older GPU, good availability
V100_MULTI_REGION_ZONES = [
    # US East
    "us-east-1a",
    "us-east-1b",
    "us-east-1c",
    "us-east-1d",
    "us-east-1e",
    "us-east-1f",
    "us-east-2a",
    "us-east-2b",
    "us-east-2c",
    # US West
    "us-west-2a",
    "us-west-2b",
    "us-west-2c",
    "us-west-2d",
    # EU Central
    "eu-central-1a",
    "eu-central-1b",
    # Asia Pacific
    "ap-northeast-1a",
    "ap-northeast-1c",
]

# Select zones based on USE_GPU
MULTI_REGION_ZONES_MAP = {
    "A100": A100_MULTI_REGION_ZONES,
    "L4": L4_MULTI_REGION_ZONES,
    "A10G": A10G_MULTI_REGION_ZONES,
    "V100": V100_MULTI_REGION_ZONES,
}
MULTI_REGION_ZONES = MULTI_REGION_ZONES_MAP[USE_GPU]

# ============================================================================
# TASK CONFIGURATION
# ============================================================================

# Trace files correspond to zones
trace_files = MULTI_REGION_ZONES

# Training configuration - loaded dynamically based on workload mode
run_suffix = time.strftime("%Y%m%d-%H%M%S")

if USE_TRAINING_WORKLOAD:
    setup_finished = '"event": "train_begin"'
    time_scale = 1.0

    # Load training config
    cfg_path = os.path.expanduser("eval/H100-finetune/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as cf:
        train_cfg = yaml.safe_load(cf) or {}

    # Select configuration based on USE_GPU
    GPU_CONFIG = {
        "A100": {
            "task_name": f"qwen3-14b-sft-a100-{run_suffix}",
            "instance_type": "p4d.24xlarge",
            "yaml_file": "eval/H100-finetune/A100.yaml",
            "cfg_prefix": "",
        },
        "L4": {
            "task_name": f"qwen3-4b-sft-l4-{run_suffix}",
            "instance_type": "g6.12xlarge",
            "yaml_file": "eval/H100-finetune/L4.yaml",
            "cfg_prefix": "L4_",
        },
        "A10G": {
            "task_name": f"qwen3-4b-sft-a10g-{run_suffix}",
            "instance_type": "g5.12xlarge",
            "yaml_file": "eval/H100-finetune/A10G.yaml",
            "cfg_prefix": "A10G_",
        },
        "V100": {
            "task_name": f"qwen3-4b-sft-v100-{run_suffix}",
            "instance_type": "p3.8xlarge",
            "yaml_file": "eval/H100-finetune/V100.yaml",
            "cfg_prefix": "V100_",
        },
    }
    gpu_cfg = GPU_CONFIG[USE_GPU]
    task_name = gpu_cfg["task_name"]
    instance_type = gpu_cfg["instance_type"]
    yaml_file = gpu_cfg["yaml_file"]
    cfg_prefix = gpu_cfg["cfg_prefix"]

    logger.info(f"[config] Using {USE_GPU} configuration: {instance_type}")

    # Load task YAML once at startup and cache it
    with open(os.path.expanduser(yaml_file), "r", encoding="utf-8") as f:
        _cached_task_yaml_config = yaml.safe_load(f.read())

    # Helper function to read config values with prefix
    def get_cfg(key: str) -> Any:
        prefixed_key = f"{cfg_prefix}{key}"
        if prefixed_key in train_cfg:
            return train_cfg[prefixed_key]
        return train_cfg[key]

    epochs_cfg = int(get_cfg("EPOCHS"))
    grad_accum_cfg = int(get_cfg("GRAD_ACCUM_STEPS"))
    batch_size_cfg = int(get_cfg("BATCH_SIZE"))
    max_train_samples_cfg = int(get_cfg("MAX_TRAIN_SAMPLES"))
    save_total_limit = int(get_cfg("SAVE_TOTAL_LIMIT"))
    MODEL_SIZE = int(get_cfg("MODELSIZE"))
    actual_cold_start_time_seconds = int(get_cfg("COLD_START_SECONDS"))
    step_seconds = int(get_cfg("STEP_SECONDS"))
    world_size_cfg = int(get_cfg("WORLD_SIZE"))
    DEADLINE_RATIO = float(train_cfg["DEADLINE_RATIO"])

    cfg_steps_float = (
        epochs_cfg
        * max_train_samples_cfg
        / (batch_size_cfg * world_size_cfg * grad_accum_cfg)
    )
    total_steps = int(round(cfg_steps_float))
    if abs(cfg_steps_float - total_steps) > 1e-6:
        raise ValueError(f"Config-implied num_steps is non-integer: {cfg_steps_float}")
    if total_steps <= 0:
        raise ValueError(
            f"total_steps must be > 0, got {total_steps} (check config values)"
        )

    actual_job_duration_seconds = step_seconds * total_steps
    CHECKPOINT_SIZE_GB = save_total_limit * MODEL_SIZE * 10  # 4 + 4 + 2

    logger.info(
        "[config] config: epochs=%d, max_train_samples=%d, "
        "batch_size=%d, grad_accum=%d, world_size=%d",
        epochs_cfg,
        max_train_samples_cfg,
        batch_size_cfg,
        grad_accum_cfg,
        world_size_cfg,
    )
    logger.info(
        "[config] timing: num_steps=%d, step_seconds=%d, job_duration=%.1fs (%.2fh)",
        total_steps,
        step_seconds,
        actual_job_duration_seconds,
        actual_job_duration_seconds / 3600.0,
    )

elif USE_FAKE_WORKLOAD:
    _cached_task_yaml_config = {}
    task_name = f"fake-train-{run_suffix}"
    setup_finished = '"event": "train_begin"'
    instance_type = "m6i.large"
    time_scale = 1.0

    # Fake training parameters
    total_steps = 4200
    step_seconds = 6.0
    save_steps = 100
    save_total_limit = 3
    checkpoint_size_mb = 100

    actual_cold_start_time_seconds = 30.0
    actual_job_duration_seconds = total_steps * step_seconds
    CHECKPOINT_SIZE_GB = save_total_limit * checkpoint_size_mb / 1024.0

    logger.info(
        "[config] FAKE WORKLOAD: steps=%d, step_sec=%.1f, ckpt_size=%.0fMB, time_scale=%.1f",
        total_steps,
        step_seconds,
        checkpoint_size_mb,
        time_scale,
    )

else:
    _cached_task_yaml_config = {}
    task_name = "test"
    setup_finished = "Hello, World!"
    instance_type = "m6i.2xlarge"
    time_scale = 100.0
    actual_cold_start_time_seconds = 60.0
    actual_job_duration_seconds = 600.0
    CHECKPOINT_SIZE_GB = 0.01

# ============================================================================
# TIMING CONFIGURATION
# ============================================================================

GAP_SECONDS = 60.0
PROBE_INTERVAL_TICKS = 12

# Cold start detection
DEFAULT_COLD_START_SECONDS = 300.0  # 5 minutes default
_launch_timestamps: Dict[str, float] = {}  # {cluster_name: launch_time}
_cold_start_measured: Dict[int, float] = {}  # {region_idx: cold_start_seconds}
_first_progress_seen: Dict[str, bool] = {}  # {cluster_name: True}


def get_cold_start_estimate(region: Optional[int] = None) -> float:
    """Get best estimate of cold start time.

    Returns measured cold start for specific region if available,
    otherwise max measured value, otherwise default.
    """
    if region is not None and region in _cold_start_measured:
        return _cold_start_measured[region]
    if _cold_start_measured:
        return max(_cold_start_measured.values())
    return DEFAULT_COLD_START_SECONDS

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Global output directory, set by run_simulation
current_output_dir: str = ""

# YAML task config (set by run_from_yaml, takes priority over USE_TRAINING_WORKLOAD)
_yaml_task_config: Optional[Dict[str, Any]] = None

# Latest probe results
latest_probe_result: Dict[str, bool] = {}

# Track all clusters launched
_launched_clusters_lock = threading.Lock()
_launched_clusters = set()

# Track preemption/termination events
_preemption_events: List[Dict[str, Any]] = []
_terminate_events: List[Dict[str, Any]] = []
_preemption_displayed_regions: set = set()

# Global reference to current env for threads
_current_env: Optional[Any] = None

# ============================================================================
# STRATEGY CONFIGURATION
# ============================================================================

# Available strategies for E2E testing (alias -> full name)
STRATEGY_ALIASES = {
    "risk": "unified_cost_model_rate_ratio",
    "single": "multi_region_single_rc_cr_threshold",
    "multibase": "multi_region_rc_cr_threshold_eager_failover",
}
DEFAULT_STRATEGY = "risk"

# ============================================================================
# ERROR CLASSIFICATION
# ============================================================================

# Error classification for launch failures
QUOTA_ERRORS = [
    "MaxSpotInstanceCountExceeded",
    "VcpuLimitExceeded",
    "VolumeLimitExceeded",
    "InsufficientAddressCapacity",
    "AddressLimitExceeded",
]

CAPACITY_ERRORS = [
    "InsufficientInstanceCapacity",
    "CapacityNotAvailable",
    "SpotMaxPriceTooLow",
    "ResourcesUnavailableError",
    "Failed to acquire resources",
    "Failed to provision",
]


def _classify_launch_error(log_content: str) -> str:
    """Classify launch error from log content.

    Returns: 'quota', 'capacity', or 'unknown'
    """
    if any(err in log_content for err in QUOTA_ERRORS):
        return "quota"
    if any(err in log_content for err in CAPACITY_ERRORS):
        return "capacity"
    return "unknown"


def _resolve_strategy(name: str) -> str:
    """Resolve strategy alias to full name."""
    return STRATEGY_ALIASES.get(name, name)
