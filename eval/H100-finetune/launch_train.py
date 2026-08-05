"""
Local launcher for SageMaker PyTorch Estimator to continue SFT from S3 checkpoints
or start fresh, using the train_entry.py entrypoint.

Usage:
  python launch_train.py -g A10G              # A10G (default)
  python launch_train.py -g L4                # L4
  python launch_train.py -g A100              # A100
  python launch_train.py --fake               # Fake workload for testing
  python launch_train.py --use-spot false     # On-demand instead of spot
"""

import argparse
import os
import time
import logging
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import yaml


DEFAULT_REGION = "us-west-2"
DEFAULT_GPU = "A10G"
logger = logging.getLogger(__name__)

# Fake workload config (only used when --fake is passed)
FAKE_TOTAL_STEPS = 4200
FAKE_STEP_SECONDS = 6.0
FAKE_SAVE_STEPS = 100
FAKE_SAVE_TOTAL_LIMIT = 3
FAKE_CHECKPOINT_SIZE_MB = 100
FAKE_INSTANCE_TYPE = "ml.m6i.large"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch SageMaker SFT training job")
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument(
        "--role",
        default=os.getenv(
            "SAGEMAKER_EXECUTION_ROLE_ARN",
            # Replace with your own SageMaker execution role ARN.
            "arn:aws:iam::<ACCOUNT_ID>:role/service-role/<SAGEMAKER_EXECUTION_ROLE>",
        ),
    )
    p.add_argument("--bucket", default="")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument(
        "--use-spot",
        type=lambda s: str(s).lower() in {"1", "true", "yes", "y"},
        default=True,
    )
    p.add_argument(
        "-g", "--gpu",
        choices=["A100", "L4", "A10G"],
        default=DEFAULT_GPU,
        help=f"GPU type: A100, L4, or A10G (default: {DEFAULT_GPU})",
    )
    p.add_argument(
        "--fake",
        action="store_true",
        help="Use fake workload instead of real GPU training",
    )
    return p.parse_args()


def launch_training_job(
    *,
    region: str,
    role: str,
    bucket: str | None,
    instance_count: int = 1,
    use_spot: bool = True,
    gpu: str = DEFAULT_GPU,
    use_fake_workload: bool = False,
) -> dict:
    if not role:
        raise ValueError("role is required (SageMaker execution role ARN)")

    # IMPORTANT: avoid global default session (thread-unsafe across regions)
    # Use an explicit regional boto session and pass it into SageMaker Session
    boto_sess = boto3.session.Session(region_name=region)
    sm_sess = sagemaker.Session(boto_session=boto_sess)
    _bucket = bucket or sm_sess.default_bucket()

    if use_fake_workload:
        # ========== FAKE WORKLOAD MODE ==========
        _job_prefix = f"fake-train-{time.strftime('%Y%m%d-%H%M%S')}"
        ckpt_s3_uri = f"s3://{_bucket}/{_job_prefix}/checkpoints/"

        expected_duration_seconds = FAKE_TOTAL_STEPS * FAKE_STEP_SECONDS
        deadline_ratio = 1.5
        deadline_seconds = int(expected_duration_seconds * deadline_ratio)
        cold_start_seconds = 30  # Fake workload uses shorter cold start

        logger.info(
            "[launch_train] FAKE MODE: steps=%d, step_sec=%.1f, save_steps=%d, ckpt_size=%.0fMB",
            FAKE_TOTAL_STEPS,
            FAKE_STEP_SECONDS,
            FAKE_SAVE_STEPS,
            FAKE_CHECKPOINT_SIZE_MB,
        )
        logger.info(
            "[launch_train] timing: expected_duration=%.1fs (%.2fh), deadline=%.1fs (%.2fh)",
            expected_duration_seconds,
            expected_duration_seconds / 3600.0,
            deadline_seconds,
            deadline_seconds / 3600.0,
        )

        # Use fake_train.py from eval/fake_workload/
        fake_workload_dir = str(Path(__file__).parent.parent / "fake_workload")

        est = PyTorch(
            entry_point="fake_train.py",
            source_dir=fake_workload_dir,
            role=role,
            framework_version="2.5.1",
            py_version="py311",
            instance_type=FAKE_INSTANCE_TYPE,
            instance_count=1,  # Fake workload is single-instance
            sagemaker_session=sm_sess,
            hyperparameters={},  # Fake train uses env vars, not hyperparameters
            checkpoint_s3_uri=ckpt_s3_uri,
            checkpoint_local_path="/opt/ml/checkpoints",
            keep_alive_period_in_seconds=0,
            disable_profiler=True,
            debugger_hook_config=False,
            use_spot_instances=bool(use_spot),
            max_run=deadline_seconds,
            max_wait=deadline_seconds if use_spot else None,
            environment={
                "TOTAL_STEPS": str(FAKE_TOTAL_STEPS),
                "STEP_SECONDS": str(FAKE_STEP_SECONDS),
                "SAVE_STEPS": str(FAKE_SAVE_STEPS),
                "SAVE_TOTAL_LIMIT": str(FAKE_SAVE_TOTAL_LIMIT),
                "OUTPUT_DIR": "/opt/ml/checkpoints",
                "CHECKPOINT_SIZE_MB": str(FAKE_CHECKPOINT_SIZE_MB),
            },
        )
    else:
        # ========== REAL TRAINING MODE ==========
        # Load shared training hyperparameters from config.yaml (no fallbacks)
        config_path = Path(__file__).with_name("config.yaml")
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Missing training config: {config_path}. "
                "This file is the single source of truth for training hyperparameters."
            )
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # GPU-specific configuration
        GPU_CONFIG = {
            "A100": {
                "instance_type": "ml.p4d.24xlarge",
                "job_prefix": "qwen3-14b-sft-a100",
                "cfg_prefix": "",  # No prefix for A100 (default)
            },
            "L4": {
                "instance_type": "ml.g6.12xlarge",
                "job_prefix": "qwen3-4b-sft-l4",
                "cfg_prefix": "L4_",
            },
            "A10G": {
                "instance_type": "ml.g5.12xlarge",
                "job_prefix": "qwen3-4b-sft-a10g",
                "cfg_prefix": "A10G_",
            },
        }
        if gpu not in GPU_CONFIG:
            raise ValueError(
                f"Unknown GPU type: {gpu}. Must be one of {list(GPU_CONFIG.keys())}"
            )

        gpu_cfg = GPU_CONFIG[gpu]
        _instance_type = gpu_cfg["instance_type"]
        _job_prefix = f"{gpu_cfg['job_prefix']}-{time.strftime('%Y%m%d-%H%M%S')}"
        cfg_prefix = gpu_cfg["cfg_prefix"]
        ckpt_s3_uri = f"s3://{_bucket}/{_job_prefix}/checkpoints/"

        # Helper to read config with prefix (L4_ or empty for A100)
        def get_cfg(key: str):
            prefixed_key = f"{cfg_prefix}{key}"
            if prefixed_key in cfg:
                return cfg[prefixed_key]
            return cfg[key]

        epochs = int(get_cfg("EPOCHS"))
        grad_accum_steps = int(get_cfg("GRAD_ACCUM_STEPS"))
        batch_size = int(get_cfg("BATCH_SIZE"))
        max_train_samples = int(get_cfg("MAX_TRAIN_SAMPLES"))
        _save_steps = int(get_cfg("SAVE_STEPS"))
        _save_total_limit = int(get_cfg("SAVE_TOTAL_LIMIT"))
        max_seq_len = int(get_cfg("MAX_SEQ_LEN"))
        model_size = int(get_cfg("MODELSIZE"))
        cold_start_seconds = int(get_cfg("COLD_START_SECONDS"))
        world_size = int(get_cfg("WORLD_SIZE"))

        # Estimate expected training duration from config (seconds).
        # num_steps = EPOCHS * MAX_TRAIN_SAMPLES / (BATCH_SIZE * world_size * GRAD_ACCUM_STEPS)
        cfg_steps_float = (
            epochs * max_train_samples / (batch_size * world_size * grad_accum_steps)
        )
        cfg_steps = int(round(cfg_steps_float))
        if abs(cfg_steps_float - cfg_steps) > 1e-6:
            raise ValueError(
                f"Config-implied num_steps is non-integer: {cfg_steps_float}"
            )
        _step_seconds = int(get_cfg("STEP_SECONDS"))
        expected_duration_seconds = cfg_steps * _step_seconds
        deadline_ratio = float(cfg["DEADLINE_RATIO"])
        deadline_seconds = int(expected_duration_seconds * deadline_ratio)

        logger.info(
            "[launch_train] GPU=%s, instance_type=%s",
            gpu,
            _instance_type,
        )
        logger.info(
            "[launch_train] config: epochs=%d, max_train_samples=%d, "
            "batch_size=%d, grad_accum=%d, world_size=%d, model_size=%d",
            epochs,
            max_train_samples,
            batch_size,
            grad_accum_steps,
            world_size,
            model_size,
        )
        logger.info(
            "[launch_train] timing: num_steps=%d, step_seconds=%d, "
            "expected_duration=%.1fs (%.2fh), deadline_ratio=%.2f, "
            "deadline=%.1fs (%.2fh)",
            cfg_steps,
            _step_seconds,
            expected_duration_seconds,
            expected_duration_seconds / 3600.0,
            deadline_ratio,
            deadline_seconds,
            deadline_seconds / 3600.0,
        )

        est = PyTorch(
            entry_point="train_qwen3_4b_qlora_single_gpu.py",
            source_dir=os.path.dirname(__file__),
            role=role,
            framework_version="2.5.1",
            py_version="py311",
            instance_type=_instance_type,
            instance_count=int(instance_count),
            sagemaker_session=sm_sess,
            hyperparameters={
                # Model/dataset
                "model-id": f"Qwen/Qwen3-{model_size}B",
                "dataset-id": "microsoft/orca-math-word-problems-200k",
                # Output / checkpoint directory (matches A100.yaml mount point)
                "output-dir": "/opt/ml/checkpoints/qwen3_sft",
                # Short SFT config matching A100.yaml run block
                "epochs": epochs,
                "optim": "adamw_torch",
                "per-device-train-batch-size": batch_size,
                "grad-accum-steps": grad_accum_steps,
                # "learning-rate": 2e-4,
                "gradient-checkpointing": True,
                # Precision / FSDP
                # "bf16": True,
                # "fp16": False,
                "use-fsdp": True,
                "fsdp-offload": True,
                # Sequence length & sampling
                "max-seq-length": max_seq_len,
                "max-train-samples": max_train_samples,
                "max-eval-samples": 1000,
                # Checkpointing cadence (A100.yaml uses 15)
                "save-steps": _save_steps,
                "save-total-limit": _save_total_limit,
                # Async sharded checkpointing similar to A100.yaml's --async-checkpoint true
                "async-checkpoint": True,
                "dcp-resume-optim": False,
            },
            checkpoint_s3_uri=ckpt_s3_uri,
            checkpoint_local_path="/opt/ml/checkpoints",
            keep_alive_period_in_seconds=0,
            disable_profiler=True,
            debugger_hook_config=False,
            enable_sagemaker_metrics=True,
            use_spot_instances=bool(use_spot),
            # Align SageMaker max_run/max_wait with simulated deadline:
            # deadline = deadline_ratio * expected_duration.
            max_run=deadline_seconds,
            max_wait=deadline_seconds if use_spot else None,
            distribution={"torch_distributed": {"enabled": True}},
            environment={
                "HF_TOKEN": os.getenv("HF_TOKEN", ""),
                "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
                "TOKENIZERS_PARALLELISM": os.getenv("TOKENIZERS_PARALLELISM", "false"),
                # Matches A100.yaml defaults to reduce CUDA allocator fragmentation
                "PYTORCH_CUDA_ALLOC_CONF": os.getenv(
                    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
                ),
            },
        )

    # Always start async, monitor handles waiting
    est.fit(wait=False)

    return {
        "job_name": est.latest_training_job.name,
        "bucket": _bucket,
        "checkpoint_s3_uri": ckpt_s3_uri,
        "region": region,
        "instance_type": FAKE_INSTANCE_TYPE if use_fake_workload else _instance_type,
        "use_spot": bool(use_spot),
        "task_duration_hours": expected_duration_seconds / 3600.0,
        "deadline_hours": deadline_seconds / 3600.0,
        "cold_start_seconds": cold_start_seconds,
    }


def main() -> None:
    from monitor_sagemaker_job import stream_monitor

    args = parse_args()
    res = launch_training_job(
        region=args.region,
        role=args.role,
        bucket=args.bucket or None,
        instance_count=args.instance_count,
        use_spot=bool(args.use_spot),
        gpu=args.gpu,
        use_fake_workload=args.fake,
    )

    # Stream monitor until job completes
    stream_monitor(
        job_name=res["job_name"],
        region=res["region"],
        deadline_hours=res["deadline_hours"],
        task_duration_hours=res["task_duration_hours"],
        restart_overhead_hours=res["cold_start_seconds"] / 3600.0,
    )


if __name__ == "__main__":
    main()
