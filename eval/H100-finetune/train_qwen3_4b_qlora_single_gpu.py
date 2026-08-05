"""
Multi-GPU FSDP training script for Qwen3 (14B/30B/MoE).

Goal
- Run on 8x H100/A100 GPUs using FSDP for memory-efficient training
- Full-parameter fine-tuning of Qwen3-32B in fp16 precision

Features
- Loads chat data from a Hugging Face dataset
- Converts conversations/messages into a chat template with the Qwen tokenizer
- FSDP (Fully Sharded Data Parallel) for large model training
- Gradient checkpointing for memory efficiency
- Mixed precision training (fp16)

Usage (example)
  torchrun --nproc_per_node=8 train_qwen3_4b_qlora_single_gpu.py \
    --model-id Qwen/Qwen3-14B \
    --dataset-id microsoft/orca-math-word-problems-200k \
    --output-dir ./outputs/qwen3_32b_sft \
    --epochs 1 --per-device-train-batch-size 1 --grad-accum-steps 4 \
    --gradient-checkpointing true --use-fsdp true

Environment
- Requires PyTorch 2.5+ with CUDA support
- transformers, datasets, accelerate
- Run with torchrun for multi-GPU training
"""

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
)
# NOTE: Do not import a specific decoder layer class here.
# Qwen3 has both dense and MoE variants. The class used for FSDP
# auto-wrapping must match the actual architecture:
# - Dense:  Qwen3DecoderLayer (model_type: 'qwen3')
# - MoE:    Qwen3MoeDecoderLayer (model_type: 'qwen3_moe')
# We detect the correct class name from the model config below and pass
# it to Accelerate via TrainingArguments.fsdp_config.

# Allow RNG checkpoint files saved with older numpy pickling format to be restored safely.
torch.serialization.add_safe_globals(
    [np.core.multiarray._reconstruct, np.ndarray, np.core.multiarray.scalar]
)


def _log_event(event: str, **payload: Any) -> None:
    # Only rank 0 should log to avoid duplicate logs in distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        return
    body = {"ts": time.time(), "event": event, **payload}
    print(json.dumps(body, default=str), flush=True)


def _get_availability_zone() -> str:
    """Get AWS availability zone from instance metadata."""
    import urllib.request
    try:
        # IMDSv1 (simple GET)
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/placement/availability-zone",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception:
        pass
    # Try IMDSv2 with token
    try:
        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        with urllib.request.urlopen(token_req, timeout=2) as resp:
            token = resp.read().decode("utf-8")
        az_req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/placement/availability-zone",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(az_req, timeout=2) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception:
        return "unknown"


def _log_cuda_mem(stage: str, has_optim: bool) -> None:
    """Log current CUDA memory usage for debugging DCP resume.

    Only logs on rank 0 and when CUDA is available.
    """
    try:
        if not torch.cuda.is_available():
            return
        # Ensure all pending work is accounted for
        torch.cuda.synchronize()
        device = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        max_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        _log_event(
            "cuda_mem_snapshot",
            stage=stage,
            has_optim=bool(has_optim),
            cuda_alloc_gb=round(float(alloc), 3),
            cuda_max_alloc_gb=round(float(max_alloc), 3),
        )
    except Exception:
        # Never let logging failures break training
        return


def _dcp_save_sync(sd: Dict[str, Any], ckpt_dir: str) -> str:
    """Version-compatible DCP sync save.

    Tries high-level API first; falls back to FileSystemWriter if needed.
    Returns a short string indicating which path was used.
    """
    import inspect
    import torch.distributed.checkpoint as dcp

    # 1) Newer API: save_state_dict(state_dict, checkpoint_id=...)
    if hasattr(dcp, "save_state_dict"):
        try:
            sig = inspect.signature(dcp.save_state_dict)
            if "checkpoint_id" in sig.parameters:
                dcp.save_state_dict(sd, checkpoint_id=ckpt_dir)
                return "save_state_dict_checkpoint_id"
        except Exception:
            pass
        # 2) Older API: save_state_dict(..., storage_writer=FileSystemWriter(...))
        try:
            from torch.distributed.checkpoint.filesystem import FileSystemWriter

            writer = FileSystemWriter(ckpt_dir)
            try:
                dcp.save_state_dict(sd, storage_writer=writer)
            except TypeError:
                # some builds may use 'writer' kw
                dcp.save_state_dict(sd, writer=writer)
            return "save_state_dict_filesystem_writer"
        except Exception:
            pass
    # 3) Fallback: dcp.save(state_dict, checkpoint_id=...)
    if hasattr(dcp, "save"):
        dcp.save(sd, checkpoint_id=ckpt_dir)
        return "save_checkpoint_id"
    raise RuntimeError("No compatible DCP save API found")


def _dcp_load_sync(ckpt_dir: str, model, optimizer) -> str:
    """Version-compatible DCP load into (model, optimizer).

    Note:
        When ``optimizer`` is None (which is the case before Trainer/Accelerate
        has constructed it), we only restore the model weights. This avoids
        both:
          * set_state_dict(..., optimizers=None) API misuse, and
          * forcing early optimizer construction (which caused FSDP OOMs).
    """
    import inspect
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_state_dict,
        get_model_state_dict,
        set_state_dict,
        set_model_state_dict,
    )

    has_optim = optimizer is not None
    _log_cuda_mem("dcp_load_start", has_optim)

    # Build initial state dict to load into.
    if optimizer is None:
        model_sd = get_model_state_dict(model)
        sd = {"model": model_sd}
    else:
        model_sd, optim_sd = get_state_dict(model, optimizer)
        sd = {"model": model_sd, "optim": optim_sd}

    # 1) Newer API: load_state_dict(state_dict, checkpoint_id=...)
    if hasattr(dcp, "load_state_dict"):
        try:
            sig = inspect.signature(dcp.load_state_dict)
            if "checkpoint_id" in sig.parameters:
                dcp.load_state_dict(sd, checkpoint_id=ckpt_dir)
                _log_cuda_mem("after_dcp_load_state_dict", has_optim)
            else:
                # 2) Older API: load_state_dict(..., storage_reader=FileSystemReader(...))
                from torch.distributed.checkpoint.filesystem import FileSystemReader

                reader = FileSystemReader(ckpt_dir)
                try:
                    dcp.load_state_dict(sd, storage_reader=reader)
                except TypeError:
                    dcp.load_state_dict(sd, reader=reader)
                _log_cuda_mem("after_dcp_load_state_dict", has_optim)

            # Apply loaded state back to model / optimizer.
            if optimizer is None:
                set_model_state_dict(model, sd["model"])
            else:
                set_state_dict(
                    model,
                    optimizer,
                    model_state_dict=sd["model"],
                    optim_state_dict=sd["optim"],
                )
            _log_cuda_mem("after_set_state_dict", has_optim)
            return "load_state_dict"
        except Exception:
            pass

    # 3) Fallback: dcp.load(state_dict, checkpoint_id=...)
    if hasattr(dcp, "load"):
        dcp.load(sd, checkpoint_id=ckpt_dir)
        _log_cuda_mem("after_dcp_load", has_optim)
        if optimizer is None:
            set_model_state_dict(model, sd["model"])
        else:
            set_state_dict(
                model,
                optimizer,
                model_state_dict=sd["model"],
                optim_state_dict=sd["optim"],
            )
        _log_cuda_mem("after_set_state_dict", has_optim)
        return "load_checkpoint_id"

    raise RuntimeError("No compatible DCP load API found")


class ProgressLoggerCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self._async_future = None  # for DCP async checkpointing

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        _log_event("train_begin", step=state.global_step)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        payload = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        payload["step"] = state.global_step
        _log_event("progress", **payload)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # No-op when using pure DCP checkpointing; Trainer save_strategy='no'.
        return

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # Wait for any in-flight async checkpoint to finish.
        try:
            if getattr(self, "_async_future", None) is not None:
                self._async_future.result()
                self._async_future = None
        except Exception as _e:  # noqa: N806
            print(f"[WARN] Waiting on async checkpoint failed: {_e}")
        _log_event("train_end", step=state.global_step)


class WatchdogCallback(TrainerCallback):
    """Kill training if no progress for too long (detects NCCL hangs, etc.).

    Uses two timeout phases:
    - Initial phase (resume + first step): longer timeout (default 15 min)
    - Training phase (after first step): shorter timeout (default 5 min)
    """

    def __init__(
        self, timeout_minutes: float = 5.0, initial_timeout_minutes: float = 15.0
    ) -> None:
        super().__init__()
        self.timeout = timeout_minutes * 60
        self.initial_timeout = initial_timeout_minutes * 60
        self._last_heartbeat: float = 0
        self._stop = False
        self._first_step_done = False
        self._thread: Optional[threading.Thread] = None

    def _watchdog_loop(self) -> None:
        while not self._stop:
            time.sleep(30)  # check every 30 seconds
            if self._stop:
                break
            elapsed = time.time() - self._last_heartbeat
            current_timeout = self.timeout if self._first_step_done else self.initial_timeout
            if elapsed > current_timeout:
                phase = "training" if self._first_step_done else "initial"
                _log_event(
                    "watchdog_timeout",
                    phase=phase,
                    elapsed_seconds=elapsed,
                    timeout_seconds=current_timeout,
                )
                print(
                    f"WATCHDOG: No progress for {elapsed:.0f}s in {phase} phase "
                    f"(limit {current_timeout:.0f}s), aborting",
                    flush=True,
                )
                os._exit(1)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._last_heartbeat = time.time()
        self._stop = False
        self._first_step_done = False
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._thread.start()
        print(
            f"WATCHDOG: Started (initial={self.initial_timeout:.0f}s, training={self.timeout:.0f}s)",
            flush=True,
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not self._first_step_done:
            self._first_step_done = True
            print(f"WATCHDOG: First step done, switching to {self.timeout:.0f}s timeout", flush=True)
        self._last_heartbeat = time.time()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=5)


class DCPCheckpointCallback(TrainerCallback):
    """Pure DCP checkpointing: trigger async sharded saves every N steps.

    - Deduplicates multi-callback firing (on_step_end + on_log) at the same step.
    - Optional backpressure policy when a previous async save is in flight:
      * 'wait' (default): wait for the previous save to complete, then save now.
      * 'skip': skip this scheduled save to avoid long stalls.
    - Writes a minimal trainer_state.json on rank 0 for human/debug use.
    """

    def __init__(
        self,
        save_steps: int,
        async_enabled: bool,
        output_dir: str,
        keep_last: int,
        backpressure: str = "wait",
    ) -> None:
        super().__init__()
        self.save_steps = max(int(save_steps), 1)
        self.async_enabled = bool(async_enabled)
        self.output_dir = output_dir
        self.keep_last = max(int(keep_last), 0)
        self.backpressure = backpressure if backpressure in {"wait", "skip"} else "wait"
        self._future = None
        self._future_started_at: Optional[float] = None
        self._future_step: Optional[int] = None
        self._future_ckpt: Optional[str] = None
        self._future_logged: bool = False
        self._last_saved_step: int = -1

    def _wait_prev(self) -> None:
        if self._future is not None:
            try:
                # Block until previous async save completes; do not log here
                # to avoid duplicate with background watcher.
                self._future.result()
            finally:
                self._future = None
                self._future_started_at = None
                self._future_step = None
                self._future_ckpt = None
                self._future_logged = False

    def _spawn_completion_watcher(
        self, fut, step: int, ckpt_dir: str, t0: float
    ) -> None:
        try:
            import threading

            def _watch() -> None:
                try:
                    fut.result()
                    # Only rank0 prints
                    _log_event(
                        "dcp_async_save_completed",
                        step=int(step),
                        output_dir=ckpt_dir,
                        write_seconds=round(time.time() - t0, 3),
                    )
                except Exception as e:  # noqa: BLE001
                    _log_event(
                        "dcp_async_save_failed",
                        step=int(step),
                        output_dir=ckpt_dir,
                        error=str(e),
                    )
                finally:
                    self._future_logged = True

            th = threading.Thread(target=_watch, name=f"dcp-watch-{step}", daemon=True)
            th.start()
        except Exception:
            pass

    def _maybe_save(self, args: TrainingArguments, state: TrainerState) -> None:
        if state.global_step == 0 or (state.global_step % self.save_steps != 0):
            return
        # Deduplicate double firing at the same step (on_step_end + on_log)
        if self._last_saved_step == int(state.global_step):
            _log_event(
                "dcp_skip",
                reason="duplicate_step",
                step=int(state.global_step),
            )
            return
        tr = getattr(self, "trainer", None)
        if tr is None:
            _log_event("dcp_skip", reason="no_trainer", step=int(state.global_step))
            return

        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        try:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict

            # If a previous async save is still in flight, either wait or skip
            if self._future is not None:
                if self.async_enabled and self.backpressure == "skip":
                    _log_event(
                        "dcp_skip",
                        reason="previous_inflight",
                        step=int(state.global_step),
                    )
                    return
                # default/backpressure == 'wait'
                t0w = time.time()
                self._wait_prev()
                _log_event(
                    "dcp_wait_prev_completed",
                    step=int(state.global_step),
                    wait_seconds=round(time.time() - t0w, 3),
                )

            # Build DCP state dicts (FSDP-aware; sharded by default)
            t0 = time.time()
            model_sd, optim_sd = get_state_dict(tr.model, tr.optimizer)
            _log_event(
                "dcp_state_dict_built",
                step=int(state.global_step),
                build_seconds=round(time.time() - t0, 3),
            )
            sd = {"model": model_sd, "optim": optim_sd}

            if self.async_enabled:
                try:
                    t_write0 = time.time()
                    self._future = dcp.async_save(sd, checkpoint_id=ckpt_dir)
                    self._future_started_at = t_write0
                    self._future_step = int(state.global_step)
                    self._future_ckpt = ckpt_dir
                    self._future_logged = False
                    _log_event(
                        "dcp_async_save_started",
                        step=state.global_step,
                        output_dir=ckpt_dir,
                    )
                    # start background watcher to log completion time
                    self._spawn_completion_watcher(
                        self._future, int(state.global_step), ckpt_dir, t_write0
                    )
                except Exception as e_async:
                    # If async path is unsupported (e.g., no CPU:gloo backend),
                    # fall back to synchronous DCP save for this and future saves.
                    _log_event(
                        "dcp_async_unavailable",
                        error=str(e_async),
                        step=state.global_step,
                    )
                    self.async_enabled = False
                    t_write0 = time.time()
                    method = _dcp_save_sync(sd, ckpt_dir)
                    _log_event(
                        "dcp_save_completed",
                        step=state.global_step,
                        output_dir=ckpt_dir,
                        method=method,
                        write_seconds=round(time.time() - t_write0, 3),
                    )
            else:
                t_write0 = time.time()
                method = _dcp_save_sync(sd, ckpt_dir)
                _log_event(
                    "dcp_save_completed",
                    step=state.global_step,
                    output_dir=ckpt_dir,
                    method=method,
                    write_seconds=round(time.time() - t_write0, 3),
                )

            # Mark that we have scheduled/saved a checkpoint for this step
            self._last_saved_step = int(state.global_step)

            # rank0 writes minimal trainer_state
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                # prune old checkpoints, keep latest N including current
                try:
                    if self.keep_last > 0:
                        from pathlib import Path
                        import shutil

                        base = Path(self.output_dir)
                        # list all checkpoint-* dirs excluding current
                        items = [
                            p
                            for p in base.glob("checkpoint-*")
                            if p.is_dir() and str(p) != ckpt_dir
                        ]

                        def _step_num(p: Path) -> int:
                            try:
                                return int(p.name.split("-")[-1])
                            except Exception:
                                return -1

                        items.sort(key=_step_num, reverse=True)
                        # keep_last includes current; here we keep (keep_last-1) of the rest
                        to_delete = (
                            items[self.keep_last - 1 :] if self.keep_last > 0 else items
                        )
                        for old in to_delete:
                            try:
                                shutil.rmtree(old, ignore_errors=True)
                            except Exception:
                                pass
                except Exception:
                    pass
                meta = {
                    "global_step": int(state.global_step),
                    "timestamp": time.time(),
                }
                with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
                    json.dump(meta, f)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] DCP checkpoint failed at step {state.global_step}: {e}")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._maybe_save(args, state)

    # Fallback for older Trainer versions or if on_step_end is not dispatched
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self._maybe_save(args, state)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # Ensure last async save completed
        self._wait_prev()


def _validate_dcp_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Validate DCP checkpoint integrity before attempting distributed load.

    Returns None if valid, or an error message string if invalid.
    This prevents distributed collective operations from crashing when
    checkpoint files are missing.

    Note: DCP .metadata is NOT standard pickle format, so we just check
    file existence and .distcp shard count rather than parsing metadata.
    """
    metadata_path = os.path.join(ckpt_dir, ".metadata")

    # Check .metadata exists and has content
    if not os.path.exists(metadata_path):
        return "missing .metadata file"
    if os.path.getsize(metadata_path) == 0:
        return ".metadata file is empty"

    # Check .distcp shard files exist (expect world_size shards)
    distcp_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".distcp")]
    if len(distcp_files) == 0:
        return "no .distcp shard files found"
    # Get expected shard count from world size
    import torch.distributed as dist
    expected_shards = dist.get_world_size() if dist.is_initialized() else 1
    if len(distcp_files) < expected_shards:
        return f"incomplete checkpoint: only {len(distcp_files)}/{expected_shards} .distcp shards"

    return None


class DCPResumeCallback(TrainerCallback):
    """Apply a DCP checkpoint into (model, optimizer) at train start.

    This must run after FSDP wrapping and optimizer construction, so we hook
    it in via Trainer callbacks rather than doing resume in main() before
    `trainer.train()`.

    If multiple checkpoint directories are provided, tries them in order
    (newest first). If all fail, raises an error to abort training.
    """

    def __init__(self, ckpt_dirs: List[str], resume_optim: bool) -> None:
        super().__init__()
        # Accept both single path (legacy) and list of paths
        if isinstance(ckpt_dirs, str):
            ckpt_dirs = [ckpt_dirs]
        self.ckpt_dirs = ckpt_dirs
        self.resume_optim = bool(resume_optim)
        self._done = False

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self._done:
            return
        trainer = getattr(self, "trainer", None)
        model = getattr(trainer, "model", None) if trainer is not None else None
        optimizer = getattr(trainer, "optimizer", None) if trainer is not None else None
        if not self.resume_optim:
            # Only restore model weights; skip optimizer state to reduce peak memory.
            optimizer = None
        if model is None:
            raise RuntimeError(
                "trainer.model is None in DCPResumeCallback; cannot resume"
            )

        # Try each checkpoint in order (newest first), stop on first success
        errors: List[str] = []
        for ckpt_dir in self.ckpt_dirs:
            # Pre-validate checkpoint BEFORE calling distributed DCP load.
            # DCP load is a collective operation - if one rank fails mid-load,
            # the entire process group crashes via NCCL/torchrun before Python's
            # exception handler can catch it. So we validate first.
            validation_error = _validate_dcp_checkpoint(ckpt_dir)
            if validation_error:
                err_msg = f"{ckpt_dir}: {validation_error}"
                errors.append(err_msg)
                _log_event(
                    "resume_checkpoint_skipped",
                    checkpoint=ckpt_dir,
                    reason=validation_error,
                )
                continue  # Skip to next checkpoint

            try:
                method = _dcp_load_sync(ckpt_dir, model, optimizer)

                # Restore global_step from trainer_state.json
                trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
                restored_step = 0
                if os.path.exists(trainer_state_path):
                    with open(trainer_state_path) as f:
                        saved_state = json.load(f)
                    restored_step = saved_state.get("global_step", 0)
                    if trainer is not None and hasattr(trainer, "state"):
                        trainer.state.global_step = restored_step

                # Step lr_scheduler to match restored step
                # Scheduler is created before on_train_begin, so it starts at step 0
                lr_scheduler = getattr(trainer, "lr_scheduler", None)
                if lr_scheduler is not None and restored_step > 0:
                    for _ in range(restored_step):
                        lr_scheduler.step()
                    _log_event(
                        "lr_scheduler_stepped",
                        to_step=restored_step,
                    )

                _log_event(
                    "resume_from_checkpoint",
                    checkpoint=ckpt_dir,
                    method=method,
                    restored_step=restored_step,
                )
                self._done = True
                return  # Success!
            except Exception as e:  # noqa: BLE001
                err_msg = f"{ckpt_dir}: {e}"
                errors.append(err_msg)
                _log_event(
                    "resume_checkpoint_failed",
                    checkpoint=ckpt_dir,
                    error=str(e),
                )
                # Continue to next checkpoint

        # All checkpoints failed - abort training
        self._done = True
        error_summary = "; ".join(errors)
        _log_event(
            "resume_all_failed",
            checkpoints=self.ckpt_dirs,
            errors=errors,
        )
        raise RuntimeError(
            f"All {len(self.ckpt_dirs)} checkpoint(s) failed to load: {error_summary}"
        )


try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print(
        "Warning: wandb not available. Install with 'pip install wandb' for experiment tracking."
    )


def _bool(x: Optional[str]) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


@dataclass
class Args:
    model_id: str
    dataset_id: str
    output_dir: str
    test_size: float
    seed: int
    epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    grad_accum_steps: int
    learning_rate: float
    gradient_checkpointing: bool
    use_fsdp: bool
    fsdp: str
    fsdp_offload: bool
    bf16: bool
    fp16: bool
    tf32: bool
    use_flash_attn2: bool
    max_train_samples: Optional[int]
    max_eval_samples: Optional[int]
    max_seq_length: int
    optim: str
    # checkpointing
    save_steps: int
    save_total_limit: int
    resume_from_checkpoint: bool
    resume_path: Optional[str]
    async_checkpoint: bool
    checkpoint_backpressure: str
    dcp_resume_optim: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="Multi-GPU FSDP SFT for Qwen3 (14B/30B/MoE)"
    )
    p.add_argument("--model-id", default="Qwen/Qwen3-14B")
    p.add_argument("--dataset-id", default="microsoft/orca-math-word-problems-200k")
    p.add_argument("--output-dir", default="./outputs/qwen3_32b_sft")
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--gradient-checkpointing", type=_bool, default=True)
    p.add_argument("--use-fsdp", type=_bool, default=True)
    p.add_argument("--fsdp", default="full_shard auto_wrap")
    p.add_argument("--fsdp-offload", type=_bool, default=False)
    p.add_argument("--bf16", type=_bool, default=True)
    p.add_argument("--fp16", type=_bool, default=False)
    p.add_argument("--tf32", type=_bool, default=True)  # Requires Ampere+ GPUs
    p.add_argument("--use-flash-attn2", type=_bool, default=False)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="Optimizer for Trainer (e.g., adamw_torch, adamw_bnb_8bit)",
    )
    p.add_argument(
        "--async-checkpoint",
        type=_bool,
        default=False,
        help="Use PyTorch Distributed Checkpoint async_save for sharded checkpoints.",
    )
    p.add_argument(
        "--checkpoint-backpressure",
        type=str,
        choices=["wait", "skip"],
        default="wait",
        help=(
            "When a previous async checkpoint is still running: 'wait' to block at the next save step, "
            "or 'skip' to skip this scheduled save to avoid long stalls."
        ),
    )

    p.add_argument("--max-train-samples", type=int, default=32000)
    p.add_argument("--max-eval-samples", type=int, default=None)
    # checkpointing
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument(
        "--resume-from-checkpoint",
        type=_bool,
        default=True,
        help="Resume automatically from the latest checkpoint under output_dir if available.",
    )
    p.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Explicit checkpoint path to resume from (overrides --resume-from-checkpoint if set).",
    )
    p.add_argument(
        "--dcp-resume-optim",
        type=_bool,
        default=True,
        help=(
            "If true, DCP resume restores both model and optimizer state. "
            "If false, only model weights are restored to reduce peak memory."
        ),
    )

    a = p.parse_args()
    return Args(
        model_id=a.model_id,
        dataset_id=a.dataset_id,
        output_dir=a.output_dir,
        test_size=a.test_size,
        seed=a.seed,
        epochs=a.epochs,
        per_device_train_batch_size=a.per_device_train_batch_size,
        per_device_eval_batch_size=a.per_device_eval_batch_size,
        grad_accum_steps=a.grad_accum_steps,
        learning_rate=a.learning_rate,
        gradient_checkpointing=bool(a.gradient_checkpointing),
        use_fsdp=bool(a.use_fsdp),
        fsdp=a.fsdp,
        fsdp_offload=bool(a.fsdp_offload),
        bf16=bool(a.bf16),
        fp16=bool(a.fp16),
        tf32=bool(a.tf32),
        use_flash_attn2=bool(a.use_flash_attn2),
        max_seq_length=a.max_seq_length,
        optim=a.optim,
        max_train_samples=a.max_train_samples,
        max_eval_samples=a.max_eval_samples,
        save_steps=a.save_steps,
        save_total_limit=a.save_total_limit,
        resume_from_checkpoint=bool(a.resume_from_checkpoint),
        resume_path=a.resume_path,
        async_checkpoint=bool(a.async_checkpoint),
        checkpoint_backpressure=str(a.checkpoint_backpressure),
        dcp_resume_optim=bool(a.dcp_resume_optim),
    )


def normalize_conversations(row: Dict[str, Any]) -> List[Dict[str, str]]:
    # Handle conversations/messages format (original)
    conv = row.get("conversations") or row.get("messages")
    if isinstance(conv, list):
        out: List[Dict[str, str]] = []
        for msg in conv:
            role = msg.get("role")
            if not role and "from" in msg:
                role = (
                    "user"
                    if msg["from"] == "human"
                    else ("assistant" if msg["from"] == "gpt" else "system")
                )
            content = msg.get("content") or msg.get("value") or ""
            out.append({"role": role or "user", "content": content})
        return out

    # Handle question/answer format (orca-math dataset)
    question = row.get("question")
    answer = row.get("answer")
    if question is not None and answer is not None:
        return [
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": str(answer)},
        ]

    # Fallback: return empty list if no recognized format
    return []


def build_text_column(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def to_text(example: Dict[str, Any]) -> Dict[str, str]:
        convs = normalize_conversations(example)
        try:
            text = tokenizer.apply_chat_template(convs, tokenize=False)
        except Exception:
            text = "\n".join(f"[{m['role']}] {m['content']}" for m in convs)
        return {"text": text}

    with_text = dataset.map(to_text)
    return with_text.remove_columns(
        [c for c in with_text.column_names if c != "text"]
    )  # keep only text


def main() -> None:
    args = parse_args()

    # Initialize device context (torch.distributed initialization is handled by accelerate)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"Local rank {local_rank} / world size {world_size} configured")

    assert torch.cuda.is_available(), "CUDA GPU is required (H100 recommended)."

    # Initialize default process group with combined backends (PyTorch 2.8+):
    # 'cpu:gloo,cuda:nccl' to satisfy DCP async_save CPU backend requirement.
    # If this fails (older torch), we fall back to NCCL-only and our DCP path
    # will automatically downgrade to sync saves.
    try:
        import torch.distributed as dist

        if dist.is_available() and not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend="cpu:gloo,cuda:nccl", init_method="env://"
                )
                _log_event("dist_init", backend="cpu:gloo,cuda:nccl")
            except Exception as e:
                _log_event("dist_init_fallback", error=str(e))
                dist.init_process_group(backend="nccl", init_method="env://")
                _log_event("dist_init", backend="nccl")
        elif dist.is_initialized():
            _log_event(
                "dist_already_initialized",
                backend=str(getattr(dist, "get_backend", lambda: "unknown")()),
            )
    except Exception as e:
        _log_event("dist_init_error", error=str(e))

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    if os.getenv("HF_TOKEN"):
        os.environ.setdefault(
            "HUGGINGFACE_HUB_TOKEN", os.environ["HF_TOKEN"]
        )  # for older libs

    set_seed(args.seed)
    _log_event("job_config", hyperparameters=vars(args), world_size=world_size)

    # Initialize wandb if available (only on rank 0)
    if WANDB_AVAILABLE and local_rank == 0:
        import wandb

        wandb.init(
            project="qwen3-finetune",
            tags=["h100", "full-finetune", args.model_id],
            config={
                "model_id": args.model_id,
                "dataset_id": args.dataset_id,
                "epochs": args.epochs,
                "batch_size": args.per_device_train_batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "learning_rate": args.learning_rate,
                "max_train_samples": args.max_train_samples,
                "bf16": args.bf16,
                "flash_attention": args.use_flash_attn2,
                "gradient_checkpointing": args.gradient_checkpointing,
            },
        )
    elif local_rank == 0:
        print("W&B not available - training will proceed without experiment tracking")

    # Load tokenizer / ensure pad token
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare dataset
    raw = load_dataset(args.dataset_id)
    train_raw = raw["train"]
    # Build 'text' column via chat template
    train_text = build_text_column(train_raw, tokenizer)
    eval_text = None
    if "validation" in raw:
        eval_text = build_text_column(raw["validation"], tokenizer)
    elif args.test_size and args.test_size > 0:
        # Simple split
        split = train_text.train_test_split(test_size=args.test_size, seed=args.seed)
        train_text, eval_text = split["train"], split["test"]

    if args.max_train_samples:
        train_text = train_text.select(
            range(min(args.max_train_samples, len(train_text)))
        )
    if eval_text is not None and args.max_eval_samples:
        eval_text = eval_text.select(range(min(args.max_eval_samples, len(eval_text))))

    # Tokenize
    def tok_fn(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            max_length=args.max_seq_length,
        )  # returns input_ids/attention_mask

    lm_train = train_text.map(tok_fn, remove_columns=list(train_text.features))
    lm_eval = (
        eval_text.map(tok_fn, remove_columns=list(eval_text.features))
        if eval_text is not None
        else None
    )

    # Setup mixed precision for FSDP
    torch_dtype = (
        torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    fsdp_mode: Optional[str] = None
    fsdp_config: Optional[Dict[str, Any]] = None
    if args.use_fsdp:
        # Detect the correct decoder layer class name for FSDP auto-wrap.
        # Accelerate's FSDP plugin requires the exact transformer block class name.
        # For Qwen/Qwen3-30B-A3B (MoE), the class is 'Qwen3MoeDecoderLayer'.
        # For dense Qwen3, it's 'Qwen3DecoderLayer'.
        from transformers import AutoConfig

        try:
            cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
            model_type = getattr(cfg, "model_type", "") or ""
        except Exception as _e:  # noqa: N806,BLE001
            model_type = ""

        if "qwen3_moe" in model_type or "moe" in args.model_id.lower():
            decoder_layer_cls_name = "Qwen3MoeDecoderLayer"
        else:
            decoder_layer_cls_name = "Qwen3DecoderLayer"

        base_mode = args.fsdp.strip()
        fsdp_tokens = base_mode.split()
        if args.fsdp_offload and "offload" not in fsdp_tokens:
            fsdp_tokens.append("offload")
        fsdp_mode = " ".join(fsdp_tokens)
        fsdp_config = {
            # Important: Must exactly match the layer class used by the model
            # or Accelerate will raise: "Could not find the transformer layer class ... in the model."
            "transformer_layer_cls_to_wrap": [decoder_layer_cls_name],
            # Use POST to lower peak memory (prefetch happens later).
            "backward_prefetch": "BACKWARD_POST",
            "forward_prefetch": False,
            "sync_module_states": True,
            "use_orig_params": False,
            "limit_all_gathers": True,
            # Extra memory reductions
            "activation_checkpointing": True,
        }
        if args.bf16:
            fsdp_config["mixed_precision"] = "bf16"
        elif args.fp16:
            fsdp_config["mixed_precision"] = "fp16"
        else:
            fsdp_config["mixed_precision"] = "fp32"
        if local_rank == 0:
            print(f"Resolved FSDP mode: {fsdp_mode}")
            print(f"Resolved FSDP config: {fsdp_config}")
            print(f"FSDP wrap class: {decoder_layer_cls_name}")

        # Guard: bitsandbytes 8-bit optimizers are not compatible with
        # FSDP+CPU offload; they expect all tensors/states on the same GPU.
        # If user asked for bnb while offload is enabled, auto-switch to adamw_torch.
        if (
            ("offload" in fsdp_tokens or args.fsdp_offload)
            and isinstance(args.optim, str)
            and "bnb" in args.optim.lower()
        ):
            if local_rank == 0:
                print(
                    "[WARN] bitsandbytes optimizers are incompatible with FSDP CPU offload; "
                    "switching optim to 'adamw_torch' to avoid device mismatch errors."
                )
            args.optim = "adamw_torch"

    # Load model
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "use_cache": not args.gradient_checkpointing,
    }
    if args.use_flash_attn2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if args.use_fsdp and local_rank == 0:
        print(f"Loading model {args.model_id} with FSDP...")

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    # Enable gradient checkpointing before FSDP wrapping
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Training arguments
    tr_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        logging_strategy="steps",
        logging_steps=10,
        log_on_each_node=False,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        optim=args.optim,
        warmup_ratio=0.05,
        weight_decay=0.01,
        # FSDP configuration (handled by Trainer/Accelerate)
        fsdp=fsdp_mode,
        fsdp_config=fsdp_config,
        # Disable Trainer's own checkpointing; we use pure DCP callback for checkpoints
        save_strategy="no",
        save_safetensors=True,
        # Distributed training
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        # Report to wandb if available
        report_to="wandb" if WANDB_AVAILABLE else "none",
    )

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=lm_train,
        eval_dataset=lm_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[
            ProgressLoggerCallback(),
            WatchdogCallback(timeout_minutes=5.0),
            DCPCheckpointCallback(
                save_steps=args.save_steps,
                async_enabled=args.async_checkpoint,
                output_dir=args.output_dir,
                keep_last=args.save_total_limit,
                backpressure=args.checkpoint_backpressure,
            ),
        ],
    )
    # Ensure our DCP callbacks have access to the trainer instance (older HF
    # versions may not set callback.trainer automatically).
    try:
        for _cb in getattr(trainer.callback_handler, "callbacks", []):
            if isinstance(_cb, (DCPCheckpointCallback, DCPResumeCallback)):
                setattr(_cb, "trainer", trainer)
    except Exception:
        pass

    # Log planned training steps and dataset cardinality for external monitors
    try:
        num_train_samples = len(lm_train)
    except Exception:
        num_train_samples = None
    total_train_bsz = (
        args.per_device_train_batch_size
        * max(world_size, 1)
        * max(args.grad_accum_steps, 1)
    )
    try:
        # Fallback estimate before Trainer computes internal max_steps
        import math

        steps_per_epoch = (
            math.ceil(num_train_samples / total_train_bsz)
            if num_train_samples
            else None
        )
        planned_max_steps = (
            int(args.epochs * steps_per_epoch) if steps_per_epoch is not None else None
        )
    except Exception:
        planned_max_steps = None
    _log_event(
        "train_plan",
        planned_max_steps=planned_max_steps,
        per_device_bsz=args.per_device_train_batch_size,
        grad_accum=args.grad_accum_steps,
        world_size=world_size,
        num_train_samples=num_train_samples,
        epochs=args.epochs,
    )

    # Print trainable params
    try:
        trainer.model.print_trainable_parameters()
    except Exception:
        pass

    # Trainer delays optimizer construction until .train(); build it early so
    # resume logic hands a real optimizer to DCP load helpers.
    # try:
    #     trainer.create_optimizer()
    # except Exception as _e:  # noqa: N806,BLE001
    #     _log_event("optimizer_init_warning", error=str(_e))

    # Create output dir and decide resume target (ignore if none exists)
    os.makedirs(args.output_dir, exist_ok=True)
    from pathlib import Path

    # Pure DCP resume logic: collect all available checkpoints, sorted newest first
    ckpt_candidates: List[str] = []
    if args.resume_path:
        # Explicit path takes priority
        ckpt_candidates = [args.resume_path]
    elif args.resume_from_checkpoint:
        ckpts = [p for p in Path(args.output_dir).glob("checkpoint-*") if p.is_dir()]
        if ckpts:

            def _step(p: Path) -> int:
                try:
                    return int(p.name.split("-")[-1])
                except Exception:
                    return -1

            # Sort descending (newest first) so we try latest checkpoint first
            ckpts.sort(key=_step, reverse=True)
            ckpt_candidates = [str(p) for p in ckpts]

    if ckpt_candidates:
        _log_event(
            "resume_candidates",
            checkpoints=ckpt_candidates,
            count=len(ckpt_candidates),
        )
        # Defer actual DCP load into (model, optimizer) to a Trainer callback
        # that runs after FSDP wrapping and optimizer construction.
        # Pass all candidates; callback will try each until one succeeds.
        resume_cb = DCPResumeCallback(ckpt_candidates, resume_optim=args.dcp_resume_optim)
        trainer.add_callback(resume_cb)
        try:
            setattr(resume_cb, "trainer", trainer)
        except Exception:
            pass
    else:
        _log_event("fresh_start")

    # Train
    try:
        az = _get_availability_zone()
        _log_event("train_start", availability_zone=az)
        trainer.train(resume_from_checkpoint=None)
    except Exception as exc:  # noqa: BLE001
        _log_event("train_exception", error=str(exc))
        raise

    # Save full fine-tuned weights.
    # NOTE: Disabled for large FSDP runs (e.g., Qwen3-14B) because
    # unsharding the full state dict to a single rank can trigger CUDA OOM
    # at the very end of training. We rely on DCP/Accelerate sharded
    # checkpoints for resume and export instead.
    # trainer.model.save_pretrained(args.output_dir, safe_serialization=True)
    # _log_event("model_saved", output_dir=args.output_dir)

    # Final sharded checkpoint for resume (best practice with FSDP)
    try:
        trainer.accelerator.save_state(args.output_dir)
        _log_event("accelerate_save_state", step=None, output_dir=args.output_dir)
    except Exception as _e:  # noqa: N806,BLE001
        if local_rank == 0:
            print(f"[WARN] Final sharded checkpoint save failed: {_e}")

    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
