#!/usr/bin/env python3
"""Fake training workload that mimics real training checkpoint behavior.

Outputs JSON logs in the same format as train_qwen3_4b_qlora_single_gpu.py
and writes real-sized checkpoints to support migration testing.

Environment variables:
    TOTAL_STEPS: Total training steps (default: 100)
    STEP_SECONDS: Seconds per step (default: 1.0)
    SAVE_STEPS: Save checkpoint every N steps (default: 10)
    SAVE_TOTAL_LIMIT: Keep only last N checkpoints (default: 3)
    OUTPUT_DIR: Checkpoint output directory (default: /tmp/checkpoints)
    CHECKPOINT_SIZE_MB: Size of each checkpoint in MB (default: 100)
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def log_event(event: str, **payload: Any) -> None:
    """Log event in the same JSON format as real training."""
    body = {"ts": time.time(), "event": event, **payload}
    print(json.dumps(body, default=str), flush=True)


def create_checkpoint(output_dir: str, step: int, checkpoint_size_mb: float) -> str:
    """Create a checkpoint directory with realistic size.

    Mimics DCP checkpoint structure:
    - checkpoint-{step}/
      - trainer_state.json (metadata)
      - model_shard_0.pt (fake model data)
    """
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Write trainer_state.json (matches real training format)
    trainer_state = {
        "global_step": step,
        "timestamp": time.time(),
    }
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
        json.dump(trainer_state, f)

    # Write fake model data to simulate checkpoint size
    # Use random-ish data to avoid compression
    model_file = os.path.join(ckpt_dir, "model_shard_0.pt")
    chunk_size = 1024 * 1024  # 1MB chunks
    total_bytes = int(checkpoint_size_mb * 1024 * 1024)

    with open(model_file, "wb") as f:
        written = 0
        while written < total_bytes:
            # Write pseudo-random data (based on position to be reproducible)
            chunk = bytes((i % 256) for i in range(min(chunk_size, total_bytes - written)))
            f.write(chunk)
            written += len(chunk)

    return ckpt_dir


def prune_old_checkpoints(output_dir: str, keep_last: int) -> None:
    """Keep only the last N checkpoints, delete older ones."""
    if keep_last <= 0:
        return

    base = Path(output_dir)
    ckpts = [p for p in base.glob("checkpoint-*") if p.is_dir()]

    def get_step(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except Exception:
            return -1

    ckpts.sort(key=get_step, reverse=True)

    # Delete checkpoints beyond keep_last
    to_delete = ckpts[keep_last:]
    for old in to_delete:
        try:
            shutil.rmtree(old, ignore_errors=True)
            log_event("checkpoint_pruned", path=str(old))
        except Exception as e:
            print(f"[WARN] Failed to prune {old}: {e}", flush=True)


def find_latest_checkpoint(output_dir: str) -> Optional[Dict[str, Any]]:
    """Find the latest checkpoint to resume from.

    Returns dict with 'path' and 'step' if found, None otherwise.
    """
    base = Path(output_dir)
    if not base.exists():
        return None

    ckpts = [p for p in base.glob("checkpoint-*") if p.is_dir()]
    if not ckpts:
        return None

    def get_step(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except Exception:
            return -1

    # Sort by step descending
    ckpts.sort(key=get_step, reverse=True)

    # Try each checkpoint until we find a valid one
    for ckpt in ckpts:
        trainer_state_path = ckpt / "trainer_state.json"
        if trainer_state_path.exists():
            try:
                with open(trainer_state_path) as f:
                    state = json.load(f)
                return {
                    "path": str(ckpt),
                    "step": state.get("global_step", get_step(ckpt)),
                }
            except Exception as e:
                log_event("checkpoint_invalid", path=str(ckpt), error=str(e))
                continue

    return None


def main():
    # Parse configuration from environment
    total_steps = int(os.environ.get('TOTAL_STEPS', 100))
    step_seconds = float(os.environ.get('STEP_SECONDS', 1.0))
    save_steps = int(os.environ.get('SAVE_STEPS', 10))
    save_total_limit = int(os.environ.get('SAVE_TOTAL_LIMIT', 3))
    output_dir = os.environ.get('OUTPUT_DIR', '/tmp/checkpoints')
    checkpoint_size_mb = float(os.environ.get('CHECKPOINT_SIZE_MB', 100))

    os.makedirs(output_dir, exist_ok=True)

    print(f"[fake_train] Configuration:", flush=True)
    print(f"  total_steps: {total_steps}", flush=True)
    print(f"  step_seconds: {step_seconds}", flush=True)
    print(f"  save_steps: {save_steps}", flush=True)
    print(f"  save_total_limit: {save_total_limit}", flush=True)
    print(f"  output_dir: {output_dir}", flush=True)
    print(f"  checkpoint_size_mb: {checkpoint_size_mb}", flush=True)

    # Check for existing checkpoint to resume from
    start_step = 0
    resume_info = find_latest_checkpoint(output_dir)

    if resume_info:
        start_step = resume_info["step"]
        log_event("resume_from_checkpoint",
                  checkpoint=resume_info["path"],
                  restored_step=start_step)
        print(f"[fake_train] Resuming from step {start_step}", flush=True)
    else:
        log_event("fresh_start")
        print("[fake_train] Starting fresh (no checkpoint found)", flush=True)

    # Signal training start (test.py looks for this)
    log_event("train_begin", step=start_step)

    # Training loop
    for step in range(start_step + 1, total_steps + 1):
        time.sleep(step_seconds)

        # Log progress periodically
        if step % 10 == 0 or step == total_steps:
            log_event("progress",
                      step=step,
                      total=total_steps,
                      pct=round(step / total_steps * 100, 1))

        # Save checkpoint at intervals
        if step % save_steps == 0:
            t0 = time.time()
            ckpt_dir = create_checkpoint(output_dir, step, checkpoint_size_mb)
            write_seconds = round(time.time() - t0, 3)

            # This is the key event test.py detects for progress
            log_event("dcp_async_save_completed",
                      step=step,
                      output_dir=ckpt_dir,
                      write_seconds=write_seconds)

            # Prune old checkpoints
            prune_old_checkpoints(output_dir, save_total_limit)

    log_event("train_end", step=total_steps)
    print(f"[fake_train] Training complete: {total_steps} steps", flush=True)


if __name__ == "__main__":
    main()
