"""SkyPilot-based parallel execution module for benchmarks using Python API."""

import os
import uuid  # noqa: F401
import getpass  # noqa: F401
import time
import json
import logging
import io
import sys
import re
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import sky

logger = logging.getLogger(__name__)
GLOBAL_ACTIVE_CLUSTERS: set[str] = set()

DISK_TIER = 'low' # HDD


def get_active_clusters() -> List[str]:
    """Return a snapshot of clusters currently launched by the executor."""
    return sorted(GLOBAL_ACTIVE_CLUSTERS)



class _ProgressStream(io.TextIOBase):
    """Stream wrapper that parses remote logs for progress markers."""

    def __init__(self, cluster_name: str, job_id: int):
        super().__init__()
        self._cluster_name = cluster_name
        self._job_id = job_id
        self._buffer = ""
        self._last_reported = 0
        self._threshold = None

    def write(self, s: str) -> int:
        sys.stdout.write(s)
        sys.stdout.flush()
        self._buffer += s
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            self._handle_line(line.strip())
        return len(s)

    def flush(self) -> None:
        sys.stdout.flush()

    def _handle_line(self, line: str) -> None:
        if not line:
            return
        if "✅ [" not in line:
            return
        fragment = line.split("✅ ")[-1]
        if "]" not in fragment or "/" not in fragment:
            return
        bracket = fragment.split("]")[0].strip("[]")
        try:
            done_str, total_str = bracket.split("/")
            done = int(done_str)
            total = int(total_str)
        except ValueError:
            return
        if total <= 0:
            return
        if self._threshold is None:
            self._threshold = max(total // 100, 1)
        if done >= total or done - self._last_reported >= self._threshold:
            self._last_reported = done
            pct = done / total * 100.0
            logger.info(f"[{self._cluster_name} job {self._job_id}] progress: {done}/{total} ({pct:.1f}%)")


def check_skypilot_api_compatibility():
    """Check SkyPilot API compatibility before launching tasks."""
    try:
        sky.status()
        return True
    except Exception as e:
        if "version mismatch" in str(e).lower():
            logger.error("🚨 SkyPilot API version mismatch detected!")
            logger.error(str(e))
            logger.info("💡 Fix: sky api stop; sky api start")
            return False
        logger.warning(f"⚠️  SkyPilot API issue (proceeding anyway): {e}")
        return True


class SkyPilotExecutor:
    """Manages parallel execution of simulation tasks using SkyPilot Python API."""
    
    def __init__(self, max_parallel_clusters: int = 10, cloud: Optional[str] = 'gcp', instance_type: Optional[str] = None, auto_down: bool = True, *, cpus: str = '16+', memory: str = '32+', tasks_per_cluster: int = 50):
        self.max_parallel_clusters = max_parallel_clusters
        self.cloud = cloud  # None means auto-select
        self.instance_type = instance_type  # None means auto-select
        self.auto_down = auto_down  # Whether to auto-terminate clusters
        self.active_clusters = []
        self.temp_dirs = []  # Track temp directories for cleanup
        self.cpus = cpus
        self.memory = memory
        self.tasks_per_cluster = tasks_per_cluster
        # Track clusters that should be kept alive due to failures
        self._clusters_keep_alive: set[str] = set()
        # Track run_id and batch index for each cluster for easy diagnostics
        self._cluster_run_map: dict[str, str] = {}
        self._cluster_batch_map: dict[str, int] = {}
        self._cluster_workdirs: dict[str, str] = {}
        # Configure results bucket for storage mounts and downloads
        self._results_bucket_name, self._results_storage = self._configure_results_storage()
        # Lazily populated data mount details (resolved per-execution)
        self._data_source_path: Optional[Path] = None
        self._remote_data_mount: Optional[str] = None

    def _configure_results_storage(self):
        """Always mount the external bucket gs://skypilot-benchmark-results.

        Rationale: user already has a Sky-managed bucket with this name in another
        context. Using `source=gs://...` avoids Sky trying to re-create it.
        """
        bucket_uri = 'gs://skypilot-benchmark-results'
        bucket_name = 'skypilot-benchmark-results'
        storage = sky.Storage(
            source=bucket_uri,
            mode=sky.StorageMode.MOUNT
        )
        return bucket_name, storage

    def _ensure_data_mount_paths(self, params: Dict) -> None:
        """Resolve local data source path and matching remote mount."""
        if self._data_source_path is not None and self._remote_data_mount is not None:
            return

        project_root = Path(__file__).resolve().parents[2]
        data_path = params.get("DATA_PATH")
        if not data_path:
            raise ValueError("DATA_PATH must be specified in params before launching SkyPilot tasks.")

        candidate = Path(data_path)
        if not candidate.is_absolute():
            candidate = project_root / candidate

        if not candidate.exists():
            raise FileNotFoundError(f"Trace data directory not found: {candidate}")

        self._data_source_path = candidate
        self._remote_data_mount = f"/tmp/data/{candidate.name}"

        if candidate.name == 'converted_multi_region_aligned':
            logger.warning(
                "Using legacy trace dataset 'converted_multi_region_aligned'; "
                "consider switching to 'converted_multi_region_aligned_h100_16_merged' for longer traces."
            )
        
    def create_batch_workdir(self, tasks: List[Dict], params: Dict, batch_id: str) -> str:
        """Create a working directory with necessary task and parameter files."""
        
        # Create temporary directory
        workdir = tempfile.mkdtemp(prefix=f"skypilot_batch_{batch_id}_")
        self.temp_dirs.append(workdir)
        
        # Add task indices to preserve order
        for i, task in enumerate(tasks):
            task['_task_index'] = task.get('_task_index', i)  # Preserve global index if exists
        
        # Save tasks file
        tasks_file = os.path.join(workdir, "tasks.json")
        with open(tasks_file, 'w') as f:
            json.dump(tasks, f)
        
        # Ensure data mount is resolved before writing params file
        self._ensure_data_mount_paths(params)

        # Save params file (override paths for remote execution)
        params_file = os.path.join(workdir, "params.json")
        params_to_write = dict(params)
        # Ensure DATA_PATH points to the mounted data directory on remote
        params_to_write["DATA_PATH"] = self._remote_data_mount
        # Keep remote behavior consistent with local: remove TASK_DURATION_HOURS to avoid mismatched recomputation
        params_to_write.pop("TASK_DURATION_HOURS", None)
        # Keep OUTPUT_DIR default; batch worker passes its own --output-dir for results
        with open(params_file, 'w') as f:
            json.dump(params_to_write, f)
        
        return workdir
    
    def execute_batch(self, tasks: List[Dict], cache_dir: Path, params: Dict) -> List[Dict]:
        """Execute a batch of tasks in parallel using SkyPilot Python API."""
        
        # Pre-check: Verify SkyPilot API compatibility
        if not check_skypilot_api_compatibility():
            logger.error("❌ SkyPilot API compatibility check failed. Please fix the version mismatch and try again.")
            # Return failed results for all tasks
            return [{**task, "cost": float('nan'), "error": "SkyPilot API version mismatch"} for task in tasks]
        
        # Add global indices to all tasks to preserve order
        for i, task in enumerate(tasks):
            task['_global_index'] = i
            
        results = []
        
        # Calculate optimal batch size based on tasks per cluster
        tasks_per_cluster = self.tasks_per_cluster
        num_clusters = min(
            self.max_parallel_clusters,
            (len(tasks) + tasks_per_cluster - 1) // tasks_per_cluster
        )
        
        # Distribute tasks across clusters
        task_batches = []
        for i in range(num_clusters):
            start_idx = i * len(tasks) // num_clusters
            end_idx = (i + 1) * len(tasks) // num_clusters
            if start_idx < end_idx:
                task_batches.append(tasks[start_idx:end_idx])
        
        logger.info(f"Distributing {len(tasks)} tasks across {len(task_batches)} clusters")
        
        # Phase 1: Launch all clusters in parallel (non-blocking)
        logger.info("📤 Phase 1: Launching all clusters in parallel...")
        cluster_info = []  # Store (batch_idx, batch_tasks, cluster_name, request_id, run_id)
        
        with ThreadPoolExecutor(max_workers=min(len(task_batches), self.max_parallel_clusters)) as executor:
            launch_futures = []
            for batch_idx, batch_tasks in enumerate(task_batches):
                future = executor.submit(
                    self._launch_cluster,  # Only launch, don't wait
                    batch_idx,
                    batch_tasks,
                    cache_dir,
                    params
                )
                launch_futures.append((batch_idx, batch_tasks, future))
            
            # Collect launch results
            for batch_idx, batch_tasks, future in launch_futures:
                try:
                    cluster_name, request_id, run_id = future.result()
                    cluster_info.append((batch_idx, batch_tasks, cluster_name, request_id, run_id))
                    logger.info(f"✅ Launch submitted for {cluster_name}, request_id: {request_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to launch batch {batch_idx}: {e}")
                    results.extend([{**task, "cost": float('nan'), "error": str(e)} for task in batch_tasks])
        
        # Phase 2: Wait for all jobs to complete in parallel
        logger.info("⏳ Phase 2: Waiting for all jobs to complete...")
        if not cluster_info:
            logger.error("No clusters launched successfully; skipping wait phase.")
            # Mark all tasks in this batch as failed with a clear error.
            return [{**task, "cost": float('nan'), "error": "No clusters launched (see earlier errors)"} for task in tasks]

        wait_futures = []
        with ThreadPoolExecutor(max_workers=min(len(cluster_info), self.max_parallel_clusters)) as executor:
            for batch_idx, batch_tasks, cluster_name, request_id, run_id in cluster_info:
                future = executor.submit(
                    self._wait_and_download,  # Wait and download results
                    batch_idx,
                    batch_tasks,
                    cluster_name,
                    request_id,
                    run_id,
                    cache_dir,
                    params
                )
                wait_futures.append(future)
            
            # Collect results from all clusters
            for future in as_completed(wait_futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Cluster batch failed: {e}")
        
        # Sort results by global index to maintain order
        results.sort(key=lambda x: x.get('_global_index', 0))
        
        # Remove the temporary index field
        for result in results:
            result.pop('_global_index', None)
            result.pop('_task_index', None)
        
        # Cost values should now be scalar floats from the corrected batch_worker
        
        # Summarize failures with richer details
        def _result_failed(res: dict) -> bool:
            return pd.isna(res.get('cost', float('nan'))) or 'error' in res or 'error_type' in res

        failed_tasks = [r for r in results if _result_failed(r)]
        if failed_tasks:
            logger.error(f"🚨 {len(failed_tasks)}/{len(results)} tasks failed:")
            for task in failed_tasks[:10]:  # Show first 10 failures
                strategy = task.get('strategy', 'unknown')
                trace = task.get('trace_index', 'unknown')
                error_msg = task.get('error') or task.get('error_details') or task.get('error_type') or 'Unknown error'
                logger.error(f"   - Strategy: {strategy}, Trace: {trace}, Error: {error_msg}")
            if len(failed_tasks) > 10:
                logger.error(f"   ... and {len(failed_tasks) - 10} more failures")
        else:
            logger.info(f"✅ All {len(results)} tasks completed successfully")
        
        return results
    
    def _launch_cluster(self, batch_idx: int, tasks: List[Dict], 
                       cache_dir: Path, params: Dict) -> tuple[str, Any, str]:
        """Launch a cluster for a batch of tasks and return immediately."""
        cluster_name = f"benchmark-batch-{batch_idx}"
        workdir = None
        
        try:
            # Generate unique run_id for this execution
            run_id = str(int(time.time()))

            # Record mappings for summary later
            self._cluster_run_map[cluster_name] = run_id
            self._cluster_batch_map[cluster_name] = batch_idx
            # Create working directory with all files
            workdir = self.create_batch_workdir(tasks, params, str(batch_idx))
            
            # Use a single persistent bucket with unique paths (preconfigured in __init__)
            base_bucket_name = self._results_bucket_name
            
            # Create run script - no need for git clone since we'll sync the code
            output_mount_path = f"/tmp/results_{cluster_name}"
            # Add run_id to the output path for unique results
            results_subdir = f"{run_id}/batch_{batch_idx}"

            # Compose optional env assignments (only include if defined locally)
            _env_assignments: list[str] = []
            # Default LOG_LEVEL to WARNING to reduce log volume (download is slow)
            _log_level = os.environ.get('LOG_LEVEL', 'WARNING')
            _env_assignments.append(f"LOG_LEVEL={_log_level}")
            for _var in ('WORKERS', 'OS_RESERVED_GB', 'WORKER_PER_GB', 'MEM_UTIL'):
                _val = os.environ.get(_var)
                if _val is not None and str(_val).strip() != '':
                    _env_assignments.append(f"{_var}={_val}")
            env_prefix = ' '.join(_env_assignments)
            env_prefix_with_space = (env_prefix + ' ') if env_prefix else ''

            run_script = f"""#!/bin/bash
set -e
set -o pipefail

# Minimal venv with only runtime deps for the worker
cd /tmp
uv venv .venv
source .venv/bin/activate
# Install selected wheels directly instead of `uv sync` to keep startup fast
uv pip install -q numpy pandas tqdm filelock configargparse

# Create output directory (for legacy backup, though we'll write to S3 mount)
mkdir -p /tmp/results

# Set PYTHONPATH so that both `scripts_multi` and its subpackages
# can be imported as top-level modules (e.g., `benchmark_components`)
export PYTHONPATH=/tmp:/tmp/scripts_multi:$PYTHONPATH

# Disable wandb to avoid API key issues
export WANDB_MODE=offline
export WANDB_DISABLED=1

# Create subdirectory for this run and batch
mkdir -p {output_mount_path}/{results_subdir}

# Run batch worker in the created venv
cd /tmp
set +e
{env_prefix_with_space}python -u -m scripts_multi.benchmark_components.batch_worker \\
  --tasks-file /tmp/tasks.json \\
  --params-json /tmp/params.json \\
  --output-dir {output_mount_path}/{results_subdir} 2>&1 | tee /tmp/batch_worker.log
rc=$?
set -e
# Always copy worker log and exit code to mounted storage for debugging
cp /tmp/batch_worker.log {output_mount_path}/{results_subdir}/batch_worker.log || true
echo "$rc" > {output_mount_path}/{results_subdir}/exit_code.txt || true
  
# Debug: Check if results were written
echo "=== Checking results after batch processing ===" 2>&1 | tee -a /tmp/debug.log
ls -la {output_mount_path}/ 2>&1 | tee -a /tmp/debug.log
"""
            
            # Create SkyPilot task without workdir (will use file_mounts instead)
            task = sky.Task(
                name=f'benchmark-batch-{batch_idx}',
                run=run_script
            )
            
            # Set file mounts
            # Create results directory in workdir  
            results_dir = os.path.join(workdir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Set file mounts for task and param files
            project_root = Path(__file__).parent.parent.parent
            
            # Use the same persistent bucket for all runs (for caching)
            s3_bucket_name = base_bucket_name
            
            file_mounts: dict[str, Any] = {
                '/tmp/tasks.json': os.path.join(workdir, 'tasks.json'),
                '/tmp/params.json': os.path.join(workdir, 'params.json'),
                # Mount specific directories needed for the task
                '/tmp/scripts_multi': str(project_root / 'scripts_multi'),
                '/tmp/sky_spot': str(project_root / 'sky_spot'),
                '/tmp/pyproject.toml': str(project_root / 'pyproject.toml'),
                '/tmp/uv.lock': str(project_root / 'uv.lock'),
                '/tmp/main.py': str(project_root / 'main.py')
            }
            
            # Mount the entire data directory for multi-region tasks
            # NOTE: Mounting directly from inside the Git repo triggers a git ls-files
            # internal error on some setups (exit 128: "directory entry not superset of prefix").
            # To avoid SkyPilot using git-based sync on this path, stage a copy into the
            # temporary workdir (which is outside the repo), then mount that copy.
            data_dir = self._data_source_path
            remote_mount = self._remote_data_mount
            if data_dir is not None and remote_mount is not None and data_dir.exists():
                staged_data_dir = Path(workdir) / 'data' / data_dir.name
                try:
                    os.makedirs(staged_data_dir.parent, exist_ok=True)
                    # Copy tree into temp workdir; avoid re-copying if already present
                    if not staged_data_dir.exists():
                        shutil.copytree(str(data_dir), str(staged_data_dir))
                    file_mounts[remote_mount] = str(staged_data_dir)
                except Exception as e:
                    logger.warning(f"Failed to stage data directory for mounting: {e}. Mounting original path instead.")
                    file_mounts[remote_mount] = str(data_dir)
            else:
                logger.error("Data directory is not resolved; remote job may fail due to missing traces.")
            
            task.set_file_mounts(file_mounts)
            
            # Set up storage using storage_mounts
            task.set_storage_mounts({output_mount_path: self._results_storage})
            
            # Track workdir for later cleanup after job completion
            self._cluster_workdirs[cluster_name] = workdir
            
            # Set resources - force GCP
            resources_kwargs = {
                'cpus': self.cpus,
                'memory': self.memory,
                'cloud': sky.GCP(),  # Always use GCP
                'disk_tier': DISK_TIER,
            }
                    
            if self.instance_type:
                resources_kwargs['instance_type'] = self.instance_type
                
            task.set_resources(sky.Resources(**resources_kwargs))  # type: ignore
            
            # Launch cluster
            logger.info(f"🚀 Launching cluster {cluster_name} with {len(tasks)} tasks")
            logger.info(f"📍 Cloud: GCP")
            cpu_info = resources_kwargs.get('cpus', '16+')
            mem_info = resources_kwargs.get('memory', '32+')
            logger.info(f"📊 Resources: CPUs {cpu_info}, Memory {mem_info}GB")
            
            # Launch (async); do not wait for job_id here to maximize parallelism
            logger.info(f"⏳ Starting cluster provisioning...")
            launch_start = time.time()
            request_id = sky.launch(task, cluster_name=cluster_name)
            self.active_clusters.append(cluster_name)
            GLOBAL_ACTIVE_CLUSTERS.add(cluster_name)
            logger.info(f"✅ Launch command sent, request_id: {request_id}")
            return cluster_name, request_id, run_id
            
        except Exception as e:
            logger.error(f"Failed to launch batch {batch_idx}: {e}")
            raise e
            
        finally:
            # If launch failed before registering workdir, clean it up here
            if workdir and os.path.exists(workdir) and cluster_name not in self._cluster_workdirs:
                shutil.rmtree(workdir)
    
    def _wait_and_download(self, batch_idx: int, tasks: List[Dict], 
                          cluster_name: str, request_id: Any, run_id: str,
                          cache_dir: Path, params: Dict) -> List[Dict]:
        """Wait for job completion and download results."""
        results: List[Dict] = []
        had_failure = False
        local_debug_root: Optional[Path] = None
        try:
            # Wait for job completion by tailing logs synchronously
            # Wait for job to start and get job_id
            try:
                job_id, handle = sky.stream_and_get(request_id)
                assert job_id is not None, "Job ID is None"
                logger.info(f"✅ Job launched on {cluster_name} with job_id: {job_id}")
            except Exception as e:
                raise RuntimeError(f"Failed to get job_id for {cluster_name}: {e}")

            logger.info(f"⏳ Waiting for job {job_id} completion on {cluster_name}...")

            # Tail logs; if job fails, we still proceed to fetch artifacts for precise errors
            try:
                progress_stream = _ProgressStream(cluster_name, job_id)
                sky.tail_logs(cluster_name, job_id, follow=True, output_stream=progress_stream)
            except Exception as e:
                logger.error(f"tail_logs error for job {job_id}: {e}")
            
            # Removed fixed 120s sleep; _download_results_from_s3 will poll until files appear
            
            # Use the same persistent bucket for all runs
            s3_bucket_name = self._results_bucket_name
            
            # Prepare local debug dir to persist fetched artifacts
            local_debug_root = Path(params.get('OUTPUT_DIR', 'outputs/multi_region_scenario_analysis')) / 'skypilot_runs' / run_id / f"batch_{batch_idx}"
            local_debug_root.mkdir(parents=True, exist_ok=True)

            # Download results and diagnostics
            download_histories = bool(params.get('SEGMENT_VIZ_ENABLED', False))
            results = self._download_results_from_s3(s3_bucket_name, tasks, run_id, batch_idx, local_debug_root, download_histories=download_histories)

            # Determine if any task in this cluster failed, or no results produced
            if not results:
                had_failure = True
            else:
                for r in results:
                    if pd.isna(r.get('cost', float('nan'))) or ('error' in r) or ('error_type' in r):
                        had_failure = True
                        break
            return results
            
        except Exception as e:
            logger.error(f"Failed to wait/download batch {batch_idx}: {e}")
            had_failure = True
            results = [{**task, "cost": float('nan'), "error": str(e)} for task in tasks]
            return results
            
        finally:
            # If any failure occurred for this cluster, keep it for debugging
            if had_failure:
                self._clusters_keep_alive.add(cluster_name)
                gcs_uri = f"gs://{self._results_bucket_name}/{run_id}/batch_{batch_idx}/"
                logger.warning(
                    f"⚠️  Keeping cluster {cluster_name} alive due to failures in batch {batch_idx}.\n"
                    f"    • Run ID: {run_id}\n"
                    f"    • Results prefix: {gcs_uri}\n"
                    f"    • Local downloads: {local_debug_root if local_debug_root is not None else 'N/A'}\n"
                    f"    • Inspect logs: sky logs {cluster_name}  (or: sky tail-logs {cluster_name})\n"
                    f"    • Terminate when done: sky down {cluster_name}"
                )
            else:
                if self.auto_down:
                    self._terminate_cluster(cluster_name)
                else:
                    logger.info(f"⚠️  Cluster {cluster_name} is still running for debugging. Run 'sky down {cluster_name}' to terminate it.")

            workdir_path = self._cluster_workdirs.pop(cluster_name, None)
            if workdir_path and os.path.exists(workdir_path):
                try:
                    shutil.rmtree(workdir_path)
                except Exception as cleanup_err:
                    logger.warning(f"Failed to clean temporary workdir {workdir_path}: {cleanup_err}")
    
    def _download_results_from_s3(self, s3_bucket_name: str, tasks: List[Dict], run_id: str, batch_idx: int, local_debug_dir: Path, download_histories: bool = False) -> List[Dict]:
        """Download only batch summary and diagnostics from GCS. No per-task fallbacks.

        Simpler flow: we expect worker to always write batch_summary.json. If missing,
        we fetch logs and return precise errors per task.

        Args:
            download_histories: If True, also download history files for segment visualization.
        """
        try:
            from google.cloud import storage as gcs_storage
            gcs_client = gcs_storage.Client()
            bucket_name = s3_bucket_name

            summary_key = f'{run_id}/batch_{batch_idx}/batch_summary.json'
            results_key = f'{run_id}/batch_{batch_idx}/results.jsonl'
            worker_log_key = f'{run_id}/batch_{batch_idx}/batch_worker.log'
            exit_code_key = f'{run_id}/batch_{batch_idx}/exit_code.txt'

            os.makedirs(local_debug_dir, exist_ok=True)
            summary_file = os.path.join(local_debug_dir, 'batch_summary.json')
            results_file = os.path.join(local_debug_dir, 'results.jsonl')
            worker_log_file = os.path.join(local_debug_dir, 'batch_worker.log')
            exit_code_file = os.path.join(local_debug_dir, 'exit_code.txt')

            bucket = gcs_client.bucket(bucket_name)
            
            # Poll for results/summary availability to avoid fixed sleeps
            max_wait_s = int(os.environ.get('SKY_RESULTS_SYNC_MAX_WAIT_S', '600'))
            poll_interval_s = 3
            waited = 0
            summary = None
            results: list[dict] = []
            while waited <= max_wait_s:
                try:
                    # Prefer results.jsonl; it's the definitive output
                    blob = bucket.blob(results_key)
                    if blob.exists():
                        blob.download_to_filename(results_file)
                        logger.info(f"✅ Downloaded results.jsonl from GCS: {bucket_name}/{results_key}")
                        with open(results_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    results.append(json.loads(line))
                        break
                except Exception:
                    pass
                try:
                    # Fallback: summary presence indicates worker finished
                    blob_s = bucket.blob(summary_key)
                    if blob_s.exists():
                        blob_s.download_to_filename(summary_file)
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                        logger.info(f"✅ batch_summary.json present: {bucket_name}/{summary_key}")
                        # Some worker versions embed results in summary
                        if 'results' in summary:
                            results = summary['results']
                            break
                except Exception:
                    pass
                time.sleep(poll_interval_s)
                waited += poll_interval_s
            
            if results:
                logger.info(f"✅ Retrieved {len(results)} results")
            else:
                logger.error("No results found after polling. Fetching diagnostics only.")

            # Fetch diagnostics only when results are missing (to speed up happy path)
            if not results:
                try:
                    bucket.blob(worker_log_key).download_to_filename(worker_log_file)
                    logger.info(f"📥 Downloaded worker log for batch {batch_idx}")
                except Exception:
                    pass
                try:
                    bucket.blob(exit_code_key).download_to_filename(exit_code_file)
                    with open(exit_code_file, 'r') as f:
                        logger.info(f"Batch {batch_idx} worker exit code: {f.read().strip()}")
                except Exception:
                    pass

            # Download histories only if segment visualization is enabled
            if download_histories:
                try:
                    hist_prefix = f"{run_id}/batch_{batch_idx}/histories/"
                    blobs = list(bucket.list_blobs(prefix=hist_prefix))
                    # Write histories under the standard analysis output dir
                    local_hist_root = Path('outputs/multi_region_scenario_analysis') / 'histories'
                    for b in blobs:
                        if not b.name.endswith('.json') and not b.name.endswith('.json.gz'):
                            continue
                        rel = b.name[len(hist_prefix):]
                        local_path = local_hist_root / rel
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        b.download_to_filename(str(local_path))
                    if blobs:
                        logger.info(f"📥 Downloaded {len(blobs)} history files to {local_hist_root}")
                except Exception as e:
                    logger.warning(f"Failed to download histories: {e}")

            if results:
                return results
            # Return per-task errors if still no results
            return [{**task, "cost": float('nan'), "error": "No results produced (sync timeout)"} for task in tasks]

        except Exception as e:
            logger.error(f"Error downloading from bucket {s3_bucket_name}: {e}")
            return [{**task, "cost": float('nan'), "error": f"Download error: {e}"} for task in tasks]
    
    def _terminate_cluster(self, cluster_name: str):
        """Terminate a SkyPilot cluster using Python API."""
        try:
            # sky.down returns a request ID
            down_request_id = sky.down(cluster_name)
            # Fast teardown: don't wait for termination to finish
            if os.environ.get('SKY_FAST_TEARDOWN', '').lower() not in ('1', 'true', 'yes'):
                try:
                    sky.get(down_request_id)  # Wait for completion
                except Exception as e:
                    logger.warning(f"Error during cluster termination: {e}")
            if cluster_name in self.active_clusters:
                self.active_clusters.remove(cluster_name)
            GLOBAL_ACTIVE_CLUSTERS.discard(cluster_name)
            logger.info(f"Terminated cluster {cluster_name}")
        except Exception as e:
            logger.error(f"Failed to terminate cluster {cluster_name}: {e}")
    
    def cleanup(self):
        """Clean up all active clusters and temporary directories."""
        # Clean up clusters
        for cluster_name in self.active_clusters[:]:
            if cluster_name in self._clusters_keep_alive:
                logger.warning(f"⚠️  Preserving cluster {cluster_name} after failures. Remember to terminate it manually: sky down {cluster_name}")
                continue
            if self.auto_down:
                self._terminate_cluster(cluster_name)
            else:
                logger.info(f"⚠️  Cluster {cluster_name} is still running for debugging. Run 'sky down {cluster_name}' to terminate it.")
        # Summary of preserved clusters
        if self._clusters_keep_alive:
            logger.warning("==== Preserved clusters with failures ====")
            for name in sorted(self._clusters_keep_alive):
                run_id = self._cluster_run_map.get(name, '?')
                batch_idx = self._cluster_batch_map.get(name, -1)
                gcs_uri = f"gs://{self._results_bucket_name}/{run_id}/batch_{batch_idx}/"
                logger.warning(f"• {name} | run_id={run_id} | batch={batch_idx} | {gcs_uri}")
            
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Failed to clean up temp dir {temp_dir}: {e}")


def execute_tasks_with_skypilot(
    tasks: List[Dict], 
    cache_dir: Path, 
    params: Dict,
    auto_down: bool = True
) -> List[Dict]:
    """Execute tasks in parallel using SkyPilot on GCP (no env var dependencies)."""

    # Fixed, simple defaults (edit here if needed)
    # Defaults tuned for single remote machine execution; env vars can override.
    max_clusters = int(os.environ.get('SKY_MAX_CLUSTERS', '1'))
    cpus = os.environ.get('SKY_CPUS', '64+')
    memory = os.environ.get('SKY_MEMORY', '128+')
    instance_type = os.environ.get('SKY_INSTANCE_TYPE') or None
    tasks_per_cluster = int(os.environ.get('SKY_TASKS_PER_CLUSTER', str(max(1, len(tasks)))))

    executor = SkyPilotExecutor(
        max_parallel_clusters=max_clusters,
        cloud='gcp',
        instance_type=instance_type,
        auto_down=auto_down,
        cpus=cpus,
        memory=memory,
        tasks_per_cluster=tasks_per_cluster,
    )

    try:
        results = executor.execute_batch(tasks, cache_dir, params)
        return results
    finally:
        executor.cleanup()
