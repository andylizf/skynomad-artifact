"""Cache management for benchmark simulations - self-contained module."""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

# Cache behaviour toggles
def _compute_cache_version() -> str:
    env_version = os.environ.get('CACHING_VERSION')
    if env_version:
        return env_version

    branch = os.environ.get('GIT_BRANCH')
    if not branch:
        head_file = Path('.git/HEAD')
        if head_file.exists():
            try:
                content = head_file.read_text().strip()
                if content.startswith('ref:'):
                    branch = content.split('/', 2)[-1]
                elif content:
                    branch = content[:8]
            except Exception:
                branch = None
    if not branch:
        branch = 'unknown'
    # Bumped 2 -> 3 when generate_cache_filename stopped sorting the trace list.
    # Entries written by the old, order-insensitive key are not interchangeable
    # with the new ones, so they must not be reused.
    return f"3_{branch}"


CACHE_VERSION = _compute_cache_version()

GCS_BUCKET_NAME = os.environ.get('CACHE_GCS_BUCKET', 'skypilot-benchmark-results')

_TRUTHY = ('1', 'true', 'yes', 'on')


def _remote_cache_requested() -> bool:
    """Off unless explicitly asked for.

    The default bucket is the authors' private GCS bucket, which nobody
    reproducing this artifact can read. Attempting it on every task only costs
    the reviewer a credential lookup and a network round trip per simulation,
    and the artifact should not reach out to external services during review.
    The local cache under the output directory is unaffected and still used.

    Set ENABLE_REMOTE_CACHE=1 (and CACHE_GCS_BUCKET to a bucket you can write)
    to turn it back on. DISABLE_REMOTE_CACHE=1 still forces it off and wins.
    """
    if os.environ.get('DISABLE_REMOTE_CACHE', '').strip().lower() in _TRUTHY:
        return False
    return os.environ.get('ENABLE_REMOTE_CACHE', '').strip().lower() in _TRUTHY


if not _remote_cache_requested():
    logger.info(
        "Caching: local only. The remote GCS cache is opt-in "
        "(ENABLE_REMOTE_CACHE=1); its default bucket is not publicly readable."
    )
    _bucket = None
    _GCS_AVAILABLE = False
else:
    try:
        from google.cloud import storage as gcs_storage

        _gcs_client = gcs_storage.Client()
        _bucket = _gcs_client.bucket(GCS_BUCKET_NAME)
        _GCS_AVAILABLE = True
        logger.info(f"Caching: using GCS bucket {GCS_BUCKET_NAME}")
    except Exception as e:  # pragma: no cover - network/auth issues only at runtime
        logger.warning(f"Caching: GCS unavailable ({e}); falling back to local cache only")
        _bucket = None
        _GCS_AVAILABLE = False


def generate_cache_filename(
    strategy: str,
    env: str,
    traces: List[str],
    checkpoint_size: float = 50.0,
    restart_overhead: float = 0.2,
    deadline_hours: Optional[float] = None,
    task_duration_hours: Optional[float] = None,
    env_start_hours: float = 0.0,
    count_cross_region_migration_time: bool = False,
) -> str:
    """Generates a cache filename, using hash if too long.

    `traces` is ordered, and the order is part of the key. Region order is a
    real input to the simulation (it decides how a strategy breaks ties between
    equally-priced regions and which region a scenario starts in), so two
    scenarios over the same SET of regions in a different ORDER are different
    experiments and must not share a cache entry. This used to sort
    `trace_descs` before joining, which made the key order-insensitive: e.g.
    region_selection's "Global (13)" and region_scaling's "Progressive_13" are
    the same 13 zones in different orders and hashed identically, so whichever
    config ran second read the other's results out of the shared cache.
    """
    trace_descs = []
    for trace_path_str in traces:
        trace_path = Path(trace_path_str)
        if "union" in trace_path.name:
            # For union traces, the name itself is descriptive
            trace_descs.append(trace_path.stem)
        else:
            trace_descs.append(f"{trace_path.parent.name}_{trace_path.stem}")

    trace_identifier = "+".join(trace_descs)
    safe_strategy_name = strategy.replace("/", "_").replace("\\", "_")
    
    # Build filename with all parameters
    filename_parts = [
        f"cv{CACHE_VERSION}",
        safe_strategy_name,
        env,
        trace_identifier,
        f"ckpt{checkpoint_size}gb",
        f"ro{restart_overhead}h",
        f"start{env_start_hours}h",
    ]
    if count_cross_region_migration_time:
        filename_parts.append('count_migtime')
    
    if task_duration_hours is not None:
        filename_parts.append(f"task{task_duration_hours}h")
    if deadline_hours is not None:
        filename_parts.append(f"ddl{deadline_hours}h")
    
    filename = "_".join(filename_parts) + ".json"
    
    # If filename is too long, use hash
    if len(filename) > 200:  # Conservative limit to avoid filesystem issues
        # Include all parameters in hash calculation
        hash_input = (
            f"{strategy}_{env}_{trace_identifier}_ckpt{checkpoint_size}gb_"
            f"ro{restart_overhead}h_start{env_start_hours}h"
        )
        if count_cross_region_migration_time:
            hash_input += '_count_migtime'
        if task_duration_hours is not None:
            hash_input += f"_task{task_duration_hours}h"
        if deadline_hours is not None:
            hash_input += f"_ddl{deadline_hours}h"
        content_hash = hashlib.md5(hash_input.encode()).hexdigest()
        # Keep strategy name but shorten traces part
        short_strategy = safe_strategy_name[-30:] if len(safe_strategy_name) > 30 else safe_strategy_name
        filename = f"cv{CACHE_VERSION}_{short_strategy}_{env}_{content_hash}.json"

    return filename


def _validate_payload(data: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    version = str(data.get('cache_version', ''))
    if version != str(CACHE_VERSION):
        logger.debug(
            "Cache version mismatch (have=%s, expected=%s) for %s",
            version,
            CACHE_VERSION,
            source,
        )
        return None
    return data


def _read_local_cache(cache_file: Path) -> Optional[Dict[str, Any]]:
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                return _validate_payload(data, str(cache_file))
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}")
    return None


def _write_local_cache(cache_file: Path, payload: Dict[str, Any]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(payload, f)


def _read_cache_payload(cache_file: Path) -> Optional[Dict[str, Any]]:
    """Load raw cache payload from local disk or GCS."""

    payload = _read_local_cache(cache_file)
    if payload is not None:
        return payload

    if not _GCS_AVAILABLE or _bucket is None:
        return None

    blob_name = f"cache/{cache_file.name}"
    try:
        blob = _bucket.blob(blob_name)
        if not blob.exists():
            logger.debug(f"GCS cache MISS: {blob_name}")
            return None

        content = blob.download_as_text()
        data = json.loads(content)
        payload = _validate_payload(data, blob_name)
        if payload is None:
            return None

        try:
            _write_local_cache(cache_file, payload)
        except Exception as e:  # pragma: no cover - disk issues are rare
            logger.warning(f"Failed to write local cache copy {cache_file}: {e}")
        logger.debug(f"GCS cache HIT: {blob_name}")
        return payload
    except Exception as e:
        logger.warning(f"Failed to load from GCS cache {cache_file.name}: {e}")
        return None


def _write_cache_payload(cache_file: Path, payload: Dict[str, Any]) -> None:
    """Persist payload to local disk and, when available, to GCS."""
    payload = {**payload, "cache_version": str(CACHE_VERSION)}

    try:
        _write_local_cache(cache_file, payload)
    except Exception as e:
        logger.warning(f"Failed to save local cache {cache_file}: {e}")

    if not _GCS_AVAILABLE or _bucket is None:
        return

    blob_name = f"cache/{cache_file.name}"
    try:
        final_blob = _bucket.blob(blob_name)
        final_blob.upload_from_string(
            json.dumps(payload),
            content_type='application/json'
        )
        logger.debug(f"Saved to GCS cache: {blob_name}")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_file} to GCS: {e}")


def load_from_cache(cache_file: Path) -> Optional[float]:
    """Load cost from cache file if it exists."""
    payload = _read_cache_payload(cache_file)
    if payload is None:
        return None
    return payload.get("mean_cost")


def save_to_cache(cache_file: Path, mean_cost: float) -> None:
    """Save cost to cache file only if it's a valid finite number."""
    # Validate input before saving
    if pd.isna(mean_cost) or not pd.notna(mean_cost):
        logger.warning(f"Refusing to cache invalid cost: {mean_cost}")
        return

    data = {"mean_cost": float(mean_cost)}
    _write_cache_payload(cache_file, data)


def load_result_entry(cache_file: Path) -> Optional[Dict[str, Any]]:
    """Load a cached simulation result if available."""
    payload = _read_cache_payload(cache_file)
    if payload is None:
        return None
    result = payload.get("result")
    if result is None:
        return None
    return result


def save_result_entry(cache_file: Path, result: Dict[str, Any]) -> None:
    """Persist a simulation result entry to cache."""
    if result is None:
        return
    if pd.isna(result.get("cost", float("nan"))):
        logger.debug("Skipping cache save for NaN cost result")
        return
    payload = {
        "result": {
            "cost": float(result.get("cost")),
            "migrations": result.get("migrations"),
            "error_type": result.get("error_type"),
            "error_details": result.get("error_details"),
        },
        "mean_cost": float(result.get("cost")),
    }
    _write_cache_payload(cache_file, payload)
