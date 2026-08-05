"""Shared helpers for trace path resolution and union trace construction."""
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections.abc import Sequence
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)


def _collect_path_entries(value) -> List[Path]:
    """Internal helper to extract path entries from mixed specifications."""

    entries: List[Path] = []

    def _append_from_string(raw: str) -> None:
        # Support os.pathsep-delimited (':' on Unix) and comma-separated lists.
        chunks = []
        for part in raw.split(os.pathsep):
            chunks.extend(part.split(','))
        for chunk in chunks:
            candidate = chunk.strip()
            if candidate:
                entries.append(Path(candidate))

    if value is None:
        return entries

    if isinstance(value, Path):
        entries.append(value)
        return entries

    if isinstance(value, os.PathLike):
        entries.append(Path(value))
        return entries

    if isinstance(value, str):
        _append_from_string(value)
        return entries

    if isinstance(value, Sequence):
        for item in value:
            entries.extend(_collect_path_entries(item))
        return entries

    raise TypeError(f"Unsupported data path specification: {value!r}")


def normalize_data_roots(data_path_spec) -> List[Path]:
    """Normalize a data path specification into an ordered, de-duplicated list of Paths."""

    collected = _collect_path_entries(data_path_spec)
    normalized: List[Path] = []
    seen: set[Path] = set()
    for entry in collected:
        expanded = entry.expanduser()
        if expanded in seen:
            continue
        normalized.append(expanded)
        seen.add(expanded)
    return normalized


def find_region_base_path(region: str, data_roots: Sequence[Path]) -> Path | None:
    """Return the first data root containing the given region, or None if not found."""

    for base in data_roots:
        candidate = base / region
        if candidate.exists():
            return candidate
    return None


def resolve_region_trace_path(region: str, file_name: str, data_roots: Sequence[Path]) -> Path:
    """Return the first existing trace path for region/file across the provided data roots."""

    attempted: List[Path] = []
    for base in data_roots:
        candidate = base / region / file_name
        if candidate.exists():
            return candidate
        attempted.append(candidate)

    if attempted:
        attempted_str = ", ".join(str(p) for p in attempted)
    else:
        attempted_str = "<no data paths provided>"
    raise FileNotFoundError(
        f"Required trace file not found for region '{region}' ({file_name}). Checked: {attempted_str}"
    )


def get_or_create_union_trace(
    regions: List[str],
    trace_index: int,
    data_path,
    use_full: bool = False,
    window_start_ticks: int | None = None,
    window_length_ticks: int | None = None,
) -> str:
    """Create or retrieve a union trace for a set of regions.

    Rule: 1 only if ALL regions are preempted, 0 if ANY region is available.
    Output path: outputs/multi_region_scenario_analysis/union_traces
    """
    data_roots = normalize_data_roots(data_path)
    if not data_roots:
        raise ValueError("No data paths provided for union trace construction.")
    union_traces_dir = Path("outputs/multi_region_scenario_analysis/union_traces")
    union_traces_dir.mkdir(parents=True, exist_ok=True)

    regions_str = "+".join(sorted(regions))
    # Different data_path may produce different unions for same regions/index; add a short hash
    import hashlib
    dp_signature_seed = "||".join(str(p.resolve()) for p in data_roots)
    dp_sig = hashlib.md5(dp_signature_seed.encode()).hexdigest()[:8]
    if use_full and window_start_ticks is not None and window_length_ticks is not None:
        union_trace_filename = f"union_{dp_sig}_{regions_str}_full_s{window_start_ticks}_l{window_length_ticks}.json"
    else:
        union_trace_filename = (
            f"union_{dp_sig}_{regions_str}_full.json" if use_full else f"union_{dp_sig}_{regions_str}_{trace_index}.json"
        )
    union_trace_path = union_traces_dir / union_trace_filename

    # Use a file lock + atomic rename to avoid concurrent writers creating
    # truncated/empty JSON files under high parallelism.
    lock_path = str(union_trace_path) + '.lock'
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=120):
            # If file exists, verify it is valid JSON; if invalid, recreate.
            if union_trace_path.exists():
                try:
                    with open(union_trace_path, 'r') as f:
                        json.load(f)
                    return str(union_trace_path)
                except Exception:
                    try:
                        union_trace_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                        logger.warning(f"Invalid union trace detected. Recreating: {union_trace_path}")
                    except Exception:
                        pass
            # Fall through to (re)create
    except Timeout:
        logger.warning(f"Timeout acquiring lock for union trace: {union_trace_path}. Proceeding without lock.")

    # Load all regional traces and optionally slice window
    all_data: List[List[int]] = []
    all_prices: List[List[float] | None] = []
    metadata_template = None
    min_len = float("inf")
    for region_name in regions:
        file_name = "full.json" if use_full else f"{trace_index}.json"
        trace_file_path = resolve_region_trace_path(region_name, file_name, data_roots)
        with open(trace_file_path, "r") as f:
            trace = json.load(f)
        data_seq: List[int] = trace["data"]
        price_seq: List[float] = trace.get("prices")
        if use_full and window_start_ticks is not None and window_length_ticks is not None:
            end = window_start_ticks + window_length_ticks
            data_seq = data_seq[window_start_ticks:end]
            price_seq = price_seq[window_start_ticks:end]
        all_data.append(data_seq)
        all_prices.append(price_seq)
        min_len = min(min_len, len(data_seq))
        if metadata_template is None:
            metadata_template = trace["metadata"]

    trimmed_data = [data[: int(min_len)] for data in all_data]
    final_union = [1 if all(d[i] == 1 for d in trimmed_data) else 0 for i in range(int(min_len))]
    # Compute union price as per-tick minimum across regions where price exists
    union_price: List[float] | None = None
    if any(p is not None for p in all_prices):
        union_price = []
        # Trim prices to min_len
        trimmed_prices: List[List[float] | None] = [
            (p[: int(min_len)] if p is not None else None) for p in all_prices
        ]
        for i in range(int(min_len)):
            vals = [p[i] for p in trimmed_prices if p is not None and p[i] is not None]
            if len(vals) == 0:
                # If no region has price at this tick, omit by repeating last or set None
                # We choose to set None-equivalent by skipping price emission entirely below
                union_price = None
                break
            union_price.append(float(min(vals)))

    payload = {"metadata": metadata_template, "data": final_union}
    if union_price is not None:
        # Use 'prices' key to be consistent with sky_spot.trace.Trace
        payload["prices"] = union_price

    # Write atomically: write to temp then replace
    tmp_path = Path(str(union_trace_path) + '.tmp')
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, union_trace_path)

    logger.info(f"Created union trace for {len(regions)} regions at {union_trace_path}")
    return str(union_trace_path)


def get_trace_paths_for_task(task: Dict, data_path) -> Tuple[str, List[str]]:
    """Get env type and trace paths for a task.

    - single_region: single trace
    - multi_region + union: single union trace
    - multi_region + single/best_single: single region trace (first region)
    - multi_region default: multi_trace with all regions
    """
    use_full = bool(task.get("use_full_trace", False))
    data_roots = normalize_data_roots(data_path)
    if not data_roots:
        raise ValueError("No data paths provided for trace resolution.")

    if task["task_type"] == "single_region":
        env_type = "trace"
        file_name = "full.json" if use_full else f"{task['trace_index']}.json"
        trace_path = resolve_region_trace_path(task["region"], file_name, data_roots)
        trace_paths = [str(trace_path)]
        return env_type, trace_paths

    regions = task["regions_in_scenario"]

    # Union trace mode
    if task.get("strategy") == "quick_optimal" or task.get("trace_mode") == "union":
        env_type = "trace"
        window_start_ticks = None
        window_length_ticks = None
        if use_full:
            # Require explicit window ticks for full-trace union to avoid implicit fallbacks
            if "window_start_ticks" not in task or "window_length_ticks" not in task:
                raise ValueError("Missing required window ticks for full-trace union: window_start_ticks, window_length_ticks")
            window_start_ticks = int(task["window_start_ticks"])  # type: ignore
            window_length_ticks = int(task["window_length_ticks"])  # type: ignore
        union_trace_path = get_or_create_union_trace(
            regions, task["trace_index"], data_roots, use_full=use_full,
            window_start_ticks=window_start_ticks, window_length_ticks=window_length_ticks,
        )
        trace_paths = [union_trace_path]
        return env_type, trace_paths

    # Single/best single mode
    if task.get("trace_mode") in ("single", "best_single"):
        env_type = "trace"
        target_region = regions[0]
        file_name = "full.json" if use_full else f"{task['trace_index']}.json"
        trace_path = resolve_region_trace_path(target_region, file_name, data_roots)
        trace_paths = [str(trace_path)]
        return env_type, trace_paths

    # Default multi-region: multi_trace
    env_type = "multi_trace"
    if use_full:
        trace_paths = [
            str(resolve_region_trace_path(r, "full.json", data_roots))
            for r in regions
        ]
    else:
        trace_paths = [
            str(resolve_region_trace_path(r, f"{task['trace_index']}.json", data_roots))
            for r in regions
        ]
    return env_type, trace_paths


def get_min_full_trace_hours(regions: List[str], data_path) -> Tuple[float, float]:
    """Return (min_hours_across_regions, gap_seconds) for full.json traces.

    Raises FileNotFoundError if any region lacks full.json.
    """
    import json

    min_len = float("inf")
    gap_seconds: float | None = None
    data_roots = normalize_data_roots(data_path)
    if not data_roots:
        raise ValueError("No data paths provided for trace duration inspection.")
    for region_name in regions:
        trace_file_path = resolve_region_trace_path(region_name, "full.json", data_roots)
        with open(trace_file_path, "r") as f:
            t = json.load(f)
        if gap_seconds is None:
            gap_seconds = float(t["metadata"]["gap_seconds"])  # type: ignore
        else:
            assert abs(gap_seconds - float(t["metadata"]["gap_seconds"])) < 1e-6, (
                f"Mismatched gap_seconds in {trace_file_path}"
            )
        min_len = min(min_len, len(t["data"]))
    assert gap_seconds is not None
    min_hours = min_len * gap_seconds / 3600.0
    return min_hours, gap_seconds
