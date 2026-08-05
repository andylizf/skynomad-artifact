#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def _iter_region_files(region_dir: Path) -> List[Path]:
    files = []
    if not region_dir.exists() or not region_dir.is_dir():
        return files
    for name in sorted(region_dir.iterdir(), key=lambda p: p.name):
        if name.is_file() and name.suffix == ".json":
            files.append(name)
    return files


def _load_trace(path: Path) -> Tuple[List[int], float]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj["data"]
    # Availability in Env is defined as `not trace[tick]`
    # Normalize to integers 0/1 for logic simplicity
    data01 = [1 if (not bool(x)) else 0 for x in data]
    gap_seconds = float(obj["metadata"]["gap_seconds"])
    return data01, gap_seconds


def _compute_stats_for_region(region_dir: Path) -> Tuple[float, float]:
    total_ticks = 0
    total_available = 0
    run_durations_hours: List[float] = []
    gap_s_seen: float = 0.0

    for f in _iter_region_files(region_dir):
        data01, gap_s = _load_trace(f)
        if gap_s_seen == 0.0:
            gap_s_seen = gap_s
        # Aggregate availability
        total_ticks += len(data01)
        total_available += sum(data01)

        # Collect contiguous availability run lengths
        run_len = 0
        for v in data01:
            if v == 1:
                run_len += 1
            else:
                if run_len > 0:
                    run_durations_hours.append(run_len * gap_s / 3600.0)
                    run_len = 0
        if run_len > 0:
            run_durations_hours.append(run_len * gap_s / 3600.0)

    if total_ticks == 0:
        raise ValueError(f"No trace data in {region_dir}")
    avg_availability = total_available / total_ticks
    if len(run_durations_hours) == 0:
        avg_duration = 0.0
    else:
        avg_duration = sum(run_durations_hours) / len(run_durations_hours)
    return avg_availability, avg_duration


def _round4(x: float) -> float:
    return float(f"{x:.4f}")


def main():
    ap = argparse.ArgumentParser("Compute region availability and duration stats from aligned traces")
    ap.add_argument("--data-root", type=str, default="data/converted_multi_region_aligned",
                    help="Root directory containing <region>/<trace_id>.json files")
    ap.add_argument("--regions", type=str, nargs="*", default=None,
                    help="Optional explicit region directory names to compute. Default: all under data-root")
    ap.add_argument("--output", type=str, default=None,
                    help="Optional path to write JSON of computed stats")
    ap.add_argument("--verify", action="store_true",
                    help="Verify against values hardcoded in unified_cost_model.py and fail on mismatch")
    args = ap.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    if args.regions:
        region_names = args.regions
    else:
        region_names = [p.name for p in sorted(root.iterdir()) if p.is_dir()]

    results: Dict[str, Dict[str, float]] = {}
    for rname in region_names:
        rdir = root / rname
        if not rdir.exists():
            continue
        avail, dur = _compute_stats_for_region(rdir)
        results[rname] = {
            "avg_availability": _round4(avail),
            "avg_duration": _round4(dur),
        }

    # Print summary
    for rname in sorted(results.keys()):
        s = results[rname]
        print(f"{rname}: availability={s['avg_availability']:.4f}, duration={s['avg_duration']:.4f} h")

    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {outp}")

    if args.verify:
        from sky_spot.strategies.unified_cost_model import REGION_CHARACTERISTICS_BY_NAME
        mismatches = []
        for name, vals in results.items():
            if name not in REGION_CHARACTERISTICS_BY_NAME:
                continue
            code_vals = REGION_CHARACTERISTICS_BY_NAME[name]
            ca = _round4(float(code_vals["avg_availability"]))
            cd = _round4(float(code_vals["avg_duration"]))
            if abs(vals["avg_availability"] - ca) > 1e-4 or abs(vals["avg_duration"] - cd) > 1e-3:
                mismatches.append((name, vals, {"avg_availability": ca, "avg_duration": cd}))
        if mismatches:
            print("Verification mismatches:")
            for name, got, exp in mismatches:
                print(f"  {name}: got={got}, expected={exp}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()


