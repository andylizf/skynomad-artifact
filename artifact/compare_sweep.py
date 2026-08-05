#!/usr/bin/env python3
"""Compare a re-run sweep against the shipped CSV it should reproduce.

`README.md` states that each `--config` sweep regenerates
`research/data/<config>.csv` from the traces. This is how to check that claim
rather than take it: it matches rows across the two files on everything that
identifies a simulation, then reports the largest disagreement in cost.

    python artifact/compare_sweep.py --config deadline_sensitivity
    python artifact/compare_sweep.py --all

Rows are matched on strategy, task type, region, window and every swept
parameter -- not on file order, which differs between runs. A row present on one
side only is reported rather than skipped, since a sweep that quietly drops
simulations would otherwise pass.

Exits non-zero if anything is outside --tolerance, so it can be used as a check.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BM_HINT = "scripts_multi/benchmark_multi_region_modular.py"

CONFIGS = [
    "deadline_sensitivity",
    "region_scaling",
    "region_selection",
    "checkpoint_sensitivity",
    "v100",
]

# Everything that identifies one simulation. Leaving any of these out merges
# distinct rows and silently compares the wrong pairs.
KEY_COLUMNS = (
    "strategy",
    "task_type",
    "trace_mode",
    "region",
    "window_index",
    "trace_index",
    "deadline_ratio",
    "checkpoint_size",
    "restart_overhead",
    "num_regions",
    "scenario_name",
)


def load(path: Path) -> dict[tuple, float]:
    rows: dict[tuple, float] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            cost = r.get("cost", "")
            if cost in ("", None):
                continue  # summary lines carry no cost
            key = tuple(r.get(c, "") for c in KEY_COLUMNS)
            if key in rows:
                raise SystemExit(
                    f"{path}: duplicate key {key}; KEY_COLUMNS does not identify a row"
                )
            rows[key] = float(cost)
    return rows


def find_rerun(config: str, rerun_dir: Path) -> Path | None:
    hits = sorted((rerun_dir / config).glob("scenario_results_t*.csv"))
    return hits[0] if hits else None


def compare(config: str, rerun_dir: Path, tolerance: float) -> bool | None:
    """True if the config agrees, False if it differs, None if there was no rerun."""
    shipped_path = REPO / "research" / "data" / f"{config}.csv"
    rerun_path = find_rerun(config, rerun_dir)
    if rerun_path is None:
        print(f"{config:24s} SKIP   no rerun under {rerun_dir / config}")
        return None
    if not shipped_path.is_file():
        print(f"{config:24s} ERROR  missing {shipped_path}")
        return False

    shipped, rerun = load(shipped_path), load(rerun_path)
    common = shipped.keys() & rerun.keys()
    only_shipped = len(shipped) - len(common)
    only_rerun = len(rerun) - len(common)

    worst_rel = worst_abs = 0.0
    exact = 0
    for k in common:
        a, b = shipped[k], rerun[k]
        if a == b:
            exact += 1
        worst_abs = max(worst_abs, abs(b - a))
        if a:
            worst_rel = max(worst_rel, abs(b - a) / abs(a))

    ok = worst_rel <= tolerance and not only_shipped and not only_rerun
    print(
        f"{config:24s} {'OK  ' if ok else 'DIFF'} "
        f"{len(common):5d} rows  {exact:5d} bit-identical  "
        f"max {worst_rel * 100:.4f}%  (${worst_abs:.6f})"
        + (f"  unmatched: {only_shipped} shipped / {only_rerun} rerun" if not ok else "")
    )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", help="one config name")
    ap.add_argument("--all", action="store_true", help="every config in the README table")
    ap.add_argument("--rerun-dir", default="outputs/rerun", type=Path,
                    help="where the sweeps were written (default outputs/rerun)")
    ap.add_argument("--tolerance", type=float, default=0.005,
                    help="largest relative cost difference to accept (default 0.005)")
    args = ap.parse_args()

    if not args.config and not args.all:
        ap.error("pass --config <name> or --all")

    targets = CONFIGS if args.all else [args.config]
    print(f"shipped: research/data/<config>.csv   rerun: {args.rerun_dir}/<config>/\n")
    results = [compare(c, args.rerun_dir, args.tolerance) for c in targets]
    compared = [r for r in results if r is not None]

    # A run that compared nothing is not a pass. Reporting "agrees" after skipping
    # every config would give a green light to an empty check, which is worse than
    # failing: it reads as verification and is not.
    if not compared:
        print("\nnothing was compared -- run the sweeps first, e.g.\n"
              f"  python {BM_HINT} --config {targets[0]} --output-dir {args.rerun_dir}")
        sys.exit(2)

    ok = all(compared)
    skipped = len(results) - len(compared)
    tail = f" ({skipped} config(s) had no rerun and were skipped)" if skipped else ""
    print("\n" + ("every compared row agrees" + tail if ok
                  else "differences above tolerance -- see the DIFF lines"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
