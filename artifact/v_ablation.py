#!/usr/bin/env python3
"""
Reproduce the Progress Value Ablation table (appendix A.7).

Six closed-form progress values run through the same utility pipeline, so V is
the only thing that varies. Four cells: V100 4-region and H100 8-region, each at
deadline ratio 1.5 and 2.0, twelve stratified windows per cell. The table reports
each candidate's mean cost minus Optimal's on the same windows, and the row sum
across the four cells.

The formulas are in sky_spot/strategies/_v_candidates.py and are wired into the
pipeline by sky_spot/strategies/unified_cost_model_f_ablation.py; the scenarios
are scripts_multi/benchmark_components/configs/v_ablation_{v100,h100}.py.

Usage:
    .venv/bin/python artifact/v_ablation.py
    .venv/bin/python artifact/v_ablation.py --skip-run    # reuse outputs/v_ablation
"""

import argparse
import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Every gap this prints matches the appendix to the two decimals it shows; the
# sum column adds the unrounded gaps, so it can differ by a cent from the
# appendix's, which adds the rounded ones.
CELLS = [
    ("v_ablation_v100_4node", "V100 4-node", ["1.5", "2.0"]),
    ("v_ablation_h100", "H100 8-region", ["1.5", "2.0"]),
]

# The appendix defines the gap as "the mean per-trace cost minus the Optimal cost"
# -- the same quantity on both sides, which is the `cost` column. `--with-billed`
# adds each candidate's idle time, transfer and probing on top, for inspection;
# that is an asymmetric comparison, since Optimal never migrates.

# Display order matches the appendix table: default first, then alternatives.
CANDIDATES = [
    ("ucm_v_log_barrier", "Log-barrier (default)"),
    ("ucm_v_alpha_05", "alpha=0.5"),
    ("ucm_v_s2t1", "Quadratic surrogate"),
    ("ucm_v_hjb_exp", "HJB exponential"),
    ("ucm_v_s5t1", "Time-only"),
    ("ucm_v_neely_dpp", "Neely DPP"),
]

OPTIMAL = "multi_region_oracle_dp"


def run_cell(config: str, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        "scripts_multi/benchmark_multi_region_modular.py",
        "--config", config,
        "--output-dir", str(out_dir),
    ]
    print(f"  running {config} -> {out_dir}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:] + proc.stderr[-3000:])
        sys.exit(f"error: {config} failed with code {proc.returncode}")


BILLED = ("downtime_cost", "transfer_cost", "probe_cost")


def load_cell(out_dir: Path, compute_only: bool = False) -> dict:
    """Return {(ratio, strategy): [cost, ...]} for one cell.

    With `compute_only` -- the default, and what the appendix compares -- every
    strategy is read off the `cost` column alone. Without it, candidates also carry
    their idle-time, transfer and probe charges while Optimal stays on compute.
    """
    hits = sorted(out_dir.rglob("scenario_results_*.csv"))
    if not hits:
        sys.exit(f"error: no results under {out_dir}")
    # Newest wins. Re-running a cell into a directory that already holds results
    # leaves both files behind, and taking the first in sort order silently
    # reports the older one -- a run that looks like it happened but did not.
    latest = max(hits, key=lambda p: p.stat().st_mtime)
    if len(hits) > 1:
        print(f"  note: {len(hits)} result files under {out_dir}, reading {latest.name} "
              f"(newest); delete the directory for a clean run", file=sys.stderr)
    by_key = defaultdict(list)
    for row in csv.DictReader(latest.open()):
        if row.get("task_type") != "multi_region":
            continue
        total = float(row["cost"])
        if not compute_only and row["strategy"] != OPTIMAL:
            total += sum(float(row.get(k) or 0.0) for k in BILLED)
        by_key[(row["deadline_ratio"], row["strategy"])].append(total)
    return by_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/v_ablation")
    ap.add_argument("--skip-run", action="store_true",
                    help="reuse an existing sweep instead of re-running it")
    ap.add_argument("--with-billed", action="store_true",
                    help="add each candidate's idle-time, transfer and probe "
                         "cost to its gap (the appendix compares cost alone)")
    args = ap.parse_args()

    candidates = CANDIDATES

    root = REPO / args.out_dir
    columns = []      # (cell label, ratio)
    optimal = {}      # (cell, ratio) -> optimal mean
    gaps = defaultdict(dict)  # strategy -> (cell, ratio) -> gap

    for config, label, ratios in CELLS:
        out_dir = root / config
        if not args.skip_run:
            run_cell(config, out_dir)
        data = load_cell(out_dir, not args.with_billed)
        for ratio in ratios:
            key = (label, ratio)
            opt_costs = data.get((ratio, OPTIMAL))
            if not opt_costs:
                sys.exit(f"error: no Optimal rows for {label} at ratio {ratio}")
            opt = statistics.mean(opt_costs)
            optimal[key] = opt
            columns.append(key)
            for strategy, _ in candidates:
                costs = data.get((ratio, strategy))
                if costs:
                    gaps[strategy][key] = statistics.mean(costs) - opt

    name_w = max(len(n) for _, n in candidates) + 2
    head = "".join(f"{lab} {r}x".rjust(20) for lab, r in columns)
    print()
    print(f"{'V candidate':<{name_w}}{head}{'Sum':>12}")
    print(f"{'Optimal':<{name_w}}" + "".join(f"${optimal[c]:>19.2f}" for c in columns))
    print("-" * (name_w + 20 * len(columns) + 12))

    rows = []
    for strategy, label in candidates:
        cells = gaps.get(strategy, {})
        if len(cells) != len(columns):
            print(f"{label:<{name_w}}  (missing cells; run without --skip-run)")
            continue
        total = sum(cells.values())
        rows.append((total, label, [cells[c] for c in columns]))

    for total, label, vals in sorted(rows):
        print(f"{label:<{name_w}}" + "".join(f"{v:>+20.2f}" for v in vals) + f"{total:>+12.2f}")

    out_csv = root / "v_ablation_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate"] + [f"{lab} {r}x" for lab, r in columns] + ["sum"])
        w.writerow(["Optimal"] + [f"{optimal[c]:.4f}" for c in columns] + [""])
        for total, label, vals in sorted(rows):
            w.writerow([label] + [f"{v:.4f}" for v in vals] + [f"{total:.4f}"])
    print(f"\nSaved: {out_csv}")
    print("Lower is better. Each entry is the candidate's mean cost minus "
          "Optimal's on the same twelve windows.")


if __name__ == "__main__":
    main()
