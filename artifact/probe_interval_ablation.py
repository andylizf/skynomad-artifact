#!/usr/bin/env python3
"""
Reproduce Figure 14 (probe_interval_ablation): cost breakdown vs probe interval.

The plotting script was not committed, so this one re-runs the sweep rather than
replaying stored numbers. It sweeps the probe interval over the same seven points
as the paper (10 min .. 16 h) and plots compute+egress against probing overhead.

Configuration: 13 H100 zones, 92 h job, 138 h deadline, 500 GB checkpoint, the
`unified_cost_model_risk_legacy` arm whose probe cadence is set by
--risk-probe-interval-ticks (default 12 ticks = 2 h, the paper's default).

Measured against the published figure -- compute cost $1,546-$1,632 with 6%
variation, probing overhead falling from $2,011 at 10 minutes to $170 at the
2-hour default:

    interval   probe (here / paper)   compute (here / paper)
      10min       1919 / 2011            1585 / 1546
      30min        644 /  671            1536 / 1559
         1h        318 /  337            1543 / 1559
         2h        159 /  170            1552 / 1558
         4h         82 /   86            1577 / 1583
         8h         41 /   44            1608 / 1604
        16h         24 /   24            1594 / 1632

Usage:
    python artifact/probe_interval_ablation.py --output outputs/figures/probe_interval_ablation.pdf
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

# One tick is gap_seconds = 600 s = 10 min in the H100 traces.
TICKS_PER_10MIN = 1
INTERVALS = [  # (label, ticks)
    ("10min", 1),
    ("30min", 3),
    ("1h", 6),
    ("2h", 12),   # simulator/e2e default
    ("4h", 24),
    ("8h", 48),
    ("16h", 96),
]
DEFAULT_LABEL = "2h"


def run_one(repo: Path, python: str, ticks: int, args) -> dict:
    """Run the simulator once at a given probe interval; return its result JSON."""
    traces = sorted(
        (repo / "data/converted_multi_region_aligned_h100_16_merged").glob("*/full.json")
    )
    if not traces:
        sys.exit("error: H100 traces not found. Run artifact/prepare_data.sh first.")

    with tempfile.TemporaryDirectory() as td:
        cmd = [
            python, "main.py",
            "--strategy=unified_cost_model_risk_legacy",
            "--env=multi_trace",
            "--trace-files", *[str(t) for t in traces],
            "--task-duration-hours", str(args.task_duration_hours),
            "--deadline-hours", str(args.deadline_hours),
            "--restart-overhead-hours", str(args.restart_overhead_hours),
            "--checkpoint-size-gb", str(args.checkpoint_size_gb),
            "--risk-probe-interval-ticks", str(ticks),
            "--output-dir", td,
            "--no-history",
        ]
        if args.probe_revalidate:
            # Without this a region is probed once for the whole job, so the
            # interval governs almost nothing and the sweep comes out flat.
            cmd.append("--probe-revalidate")
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        results = list(Path(td).glob("*.json"))
        if not results:
            sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
            sys.exit(f"error: simulator produced no result at ticks={ticks}")
        return json.loads(results[0].read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/figures/probe_interval_ablation.pdf")
    ap.add_argument("--task-duration-hours", type=float, default=92.0)
    ap.add_argument("--deadline-hours", type=float, default=138.0)
    ap.add_argument("--restart-overhead-hours", type=float, default=0.2)
    ap.add_argument("--checkpoint-size-gb", type=float, default=500.0)
    ap.add_argument("--python", default=".venv/bin/python")
    ap.add_argument(
        "--probe-revalidate",
        action="store_true",
        help="Pass --probe-revalidate to the simulator, so a region with a live "
             "virtual instance is re-probed and re-billed every interval, the way "
             "the sampling-based baselines are. Off by default, matching the "
             "shipped results.",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python = args.python

    labels, compute, probing = [], [], []
    for label, ticks in INTERVALS:
        d = run_one(repo, python, ticks, args)
        total = float(d["costs"][0])
        probe = float(d["probe_costs"][0])
        labels.append(label)
        probing.append(probe)
        compute.append(total - probe)  # total already includes probe charges
        print(f"  {label:>6s} (ticks={ticks:3d}): "
              f"total={total:9.2f}  compute+egress={total-probe:9.2f}  probe={probe:8.2f}  "
              f"probes={d['probe_tick_counts'][0]}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, compute, label="Compute + egress")
    ax.bar(labels, probing, bottom=compute, label="Probing overhead")

    if DEFAULT_LABEL in labels:
        i = labels.index(DEFAULT_LABEL)
        ax.annotate("default", xy=(i, compute[i] + probing[i]),
                    xytext=(i, (compute[i] + probing[i]) * 1.08),
                    ha="center", fontsize=9)

    # Match the published figure's y range so the tick labels agree.
    ax.set_ylim(0, 4000)
    ax.set_xlabel("Probe interval")
    ax.set_ylabel("Total cost (USD)")
    ax.legend(frameon=False)
    fig.tight_layout()

    out = repo / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved: {out}")

    lo, hi = min(compute), max(compute)
    print(f"\nCompute+egress varies {lo:.2f}-{hi:.2f} USD "
          f"({(hi-lo)/lo*100:.1f}% across intervals)")
    print(f"Probing overhead: {probing[0]:.2f} USD at 10min -> "
          f"{probing[-1]:.2f} USD at 16h")
    if args.probe_revalidate:
        print(
            "Note: --probe-revalidate re-probes and re-bills a region even while "
            "its virtual instance is still alive. The published figure runs with "
            "it off."
        )


if __name__ == "__main__":
    main()
