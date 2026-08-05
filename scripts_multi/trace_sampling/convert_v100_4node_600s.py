#!/usr/bin/env python3
"""Build the V100 4-node traces the appendix's V100 ablation cell runs on.

Source: three AWS availability zones captured at four requested nodes, 3156 ticks
of 300 s each. Four steps get from there to what the cell needs:

    capacity -> binary   a zone is usable when at least one node is free
    300 s -> 600 s       coarsen the grid, requiring the whole coarse tick
    price                one constant per zone, the mean of that zone's series
    truncate             all three zones to the shortest, 1578 ticks = 263.0 h

The result is written to data/converted_v100_4node_600s/ and consumed by
--config v_ablation_v100_4node.

Usage:
    .venv/bin/python scripts_multi/trace_sampling/convert_v100_4node_600s.py \
        --out data/converted_v100_4node_600s
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# The four-node capture of the three zones. data/v100_multinode_raw/ holds a
# different capture of the same zones (3664/5131/5131 ticks) which is not
# interchangeable: its availability pattern sits elsewhere on the timeline.
RAW = REPO / "data/spot_hedge_4node_src"
ZONES = ["us-east-1f_v100_1", "us-east-2a_v100_1", "us-west-2c_v100_1"]
# capacity -> binary threshold: a zone is usable when any of the four nodes is
# free. At this threshold us-east-1f is 40.1% available with 34 preemptions, which
# is the availability the cell is characterised by; requiring all four gives
# 32.9% and 45.
GANG = int(os.environ.get("GANG_THRESHOLD", "1"))
# All three source files are exactly 3156 ticks, which halves to 1578 coarse
# ticks. At a 310-tick deadline that leaves 1578 - 310 + 1 = 1269 start positions
# for the window sampler.
SRC_TICKS = 3156
SRC_GAP = 300     # the raw grid
DST_GAP = 600     # the grid the ablation cell runs on


def load_capacity(zone: str) -> list[int]:
    payload = json.loads((RAW / f"{zone}.json").read_text())
    return list(payload["data"])


def to_binary(capacity: list[int], gang: int) -> list[int]:
    """0 = available, 1 = preempted -- the convention Trace expects."""
    return [0 if c >= gang else 1 for c in capacity]


RULE = os.environ.get("RESAMPLE_RULE", "all")

# One constant price per zone -- the mean of that zone's single-node series --
# rather than a per-tick series. The cell is characterised by a single figure per
# zone ($0.93 for us-east-2a to $1.27 for us-east-1f, a 1.36x spread), and a
# constant is what makes windows that cover the same work cost the same.
# CONSTANT_PRICE=0 attaches the per-tick series instead, for comparison.
CONSTANT_PRICE = os.environ.get("CONSTANT_PRICE", "1") == "1"


def resample(binary: list[int], factor: int) -> list[int]:
    """Coarsen the grid. RESAMPLE_RULE picks the rule; the default is `all`.

      all      a coarse tick is available only if every fine tick in it was --
               a job spanning the whole tick needs the capacity throughout
      majority available if at least half the fine ticks were
      first    take the first fine tick of each group
      any      available if any fine tick was
    """
    out = []
    for i in range(0, len(binary) - factor + 1, factor):
        chunk = binary[i:i + factor]
        free = sum(1 for v in chunk if v == 0)
        if RULE == "majority":
            out.append(0 if free * 2 >= len(chunk) else 1)
        elif RULE == "first":
            out.append(chunk[0])
        elif RULE == "any":
            out.append(0 if free else 1)
        else:
            out.append(0 if free == len(chunk) else 1)
    return out


def price_series(zone: str, length: int) -> list[float] | None:
    """Per-tick spot price from the 1-node traces of the same zone, if present.

    Align by wall-clock, not by index: the 1-node traces sit on a 195 s grid
    (20158 ticks), not the 300 s grid of the capacity data, so stepping by
    DST_GAP // SRC_GAP stretched the price axis by 600/195 vs 600/300 -- a factor
    of 1.54 -- and slid every price away from the tick it belongs to.
    """
    # data/v100_1node_prices/ is the price capture that goes with the four-node
    # availability data. converted_multi_region_aligned/ is a later capture of the
    # same zones on the same 195 s grid, about 1.6% dearer, and is the fallback.
    primary = REPO / "data/v100_1node_prices" / f"{zone}.json"
    src = REPO / "data/converted_multi_region_aligned" / zone / "full.json"
    if primary.is_file():
        payload = json.loads(primary.read_text())
    elif src.is_file():
        payload = json.loads(src.read_text())
    else:
        return None
    prices = payload.get("prices")
    if not prices:
        return None
    src_gap = (payload.get("gap_seconds")
               or payload.get("metadata", {}).get("gap_seconds") or SRC_GAP)
    if CONSTANT_PRICE:
        return [sum(prices) / len(prices)] * length
    coarse = []
    for i in range(length):
        idx = int(i * DST_GAP / src_gap)
        coarse.append(prices[min(idx, len(prices) - 1)])
    return coarse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/converted_v100_4node_600s")
    args = ap.parse_args()
    out_root = REPO / args.out

    factor = DST_GAP // SRC_GAP
    series = {}
    for zone in ZONES:
        binary = to_binary(load_capacity(zone)[:SRC_TICKS], GANG)
        series[zone] = resample(binary, factor)

    shortest = min(len(v) for v in series.values())
    for zone, data in series.items():
        data = data[:shortest]
        prices = price_series(zone, shortest)
        payload = {
            "metadata": {
                "gap_seconds": DST_GAP,
                # TraceEnv needs both rates. On-demand for these p3 zones is
                # $3.06/hr; spot runs $0.93-$1.27 across the three.
                "price_info": {
                    "price": (prices[0] if prices else 0.918),
                    "on_demand_price": 3.06,
                },
                "source_file": str(RAW / f"{zone}.json"),
                "note": f"capacity>={GANG} -> available; resampled "
                        f"{SRC_GAP}s->{DST_GAP}s; one constant price per zone",
            },
            "data": data,
        }
        if prices:
            payload["prices"] = prices
        dest = out_root / zone
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "full.json").write_text(json.dumps(payload))
        avail = sum(1 for v in data if v == 0) / len(data) * 100
        print(f"  {zone}: {len(data)} ticks x {DST_GAP}s = "
              f"{len(data) * DST_GAP / 3600:.1f}h, {avail:.1f}% available"
              f"{' , prices attached' if prices else ''}")

    print(f"\nshortest trace: {shortest} ticks = {shortest * DST_GAP / 3600:.1f}h")
    print(f"at a 51.67 h deadline that leaves "
          f"{shortest - 310 + 1} window start positions")


if __name__ == "__main__":
    main()
