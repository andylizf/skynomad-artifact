#!/usr/bin/env python3
"""
Regenerate data/v100_spot_price_archive_summary.csv from the public price archive.

The V100 traces carry availability only, so `create_aligned_traces.py` attaches
one price per zone from `sky_spot.utils.ACTUAL_COSTS['v100_1']` ($3.06/hr
on-demand, $0.918/hr spot). This script checks that figure against what AWS
actually charged for p3.2xlarge over the trace's own window, using Eric Pauley's
AWS Spot Price History archive (concept DOI 10.5281/zenodo.14198917), and writes
the per-AZ summary the repository ships.

What it finds, on the 2023-02-15T22:08:16Z + 1091.9h window the traces cover:
$0.918 is the floor across all nine availability zones and the standing price in
five of them (both us-east-2 zones and all three us-west-2 zones). The four
us-east-1 zones run higher, $1.23 to $1.65 time-weighted. So the uniform price
understates us-east-1 and matches the rest, which biases against the paper's
result rather than for it: with true prices the expensive zones get more
expensive, and the policies that price zones (SkyNomad, UP(AP)) gain over the one
that does not (UP(A)).

The archive keys rows by availability-zone ID (`use1-az1`), not zone name
(`us-east-1a`). AWS maps IDs to names per account, so the nine IDs cannot be
matched to the nine zone names from public data alone. That is why this produces
a summary rather than a per-zone price series to feed back into the traces.

The 2023 archive file is 511 MB, so it is not downloaded automatically. Fetch it
once and point this script at it:

    curl -L -o 2023.tsv.zst \\
      https://zenodo.org/api/records/18821638/files/2023.tsv.zst/content

    .venv/bin/python scripts_multi/trace_sampling/check_v100_archive_prices.py \\
        --archive 2023.tsv.zst

Requires the `zstd` binary on PATH (macOS: brew install zstd).
"""

import argparse
import csv
import datetime as dt
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The aligned V100 traces start here and run 1091.9 hours; see the metadata of
# data/converted_multi_region_aligned/*/full.json.
WINDOW_START = dt.datetime(2023, 2, 15, 22, 8, 16, tzinfo=dt.timezone.utc)
WINDOW_HOURS = 1091.9

INSTANCE_TYPE = "p3.2xlarge"
PRODUCT = "Linux/UNIX"
# us-east-1, us-east-2, us-west-2: the three regions the traces cover.
AZ_PREFIXES = ("use1-", "use2-", "usw2-")

DEFAULT_OUT = REPO / "data" / "v100_spot_price_archive_summary.csv"


def rows_from_archive(archive: Path):
    """Yield (az_id, price, timestamp) for the instance type we care about."""
    if shutil.which("zstd") is None:
        sys.exit("error: zstd not found on PATH (macOS: brew install zstd)")
    proc = subprocess.Popen(
        ["zstd", "-dc", str(archive)], stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 5:
            continue
        az, itype, product, price, ts = parts
        if itype != INSTANCE_TYPE or product != PRODUCT:
            continue
        if not az.startswith(AZ_PREFIXES):
            continue
        yield az, float(price), dt.datetime.fromisoformat(ts)
    proc.stdout.close()
    if proc.wait() != 0:
        sys.exit(f"error: zstd exited {proc.returncode} reading {archive}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, type=Path,
                    help="path to the archive's 2023.tsv.zst")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.archive.is_file():
        sys.exit(f"error: archive not found: {args.archive}")

    window_end = WINDOW_START + dt.timedelta(hours=WINDOW_HOURS)
    print(f"window: {WINDOW_START.isoformat()} -> {window_end.isoformat()}")

    in_window = defaultdict(list)
    carried = {}  # last price seen before the window opens
    for az, price, ts in rows_from_archive(args.archive):
        if ts < WINDOW_START:
            carried[az] = price
        elif ts <= window_end:
            in_window[az].append((ts, price))

    zones = sorted(set(in_window) | set(carried))
    if not zones:
        sys.exit("error: no matching rows; is this the right archive file?")

    out_rows = []
    for az in zones:
        points = []
        if az in carried:
            points.append((WINDOW_START, carried[az]))
        points.extend(in_window.get(az, []))
        if not points:
            continue
        # Time-weighted mean: each quoted price holds until the next quote.
        total = weighted = 0.0
        for i, (ts, price) in enumerate(points):
            end = points[i + 1][0] if i + 1 < len(points) else window_end
            span = (end - ts).total_seconds()
            total += span
            weighted += price * span
        prices = [p for _, p in points]
        out_rows.append({
            "az_id": az,
            "price_at_window_start": f"{points[0][1]:.4f}",
            "min": f"{min(prices):.4f}",
            "max": f"{max(prices):.4f}",
            "time_weighted_mean": f"{(weighted / total if total else float('nan')):.4f}",
            "price_changes_in_window": len(in_window.get(az, [])),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    for r in out_rows:
        print(f"  {r['az_id']:<10s} start={r['price_at_window_start']} "
              f"min={r['min']} max={r['max']} twmean={r['time_weighted_mean']} "
              f"changes={r['price_changes_in_window']}")
    floor = min(float(r["min"]) for r in out_rows)
    print(f"\nfloor across all {len(out_rows)} zones: ${floor:.4f}"
          f"   (ACTUAL_COSTS['v100_1'] spot price: $0.918)")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
