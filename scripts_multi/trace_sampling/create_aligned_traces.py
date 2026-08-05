"""
Create aligned trace files from 2023 availability CSV data.

IMPORTANT: This script converts availability data (1=available, 0=unavailable) to
the preempted format (1=preempted, 0=available) used by TraceEnv, because
TraceEnv.spot_available() returns "not trace[tick]".

Pricing has two parts. `metadata.price_info` carries the p3.2xlarge (1xV100)
reference rates from `sky_spot.utils.ACTUAL_COSTS['v100_1']`, $3.06/hr on-demand
and $0.918/hr spot; `TraceEnv` refuses to load a trace without that block
(sky_spot/env.py has no price fallback by design). What the simulator actually
bills against is the per-tick `prices` array attached by `_price_series` below:
one series per zone, spanning $0.918 to $1.85. Both are needed -- under a flat
$0.918 the price-sensitive policies do not reproduce research/data/v100.csv.
"""
import os
import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Make `sky_spot` importable when this file is run as a script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sky_spot.utils import ACTUAL_COSTS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
SOURCE_CSV_DIR = Path("data/real/availability/2023-02-15/processed")
TARGET_DATA_DIR = Path("data/converted_multi_region_aligned")

# AWS p3.2xlarge (1x V100): (on-demand $/hr, spot $/hr).
V100_ON_DEMAND_PRICE, V100_SPOT_PRICE = ACTUAL_COSTS["v100_1"]

# The numbered window files are random samples of the aligned trace. Seed so a
# second run of prepare_data.sh produces byte-identical output.
RANDOM_SEED = 0


PRICE_SERIES_FILE = Path("data/v100_spot_prices.json")
_PRICE_SERIES: dict[str, list[float]] | None = None


def _price_series() -> dict[str, list[float]]:
    """Per-tick spot prices for the nine zones, keyed by region name.

    The source availability CSVs carry availability only. The published traces
    that produced research/data/v100.csv also carried a `prices` array, one entry
    per tick, and that array is what makes the price-sensitive policies (Next
    Spot Oracle above all) reproduce. It ships here as data/v100_spot_prices.json,
    reconstructed from the 100 windowed traces of the original run and checked
    against the public AWS spot price archive: the per-zone maxima agree exactly
    ($1.8464 for us-east-1a, $1.4053 for us-east-1c, $1.3661 for us-east-1d,
    $1.4003 for us-east-1f).
    """
    global _PRICE_SERIES
    if _PRICE_SERIES is None:
        if PRICE_SERIES_FILE.is_file():
            with open(PRICE_SERIES_FILE) as f:
                _PRICE_SERIES = json.load(f)
        else:
            _PRICE_SERIES = {}
    return _PRICE_SERIES


def _price_info(region_name: str) -> dict:
    """Price block written into every generated trace's metadata.

    `region_name` looks like `us-west-2a_v100_1`; the zone is everything before
    the device suffix, matching what sky_spot/env.py expects.
    """
    zone = region_name.rsplit("_", 2)[0]
    return {
        "zone": zone,
        "device": "v100_1",
        "instance_type": "p3.2xlarge",
        "on_demand_price": V100_ON_DEMAND_PRICE,
        "price": V100_SPOT_PRICE,
        "price_source": (
            "reference rates from sky_spot.utils.ACTUAL_COSTS['v100_1']; the "
            "per-tick series billed against is the trace's own `prices` array"
        ),
    }

def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Loads a CSV and prepares the DataFrame."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def main():
    """Main function to create aligned traces from original CSVs."""
    if not SOURCE_CSV_DIR.exists():
        logging.error(f"Source CSV directory not found: {SOURCE_CSV_DIR}")
        return

    # 1. Load all region dataframes
    region_dfs = {}
    for csv_file in SOURCE_CSV_DIR.glob("*.csv"):
        region_name = csv_file.stem
        logging.info(f"Loading data for {region_name}...")
        try:
            region_dfs[region_name] = load_and_prepare_data(csv_file)
        except Exception as e:
            logging.error(f"Failed to load {csv_file}: {e}")
            continue

    if not region_dfs:
        logging.error("No data loaded. Exiting.")
        return

    # 2. Find the common time window
    common_start = max(df.index.min() for df in region_dfs.values())
    common_end = min(df.index.max() for df in region_dfs.values())

    if common_start >= common_end:
        logging.error("No common time window found across all CSV files. Exiting.")
        return

    logging.info(f"Common time window found: {common_start} to {common_end}")

    # 3. Generate aligned traces
    np.random.seed(RANDOM_SEED)
    num_traces = 100  # Number of trace files to generate
    trace_length_hours = 60
    
    # Assuming gap_seconds is consistent, get it from the first dataframe
    first_df = next(iter(region_dfs.values()))
    gap_seconds = int((first_df.index[1] - first_df.index[0]).total_seconds())
    trace_length_ticks = int(trace_length_hours * 3600 / gap_seconds)

    # Align all dataframes to the common window
    aligned_dfs = {name: df[common_start:common_end] for name, df in region_dfs.items()}

    # 3.1 Export full-length aligned traces for each region
    for region_name, df in aligned_dfs.items():
        # Convert availability (1=available) to preempted format (1=preempted)
        availability_data = df['availability']
        trace_data = (~availability_data.astype(bool)).astype(int).tolist()

        new_metadata = {
            "gap_seconds": gap_seconds,
            "start_time": common_start.isoformat(),
            "source_file": str(SOURCE_CSV_DIR / f"{region_name}.csv"),
            "price_info": _price_info(region_name),
        }

        new_content = {
            "metadata": new_metadata,
            "data": trace_data
        }
        series = _price_series().get(region_name)
        if series:
            new_content["prices"] = series[:len(trace_data)]

        target_dir = TARGET_DATA_DIR / region_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "full.json"

        with open(target_file, 'w') as f:
            json.dump(new_content, f)
        logging.info(f"Wrote full trace for {region_name} to {target_file}")

    max_start_index = max(0, len(next(iter(aligned_dfs.values()))) - trace_length_ticks)

    for i in range(num_traces):
        logging.info(f"Generating trace set {i}/{num_traces}...")
        start_idx = np.random.randint(0, max_start_index)
        
        # Get the absolute start time for this trace from the first aligned df
        trace_start_time = next(iter(aligned_dfs.values())).index[start_idx]

        for region_name, df in aligned_dfs.items():
            end_idx = min(start_idx + trace_length_ticks, len(df))
            # Convert availability (1=available) to preempted format (1=preempted)
            # because TraceEnv.spot_available() returns "not trace[tick]"
            availability_data = df['availability'].iloc[start_idx:end_idx]
            trace_data = (~availability_data.astype(bool)).astype(int).tolist()

            new_metadata = {
                "gap_seconds": gap_seconds,
                "start_time": trace_start_time.isoformat(),
                "source_file": str(SOURCE_CSV_DIR / f"{region_name}.csv"),
                "price_info": _price_info(region_name),
            }

            new_content = {
                "metadata": new_metadata,
                "data": trace_data
            }
            series = _price_series().get(region_name)
            if series:
                new_content["prices"] = series[start_idx:start_idx + len(trace_data)]

            target_dir = TARGET_DATA_DIR / region_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / f"{i}.json"

            with open(target_file, 'w') as f:
                json.dump(new_content, f)

    logging.info(f"Successfully generated {num_traces} sets of aligned traces in {TARGET_DATA_DIR}")

if __name__ == "__main__":
    main()
