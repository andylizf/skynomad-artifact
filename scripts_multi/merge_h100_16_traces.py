#!/usr/bin/env python3
"""
Merge H100-16 traces from first and second half collections.
Fills the gap between collections using the last state of the first half.

This is the script that produced `data/converted_multi_region_aligned_h100_16_merged`,
the H100 traces the simulation sweeps read. It is included so the gap filling that
directory's README describes can be read rather than taken on trust.

It is not runnable from this package as shipped: its second input, the second-half
collection `data/converted_multi_region_aligned_new`, is not included -- only the
merged output is. What the merge did is recorded per zone in
`<zone>/full.json` under `metadata.gap_filled`: a 36,599 s (10.2 h) gap at tick
indices 696-755, 60 filled ticks per zone across 13 zones (780 of 27,482 ticks,
2.8%), each filled with that zone's state immediately before the gap
(`fill_strategy: last_state_0` in 6 zones, `last_state_1` in 7).
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import shutil


def calculate_gap(end_time_str, start_time_str, interval_seconds):
    """Calculate gap between two timestamps and return number of data points needed."""
    end_time = datetime.fromisoformat(end_time_str)
    start_time = datetime.fromisoformat(start_time_str)

    gap_duration = start_time - end_time
    gap_seconds = int(gap_duration.total_seconds())
    gap_points = gap_seconds // interval_seconds

    print(f"Gap calculation:")
    print(f"  First half ends:  {end_time}")
    print(f"  Second half starts: {start_time}")
    print(f"  Gap duration: {gap_duration} ({gap_seconds} seconds)")
    print(f"  Data interval: {interval_seconds} seconds")
    print(f"  Gap data points needed: {gap_points}")

    return gap_points, gap_seconds


def merge_traces(first_dir, second_dir, output_dir):
    """Merge traces from two directories with gap filling."""

    print(f"\n=== Merging H100-16 traces ===")
    print(f"First half:  {first_dir}")
    print(f"Second half: {second_dir}")
    print(f"Output:      {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of regions from first directory (should match second)
    first_regions = [d for d in os.listdir(first_dir) if d.endswith("_h100_16")]
    second_regions = [d for d in os.listdir(second_dir) if d.endswith("_h100_16")]

    print(f"\nRegions found:")
    print(f"  First half: {len(first_regions)} regions")
    print(f"  Second half: {len(second_regions)} regions")

    # Verify same regions in both
    missing_in_second = set(first_regions) - set(second_regions)
    missing_in_first = set(second_regions) - set(first_regions)

    if missing_in_second:
        print(f"WARNING: Regions missing in second half: {missing_in_second}")
    if missing_in_first:
        print(f"WARNING: Regions missing in first half: {missing_in_first}")

    common_regions = sorted(set(first_regions) & set(second_regions))
    print(f"  Common regions: {len(common_regions)}")

    # Process each region
    total_original_windows = 0
    total_merged_windows = 0

    for region in common_regions:
        print(f"\n--- Processing {region} ---")

        region_first_dir = os.path.join(first_dir, region)
        region_second_dir = os.path.join(second_dir, region)
        region_output_dir = os.path.join(output_dir, region)
        os.makedirs(region_output_dir, exist_ok=True)

        # Read full.json files
        with open(os.path.join(region_first_dir, "full.json"), "r") as f:
            data1 = json.load(f)

        with open(os.path.join(region_second_dir, "full.json"), "r") as f:
            data2 = json.load(f)

        # Calculate gap
        interval = data1["metadata"]["gap_seconds"]
        end_time_first = datetime.fromisoformat(
            data1["metadata"]["start_time"]
        ) + timedelta(seconds=(len(data1["data"]) - 1) * interval)
        start_time_second = data2["metadata"]["start_time"]

        gap_points, gap_seconds = calculate_gap(
            end_time_first.isoformat(), start_time_second, interval
        )

        # Create gap fill data using last state of first half
        last_state = data1["data"][-1]
        last_price = data1["prices"][-1]

        print(f"Gap fill strategy:")
        print(f"  Last state of first half: {last_state}")
        print(f"  Last price of first half: ${last_price}")
        print(
            f"  Will fill {gap_points} points with state={last_state}, price=${last_price}"
        )

        gap_data = [last_state] * gap_points
        gap_prices = [last_price] * gap_points

        # Merge data
        merged_data = data1["data"] + gap_data + data2["data"]
        merged_prices = data1["prices"] + gap_prices + data2["prices"]

        print(f"Merged statistics:")
        print(f"  First half points: {len(data1['data'])}")
        print(f"  Gap fill points: {gap_points}")
        print(f"  Second half points: {len(data2['data'])}")
        print(f"  Total merged points: {len(merged_data)}")

        # Create merged metadata
        merged_metadata = data1["metadata"].copy()
        merged_metadata["gap_filled"] = {
            "gap_start_index": len(data1["data"]),
            "gap_end_index": len(data1["data"]) + gap_points - 1,
            "gap_points": gap_points,
            "gap_seconds": gap_seconds,
            "fill_strategy": f"last_state_{last_state}",
            "original_first_half_points": len(data1["data"]),
            "original_second_half_points": len(data2["data"]),
            "second_half_start_time": data2["metadata"]["start_time"],
        }

        # Use price_info from second half if it has more complete info
        if "on_demand_price" in data2["metadata"].get("price_info", {}):
            merged_metadata["price_info"] = data2["metadata"]["price_info"]

        # Create merged full.json
        merged_full = {
            "metadata": merged_metadata,
            "data": merged_data,
            "prices": merged_prices,
        }

        with open(os.path.join(region_output_dir, "full.json"), "w") as f:
            json.dump(merged_full, f)

        # Copy and renumber window files
        # First half: 0.json -> 7.json (keep as is)
        windows_copied = 0
        for i in range(8):
            src_file = os.path.join(region_first_dir, f"{i}.json")
            dst_file = os.path.join(region_output_dir, f"{i}.json")
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                windows_copied += 1

        # Second half: 0.json -> 7.json becomes 8.json -> 15.json
        for i in range(8):
            src_file = os.path.join(region_second_dir, f"{i}.json")
            dst_file = os.path.join(region_output_dir, f"{i + 8}.json")
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                windows_copied += 1

        total_original_windows += 16  # 8 from each half
        total_merged_windows += windows_copied

        print(
            f"  Copied {windows_copied} window files (0-7 from first half, 8-15 from second half)"
        )

    print(f"\n=== Merge Summary ===")
    print(f"Processed regions: {len(common_regions)}")
    print(f"Window files copied: {total_merged_windows}/{total_original_windows}")
    print(f"Output directory: {output_dir}")

    # Create README
    readme_content = f"""# Merged H100-16 Traces

This directory contains merged traces from two consecutive H100 data collection periods.

## Source Data
- **First half**: {first_dir}
- **Second half**: {second_dir}

## Merge Strategy
- Combined {len(common_regions)} regions with H100-16 data
- Gap between collections filled using last state of first half
- Window files: 0-7 from first half, 8-15 from second half
- Each full.json contains gap_filled metadata with details

## Files Structure
- `full.json`: Complete merged trace with gap filling
- `0.json` - `7.json`: Windows from first half
- `8.json` - `15.json`: Windows from second half

## Gap Filling Details
See `gap_filled` metadata in each region's full.json for:
- Gap location (start/end indices)
- Fill strategy used
- Original data point counts
- Gap duration and reasoning

Generated by: scripts/merge_h100_16_traces.py
"""

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)

    print(f"Created README.md with merge details")


def main():
    # Default paths
    first_dir = "data/converted_multi_region_aligned"
    second_dir = "data/converted_multi_region_aligned_new"
    output_dir = "data/converted_multi_region_aligned_h100_16_merged"

    # Check if input directories exist
    if not os.path.exists(first_dir):
        print(f"ERROR: First half directory not found: {first_dir}")
        return 1

    if not os.path.exists(second_dir):
        print(f"ERROR: Second half directory not found: {second_dir}")
        return 1

    if os.path.exists(output_dir):
        response = input(f"Output directory {output_dir} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return 0
        shutil.rmtree(output_dir)

    try:
        merge_traces(first_dir, second_dir, output_dir)
        print(f"\n✓ Merge completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Error during merge: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
