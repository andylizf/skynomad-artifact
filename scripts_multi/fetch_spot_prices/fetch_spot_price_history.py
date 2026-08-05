#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import sys
from typing import List, Optional, Tuple, Dict


DEVICE_TO_AWS_INSTANCE = {
    # GPU
    'v100_1': 'p3.2xlarge',
    'v100_8': 'p3.16xlarge',
    'k80_1': 'p2.xlarge',
    'k80_8': 'p2.8xlarge',
    't4_4': 'g4dn.12xlarge',
    't4_8': 'g4dn.metal',
    'a10g_4': 'g5.12xlarge',
    # CPU (best-effort mapping; may not be used for AWS spot)
    'intel_64': 'r5.16xlarge',
    'intel_48': 'r5.12xlarge',
}


def _split_region_dir_name(region_dir_name: str) -> Tuple[str, str]:
    # e.g. 'us-east-1a_v100_1' -> ('us-east-1a', 'v100_1')
    try:
        zone, device = region_dir_name.split('_', 1)
        return zone, device
    except Exception:
        raise ValueError(f"Unexpected region dir name format: {region_dir_name}")


def _aws_region_from_zone(zone: str) -> str:
    # 'us-east-1a' -> 'us-east-1'
    return zone[:-1]


def _load_az_map_from_file(map_path: str, aws_region: str) -> Dict[str, str]:
    # Supports TSV/CSV with headers containing ZoneName and ZoneId or two-column without headers
    mapping: Dict[str, str] = {}
    try:
        import csv
        is_tsv = map_path.lower().endswith('.tsv')
        with open(map_path, 'r', encoding='utf-8', errors='ignore') as f:
            first = f.readline()
            if not first:
                return mapping
            f.seek(0)
            has_header = ('ZoneName' in first and 'ZoneId' in first) or ('zone' in first and 'zoneid' in first)
            if has_header:
                reader = csv.DictReader(f, delimiter='\t' if is_tsv else ',')
                for row in reader:
                    name = str(row.get('ZoneName') or row.get('zone') or '').strip()
                    zid = str(row.get('ZoneId') or row.get('zoneid') or '').strip()
                    if not name or not zid:
                        continue
                    if name.startswith(aws_region):
                        mapping[name] = zid
            else:
                # Expect two columns: ZoneName, ZoneId
                for line in f:
                    parts = line.rstrip('\n').split('\t' if is_tsv else ',')
                    if len(parts) < 2:
                        continue
                    name, zid = parts[0].strip(), parts[1].strip()
                    if name.startswith(aws_region) and zid:
                        mapping[name] = zid
    except Exception:
        return {}
    return mapping


def _resolve_global_az_id(local_zone: str, aws_region: str, map_file: Optional[str]) -> Optional[str]:
    # First try map file if provided; otherwise try boto3; else None
    if map_file:
        m = _load_az_map_from_file(map_file, aws_region)
        if local_zone in m:
            return m[local_zone]
    try:
        import boto3  # type: ignore
        ec2 = boto3.client('ec2', region_name=aws_region)
        resp = ec2.describe_availability_zones(AllAvailabilityZones=True)
        for z in resp.get('AvailabilityZones', []):
            if z.get('RegionName') == aws_region and z.get('ZoneName') == local_zone:
                zid = z.get('ZoneId')
                if isinstance(zid, str) and zid:
                    return zid
    except Exception:
        return None
    return None


def _fetch_history_boto3(
    aws_region: str,
    availability_zone: str,
    instance_type: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
    product_description: str = 'Linux/UNIX',
) -> List[Tuple[dt.datetime, float]]:
    try:
        import boto3  # type: ignore
    except Exception as e:
        raise RuntimeError("boto3 is required. Install with: uv pip install boto3") from e

    ec2 = boto3.client('ec2', region_name=aws_region)

    events: List[Tuple[dt.datetime, float]] = []
    next_token: Optional[str] = None
    while True:
        kwargs = dict(
            StartTime=start_time,
            EndTime=end_time,
            ProductDescriptions=[product_description],
            InstanceTypes=[instance_type],
            AvailabilityZone=availability_zone,
            MaxResults=1000,
        )
        if next_token:
            kwargs['NextToken'] = next_token
        resp = ec2.describe_spot_price_history(**kwargs)
        for rec in resp.get('SpotPriceHistory', []):
            ts = rec['Timestamp']
            price = float(rec['SpotPrice'])
            events.append((ts, price))
        next_token = resp.get('NextToken')
        if not next_token:
            break

    # Sort ascending by timestamp
    events.sort(key=lambda x: x[0])
    return events


def _interpolate_prices(
    events: List[Tuple[dt.datetime, float]],
    start_time: dt.datetime,
    gap_seconds: int,
    num_ticks: int,
    default_price: Optional[float] = None,
) -> List[float]:
    prices: List[float] = []
    if not events:
        if default_price is None:
            raise ValueError("No spot price events returned and no default_price provided")
        return [default_price] * num_ticks

    def to_secs(d: dt.datetime) -> float:
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.timestamp()

    event_secs = [to_secs(e[0]) for e in events]
    start_sec = to_secs(start_time)

    # We guarantee by loader that events[0] <= start_time by inserting last_before if needed

    # Walk the events once while emitting per-tick prices (step function)
    idx = 0
    current_price = events[0][1]
    for i in range(num_ticks):
        t_sec = start_sec + i * gap_seconds
        while idx + 1 < len(events) and event_secs[idx + 1] <= t_sec:
            idx += 1
            current_price = events[idx][1]
        prices.append(current_price)
    return prices


def _region_to_global_prefix(region: str) -> Optional[str]:
    # Map 'us-east-1' -> 'use1', 'us-west-2' -> 'usw2', 'eu-west-3' -> 'euw3'
    try:
        parts = region.split('-')  # e.g., ['us', 'east', '1']
        if len(parts) != 3:
            return None
        return parts[0][:2] + parts[1][0] + parts[2]
    except Exception:
        return None


def _load_archive_csv(
    archive_dir: str,
    filter_zone: Optional[str],
    filter_it: Optional[str],
    start_time: dt.datetime,
    end_time: dt.datetime,
    filter_zone_global_id: Optional[str],
) -> List[Tuple[dt.datetime, float]]:
    # Eric Pauley dataset is tab-separated without headers: zone, instanceType, product, price, timestamp
    # Also support CSV/header form if present.
    events: List[Tuple[dt.datetime, float]] = []
    import os, io, csv

    # Derive region prefix for global AZ id filtering (fallback mode)
    region = filter_zone[:-1] if filter_zone else None  # 'us-east-1a' -> 'us-east-1'

    def parse_ts(s: str) -> dt.datetime:
        s = s.strip()
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        if s.isdigit():
            return dt.datetime.fromtimestamp(int(s), tz=dt.timezone.utc)
        try:
            return dt.datetime.fromisoformat(s)
        except Exception:
            return dt.datetime.fromisoformat(s.split('.')[0])

    # Build candidate file list constrained by time window
    paths: List[str] = []
    if os.path.isdir(archive_dir):
        # Collect year and month files intersecting [start_time, end_time]
        years = set()
        months = []
        y0, y1 = start_time.year, end_time.year
        for y in range(y0, y1 + 1):
            years.add(y)
        # Monthly files for years >= 2024
        cur = dt.datetime(start_time.year, start_time.month, 1)
        last = dt.datetime(end_time.year, end_time.month, 1)
        while cur <= last:
            months.append(cur.strftime('%Y-%m'))
            # Advance one month
            if cur.month == 12:
                cur = dt.datetime(cur.year + 1, 1, 1)
            else:
                cur = dt.datetime(cur.year, cur.month + 1, 1)
        for root, _, files in os.walk(archive_dir):
            for name in files:
                low = name.lower()
                if low.endswith('.tsv') or low.endswith('.csv'):
                    # Select if matches year rollup (e.g., 2023.tsv) or monthly within window (e.g., 2024-03.tsv)
                    stem = os.path.splitext(name)[0]
                    if stem.isdigit():
                        if int(stem) in years:
                            paths.append(os.path.join(root, name))
                    else:
                        if any(stem.startswith(m) for m in months):
                            paths.append(os.path.join(root, name))
    else:
        paths = [archive_dir]

    # Helper to normalize datetimes to epoch seconds
    def to_secs(d: dt.datetime) -> float:
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.timestamp()
    start_sec = to_secs(start_time)
    end_sec = to_secs(end_time)

    # Always allow seeding from a real event prior to window start
    allow_seed = True
    last_before: Optional[Tuple[dt.datetime, float]] = None
    last_before_sec: float = float('-inf')

    for p in paths:
        try:
            is_tsv = str(p).lower().endswith('.tsv')
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                first = f.readline()
                if not first:
                    continue
                # Header detection
                has_header = (
                    'zone' in first or 'AvailabilityZone' in first or 'ZoneId' in first or 'Timestamp' in first or 'timestamp' in first
                )
                # Reset file
                f.seek(0)
                if has_header:
                    reader = csv.DictReader(f, delimiter='\t' if is_tsv else ',')
                    for row in reader:
                        try:
                            z = str(
                                row.get('zone')
                                or row.get('ZoneId')
                                or row.get('AvailabilityZone')
                                or ''
                            ).strip()
                            it = str(row.get('instanceType') or row.get('InstanceType') or '').strip()
                            ts_s = str(row.get('timestamp') or row.get('Timestamp') or '').strip()
                            pr_s = str(row.get('price') or row.get('SpotPrice') or '').strip()
                            product = str(row.get('product') or row.get('ProductDescription') or '').strip()
                        except Exception:
                            continue
                        if filter_it and it != filter_it:
                            continue
                        # Strict exact AZ match using global ZoneId
                        if filter_zone_global_id and z != filter_zone_global_id:
                            continue
                        # Product filter if available
                        if product and product != 'Linux/UNIX':
                            continue
                        try:
                            ts = parse_ts(ts_s)
                            price = float(pr_s)
                        except Exception:
                            continue
                        tsec = to_secs(ts)
                        if tsec <= start_sec:
                            if allow_seed:
                                # Track the latest price not after the window start
                                if tsec > last_before_sec:
                                    last_before = (ts, price)
                                    last_before_sec = tsec
                            continue
                        if tsec > end_sec:
                            break
                        events.append((ts, price))
                else:
                    # No header: assume fixed columns TSV: zone, instanceType, product, price, timestamp
                    for line in f:
                        parts = line.rstrip('\n').split('\t') if is_tsv else line.rstrip('\n').split(',')
                        if len(parts) < 5:
                            continue
                        z, it, product, pr_s, ts_s = parts[0], parts[1], parts[2], parts[3], parts[4]
                        if filter_it and it != filter_it:
                            continue
                        if filter_zone_global_id and z != filter_zone_global_id:
                            continue
                        if product != 'Linux/UNIX':
                            continue
                        try:
                            ts = parse_ts(ts_s)
                            price = float(pr_s)
                        except Exception:
                            continue
                        # Fast path: skip before window, break after window (files are time-ordered)
                        tsec = to_secs(ts)
                        if tsec <= start_sec:
                            if allow_seed:
                                if tsec > last_before_sec:
                                    last_before = (ts, price)
                                    last_before_sec = tsec
                            continue
                        if tsec > end_sec:
                            break
                        events.append((ts, price))
        except Exception:
            continue

    # Ensure we have a valid starting price: seed with last known price before window start
    if allow_seed and last_before is not None:
        if not events or to_secs(events[0][0]) > start_sec:
            events.insert(0, last_before)
    events.sort(key=lambda x: x[0])
    return events


def fill_trace_prices(trace_path: str, region_dir_name: str) -> None:
    data = json.loads(open(trace_path, 'r').read())
    meta = data.get('metadata', {})
    gap_seconds = int(meta.get('gap_seconds'))
    start_time_str = meta.get('start_time') or meta.get('date') or meta.get('start')
    if not start_time_str:
        raise ValueError(f"Trace missing start_time-like metadata: {trace_path}")
    start_time = dt.datetime.fromisoformat(start_time_str)
    n_ticks = len(data.get('data', []))
    if n_ticks <= 0:
        raise ValueError(f"Trace has no data points: {trace_path}")

    zone, device = _split_region_dir_name(region_dir_name)
    aws_region = _aws_region_from_zone(zone)
    instance_type = DEVICE_TO_AWS_INSTANCE.get(device)
    if not instance_type:
        raise ValueError(f"Unsupported device for AWS instance mapping: {device}")

    end_time = start_time + dt.timedelta(seconds=n_ticks * gap_seconds)

    # Strict mode: do not fallback to any synthetic price
    default_price: Optional[float] = None

    # Strict AZ matching behavior (always on), resolve global AZ id via optional map file or boto3
    az_map_file = os.environ.get('SPOT_AZ_MAP_FILE')
    global_zone_id: Optional[str] = _resolve_global_az_id(zone, aws_region, az_map_file)
    if global_zone_id is None:
        raise RuntimeError(
            f"Failed to resolve global AZ id for {zone}. Provide SPOT_AZ_MAP_FILE or allow boto3 to query."
        )

    # Choose source: archive CSV or live AWS API
    archive_dir = os.environ.get('SPOT_ARCHIVE_CSV')
    if archive_dir:
        events = _load_archive_csv(
            archive_dir,
            filter_zone=zone,
            filter_it=instance_type,
            start_time=start_time,
            end_time=end_time,
            filter_zone_global_id=global_zone_id,
        )
    else:
        events = _fetch_history_boto3(
            aws_region=aws_region,
            availability_zone=zone,
            instance_type=instance_type,
            start_time=start_time,
            end_time=end_time,
        )
    prices = _interpolate_prices(
        events=events,
        start_time=start_time,
        gap_seconds=gap_seconds,
        num_ticks=n_ticks,
        default_price=default_price,
    )

    data['prices'] = prices
    with open(trace_path, 'w') as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(description="Fill trace JSON with AWS spot price series")
    parser.add_argument('--region-dir', required=True, help="Region directory name, e.g., us-east-1a_v100_1")
    parser.add_argument('--trace-files', nargs='+', required=True, help="Trace JSON files to fill")
    parser.add_argument('--archive-dir', type=str, default=None, help="Directory containing Eric Pauley spot price TSV/CSV files")
    parser.add_argument('--az-map-file', type=str, default=None, help="Optional ZoneName->ZoneId mapping file (TSV/CSV)")
    args = parser.parse_args()

    # Apply CLI overrides (env still supported but we prefer explicit CLI)
    if args.archive_dir:
        os.environ['SPOT_ARCHIVE_CSV'] = args.archive_dir
    if args.az_map_file:
        os.environ['SPOT_AZ_MAP_FILE'] = args.az_map_file

    for p in args.trace_files:
        fill_trace_prices(p, args.region_dir)
        print(f"Filled prices for {p}")


if __name__ == '__main__':
    main()


