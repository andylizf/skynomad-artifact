#!/usr/bin/env python3
"""Convert GCP H100 probing logs into TraceEnv-compatible aligned traces.

The converter expects the probing scheduler output (10 minute cadence JSON files)
and produces per-zone traces under ``data/converted_multi_region_aligned``. Each
trace contains ``data`` (1 = preempted, 0 = available) and a matching ``prices``
array populated from the Google Cloud Billing Catalog API. The API only exposes
the most recent list prices, so each tick in a trace uses the latest spot rate
(recorded in ``metadata.price_info.price``) along with the companion on-demand
rate (``metadata.price_info.on_demand_price``) for accurate billing downstream.

Usage example (regenerate the 13 shipped H100 zones; no network, no credentials)::

    .venv/bin/python scripts_multi/trace_sampling/convert_gcp_h100_spot.py \
        --input-dir data/h100_16_runs \
        --output-dir outputs/h100_regenerated \
        --device-name h100_16 \
        --machine-type a3-highgpu-8g

Add ``--gang-threshold 16`` to binarise a zone as available only when all 16
requested instances launched, instead of the shipped at-least-one definition.

Pricing: prices come from ``--cache-dir`` (default ``data/spot_price_cache/gcp``),
which ships with this artifact and covers every region in ``data/h100_16_runs``,
so the command above runs offline. ``--api-key`` (a Google Cloud Billing API key)
is needed only for a region that is not in the cache; the Catalog API exposes
only the current list price, so every tick of a trace carries the latest spot
rate, recorded in ``metadata.price_info``.

The script writes ``full.json`` plus a configurable number of fixed-length
windows (``0.json``, ``1.json`` ...) for each zone.
"""

import argparse
import dataclasses
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib import parse, request


# Google Cloud Billing Catalog service id for Compute Engine
_COMPUTE_SERVICE_ID = "6F81-5844-456A"


_MACHINE_SHAPES = {
    "a3-highgpu-8g": {
        "Preemptible": [
            {
                "label": "cpu",
                "match_any": [
                    "Spot Preemptible A3 Instance Core running in",
                ],
                "quantity": 96,
            },
            {
                "label": "ram",
                "match_any": [
                    "Spot Preemptible A3 Instance Ram running in",
                ],
                "quantity": 768,
            },
            {
                "label": "gpu",
                "match_any": [
                    "Nvidia H100 80GB GPU attached to Spot Preemptible VMs running in",
                ],
                "quantity": 8,
            },
        ],
        "OnDemand": [
            {
                "label": "cpu",
                "match_any": [
                    "A3 Instance Core running in",
                ],
                "quantity": 96,
            },
            {
                "label": "ram",
                "match_any": [
                    "A3 Instance Ram running in",
                ],
                "quantity": 768,
            },
            {
                "label": "gpu",
                "match_any": [
                    "Nvidia H100 80GB GPU attached to VMs running in",
                    "Nvidia H100 80GB GPU attached to A3 VMs running in",
                    "Nvidia H100 80GB GPU running in",
                ],
                "quantity": 8,
            },
        ],
    },
}


def _parse_timestamp(value: str) -> datetime:
    # The pricing API stamps UTC as a trailing "Z", which fromisoformat only
    # accepts from 3.11 on; normalise it so the 3.10 floor in pyproject holds.
    if value.endswith(("Z", "z")):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _price_from_unit_price(unit_price: Dict[str, object]) -> float:
    units = int(unit_price.get("units", "0"))
    nanos = int(unit_price.get("nanos", 0))
    return float(units) + nanos / 1_000_000_000


def _normalize_unit_price(price: float, usage_unit: str) -> float:
    unit = (usage_unit or "").lower()
    if not unit:
        return price
    if unit in {"s", "sec", "secs", "second", "seconds"}:
        return price * 3600.0
    if unit.endswith("/s") or unit.endswith("_s") or unit.endswith(".s"):
        return price * 3600.0
    if " per second" in unit:
        return price * 3600.0
    return price


@dataclasses.dataclass
class RunRecord:
    timestamp: datetime
    interval_seconds: int
    zone_results: Dict[str, Dict[str, object]]


@dataclasses.dataclass
class ZoneSeries:
    zone: str
    data: List[int]
    timestamps: List[datetime]

    def slice(self, start_tick: int, end_tick: int) -> "ZoneSeries":
        return ZoneSeries(
            zone=self.zone,
            data=self.data[start_tick:end_tick],
            timestamps=self.timestamps[start_tick:end_tick],
        )


class GCPSpotPriceFetcher:
    """Fetch the latest published GCP spot price for a given region.

    The Cloud Billing Catalog API only exposes the current list price for a
    preemptible SKU (refreshed roughly daily). Historical price deltas are not
    available, so the best we can do is retrieve the latest rate and assume it
    applies to the entire trace window. We record the metadata alongside the
    trace so downstream analyses are aware of the limitation.
    """

    def __init__(
        self,
        api_key: Optional[str],
        machine_type: str,
        cache_dir: Path,
        currency: str = "USD",
        sku_filter: Optional[str] = None,
        request_timeout: int = 30,
    ) -> None:
        # An API key is optional: cached prices are served without one. It only
        # becomes necessary for a region with no cache entry, which is where the
        # error is raised (see _price_entry_for_region).
        self._api_key = api_key
        self._machine_type = machine_type
        self._cache_dir = cache_dir
        self._currency = currency
        self._sku_filter = sku_filter
        self._timeout = request_timeout
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._price_cache: Dict[str, dict] = {}

    def prices_for_zone(
        self,
        zone: str,
        timestamps: Sequence[datetime],
    ) -> Tuple[List[float], dict]:
        region = zone.rsplit("-", 1)[0]
        entry = self._price_entry_for_region(region)
        price = float(entry["price"])
        return [price] * len(timestamps), entry

    def _price_entry_for_region(self, region: str) -> dict:
        if region in self._price_cache:
            return self._price_cache[region]

        cache_path = self._cache_dir / f"{region}_{self._machine_type}.json"
        cached = self._load_cache_entry(cache_path)
        if cached is None or "on_demand_price" not in cached:
            if not self._api_key:
                raise SystemExit(
                    f"No cached price for region={region} machine={self._machine_type} "
                    f"at {cache_path}, and no --api-key was supplied to fetch one. "
                    f"The cache shipped with this artifact covers every region in "
                    f"data/h100_16_runs; a new region needs a Google Cloud Billing "
                    f"API key (--api-key, e.g. $GCP_BILLING_API_KEY)."
                )
            cached = self._fetch_price_entry(region)
            cache_path.write_text(json.dumps(cached, indent=2))
        self._price_cache[region] = cached
        return cached

    def _load_cache_entry(self, cache_path: Path) -> Optional[dict]:
        if not cache_path.exists():
            return None
        try:
            entry = json.loads(cache_path.read_text())
        except Exception:
            return None
        if entry.get("machine_type") != self._machine_type:
            return None
        if entry.get("currency") != self._currency:
            return None
        if "price" not in entry:
            return None
        if "on_demand_price" not in entry:
            return None
        return entry

    def _fetch_price_entry(self, region: str) -> dict:
        fetched_at = datetime.now(timezone.utc).isoformat()
        shape = _MACHINE_SHAPES.get(self._machine_type, {})

        spot_skus = self._list_skus(region, usage_type="Preemptible")
        spot_total, spot_components = self._component_pricing_from_shape(
            region,
            spot_skus,
            shape.get("Preemptible") or shape.get("components"),
            usage_label="Preemptible",
        )
        if spot_total is None:
            candidate = self._select_price_entry(spot_skus)
            if candidate is None:
                raise RuntimeError(
                    f"Could not find preemptible pricing for machine={self._machine_type} region={region}. "
                    "Supply --sku-filter or populate cache manually."
                )
            unit_price, usage_unit, sku_id, description, effective = candidate
            normalized_price = _normalize_unit_price(unit_price, usage_unit)
            spot_total = normalized_price
            spot_components = [
                {
                    "label": "instance",
                    "sku_id": sku_id,
                    "description": description,
                    "usage_unit": usage_unit,
                    "billing_unit": "h",
                    "unit_price": normalized_price,
                    "raw_unit_price": unit_price,
                    "quantity": 1.0,
                    "component_total": normalized_price,
                    "effective_time": effective.isoformat(),
                }
            ]

        ondemand_skus = self._list_skus(region, usage_type="OnDemand")
        ondemand_total, ondemand_components = self._component_pricing_from_shape(
            region,
            ondemand_skus,
            shape.get("OnDemand"),
            usage_label="OnDemand",
        )
        if ondemand_total is None:
            candidate = self._select_price_entry(ondemand_skus)
            if candidate is not None:
                unit_price, usage_unit, sku_id, description, effective = candidate
                normalized_price = _normalize_unit_price(unit_price, usage_unit)
                ondemand_total = normalized_price
                ondemand_components = [
                    {
                        "label": "instance",
                        "sku_id": sku_id,
                        "description": description,
                        "usage_unit": usage_unit,
                        "billing_unit": "h",
                        "unit_price": normalized_price,
                        "raw_unit_price": unit_price,
                        "quantity": 1.0,
                        "component_total": normalized_price,
                        "effective_time": effective.isoformat(),
                    }
                ]

        entry = {
            "region": region,
            "machine_type": self._machine_type,
            "currency": self._currency,
            "price": float(spot_total),
            "price_unit": "h",
            "fetched_at": fetched_at,
        }
        if spot_components is not None:
            entry["components"] = spot_components
        if ondemand_total is not None:
            entry["on_demand_price"] = float(ondemand_total)
            entry["on_demand_price_unit"] = "h"
        if spot_components:
            try:
                latest = max(
                    spot_components, key=lambda comp: comp.get("effective_time", "")
                )
                if latest.get("effective_time"):
                    entry["effective_time"] = latest["effective_time"]
            except Exception:
                pass
        return entry

    def _component_pricing_from_shape(
        self,
        region: str,
        skus: Sequence[dict],
        components_cfg: Optional[Sequence[dict]],
        usage_label: str,
    ) -> Tuple[Optional[float], Optional[List[dict]]]:
        if not components_cfg:
            return None, None
        total_price = 0.0
        components_meta: List[dict] = []
        for component in components_cfg:
            patterns = []
            if component.get("match_any"):
                patterns.extend(component["match_any"])
            elif component.get("match"):
                patterns.append(component["match"])
            subset = (
                [
                    sku
                    for sku in skus
                    if any(
                        pattern in sku.get("description", "") for pattern in patterns
                    )
                ]
                if patterns
                else list(skus)
            )
            candidate = self._select_price_entry(subset) if subset else None
            if candidate is None:
                return None, None
            unit_price, usage_unit, sku_id, description, effective = candidate
            normalized_price = _normalize_unit_price(unit_price, usage_unit)
            quantity = float(component.get("quantity", 1.0))
            component_total = normalized_price * quantity
            total_price += component_total
            components_meta.append(
                {
                    "label": component.get("label") or usage_label.lower(),
                    "sku_id": sku_id,
                    "description": description,
                    "usage_unit": usage_unit,
                    "billing_unit": "h",
                    "raw_unit_price": unit_price,
                    "unit_price": normalized_price,
                    "quantity": quantity,
                    "component_total": component_total,
                    "effective_time": effective.isoformat(),
                }
            )
        return total_price, components_meta

    def _list_skus(self, region: str, usage_type: Optional[str] = None) -> List[dict]:
        params = {
            "currencyCode": self._currency,
            "pageSize": 500,
        }

        url = f"https://cloudbilling.googleapis.com/v1/services/{_COMPUTE_SERVICE_ID}/skus"
        all_skus: List[dict] = []
        page_token: Optional[str] = None
        while True:
            full_params = dict(params)
            if page_token:
                full_params["pageToken"] = page_token
            query = parse.urlencode(full_params)
            api_url = f"{url}?{query}&key={self._api_key}"
            req = request.Request(api_url, method="GET")
            with request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            all_skus.extend(payload.get("skus", []))
            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        def _matches_filters(sku: dict) -> bool:
            category = sku.get("category", {})
            usage = category.get("usageType")
            if usage_type:
                if usage != usage_type:
                    return False
            else:
                if usage != "Preemptible":
                    return False
            description = sku.get("description", "")
            if self._sku_filter and self._sku_filter not in description:
                return False
            return True

        filtered = [
            sku
            for sku in all_skus
            if _matches_filters(sku) and region in sku.get("serviceRegions", [])
        ]
        if not filtered and usage_type:
            filtered = [
                sku
                for sku in all_skus
                if region in sku.get("serviceRegions", [])
                and sku.get("category", {}).get("usageType") == usage_type
            ]
        if not filtered:
            filtered = [
                sku for sku in all_skus if region in sku.get("serviceRegions", [])
            ]
        return filtered or all_skus

    def _select_price_entry(
        self, skus: Sequence[dict]
    ) -> Optional[Tuple[float, str, str, str, datetime]]:
        best: Optional[Tuple[float, str, str, str, datetime]] = None
        for sku in skus:
            description = sku.get("description", "")
            sku_id = sku.get("skuId", "")
            pricing_infos = sku.get("pricingInfo", [])
            for info in pricing_infos:
                pricing_expr = info.get("pricingExpression", {})
                usage_unit = pricing_expr.get("usageUnit") or ""
                tiered = pricing_expr.get("tieredRates") or []
                if not tiered:
                    continue
                unit_price = tiered[0].get("unitPrice", {})
                price = _price_from_unit_price(unit_price)
                currency_rate = float(info.get("currencyConversionRate", 1.0))
                effective_raw = info.get("effectiveTime")
                if not effective_raw:
                    continue
                try:
                    effective = _parse_timestamp(effective_raw)
                except Exception:
                    continue
                candidate = (
                    price * currency_rate,
                    usage_unit,
                    sku_id,
                    description,
                    effective,
                )
                if best is None or candidate[-1] > best[-1]:
                    best = candidate
        return best


def load_run_records(input_dir: Path) -> List[RunRecord]:
    run_paths = sorted(input_dir.glob("*.json"))
    if not run_paths:
        raise FileNotFoundError(f"No run files found under {input_dir}")
    records: List[RunRecord] = []
    for path in run_paths:
        obj = json.loads(path.read_text())
        timestamp = _parse_timestamp(obj["scheduled_at"])
        interval = int(obj["interval_sec"])
        zone_results = obj.get("zone_results")
        if not isinstance(zone_results, dict):
            raise ValueError(f"Malformed zone_results in {path}")
        records.append(
            RunRecord(
                timestamp=timestamp,
                interval_seconds=interval,
                zone_results=zone_results,
            )
        )
    return records


def build_zone_series(
    records: Sequence[RunRecord], gang_threshold: int = 1
) -> Dict[str, ZoneSeries]:
    """Binarise each probe into available (0) / preempted (1).

    `gang_threshold` is how many of the requested instances must have launched
    for the tick to count as available. The traces shipped with this artifact use
    the default of 1.
    """
    if gang_threshold < 1:
        raise ValueError(f"gang_threshold must be >= 1, got {gang_threshold}")
    zones: Dict[str, List[int]] = {}
    timestamps = [rec.timestamp for rec in records]
    for rec in records:
        for zone, stats in rec.zone_results.items():
            data_list = zones.setdefault(zone, [])
            instances_created = int(stats.get("instances_created", 0))
            data_list.append(0 if instances_created >= gang_threshold else 1)
    return {
        zone: ZoneSeries(zone=zone, data=data, timestamps=list(timestamps))
        for zone, data in zones.items()
    }


def write_trace(
    path: Path,
    series: ZoneSeries,
    prices: List[float],
    gap_seconds: int,
    source_dir: Path,
    price_info: dict,
) -> None:
    if len(prices) != len(series.data):
        raise ValueError(
            f"Price length {len(prices)} does not match data length {len(series.data)} for {series.zone}"
        )
    metadata = {
        "gap_seconds": gap_seconds,
        "start_time": series.timestamps[0].isoformat(),
        "source_dir": str(source_dir),
        "zone": series.zone,
        "price_info": price_info,
    }
    payload = {"metadata": metadata, "data": series.data, "prices": prices}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def compute_windows(
    total_ticks: int,
    gap_seconds: int,
    window_hours: float,
    num_windows: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    window_ticks = max(1, int(round(window_hours * 3600 / gap_seconds)))
    if window_ticks > total_ticks:
        return [(0, total_ticks)]
    max_start = total_ticks - window_ticks
    if max_start <= 0:
        return [(0, total_ticks)]
    if num_windows <= 0:
        return [(0, window_ticks)]
    starts = sorted({rng.randint(0, max_start) for _ in range(num_windows)})
    windows = [(s, s + window_ticks) for s in starts]
    return windows


def convert(
    input_dir: Path,
    output_base: Path,
    device_name: str,
    machine_type: str,
    api_key: Optional[str],
    cache_dir: Path,
    trace_length_hours: float,
    num_traces: int,
    seed: int,
    sku_filter: Optional[str] = None,
    price_fetcher: Optional["GCPSpotPriceFetcher"] = None,
    gang_threshold: int = 1,
    print_stats: bool = False,
) -> None:
    records = load_run_records(input_dir)
    if not records:
        raise RuntimeError("No run records to convert")
    gap_seconds = records[0].interval_seconds
    zone_series = build_zone_series(records, gang_threshold=gang_threshold)

    if print_stats:
        # The availability split quoted in README.md ("76.7% at gang-threshold
        # 1, 43.6% at 16"). Without this the script writes traces and prints
        # nothing, so the README's "both numbers are reproducible offline"
        # required the reader to write their own aggregation.
        #
        # `data` is a PREEMPTION series: 0 means the zone was available for that
        # tick, 1 means it was not. Availability is therefore the count of zeros.
        total_ticks = sum(len(s.data) for s in zone_series.values())
        available_ticks = sum(
            sum(1 for v in s.data if v == 0) for s in zone_series.values()
        )
        pct = (100.0 * available_ticks / total_ticks) if total_ticks else 0.0
        print(f"zones={len(zone_series)} gang-threshold={gang_threshold}")
        print(
            f"zone-ticks={total_ticks} available={available_ticks} "
            f"({pct:.1f}%) at gang-threshold={gang_threshold}"
        )
        for zone in sorted(zone_series):
            s = zone_series[zone]
            n = len(s.data)
            a = sum(1 for v in s.data if v == 0)
            print(f"  {zone:<22} ticks={n:<6} available={a:<6} ({100.0*a/n:.1f}%)")
    fetcher = price_fetcher or GCPSpotPriceFetcher(
        api_key=api_key,
        machine_type=machine_type,
        cache_dir=cache_dir,
        sku_filter=sku_filter,
    )
    rng = random.Random(seed)

    for zone, series in zone_series.items():
        prices, price_info = fetcher.prices_for_zone(zone, series.timestamps)
        region_dir = output_base / f"{zone}_{device_name}"
        write_trace(
            region_dir / "full.json", series, prices, gap_seconds, input_dir, price_info
        )

        windows = compute_windows(
            len(series.data), gap_seconds, trace_length_hours, num_traces, rng
        )
        for idx, (start_tick, end_tick) in enumerate(windows):
            window_series = series.slice(start_tick, end_tick)
            window_prices = prices[start_tick:end_tick]
            write_trace(
                region_dir / f"{idx}.json",
                window_series,
                window_prices,
                gap_seconds,
                input_dir,
                price_info,
            )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert GCP H100 probing runs to aligned traces"
    )
    parser.add_argument(
        # The probe records that ship with this artifact. The old default
        # (data/h100x8_5days_8region/runs) is a directory that does not exist
        # here, so anyone following the README hit a missing-path error.
        "--input-dir", type=Path, default=Path("data/h100_16_runs")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/converted_multi_region_aligned")
    )
    parser.add_argument("--device-name", default="h100_8")
    parser.add_argument("--machine-type", default="a3-highgpu-8g")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google Cloud Billing API key. Only needed for a region whose price "
             "is not already in --cache-dir; the cache shipped with this artifact "
             "covers all 13 zones in data/h100_16_runs, so the default run needs "
             "no key and no network.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/spot_price_cache/gcp")
    )
    parser.add_argument(
        "--gang-threshold",
        type=int,
        default=1,
        help="Instances that must launch for a tick to count as available. The "
             "shipped traces use 1 (at least one of 16); pass 16 for the strict "
             "gang definition. See README.md.",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print the aggregate zone-tick availability split (total ticks, "
             "available ticks, percentage) for the chosen --gang-threshold, plus "
             "a per-zone breakdown. This is what reproduces the 76.7%% / 43.6%% "
             "numbers quoted in README.md.",
    )
    parser.add_argument("--trace-length-hours", type=float, default=52.0)
    parser.add_argument("--num-traces", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20250924)
    parser.add_argument(
        "--sku-filter",
        default=None,
        help="Additional filter clause for Cloud Billing Catalog (ANDed to base filter)",
    )
    args = parser.parse_args(argv)

    # No hard --api-key requirement: prices come from --cache-dir when a cached
    # entry exists, and the cache for every region in the shipped probe set is
    # part of this artifact. A key is only needed for a region that is not
    # cached, and _price_entry_for_region raises a clear error in that case.

    convert(
        input_dir=args.input_dir,
        output_base=args.output_dir,
        device_name=args.device_name,
        machine_type=args.machine_type,
        api_key=args.api_key,
        gang_threshold=args.gang_threshold,
        print_stats=args.print_stats,
        cache_dir=args.cache_dir,
        trace_length_hours=args.trace_length_hours,
        num_traces=args.num_traces,
        seed=args.seed,
        sku_filter=args.sku_filter,
    )


if __name__ == "__main__":
    main()
