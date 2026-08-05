import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts_multi.trace_sampling.convert_gcp_h100_spot import (
    GCPSpotPriceFetcher,
    convert,
)


class _DummyFetcher:
    def __init__(self, price: float = 2.5) -> None:
        self.price = price
        self.calls = {}
        now = datetime(2025, 9, 24, tzinfo=timezone.utc).isoformat()
        self._info = {
            "region": None,
            "machine_type": "a3-highgpu-8g",
            "currency": "USD",
            "price": price,
            "components": [
                {
                    "label": "instance",
                    "sku_id": "dummySpotSku",
                    "description": "Spot Preemptible a3-highgpu-8g",
                    "usage_unit": "h",
                    "unit_price": price,
                    "quantity": 1.0,
                    "component_total": price,
                    "effective_time": now,
                }
            ],
            "on_demand_price": price * 2,
            "effective_time": now,
            "fetched_at": now,
        }

    def prices_for_zone(self, zone: str, timestamps):  # pragma: no cover - exercised via convert
        self.calls[zone] = len(timestamps)
        info = dict(self._info)
        info["region"] = zone.rsplit('-', 1)[0]
        return [self.price] * len(timestamps), info


def _write_run(path: Path, scheduled_at: str, interval: int, zone_stats: dict) -> None:
    payload = {
        "scheduled_at": scheduled_at,
        "interval_sec": interval,
        "zone_results": zone_stats,
    }
    path.write_text(json.dumps(payload))


def test_convert_writes_constant_price_metadata(tmp_path):
    input_dir = tmp_path / "runs"
    input_dir.mkdir()
    interval = 600
    base = datetime(2025, 9, 20, 0, 0, tzinfo=timezone.utc)
    zones = ["us-central1-a", "us-central1-b"]
    stats = [
        {zones[0]: {"instances_created": 1}, zones[1]: {"instances_created": 0}},
        {zones[0]: {"instances_created": 0}, zones[1]: {"instances_created": 0}},
        {zones[0]: {"instances_created": 2}, zones[1]: {"instances_created": 1}},
    ]
    for idx, zone_stats in enumerate(stats):
        ts = (base.replace()) + idx * timedelta(seconds=interval)  # type: ignore[name-defined]
        # avoid importing timedelta at module scope just for the test helper
        scheduled_at = ts.isoformat()
        _write_run(input_dir / f"{idx:03d}.json", scheduled_at, interval, zone_stats)

    output_dir = tmp_path / "aligned"
    dummy_fetcher = _DummyFetcher(price=3.7)
    convert(
        input_dir=input_dir,
        output_base=output_dir,
        device_name="h100_8",
        machine_type="a3-highgpu-8g",
        api_key="unused",
        cache_dir=tmp_path / "cache",
        trace_length_hours=0.5,
        num_traces=1,
        seed=123,
        sku_filter=None,
        price_fetcher=dummy_fetcher,
    )

    for zone, expected in {
        "us-central1-a": [0, 1, 0],
        "us-central1-b": [1, 1, 0],
    }.items():
        region_dir = output_dir / f"{zone}_h100_8"
        full_path = region_dir / "full.json"
        window_path = region_dir / "0.json"
        assert full_path.exists()
        assert window_path.exists()

        full_obj = json.loads(full_path.read_text())
        window_obj = json.loads(window_path.read_text())

        assert full_obj["data"] == expected
        assert full_obj["prices"] == pytest.approx([dummy_fetcher.price] * len(expected))
        assert window_obj["prices"]
        assert len(window_obj["prices"]) == len(window_obj["data"])
        assert window_obj["prices"] == pytest.approx([dummy_fetcher.price] * len(window_obj["data"]))

        info = full_obj["metadata"]["price_info"]
        assert info["price"] == pytest.approx(dummy_fetcher.price)
        assert info.get("on_demand_price") == pytest.approx(dummy_fetcher.price * 2)
        assert info["region"] == zone.rsplit('-', 1)[0]
        assert "fetched_at" in info
        assert "effective_time" in info

    assert set(dummy_fetcher.calls.keys()) == set(zones)


def test_select_price_entry_prefers_latest_effective_time(tmp_path):
    fetcher = GCPSpotPriceFetcher(
        api_key="dummy",
        machine_type="a3-highgpu-8g",
        cache_dir=tmp_path / "cache",
    )

    earlier = {
        "skuId": "skuEarly",
        "description": "Spot Preemptible Instance",
        "pricingInfo": [
            {
                "effectiveTime": "2025-08-01T00:00:00Z",
                "currencyConversionRate": 1.0,
                "pricingExpression": {
                    "usageUnit": "h",
                    "tieredRates": [{"unitPrice": {"units": "3", "nanos": 500000000}}],
                },
            }
        ],
    }
    later = {
        "skuId": "skuLate",
        "description": "Spot Preemptible Instance",
        "pricingInfo": [
            {
                "effectiveTime": "2025-09-01T00:00:00Z",
                "currencyConversionRate": 1.0,
                "pricingExpression": {
                    "usageUnit": "h",
                    "tieredRates": [{"unitPrice": {"units": "2", "nanos": 750000000}}],
                },
            }
        ],
    }

    best = fetcher._select_price_entry([earlier, later])
    assert best is not None
    price, usage_unit, sku_id, _, effective = best
    assert sku_id == "skuLate"
    assert usage_unit == "h"
    assert price == pytest.approx(2.75)
    assert effective.isoformat() == "2025-09-01T00:00:00+00:00"
