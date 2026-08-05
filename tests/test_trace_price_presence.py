"""The aligned V100 traces must carry prices the simulator can actually use.

This test used to `continue` past any trace that had no price data at all, which
made it pass on exactly the defect it is named for: `sky_spot/env.py` has no
price fallback, so a price-less trace looked fine here and then made every
simulation over it raise `ValueError: No on_demand_price in price_info`. The
check below is the one that matters -- it constructs a real `TraceEnv` per
region, so it fails with the same error the simulator would produce.
"""

import json
import os
import sys
from glob import glob

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_all_aligned_traces_have_prices():
    base = os.path.join(REPO_ROOT, "data", "converted_multi_region_aligned")
    if not os.path.isdir(base):
        pytest.skip("run artifact/prepare_data.sh first")

    from sky_spot.env import TraceEnv

    paths = sorted(glob(os.path.join(base, "*", "*.json")))
    assert paths, f"no traces found under {base}"

    bad = []
    for path in paths:
        try:
            with open(path, "r") as f:
                obj = json.load(f)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{path}: unreadable ({exc})")
            continue

        data = obj.get("data")
        if not isinstance(data, list) or not data:
            bad.append(f"{path}: no trace data")
            continue

        price_info = (obj.get("metadata") or {}).get("price_info")
        if not isinstance(price_info, dict):
            bad.append(f"{path}: no metadata.price_info")
            continue
        for key in ("on_demand_price", "price"):
            value = price_info.get(key)
            if not isinstance(value, (int, float)) or value <= 0:
                bad.append(f"{path}: price_info[{key!r}]={value!r}")

        # A per-tick price array is optional, but if one is present it must line
        # up with the availability trace or billing reads the wrong tick.
        prices = obj.get("prices")
        if prices is not None and (
            not isinstance(prices, list) or len(prices) != len(data)
        ):
            bad.append(f"{path}: prices length {len(prices)} != data length {len(data)}")

    assert not bad, f"{len(bad)} trace(s) unusable, e.g. {bad[:5]}"

    # The assertion that actually mirrors the simulator: every region's full
    # trace must construct without raising.
    for full in sorted(glob(os.path.join(base, "*", "full.json"))):
        TraceEnv(full, 0.0, None)
