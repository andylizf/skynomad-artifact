"""Scenario configuration module - self-contained."""

import os
from collections.abc import Sequence
from pathlib import Path

from benchmark_components.trace_utils import (
    normalize_data_roots,
    resolve_region_trace_path,
)

# Multi-region strategies (run on multi_trace environment)
MULTI_REGION_STRATEGIES = [
    "multi_region_rc_cr_threshold",
    "unified_cost_model_oracle",
    "multi_region_oracle_dp",
    "unified_cost_model_risk",
    "unified_cost_model_rate_ratio",
    "unified_cost_model_rate_avail",
    "unified_cost_model_ramp_v",
    "unified_cost_model_rate_past",
    "unified_cost_model_risk_legacy",
]

# Segment visualization defaults (can be overridden per-scenario by adding
# 'segment_viz_strategies' in EXPERIMENT_SCENARIOS entries)
SEGMENT_VIZ_STRATEGIES = [
    "unified_cost_model_risk",
    # "multi_region_oracle_dp",
    "multi_region_rc_cr_threshold",
    "unified_cost_model_oracle",
    "unified_cost_model_rate_ratio",
    "unified_cost_model_rate_avail",
    "unified_cost_model_ramp_v",
    "unified_cost_model_rate_past",
    "unified_cost_model_risk_legacy",
]


# Union pool strategies (run on union trace - single trace of all regions combined)
UNION_POOL_STRATEGIES = []

# Single-region strategies with aggregation mode: (strategy_name, mode)
# Each entry creates one baseline - add multiple entries for multiple baselines
SINGLE_REGION_STRATEGIES = [
    ("rc_cr_threshold", "best"),
    ("rc_cr_threshold", "average"),
]

# ============================================================================
# PAPER BASELINE STRATEGIES (shared across configs)
# ============================================================================
PAPER_STRATEGIES = [
    # Our method
    "unified_cost_model_risk",
    # Oracle baselines.
    #
    # This must be `multi_region_oracle_dp`, not the `..._safety_net` variant:
    # the shipped research/data/*.csv were produced with the former, the
    # plotting scripts look up the Optimal series by that exact name (and used
    # to drop it silently when a sweep emitted the other one), and the two are
    # different policies -- the safety-net variant restricts DP transitions at
    # the safety-net boundary. Verified on the H100 Global Diverse (6) rows of
    # deadline_sensitivity.csv: `multi_region_oracle_dp` reproduces the shipped
    # costs to the last digit, `..._safety_net` is up to 0.1% above them.
    "multi_region_oracle_dp",  # DP (Oracle)
    "unified_cost_model_oracle",  # Next Spot Oracle
    # Multi-region baselines
    "multi_region_rc_cr_threshold_eager_failover",  # Eager Failover
    "multi_region_availability_probe_simple",  # Probe (Avail)
    "multi_region_probe_cost_ratio",  # Probe (Cost)
]

DEFAULT_CHECKPOINT_SIZE = 50.0
DEFAULT_RESTART_OVERHEAD = 0.2

US_EAST_1_REGIONS = [
    "us-east-1a_v100_1",
    "us-east-1c_v100_1",
    "us-east-1d_v100_1",
    "us-east-1f_v100_1",
]

US_EAST_2_REGIONS = ["us-east-2a_v100_1", "us-east-2b_v100_1"]

US_WEST_2_REGIONS = ["us-west-2a_v100_1", "us-west-2b_v100_1", "us-west-2c_v100_1"]

# Batch comparison default regions (control set used by batch_strategy_comparison.py)
BATCH_DEFAULT_REGIONS = [
    "us-east-1a_v100_1",
    "us-east-1d_v100_1",
    "us-east-1c_v100_1",
    "us-east-1f_v100_1",
    "us-east-2a_v100_1",
]


GCP_A3_H100_REGIONS = [
    # "asia-east1-c_h100_8",
    # "asia-northeast1-b_h100_8",
    "europe-west1-c_h100_8",
    # "europe-west4-c_h100_8",
    "us-central1-a_h100_8",
    "us-central1-b_h100_8",
    "us-east4-a_h100_8",
    "us-west1-a_h100_8",
]


def with_suffix(accerlerator: str, num_nodes: int, regions: list[str]):
    def _suffix(region: str):
        return f"{region}_{accerlerator}_{num_nodes}"

    return [_suffix(region) for region in regions]


GCP_A3_H100_16_REGIONS = [
    "asia-south2-b_h100_16",
    "asia-southeast1-b_h100_16",
    "asia-southeast1-c_h100_16",
    "europe-west1-c_h100_16",
    "europe-west3-c_h100_16",
    "us-central1-a_h100_16",
    "us-central1-b_h100_16",
    "us-central1-c_h100_16",
    "us-east4-a_h100_16",
    "us-east4-b_h100_16",
    "us-east5-a_h100_16",
    "us-west1-a_h100_16",
    "us-west1-b_h100_16",
]

VOLATILE_LOW_GCP_A3_H100_REGIONS = with_suffix(
    "h100",
    16,
    [
        "asia-southeast1-c",
        "us-west1-b",
    ],
)

VOLATILE_MIDDLE_GCP_A3_H100_REGIONS = with_suffix(
    "h100",
    16,
    [
        "europe-west1-c",
        "europe-west3-c",
        "us-west1-a",
    ],
)

STABLE_HIGH_GCP_A3_H100_REGION = with_suffix(
    "h100",
    16,
    [
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
        "us-east4-a",
        "us-east4-b",
        "us-east5-a",
    ],
)

US_EAST_WEST_REGIONS = with_suffix(
    "h100",
    16,
    [
        "us-east4-a",
        "us-east4-b",
        "us-east5-a",
        "us-west1-a",
        "us-west1-b",
    ],
)

# All US regions (Central + East + West)
ALL_US_REGIONS = with_suffix(
    "h100",
    16,
    [
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
        "us-east4-a",
        "us-east4-b",
        "us-east5-a",
        "us-west1-a",
        "us-west1-b",
    ],
)

# All Europe regions
EUROPE_REGIONS = with_suffix(
    "h100",
    16,
    [
        "europe-west1-c",
        "europe-west3-c",
    ],
)

# All Asia regions
ASIA_REGIONS = with_suffix(
    "h100",
    16,
    [
        "asia-south2-b",
        "asia-southeast1-b",
        "asia-southeast1-c",
    ],
)

ALL_AVAIL_BUT_FAR_GCP_A3_H100_REGION = with_suffix(
    "h100",
    16,
    [
        "asia-south2-b",
        "asia-southeast1-b",
    ],
)

# ============================================================================
# DATA SOURCES
# ============================================================================
# 1. MERGED H100_16 (GCP) - 352h of data
#    Path: data/converted_multi_region_aligned_h100_16_merged/
#    Source: GCP A3 H100 spot instances (16-node clusters)
#    Regions: us-central1, us-east4, us-east5, us-west1, europe-west1/3, asia-southeast1, asia-south2
#    Note: First 116h matches OLD_DATA, extended with additional 236h of more stable data
#    Availability: Generally high (88-100%), good for production scenarios
#
# 2. OLD DATA (GCP H100 + AWS V100) - mixed duration
#    Path: data/converted_multi_region_aligned/
#    Contains:
#      - H100_16 (GCP): 116h, 13 regions - MORE VOLATILE than merged data
#      - H100_8 (GCP): 120h, 8 regions
#      - V100_1 (AWS): 1092h (45 days), 9 regions - longest traces, very volatile
#
#    H100 regions (GCP): us-central1, us-east4/5, us-west1, europe-west1/3/4, asia-*
#    V100 regions (AWS): us-east-1a/c/d/f, us-east-2a/b, us-west-2a/b/c
#
#    Key insight: The first 116h of merged data == old H100_16 data
#                 Old data is more volatile, better for strategy differentiation
# ============================================================================

MERGED_A100_DATA_PATH = "data/converted_multi_region_aligned_h100_16_merged"

# Old data with more volatile traces.
#
# As shipped in this artifact this directory holds the NINE V100 zones only --
# artifact/prepare_data.sh regenerates them from data/real/availability. The
# 116h H100_16 snapshot that used to live here is not part of the artifact; the
# scenarios below that name h100_16 regions therefore read MERGED_A100_DATA_PATH
# instead, whose first 116h are the same probes (see the header comment above).
VOLATILE_OLD_DATA_PATH = "data/converted_multi_region_aligned"

# ============================================================================
# Progressive region list - ordered by SINGLE-REGION COST (HIGH to LOW)
#
# This is the ordering behind research/data/region_scaling.csv and Figure 10b:
# each step down the list adds the next cheapest zone to run alone in.
# PROGRESSIVE_REGIONS_BY_AVAILABILITY below is the alternative axis, selectable
# with PROGRESSIVE_REGION_ORDER=availability.
#
# Based on actual UP (single-region) benchmark results at deadline_ratio=1.2:
#   1. asia-southeast1-c  ($4341) - worst single-region cost
#   2. asia-south2-b      ($4206) - 100% availability, 4x the cheapest price
#   3. us-west1-b         ($4127)
#   4. europe-west3-c     ($2978)
#   5. us-west1-a         ($2210)
#   6. us-east4-a         ($1732)
#   7. europe-west1-c     ($1405)
#   8. us-central1-c      ($1323)
#   9. us-central1-b      ($1309)
#  10. asia-southeast1-b  ($1272)
#  11. us-central1-a      ($1248)
#  12. us-east5-a         ($1124)
#  13. us-east4-b         ($1078) - best single-region cost
# ============================================================================
PROGRESSIVE_REGIONS_BY_SINGLE_COST = with_suffix(
    "h100",
    16,
    [
        "asia-southeast1-c",  # $4341 - worst single-region cost
        "asia-south2-b",      # $4206 - 100% avail, 4x the cheapest price
        "us-west1-b",         # $4127
        "europe-west3-c",     # $2978
        "us-west1-a",         # $2210
        "us-east4-a",         # $1732
        "europe-west1-c",     # $1405
        "us-central1-c",      # $1323
        "us-central1-b",      # $1309
        "asia-southeast1-b",  # $1272
        "us-central1-a",      # $1248
        "us-east5-a",         # $1124
        "us-east4-b",         # $1078 - best single-region cost
    ],
)

# ============================================================================
# Progressive region list - ordered by AVERAGE SPOT AVAILABILITY (HIGH to LOW)
#
# The alternative axis ordering for the region-scaling sweep. Select it with:
#
#   PROGRESSIVE_REGION_ORDER=availability \
#   python \
#     scripts_multi/benchmark_multi_region_modular.py \
#     --config region_scaling --output-dir outputs/rerun_availability_order
#
# Availability is measured over the shipped merged traces
# (data/converted_multi_region_aligned_h100_16_merged, 2114 ticks per zone,
# availability = fraction of ticks with data[i] == 0, since the trace records
# preemption not availability).
# ============================================================================
PROGRESSIVE_REGIONS_BY_AVAILABILITY = with_suffix(
    "h100",
    16,
    [
        "asia-south2-b",      # 100.0% available
        "us-central1-a",      #  95.8%
        "us-central1-c",      #  91.9%
        "us-central1-b",      #  91.2%
        "us-east5-a",         #  89.0%
        "europe-west1-c",     #  88.0%
        "us-east4-a",         #  86.2%
        "us-east4-b",         #  84.8%
        "asia-southeast1-b",  #  78.4%
        "us-west1-a",         #  60.0%
        "asia-southeast1-c",  #  54.4%
        "europe-west3-c",     #  47.4%
        "us-west1-b",         #  24.1%
    ],
)

PROGRESSIVE_REGION_ORDERS = {
    "single_cost": PROGRESSIVE_REGIONS_BY_SINGLE_COST,
    "availability": PROGRESSIVE_REGIONS_BY_AVAILABILITY,
}

# Which ordering the progressive scenarios use. "single_cost" is the default and
# what research/data/region_scaling.csv holds; set
# PROGRESSIVE_REGION_ORDER=availability in the environment to switch.
PROGRESSIVE_ORDER = os.environ.get("PROGRESSIVE_REGION_ORDER", "single_cost")
if PROGRESSIVE_ORDER not in PROGRESSIVE_REGION_ORDERS:
    raise ValueError(
        f"PROGRESSIVE_REGION_ORDER={PROGRESSIVE_ORDER!r} is not one of "
        f"{sorted(PROGRESSIVE_REGION_ORDERS)}"
    )
_PROGRESSIVE_REGIONS = PROGRESSIVE_REGION_ORDERS[PROGRESSIVE_ORDER]
_PROGRESSIVE_ORDER_DESC = {
    "single_cost": "single-region cost (high to low)",
    "availability": "average spot availability (high to low)",
}[PROGRESSIVE_ORDER]

# Progressive scenarios: 1 region -> 13 regions
PROGRESSIVE_EXPERIMENT_SCENARIOS = [
    {
        "name": f"Progressive_{i+1}",
        "regions": _PROGRESSIVE_REGIONS[:i+1],
        "description": f"Progressive scaling: {i+1} region(s), ordered by {_PROGRESSIVE_ORDER_DESC}",
        "data_paths": (MERGED_A100_DATA_PATH,),
    }
    for i in range(13)
]

# ============================================================================
# Single-region mode toggle
# When True, test multi-region strategies with only one region each
# ============================================================================
SINGLE_REGION_MODE = False

# Single-region experiment scenarios - one scenario per region
SINGLE_REGION_EXPERIMENT_SCENARIOS = [
    {
        "name": f"Single: {region}",
        "regions": [region],
        "description": f"Single-region test with {region} only",
    }
    for region in GCP_A3_H100_16_REGIONS
]

# Multi-region experiment scenarios
MULTI_REGION_EXPERIMENT_SCENARIOS = [
    # === Full coverage ===
    {
        "name": "All 13 Regions",
        "regions": GCP_A3_H100_16_REGIONS,
        "description": "All thirteen GCP H100 16-node regions",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    # === By stability ===
    {
        "name": "Volatile Only (5)",
        "regions": VOLATILE_MIDDLE_GCP_A3_H100_REGIONS
        + VOLATILE_LOW_GCP_A3_H100_REGIONS,
        "description": "All volatile regions (low + middle availability)",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Stable Only (6)",
        "regions": STABLE_HIGH_GCP_A3_H100_REGION,
        "description": "Stable US regions with high availability",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Volatile + Stable (11)",
        "regions": VOLATILE_MIDDLE_GCP_A3_H100_REGIONS
        + VOLATILE_LOW_GCP_A3_H100_REGIONS
        + STABLE_HIGH_GCP_A3_H100_REGION,
        "description": "Mix of volatile and stable regions",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Volatile Low Only (2)",
        "regions": VOLATILE_LOW_GCP_A3_H100_REGIONS,
        "description": "Only the most volatile regions",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Volatile Middle Only (3)",
        "regions": VOLATILE_MIDDLE_GCP_A3_H100_REGIONS,
        "description": "Middle-tier volatile regions",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    # === By geography ===
    {
        "name": "All US (8)",
        "regions": ALL_US_REGIONS,
        "description": "All US regions (Central + East + West)",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "US Central Only (3)",
        "regions": with_suffix(
            "h100", 16, ["us-central1-a", "us-central1-b", "us-central1-c"]
        ),
        "description": "US Central regions only",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "US East Only (3)",
        "regions": with_suffix("h100", 16, ["us-east4-a", "us-east4-b", "us-east5-a"]),
        "description": "US East regions only",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "US West Only (2)",
        "regions": with_suffix("h100", 16, ["us-west1-a", "us-west1-b"]),
        "description": "US West regions only",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "US East + West (5)",
        "regions": US_EAST_WEST_REGIONS,
        "description": "US East and West regions (no Central)",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Europe Only (2)",
        "regions": EUROPE_REGIONS,
        "description": "European regions only",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Asia Only (3)",
        "regions": ASIA_REGIONS,
        "description": "Asian regions only",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "US + Europe (10)",
        "regions": ALL_US_REGIONS + EUROPE_REGIONS,
        "description": "US and European regions",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "US + Asia (11)",
        "regions": ALL_US_REGIONS + ASIA_REGIONS,
        "description": "US and Asian regions",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Europe + Asia (5)",
        "regions": EUROPE_REGIONS + ASIA_REGIONS,
        "description": "European and Asian regions (non-US)",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    # === Cross-continental with different sizes ===
    {
        "name": "Global Diverse (6)",
        "regions": with_suffix(
            "h100",
            16,
            [
                "us-central1-a",
                "us-west1-a",  # US
                "europe-west1-c",
                "europe-west3-c",  # Europe
                "asia-southeast1-b",
                "asia-south2-b",  # Asia
            ],
        ),
        "description": "2 regions from each continent",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Global Minimal (3)",
        "regions": with_suffix(
            "h100",
            16,
            [
                "us-central1-a",  # US (stable)
                "europe-west1-c",  # Europe (volatile)
                "asia-southeast1-b",  # Asia (available)
            ],
        ),
        "description": "1 region from each continent",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    # === Special combinations ===
    {
        "name": "Volatile + Far Available (7)",
        "regions": VOLATILE_MIDDLE_GCP_A3_H100_REGIONS
        + VOLATILE_LOW_GCP_A3_H100_REGIONS
        + ALL_AVAIL_BUT_FAR_GCP_A3_H100_REGION,
        "description": "Volatile regions plus distant high-availability regions",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "High Availability Far (2)",
        "regions": ALL_AVAIL_BUT_FAR_GCP_A3_H100_REGION,
        "description": "Far but highly available Asian regions",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Stable + Far Available (8)",
        "regions": STABLE_HIGH_GCP_A3_H100_REGION
        + ALL_AVAIL_BUT_FAR_GCP_A3_H100_REGION,
        "description": "US stable + Asia high-availability",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    # === Size variations (for scaling analysis) ===
    {
        "name": "Size 4 Mixed",
        "regions": with_suffix(
            "h100",
            16,
            [
                "us-central1-a",  # stable
                "us-west1-b",  # volatile
                "europe-west1-c",  # volatile
                "asia-southeast1-b",  # available
            ],
        ),
        "description": "4 regions with mixed stability",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Size 6 US-focused",
        "regions": with_suffix(
            "h100",
            16,
            [
                "us-central1-a",
                "us-central1-b",
                "us-east4-a",
                "us-east4-b",
                "us-west1-a",
                "us-west1-b",
            ],
        ),
        "description": "6 US regions across all coasts",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Size 9 Balanced",
        "regions": with_suffix(
            "h100",
            16,
            [
                "us-central1-a",
                "us-central1-b",
                "us-central1-c",  # 3 central
                "us-east4-a",
                "us-west1-a",  # 2 coasts
                "europe-west1-c",
                "europe-west3-c",  # 2 europe
                "asia-southeast1-b",
                "asia-south2-b",  # 2 asia
            ],
        ),
        "description": "9 regions balanced across geographies",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    # === Very low availability scenarios ===
    {
        "name": "Worst 3 Regions",
        "regions": with_suffix(
            "h100",
            16,
            [
                "asia-southeast1-b",  # ~3%
                "us-west1-b",  # ~22%
                "europe-west3-c",  # ~25%
            ],
        ),
        "description": "3 regions with lowest availability (<30%)",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Worst + 1 Stable (4)",
        "regions": with_suffix(
            "h100",
            16,
            [
                "asia-southeast1-b",  # ~3%
                "us-west1-b",  # ~22%
                "europe-west3-c",  # ~25%
                "us-central1-a",  # ~100% stable fallback
            ],
        ),
        "description": "3 worst regions + 1 stable fallback",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Low Avail + Far Stable (5)",
        "regions": with_suffix(
            "h100",
            16,
            [
                "asia-southeast1-b",  # ~3%
                "us-west1-b",  # ~22%
                "europe-west3-c",  # ~25%
                "asia-south2-b",  # 100% but far
                "us-central1-a",  # ~100% stable
            ],
        ),
        "description": "3 worst + 2 stable (near and far)",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
]

# Main experiment scenario (6 regions, 2 from each continent)
GLOBAL_DIVERSE_SCENARIO = {
    "name": "Global Diverse (6)",
    "regions": with_suffix(
        "h100",
        16,
        [
            "us-central1-a",
            "us-west1-a",  # US
            "europe-west1-c",
            "europe-west3-c",  # Europe
            "asia-southeast1-b",
            "asia-south2-b",  # Asia
        ],
    ),
    "description": "2 regions from each continent",
    "data_paths": (MERGED_A100_DATA_PATH,),
}

# Volatile scenario for V-ablation (uses old data with more preemptions)
# Regions sorted by availability (low to high):
#   europe-west3-c: 16.2% avail, 12.2h max run
#   us-west1-b: 24.0% avail, 3.7h max run
#   asia-southeast1-b: 34.6% avail, 35.8h max run
#   us-west1-a: 38.1% avail, 27.0h max run
#   asia-southeast1-c: 75.4% avail, 46.8h max run
#   us-central1-a: 97.3% avail, 40.7h max run (stable fallback)
# The two scenarios below name h100_16 zones. Their availability annotations
# used to be the ones measured on the 116h VOLATILE_OLD_DATA_PATH snapshot,
# which this artifact does not ship; both now read MERGED_A100_DATA_PATH and the
# annotations are re-measured on those traces. Pointing them at
# VOLATILE_OLD_DATA_PATH builds zero simulation tasks.
VOLATILE_GLOBAL_SCENARIO = {
    "name": "Volatile Global (6)",
    "regions": with_suffix(
        "h100",
        16,
        [
            "us-west1-b",          # 24.1% avail - most volatile
            "europe-west3-c",      # 47.4% avail
            "asia-southeast1-c",   # 54.4% avail
            "us-west1-a",          # 60.0% avail
            "asia-southeast1-b",   # 78.4% avail
            "us-central1-a",       # 95.8% avail - stable fallback
        ],
    ),
    "description": "6 regions with high volatility (24-96% availability)",
    "data_paths": (MERGED_A100_DATA_PATH,),
}

# All-volatile scenario (no stable fallback) for V-ablation differentiation.
# On the merged traces these are the four least available zones outside
# us-central1; none of them has a long stable stretch, so preemptions are
# frequent. (On the unshipped 116h snapshot all four were below 40%
# availability, which is where the old "<40%" description came from.)
VOLATILE_NO_FALLBACK_SCENARIO = {
    "name": "Volatile No Fallback (4)",
    "regions": with_suffix(
        "h100",
        16,
        [
            "us-west1-b",          # 24.1% avail, 30.0h max run
            "europe-west3-c",      # 47.4% avail, 112.0h max run
            "asia-southeast1-c",   # 54.4% avail, 104.2h max run
            "us-west1-a",          # 60.0% avail, 31.8h max run
        ],
    ),
    "description": "4 volatile regions, no stable fallback (24-60% availability each)",
    "data_paths": (MERGED_A100_DATA_PATH,),
    "segment_viz_strategies": [
        "multi_region_rc_cr_threshold",
        "unified_cost_model_rate_past",
        "unified_cost_model_rate_ratio",
        "unified_cost_model_ramp_v",
        "unified_cost_model_risk",
        "unified_cost_model_rate_avail",
    ],
}

# V100 volatile scenario - 1092h of data, very volatile regions
# us-east-1a: 16.7% avail, 8.1h max run
# us-east-1d: 44.7% avail, 41.8h max run
# us-east-1c: 46.7% avail, 30.2h max run
# us-east-1f: 59.1% avail, 34.9h max run
V100_VOLATILE_SCENARIO = {
    "name": "V100 Volatile (4)",
    "regions": [
        "us-east-1a_v100_1",    # 16.7% avail - most volatile
        "us-east-1d_v100_1",    # 44.7% avail
        "us-east-1c_v100_1",    # 46.7% avail
        "us-east-1f_v100_1",    # 59.1% avail
    ],
    "description": "4 volatile V100 regions (1092h traces)",
    "data_paths": (VOLATILE_OLD_DATA_PATH,),
}

# Default scenarios: Global Diverse (main experiment) + Progressive (region scaling)
DEFAULT_EXPERIMENT_SCENARIOS = [GLOBAL_DIVERSE_SCENARIO, VOLATILE_GLOBAL_SCENARIO, VOLATILE_NO_FALLBACK_SCENARIO, V100_VOLATILE_SCENARIO] + PROGRESSIVE_EXPERIMENT_SCENARIOS

# Select which scenarios to use based on mode
EXPERIMENT_SCENARIOS = (
    SINGLE_REGION_EXPERIMENT_SCENARIOS
    if SINGLE_REGION_MODE
    else DEFAULT_EXPERIMENT_SCENARIOS  # Changed from MULTI_REGION_EXPERIMENT_SCENARIOS
)

# Default benchmark parameters
DEFAULT_PARAMS = {
    "DEADLINE_HOURS": None,  # Will be dynamically calculated
    "ENV_START_HOURS": 0.0,
    "MAX_WORKERS": 4,
    "DATA_PATH": "data/converted_multi_region_aligned_h100_16_merged",
    "OUTPUT_DIR": "outputs/multi_region_scenario_analysis",
}


def get_data_roots_for_scenario(
    scenario: dict,
    default_spec: Sequence[str] | Sequence[Path] | str | Path | None = None,
) -> list[Path]:
    """Return normalized data roots for a scenario with fallback to defaults."""

    if default_spec is None:
        default_spec = DEFAULT_PARAMS["DATA_PATH"]
    spec = scenario.get("data_paths", default_spec)
    roots = normalize_data_roots(spec)
    if roots:
        return roots
    fallback_roots = normalize_data_roots(default_spec)
    if fallback_roots:
        return fallback_roots
    # Final fallback: default configuration
    return normalize_data_roots(DEFAULT_PARAMS["DATA_PATH"])


def _dedupe_preserve_order(strategies: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in strategies:
        if not item:
            continue
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def get_segment_viz_strategies(scenario_name: str | None = None) -> list[str]:
    """Return the ordered list of strategies to render in segment visualizations."""
    strategies = list(SEGMENT_VIZ_STRATEGIES)
    if scenario_name:
        for scenario in EXPERIMENT_SCENARIOS:
            if scenario.get("name") != scenario_name:
                continue
            override = scenario.get("segment_viz_strategies")
            if isinstance(override, (list, tuple)):
                strategies = [str(s) for s in override if s]
            break
    return _dedupe_preserve_order(strategies)


def get_trace_duration_hours(trace_path: str) -> float:
    """Get the duration of a trace file in hours."""
    import json

    with open(trace_path, "r") as f:
        trace = json.load(f)
    gap_seconds = trace["metadata"]["gap_seconds"]
    length = len(trace["data"])
    return length * gap_seconds / 3600


def calculate_global_deadline(
    scenarios: list,
    num_traces: int | None,
    scenario_data_roots_map: dict[str, Sequence[Path] | Sequence[str]] | None = None,
    default_spec: Sequence[str] | Sequence[Path] | str | Path | None = None,
) -> float:
    """Calculate the safest deadline across all scenarios and traces using discrete tick constraints.

    This finds the minimum trace duration across all scenarios, ensuring the deadline
    is feasible for discrete tick execution and prevents 'Timestamp out of range' errors.
    """
    import json

    min_safe_deadline_hours = float("inf")
    trace_count = num_traces if num_traces is not None else 5

    for scenario in scenarios:
        roots_spec = None
        if scenario_data_roots_map is not None:
            roots_spec = scenario_data_roots_map.get(scenario.get("name"))
        if roots_spec is not None:
            data_roots = normalize_data_roots(roots_spec)
        else:
            data_roots = get_data_roots_for_scenario(scenario, default_spec)
        if not data_roots:
            raise ValueError(
                f"No data paths available for scenario '{scenario.get('name', 'unknown')}'."
            )
        for trace_index in range(trace_count):
            for region in scenario["regions"]:
                file_name = f"{trace_index}.json"
                trace_path = resolve_region_trace_path(region, file_name, data_roots)

                with open(trace_path, "r") as f:
                    trace = json.load(f)

                gap_seconds = trace["metadata"]["gap_seconds"]
                total_ticks = len(trace["data"])

                # Calculate maximum safe deadline based on available ticks
                # Use (total_ticks - 1) to leave buffer for final tick processing
                safe_deadline_hours = (total_ticks - 1) * gap_seconds / 3600
                min_safe_deadline_hours = min(
                    min_safe_deadline_hours, safe_deadline_hours
                )

    return min_safe_deadline_hours
