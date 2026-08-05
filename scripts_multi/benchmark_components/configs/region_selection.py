"""
Region Selection Experiment

Compare strategy performance across different geographic region configurations:
- US only (8 regions)
- Europe only (2 regions) - GDPR compliant
- Asia only (3 regions)
- Global (all 13 regions)
"""

from benchmark_components.scenario_config import (
    ALL_US_REGIONS,
    EUROPE_REGIONS,
    ASIA_REGIONS,
    GCP_A3_H100_16_REGIONS,
    MERGED_A100_DATA_PATH,
    PAPER_STRATEGIES,
    SINGLE_REGION_STRATEGIES,
    DEFAULT_CHECKPOINT_SIZE,
    DEFAULT_RESTART_OVERHEAD,
)

DESCRIPTION = "Geographic region selection analysis (US, EU, Asia, Global)"

# ============================================================================
# STRATEGIES TO TEST
# ============================================================================
STRATEGIES = PAPER_STRATEGIES

# Single-region baselines
SINGLE_REGION_STRATEGIES = SINGLE_REGION_STRATEGIES

# ============================================================================
# SCENARIOS
# ============================================================================
SCENARIOS = [
    {
        "name": "US Only (8)",
        "regions": ALL_US_REGIONS,
        "description": "All US regions (Central + East + West)",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Europe Only (2)",
        "regions": EUROPE_REGIONS,
        "description": "GDPR-compliant European regions only",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Asia Only (3)",
        "regions": ASIA_REGIONS,
        "description": "Asian regions only",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
    {
        "name": "Global (13)",
        "regions": GCP_A3_H100_16_REGIONS,
        "description": "All 13 GCP H100 16-node regions",
        "data_paths": (MERGED_A100_DATA_PATH,),
    },
]

# ============================================================================
# PARAMETERS
# ============================================================================
PARAMS = {
    "deadline_ratios": [1.2],
    "checkpoint_sizes": [DEFAULT_CHECKPOINT_SIZE],
    "restart_overhead_hours": [DEFAULT_RESTART_OVERHEAD],
    "data_path": MERGED_A100_DATA_PATH,
}
