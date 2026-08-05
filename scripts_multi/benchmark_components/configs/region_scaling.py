"""
Region Scaling Experiment

Analyze how strategy performance changes as the number of available regions increases.
Uses progressive scenarios from 1 region to 13 regions.
"""

from benchmark_components.scenario_config import (
    PROGRESSIVE_EXPERIMENT_SCENARIOS,
    MERGED_A100_DATA_PATH,
    PAPER_STRATEGIES,
    SINGLE_REGION_STRATEGIES,
    DEFAULT_CHECKPOINT_SIZE,
    DEFAULT_RESTART_OVERHEAD,
)

DESCRIPTION = "Progressive region scaling analysis (1 to 13 regions)"

# ============================================================================
# STRATEGIES TO TEST
# ============================================================================
STRATEGIES = PAPER_STRATEGIES

# Single-region baselines
SINGLE_REGION_STRATEGIES = SINGLE_REGION_STRATEGIES

# ============================================================================
# SCENARIOS
# ============================================================================
# Progressive: 1 region, 2 regions, ..., 13 regions
SCENARIOS = PROGRESSIVE_EXPERIMENT_SCENARIOS

# ============================================================================
# PARAMETERS
# ============================================================================
PARAMS = {
    "deadline_ratios": [1.2],  # Fix deadline ratio
    "checkpoint_sizes": [DEFAULT_CHECKPOINT_SIZE],
    "restart_overhead_hours": [DEFAULT_RESTART_OVERHEAD],
    "data_path": MERGED_A100_DATA_PATH,
}
