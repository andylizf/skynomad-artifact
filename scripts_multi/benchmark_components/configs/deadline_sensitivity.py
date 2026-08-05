"""
Deadline Sensitivity Experiment

Analyze how strategy performance changes across different deadline pressures.
Tests a wide range of deadline ratios from tight (1.05) to relaxed (3.0).
"""

from benchmark_components.scenario_config import (
    GLOBAL_DIVERSE_SCENARIO,
    MERGED_A100_DATA_PATH,
    PAPER_STRATEGIES,
    SINGLE_REGION_STRATEGIES,
    DEFAULT_CHECKPOINT_SIZE,
    DEFAULT_RESTART_OVERHEAD,
)

DESCRIPTION = "Deadline sensitivity analysis across multiple deadline ratios"

# ============================================================================
# STRATEGIES TO TEST
# ============================================================================
STRATEGIES = PAPER_STRATEGIES

# Single-region baselines
SINGLE_REGION_STRATEGIES = SINGLE_REGION_STRATEGIES

# ============================================================================
# SCENARIOS
# ============================================================================
# Use one representative scenario to isolate deadline effect
SCENARIOS = [GLOBAL_DIVERSE_SCENARIO]

# ============================================================================
# PARAMETERS
# ============================================================================
PARAMS = {
    # Wide range of deadline ratios: tight → relaxed
    "deadline_ratios": [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0, 3.0],
    "checkpoint_sizes": [DEFAULT_CHECKPOINT_SIZE],
    "restart_overhead_hours": [DEFAULT_RESTART_OVERHEAD],
    "data_path": MERGED_A100_DATA_PATH,
}
