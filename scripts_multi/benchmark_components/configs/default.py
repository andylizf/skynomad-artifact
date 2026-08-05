"""
Default Experiment Configuration

Full benchmark with all strategies and scenarios (same as original behavior).
"""

from benchmark_components.scenario_config import (
    MULTI_REGION_STRATEGIES,
    SINGLE_REGION_STRATEGIES as _SINGLE_REGION_STRATEGIES,
    DEFAULT_EXPERIMENT_SCENARIOS,
    DEFAULT_PARAMS,
)

DESCRIPTION = "Full benchmark (all strategies, default scenarios)"

# ============================================================================
# STRATEGIES TO TEST
# ============================================================================
STRATEGIES = MULTI_REGION_STRATEGIES

# Single-region baselines
SINGLE_REGION_STRATEGIES = _SINGLE_REGION_STRATEGIES

# ============================================================================
# SCENARIOS
# ============================================================================
SCENARIOS = DEFAULT_EXPERIMENT_SCENARIOS

# ============================================================================
# PARAMETERS
# ============================================================================
PARAMS = {
    "deadline_ratios": [1.083],
    "checkpoint_sizes": [50.0],
    "restart_overhead_hours": [0.2],
    "data_path": DEFAULT_PARAMS["DATA_PATH"],
}
