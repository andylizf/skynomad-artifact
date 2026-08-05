"""
Checkpoint Sensitivity Experiment

Analyze how strategy performance changes with different checkpoint sizes.
Larger checkpoints mean higher transfer costs when migrating between regions.
"""

from benchmark_components.scenario_config import (
    GLOBAL_DIVERSE_SCENARIO,
    MERGED_A100_DATA_PATH,
    PAPER_STRATEGIES,
    SINGLE_REGION_STRATEGIES,
    DEFAULT_RESTART_OVERHEAD,
)

DESCRIPTION = "Checkpoint size sensitivity analysis"

# ============================================================================
# STRATEGIES TO TEST
# ============================================================================
STRATEGIES = PAPER_STRATEGIES

# Single-region baselines
SINGLE_REGION_STRATEGIES = SINGLE_REGION_STRATEGIES

# ============================================================================
# SCENARIOS
# ============================================================================
# Use one representative scenario to isolate checkpoint effect
SCENARIOS = [GLOBAL_DIVERSE_SCENARIO]

# ============================================================================
# PARAMETERS
# ============================================================================
PARAMS = {
    "deadline_ratios": [1.2],  # Use tighter deadline for more migrations
    # Range of checkpoint sizes based on real model sizes (H100×8 training)
    # 50GB ≈ 7B model, 200GB ≈ 13B, 500GB ≈ 40B, 1000GB ≈ 70B×1.5, 2000GB ≈ 70B×3, 4000GB ≈ 70B×5
    # 0 GB is the no-checkpoint-transfer reference point that Figure 12's leftmost
    # column plots.
    "checkpoint_sizes": [0.0, 50.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0],
    "restart_overhead_hours": [DEFAULT_RESTART_OVERHEAD],
    "data_path": MERGED_A100_DATA_PATH,
}
