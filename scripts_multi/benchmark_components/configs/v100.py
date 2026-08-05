"""
V100 Cross-Area Experiment (Figure 9b)

Re-runs the AWS V100 sweep behind `research/data/v100.csv`: three p3.2xlarge
(1x V100) availability zones across three AWS areas, at a 1.5x deadline ratio.

One thing about this config differs from the rest: the traces are the ones
`artifact/prepare_data.sh` regenerates from the raw 2023 data. Each zone carries its
own per-tick spot price series -- $0.918/hr at the floor in us-east-2 and us-west-2,
up to $1.85/hr in us-east-1a -- against $3.06/hr on-demand. The per-strategy
comparison against the shipped CSV is in the `--config v100` paragraph of README.md.

Usage:
    python \\
        scripts_multi/benchmark_multi_region_modular.py \\
        --config v100 --output-dir outputs/rerun
"""

from benchmark_components.scenario_config import (
    VOLATILE_OLD_DATA_PATH,
    SINGLE_REGION_STRATEGIES,
    DEFAULT_RESTART_OVERHEAD,
)

DESCRIPTION = "AWS V100 cross-area sweep (Figure 9b), 3 zones, deadline ratio 1.5"

# ============================================================================
# STRATEGIES TO TEST
# ============================================================================
# The arms Figure 9(b) reports.
STRATEGIES = [
    "unified_cost_model_rate_ratio",
    "multi_region_oracle_dp",  # DP (Oracle)
    "unified_cost_model_oracle",  # Next Spot Oracle
    "multi_region_rc_cr_threshold_eager_failover",  # Eager Failover / UP(S)
    "multi_region_availability_probe_simple",  # UP(A)
    "multi_region_probe_cost_ratio",  # UP(AP)
]

SINGLE_REGION_STRATEGIES = SINGLE_REGION_STRATEGIES

# ============================================================================
# SCENARIOS
# ============================================================================
SCENARIOS = [
    {
        "name": "V100 Cross-Area (3 regions)",
        "regions": [
            "us-east-1c_v100_1",
            "us-east-2b_v100_1",
            "us-west-2a_v100_1",
        ],
        "description": "One AZ each in us-east-1, us-east-2 and us-west-2",
        "data_paths": (VOLATILE_OLD_DATA_PATH,),
    },
]

# ============================================================================
# PARAMETERS
# ============================================================================
PARAMS = {
    "deadline_ratios": [1.5],
    # The five windows Figure 9(b) reports, as indices into the sequential tiling
    # of the trace. Named here so the panel is one `--config v100` away.
    "full_trace_windows": [3, 5, 7, 8, 14],
    # research/data/v100.csv was run at a 50 GB checkpoint.
    "checkpoint_sizes": [50.0],
    "restart_overhead_hours": [DEFAULT_RESTART_OVERHEAD],
    "data_path": VOLATILE_OLD_DATA_PATH,
    # The three settings this panel runs with. Together they put every arm and
    # every window on research/data/v100.csv to the cent.
    #
    # --oracle-v rate_ratio scores the oracle arm with the same V as SkyNomad
    #   here, so the two differ in lifetime input alone.
    # --max-prediction-hours lifts the 2 h ceiling on the lifetime estimate.
    # --virtual-availability keeps a region under observation after the policy
    #   stops launching there, so one written off early can be found to have
    #   recovered. Each config carries the settings its own panel uses, which is
    #   why these sit here rather than in the strategy defaults.
    "strategy_args": ("--oracle-v rate_ratio --max-prediction-hours 1e9 "
                      "--virtual-availability"),
}
