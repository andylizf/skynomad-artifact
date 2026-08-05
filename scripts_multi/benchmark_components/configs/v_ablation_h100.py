"""Progress-value ablation, the appendix's H100 8-region cell.

The eight GCP H100 zones of Figure 2: three in Asia, one in Europe, four in the
US. `asia-south2-b` is the price outlier at 3.6x the cheapest zone's spot rate.
Not the same eight as Figure 10b's 8-region point, which is ordered by single-zone
cost and shares only five of them.

Driven by artifact/v_ablation.py; the numbers it produces are pinned in
tests/test_a7_table.py.
"""
from benchmark_components.scenario_config import (
    MERGED_A100_DATA_PATH,
    DEFAULT_CHECKPOINT_SIZE,
    DEFAULT_RESTART_OVERHEAD,
)
from sky_spot.strategies.unified_cost_model_f_ablation import ABLATION_STRATEGIES

DESCRIPTION = "Progress value ablation on 8 GCP H100 regions"

STRATEGIES = ABLATION_STRATEGIES + [
    "multi_region_oracle_dp",
    "unified_cost_model_rate_ratio",
]

SINGLE_REGION_STRATEGIES = []

PAPER_8_REGIONS = [
    "asia-south2-b_h100_16",
    "asia-southeast1-b_h100_16",
    "asia-southeast1-c_h100_16",
    "europe-west1-c_h100_16",
    "us-central1-a_h100_16",
    "us-east4-b_h100_16",
    "us-west1-a_h100_16",
    "us-west1-b_h100_16",
]

SCENARIOS = [
    {
        "name": "H100 8-region",
        "regions": PAPER_8_REGIONS,
        "description": "Figure 2's eight GCP H100 regions",
        "data_paths": (MERGED_A100_DATA_PATH,),
    }
]

PARAMS = {
    # The global deadline of these traces: 311 ticks of 600 s, which the driver
    # derives from the shortest trace when deadline_hours is left out.
    "deadline_hours": 51.833333333333336,
    "deadline_ratios": [1.5, 2.0],
    "checkpoint_sizes": [DEFAULT_CHECKPOINT_SIZE],
    "restart_overhead_hours": [DEFAULT_RESTART_OVERHEAD],
    "data_path": MERGED_A100_DATA_PATH,
    "num_traces": 12,
    "start_mode": "stratified",
    "random_seed": 42,
    # Both settings as in v_ablation_v100_4node.py, which explains them.
    "strategy_args": "--max-prediction-hours 1e9 --virtual-availability",
    "dump_history": True,
}
