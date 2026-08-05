"""Progress-value ablation, the appendix's V100 4-region cell.

Three AWS availability zones from a four-node capture: each trace records how
many of four requested nodes were up, and a zone counts as usable when at least
one was. Built by scripts_multi/trace_sampling/convert_v100_4node_600s.py, which
has to run before this config.

Driven by artifact/v_ablation.py; the numbers it produces are pinned in
tests/test_a7_table.py.
"""
from sky_spot.strategies.unified_cost_model_f_ablation import ABLATION_STRATEGIES

DESCRIPTION = "Progress value ablation on the 3-zone V100 4-node traces"

# unified_cost_model_rate_ratio is the appendix's log-barrier default: V = C_od *
# theta/theta_tilde with theta_tilde the planned rate P/T, which is the closed
# form C_od * T/(T-t) * (P-p)/P. ucm_v_log_barrier is the same function under the
# ablation's own naming, and the two agree window for window.
STRATEGIES = ABLATION_STRATEGIES + [
    "multi_region_oracle_dp",
    "unified_cost_model_rate_ratio",
]
SINGLE_REGION_STRATEGIES = []

DATA_PATH = "data/converted_v100_4node_600s"

SCENARIOS = [
    {
        "name": "V100 4-node AWS (3 zones)",
        "regions": [
            "us-east-1f_v100_1",
            "us-east-2a_v100_1",
            "us-west-2c_v100_1",
        ],
        "description": "Three AWS AZs, four nodes each",
        "data_paths": (DATA_PATH,),
    }
]

PARAMS = {
    # The global deadline of these traces: 310 ticks of 600 s. The driver derives
    # the same value from the shortest trace when deadline_hours is left out; it is
    # written down so the window sampler, which keys off the deadline, does not
    # move if a trace is regenerated at a different length.
    "deadline_hours": 51.666666666666664,
    "deadline_ratios": [1.5, 2.0],
    "checkpoint_sizes": [50.0],
    "restart_overhead_hours": [0.2],
    "data_path": DATA_PATH,
    "num_traces": 12,
    "start_mode": "stratified",
    "random_seed": 42,
    # Two settings that differ from the other sweeps in this directory.
    #
    # --max-prediction-hours lifts the 2 h ceiling on the lifetime estimate. On
    # these traces the long contiguous availability of the stable zones is the
    # signal region choice runs on, and truncating it to two hours flattens those
    # zones against the volatile ones; the ceiling binds on 90.8% of the estimates
    # here.
    #
    # --virtual-availability keeps a region's availability under observation after
    # the policy stops launching there, so the survival estimate of a region it has
    # written off can still recover: otherwise the estimate that drove the policy
    # away is also the last observation it ever takes for that region.
    "strategy_args": "--max-prediction-hours 1e9 --virtual-availability",
    # Read the cost from the snapshot rebuilt after the loop, so it runs through
    # the tick that completes the task. See simulation_runner.
    "dump_history": True,
}
