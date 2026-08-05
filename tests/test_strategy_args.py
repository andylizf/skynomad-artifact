"""--strategy-args reaches the strategies that declare a flag, and only those.

One --strategy-args list is handed to every strategy a sweep runs, but they do
not share a flag set. Each _from_args ends in parse_args(), which exits the
process on an unknown flag, so an unfiltered list would kill the DP oracle the
moment anyone tuned a unified-cost-model knob.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts_multi"))

from benchmark_components.simulation_runner import _accepted_strategy_argv  # noqa: E402
from sky_spot.strategies import strategy as strategy_lib  # noqa: E402

BASE_ARGS = (70.0, 0.2, REPO / "outputs" / "sim_temp", 0.0, 50.0)


def _accept(strategy_name, argv):
    cls = strategy_lib.Strategy.get(strategy_name)
    return _accepted_strategy_argv(cls, BASE_ARGS, argv)


def test_empty_argv_is_a_no_op():
    assert _accept("unified_cost_model_risk", []) == []


def test_declared_store_true_flag_is_kept():
    assert _accept("unified_cost_model_risk", ["--probe-revalidate"]) == ["--probe-revalidate"]


def test_flag_with_a_value_keeps_both_tokens():
    argv = ["--probe-interval-ticks", "12"]
    assert _accept("unified_cost_model_risk", argv) == argv


def test_undeclared_flag_is_dropped_rather_than_killing_the_run():
    """The DP oracle has no probing knobs; it must still run."""
    assert _accept("multi_region_oracle_dp", ["--probe-revalidate"]) == []


def test_mixed_list_splits_per_strategy():
    argv = ["--probe-revalidate", "--deadline-hours", "70"]
    # --deadline-hours is on the shared base parser, so both strategies take it.
    assert _accept("unified_cost_model_risk", argv) == argv
    assert _accept("multi_region_oracle_dp", argv) == ["--deadline-hours", "70"]


@pytest.mark.parametrize("name", ["unified_cost_model_risk", "multi_region_oracle_dp"])
def test_filtering_does_not_leave_sys_argv_modified(name):
    before = list(sys.argv)
    _accept(name, ["--probe-revalidate"])
    assert sys.argv == before
