"""The appendix's progress-value ablation table.

Every number below is the appendix's own, and each cell is produced by the config
named beside it. The table depends on a long chain of inputs -- source capture,
binarisation threshold, resampling rule, one constant price per zone, the deadline,
the window sampler's seed, the region set, two strategy settings, and the three V
formulas that are not plain powers of the schedule ratio -- so it is pinned here
rather than left to a manual check.

Marked slow because it runs 192 simulations. `pytest -m "not slow"` skips it.
"""

import csv
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

V100 = "v_ablation_v100_4node"
H100 = "v_ablation_h100"
V100_DATA = REPO / "data/converted_v100_4node_600s"
H100_DATA = REPO / "data/converted_multi_region_aligned_h100_16_merged"

# candidate -> {(cell, ratio): the appendix's gap}
PAPER = {
    "unified_cost_model_rate_ratio": {
        (V100, "1.5"): 9.60, (V100, "2.0"): 5.83,
        (H100, "1.5"): 62.59, (H100, "2.0"): 30.92,
    },
    "ucm_v_alpha_05": {
        (V100, "1.5"): 9.86, (V100, "2.0"): 5.54,
        (H100, "1.5"): 64.70, (H100, "2.0"): 31.26,
    },
    "ucm_v_s2t1": {
        (V100, "1.5"): 30.95, (V100, "2.0"): 5.11,
        (H100, "1.5"): 38.42, (H100, "2.0"): 35.26,
    },
    "ucm_v_hjb_exp": {
        (V100, "1.5"): 9.90, (V100, "2.0"): 5.50,
        (H100, "1.5"): 66.20, (H100, "2.0"): 33.91,
    },
    "ucm_v_s5t1": {
        (V100, "1.5"): 27.73, (V100, "2.0"): 8.52,
        (H100, "1.5"): 40.91, (H100, "2.0"): 44.26,
    },
    "ucm_v_neely_dpp": {
        (V100, "1.5"): 10.67, (V100, "2.0"): 9.98,
        (H100, "1.5"): 68.88, (H100, "2.0"): 35.67,
    },
}
LABEL = {
    "unified_cost_model_rate_ratio": "Log-barrier (default)",
    "ucm_v_alpha_05": "alpha=0.5",
    "ucm_v_s2t1": "Quadratic surrogate",
    "ucm_v_hjb_exp": "HJB exponential",
    "ucm_v_s5t1": "Time-only",
    "ucm_v_neely_dpp": "Neely DPP",
}
PAPER_OPTIMAL = {
    (V100, "1.5"): 33.15, (V100, "2.0"): 24.76,
    (H100, "1.5"): 522.72, (H100, "2.0"): 390.93,
}
OPTIMAL = "multi_region_oracle_dp"
TOL = 0.01  # 1%, wide enough for the appendix's two decimals


def _run(config: str, out: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "scripts_multi/benchmark_multi_region_modular.py",
         "--config", config, "--output-dir", str(out)],
        cwd=REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    hits = list(out.rglob("scenario_results_*.csv"))
    assert hits, f"no results under {out}"
    costs = defaultdict(list)
    for row in csv.DictReader(hits[0].open()):
        if row.get("task_type") != "multi_region":
            continue
        costs[(row["strategy"], row["deadline_ratio"])].append(float(row["cost"]))
    return costs


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    for data, config in ((V100_DATA, V100), (H100_DATA, H100)):
        if not data.is_dir():
            pytest.skip(f"{data.relative_to(REPO)} missing; run prepare_data.sh first")
    root = tmp_path_factory.mktemp("a7")
    optimal, gap = {}, {}
    for config in (V100, H100):
        costs = _run(config, root / config)
        for ratio in ("1.5", "2.0"):
            opt = statistics.mean(costs[(OPTIMAL, ratio)])
            optimal[(config, ratio)] = opt
            for (strategy, r), values in costs.items():
                if r == ratio and strategy != OPTIMAL:
                    gap[(strategy, config, ratio)] = statistics.mean(values) - opt
    return optimal, gap


@pytest.mark.slow
@pytest.mark.parametrize("cell", sorted(PAPER_OPTIMAL), ids=lambda c: f"{c[0]}-{c[1]}")
def test_optimal_matches_the_appendix(table, cell):
    optimal, _ = table
    assert optimal[cell] == pytest.approx(PAPER_OPTIMAL[cell], rel=TOL)


@pytest.mark.slow
@pytest.mark.parametrize("strategy", list(PAPER), ids=lambda s: LABEL[s])
def test_candidate_gaps_match_the_appendix(table, strategy):
    _, gap = table
    for (config, ratio), expected in PAPER[strategy].items():
        got = gap.get((strategy, config, ratio))
        assert got is not None, f"{strategy} missing in {config} at ratio {ratio}"
        assert got == pytest.approx(expected, rel=TOL), (
            f"{LABEL[strategy]}, {config} ratio {ratio}: "
            f"{got:+.2f} against the appendix's {expected:+.2f}"
        )


@pytest.mark.slow
def test_log_barrier_wins_the_sum_column(table):
    """The appendix's conclusion, which is a four-cell result.

    It does not hold cell by cell -- on V100 alone alpha=0.5 is three cents ahead
    -- so the row sum is the thing to pin.
    """
    _, gap = table
    total = {
        s: sum(gap[(s, c, r)] for (c, r) in PAPER[s]) for s in PAPER
    }
    assert min(total, key=total.get) == "unified_cost_model_rate_ratio"
    assert total["unified_cost_model_rate_ratio"] == pytest.approx(108.94, rel=TOL)


@pytest.mark.slow
def test_the_two_degenerate_forms_are_the_worst(table):
    """Neely ignores time and Time-only ignores progress; both should trail."""
    _, gap = table
    total = {s: sum(gap[(s, c, r)] for (c, r) in PAPER[s]) for s in PAPER}
    worst_two = sorted(total, key=total.get)[-2:]
    assert set(worst_two) == {"ucm_v_s5t1", "ucm_v_neely_dpp"}
