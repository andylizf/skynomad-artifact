"""Additional tests for the independent ProbeLaunch type.

Covers semantics:
- Probes do not create active instances and do not contribute progress/leader.
- Probes bill a fixed 1-minute unit regardless of env gap.
- Multiple probes in the same tick accumulate cost per region.
- Probes do not trigger migrations/egress and do not change migration counters.
- Terminating after a probe in the same tick is invalid, but the probe billing remains.
"""

import json
import os
import tempfile
import unittest
from typing import Optional, Generator

from sky_spot.env import MultiTraceEnv, PROBE_BILLING_SECONDS
from sky_spot.multi_region_types import (
    Action,
    LaunchResult,
    ProbeLaunch,
    TryLaunch,
)
from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType
from sky_spot import task as task_lib


class MockTask(task_lib.Task):
    def __init__(self, duration_seconds: float):
        self.duration_seconds = duration_seconds
        self.checkpoint_size_gb = 50.0
        self._progress_source = []
    def reset(self):
        self._progress_source = []
    def set_progress_source(self, progress_source):
        self._progress_source = progress_source
    def get_total_duration_seconds(self) -> float:
        return self.duration_seconds
    def get_info(self) -> dict:
        return {}
    @property
    def is_done(self) -> bool:
        return sum(self._progress_source) >= self.duration_seconds
    def get_config(self) -> dict:
        return {"duration_seconds": self.duration_seconds}
    def __str__(self) -> str:
        return f"MockTask(duration_seconds={self.duration_seconds})"


class Args:
    def __init__(self, *, deadline_hours=10.0, restart_overhead_hours=0.0):
        self.deadline_hours = deadline_hours
        self.restart_overhead_hours = [restart_overhead_hours]
        self.inter_task_overhead = [0.0]


class TestProbeIndependentType(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # 10-minute ticks to make 1-minute billing difference salient
        self.gap_seconds = 600
        for i in range(3):
            data = {
                "metadata": {
                    "price_info": {"on_demand_price": 3.06, "price": 0.918},"gap_seconds": self.gap_seconds, "device": "v100_1"},
                "data": [0] * 30,  # available
            }
            with open(os.path.join(self.temp_dir, f"region_{i}_v100_1.json"), "w") as f:
                json.dump(data, f)
        self.trace_files = [
            os.path.join(self.temp_dir, f"region_{i}_v100_1.json") for i in range(3)
        ]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_probe_alone_creates_no_active_and_no_progress(self):
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        class OnlyProbe(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                _ = yield ProbeLaunch(region=0)

        strat = OnlyProbe(Args())
        task = MockTask(duration_seconds=3 * self.gap_seconds)
        strat.reset(env, task)

        # Tick 0: run a probe only
        env.observe()
        env.update_strategy_progress(strat)
        env.execute_multi_strategy(strat)
        # Probes should not create active instances during the tick
        self.assertEqual(env.get_active_instances(), {})
        env.tick += 1
        # Tick 1: finalize and settle progress for tick 0
        env.observe()
        env.update_strategy_progress(strat)
        # No active instance in [0,1] -> progress 0 for that tick
        self.assertEqual(strat.task_done_time[-1], 0.0)

    def test_probe_does_not_change_leader_and_progress(self):
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        class MainThenProbe(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self._phase = 0
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if self._phase == 0:
                    res = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
                    assert res and res.success
                    self._phase = 1
                else:
                    _ = yield ProbeLaunch(region=0)

        strat = MainThenProbe(Args(restart_overhead_hours=0.0))
        task = MockTask(duration_seconds=3 * self.gap_seconds)
        strat.reset(env, task)

        # Tick 0: launch main in region 1
        env.observe(); env.update_strategy_progress(strat); env.execute_multi_strategy(strat); env.tick += 1
        # Tick 1: run a probe in region 0
        env.observe(); env.update_strategy_progress(strat)
        # With zero overhead, last tick progress should equal full gap_seconds
        self.assertEqual(strat.task_done_time[-1], self.gap_seconds)
        env.execute_multi_strategy(strat); env.tick += 1
        # Verify leader remains region 1 (only active)
        env.observe(); env.update_strategy_progress(strat)
        self.assertEqual(env._current_leader_region, 1)

    def test_multiple_probes_in_same_tick_bill_per_region(self):
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        class TwoProbes(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                _ = yield ProbeLaunch(region=0)
                _ = yield ProbeLaunch(region=1)

        strat = TwoProbes(Args())
        task = MockTask(duration_seconds=self.gap_seconds)
        strat.reset(env, task)

        env.observe(); env.execute_multi_strategy(strat); env.tick += 1; env.observe()

        # Spot price per region (constant inferred when no prices[] provided)
        s0 = env.envs[0]._spot_price if env.envs[0]._spot_price is not None else env.envs[0].trace.get_price(env.envs[0]._start_index)
        s1 = env.envs[1]._spot_price if env.envs[1]._spot_price is not None else env.envs[1].trace.get_price(env.envs[1]._start_index)
        expected = (float(s0) + float(s1)) * PROBE_BILLING_SECONDS / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected, places=6)
        # probe_total reflects sum of both
        breakdown = env.get_cost_breakdown()
        self.assertAlmostEqual(float(breakdown.get("probe_total", 0.0)), expected, places=6)
        # No active instances recorded
        self.assertEqual(env.get_active_instances(), {})

    def test_probe_does_not_increase_migrations_or_transfer(self):
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        class MainAndProbe(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                res = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                assert res and res.success
                _ = yield ProbeLaunch(region=1)

        strat = MainAndProbe(Args())
        task = MockTask(duration_seconds=self.gap_seconds)
        strat.reset(env, task)

        env.observe(); env.execute_multi_strategy(strat); env.tick += 1; env.observe()
        breakdown = env.get_cost_breakdown()
        self.assertEqual(env.cross_region_migrations_count, 0)
        self.assertEqual(int(breakdown.get("CrossRegionMigrations", 0)), 0)
        self.assertAlmostEqual(float(breakdown.get("transfer_total", 0.0)), 0.0, places=6)

    def test_terminate_after_probe_same_tick_keeps_probe_billing(self):
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        class ProbeThenTerminate(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                _ = yield ProbeLaunch(region=2)
                # Terminating immediately after a probe should raise an error
                from sky_spot.multi_region_types import Terminate
                _ = yield Terminate(region=2)

        strat = ProbeThenTerminate(Args())
        task = MockTask(duration_seconds=self.gap_seconds)
        strat.reset(env, task)

        env.observe()
        with self.assertRaisesRegex(ValueError, "No instance to terminate"):
            env.execute_multi_strategy(strat)
        # Finalize tick to ensure probe billing still applies
        env.tick += 1
        env.observe()
        sub_env = env.envs[2]
        price = (
            float(sub_env._spot_price)
            if sub_env._spot_price is not None
            else float(sub_env.trace.get_price(sub_env._start_index))
        )
        expected = price * PROBE_BILLING_SECONDS / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected, places=6)
        breakdown = env.get_cost_breakdown()
        self.assertAlmostEqual(float(breakdown.get("probe_total", 0.0)), expected, places=6)
