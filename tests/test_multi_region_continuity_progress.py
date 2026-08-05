"""Tests for progress continuity with parallel background probes.

Validates that when a SPOT instance continues running across ticks while
additional ProbeLaunch actions occur in other regions (background probes),
the restart overhead is NOT applied to the continued main run's progress.
"""

import json
import os
import tempfile
import unittest
from typing import Optional, Generator

from sky_spot.env import MultiTraceEnv
from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.multi_region_types import TryLaunch, Terminate, LaunchResult, Action, ProbeLaunch
from sky_spot.utils import ClusterType
from sky_spot import task as task_lib


class MockTask(task_lib.Task):
    def __init__(self, duration_seconds: float):
        self.duration_seconds = duration_seconds
        self.checkpoint_size_gb = 50.0
        self._progress_source = []
    def reset(self):
        """Reset state for abstract base compatibility."""
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
        return {'duration_seconds': self.duration_seconds}
    def __str__(self) -> str:
        return f"MockTask(duration_seconds={self.duration_seconds})"


class MockArgs:
    def __init__(self, deadline_hours=10.0, restart_overhead_hours=0.0):
        # Use zero restart overhead to isolate continuity effect of background probes.
        # This test focuses on ensuring probes do NOT penalize ongoing progress.
        self.deadline_hours = deadline_hours
        self.restart_overhead_hours = [restart_overhead_hours]
        self.inter_task_overhead = [0.0]


class TestProgressContinuityWithProbes(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.gap_seconds = 60  # 1 minute ticks
        # region 0/1 always available
        for i in range(2):
            data = {
                'metadata': {
                    'price_info': {'on_demand_price': 3.06, 'price': 0.918},
                    'gap_seconds': self.gap_seconds,
                    'device': 'v100_1'
                },
                'data': [0] * 30  # plenty of ticks
            }
            with open(os.path.join(self.temp_dir, f'region_{i}_v100_1.json'), 'w') as f:
                json.dump(data, f)
        self.trace_files = [
            os.path.join(self.temp_dir, f'region_{i}_v100_1.json')
            for i in range(2)
        ]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_continuity_skips_restart_overhead(self):
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        class ProbeStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self._phase = 0
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                # phase 0: launch main in region 0
                if self._phase == 0:
                    res = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert res is not None and res.success
                    self._phase = 1
                else:
                    # phase >=1: keep main; launch a background probe in region 1
                    _ = yield ProbeLaunch(region=1)
                # no explicit terminate here; test continuity only

        # Use zero overhead to isolate probe continuity effect
        args = MockArgs(deadline_hours=10.0, restart_overhead_hours=0.0)
        task = MockTask(duration_seconds=5 * self.gap_seconds)
        strat = ProbeStrategy(args)
        strat.reset(env, task)

        # Tick 0
        env.observe()
        env.update_strategy_progress(strat)
        env.execute_multi_strategy(strat)
        env.tick += 1

        # Tick 1
        env.observe()
        env.update_strategy_progress(strat)
        # After continuity fix, we expect full gap_seconds progress on this tick
        self.assertEqual(strat.task_done_time[-1], self.gap_seconds)
        env.execute_multi_strategy(strat)
        env.tick += 1

        # Tick 2
        env.observe()
        env.update_strategy_progress(strat)
        # Still full progress (continuity maintained)
        self.assertEqual(strat.task_done_time[-1], self.gap_seconds)

    def test_cold_start_and_probe_do_not_penalize_running_instance(self):
        """Cold start overhead applies, but later background probes shouldn't penalize ongoing run.

        - Use 10-minute ticks and 12-minute restart overhead.
        - Launch main at tick 0; expect cold start overhead at tick 1, partial progress at tick 2.
        - Then start a background probe in another region; at the next tick, main progress should be full.
        """
        # Set up a separate environment with 10-minute ticks
        gap_seconds = 600
        temp_dir = tempfile.mkdtemp()
        try:
            for i in range(2):
                data = {
                    'metadata': {
                        'price_info': {'on_demand_price': 3.06, 'price': 0.918},
                        'gap_seconds': gap_seconds,
                        'device': 'v100_1'
                    },
                    'data': [0] * 30
                }
                with open(os.path.join(temp_dir, f'region_{i}_v100_1.json'), 'w') as f:
                    json.dump(data, f)
            trace_files = [
                os.path.join(temp_dir, f'region_{i}_v100_1.json')
                for i in range(2)
            ]

            env = MultiTraceEnv(trace_files, env_start_hours=0)

            class ProbeAfterColdStart(MultiRegionStrategy):
                def __init__(self, args):
                    super().__init__(args)
                    self.launched_main = False
                def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                    # Launch main once at tick 0
                    if not self.launched_main:
                        res = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                        assert res is not None and res.success
                        self.launched_main = True
                        return
                    # After overhead finishes (tick >= 2), kick off a background probe in region 1
                    if self.env.tick >= 2:
                        _ = yield ProbeLaunch(region=1)
                    # No explicit terminate; we only test continuity impact

            # 12 minutes overhead -> two ticks to clear
            args = MockArgs(deadline_hours=10.0, restart_overhead_hours=0.2)
            task = MockTask(duration_seconds=5 * gap_seconds)
            strat = ProbeAfterColdStart(args)
            strat.reset(env, task)

            # Tick 0
            env.observe()
            env.update_strategy_progress(strat)
            env.execute_multi_strategy(strat)
            env.tick += 1

            # Tick 1: cold start overhead (600s consumed out of 720s)
            env.observe()
            env.update_strategy_progress(strat)
            self.assertEqual(strat.task_done_time[-1], 0)  # No progress during full-overhead tick
            env.execute_multi_strategy(strat)
            env.tick += 1

            # Tick 2: finish overhead (remaining 120s), so partial progress 480s
            env.observe()
            env.update_strategy_progress(strat)
            self.assertGreater(strat.task_done_time[-1], 0)
            self.assertLess(strat.task_done_time[-1], gap_seconds)
            env.execute_multi_strategy(strat)  # This will trigger a background probe in region 1
            env.tick += 1

            # Tick 3: background probe was launched last tick; main should not be penalized
            env.observe()
            env.update_strategy_progress(strat)
            self.assertEqual(strat.task_done_time[-1], gap_seconds)
        finally:
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
