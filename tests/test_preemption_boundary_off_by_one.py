"""Test for off-by-one progress loss when preemption happens at tick t.

Before the fix, MultiTraceEnv.observe() applies preemptions at tick t and then
update_strategy_progress() uses the post-preemption active set to attribute
progress for the previous interval [t-1, t], incorrectly dropping that tick's
work. This test reproduces that case.
"""

import json
import os
import tempfile
import unittest
from typing import Optional, Generator

from sky_spot.env import MultiTraceEnv
from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.multi_region_types import TryLaunch, Terminate, LaunchResult, Action
from sky_spot.utils import ClusterType
from sky_spot import task as task_lib


class SimpleTask(task_lib.Task):
    """Simple task that tracks external progress seconds."""

    def __init__(self, duration_seconds: float):
        self.duration_seconds = float(duration_seconds)
        self._progress = []
        self.checkpoint_size_gb = 1.0
        super().__init__({'duration': duration_seconds / 3600.0})

    def set_progress_source(self, task_done_time_list):
        self._progress = task_done_time_list

    def reset(self):
        self._progress = []

    def get_total_duration_seconds(self) -> float:
        return self.duration_seconds

    @property
    def is_done(self) -> bool:
        return sum(self._progress) >= self.duration_seconds - 1e-9

    def get_info(self) -> dict:
        return {}

    def __str__(self) -> str:
        return f"SimpleTask({self.duration_seconds}s)"


class LaunchOnceKeepRunning(MultiRegionStrategy):
    """Launch SPOT in region 0 once and keep running until task_done, then terminate."""

    NAME = 'launch_once_keep_running_test'

    def __init__(self, args):
        super().__init__(args)
        self._launched = False

    def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
        if self.task_done:
            # Clean up
            for r in list(self.env.get_active_instances().keys()):
                yield Terminate(region=r)
            return
        if not self._launched:
            res = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
            assert res is not None and res.success
            self._launched = True
            return
        # Otherwise, keep running (no actions)


class TestPreemptionBoundary(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.gap = 195  # seconds per tick
        # Build a trace: available for ticks 0,1,2; then preempted at tick 3
        # This means the work done in [2,3] should be credited.
        data = {
            'metadata': {
                'price_info': {'on_demand_price': 3.06, 'price': 0.918},
                'gap_seconds': self.gap,
                'device': 'v100_1'
            },
            'data': [0, 0, 0, 1, 1, 1]  # 0=available, 1=preempted
        }
        self.trace_file = os.path.join(self.temp_dir, 'region_0_v100_1.json')
        with open(self.trace_file, 'w') as f:
            json.dump(data, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_progress_not_lost_on_boundary_preemption(self):
        # Env with one region
        env = MultiTraceEnv([self.trace_file], env_start_hours=0)

        # Task requires exactly 3 ticks of work
        task = SimpleTask(duration_seconds=3 * self.gap)

        # Args
        class Args:
            deadline_hours = 10.0
            restart_overhead_hours = [0.0]
            inter_task_overhead = [0.0]

        strat = LaunchOnceKeepRunning(Args())
        strat.reset(env, task)

        tick = 0
        # Run manual loop mirroring simulate's multi-region path
        while not strat.task_done and tick < 8:
            env.observe()
            env.update_strategy_progress(strat)
            if strat.task_done:
                break
            env.execute_multi_strategy(strat)
            env.tick += 1
            tick += 1

        # With correct accounting, the 3rd tick of progress ([2,3]) should be credited
        # even if preemption happens at t=3. So task must be done by or before tick==3.
        self.assertTrue(strat.task_done, 'Task should complete with 3 ticks of work')
        self.assertLessEqual(tick, 4, f'Should not require an extra tick; ran {tick} ticks')


if __name__ == '__main__':
    unittest.main()

