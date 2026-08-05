import json
import os
import tempfile
import unittest

from sky_spot.env import TraceEnv
from sky_spot.strategies.strategy import Strategy
from sky_spot.task import SingleTask
from sky_spot.utils import ClusterType
from sky_spot import simulate


class AlwaysSpotStrategy(Strategy):
    """Heuristic: always request SPOT, regardless of availability.

    This forces the final safety check path to run. In the critical window,
    our patched safety check must switch to ON_DEMAND if SPOT is unavailable.
    """
    NAME = 'always_spot_test'

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        return ClusterType.SPOT

    @classmethod
    def _from_args(cls, parser):  # Not used in these tests
        return cls(parser.parse_args([]))


class TestStrongGuaranteeSingleRegion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # 1 hour per tick for easy reasoning
        self.gap = 3600
        # Build a short trace: tick0 available, tick1 unavailable, tick2 available
        trace = {
            'metadata': {
                'price_info': {'on_demand_price': 3.06, 'price': 0.918},
                'gap_seconds': self.gap,
                'device': 'v100_1',  # ensure base price detection via filename substring
            },
            # 0 = available, 1 = preempted
            'data': [0, 1, 0, 0],
        }
        # Name must include 'v100_1' for pricing lookup in TraceEnv
        self.trace_file = os.path.join(self.tmp, 'single_region_v100_1.json')
        with open(self.trace_file, 'w') as f:
            json.dump(trace, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_critical_window_safety_override_switches_to_on_demand(self):
        """At the critical window with SPOT unavailable, request must flip to ON_DEMAND."""
        env = TraceEnv(self.trace_file, env_start_hours=0)

        class Args:
            deadline_hours = 2.1  # hours
            restart_overhead_hours = [0.0]
            inter_task_overhead = [0.0]

        # Task requires slightly > 2 ticks (2.1h)
        task = SingleTask({'duration': 2.1, 'checkpoint_size_gb': 1.0})
        strat = AlwaysSpotStrategy(Args())
        strat.reset(env, task)

        # Choose an initial large deadline so tick 0 is NOT in critical window,
        # ensuring last_cluster_type becomes SPOT.
        strat.deadline = 4.0 * 3600
        req0 = strat.step()  # has_spot=True at tick 0
        env.step(req0)
        self.assertEqual(req0, ClusterType.SPOT)

        # Now force a critical window at tick 1 by shrinking the deadline.
        # With elapsed=1h, remaining_time becomes 1h; remaining_task+ro ≈ 1.1h => critical.
        strat.deadline = 2.1 * 3600

        # tick 1: SPOT is unavailable (data[1] == 1) and we're in critical window.
        req1 = strat.step()
        # The patched safety check should flip SPOT->ON_DEMAND in the critical window.
        self.assertEqual(req1, ClusterType.ON_DEMAND, 'Must choose ON_DEMAND in critical window when SPOT unavailable')
        env.step(req1)

        # Ensure we did not get NONE (which would waste the last usable tick)
        self.assertNotEqual(req1, ClusterType.NONE)

    def test_simulate_completes_without_deadline_error(self):
        """Full simulate() run should complete without Deadline exceeded in the same setup."""
        env = TraceEnv(self.trace_file, env_start_hours=0)

        class Args:
            deadline_hours = 2.1
            restart_overhead_hours = [0.0]
            inter_task_overhead = [0.0]

        task = SingleTask({'duration': 2.1, 'checkpoint_size_gb': 1.0})
        strat = AlwaysSpotStrategy(Args())

        # Run full pipeline (silent, small output dir)
        stats = simulate.simulate(
            envs=[env],
            strategy=strat,
            task=task,
            trace_file=os.path.basename(self.trace_file),
            deadline_hours=Args.deadline_hours,
            restart_overhead_hours=Args.restart_overhead_hours,
            env_start_hours=0.0,
            output_dir=self.tmp,
            kwargs={'deadline_hours': Args.deadline_hours, 'restart_overhead_hours': Args.restart_overhead_hours},
            silent=True,
            dump_history=True,
        )
        # If simulate() returns, no deadline exception occurred; task should be done.
        self.assertTrue(task.is_done)
        self.assertIn('costs', stats)


if __name__ == '__main__':
    unittest.main()
