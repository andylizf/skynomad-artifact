import json
import math
import os
import tempfile
import unittest
from typing import Optional, Generator

from sky_spot.env import MultiTraceEnv
from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.multi_region_types import TryLaunch, Terminate, LaunchResult, Action, ProbeLaunch
from sky_spot.utils import ClusterType
from sky_spot import task as task_lib
from sky_spot import simulate


class MockTask(task_lib.Task):
    def __init__(self, duration_seconds: float):
        self.duration_seconds = duration_seconds
        self.checkpoint_size_gb = 50.0
        self._progress_source = []

    def reset(self):
        self._progress_source = []

    def set_progress_source(self, src):
        self._progress_source = src

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


class StrictInvariantsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.gap_seconds = 300
        # Two simple regions always available
        for i in range(2):
            data = {
                'metadata': {
                    'price_info': {'on_demand_price': 3.06, 'price': 0.918},
                    'gap_seconds': self.gap_seconds,
                    'device': 'v100_1',
                },
                'data': [0] * 10,
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

    def test_update_requires_observe_first(self):
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        class NoopStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if False:
                    yield  # pragma: no cover
                return

        class Args:
            deadline_hours = 1.0
            restart_overhead_hours = [0.1]
            inter_task_overhead = [0.0]

        strat = NoopStrategy(Args())
        task = MockTask(duration_seconds=self.gap_seconds)
        strat.reset(env, task)

        with self.assertRaises(AssertionError):
            env.update_strategy_progress(strat)

        # Now follow correct order should succeed
        env.observe()
        env.update_strategy_progress(strat)

    def test_same_tick_terminate_after_launch_same_type_disallowed(self):
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        class TestStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                # Launch ON_DEMAND then try to terminate in same tick
                # This should NOT be allowed
                res = yield TryLaunch(region=0, cluster_type=ClusterType.ON_DEMAND)
                assert res.success
                yield Terminate(region=0, cluster_type=ClusterType.ON_DEMAND)

        class Args:
            deadline_hours = 10.0
            restart_overhead_hours = [0.1]
            inter_task_overhead = [0.0]

        strat = TestStrategy(Args())
        task = MockTask(duration_seconds=self.gap_seconds)
        strat.reset(env, task)

        env.observe()
        
        # Should raise ValueError when trying to terminate just-launched instance
        with self.assertRaises(ValueError) as ctx:
            env.execute_multi_strategy(strat)
        
        self.assertIn("just launched in the same tick", str(ctx.exception))

    def test_safety_net_zero_slack_switches_to_ondemand(self):
        """Simplified safety net: always switch to ON_DEMAND at boundary, regardless of SPOT state."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        class Args:
            deadline_hours = (3000.0 / 3600.0)
            restart_overhead_hours = [0.0]
            inter_task_overhead = [0.0]

        class SingleLaunchStrategy(MultiRegionStrategy):
            def _step_multi(self):  # pragma: no cover - not used directly
                if False:
                    yield  # pragma: no cover
                return

        strat = SingleLaunchStrategy(Args())
        task = MockTask(duration_seconds=1800.0)
        strat.reset(env, task)

        gap = env.gap_seconds  # 300s from setUp traces
        self.assertEqual(gap, 300)

        # Fabricate state: leader spot with zero overhead, zero slack remaining
        env.tick = 9  # elapsed = 2700s, deadline = 3000s -> remaining_time = 300s
        env.observed_tick = env.tick
        strat.deadline = 3000.0
        strat.restart_overhead = 0.0

        strat.task_done_time.clear()
        strat.task_done_time.extend([300.0, 300.0, 300.0, 300.0, 300.0])  # sum = 1500
        env._current_leader_region = 0
        env._current_leader_progress = 1500.0

        # No remaining restart overhead on the leader
        strat.remaining_restart_overhead = 0.0
        env.active_instances = {(0, ClusterType.SPOT): env.tick}
        env._safety_net_latched = False

        # Early safety-net check happens after observe() in the main loop.
        # For direct calls in tests, trigger it explicitly before executing.
        env.maybe_trigger_safety_net(strat)
        env.execute_multi_strategy(strat)

        # Simplified safety net: always switch to ON_DEMAND at boundary
        self.assertTrue(env._safety_net_latched)
        self.assertEqual(env.get_active_instances(), {0: ClusterType.ON_DEMAND})

    def test_safety_net_zero_slack_switches_when_overhead_left(self):
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        class Args:
            deadline_hours = (3000.0 / 3600.0)
            restart_overhead_hours = [0.1]
            inter_task_overhead = [0.0]

        class SingleLaunchStrategy(MultiRegionStrategy):
            def _step_multi(self):
                if False:
                    yield  # pragma: no cover
                return

        strat = SingleLaunchStrategy(Args())
        task = MockTask(duration_seconds=1800.0)
        strat.reset(env, task)

        gap = env.gap_seconds
        self.assertEqual(gap, 300)

        env.tick = 8  # remaining time = 600s (2 ticks)
        env.observed_tick = env.tick
        strat.deadline = 3000.0
        strat.restart_overhead = 0.1 * 3600.0

        strat.task_done_time.clear()
        strat.task_done_time.extend([300.0, 300.0, 300.0, 300.0, 300.0, 60.0])  # sum = 1560
        env._current_leader_region = 0
        env._current_leader_progress = 1560.0

        # Leader still has outstanding restart overhead
        strat.remaining_restart_overhead = strat.restart_overhead
        env.active_instances = {(0, ClusterType.SPOT): env.tick}
        env._safety_net_latched = False

        # Trigger the early safety net explicitly (normally done in simulate loop)
        env.maybe_trigger_safety_net(strat)
        env.execute_multi_strategy(strat)

        self.assertTrue(env._safety_net_latched)
        self.assertEqual(env.get_active_instances(), {0: ClusterType.ON_DEMAND})


    def test_snapshots_and_prices(self):
        # Two regions with dynamic prices
        temp_dir = tempfile.mkdtemp()
        try:
            gap_seconds = 60
            def write(idx, prices):
                trace_data = {
                    'metadata': {
                        'price_info': {'on_demand_price': 3.06, 'price': 0.918},
                        'gap_seconds': gap_seconds,
                        'region': f'test-{idx}',
                        'zone': f'test-{idx}a',
                        'instance_type': 'v100',
                        'device': 'v100_1'
                    },
                    'data': [0, 1, 0],
                    'prices': prices,
                }
                fp = os.path.join(temp_dir, f'region_{idx}_v100_1.json')
                with open(fp, 'w') as f:
                    json.dump(trace_data, f)
                return fp

            files = [write(0, [0.5, 0.6, 0.7]), write(1, [0.7, 0.8, 0.9])]
            env = MultiTraceEnv(files, env_start_hours=0)

            # Snapshot at tick 0 should reflect data[0] (available)
            snap = env.get_spot_availability_snapshot()
            self.assertEqual(snap, {0: True, 1: True})

            # Prices list always returned for all regions
            prices0 = env.get_all_regions_spot_prices()
            self.assertEqual(len(prices0), 2)
            self.assertTrue(all(isinstance(x, (float, int)) for x in prices0))

            # Advance to next completed tick 1 (tick needs to be 2 for snapshot to read index 1)
            env.observe(); env.tick += 1  # complete tick 0
            env.observe(); env.tick += 1  # complete tick 1
            snap1 = env.get_spot_availability_snapshot()
            self.assertEqual(snap1, {0: False, 1: False})

            # Advance to next completed tick 2 (tick needs to be 3 for snapshot to read index 2)
            env.observe(); env.tick += 1
            snap2 = env.get_spot_availability_snapshot()
            self.assertEqual(snap2, {0: True, 1: True})
        finally:
            import shutil
            shutil.rmtree(temp_dir)

    def test_leader_tie_breaker_prefers_lower_region_index(self):
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        class TwoLaunchStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                # Launch both regions in the same tick
                res = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                assert res and res.success
                _ = yield ProbeLaunch(region=1)

        class Args:
            deadline_hours = 10.0
            restart_overhead_hours = [0.0]  # ensure equal instantaneous progress
            inter_task_overhead = [0.0]

        strat = TwoLaunchStrategy(Args())
        task = MockTask(duration_seconds=10 * self.gap_seconds)
        strat.reset(env, task)

        # Tick 0: launch in both regions
        env.observe()
        env.execute_multi_strategy(strat)
        env.tick += 1

        # Tick 1: with equal progress/uptime, leader should be lower index (0)
        env.observe()
        env.update_strategy_progress(strat)
        self.assertEqual(env._current_leader_region, 0)

    def test_background_probe_overhead_tracks_and_decrements(self):
        # gap=300, overhead=0.2h=720s -> two ticks to clear for the probe
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        class ProbeStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self.launched_main = False
                self.probe_launched = False
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if not self.launched_main:
                    res = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert res and res.success
                    self.launched_main = True
                    return
                # After two ticks, launch a probe at region 1
                if env.tick >= 2 and not self.probe_launched:
                    res = yield ProbeLaunch(region=1)
                    if res and res.success:
                        self.probe_launched = True

        class Args:
            deadline_hours = 10.0
            restart_overhead_hours = [0.2]
            inter_task_overhead = [0.0]

        strat = ProbeStrategy(Args())
        task = MockTask(duration_seconds=10 * self.gap_seconds)
        strat.reset(env, task)

        # Tick 0: launch main
        env.observe(); env.update_strategy_progress(strat); env.execute_multi_strategy(strat); env.tick += 1
        # Tick 1: overhead applies; still no probe
        env.observe(); env.update_strategy_progress(strat); env.execute_multi_strategy(strat); env.tick += 1
        # Tick 2: finish overhead; launch probe
        env.observe(); env.update_strategy_progress(strat); env.execute_multi_strategy(strat); env.tick += 1
        # Tick 3: after probe launched last tick, its overhead should be initialized and decremented by gap
        env.observe(); env.update_strategy_progress(strat)
        # Probe launches should not introduce restart overhead; remaining overhead
        # refers only to the leader and is unaffected by the probe.
        self.assertLessEqual(strat.remaining_restart_overhead, 1e-6)
        # Next tick: still no overhead for probe region
        env.execute_multi_strategy(strat); env.tick += 1
        env.observe(); env.update_strategy_progress(strat)
        self.assertLessEqual(strat.remaining_restart_overhead, 1e-6)
        # Next tick also remains unaffected
        env.execute_multi_strategy(strat); env.tick += 1
        env.observe(); env.update_strategy_progress(strat)
        self.assertLessEqual(strat.remaining_restart_overhead, 1e-6)

    def test_safety_net_allows_working_spot(self):
        # Working SPOT with no remaining overhead should not be overridden by safety net
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        class IdleStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if False:
                    yield  # pragma: no cover
                return

        class Args:
            # Tight deadline to trigger safety net check
            deadline_hours = 1.0
            restart_overhead_hours = [0.0]
            inter_task_overhead = [0.0]

        strat = IdleStrategy(Args())
        task = MockTask(duration_seconds=10 * self.gap_seconds)
        strat.reset(env, task)

        # Tick 0: launch a SPOT manually to simulate working instance
        env.observe()
        ok = env._try_launch_internal(0, ClusterType.SPOT)
        self.assertTrue(ok)
        env.tick += 1

        # At tick 1, remaining_restart_overhead is 0 and safety net should allow SPOT
        env.observe(); env.update_strategy_progress(strat)
        env.execute_multi_strategy(strat)
        active = env.get_active_instances()
        # Should still be SPOT only, not forced to ON_DEMAND
        self.assertEqual(active, {0: ClusterType.SPOT})

    def test_safety_net_emergency_recovery_complete(self):
        """Test complete safety net emergency recovery flow"""
        
        gap_seconds = 600
        deadline_hours = 15.0  
        task_duration_hours = 9.5  
        
        # Create a simple trace
        trace_data = {
            'metadata': {
                'price_info': {'on_demand_price': 3.06, 'price': 0.918},
                'gap_seconds': gap_seconds,
                'region': 'test-0',
                'zone': 'test-0a', 
                'instance_type': 'v100',
                'device': 'v100_1'
            },
            'data': [0] * 200,  
            'price': [1.0] * 200,
        }
        fp = os.path.join(self.temp_dir, 'region_0_v100_1.json')
        with open(fp, 'w') as f:
            json.dump(trace_data, f)
        
        env = MultiTraceEnv([fp], env_start_hours=0)
        
        class DummyStrategy(MultiRegionStrategy):
            NAME = 'dummy_emergency'
            def _step_multi(self): 
                return
                yield  # Make it a generator
            @classmethod
            def _from_args(cls, parser): 
                return cls(parser.parse_args())
        
        task = MockTask(duration_seconds=task_duration_hours * 3600)
        
        class Args:
            def __init__(self):
                self.deadline_hours = deadline_hours
                self.restart_overhead_hours = [0.2]
                self.inter_task_overhead = [0.0]
            
        strat = DummyStrategy(Args())
        strat.reset(env, task)
        
        # === PHASE 1: Set up safety net latched state with SPOT instance ===
        
        # Launch SPOT and set up latched state manually
        env.observe()
        ok = env._try_launch_internal(0, ClusterType.SPOT)
        self.assertTrue(ok)
        env.tick += 1
        
        # Manually set safety net to latched state
        env._safety_net_latched = True
        
        # Verify normal latched behavior with 1 instance
        env.observe()
        env.update_strategy_progress(strat)
        
        latched = env.maybe_trigger_safety_net(strat)
        self.assertTrue(latched, "Safety net should remain latched with 1 instance")
        
        active = env.get_active_instances()
        self.assertEqual(active, {0: ClusterType.SPOT}, "Should have SPOT in region 0")
        
        # === PHASE 2: Test emergency recovery when latched instance is preempted ===
        
        # Simulate preemption by clearing active instances
        env.active_instances.clear()
        
        # Safety net should detect 0 instances and launch emergency ON_DEMAND
        env.tick += 1
        env.observe()
        env.update_strategy_progress(strat)
        
        latched_after_emergency = env.maybe_trigger_safety_net(strat)
        
        # Verify emergency recovery worked
        self.assertTrue(latched_after_emergency, "Safety net should remain latched during emergency recovery")
        
        # Should have launched ON_DEMAND in region 0
        final_active = env.get_active_instances()
        self.assertEqual(len(final_active), 1, "Should have exactly one instance after emergency recovery")
        self.assertIn(0, final_active, "Should launch in region 0 for emergency recovery")
        self.assertEqual(final_active[0], ClusterType.ON_DEMAND, "Should be ON_DEMAND for emergency recovery")

    def test_safety_net_assertion_error(self):
        """Test the assertion error when safety net is latched but no instances exist"""
        
        gap_seconds = 600
        deadline_hours = 15.0  # Large deadline to avoid infeasible error
        task_duration_hours = 9.5
        
        # Create a simple trace
        trace_data = {
            'metadata': {
                'price_info': {'on_demand_price': 3.06, 'price': 0.918},
                'gap_seconds': gap_seconds,
                'region': 'test-0',
                'zone': 'test-0a', 
                'instance_type': 'v100',
                'device': 'v100_1'
            },
            'data': [0] * 100,  # Always available
            'price': [1.0] * 100,
        }
        fp = os.path.join(self.temp_dir, 'region_0_v100_1.json')
        with open(fp, 'w') as f:
            json.dump(trace_data, f)
        
        env = MultiTraceEnv([fp], env_start_hours=0)
        
        class DummyStrategy(MultiRegionStrategy):
            NAME = 'dummy_assertion'
            def _step_multi(self): 
                return
                yield  # Make it a generator
            @classmethod
            def _from_args(cls, parser): 
                return cls(parser.parse_args())
        
        task = MockTask(duration_seconds=task_duration_hours * 3600)
        
        class Args:
            def __init__(self):
                self.deadline_hours = deadline_hours
                self.restart_overhead_hours = [0.2]
                self.inter_task_overhead = [0.0]
            
        strat = DummyStrategy(Args())
        strat.reset(env, task)
        
        # Manually set up the problematic state:
        # 1. Safety net is latched
        # 2. But no active instances
        env._safety_net_latched = True
        env.active_instances.clear()  # No instances
        
        # This should trigger emergency recovery, not assertion error
        env.observe()
        env.update_strategy_progress(strat)
        
        # Safety net should detect 0 instances and launch emergency ON_DEMAND
        latched = env.maybe_trigger_safety_net(strat)
        
        # Verify emergency recovery worked
        self.assertTrue(latched, "Safety net should remain latched during emergency recovery")
        active_after_recovery = env.get_active_instances()
        self.assertEqual(len(active_after_recovery), 1, "Should have exactly one instance after emergency recovery")
        self.assertIn(0, active_after_recovery, "Should launch in region 0 for emergency recovery")
        self.assertEqual(active_after_recovery[0], ClusterType.ON_DEMAND, "Should be ON_DEMAND for emergency recovery")
        
    def test_safety_net_boundary_calculation(self):
        """Test that boundary=True is triggered at the correct time"""
        
        gap_seconds = 600
        task_duration_hours = 2.0  # 2 hour task  
        deadline_hours = 2.5       # 2.5 hour deadline (tight but feasible)
        restart_overhead_seconds = 0.2 * 3600  # 12 minutes
        
        # Create a simple trace
        trace_data = {
            'metadata': {
                'price_info': {'on_demand_price': 3.06, 'price': 0.918},
                'gap_seconds': gap_seconds,
                'region': 'test-0',
                'zone': 'test-0a', 
                'instance_type': 'v100',
                'device': 'v100_1'
            },
            'data': [0] * 100,  # Always available
            'price': [1.0] * 100,
        }
        fp = os.path.join(self.temp_dir, 'region_0_v100_1.json')
        with open(fp, 'w') as f:
            json.dump(trace_data, f)
        
        env = MultiTraceEnv([fp], env_start_hours=0)
        
        class DummyStrategy(MultiRegionStrategy):
            NAME = 'dummy_boundary'
            def _step_multi(self): 
                return
                yield  # Make it a generator
            @classmethod
            def _from_args(cls, parser): 
                return cls(parser.parse_args())
        
        task = MockTask(duration_seconds=task_duration_hours * 3600)
        
        class Args:
            def __init__(self):
                self.deadline_hours = deadline_hours
                self.restart_overhead_hours = [restart_overhead_seconds / 3600]  # Convert to hours
                self.inter_task_overhead = [0.0]
            
        strat = DummyStrategy(Args())
        strat.reset(env, task)
        
        # Launch an instance to test with
        env.observe()
        ok = env._try_launch_internal(0, ClusterType.SPOT)
        self.assertTrue(ok)
        env.tick += 1
        
        # Run for several ticks until we hit the boundary
        import math
        found_boundary = False
        for _ in range(20):  # Reasonable upper limit
            env.observe()
            env.update_strategy_progress(strat)
            
            # Check if we've hit the boundary condition
            elapsed_seconds = env.tick * gap_seconds
            remaining_time_seconds = strat.deadline - elapsed_seconds
            remaining_task_seconds = task.duration_seconds - env._current_leader_progress
            needed = math.ceil((remaining_task_seconds + restart_overhead_seconds) / gap_seconds) * gap_seconds
            
            latched = env.maybe_trigger_safety_net(strat)
            
            if needed >= remaining_time_seconds:
                # We've hit the boundary - safety net should latch
                self.assertTrue(latched, f"Should latch when boundary condition met: needed={needed} >= remaining_time={remaining_time_seconds}")
                found_boundary = True
                break
            else:
                # Not at boundary yet - should not latch
                self.assertFalse(latched, f"Should not latch before boundary: needed={needed} < remaining_time={remaining_time_seconds}")
            
            env.tick += 1
        
        self.assertTrue(found_boundary, "Should reach boundary condition within reasonable time")


if __name__ == '__main__':
    unittest.main()
