"""Tests for multi-region billing system."""

import json
import os
import tempfile
import unittest
from typing import Dict, Optional, Generator

from sky_spot.env import MultiTraceEnv, PROBE_BILLING_SECONDS
from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.multi_region_types import TryLaunch, Terminate, LaunchResult, Action, ProbeLaunch
from sky_spot.utils import ClusterType
from sky_spot.trace import Trace
from sky_spot import task as task_lib


class MockTask(task_lib.Task):
    """Mock task for testing."""
    
    def __init__(self, duration_seconds: float):
        self.duration_seconds = duration_seconds
        self.checkpoint_size_gb = 50.0
        self._progress_source = []
    
    def set_progress_source(self, progress_source):
        self._progress_source = progress_source
    
    def get_total_duration_seconds(self) -> float:
        return self.duration_seconds
    
    def get_info(self) -> dict:
        progress = sum(self._progress_source)
        return {
            'progress': progress,
            'remaining': self.duration_seconds - progress,
            'total': self.duration_seconds
        }
    
    @property
    def is_done(self) -> bool:
        return sum(self._progress_source) >= self.duration_seconds
    
    def get_config(self) -> dict:
        return {'duration_seconds': self.duration_seconds}
    
    def reset(self):
        """Reset the task state."""
        self._progress_source = []
    
    def __str__(self) -> str:
        """String representation of the task."""
        return f"MockTask(duration={self.duration_seconds}s)"


class MockArgs:
    """Mock arguments for strategy initialization."""
    def __init__(self):
        self.deadline_hours = 2.0  # Increased to prevent SAFETY NET triggering
        self.restart_overhead_hours = [0.05]  # 3 minutes
        self.inter_task_overhead = [0.0]


class TestMultiRegionBilling(unittest.TestCase):
    """Test multi-region billing scenarios."""
    
    def setUp(self):
        """Create mock trace files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.gap_seconds = 60  # 1 minute per tick
        
        # Keep compatibility with static per-device pricing from env
        # Specific per-tick spot price will override if present in traces
        self.base_price = 3.06
        self.ondemand_price = self.base_price
        
        # Create mock traces with known availability patterns
        # Region 0: Always available
        self.create_trace_file(0, [0] * 100)  # 0 = available
        
        # Region 1: Available from tick 0-49, unavailable 50-99
        self.create_trace_file(1, [0] * 50 + [1] * 50)  # 1 = unavailable
        
        # Region 2: Alternating availability
        self.create_trace_file(2, [0, 1] * 50)
        
        self.trace_files = [
            os.path.join(self.temp_dir, f"us-east-{i}a_v100_1", "0.json")
            for i in range(3)
        ]
        
    def create_trace_file(self, region_id: int, availability: list, *, metadata_override: Optional[Dict] = None):
        """Create a mock trace file."""
        trace_data = {
            "metadata": {
                "price_info": {"on_demand_price": 3.06, "price": 0.918},
                "region": f"us-east-{region_id}a",
                "instance_type": "v100",
                "start_time": "2024-01-01T00:00:00Z",
                "gap_seconds": self.gap_seconds,
                "device": "v100_1"
            },
            "data": availability
            # Don't include prices by default - TraceEnv will infer spot price
        }
        if metadata_override:
            trace_data["metadata"].update(metadata_override)

        # Create directory structure like real traces
        region_dir = os.path.join(self.temp_dir, f"us-east-{region_id}a_v100_1")
        os.makedirs(region_dir, exist_ok=True)
        filepath = os.path.join(region_dir, "0.json")
        with open(filepath, 'w') as f:
            json.dump(trace_data, f)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_single_region_single_tick(self):
        """Test billing for single region, single tick."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        # Define a simple strategy that launches SPOT in region 0
        class SimpleStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                assert result.success
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)  # 1 hour task
        strategy = SimpleStrategy(args)
        strategy.reset(env, task)
        
        # Execute one tick
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Need to observe again to finalize costs from previous tick
        env.observe()
        
        # Check costs (1 minute of SPOT = spot_price[tick0] * 60 / 3600)
        # If dynamic price exists, use it; else use env's inferred spot price
        if env.envs[0]._spot_price is not None:
            price0 = env.envs[0]._spot_price
        else:
            price0 = env.envs[0].trace.get_price(env.envs[0]._start_index + 0)
        expected_cost = float(price0) * self.gap_seconds / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected_cost, places=6)
        
        # Check cost breakdown
        breakdown = env.get_cost_breakdown()
        self.assertEqual(breakdown['tick_count'], 1)
        self.assertAlmostEqual(breakdown['by_region'][0], expected_cost, places=6)
        self.assertAlmostEqual(breakdown['by_type'][ClusterType.SPOT], expected_cost, places=6)

    def test_probe_billing_fixed_one_minute(self):
        """Ensure probe launches are billed as a fixed one-minute charge."""

        class ProbeStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                result = yield ProbeLaunch(region=0)
                assert result.success

        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        args = MockArgs()
        task = MockTask(duration_seconds=self.gap_seconds)
        strategy = ProbeStrategy(args)
        strategy.reset(env, task)

        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        env.observe()

        sub_env = env.envs[0]
        if sub_env._spot_price is not None:
            spot_price = float(sub_env._spot_price)
        else:
            spot_price = float(sub_env.trace.get_price(sub_env._start_index))

        expected_probe_cost = spot_price * PROBE_BILLING_SECONDS / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected_probe_cost, places=6)
        breakdown = env.get_cost_breakdown()
        self.assertIn('probe_total', breakdown)
        self.assertAlmostEqual(float(breakdown['probe_total']), expected_probe_cost, places=6)
        self.assertAlmostEqual(float(breakdown['transfer_total']), 0.0, places=6)

    def test_probe_does_not_incurs_transfer_fee(self):
        """Probe launches should not add transfer cost even across regions."""

        class ProbeWithMigration(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                assert result.success
                result = yield ProbeLaunch(region=1)
                assert result.success

        env = MultiTraceEnv(self.trace_files, env_start_hours=0)

        args = MockArgs()
        task = MockTask(duration_seconds=self.gap_seconds)
        strategy = ProbeWithMigration(args)
        strategy.reset(env, task)

        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        env.observe()

        breakdown = env.get_cost_breakdown()
        self.assertEqual(int(breakdown.get('CrossRegionMigrations', 0)), 0)
        self.assertEqual(float(breakdown.get('transfer_total', 0.0)), 0.0)

    def test_metadata_on_demand_price(self):
        """Trace metadata on_demand_price should override device defaults."""

        override_price = 42.123
        self.create_trace_file(5, [0, 0], metadata_override={'price_info': {'on_demand_price': override_price, 'price': 0.918}})
        env = MultiTraceEnv([
            os.path.join(self.temp_dir, 'us-east-5a_v100_1', '0.json')
        ], env_start_hours=0)

        self.assertAlmostEqual(env.envs[0]._base_price, override_price, places=6)

    def test_region_specific_dynamic_prices(self):
        """Different regions with different per-tick prices should bill accordingly."""
        # Create two traces with identical availability but different dynamic prices
        # Region 0: prices [0.5, 0.6]
        # Region 1: prices [0.7, 0.8]
        def write_trace_with_prices(idx: int, prices: list[float]):
            trace_data = {
                "metadata": {
                    "price_info": {"on_demand_price": 3.06, "price": 0.918},
                    "region": f"test-region-{idx}",
                    "instance_type": "test.large",
                    "start_time": "2024-01-01T00:00:00Z",
                    "gap_seconds": self.gap_seconds,
                    "device": "v100_1"
                },
                "data": [0] * len(prices),
                "prices": prices,
            }
            filepath = os.path.join(self.temp_dir, f"region_{idx}_v100_1.json")
            with open(filepath, 'w') as f:
                json.dump(trace_data, f)

        # Use length 3 to allow a final observe() without hitting bounds
        write_trace_with_prices(0, [0.5, 0.6, 0.6])
        write_trace_with_prices(1, [0.7, 0.8, 0.8])
        trace_files = [
            os.path.join(self.temp_dir, f"region_{i}_v100_1.json") for i in range(2)
        ]

        env = MultiTraceEnv(trace_files, env_start_hours=0)

        class AlternatingStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self._tick = 0
            def _step_multi(self):
                if self._tick == 0:
                    # Launch in region 0 first
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result.success
                elif self._tick == 1:
                    # Switch to region 1
                    yield Terminate(region=0)
                    result = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
                    assert result.success
                self._tick += 1

        args = MockArgs()
        task = MockTask(duration_seconds=2 * self.gap_seconds)  # 2 ticks
        strategy = AlternatingStrategy(args)
        strategy.reset(env, task)

        # Tick 0: Region 0
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        # Tick 1: Switch to region 1
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        # Finalize
        env.observe()

        # Expected: Tick 0 uses price 0.5 from region 0, Tick 1 uses price 0.8 from region 1
        expected = (0.5 + 0.8) * self.gap_seconds / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected, places=6)

        breakdown = env.get_cost_breakdown()
        expected_r0 = 0.5 * self.gap_seconds / 3600  # Only tick 0
        expected_r1 = 0.8 * self.gap_seconds / 3600  # Only tick 1
        self.assertAlmostEqual(breakdown['by_region'][0], expected_r0, places=6)
        self.assertAlmostEqual(breakdown['by_region'][1], expected_r1, places=6)
    
    def test_multi_region_sequential(self):
        """Test billing for sequential execution across multiple regions."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class SequentialStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                tick = getattr(self, '_tick', 0)
                
                if tick == 0:
                    # Launch SPOT in region 0
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result.success
                elif tick == 1:
                    # Switch to ON_DEMAND in region 1
                    yield Terminate(region=0)
                    result = yield TryLaunch(region=1, cluster_type=ClusterType.ON_DEMAND)
                    assert result.success
                
                self._tick = tick + 1
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)  # 1 hour task
        strategy = SequentialStrategy(args)
        strategy.reset(env, task)
        
        # Tick 0: Launch SPOT in region 0
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Tick 1: Switch to ON_DEMAND in region 1
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Final observe to finalize costs
        env.observe()
        
        # Check costs (SPOT for tick 0, ON_DEMAND for tick 1)
        if env.envs[0]._spot_price is not None:
            actual_spot_price = env.envs[0]._spot_price
        else:
            actual_spot_price = env.envs[0].trace.get_price(env.envs[0]._start_index + 0)
        actual_ondemand_price = env.envs[1]._base_price
        expected_cost = (float(actual_spot_price) + float(actual_ondemand_price)) * self.gap_seconds / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected_cost, places=5)
        
        # Check breakdown
        breakdown = env.get_cost_breakdown()
        self.assertAlmostEqual(
            breakdown['by_region'][0], 
            float(actual_spot_price) * self.gap_seconds / 3600, 
            places=5
        )
        self.assertAlmostEqual(
            breakdown['by_region'][1], 
            actual_ondemand_price * self.gap_seconds / 3600, 
            places=5
        )
    
    def test_terminate_still_charged(self):
        """Test that terminating an instance at tick start doesn't charge for that tick."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class TerminateStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self.first_tick = True
                
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if self.first_tick:
                    # First tick: launch SPOT
                    self.first_tick = False
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result.success
                else:
                    # Second tick: terminate (should not be charged for this tick)
                    yield Terminate(region=0)
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)  # 1 hour task
        strategy = TerminateStrategy(args)
        strategy.reset(env, task)
        
        # Tick 1: Launch
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Tick 2: Terminate
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Final observe to finalize costs
        env.observe()
        
        # Should only be charged for 1 tick (terminated at start of tick 2)
        if env.envs[0]._spot_price is not None:
            p = env.envs[0]._spot_price
        else:
            p = env.envs[0].trace.get_price(env.envs[0]._start_index + 0)
        expected_cost = 1 * float(p) * self.gap_seconds / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected_cost, places=6)
        
        # Verify no active instances after termination
        self.assertEqual(len(env.get_active_instances()), 0)
    
    def test_failed_launch_no_charge(self):
        """Test that failed launches don't incur charges."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class FailedLaunchStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                # Try to launch in region 1 at tick 50 (unavailable)
                result = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
                assert not result.success  # Should fail
                
                # Fall back to ON_DEMAND
                result = yield TryLaunch(region=1, cluster_type=ClusterType.ON_DEMAND)
                assert result.success
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)  # 1 hour task
        strategy = FailedLaunchStrategy(args)
        strategy.reset(env, task)
        
        # Skip to tick 50 where region 1 SPOT is unavailable
        # Need to properly advance ticks through observe/step cycle
        for i in range(50):
            env.observe()
            env.tick += 1
        
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Final observe to finalize costs
        env.observe()
        
        # Should only be charged for ON_DEMAND, not failed SPOT
        actual_ondemand_price = env.envs[1]._base_price
        expected_cost = actual_ondemand_price * self.gap_seconds / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected_cost, places=6)
    
    def test_preemption_handling(self):
        """Test that preempted instances stop charging."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class SimpleSpotStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                active = self.env.get_active_instances()
                if not active:
                    # Try to launch SPOT in region 1 first (will fail after tick 50)
                    result = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
                    if not result.success:
                        # If region 1 fails, try region 0 (always available)
                        result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    if result.success:
                        return
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)  # 1 hour task
        strategy = SimpleSpotStrategy(args)
        strategy.reset(env, task)
        
        # Run for several ticks
        launched_ticks = 0
        for i in range(55):  # Will be preempted at tick 50
            env.observe()
            
            # CRITICAL: Must call update_strategy_progress to handle preemptions
            env.update_strategy_progress(strategy)
            
            # Check if instance is active before execution
            active_before = len(env.get_active_instances()) > 0
            
            # Strategy will try to launch if no active instances
            env.execute_multi_strategy(strategy)
            
            # Count ticks where instance is active after execution
            active_after = len(env.get_active_instances()) > 0
            if active_after:
                launched_ticks += 1
            
            env.tick += 1
        
        # Final observe to get final costs
        env.observe()
        
        # Should be charged for the number of ticks the instance was active
        # The instance gets preempted at tick 50, but may relaunch
        if env.envs[1]._spot_price is not None:
            p = float(env.envs[1]._spot_price)
        else:
            p = float(env.envs[1].trace.get_price(env.envs[1]._start_index + 0))
        
        # The actual cost should match the number of active ticks
        # Check the actual billing history
        actual_billed_ticks = len(env.cost_history)
        expected_cost = actual_billed_ticks * p * self.gap_seconds / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected_cost, places=6)
        
        # After tick 50, region 1 becomes unavailable
        # The instance should have been preempted and possibly relaunched in region 0
        final_active = env.get_active_instances()
        
        if final_active:
            # After preemption from region 1, strategy should have launched in region 0
            self.assertIn(0, final_active, "Should have relaunched in region 0 after region 1 preemption")
            self.assertNotIn(1, final_active, "Region 1 should not be active (unavailable after tick 50)")
        
    def test_complex_scenario(self):
        """Test a complex scenario with multiple operations."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class ComplexStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self.tick_count = 0
                
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                active = self.env.get_active_instances()
                
                if self.tick_count == 0:
                    # Start with SPOT in region 0
                    yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                elif self.tick_count == 2:
                    # Switch to SPOT in region 1
                    yield Terminate(region=0)
                    yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
                elif self.tick_count == 5:
                    # Switch to ON_DEMAND in region 2
                    yield Terminate(region=1)
                    yield TryLaunch(region=2, cluster_type=ClusterType.ON_DEMAND)
                
                self.tick_count += 1
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)  # 1 hour task
        strategy = ComplexStrategy(args)
        strategy.reset(env, task)
        
        # Run for 10 ticks
        costs_per_tick = []
        for i in range(10):
            env.observe()
            env.execute_multi_strategy(strategy)
            env.tick += 1
            
            # Calculate cost for this tick after incrementing tick
            # (costs are finalized in the next observe)
        
        # Final observe to finalize all costs
        env.observe()
        
        # Now calculate costs per tick from the history
        cost_history = env.cost_history
        for i in range(len(cost_history)):
            tick_costs = cost_history[i]
            tick_cost = 0
            for region, ctype in tick_costs.items():
                cost_map = env.envs[region].get_constant_cost_map()
                tick_cost += cost_map[ctype] * self.gap_seconds / 3600
            costs_per_tick.append(tick_cost)
        
        # Verify costs match expected pattern:
        # Ticks 0-1: SPOT in region 0
        # Tick 2-4: SPOT in region 1 (switched from region 0)
        # Tick 5-9: ON_DEMAND in region 2 (switched from region 1)
        
        # Get actual prices
        spot_price_r0 = float(env.envs[0]._spot_price if env.envs[0]._spot_price else 
                              env.envs[0].trace.get_price(env.envs[0]._start_index + 0))
        spot_price_r1 = float(env.envs[1]._spot_price if env.envs[1]._spot_price else
                              env.envs[1].trace.get_price(env.envs[1]._start_index + 0))
        ondemand_price = env.envs[2]._base_price
        
        # Calculate per-tick costs
        spot_r0_cost = spot_price_r0 * self.gap_seconds / 3600
        spot_r1_cost = spot_price_r1 * self.gap_seconds / 3600
        ondemand_cost = ondemand_price * self.gap_seconds / 3600
        
        expected_costs = [
            spot_r0_cost,     # 0: SPOT in region 0
            spot_r0_cost,     # 1: SPOT in region 0
            spot_r1_cost,     # 2: SPOT in region 1
            spot_r1_cost,     # 3: SPOT in region 1
            spot_r1_cost,     # 4: SPOT in region 1
            ondemand_cost,    # 5: ON_DEMAND in region 2
            ondemand_cost,    # 6: ON_DEMAND in region 2
            ondemand_cost,    # 7: ON_DEMAND in region 2
            ondemand_cost,    # 8: ON_DEMAND in region 2
            ondemand_cost,    # 9: ON_DEMAND in region 2
        ]
        
        for i, (actual, expected) in enumerate(zip(costs_per_tick, expected_costs)):
            self.assertAlmostEqual(
                actual, expected, places=6,
                msg=f"Cost mismatch at tick {i}: expected {expected}, got {actual}"
            )
        
        # Verify final state - only ON_DEMAND in region 2 should be active
        final_active = env.get_active_instances()
        self.assertEqual(len(final_active), 1, "Should have exactly one instance")
        self.assertEqual(final_active[2], ClusterType.ON_DEMAND)
        self.assertNotIn(0, final_active)
        self.assertNotIn(1, final_active)
    
    def test_same_tick_launch_terminate_error(self):
        """Test that terminating after launch in same tick fails (no instance yet)."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class SameTickStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                # Launch SPOT instance (will be buffered, not active yet)
                result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                assert result.success
                
                # Try to terminate in the same tick
                # This should fail because we can't terminate just-launched instances
                yield Terminate(region=0, cluster_type=ClusterType.SPOT)
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)
        strategy = SameTickStrategy(args)
        strategy.reset(env, task)
        
        # Should raise ValueError
        env.observe()
        with self.assertRaises(ValueError) as cm:
            env.execute_multi_strategy(strategy)
        
        # The error is because we can't terminate an instance launched in the same tick
        self.assertIn("just launched in the same tick", str(cm.exception))
    
    def test_launch_and_probe_parallel_allowed(self):
        """Test that TryLaunch and ProbeLaunch can happen in parallel (probe doesn't violate single-instance)."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class ParallelProbeStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                # Launch a real instance in region 0
                result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                assert result.success
                
                # Probe region 1 in parallel (this should be allowed)
                probe_result = yield ProbeLaunch(region=1)
                # Probe result tells us if spot is available
                
                # Can also probe multiple regions
                probe_result2 = yield ProbeLaunch(region=2)
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)
        strategy = ParallelProbeStrategy(args)
        strategy.reset(env, task)
        
        env.observe()
        # This should NOT raise an assertion error about single-instance invariant
        env.execute_multi_strategy(strategy)
        env.tick += 1
        env.observe()
        
        # Check billing: should have main instance + probe charges
        # Main instance: SPOT in region 0
        # Probes: 60 seconds each in regions 1 and 2 (if successful)
        breakdown = env.get_cost_breakdown()
        
        # Should have active instance in region 0
        active = env.get_active_instances()
        self.assertEqual(len(active), 1, "Should have exactly one active instance")
        self.assertEqual(active[0], ClusterType.SPOT)
        
        # Check probe billing was recorded
        self.assertGreater(env.probe_cost_total, 0, "Should have probe costs")
    
    def test_terminate_wrong_type_fails(self):
        """Test that terminating a non-existent cluster type fails gracefully."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class WrongTerminateStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self.tick_count = 0
                
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if self.tick_count == 0:
                    # Launch SPOT in region 0
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result.success
                elif self.tick_count == 1:
                    # Try to terminate ON_DEMAND in region 0 (but SPOT is running!)
                    # This should not crash but log that the instance wasn't found
                    yield Terminate(region=0, cluster_type=ClusterType.ON_DEMAND)
                    # The SPOT instance should still be running
                
                self.tick_count += 1
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)
        strategy = WrongTerminateStrategy(args)
        strategy.reset(env, task)
        
        # Tick 0: Launch SPOT
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Tick 1: Try to terminate wrong type
        env.observe()
        env.execute_multi_strategy(strategy)  # Should not crash
        env.tick += 1
        
        # Check that SPOT is still running (wrong terminate didn't affect it)
        active = env.get_active_instances()
        self.assertEqual(len(active), 1, "Should still have the SPOT instance")
        self.assertEqual(active[0], ClusterType.SPOT, "SPOT should still be active")


if __name__ == '__main__':
    unittest.main() 
