import json
import os
import tempfile
import unittest
from typing import Optional, Generator

from sky_spot.env import MultiTraceEnv
from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.multi_region_types import TryLaunch, Terminate, LaunchResult, Action
from sky_spot.utils import ClusterType


class TestInstanceReplacementBilling(unittest.TestCase):
    """Test that billing correctly charges for the previous tick's instance type
    when instances are replaced within the same tick."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.gap_seconds = 600  # 10 minutes
        
        # Create test trace files with different prices
        self.spot_price = 0.5  # $0.5/hour for SPOT
        self.ondemand_price = 1.5  # $1.5/hour for ON_DEMAND
        
        for i in range(2):
            trace_data = {
                'metadata': {
                    'gap_seconds': self.gap_seconds,
                    'region': f'test-region-{i}',
                    'zone': f'test-{i}a',
                    'instance_type': 'v100',
                    'device': 'v100_1',
                    'price_info': {
                        'price': self.spot_price,
                        'on_demand_price': self.ondemand_price,
                    }
                },
                'data': [0] * 100,  # Always available
                'prices': [self.spot_price] * 100,  # Dynamic spot pricing
            }
            fp = os.path.join(self.temp_dir, f'region_{i}_v100_1.json')
            with open(fp, 'w') as f:
                json.dump(trace_data, f)
        
        self.trace_files = [
            os.path.join(self.temp_dir, f'region_{i}_v100_1.json')
            for i in range(2)
        ]
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_same_tick_replacement_billing(self):
        """Test that when replacing SPOT with ON_DEMAND in the same tick,
        the billing for that tick reflects the ON_DEMAND instance that is running
        after the replacement (since active_instances represents tick [i, i+1))."""
        
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class ReplacementStrategy(MultiRegionStrategy):
            def __init__(self):
                self.tick_actions = {}
                super().__init__(None)
            
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                tick = env.tick
                
                if tick == 0:
                    # Tick 0: Launch SPOT
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result and result.success
                    
                elif tick == 2:
                    # Tick 2: Replace SPOT with ON_DEMAND in same tick
                    # First launch ON_DEMAND
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.ON_DEMAND)
                    assert result and result.success
                    # Then terminate the old SPOT (must specify type for same-tick replacement)
                    yield Terminate(region=0, cluster_type=ClusterType.SPOT)
        
        strategy = ReplacementStrategy()
        
        # Tick 0: Launch SPOT
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Tick 1: Just run SPOT (no changes)
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # At this point we should have charged for:
        # - Tick 0: SPOT ($0.5/hour * 600s/3600s = $0.0833...)
        
        # Tick 2: Replace SPOT with ON_DEMAND
        env.observe()  # This finalizes billing for tick 1 (SPOT running)
        env.execute_multi_strategy(strategy)  # Replace happens here
        env.tick += 1
        
        # At this point we should have charged for:
        # - Tick 0: SPOT
        # - Tick 1: SPOT (even though replacement happens in tick 2)
        
        # Tick 3: Run ON_DEMAND
        env.observe()  # This finalizes billing for tick 2 (should bill SPOT, not ON_DEMAND)
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Check cost history
        self.assertEqual(len(env.cost_history), 3, "Should have 3 billing entries")
        
        # Verify tick 0 billing: SPOT
        self.assertEqual(env.cost_history[0], {0: ClusterType.SPOT})
        
        # Verify tick 1 billing: SPOT
        self.assertEqual(env.cost_history[1], {0: ClusterType.SPOT})
        
        # Verify tick 2 billing: Should be ON_DEMAND (what is running during tick [2,3))
        # This is the critical test - since ON_DEMAND replaced SPOT during tick 2,
        # billing should reflect the ON_DEMAND that is running during tick 2
        self.assertEqual(env.cost_history[2], {0: ClusterType.ON_DEMAND},
                        "Tick 2 should bill ON_DEMAND (what runs during tick [2,3)), not SPOT")
        
        # Calculate expected costs
        spot_cost_per_tick = self.spot_price * self.gap_seconds / 3600
        ondemand_cost_per_tick = self.ondemand_price * self.gap_seconds / 3600
        expected_cost = spot_cost_per_tick * 2 + ondemand_cost_per_tick * 1  # 2 SPOT + 1 ON_DEMAND
        
        # Verify total cost
        actual_cost = env.accumulated_cost
        self.assertAlmostEqual(actual_cost, expected_cost, places=4,
                              msg=f"Cost should be 2 SPOT + 1 ON_DEMAND. Expected: {expected_cost}, Got: {actual_cost}")
    
    def test_next_tick_bills_replacement(self):
        """Test that the replacement instance (ON_DEMAND) is billed starting
        from the tick when the replacement happens."""
        
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class ReplacementStrategy(MultiRegionStrategy):
            def __init__(self):
                super().__init__(None)
            
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                tick = env.tick
                
                if tick == 0:
                    # Launch SPOT initially
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result and result.success
                    
                elif tick == 1:
                    # Replace with ON_DEMAND
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.ON_DEMAND)
                    assert result and result.success
                    yield Terminate(region=0, cluster_type=ClusterType.SPOT)
        
        strategy = ReplacementStrategy()
        
        # Execute through several ticks
        for i in range(4):
            env.observe()
            env.execute_multi_strategy(strategy)
            env.tick += 1
        
        # Final observe to finalize tick 3 costs
        env.observe()
        
        # Check cost history
        self.assertEqual(len(env.cost_history), 4)
        
        # Tick 0: SPOT
        self.assertEqual(env.cost_history[0], {0: ClusterType.SPOT})
        
        # Tick 1: ON_DEMAND (replacement happens during this tick)
        self.assertEqual(env.cost_history[1], {0: ClusterType.ON_DEMAND})
        
        # Tick 2: ON_DEMAND (continues running)
        self.assertEqual(env.cost_history[2], {0: ClusterType.ON_DEMAND})
        
        # Tick 3: ON_DEMAND
        self.assertEqual(env.cost_history[3], {0: ClusterType.ON_DEMAND})
        
        # Verify total cost
        spot_cost_per_tick = self.spot_price * self.gap_seconds / 3600
        ondemand_cost_per_tick = self.ondemand_price * self.gap_seconds / 3600
        expected_cost = spot_cost_per_tick * 1 + ondemand_cost_per_tick * 3  # 1 SPOT + 3 ON_DEMAND
        
        actual_cost = env.accumulated_cost
        self.assertAlmostEqual(actual_cost, expected_cost, places=4,
                              msg=f"Cost mismatch. Expected: {expected_cost}, Got: {actual_cost}")
    
    def test_terminated_instance_still_billed(self):
        """Test that a terminated instance is still billed for the tick
        in which it was terminated."""
        
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class TerminateStrategy(MultiRegionStrategy):
            def __init__(self):
                super().__init__(None)
            
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                tick = env.tick
                
                if tick == 0:
                    # Launch SPOT
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result and result.success
                    
                elif tick == 2:
                    # Terminate without replacement
                    yield Terminate(region=0)
        
        strategy = TerminateStrategy()
        
        # Execute through tick 3
        for i in range(4):
            env.observe()
            env.execute_multi_strategy(strategy)
            env.tick += 1
        
        # Final observe
        env.observe()
        
        # Check cost history
        # Should only bill for ticks 0 and 1 (SPOT running)
        # Tick 2 has no instance running after termination, so no billing
        self.assertEqual(len(env.cost_history), 2, "Should bill for 2 ticks (0, 1)")
        
        # Both ticks should bill SPOT
        for i in range(2):
            self.assertEqual(env.cost_history[i], {0: ClusterType.SPOT},
                           f"Tick {i} should bill SPOT")
        
        # Verify total cost (2 ticks of SPOT)
        spot_cost_per_tick = self.spot_price * self.gap_seconds / 3600
        expected_cost = spot_cost_per_tick * 2
        
        actual_cost = env.accumulated_cost
        self.assertAlmostEqual(actual_cost, expected_cost, places=4)


if __name__ == '__main__':
    unittest.main()