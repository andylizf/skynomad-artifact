"""Tests for dynamic migration time functionality."""

import json
import os
import tempfile
import unittest
from typing import Dict, Optional, Generator

from sky_spot.env import MultiTraceEnv
from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.multi_region_types import TryLaunch, Terminate, LaunchResult, Action
from sky_spot.utils import ClusterType
from sky_spot import task as task_lib
from sky_spot.migration_model import (
    parse_region_info,
    get_region_relationship,
    get_transfer_time_hours,
    get_transfer_cost_usd
)


class MockTask(task_lib.Task):
    """Mock task for testing with configurable checkpoint size."""
    
    def __init__(self, duration_seconds: float, checkpoint_size_gb: float = 50.0):
        self.duration_seconds = duration_seconds
        self.checkpoint_size_gb = checkpoint_size_gb
        self._progress_source = []
    
    def set_progress_source(self, task_done_time_list):
        self._progress_source = task_done_time_list
    
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
        return {
            'duration_seconds': self.duration_seconds,
            'checkpoint_size_gb': self.checkpoint_size_gb
        }
    
    def reset(self):
        """Reset the task state."""
        self._progress_source = []
    
    def __str__(self) -> str:
        """String representation of the task."""
        return f"MockTask(duration={self.duration_seconds}s, checkpoint={self.checkpoint_size_gb}GB)"


class MockArgs:
    """Mock arguments for strategy initialization."""
    def __init__(self, restart_overhead_hours=0.2):
        self.deadline_hours = 100.0  # Large deadline to avoid SAFETY NET
        self.restart_overhead_hours = [restart_overhead_hours]  # Fixed overhead (will be overridden for migrations)
        self.inter_task_overhead = [0.0]


class TestDynamicMigration(unittest.TestCase):
    """Test dynamic migration time functionality."""
    
    def setUp(self):
        """Create mock trace files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.gap_seconds = 600  # 10 minutes per tick
        
        # Create region-specific directories
        self.regions = {
            'us-east-1a_v100_1': {'region': 'us-east-1', 'zone': 'a'},
            'us-east-1c_v100_1': {'region': 'us-east-1', 'zone': 'c'},
            'us-west-2a_v100_1': {'region': 'us-west-2', 'zone': 'a'},
        }
        
        self.trace_files = []
        for region_name, info in self.regions.items():
            # Create directory structure
            region_dir = os.path.join(self.temp_dir, region_name)
            os.makedirs(region_dir, exist_ok=True)
            
            # Create trace file
            trace_file = os.path.join(region_dir, '0.json')
            self.create_trace_file(trace_file, region_name, [0] * 100)  # All available
            self.trace_files.append(trace_file)
    
    def create_trace_file(self, filepath: str, region_name: str, availability: list):
        """Create a mock trace file."""
        trace_data = {
            "metadata": {
                "price_info": {"on_demand_price": 3.06, "price": 0.918},
                "region": region_name,
                "instance_type": "v100_1",
                "start_time": "2024-01-01T00:00:00Z",
                "gap_seconds": self.gap_seconds,
                "device": "v100_1"
            },
            "data": availability
        }
        
        with open(filepath, 'w') as f:
            json.dump(trace_data, f)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _simulate_cross_region_migration(self, count_cross_region_time: bool):
        """Run the single-migration scenario and return histories."""
        env = MultiTraceEnv(
            self.trace_files[:3],
            env_start_hours=0,
            count_cross_region_migration_time=count_cross_region_time
        )

        class MigrationStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self.tick_count = 0

            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if self.tick_count == 0:
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                elif self.tick_count == 5:
                    yield Terminate(region=0)
                    result = yield TryLaunch(region=2, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success

                self.tick_count += 1

        args = MockArgs(restart_overhead_hours=0.2)
        task = MockTask(duration_seconds=36000, checkpoint_size_gb=100)
        strategy = MigrationStrategy(args)
        strategy.reset(env, task)

        progress_history = []
        overhead_history = []

        for _ in range(10):
            env.observe()
            env.update_strategy_progress(strategy)
            env.execute_multi_strategy(strategy)
            env.tick += 1

            progress_history.append(strategy.task_done_time[-1] if strategy.task_done_time else 0)
            overhead_history.append(strategy.remaining_restart_overhead)

        return env, progress_history, overhead_history

    def test_migration_model_parsing(self):
        """Test region parsing and relationship detection."""
        # Test parsing
        region, zone, instance_type = parse_region_info('us-east-1a_v100_1')
        self.assertEqual(region, 'us-east-1')
        self.assertEqual(zone, 'a')
        self.assertEqual(instance_type, 'v100')
        
        # Test relationships
        self.assertEqual(
            get_region_relationship('us-east-1a_v100_1', 'us-east-1a_v100_1'),
            'same_zone'
        )
        self.assertEqual(
            get_region_relationship('us-east-1a_v100_1', 'us-east-1c_v100_1'),
            'cross_az'
        )
        self.assertEqual(
            get_region_relationship('us-east-1a_v100_1', 'us-west-2a_v100_1'),
            'cross_region'
        )
    
    def test_migration_time_calculation(self):
        """Test migration time calculations under documented semantics.

        get_transfer_time_hours returns additional transfer time beyond
        baseline same-region S3 time (which is part of restart_overhead).
        """
        checkpoint_size = 100  # GB
        
        # Same zone: additional transfer time = 0 → total = startup only
        transfer_hours = get_transfer_time_hours(
            'us-east-1a_v100_1', 'us-east-1a_v100_1', checkpoint_size
        )
        same_zone_hours = 0.033 + transfer_hours  # Default startup time
        self.assertAlmostEqual(same_zone_hours, 0.033, places=3)
        
        # Cross-AZ: treated the same as same zone for S3 additional time
        transfer_hours = get_transfer_time_hours(
            'us-east-1a_v100_1', 'us-east-1c_v100_1', checkpoint_size
        )
        cross_az_hours = 0.033 + transfer_hours  # Default startup time
        self.assertAlmostEqual(cross_az_hours, 0.033, places=3)
        
        # Cross-region: should be slightly slower (small positive additional time)
        transfer_hours = get_transfer_time_hours(
            'us-east-1a_v100_1', 'us-west-2a_v100_1', checkpoint_size
        )
        cross_region_hours = 0.033 + transfer_hours  # Default startup time
        self.assertGreater(cross_region_hours, 0.033)
    
    def test_cold_start_vs_migration(self):
        """Cold starts use fixed overhead; cross-region migrations add dynamic time."""
        env, progress_history, overhead_history = self._simulate_cross_region_migration(True)

        # Tick 0: Launch happens, but overhead not yet applied
        self.assertEqual(progress_history[0], 0)
        self.assertEqual(overhead_history[0], 0)

        # Tick 1: Overhead is applied (0.2 hours = 720 seconds)
        self.assertEqual(progress_history[1], 0)
        self.assertAlmostEqual(overhead_history[1], 720 - 600, delta=1)

        # Tick 2: Finish cold start overhead
        self.assertGreater(progress_history[2], 0)
        self.assertAlmostEqual(overhead_history[2], 0, delta=1)

        # Ticks 3-4: Normal progress
        for i in range(3, 5):
            self.assertAlmostEqual(progress_history[i], 600, delta=1)

        # Tick 5: Migration scheduled, overhead applied next tick
        self.assertEqual(overhead_history[5], 0)

        transfer_hours = get_transfer_time_hours(
            'us-east-1a_v100_1', 'us-west-2a_v100_1', 100
        )
        expected_migration_hours = 0.2 + transfer_hours
        expected_migration_seconds = expected_migration_hours * 3600

        self.assertEqual(progress_history[6], 0)
        self.assertAlmostEqual(overhead_history[6], expected_migration_seconds - 600, delta=10)
        self.assertNotAlmostEqual(overhead_history[6], 120, delta=10)
        self.assertAlmostEqual(env.migration_hours_total, expected_migration_hours, places=3)

    def test_ignore_cross_region_migration_time_flag(self):
        """When disabled, cross-region migration downtime excludes transfer time but keeps egress cost."""
        env, progress_history, overhead_history = self._simulate_cross_region_migration(False)

        self.assertEqual(progress_history[0], 0)
        self.assertEqual(overhead_history[0], 0)

        restart_seconds = 0.2 * 3600
        self.assertEqual(progress_history[1], 0)
        self.assertAlmostEqual(overhead_history[1], restart_seconds - 600, delta=1)

        self.assertGreater(progress_history[2], 0)
        self.assertAlmostEqual(overhead_history[2], 0, delta=1)

        transfer_hours = get_transfer_time_hours(
            'us-east-1a_v100_1', 'us-west-2a_v100_1', 100
        )
        expected_with_transfer = (0.2 + transfer_hours) * 3600 - 600
        expected_restart_only = restart_seconds - 600

        self.assertEqual(progress_history[6], 0)
        self.assertAlmostEqual(overhead_history[6], expected_restart_only, delta=1)
        self.assertNotAlmostEqual(overhead_history[6], expected_with_transfer, delta=10)
        self.assertAlmostEqual(env.migration_hours_total, 0.2, places=3)
        self.assertGreater(env.transfer_cost_total, 0.0)


    def test_multiple_migrations(self):
        """Test multiple migrations with different relationships."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0, count_cross_region_migration_time=True)  # All 3 regions
        
        class MultiMigrationStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self.tick_count = 0
                
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if self.tick_count == 0:
                    # Start in us-east-1a
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                elif self.tick_count == 3:
                    # Migrate to us-east-1c (cross-AZ)
                    yield Terminate(region=0)
                    result = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                elif self.tick_count == 6:
                    # Migrate to us-west-2a (cross-region)
                    yield Terminate(region=1)
                    result = yield TryLaunch(region=2, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                
                self.tick_count += 1
        
        args = MockArgs(restart_overhead_hours=0.2)
        task = MockTask(duration_seconds=36000, checkpoint_size_gb=50)  # 50GB checkpoint
        strategy = MultiMigrationStrategy(args)
        strategy.reset(env, task)
        
        migration_overheads = []
        migration_ticks = [3, 6]  # When migrations happen
        
        # Run simulation
        for i in range(10):
            env.observe()
            env.update_strategy_progress(strategy)
            
            # Record overhead at migration ticks
            if i in migration_ticks:
                migration_overheads.append({
                    'tick': i,
                    'overhead_before': strategy.remaining_restart_overhead
                })
            
            env.execute_multi_strategy(strategy)
            env.tick += 1
            
            # Record overhead after execution
            if i in migration_ticks:
                migration_overheads[-1]['overhead_after'] = strategy.remaining_restart_overhead
        
        # Verify different migration times
        # First migration: cross-AZ (us-east-1a -> us-east-1c)
        cross_az_transfer = get_transfer_time_hours(
            'us-east-1a_v100_1', 'us-east-1c_v100_1', 50
        )
        cross_az_hours = 0.033 + cross_az_transfer  # Default startup
        cross_az_seconds = cross_az_hours * 3600
        
        # Second migration: cross-region (us-east-1c -> us-west-2a)
        cross_region_transfer = get_transfer_time_hours(
            'us-east-1c_v100_1', 'us-west-2a_v100_1', 50
        )
        cross_region_hours = 0.033 + cross_region_transfer  # Default startup
        cross_region_seconds = cross_region_hours * 3600
        
        # Cross-region should take longer than cross-AZ (though difference is smaller with fast S3)
        self.assertGreater(cross_region_seconds, cross_az_seconds)
    
    def test_migration_costs(self):
        """Test that migration costs are calculated correctly."""
        # Test various scenarios
        test_cases = [
            ('us-east-1a_v100_1', 'us-east-1a_v100_1', 100, 0.0),  # Same zone: free
            ('us-east-1a_v100_1', 'us-east-1c_v100_1', 100, 0.0),  # Cross-AZ: free
            ('us-east-1a_v100_1', 'us-west-2a_v100_1', 100, 2.0),  # Cross-region: $0.02/GB
        ]
        
        for src, dst, size_gb, expected_cost in test_cases:
            cost = get_transfer_cost_usd(src, dst, size_gb)
            self.assertEqual(cost, expected_cost)
    
    def test_checkpoint_size_impact(self):
        """Larger checkpoints increase migration time; additional time scales linearly."""
        source = 'us-east-1a_v100_1' 
        dest = 'us-west-2a_v100_1'  # Cross-region for observable additional time
        startup_hours = 0.033
        
        checkpoint_sizes = [10, 50, 100, 500]
        migration_times = []
        additional_times = []
        
        for size in checkpoint_sizes:
            transfer_hours = get_transfer_time_hours(source, dest, size)
            time_hours = startup_hours + transfer_hours
            migration_times.append(time_hours)
            additional_times.append(transfer_hours)
            print(f"Checkpoint {size}GB: {time_hours:.3f} hours ({time_hours*60:.1f} minutes)")
        
        # Verify total times are increasing
        for i in range(1, len(migration_times)):
            self.assertGreater(migration_times[i], migration_times[i-1],
                             f"Checkpoint {checkpoint_sizes[i]}GB should take longer than {checkpoint_sizes[i-1]}GB")
        
        # Additional transfer time scales strongly with size (approximately linear)
        self.assertGreater(additional_times[-1] / additional_times[0], 40,
                          "Additional transfer time should scale strongly with checkpoint size")
    
    def test_checkpoint_size_in_environment(self):
        """Test that checkpoint size affects migration overhead in the environment."""
        
        class CheckpointTestStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self.tick_count = 0
                
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if self.tick_count == 0:
                    # Start in region 0
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                elif self.tick_count == 3:
                    # Migrate cross-region to region 2
                    yield Terminate(region=0)
                    result = yield TryLaunch(region=2, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                
                self.tick_count += 1
        
        # Test with different checkpoint sizes
        results = {}
        for checkpoint_size in [10, 100, 500]:
            # Create fresh environment for each test (need cross-region dest)
            env = MultiTraceEnv(self.trace_files[:3], env_start_hours=0, count_cross_region_migration_time=True)
            
            args = MockArgs(restart_overhead_hours=0.2)
            task = MockTask(duration_seconds=36000, checkpoint_size_gb=checkpoint_size)
            strategy = CheckpointTestStrategy(args)
            strategy.reset(env, task)
            
            # Run until migration happens
            migration_overhead = None
            for i in range(8):
                env.observe()
                env.update_strategy_progress(strategy)
                
                # Check overhead after migration (at tick 4)
                if i == 4:  # One tick after migration
                    migration_overhead = strategy.remaining_restart_overhead
                
                env.execute_multi_strategy(strategy)
                env.tick += 1
            
            results[checkpoint_size] = migration_overhead
            print(f"Checkpoint {checkpoint_size}GB migration overhead: {migration_overhead:.1f} seconds")
        
        # Verify that larger checkpoints result in larger overhead
        self.assertGreater(results[100], results[10],
                          "100GB checkpoint should have more overhead than 10GB")
        self.assertGreater(results[500], results[100], 
                          "500GB checkpoint should have more overhead than 100GB")


if __name__ == '__main__':
    unittest.main()
