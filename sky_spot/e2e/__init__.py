"""E2E testing module for sky_spot.

This module provides end-to-end testing infrastructure for multi-region
spot instance simulations.
"""

from sky_spot.e2e.config import (
    USE_TRAINING_WORKLOAD,
    USE_FAKE_WORKLOAD,
    USE_GPU,
    MULTI_REGION_ZONES,
    trace_files,
    task_name,
    instance_type,
    DEFAULT_STRATEGY,
    STRATEGY_ALIASES,
    _resolve_strategy,
)

from sky_spot.e2e.console import SimulationConsole

from sky_spot.e2e.cluster import (
    _get_cluster_name,
    _get_bucket_name,
    _create_bucket,
    _actual_launch_internal,
    _actual_terminate_internal,
    _actual_probe_internal,
    _actual_check_is_preempted_internal,
    _actual_check_ondemand_health_internal,
    _cleanup_launched_clusters,
    _probe_thread,
    probe_event,
)

from sky_spot.e2e.transfer import (
    _transfer_s3_bucket,
)

from sky_spot.e2e.viz import (
    generate_timeline_plot,
    _load_history_from_file,
    _infer_region_names_from_history,
)

from sky_spot.e2e.runner import run_simulation

__all__ = [
    # Main entry point
    'run_simulation',
    # Config
    'USE_TRAINING_WORKLOAD',
    'USE_FAKE_WORKLOAD',
    'USE_GPU',
    'MULTI_REGION_ZONES',
    'trace_files',
    'task_name',
    'instance_type',
    'DEFAULT_STRATEGY',
    'STRATEGY_ALIASES',
    '_resolve_strategy',
    # Console
    'SimulationConsole',
    # Cluster operations
    '_get_cluster_name',
    '_get_bucket_name',
    '_create_bucket',
    '_actual_launch_internal',
    '_actual_terminate_internal',
    '_actual_probe_internal',
    '_actual_check_is_preempted_internal',
    '_actual_check_ondemand_health_internal',
    '_cleanup_launched_clusters',
    '_probe_thread',
    'probe_event',
    # Transfer
    '_transfer_s3_bucket',
    # Visualization
    'generate_timeline_plot',
    '_load_history_from_file',
    '_infer_region_names_from_history',
]
