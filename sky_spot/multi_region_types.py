from dataclasses import dataclass
from typing import Optional, Union
from sky_spot.utils import ClusterType


@dataclass
class TryLaunch:
    region: int
    cluster_type: ClusterType


@dataclass
class Terminate:
    region: int
    cluster_type: Optional[ClusterType] = None  # If None, terminates whatever is running


@dataclass
class LaunchResult:
    success: bool
    region: int
    cluster_type: Optional[ClusterType] = None


@dataclass
class ProbeLaunch:
    """Launch a one-tick probe in a region without affecting state.

    Semantics:
    - Does not create an active instance (no progress, no leader effects).
    - Billed at a fixed 1-minute unit if launch succeeds.
    - No migration/egress or restart overhead is applied.
    """
    region: int


# Type alias for all possible actions
Action = Union[TryLaunch, Terminate, ProbeLaunch]
