"""Multi-region probe cost ratio baseline.

A simple baseline that uses probe success rate divided by spot price to
select regions. Inherits RC/CR uniform progress safety net.

Decision logic:
1. Track probe success rate = successes / total_probes (global, not windowed)
2. Score each region by success_rate / spot_price
3. Select region with highest score
4. Use RC/CR conditions to decide SPOT vs ON_DEMAND
"""

import argparse
import logging
import typing

from sky_spot.strategies.multi_region_rc_cr_threshold import (
    MultiRegionRCCRThresholdStrategy,
)
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot.multi_region_types import Action, LaunchResult

logger = logging.getLogger(__name__)


class MultiRegionProbeCostRatioStrategy(MultiRegionRCCRThresholdStrategy):
    """Probe-based region selection using success_rate / cost."""

    NAME = "multi_region_probe_cost_ratio"

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        # Probe cadence in minutes
        self.probe_interval_min: float = getattr(args, "probe_interval_min", 60.0)

        # Runtime state
        self._probe_successes: list[int] = []
        self._probe_total: list[int] = []
        self._next_probe_tick: list[int] = []
        self._probe_interval_ticks: int = 1
        self._main_region: typing.Optional[int] = None

    def reset(self, env: "env.Env", task):
        super().reset(env, task)
        menv = typing.cast("env.MultiTraceEnv", env)
        num_regions = menv.num_regions

        self._probe_successes = [0 for _ in range(num_regions)]
        self._probe_total = [0 for _ in range(num_regions)]
        self._next_probe_tick = [0 for _ in range(num_regions)]
        self._main_region = None

        # Convert probe interval from minutes to ticks
        gap_seconds = max(1, menv.gap_seconds)
        interval_seconds = max(self.probe_interval_min, 0.0) * 60.0
        ticks = int(round(interval_seconds / gap_seconds))
        self._probe_interval_ticks = max(1, ticks)

    def _success_rate(self, ridx: int) -> float:
        """Return probe success rate for a region."""
        total = self._probe_total[ridx]
        if total == 0:
            return 0.5  # Neutral prior
        return self._probe_successes[ridx] / total

    def _record_probe(self, ridx: int, success: bool) -> None:
        """Record a probe result."""
        self._probe_total[ridx] += 1
        if success:
            self._probe_successes[ridx] += 1

    def _score_region(self, ridx: int, menv: "env.MultiTraceEnv") -> float:
        """Score = success_rate / spot_price."""
        rate = self._success_rate(ridx)
        spot_price = menv.envs[ridx].get_price()[ClusterType.SPOT]
        if spot_price <= 0:
            return float("inf") if rate > 0 else 0.0
        return rate / spot_price

    def _rank_regions(self, menv: "env.MultiTraceEnv") -> list[int]:
        """Sort regions by score (high to low)."""
        ranked = list(range(menv.num_regions))
        ranked.sort(key=lambda r: (-self._score_region(r, menv), r))
        return ranked

    def _step_multi(
        self,
    ) -> typing.Generator["Action", typing.Optional["LaunchResult"], None]:
        from sky_spot.multi_region_types import TryLaunch, Terminate, ProbeLaunch

        menv = typing.cast("env.MultiTraceEnv", self.env)
        now_tick = self.env.tick

        active_instances = menv.get_active_instances()

        # Detect preemption of main region
        if self._main_region is not None:
            reg_type = active_instances.get(self._main_region)
            if reg_type != ClusterType.SPOT:
                # Preempted - record as failure
                self._record_probe(self._main_region, False)
                self._main_region = None

        # Record heartbeat for running main region
        if self._main_region is not None:
            if now_tick >= self._next_probe_tick[self._main_region]:
                self._record_probe(self._main_region, True)
                self._next_probe_tick[self._main_region] = (
                    now_tick + self._probe_interval_ticks
                )

        # Early exit if task done
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            for region in list(active_instances.keys()):
                yield Terminate(region=region)
            self._main_region = None
            return

        # Determine desired cluster type using RC/CR conditions
        current_cluster_type = ClusterType.NONE
        current_region = None
        if active_instances:
            current_region, current_cluster_type = next(iter(active_instances.items()))
        request_type = self._compute_request_type(current_cluster_type, has_spot=True)

        if request_type == ClusterType.SPOT:
            ranked_regions = self._rank_regions(menv)

            # Terminate ON_DEMAND if switching to SPOT
            if (
                current_cluster_type == ClusterType.ON_DEMAND
                and current_region is not None
            ):
                yield Terminate(region=current_region)
                current_cluster_type = ClusterType.NONE
                current_region = None
                self._main_region = None

            # If already running SPOT, check if should switch
            if current_cluster_type == ClusterType.SPOT and current_region is not None:
                best_region = ranked_regions[0]
                current_score = self._score_region(current_region, menv)
                best_score = self._score_region(best_region, menv)

                # Keep current if score is close enough
                if current_region == best_region or current_score >= best_score * 0.9:
                    self._main_region = current_region
                else:
                    # Switch to better region
                    yield Terminate(region=current_region)
                    self._main_region = None
                    current_cluster_type = ClusterType.NONE
                    current_region = None

            # Launch SPOT if needed
            if current_cluster_type != ClusterType.SPOT:
                launched = False
                for ridx in ranked_regions:
                    if menv.get_active_instances().get(ridx) == ClusterType.SPOT:
                        continue
                    result = yield TryLaunch(region=ridx, cluster_type=ClusterType.SPOT)
                    assert result is not None
                    self._record_probe(ridx, result.success)
                    if result.success:
                        self._main_region = ridx
                        self._next_probe_tick[ridx] = (
                            now_tick + self._probe_interval_ticks
                        )
                        launched = True
                        break

                if not launched:
                    # Fallback to ON_DEMAND
                    fallback_type = self._compute_request_type(
                        ClusterType.NONE, has_spot=False
                    )
                    if fallback_type == ClusterType.ON_DEMAND:
                        fallback_region = 0
                        result = yield TryLaunch(
                            region=fallback_region, cluster_type=ClusterType.ON_DEMAND
                        )
                        assert result is not None and result.success
                        self._main_region = None

        elif request_type == ClusterType.ON_DEMAND:
            if current_cluster_type != ClusterType.ON_DEMAND:
                if current_region is not None:
                    yield Terminate(region=current_region)
                fallback_region = current_region if current_region is not None else 0
                result = yield TryLaunch(
                    region=fallback_region, cluster_type=ClusterType.ON_DEMAND
                )
                assert result is not None and result.success
                self._main_region = None
            return

        else:  # ClusterType.NONE
            if current_region is not None:
                yield Terminate(region=current_region)
            self._main_region = None
            return

        # Background probes for other regions
        for ridx in range(menv.num_regions):
            if ridx == self._main_region:
                continue
            if menv.get_active_instances().get(ridx) == ClusterType.SPOT:
                continue
            if now_tick < self._next_probe_tick[ridx]:
                continue

            result = yield ProbeLaunch(region=ridx)
            assert result is not None
            self._record_probe(ridx, result.success)
            self._next_probe_tick[ridx] = now_tick + self._probe_interval_ticks

    @classmethod
    def _from_args(cls, parser: "argparse.ArgumentParser"):
        group = parser.add_argument_group("ProbeCostRatio")
        group.add_argument(
            "--probe-interval-min",
            type=float,
            default=60.0,
            help="Probe interval in minutes",
        )
        return cls(parser.parse_args())
