"""Simple multi-region availability probe strategy.

This strategy keeps the RC/CR uniform progress safety net from
``multi_region_rc_cr_threshold`` but augments the region selection with
lightweight spot availability probes.  It maintains a short sliding window of
recent probe outcomes per region (1 = launch succeeded, 0 = failed or
preempted) and always prefers the region with the highest empirical success
ratio.  Background probes launch spot instances for a single tick to gather
fresh samples without affecting the main job.

Key behaviours:
* Maintain at most one main instance (SPOT or ON_DEMAND) at any time.
* Use RC/CR conditions to decide between SPOT / ON_DEMAND.
* When SPOT is requested, iterate regions in descending ratio order until one
  launch succeeds.
* Background probes run periodically to keep window samples up-to-date.
* On preemption of the main region, immediately record a failure sample and
  re-launch based on the latest ratios.

The implementation intentionally stays simple: no bandits, no unified cost
model parameters—just probe-based success ratios.
"""

import argparse
import logging
from collections import deque
import typing

from sky_spot.strategies.multi_region_rc_cr_threshold import (
    MultiRegionRCCRThresholdStrategy,
)
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot.multi_region_types import Action, LaunchResult

logger = logging.getLogger(__name__)


class MultiRegionAvailabilityProbeSimple(MultiRegionRCCRThresholdStrategy):
    """Uniform-progress multi-region strategy driven by probe ratios."""

    NAME = "multi_region_availability_probe_simple"

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        # Sliding window size (number of recent samples kept per region).
        self.window_size: int = getattr(args, "probe_window", 5)
        # Probe cadence in minutes; each region is probed at most once per interval.
        self.probe_interval_min: float = getattr(args, "probe_interval_min", 120.0)

        # Runtime state, initialised in reset().
        self._probe_hist: list[deque[int]] = []
        self._next_probe_tick: list[int] = []
        self._next_sample_tick: list[int] = []
        self._probes_active: set[int] = set()
        self._main_region: typing.Optional[int] = None
        self._probe_interval_ticks: int = 1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, env: "env.Env", task):
        super().reset(env, task)
        menv = typing.cast("env.MultiTraceEnv", env)
        num_regions = menv.num_regions
        self._probe_hist = [deque(maxlen=self.window_size) for _ in range(num_regions)]
        self._next_probe_tick = [0 for _ in range(num_regions)]
        self._next_sample_tick = [0 for _ in range(num_regions)]
        self._probes_active.clear()
        self._main_region = None

        # Convert probe interval from minutes to ticks (at least 1 tick).
        gap_seconds = max(1, menv.gap_seconds)
        interval_seconds = max(self.probe_interval_min, 0.0) * 60.0
        ticks = int(round(interval_seconds / gap_seconds))
        self._probe_interval_ticks = max(1, ticks)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _ratio(self, ridx: int) -> float:
        window = self._probe_hist[ridx]
        if not window:
            return 0.5  # Neutral prior when no samples yet.
        return sum(window) / float(len(window))

    def _record(self, ridx: int, success: bool) -> None:
        self._probe_hist[ridx].append(1 if success else 0)

    def _rank_regions(self, menv: "env.MultiTraceEnv") -> list[int]:
        # Sort by success ratio desc, then by region id asc for stability.
        ranked = list(range(menv.num_regions))
        ranked.sort(key=lambda r: (-self._ratio(r), r))
        return ranked

    def _active_main_region(
        self, active_map: dict[int, ClusterType]
    ) -> typing.Optional[int]:
        if self._main_region is not None:
            reg_type = active_map.get(self._main_region)
            if reg_type == ClusterType.SPOT:
                return self._main_region
        # If we do not have a tracked main region but exactly one SPOT instance is
        # running (e.g., after recovery), treat it as the main region.
        spot_regions = [r for r, t in active_map.items() if t == ClusterType.SPOT]
        if len(spot_regions) == 1:
            self._main_region = spot_regions[0]
            return self._main_region
        return None

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------
    def _step_multi(
        self,
    ) -> typing.Generator["Action", typing.Optional["LaunchResult"], None]:  # type: ignore[override]
        from sky_spot.multi_region_types import TryLaunch, Terminate

        menv = typing.cast("env.MultiTraceEnv", self.env)

        # 1) Probes are independent and do not create active instances; no cleanup needed.

        # Refresh active instances after clean-up.
        active_instances = menv.get_active_instances()

        # 2) Detect preemptions for the tracked main region.
        if self._main_region is not None:
            reg_type = active_instances.get(self._main_region)
            if reg_type != ClusterType.SPOT:
                # Main region disappeared (preempted or terminated externally).
                self._record(self._main_region, False)
                self._main_region = None

        # 3) Update running main region samples (success heartbeat).
        now_tick = self.env.tick
        current_main = self._active_main_region(active_instances)
        if (
            current_main is not None
            and now_tick >= self._next_sample_tick[current_main]
        ):
            self._record(current_main, True)
            self._next_sample_tick[current_main] = now_tick + self._probe_interval_ticks

        # 4) Early exit if task already done.
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            # Shut everything down.
            for region, ctype in list(active_instances.items()):
                yield Terminate(region=region)
            self._main_region = None
            return

        # 5) Determine desired cluster type using RC/CR conditions.
        current_cluster_type = ClusterType.NONE
        current_region = None
        if active_instances:
            current_region, current_cluster_type = next(iter(active_instances.items()))
        request_type = self._compute_request_type(current_cluster_type, has_spot=True)

        # 6) Apply decisions.
        if request_type == ClusterType.SPOT:
            ranked_regions = self._rank_regions(menv)

            # Ensure we do not keep ON_DEMAND while attempting SPOT.
            if (
                current_cluster_type == ClusterType.ON_DEMAND
                and current_region is not None
            ):
                yield Terminate(region=current_region)
                current_cluster_type = ClusterType.NONE
                current_region = None
                self._main_region = None

            if current_cluster_type == ClusterType.SPOT and current_region is not None:
                best_region = ranked_regions[0]
                current_ratio = self._ratio(current_region)
                best_ratio = self._ratio(best_region)
                # If current region is already best (within tolerance), keep it.
                if current_region == best_region or current_ratio >= best_ratio - 1e-6:
                    # Optionally run a background probe later.
                    self._main_region = current_region
                    self._hysteresis_exit = False
                else:
                    # Switch to a better region.
                    yield Terminate(region=current_region)
                    self._main_region = None
                    current_cluster_type = ClusterType.NONE
                    current_region = None

            if current_cluster_type != ClusterType.SPOT:
                launched = False
                active_map = menv.get_active_instances()
                for ridx in ranked_regions:
                    if active_map.get(ridx) == ClusterType.SPOT:
                        continue
                    result = yield TryLaunch(region=ridx, cluster_type=ClusterType.SPOT)
                    assert result is not None
                    self._record(ridx, result.success)
                    if result.success:
                        self._main_region = ridx
                        # Schedule the next heartbeat sample for this region.
                        self._next_sample_tick[ridx] = (
                            now_tick + self._probe_interval_ticks
                        )
                        launched = True
                        self._hysteresis_exit = False
                        break
                if not launched:
                    if getattr(self, "_hysteresis_exit", False):
                        self._hysteresis_exit = False
                        return
                    # Double-check RC/CR with has_spot=False for fallback.
                    fallback_type = self._compute_request_type(
                        ClusterType.NONE, has_spot=False
                    )
                    if fallback_type == ClusterType.ON_DEMAND:
                        fallback_region = current_region
                        if fallback_region is None:
                            fallback_region = getattr(
                                menv, "_current_leader_region", None
                            )
                        if fallback_region is None:
                            fallback_region = 0
                        result = yield TryLaunch(
                            region=fallback_region, cluster_type=ClusterType.ON_DEMAND
                        )
                        assert result is not None and result.success
                        self._main_region = None
                    self._hysteresis_exit = False
                    return

        elif request_type == ClusterType.ON_DEMAND:
            if current_cluster_type != ClusterType.ON_DEMAND:
                if current_region is not None:
                    yield Terminate(region=current_region)
                fallback_region = current_region
                if fallback_region is None:
                    fallback_region = getattr(menv, "_current_leader_region", None)
                if fallback_region is None:
                    fallback_region = 0
                result = yield TryLaunch(
                    region=fallback_region, cluster_type=ClusterType.ON_DEMAND
                )
                assert result is not None and result.success
                self._main_region = None
                self._hysteresis_exit = False
            return

        else:  # ClusterType.NONE
            if current_region is not None:
                yield Terminate(region=current_region)
            self._main_region = None
            return

        # 7) Schedule at most one background probe this tick.
        yield from self._run_background_probe(
            menv,
            exclude={self._main_region} if self._main_region is not None else set(),
        )

    # ------------------------------------------------------------------
    # Background probe helper
    # ------------------------------------------------------------------
    def _run_background_probe(
        self,
        menv: "env.MultiTraceEnv",
        exclude: set[int],
    ) -> typing.Generator["Action", typing.Optional["LaunchResult"], None]:
        from sky_spot.multi_region_types import ProbeLaunch

        now = self.env.tick
        active_map = menv.get_active_instances()
        due: list[tuple[int, int]] = []
        for ridx in range(menv.num_regions):
            if ridx in exclude:
                continue
            if active_map.get(ridx) == ClusterType.SPOT:
                continue
            if now >= self._next_probe_tick[ridx]:
                due.append((len(self._probe_hist[ridx]), ridx))

        if not due:
            if False:
                yield  # Make this a generator even when no probe is launched.
            return

        # Prefer regions with fewer samples to balance exploration.
        due.sort(key=lambda item: (item[0], item[1]))
        ridx = due[0][1]

        result = yield ProbeLaunch(region=ridx)
        assert result is not None
        self._record(ridx, result.success)
        self._next_probe_tick[ridx] = now + self._probe_interval_ticks
        # No active probe to track/terminate under the new semantics.

    # ------------------------------------------------------------------
    # Argparse hook
    # ------------------------------------------------------------------
    @classmethod
    def _from_args(
        cls, parser: "argparse.ArgumentParser"
    ) -> "MultiRegionAvailabilityProbeSimple":
        group = parser.add_argument_group("AvailabilityProbeSimple")
        group.add_argument(
            "--probe-window",
            type=int,
            default=5,
            help="Sliding window length for recent probe outcomes per region",
        )
        group.add_argument(
            "--probe-interval-min",
            type=float,
            default=120.0,
            help="Background probe interval (minutes)",
        )
        return cls(parser.parse_args())
