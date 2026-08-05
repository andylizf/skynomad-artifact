"""
Unified Cost Model Strategy

A unified cost framework based on time value theory with:
- Survival-based duration prediction (Nelson-Aalen + burst/jitter)
- Probe mechanism for availability tracking
- Dynamic time value computation
"""

import logging
import typing
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Set, Tuple

from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot import env, task
from sky_spot.multi_region_types import (
    TryLaunch,
    Terminate,
    Action,
    LaunchResult,
    ClusterType,
    ProbeLaunch,
)
from sky_spot.migration_model import (
    get_transfer_time_hours,
    get_transfer_cost_usd,
    get_region_relationship,
)
from sky_spot.strategies import ucm_decision_logger as dlog

logger = logging.getLogger(__name__)


# ==========================================================================
# Data structures for event recording and derived state
# ==========================================================================
@dataclass
class RunEvent:
    """A single run event (start or end)."""

    type: str  # "start" or "end"
    time: float  # elapsed_seconds
    source: str  # "spot" or "probe"
    outcome: str = ""  # "failure" or "censor" (only for end events)


@dataclass
class Segment:
    """A completed run segment."""

    start: float
    end: float
    duration_h: float
    outcome: str  # "failure" or "censor"
    source: str  # "spot" or "probe"


@dataclass
class RegionState:
    """Derived state for a region, rebuilt each step."""

    segments: List[Segment]  # completed segments
    current_age_h: Optional[float]  # age if currently running
    current_source: Optional[str]  # "spot" or "probe" if running
    current_start: Optional[float]  # start time if running


class UnifiedCostModelStrategy(MultiRegionStrategy):
    """Unified Cost Model with survival-based duration and probe mechanism."""

    NAME = "unified_cost_model_risk"
    LOG_PREFIX = "[UCM]"
    EVENT_FAILURE = "failure"
    EVENT_CENSOR = "censor"

    def __init__(self, args):
        super().__init__(args)

        # Core parameters
        self.max_attempts_per_tick: int = args.max_attempts_per_tick
        self.consider_migration_penalty: bool = not args.ignore_migration_penalty

        # VOI gating parameters
        self.voi_gating: bool = args.voi_gating
        self.voi_delta_base: float = args.voi_delta_base
        self.voi_k_checkpoint: float = args.voi_k_checkpoint
        self.voi_k_v: float = args.voi_k_v

        # Probe parameters
        self.probe_interval_ticks: int = getattr(args, "probe_interval_ticks", 6)
        # See _post_step_actions for what this changes and why the default is
        # False (the shipped results were produced with it False).
        self.probe_revalidate: bool = getattr(args, "probe_revalidate", False)
        # See _update_virtual_availability. Off by default, which is the setting
        # the paper's figures use; the appendix's ablation configs turn it on.
        self.virtual_availability: bool = getattr(args, "virtual_availability", False)
        self.max_prediction_hours: float = getattr(args, "max_prediction_hours", 2.0)
        self.progress_window_hours: float = getattr(args, "progress_window_hours", 10.0)
        self._availability_window_seconds: float = 12 * 3600.0

        # ==========================================================================
        # Raw event storage (append-only, per region)
        # ==========================================================================
        self._raw_events: List[List[RunEvent]] = []

        # ==========================================================================
        # Derived state (rebuilt at each step_multi start)
        # ==========================================================================
        self._region_state: List[RegionState] = []

        # ==========================================================================
        # Tracking state for active runs (to detect preemptions)
        # ==========================================================================
        self._active_spot_region: Optional[int] = None
        self._active_spot_start: float = 0.0
        self._spot_terminated_by_us: bool = False
        self._last_region: Optional[int] = None

        # Probe scheduling
        self._next_probe_tick: List[int] = []
        self._probes_active: Set[int] = set()
        self._last_failed_regions: Set[int] = set()

        # Virtual availability state, used only when self.virtual_availability
        self._probe_age_h: List[float] = []
        self._last_probe_cost_by_region: List[float] = []
        self._virtual_available: List[Optional[bool]] = []
        self._virtual_start_time: List[Optional[float]] = []

        # Burst model state (incremental)
        self._burst_events: List[Deque[dict]] = []

        # Jitter model state
        self._jitter_g_hat: List[float] = []
        self._jitter_last_real_age_h: List[Optional[float]] = []
        self._jitter_last_virtual_age_h: List[Optional[float]] = []
        self._jitter_tick_failures: List[int] = []
        self._jitter_last_tick: int = -1

        logger.info("Initialized Unified Cost Model strategy")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, env: "env.Env", task: "task.Task"):
        super().reset(env, task)
        menv = typing.cast("env.MultiTraceEnv", env)
        num_regions = menv.num_regions
        self._gap_seconds = menv.gap_seconds

        # Reset raw events
        self._raw_events = [[] for _ in range(num_regions)]

        # Reset derived state
        self._region_state = [
            RegionState(
                segments=[], current_age_h=None, current_source=None, current_start=None
            )
            for _ in range(num_regions)
        ]

        # Reset tracking
        self._active_spot_region = None
        self._active_spot_start = 0.0
        self._spot_terminated_by_us = False
        self._last_region = None

        # Reset probe state
        self._next_probe_tick = [0 for _ in range(num_regions)]
        self._probes_active.clear()
        self._last_failed_regions.clear()

        # Reset virtual availability state
        self._probe_age_h = [0.0 for _ in range(num_regions)]
        self._last_probe_cost_by_region = [0.0 for _ in range(num_regions)]
        self._virtual_available = [None for _ in range(num_regions)]
        self._virtual_start_time = [None for _ in range(num_regions)]

        # Reset burst model
        self._burst_events = [deque() for _ in range(num_regions)]

        # Reset jitter
        self._jitter_g_hat = [1.0 for _ in range(num_regions)]
        self._jitter_last_real_age_h = [None for _ in range(num_regions)]
        self._jitter_last_virtual_age_h = [None for _ in range(num_regions)]
        self._jitter_tick_failures = [0 for _ in range(num_regions)]
        self._jitter_last_tick = -1

        logger.info("Unified Cost Model strategy reset")

    # ==========================================================================
    # Raw event recording (THE ONLY entry points for data collection)
    # ==========================================================================
    def _record_start(self, region: int, source: str) -> None:
        """Record a run start event. source='spot' or 'probe'."""
        event = RunEvent(type="start", time=self._now(), source=source)
        self._raw_events[region].append(event)
        logger.debug(
            f"{self.LOG_PREFIX} record_start region={region} source={source} time={event.time:.1f}"
        )

    def _record_end(self, region: int, source: str, outcome: str) -> None:
        """Record a run end event. outcome='failure' or 'censor'."""
        event = RunEvent(type="end", time=self._now(), source=source, outcome=outcome)
        self._raw_events[region].append(event)
        logger.debug(
            f"{self.LOG_PREFIX} record_end region={region} source={source} outcome={outcome} time={event.time:.1f}"
        )

        # Track failures for jitter model
        if outcome == self.EVENT_FAILURE:
            self._jitter_tick_failures[region] += 1
            self._jitter_last_tick = -1  # Force jitter update

    # ==========================================================================
    # Virtual availability (the appendix's ablation configs only)
    # ==========================================================================
    def _update_virtual_availability(self, region: int, available: bool) -> None:
        """Track a region's availability as a run, even with nothing launched there.

        Without this, a region is only observed while something actually runs in
        it, so a region that gets preempted early and is then declined on price
        never produces another survival observation: its estimate stays frozen at
        the one short lifetime that drove it away, and the policy cannot discover
        that the region came back. This keeps a synthetic run open whenever
        probes or launches show the region up, and closes it as EVENT_FAILURE
        when availability is lost, so the survival curve keeps gaining knots.

        Off by default, which is the setting the paper's figures use; the
        appendix's ablation configs turn it on. Concretely, in that ablation's
        r=2.0 window at 43.0 h the only usable zone is preempted in the second
        tick and comes back nine ticks later; with this off, the estimate is stuck
        at 0.26 h against a 0.2 h restart and the job pays on-demand for the rest
        of the deadline.
        """
        if not self.virtual_availability:
            return
        # A real SPOT run in this region already supplies the signal.
        if self._active_spot_region is not None and self._active_spot_region == region:
            return

        if region >= len(self._virtual_available):
            missing = region + 1 - len(self._virtual_available)
            self._virtual_available.extend([None] * missing)
            self._virtual_start_time.extend([None] * missing)

        prev = self._virtual_available[region]
        now = self._now()

        if prev is None:
            if available:
                self._record_start(region, "probe")
                self._virtual_start_time[region] = now
            self._virtual_available[region] = available
            return

        if prev == available:
            return

        if prev and not available:
            self._record_end(region, "probe", self.EVENT_FAILURE)
            self._virtual_start_time[region] = None
        else:
            self._record_start(region, "probe")
            self._virtual_start_time[region] = now

        self._virtual_available[region] = available

    # ==========================================================================
    # Rebuild derived state from raw events (called at step start)
    # ==========================================================================
    def _rebuild_derived_state(self) -> None:
        """Rebuild segments, current state, and jitter from raw events for all regions."""
        now = self._now()
        menv = typing.cast("env.MultiTraceEnv", self.env)

        if self.virtual_availability:
            # Probes are billed a fixed slice of spot time, so the cost the env
            # has charged for a region divided by its spot price is how much
            # probed lifetime it has accumulated -- the age a virtual run has
            # reached across repeated probes, which a single segment's age does
            # not carry.
            num_regions = menv.num_regions
            if len(self._probe_age_h) != num_regions:
                self._probe_age_h = [0.0 for _ in range(num_regions)]
            if len(self._last_probe_cost_by_region) != num_regions:
                self._last_probe_cost_by_region = [0.0 for _ in range(num_regions)]

            for region in range(num_regions):
                total_cost = float(menv.probe_cost_by_region.get(region, 0.0))
                delta = total_cost - self._last_probe_cost_by_region[region]
                if delta > 1e-9:
                    try:
                        price_per_hour = menv.envs[region].get_price()[ClusterType.SPOT]
                    except Exception:
                        price_per_hour = 0.0
                    if price_per_hour > 0:
                        self._probe_age_h[region] += delta / price_per_hour
                self._last_probe_cost_by_region[region] = total_cost

        for region in range(menv.num_regions):
            events = self._raw_events[region]
            segments: List[Segment] = []
            current_start: Optional[float] = None
            current_source: Optional[str] = None

            for ev in events:
                if ev.type == "start":
                    current_start = ev.time
                    current_source = ev.source
                elif ev.type == "end":
                    if current_start is not None:
                        duration_h = max(0.0, (ev.time - current_start) / 3600.0)
                        segments.append(
                            Segment(
                                start=current_start,
                                end=ev.time,
                                duration_h=duration_h,
                                outcome=ev.outcome,
                                source=current_source or ev.source,
                            )
                        )
                    current_start = None
                    current_source = None

            # Update region state
            state = self._region_state[region]
            state.segments = segments

            if current_start is not None:
                state.current_age_h = max(0.0, (now - current_start) / 3600.0)
                state.current_source = current_source
                state.current_start = current_start
            else:
                state.current_age_h = None
                state.current_source = None
                state.current_start = None
            # Note: jitter/burst events are updated incrementally via _jitter_update_once_per_tick()
            # They should NOT be rebuilt here as they track tick-by-tick hazard accumulation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _now(self) -> float:
        return float(self.env.elapsed_seconds)

    # ------------------------------------------------------------------
    # Burst model (incremental update)
    # ------------------------------------------------------------------
    def _burst_prune(self, region: int, max_risk: float = 8.0) -> None:
        """Remove oldest events to keep total risk <= max_risk."""
        events = self._burst_events[region]
        if not events:
            return
        total_risk = sum(e["risk"] for e in events)
        while events and total_risk - events[0]["risk"] >= max_risk:
            oldest = events.popleft()
            total_risk -= oldest["risk"]

    def _burst_compute_suffix(self, region: int) -> Tuple[float, float, int, float]:
        """Return (failures, risk, length, ratio) for densest suffix with risk <= 1."""
        events = self._burst_events[region]
        if not events:
            return 0.0, 0.0, 0, 0.0

        total_risk = 0.0
        total_failures = 0.0
        suffix_len = 0
        best_failures = 0.0
        best_risk = 0.0
        best_len = 0
        best_ratio = 0.0
        eps = 1e-6

        for event in reversed(events):
            risk_i = max(0.0, float(event["risk"]))
            fail_i = max(0.0, float(event["failures"]))
            total_risk += risk_i
            total_failures += fail_i
            suffix_len += 1

            if suffix_len == 1 and total_risk <= eps and total_failures <= 0.0:
                continue

            if total_risk > 1.0 + 1e-9:
                if suffix_len == 1:
                    eff_risk = max(
                        total_risk, eps if total_failures > 0 else total_risk
                    )
                    ratio = total_failures / eff_risk if eff_risk > 0 else 0.0
                    best_failures = total_failures
                    best_risk = eff_risk
                    best_len = suffix_len
                    best_ratio = ratio
                break

            eff_risk = total_risk
            if eff_risk <= 0.0:
                eff_risk = eps if total_failures > 0 else 0.0

            ratio = total_failures / eff_risk if eff_risk > 0 else 0.0

            if (
                best_len == 0
                or ratio > best_ratio + 1e-9
                or (abs(ratio - best_ratio) <= 1e-9 and suffix_len < best_len)
            ):
                best_failures = total_failures
                best_risk = eff_risk
                best_len = suffix_len
                best_ratio = ratio

        if best_len == 0 and events:
            latest = events[-1]
            best_failures = float(latest["failures"])
            raw_risk = float(latest["risk"])
            best_risk = (
                raw_risk if raw_risk > 0 else (eps if best_failures > 0 else 0.0)
            )
            best_len = 1
            best_ratio = best_failures / best_risk if best_risk > 0 else 0.0

        return best_failures, best_risk, best_len, best_ratio

    def _burst_update_stats(self, region: int) -> None:
        """Update burst statistics for a region."""
        events = self._burst_events[region]
        total_risk = sum(e["risk"] for e in events)

        # Cold start: not enough exposure to detect burst
        if total_risk < 1.0:
            self._jitter_g_hat[region] = 1.0
        else:
            _, _, _, ratio = self._burst_compute_suffix(region)
            self._jitter_g_hat[region] = max(1.0, ratio)

    def _burst_append_event(self, region: int, *, risk: float, failures: int) -> None:
        """Append a burst event and update statistics."""
        risk = max(0.0, float(risk))
        if risk <= 0.0 and failures <= 0:
            return
        menv = typing.cast("env.MultiTraceEnv", self.env)
        event = {"risk": risk, "failures": int(failures), "tick": menv.tick}
        self._burst_events[region].append(event)
        self._burst_prune(region)
        self._burst_update_stats(region)

    # ------------------------------------------------------------------
    # Nelson-Aalen survival curve
    # ------------------------------------------------------------------
    def _na_build_knots(self, region: int) -> Tuple[List[Tuple[float, float]], float]:
        """Build NA knots from historical segments.

        Under virtual availability, the run currently in flight also counts, as a
        censored observation: it has survived this long without failing, which is
        evidence the region is healthy, and a region that has been up for hours
        should not be scored off its last preemption alone.
        """
        epsilon = 1e-8
        state = self._region_state[region]
        segments = state.segments
        if not segments and not self.virtual_availability:
            return [], epsilon

        samples = [
            (seg.duration_h, 1 if seg.outcome == "failure" else 0) for seg in segments
        ]
        if self.virtual_availability:
            if state.current_age_h is not None and state.current_age_h > 0:
                samples.append((state.current_age_h, 0))
            if not samples:
                return [], epsilon
        failure_times = sorted({d for d, f in samples if f == 1 and d >= 0})
        if not failure_times:
            return [], epsilon

        durations = sorted([d for d, _ in samples])
        knots: List[Tuple[float, float]] = []

        for tau in failure_times:
            d_j = sum(1 for d, f in samples if f == 1 and abs(d - tau) <= 1e-9)
            n_j = sum(1 for d in durations if d + 1e-9 >= tau)
            step = (d_j / max(n_j, 1e-12)) if d_j > 0 else 0.0
            if step > 0:
                knots.append((tau, step))

        tail_h = max(knots[-1][1], epsilon) if knots else epsilon
        return knots, tail_h

    def _na_H0_value(
        self, knots: List[Tuple[float, float]], t: float, tail_h: float = 0.0
    ) -> float:
        """Compute cumulative hazard H0(t)."""
        if t <= 0:
            return 0.0
        if not knots:
            return max(0.0, t * tail_h)

        s = 0.0
        last_tau = 0.0
        for tau, step in knots:
            if tau >= t:
                break
            s += step
            last_tau = tau

        if t > last_tau and tail_h > 0:
            s += (t - last_tau) * tail_h

        return s

    def _expected_duration_jitter(self, region: int, age_hours: float) -> float:
        """Compute expected remaining lifetime using NA baseline scaled by jitter."""
        knots, tail_h = self._na_build_knots(region)
        g = self._jitter_g_hat[region] if 0 <= region < len(self._jitter_g_hat) else 1.0
        g = max(1e-6, float(g))

        E = 0.0
        S_rel = 1.0
        t = float(max(0.0, age_hours))
        gap_h = self._gap_seconds / 3600.0 if self._gap_seconds else 1.0

        for tau, step in knots:
            if tau <= t + 1e-12:
                continue
            E += S_rel * (tau - t)
            S_rel *= math.exp(-g * step)
            t = tau

        if tail_h > 1e-12:
            tail_inc = max(tail_h, 1e-12)
            exp_term = math.exp(-g * tail_inc)
            denom = 1.0 - exp_term
            if denom > 1e-9:
                tail_contrib = S_rel * gap_h / denom
            else:
                tail_contrib = S_rel * gap_h / max(g * tail_inc, 1e-9)
            E += tail_contrib

        return max(E, 0.0)

    # ------------------------------------------------------------------
    # Jitter update (once per tick)
    # ------------------------------------------------------------------
    def _jitter_update_once_per_tick(self) -> None:
        """Update jitter state once per tick."""
        menv = typing.cast("env.MultiTraceEnv", self.env)
        tick = menv.tick
        if self._jitter_last_tick == tick:
            return

        dt_h = self._gap_seconds / 3600.0 if self._gap_seconds else 10.0 / 60.0

        for r in range(menv.num_regions):
            knots, tail_h = self._na_build_knots(r)
            tail_h = max(tail_h, 1e-8)

            state = self._region_state[r]
            age_now: Optional[float] = None
            age_prev: Optional[float] = None
            is_active = False

            if state.current_source == "spot":
                is_active = True
                age_now = (
                    state.current_age_h if state.current_age_h is not None else dt_h
                )
                age_prev = self._jitter_last_real_age_h[r]
                if age_prev is None:
                    age_prev = max(age_now - dt_h, 0.0)
                self._jitter_last_real_age_h[r] = age_now
                if not self.virtual_availability:
                    self._jitter_last_virtual_age_h[r] = None
            elif state.current_source == "probe":
                is_active = True
                age_now = (
                    state.current_age_h if state.current_age_h is not None else dt_h
                )
                age_prev = self._jitter_last_virtual_age_h[r]
                if age_prev is None:
                    age_prev = max(age_now - dt_h, 0.0)
                self._jitter_last_virtual_age_h[r] = age_now
                self._jitter_last_real_age_h[r] = None
            else:
                self._jitter_last_real_age_h[r] = None
                if not self.virtual_availability:
                    self._jitter_last_virtual_age_h[r] = None
                # else: keep last_virtual_age_h so repeated probes accumulate age

            # Compute delta_lambda (hazard increment)
            delta_lambda = 0.0
            if age_now is not None and age_prev is not None:
                H0_prev = self._na_H0_value(knots, age_prev, tail_h)
                H0_now = self._na_H0_value(knots, age_now, tail_h)
                delta_lambda = max(0.0, H0_now - H0_prev)

            failures = (
                self._jitter_tick_failures[r]
                if 0 <= r < len(self._jitter_tick_failures)
                else 0
            )

            # Ensure some risk even if no knots
            if delta_lambda <= 0.0 and (is_active or failures > 0):
                if not knots:
                    delta_lambda = max(delta_lambda, tail_h * dt_h)
                else:
                    last_tau = knots[-1][0]
                    if (
                        age_prev is not None
                        and age_now is not None
                        and age_prev >= last_tau
                    ):
                        delta_lambda = max(
                            delta_lambda, tail_h * max(0.0, age_now - age_prev)
                        )

            event_risk = max(0.0, delta_lambda)
            if failures > 0 and event_risk <= 0.0:
                event_risk = max(event_risk, tail_h * dt_h if tail_h > 0 else 1e-6)

            if event_risk > 0.0 or failures > 0:
                self._burst_append_event(r, risk=event_risk, failures=failures)

            self._jitter_tick_failures[r] = 0

        self._jitter_last_tick = tick

    # ------------------------------------------------------------------
    # Duration prediction
    # ------------------------------------------------------------------
    def _get_expected_duration_hours(
        self, region_id: int, remaining_time_seconds: Optional[float] = None
    ) -> float:
        """Return expected SPOT duration using survival curve + jitter."""
        state = self._region_state[region_id]
        age_hours = state.current_age_h if state.current_age_h is not None else 0.0
        if self.virtual_availability and region_id < len(self._probe_age_h):
            # Repeated probes of the same region add up to an age the current
            # segment does not carry; take whichever evidence is longer.
            age_hours = max(age_hours, float(self._probe_age_h[region_id]))
        raw_pred = self._expected_duration_jitter(region_id, age_hours)
        pred = raw_pred

        # Cap the survival prediction. The H100 results were produced with the
        # default 2 h cap; the V100 ones were not -- there the long contiguous
        # availability of the stable zones is the signal region choice runs on,
        # and truncating it to two hours flattens those zones against the
        # volatile ones. The v100 configs raise the cap out of the way. The
        # remaining deadline still caps the prediction below.
        if pred > self.max_prediction_hours:
            pred = self.max_prediction_hours

        if remaining_time_seconds is None:
            remaining_time_seconds = self.deadline - self.env.elapsed_seconds
        if remaining_time_seconds > 0:
            remaining_h = remaining_time_seconds / 3600.0
            if pred > remaining_h:
                pred = remaining_h

        return pred

    # ------------------------------------------------------------------
    # Availability estimation (rebuilt from segments)
    # ------------------------------------------------------------------
    def _estimate_region_availability(self, region: int) -> float:
        """Estimate availability based on segments from _region_state."""
        now = self._now()
        window = self._availability_window_seconds
        window_start = max(0.0, now - window)

        state = self._region_state[region]
        available_seconds = 0.0

        # Sum up time from completed segments within window
        for seg in state.segments:
            seg_start = max(seg.start, window_start)
            seg_end = min(seg.end, now)
            if seg_end > seg_start:
                available_seconds += seg_end - seg_start

        # Add current run if active
        if state.current_start is not None:
            seg_start = max(state.current_start, window_start)
            if now > seg_start:
                available_seconds += now - seg_start

        denominator = min(window, now)
        if denominator <= 0.0:
            return 0.0

        has_history = len(state.segments) > 0 or state.current_start is not None
        if not has_history:
            return 0.0

        return max(0.0, min(1.0, available_seconds / denominator))

    # ------------------------------------------------------------------
    # Time value computation
    # ------------------------------------------------------------------
    def _compute_time_value(self) -> float:
        """Progress value V, formed from an estimated reference cost.

        `_estimate_reference_cost` prices a plan that covers the remaining work by
        mixing on-demand and spot in proportion to each region's estimated
        availability and expected lifetime; V is that price scaled by 1.2.
        """
        D = self.task_duration
        T = self.deadline
        p = sum(self.task_done_time)
        t = self.env.elapsed_seconds

        remaining_task = D - p
        remaining_time = T - t

        if remaining_time <= 0:
            return float("inf")
        if remaining_task <= 0:
            return 0

        c_ref = self._estimate_reference_cost(remaining_task, remaining_time)
        return c_ref * 1.2

    def _compute_od_weight_from_geometry(self) -> float:
        """Compute OD weight using geometric interpretation."""
        D = self.task_duration
        T = self.deadline
        p = sum(self.task_done_time)
        t = self.env.elapsed_seconds
        d = 2 * self.restart_overhead

        if t <= 0:
            return 0.0
        if abs(p - t) < 1e-6:
            return 0.5

        recent_rate = self._estimate_recent_progress_rate()

        if abs(recent_rate - 1.0) < 1e-6:
            return 0

        x_intersect = (recent_rate * t - p + (D + d - T)) / (recent_rate - 1.0)

        if x_intersect >= T:
            return 0.0
        elif x_intersect <= 0:
            return 1.0
        else:
            distance = T - x_intersect
            remaining = T - t
            return max(0.0, min(1.0, distance / remaining))

    def _estimate_recent_progress_rate(self) -> float:
        """Estimate recent progress rate using sliding window."""
        window_seconds = self.progress_window_hours * 3600.0

        if self.env.elapsed_seconds < 600:
            t = self.env.elapsed_seconds
            p = sum(self.task_done_time)
            return (p / t) if t > 0 else 0.0

        current_time = self.env.elapsed_seconds
        window_start_time = max(0, current_time - window_seconds)

        progress_at_window_start = 0.0
        cumulative_progress = 0.0

        tick_duration = self._gap_seconds
        window_start_tick = int(window_start_time / tick_duration)

        for i, tick_work in enumerate(self.task_done_time):
            if i < window_start_tick:
                progress_at_window_start += tick_work
            cumulative_progress += tick_work

        work_in_window = cumulative_progress - progress_at_window_start
        time_in_window = current_time - window_start_time

        if time_in_window > 0:
            recent_rate = work_in_window / time_in_window
        else:
            recent_rate = 0.0

        return max(0.0, min(1.0, recent_rate))

    def _estimate_reference_cost(
        self, remaining_task_seconds: float, remaining_time_seconds: float
    ) -> float:
        """Estimate reference cost based on availability and durations."""
        env_t = typing.cast("env.MultiTraceEnv", self.env)
        num_regions = env_t.num_regions

        od_prices = [
            env_t.envs[r].get_price()[ClusterType.ON_DEMAND] for r in range(num_regions)
        ]
        c_od = min(od_prices)

        region_info = []
        for r in range(num_regions):
            spot_price = env_t.envs[r].get_price()[ClusterType.SPOT]
            availability = self._estimate_region_availability(r)
            expected_duration = self._get_expected_duration_hours(
                r, remaining_time_seconds
            )

            restart_hours = self.restart_overhead / 3600.0
            if expected_duration > restart_hours:
                effective_rate = (expected_duration - restart_hours) / expected_duration
            else:
                effective_rate = 0.0

            expected_contribution = (
                availability * remaining_time_seconds * effective_rate
            )

            if expected_contribution > 0:
                effective_duration_hours = max(expected_duration - restart_hours, 1e-6)
                adjusted_price = spot_price
                if effective_duration_hours > 0:
                    adjusted_price = (
                        spot_price * expected_duration / effective_duration_hours
                    )

                region_info.append(
                    {
                        "region": r,
                        "price": spot_price,
                        "adjusted_price": adjusted_price,
                        "availability": availability,
                        "duration_hours": expected_duration,
                        "effective_rate": effective_rate,
                        "contribution": expected_contribution,
                    }
                )

        region_info.sort(key=lambda x: x["adjusted_price"])

        od_weight = self._compute_od_weight_from_geometry()
        od_fraction = max(0.0, min(1.0, od_weight))

        remaining_task = remaining_task_seconds
        od_assigned = od_fraction * remaining_task
        od_total_assigned = od_assigned
        spot_target = max(0.0, remaining_task - od_assigned)

        total_cost = od_assigned / 3600.0 * c_od
        spot_assigned = 0.0
        spot_cost = 0.0
        contributing_regions = []

        for info in region_info:
            if spot_assigned >= spot_target:
                break
            available = min(info["contribution"], spot_target - spot_assigned)
            if available <= 0:
                continue
            contributing_regions.append(info)
            spot_cost += (available / 3600.0) * info["adjusted_price"]
            spot_assigned += available

        if spot_assigned < spot_target:
            deficit = spot_target - spot_assigned
            od_total_assigned += deficit
            total_cost += (deficit / 3600.0) * c_od
        total_cost += spot_cost

        c_ref = total_cost / max(remaining_task / 3600.0, 1e-6)

        # Add egress cost
        checkpoint_size = getattr(self.task, "checkpoint_size_gb", 0)
        if (contributing_regions or od_total_assigned > 0) and checkpoint_size > 0:
            num_regions_used = len(contributing_regions)
            if od_total_assigned > 0:
                num_regions_used += 1
            if num_regions_used > 1:
                num_switches = max(0, num_regions_used - 1)
                egress_per_gb = 0.02
                egress_total = num_switches * checkpoint_size * egress_per_gb
                egress_per_hour = egress_total / max(remaining_task / 3600.0, 1e-6)
                c_ref += egress_per_hour

        return c_ref

    # ------------------------------------------------------------------
    # Migration penalty
    # ------------------------------------------------------------------
    def _compute_migration_penalty(
        self,
        from_region: int,
        to_region: int,
        checkpoint_size_gb: float,
        V: float,
        duration_hours: Optional[float] = None,
    ) -> Tuple[float, Optional[dict]]:
        """Compute migration penalty."""
        if from_region == to_region:
            return 0.0, None

        env_t = typing.cast("env.MultiTraceEnv", self.env)
        from_name = env_t.get_region_name(from_region)
        to_name = env_t.get_region_name(to_region)

        try:
            transfer_time = get_transfer_time_hours(
                from_name, to_name, checkpoint_size_gb
            )
        except Exception:
            transfer_time = 0.0
        try:
            relationship = get_region_relationship(from_name, to_name)
        except Exception:
            relationship = None

        effective_transfer = transfer_time
        if (
            relationship in {"cross_region", "cross_continent"}
            and not env_t._count_cross_region_migration_time
        ):
            effective_transfer = 0.0

        transfer_usd = get_transfer_cost_usd(from_name, to_name, checkpoint_size_gb)

        if duration_hours is not None:
            downtime_value = effective_transfer * V
            penalty_usd = downtime_value + transfer_usd
            penalty = penalty_usd / max(duration_hours, 1e-6)
        else:
            instance_startup_hours = self.restart_overheads[0] / 3600.0
            migration_hours = instance_startup_hours + effective_transfer
            downtime_value = migration_hours * V
            penalty = downtime_value + transfer_usd

        if penalty > 0:
            return penalty, {
                "transfer_hours": effective_transfer,
                "transfer_cost_usd": transfer_usd,
            }
        return 0.0, None

    # ------------------------------------------------------------------
    # Cost computation
    # ------------------------------------------------------------------
    def _compute_spot_net_value(
        self, region_id: int, V: float, ignore_restart_overhead: bool = False
    ) -> float:
        """Compute expected net value of SPOT in a region."""
        duration = self._get_expected_duration_hours(region_id)
        restart_hours = self.restart_overhead / 3600.0

        if ignore_restart_overhead:
            effective_duration = max(0.0, duration)
        else:
            effective_duration = max(0.0, duration - restart_hours)

        if effective_duration <= 0:
            return -float("inf")

        env_t = typing.cast("env.MultiTraceEnv", self.env)
        c_spot = env_t.envs[region_id].get_price()[ClusterType.SPOT]

        value_gained = V * effective_duration
        cost_incurred = c_spot * duration
        net_value = value_gained - cost_incurred
        net_value_per_hour = net_value / duration

        return net_value_per_hour

    def _compute_od_net_value(self, V: float, region_id: int) -> float:
        """Compute net value of ON_DEMAND."""
        env_t = typing.cast("env.MultiTraceEnv", self.env)
        c_od = env_t.envs[region_id].get_price()[ClusterType.ON_DEMAND]

        return V - c_od

    # ------------------------------------------------------------------
    # Detect preemptions at step start
    # ------------------------------------------------------------------
    def _detect_preemptions(self, active_instances: dict[int, ClusterType]) -> None:
        """Detect if active SPOT was preempted and record end event."""
        if self._active_spot_region is not None:
            still_running = (
                self._active_spot_region in active_instances
                and active_instances[self._active_spot_region] == ClusterType.SPOT
            )
            if not still_running:
                outcome = (
                    self.EVENT_CENSOR
                    if self._spot_terminated_by_us
                    else self.EVENT_FAILURE
                )
                self._record_end(self._active_spot_region, "spot", outcome)
                # Availability is now computed from segments in _region_state
                self._update_virtual_availability(self._active_spot_region, False)
                self._active_spot_region = None
                self._active_spot_start = 0.0
                self._spot_terminated_by_us = False

    # ------------------------------------------------------------------
    # Handle action results - record data points
    # ------------------------------------------------------------------
    def _on_spot_launch_success(self, region: int) -> None:
        """Called when SPOT launch succeeds."""
        self._record_start(region, "spot")
        self._active_spot_region = region
        self._active_spot_start = self._now()
        self._spot_terminated_by_us = False
        self._last_region = region
        self._last_failed_regions.discard(region)
        # Reset jitter tracking for real run
        self._jitter_last_real_age_h[region] = 0.0
        self._jitter_last_virtual_age_h[region] = None
        # Virtual availability now represented by real run
        self._update_virtual_availability(region, False)

    def _on_spot_launch_failure(self, region: int) -> None:
        """Called when SPOT launch fails."""
        self._last_failed_regions.add(region)
        self._jitter_tick_failures[region] += 1
        self._update_virtual_availability(region, False)

    def _on_probe_success(self, region: int) -> None:
        """Called when probe succeeds (availability confirmed)."""
        if region in self._probes_active:
            # Only reachable under --probe-revalidate, where a region with a
            # live virtual instance is probed again. The virtual run is
            # continuing, so do not open a second "start" event for it -- that
            # would leave the raw event list with two starts and no end between
            # them and corrupt every duration statistic derived from it. The
            # virtual run's age is not reset either, for the same reason.
            return
        self._record_start(region, "probe")
        self._probes_active.add(region)
        # Reset jitter tracking for virtual run
        self._jitter_last_virtual_age_h[region] = 0.0
        self._update_virtual_availability(region, True)

    def _on_probe_failure(self, region: int) -> None:
        """Called when probe fails."""
        # If probe was running, record its end
        if region in self._probes_active:
            self._record_end(region, "probe", self.EVENT_FAILURE)
        self._probes_active.discard(region)
        self._jitter_tick_failures[region] += 1
        self._jitter_last_virtual_age_h[region] = None
        self._update_virtual_availability(region, False)

    def _on_spot_terminate(self, region: int) -> None:
        """Called when we terminate SPOT (voluntary)."""
        if self._active_spot_region == region:
            self._spot_terminated_by_us = True
        self._update_virtual_availability(region, False)

    # ------------------------------------------------------------------
    # Post-step: schedule probes
    # ------------------------------------------------------------------
    def _post_step_actions(
        self,
        *,
        decision_outcome: str = "",  # noqa: ARG002
        primary_region: Optional[int] = None,
        primary_type: ClusterType = ClusterType.NONE,  # noqa: ARG002
        active_instances: Optional[dict[int, ClusterType]] = None,  # noqa: ARG002
    ) -> typing.Iterable[Action]:
        """Schedule probe launches for regions.

        A region whose probe succeeds enters `self._probes_active` and, with the
        default `probe_revalidate=False`, is not probed again: the virtual
        instance built from that probe carries the region's state forward, and
        the survival model extrapolates its lifetime from there rather than from
        repeated sampling. `env._probe_internal` bills one minute of spot per
        successful ProbeLaunch, so the charge is one probe per region per job.
        The region leaves the set when a real SPOT launch happens there.

        The probe-based baselines sample instead of extrapolating, so they have
        to re-probe: `multi_region_availability_probe_simple` scores a region by
        the fraction of successful probes in a sliding window, and
        `multi_region_probe_cost_ratio` divides that by price, so both reset
        `_next_probe_tick` unconditionally and pay on every interval.

        `--probe-revalidate` makes this strategy re-probe and re-bill on
        `--probe-interval-ticks` as they do. It is off by default, which is what
        produced research/data/*.csv and the published figures.

        Note: decision_outcome, primary_type, active_instances are provided for
        subclass compatibility but not used in base implementation.
        """
        menv = typing.cast("env.MultiTraceEnv", self.env)
        now = menv.tick
        interval = max(1, self.probe_interval_ticks)
        blocked = set(self._last_failed_regions)

        for ridx in range(menv.num_regions):
            if ridx == primary_region:
                continue
            if ridx in self._probes_active and not self.probe_revalidate:
                continue
            if ridx in blocked:
                continue
            if now < self._next_probe_tick[ridx]:
                continue
            yield ProbeLaunch(region=ridx)
            self._next_probe_tick[ridx] = now + interval

        self._last_failed_regions.clear()

    def _before_spot_launch(self, region: int) -> None:
        """Clean up before launching real SPOT - end probe if running."""
        if region in self._probes_active:
            # Record probe end as censor (we're taking over)
            self._record_end(region, "probe", self.EVENT_CENSOR)
            self._probes_active.discard(region)
            self._jitter_last_virtual_age_h[region] = None

    # ------------------------------------------------------------------
    # Decision logging
    # ------------------------------------------------------------------
    def _log_burst_state(self, env_t: "env.MultiTraceEnv") -> None:
        """Log burst/jitter state for all regions."""
        dlog.log_burst_header()
        for r in range(env_t.num_regions):
            events = list(self._burst_events[r])
            # _burst_compute_suffix returns (failures, risk, length, ratio)
            fail, risk, suffix_len, _ = (
                self._burst_compute_suffix(r) if events else (0, 0, 0, 0)
            )
            g = self._jitter_g_hat[r]
            dlog.log_burst_region(r, events, suffix_len, risk, fail, g)

    def _log_duration_state(self, env_t: "env.MultiTraceEnv") -> None:
        """Log duration predictions for all regions."""
        dlog.log_duration_header()
        for r in range(env_t.num_regions):
            knots, tail = self._na_build_knots(r)
            g = self._jitter_g_hat[r]
            state = self._region_state[r]
            age = state.current_age_h if state.current_age_h else 0.0
            e_dur = self._get_expected_duration_hours(r)
            avail = self._estimate_region_availability(r)
            dlog.log_duration_region(r, knots, tail, g, age, e_dur, avail)

    def _log_time_value(self, V: float) -> None:
        """Log time value computation."""
        dlog.log_time_value_header()
        rate = self._estimate_recent_progress_rate()
        od_w = self._compute_od_weight_from_geometry()
        # Simplified logging
        c_ref = V / 1.2 if V > 0 else 0
        dlog.log_time_value(od_w, rate, None, c_ref, 0, V, "")

    def _log_candidates(
        self,
        V: float,
        overhead_h: float,
        from_region: Optional[int],
        env_t: "env.MultiTraceEnv",
        checkpoint_gb: float,
    ) -> Tuple[List[Tuple[float, int, float, float, float]], Optional[int]]:
        """Log and compute candidates. Returns (candidates, best_region)."""
        dlog.log_candidates_header(V, overhead_h, from_region)

        # Collect all candidates with their values
        all_candidates: List[
            Tuple[float, int, float, float, float, float, str]
        ] = []  # (net, r, e_dur, spot_price, egress, avail, skip_reason)

        for r in range(env_t.num_regions):
            e_dur = self._get_expected_duration_hours(r)
            avail = self._estimate_region_availability(r)
            spot_price = env_t.envs[r].get_price()[ClusterType.SPOT]

            # Egress cost
            egress = 0.0
            if from_region is not None and from_region != r:
                egress = get_transfer_cost_usd(
                    env_t.get_region_name(from_region),
                    env_t.get_region_name(r),
                    checkpoint_gb,
                )

            # Net value: V × (1 - overhead/E) - spot_price - egress/E
            # For current region: no overhead (instance already running)
            is_current_region = from_region is not None and from_region == r
            if e_dur > 0:
                if is_current_region:
                    eff_ratio = 1.0  # No overhead for current running region
                else:
                    eff_ratio = 1 - overhead_h / e_dur if e_dur > overhead_h else 0
                net = (
                    V * eff_ratio - spot_price - egress / e_dur
                    if e_dur > 0
                    else -float("inf")
                )
            else:
                net = -float("inf")

            # Determine skip reason
            skip_reason = ""
            if e_dur <= 0:
                skip_reason = "E=0"

            all_candidates.append(
                (net, r, e_dur, spot_price, egress, avail, skip_reason)
            )

        # Sort by net value (high to low)
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        # Find best viable candidate
        best_region = None
        best_value = -float("inf")
        viable_candidates = []

        for net, r, e_dur, spot_price, egress, avail, skip_reason in all_candidates:
            if not skip_reason and net > best_value:
                best_value = net
                best_region = r
            if not skip_reason:
                viable_candidates.append((net, r, e_dur, spot_price, egress))

        # Log all candidates (sorted by net value)
        for net, r, e_dur, spot_price, egress, avail, skip_reason in all_candidates:
            is_same = from_region is not None and from_region == r
            # For current region: no overhead (instance already running)
            if is_same:
                eff_ratio = 1.0
            else:
                eff_ratio = 1 - overhead_h / e_dur if e_dur > overhead_h else 0

            if skip_reason:
                # Still show formula even for skipped
                dlog.log(
                    f"    R{r}: E={e_dur:.1f}h, spot=${spot_price:.2f}/h, avail={avail * 100:.0f}% [{skip_reason}]"
                )
                if e_dur > 0:
                    egress_str = f" - {egress:.2f}/{e_dur:.1f}" if egress > 0 else ""
                    dlog.log(
                        f"        net = {V:.2f}×{eff_ratio:.2f} - {spot_price:.2f}{egress_str} = ${net:.2f}/h"
                    )
            else:
                egress_str = (
                    f", egress=${egress:.2f}"
                    if egress > 0
                    else (", egress=$0 (same)" if is_same else "")
                )
                dlog.log(
                    f"    R{r}: E={e_dur:.1f}h, spot=${spot_price:.2f}/h{egress_str}"
                )
                egress_calc = f" - {egress:.2f}/{e_dur:.1f}" if egress > 0 else ""
                dlog.log(
                    f"        net = {V:.2f}×{eff_ratio:.2f} - {spot_price:.2f}{egress_calc} = ${net:.2f}/h"
                )

        return viable_candidates, best_region

    # ------------------------------------------------------------------
    # Core decision logic
    # ------------------------------------------------------------------
    def _step_multi(self) -> typing.Generator[Action, Optional[LaunchResult], None]:
        """Pure unified cost model decision making."""
        env_t = typing.cast("env.MultiTraceEnv", self.env)

        # 1. Detect preemptions (record end events)
        active_instances = env_t.get_active_instances()
        self._detect_preemptions(dict(active_instances))

        # 2. Rebuild derived state from raw events
        self._rebuild_derived_state()

        # 3. Update jitter model
        self._jitter_update_once_per_tick()

        # 4. Check if task done
        remaining_task = self.task_duration - sum(self.task_done_time)
        if remaining_task <= 1e-3:
            for region in list(active_instances.keys()):
                yield Terminate(region=region)
            return

        # 5. Compute time value
        V = self._compute_time_value()
        logger.debug(f"[UCM] V={V:.3f}")

        od_values = [
            self._compute_od_net_value(V, rid) for rid in range(env_t.num_regions)
        ]

        # 6. Current state
        current_region = None
        current_type = ClusterType.NONE
        current_value = 0.0

        if active_instances:
            assert len(active_instances) == 1, active_instances
            current_region = list(active_instances.keys())[0]
            current_type = active_instances[current_region]

            if current_type == ClusterType.SPOT:
                current_value = self._compute_spot_net_value(
                    current_region, V, ignore_restart_overhead=True
                )
            elif current_type == ClusterType.ON_DEMAND:
                current_value = od_values[current_region]

            logger.info(
                f"Current: {current_type.name} in R{current_region} (value={current_value:.3f})"
            )
            self._last_region = current_region

        # === DECISION LOGGING ===
        progress_pct = (
            100 * sum(self.task_done_time) / self.task_duration
            if self.task_duration > 0
            else 0
        )
        dlog.log_tick_header(
            env_t.tick,
            self.env.elapsed_seconds / 3600,
            progress_pct,
            remaining_task / 3600,
        )

        # Log preemption status
        if self._active_spot_region is None and self._last_region is not None:
            dlog.log_preemption(
                self._last_region,
                "PREEMPTED (failure)"
                if not self._spot_terminated_by_us
                else "terminated (censor)",
            )
        elif current_region is not None:
            dlog.log_preemption(current_region, f"RUNNING ({current_type.name})")
        else:
            dlog.log_preemption(None, "")

        self._log_burst_state(env_t)
        self._log_duration_state(env_t)
        self._log_time_value(V)

        # 7. Build options
        wait_value = 0.0
        all_options: list[tuple[float, str, Optional[int]]] = [
            (wait_value, "NONE", None)
        ]
        checkpoint_size_gb = getattr(self.task, "checkpoint_size_gb", 0)

        # Log candidates
        origin = current_region if current_region is not None else self._last_region
        overhead_h = self.restart_overhead / 3600
        self._log_candidates(V, overhead_h, origin, env_t, checkpoint_size_gb)

        # Add SPOT options
        for region_id in range(env_t.num_regions):
            ignore_restart = (
                current_type == ClusterType.SPOT and region_id == current_region
            )
            base = self._compute_spot_net_value(
                region_id, V, ignore_restart_overhead=ignore_restart
            )

            # Migration penalty
            per_candidate_penalty = 0.0
            origin_region = (
                current_region if current_region is not None else self._last_region
            )
            if self.consider_migration_penalty and origin_region is not None:
                duration = self._get_expected_duration_hours(region_id)
                per_candidate_penalty, _ = self._compute_migration_penalty(
                    origin_region, region_id, checkpoint_size_gb, V, duration
                )

            value = base - per_candidate_penalty

            if value < 0:
                continue

            # VOI gating
            if (
                self.voi_gating
                and current_region is not None
                and region_id != current_region
            ):
                delta = (
                    self.voi_delta_base
                    + self.voi_k_checkpoint * checkpoint_size_gb
                    + self.voi_k_v * V
                )
                if value <= current_value + delta:
                    continue

            all_options.append((value, "SPOT", region_id))

        # Add ON_DEMAND options
        origin_region = (
            current_region if current_region is not None else self._last_region
        )
        for od_region in range(env_t.num_regions):
            od_base = od_values[od_region]
            od_penalty = 0.0

            if self.consider_migration_penalty and origin_region is not None:
                od_penalty, _ = self._compute_migration_penalty(
                    origin_region, od_region, checkpoint_size_gb, V
                )

            od_option_value = od_base - od_penalty

            if od_option_value < 0:
                continue
            if current_type == ClusterType.ON_DEMAND and current_region == od_region:
                continue

            if self.voi_gating and current_region is not None:
                delta = (
                    self.voi_delta_base
                    + self.voi_k_checkpoint * checkpoint_size_gb
                    + self.voi_k_v * V
                )
                if od_option_value <= current_value + delta:
                    continue

            all_options.append((od_option_value, "ON_DEMAND", od_region))

        # 8. Sort by value
        all_options.sort(key=lambda x: x[0], reverse=True)
        all_options = [opt for opt in all_options if opt[0] >= 0]

        # 9. Try options
        attempts_this_tick = 0
        for value, cluster_type_str, region in all_options:
            if value <= current_value:
                break

            if region == current_region and cluster_type_str == current_type.name:
                continue

            launch_success = False

            if cluster_type_str == "SPOT":
                assert region is not None
                # Clean up probe if running in this region
                self._before_spot_launch(region)

                logger.info(f"Trying SPOT in region {region} (value={value:.3f})")
                action = TryLaunch(region=region, cluster_type=ClusterType.SPOT)
                result = yield action
                assert result is not None
                attempts_this_tick += 1

                if result.success:
                    self._on_spot_launch_success(region)
                    launch_success = True
                    dlog.log_launch_attempt(
                        attempts_this_tick, region, "SPOT", "SUCCESS"
                    )
                else:
                    self._on_spot_launch_failure(region)
                    dlog.log_launch_attempt(attempts_this_tick, region, "SPOT", "FAIL")

            elif cluster_type_str == "ON_DEMAND":
                assert region is not None
                action = TryLaunch(region=region, cluster_type=ClusterType.ON_DEMAND)
                result = yield action
                assert result is not None
                launch_success = result.success

            else:  # None
                launch_success = True

            if launch_success:
                self._last_region = region
                # Terminate old instance only if we launched something new
                # Don't terminate if we chose NONE - keep current instance running
                if current_region is not None and cluster_type_str != "NONE":
                    logger.info(
                        f"Terminating old {current_type.name} in R{current_region}"
                    )
                    self._on_spot_terminate(current_region)
                    yield Terminate(region=current_region)

                # Post-step: schedule probes and handle results
                post_primary = (
                    region if cluster_type_str in ("SPOT", "ON_DEMAND") else None
                )
                post_type = (
                    ClusterType.SPOT
                    if cluster_type_str == "SPOT"
                    else ClusterType.ON_DEMAND
                    if cluster_type_str == "ON_DEMAND"
                    else ClusterType.NONE
                )
                active_now = dict(env_t.get_active_instances())
                for probe_action in self._post_step_actions(
                    decision_outcome="launch_success",
                    primary_region=post_primary,
                    primary_type=post_type,
                    active_instances=active_now,
                ):
                    probe_result = yield probe_action
                    if isinstance(probe_action, ProbeLaunch):
                        if probe_result and probe_result.success:
                            self._on_probe_success(probe_action.region)
                        else:
                            self._on_probe_failure(probe_action.region)
                return

            if (
                self.max_attempts_per_tick > 0
                and attempts_this_tick >= self.max_attempts_per_tick
            ):
                break

        # 10. No viable options - keep current
        logger.debug(
            f"No viable options, keeping {current_type.name} in R{current_region}"
        )

        # Post-step: schedule probes and handle results
        active_now = dict(env_t.get_active_instances())
        for probe_action in self._post_step_actions(
            decision_outcome="keep_current",
            primary_region=current_region,
            primary_type=current_type,
            active_instances=active_now,
        ):
            probe_result = yield probe_action
            if isinstance(probe_action, ProbeLaunch):
                if probe_result and probe_result.success:
                    self._on_probe_success(probe_action.region)
                else:
                    self._on_probe_failure(probe_action.region)

    @classmethod
    def _from_args(cls, parser, namespace=None):
        group = parser.add_argument_group("UnifiedCostModelStrategy")
        group.add_argument(
            "--max-attempts-per-tick",
            type=int,
            default=0,
            help="Max SPOT regions to try per tick (0=unlimited)",
        )
        group.add_argument(
            "--ignore-migration-penalty",
            action="store_true",
            help="Ignore migration time and transfer cost",
        )
        group.add_argument(
            "--voi-gating",
            action="store_true",
            default=True,
            help="Enable VOI gating for exploration",
        )
        group.add_argument("--no-voi-gating", action="store_false", dest="voi_gating")
        group.add_argument(
            "--voi-delta-base",
            type=float,
            default=0.5,
            help="Base margin (USD/h) for VOI gating",
        )
        group.add_argument(
            "--voi-k-checkpoint",
            type=float,
            default=0.0002,
            help="Additional margin per GB checkpoint",
        )
        group.add_argument(
            "--voi-k-v",
            type=float,
            default=0.05,
            help="Additional margin proportional to V",
        )
        group.add_argument(
            "--probe-interval-ticks",
            type=int,
            default=6,
            help="Ticks between probe attempts",
        )
        group.add_argument(
            "--probe-revalidate",
            action="store_true",
            default=False,
            help="Re-probe a region every --probe-interval-ticks even when it "
                 "already has a live virtual instance, and pay for each probe, "
                 "the way the sampling-based baselines do. Off by default, "
                 "which is how the shipped results and the published figures "
                 "were produced: one probe per region per job, with the virtual "
                 "instance carrying the region's state from there.",
        )
        group.add_argument(
            "--virtual-availability",
            action="store_true",
            default=False,
            help="Keep tracking a region's availability as a run even when "
                 "nothing is launched there, so the survival estimate of a "
                 "region the policy has stopped using still gains observations. "
                 "Off by default, which is the setting behind the paper's "
                 "figures; the appendix's ablation configs turn it on.",
        )
        group.add_argument(
            "--max-prediction-hours",
            type=float,
            default=2.0,
            help="Maximum duration prediction horizon",
        )
        group.add_argument(
            "--progress-window-hours",
            type=float,
            default=10.0,
            help="Sliding window for progress rate",
        )
        # If namespace provided (programmatic call), don't re-parse sys.argv
        # If namespace is None (CLI call), parse sys.argv normally
        if namespace is not None:
            return cls(parser.parse_args(args=[], namespace=namespace))
        return cls(parser.parse_args())
