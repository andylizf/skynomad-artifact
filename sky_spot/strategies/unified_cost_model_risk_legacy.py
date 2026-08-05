"""Unified Cost Model with survival-based duration estimate and probe risk blending."""

import logging
import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import typing

from sky_spot.strategies.unified_cost_model_legacy import UnifiedCostModelStrategy
from sky_spot.strategies.structured_logger import StructuredLogger
from sky_spot.multi_region_types import (
    Action,
    ClusterType,
    LaunchResult,
    Terminate,
    TryLaunch,
    ProbeLaunch,
)

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task

logger = logging.getLogger(__name__)


class UnifiedCostModelRiskStrategy(UnifiedCostModelStrategy):
    """Unified Cost Model variant using KM survival curves plus multi-window risk."""

    NAME = "unified_cost_model_risk_legacy"
    LOG_PREFIX = "[UCM_RISK]"
    EVENT_FAILURE = "failure"
    EVENT_CENSOR = "censor"

    def __init__(self, args):
        self.probe_interval_ticks: int = getattr(args, "risk_probe_interval_ticks", 12)
        # Historical decay horizon (hours). Shorter horizon forgets old failures faster.
        self.max_prediction_hours: float = getattr(args, "risk_max_cap_hours", 24.0)
        # Progress rate sliding window size (hours)
        self.progress_window_hours: float = getattr(
            args, "risk_progress_window_hours", 10.0
        )

        super().__init__(args)
        self.enable_candidate_debug_summary = True

        # Historical log stores append-only runtime events.
        self._historical_logs: List[List[dict[str, typing.Any]]] = []
        # Current status keeps the virtual probe-derived run window.
        self._current_status: List[dict[str, Optional[float]]] = []

        self._next_probe_tick: List[int] = []
        self._probes_active: Set[int] = set()
        self._last_failed_regions: Set[int] = set()
        self._region_active_since_seconds: List[Optional[float]] = []
        self._availability_window_seconds: float = 12 * 3600.0
        self._availability_segments: List[deque[Tuple[float, float]]] = []

        # Burst-based jitter model state per region
        self._burst_events: List[deque[dict[str, typing.Any]]] = []
        self._burst_suffix_failures: List[float] = []
        self._burst_suffix_risk: List[float] = []
        self._burst_suffix_len: List[int] = []
        self._burst_suffix_ratio: List[float] = []
        self._jitter_g_hat: List[float] = []
        self._jitter_g_hat_raw: List[float] = []
        self._jitter_last_real_age_h: List[Optional[float]] = []
        self._jitter_last_virtual_age_h: List[Optional[float]] = []
        self._jitter_last_subject: List[str] = []
        self._jitter_tick_failures: List[int] = []
        self._jitter_last_tick: int = -1

        # Structured logger
        self._structured_logger = StructuredLogger(
            log_level=getattr(args, "structured_log_level", "INFO")
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _format_log_message(self, message: str) -> str:
        if not message:
            return self.LOG_PREFIX
        stripped = message.lstrip()
        if stripped.startswith("["):
            return message
        return f"{self.LOG_PREFIX} {message}"

    def _log_debug(self, message: str) -> None:
        logger.debug(self._format_log_message(message))

    @classmethod
    def _from_args(cls, parser):
        group = parser.add_argument_group("UnifiedCostModelRiskStrategy")
        group.add_argument(
            "--risk-probe-interval-ticks",
            type=int,
            default=6,
            help="Spacing between probe samples in ticks (default: 12 -> 2h for 10 min traces)",
        )
        group.add_argument(
            "--risk-max-cap-hours",
            type=float,
            default=24.0,
            help="Maximum horizon the duration predictor is willing to forecast",
        )
        group.add_argument(
            "--risk-progress-window-hours",
            type=float,
            default=10.0,
            help="Sliding window size (hours) for recent progress rate calculation",
        )

        return super()._from_args(parser)

    def reset(self, env: "env.Env", task: "task.Task"):
        super().reset(env, task)
        menv = typing.cast("env.MultiTraceEnv", self.env)
        num_regions = menv.num_regions
        self._historical_logs = [[] for _ in range(num_regions)]
        self._current_status = [
            {"virtual_start_time": None, "last_updated_time": None}
            for _ in range(num_regions)
        ]
        self._next_probe_tick = [0 for _ in range(num_regions)]
        self._last_failed_regions.clear()
        # Preserve probe state across resets - DON'T clear _probes_active
        # The probe instances should continue running until they are explicitly terminated
        # Only clear other state that should be reset
        self._log_debug(
            f"reset completed - preserved _probes_active: {self._probes_active}"
        )
        self._region_active_since_seconds = [None for _ in range(num_regions)]
        self._availability_segments = [deque() for _ in range(num_regions)]
        # Initialize burst-based jitter state
        self._burst_events = [deque() for _ in range(num_regions)]
        self._burst_suffix_failures = [0.0 for _ in range(num_regions)]
        self._burst_suffix_risk = [0.0 for _ in range(num_regions)]
        self._burst_suffix_len = [0 for _ in range(num_regions)]
        self._burst_suffix_ratio = [0.0 for _ in range(num_regions)]
        self._jitter_g_hat = [1.0 for _ in range(num_regions)]
        self._jitter_g_hat_raw = [0.0 for _ in range(num_regions)]
        self._jitter_last_real_age_h = [None for _ in range(num_regions)]
        self._jitter_last_virtual_age_h = [None for _ in range(num_regions)]
        self._jitter_last_subject = ["idle" for _ in range(num_regions)]
        self._jitter_tick_failures = [0 for _ in range(num_regions)]
        self._jitter_last_tick = -1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _collect_decision_options_for_logger(
        self,
        rows: dict,
        option_details: list,
        current_region: int,
        current_type: ClusterType,
    ) -> None:
        """Collect decision options for structured logger."""
        # Add spot options from rows
        for region, data in rows.items():
            value = data.get("candidate_value", data.get("net_value_per_hour"))
            duration = data.get("duration_hours")
            spot_price = data.get("spot_price_usd_per_hour")
            effective = data.get("effective_duration_hours")
            penalty = data.get("migration_penalty_per_hour")
            is_current = region == current_region and current_type == ClusterType.SPOT

            self._structured_logger.add_decision_option(
                region=region,
                cluster_type="SPOT",
                value=value or 0,
                duration_hours=duration,
                spot_price=spot_price,
                effective_hours=effective,
                penalty=penalty,
                is_current=is_current,
                status=data.get("status", "candidate"),
            )

        # Add ON_DEMAND options
        for detail in option_details:
            if detail.get("type") == "ON_DEMAND":
                self._structured_logger.add_decision_option(
                    region=detail.get("region"),
                    cluster_type="ON_DEMAND",
                    value=detail.get("candidate_value", 0),
                    spot_price=detail.get("c_od"),
                    penalty=detail.get("penalty"),
                    status=detail.get("status", "candidate"),
                )

    def _update_logger_history(self, region: int) -> None:
        """Update structured logger with region's history."""
        assert 0 <= region < len(self._historical_logs), (
            f"Invalid region {region}, must be in range [0, {len(self._historical_logs)})"
        )

        # Replace the structured logger history so we don't accumulate duplicates
        self._structured_logger.clear_history(region)

        # Build timeline segments
        now = self._now()
        for entry in self._historical_logs[region]:
            runtime_h = entry.get("runtime_hours", 0.0)
            event_type = entry.get("event", self.EVENT_CENSOR)
            timestamp = entry.get("timestamp", now)
            start_h = (timestamp - runtime_h * 3600.0) / 3600.0

            event_char = "F" if event_type == self.EVENT_FAILURE else "C"
            self._structured_logger.add_history_segment(
                region, start_h, runtime_h, event_char
            )

        # Add live segments
        status = self._current_status_row(region)
        if status and status.get("virtual_start_time"):
            start_h = status["virtual_start_time"] / 3600.0
            duration_h = (now - status["virtual_start_time"]) / 3600.0
            self._structured_logger.add_history_segment(
                region, start_h, duration_h, "L"
            )

        # Add active spot segment
        if 0 <= region < len(self._region_active_since_seconds):
            active_start = self._region_active_since_seconds[region]
            if active_start:
                start_h = active_start / 3600.0
                duration_h = (now - active_start) / 3600.0
                self._structured_logger.add_history_segment(
                    region, start_h, duration_h, "L"
                )

    def _burst_prune(self, region: int, max_risk: float = 8.0) -> None:
        """Keep the burst event deque bounded by removing oldest risk mass."""
        events = self._burst_events[region]
        if not events:
            return

        total_risk = sum(event["risk"] for event in events)
        while events and total_risk - events[0]["risk"] >= max_risk:
            oldest = events.popleft()
            total_risk -= oldest["risk"]

    def _burst_compute_suffix(self, region: int) -> Tuple[float, float, int, float]:
        """Return (failures, risk, length, ratio) for the densest suffix with risk ≤ 1."""
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
                # no information yet – keep looking
                continue

            # Stop extending once the risk budget (1.0) is exceeded.
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

        if best_len == 0:
            # Fallback to the most recent event if nothing satisfied the constraints
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
        """Recompute burst suffix statistics and current jitter multiplier."""
        failures, risk, suffix_len, ratio = self._burst_compute_suffix(region)
        self._burst_suffix_failures[region] = failures
        self._burst_suffix_risk[region] = risk
        self._burst_suffix_len[region] = suffix_len
        self._burst_suffix_ratio[region] = ratio
        g = max(1.0, ratio)
        self._jitter_g_hat[region] = g
        self._jitter_g_hat_raw[region] = ratio

    def _burst_append_event(
        self,
        region: int,
        *,
        risk: float,
        failures: int,
    ) -> None:
        """Append a new observation to the burst deque and refresh state."""
        risk = max(0.0, float(risk))
        if risk <= 0.0 and failures <= 0:
            return

        menv = typing.cast("env.MultiTraceEnv", self.env)
        current_tick = menv.tick
        event = {"risk": risk, "failures": int(failures), "tick": current_tick}
        self._burst_events[region].append(event)
        self._burst_prune(region)
        self._burst_update_stats(region)

    # ------------------------------------------------------------------
    # Duration data model helpers
    # ------------------------------------------------------------------
    def _now(self) -> float:
        return float(self.env.elapsed_seconds)

    def _current_status_row(self, region: int) -> dict[str, Optional[float]]:
        assert 0 <= region < len(self._current_status), (
            f"Invalid region {region}, must be in range [0, {len(self._current_status)})"
        )
        return self._current_status[region]

    def _append_historical_event(
        self,
        region: int,
        runtime_hours: float,
        event_type: str,
        timestamp: Optional[float] = None,
        *,
        source: str = "unknown",
    ) -> None:
        assert 0 <= region < len(self._historical_logs), (
            f"Invalid region {region}, must be in range [0, {len(self._historical_logs)})"
        )
        runtime = max(0.0, float(runtime_hours))
        ts = self._now() if timestamp is None else float(timestamp)
        self._historical_logs[region].append(
            {
                "runtime_hours": runtime,
                "event": event_type,
                "timestamp": ts,
                "source": source,
            }
        )

    def _touch_virtual_run(self, region: int) -> None:
        status = self._current_status_row(region)
        now = self._now()
        if status["virtual_start_time"] is None:
            status["virtual_start_time"] = now
            if 0 <= region < len(self._jitter_last_virtual_age_h):
                self._jitter_last_virtual_age_h[region] = 0.0
            if 0 <= region < len(self._jitter_last_subject):
                self._jitter_last_subject[region] = "virtual"
            self._log_debug(f"[LOG] virtual_run_start region={region} start={now:.1f}")
        else:
            self._log_debug(
                f"[LOG] virtual_run_heartbeat region={region} elapsed={(now - status['virtual_start_time']) / 3600.0:.2f}h"
            )
        status["last_updated_time"] = now

    def _pause_virtual_run(
        self,
        region: int,
        event_type: Optional[str] = None,
        *,
        source: str = "unknown",
    ) -> None:
        status = self._current_status_row(region)
        now = self._now()
        start_time = status.get("virtual_start_time")
        if start_time is not None and event_type is not None:
            runtime_hours = max(0.0, (now - start_time) / 3600.0)
            self._append_historical_event(
                region, runtime_hours, event_type, now, source=source
            )
            if event_type == self.EVENT_FAILURE:
                self._jitter_last_tick = -1
        elif event_type == self.EVENT_FAILURE and event_type is not None:
            # No virtual run active: still record a minimal Failure(Δ)
            delta_hours = max(self._gap_seconds / 3600.0, 1e-6)
            self._append_historical_event(
                region, delta_hours, event_type, now, source=source
            )
            self._jitter_last_tick = -1
        if start_time is not None:
            self._record_availability_segment(region, start_time, now)
            self._log_debug(
                "[LOG] virtual_run_pause region=%s event=%s runtime=%.3fh"
                % (region, event_type or "none", (now - start_time) / 3600.0)
            )
        elif event_type == self.EVENT_FAILURE:
            self._log_debug(
                "[LOG] virtual_run_failure_without_runtime region=%s" % region
            )
        if 0 <= region < len(self._jitter_last_virtual_age_h):
            self._jitter_last_virtual_age_h[region] = None
        if 0 <= region < len(self._jitter_last_subject) and event_type is not None:
            self._jitter_last_subject[region] = "idle"
        status["virtual_start_time"] = None
        status["last_updated_time"] = now

    def _trim_availability_segments(
        self, region: int, window_start: Optional[float] = None
    ) -> None:
        assert 0 <= region < len(self._availability_segments), (
            f"Invalid region {region}, must be in range [0, {len(self._availability_segments)})"
        )
        segments = self._availability_segments[region]
        if window_start is None:
            window_start = max(
                0.0, self.env.elapsed_seconds - self._availability_window_seconds
            )
        while segments and segments[0][1] <= window_start:
            segments.popleft()

    def _record_availability_segment(
        self, region: int, start_time: Optional[float], end_time: float
    ) -> None:
        assert 0 <= region < len(self._availability_segments), (
            f"Invalid region {region}, must be in range [0, {len(self._availability_segments)})"
        )
        assert start_time is not None, "start_time cannot be None"
        start = max(0.0, float(start_time))
        end = max(start, float(end_time))
        assert end > start, f"Invalid time range: start={start}, end={end}"
        segments = self._availability_segments[region]
        if segments and start <= segments[-1][1] + 1e-6:
            last_start, last_end = segments[-1]
            segments[-1] = (last_start, max(last_end, end))
        else:
            segments.append((start, end))
        self._trim_availability_segments(region)

    def _post_step_actions(
        self,
        *,
        decision_outcome: str,
        primary_region: Optional[int],
        primary_type: ClusterType,
        active_instances: Dict[int, ClusterType],
    ) -> typing.Iterable[Action]:
        actions = list(
            super()._post_step_actions(
                decision_outcome=decision_outcome,
                primary_region=primary_region,
                primary_type=primary_type,
                active_instances=active_instances,
            )
        )

        menv = typing.cast("env.MultiTraceEnv", self.env)
        now = menv.tick
        interval = max(1, self.probe_interval_ticks)

        blocked_regions = set(self._last_failed_regions)

        # Schedule new probes for regions that are due.
        for ridx in range(menv.num_regions):
            if ridx == primary_region:
                continue
            if ridx in self._probes_active:
                continue
            if ridx in blocked_regions:
                continue
            if now < self._next_probe_tick[ridx]:
                continue
            actions.append(ProbeLaunch(region=ridx))
            self._next_probe_tick[ridx] = now + interval

        # Clear for next tick once probe scheduling is done.
        self._last_failed_regions.clear()

        # Update jitter regime once per tick (after probe scheduling)
        self._jitter_update_once_per_tick()

        return actions

    def _handle_post_step_action_result(
        self,
        action: Action,
        result: Optional[LaunchResult],
    ) -> None:
        super()._handle_post_step_action_result(action, result)

        if isinstance(action, TryLaunch):
            success = bool(result.success) if result is not None else False
            if action.cluster_type == ClusterType.SPOT:
                if success:
                    if 0 <= action.region < len(self._region_active_since_seconds):
                        self._region_active_since_seconds[action.region] = self._now()
                    if 0 <= action.region < len(self._jitter_last_real_age_h):
                        self._jitter_last_real_age_h[action.region] = 0.0
                    if 0 <= action.region < len(self._jitter_last_virtual_age_h):
                        self._jitter_last_virtual_age_h[action.region] = None
                    # Per new plan: do not create a censor event for virtual; just pause status
                    self._pause_virtual_run(action.region, None)
                    if 0 <= action.region < len(self._jitter_last_subject):
                        self._jitter_last_subject[action.region] = "real"
                    self._last_failed_regions.discard(action.region)
                    self._structured_logger.add_event(
                        action.region, "spot_launch_success"
                    )
                else:
                    if 0 <= action.region < len(self._region_active_since_seconds):
                        self._region_active_since_seconds[action.region] = None
                    # Failed launch implies immediate failure of the virtual run.
                    self._last_failed_regions.add(action.region)
                    self._pause_virtual_run(
                        action.region, self.EVENT_FAILURE, source="spot"
                    )
                    if 0 <= action.region < len(self._jitter_tick_failures):
                        self._jitter_tick_failures[action.region] += 1
                    if 0 <= action.region < len(self._jitter_last_real_age_h):
                        self._jitter_last_real_age_h[action.region] = None
                    if 0 <= action.region < len(self._jitter_last_subject):
                        self._jitter_last_subject[action.region] = "idle"
                    self._structured_logger.add_event(action.region, "spot_launch_fail")
        elif isinstance(action, ProbeLaunch):
            success = bool(result.success) if result is not None else False
            if success:
                self._touch_virtual_run(action.region)
                self._probes_active.add(action.region)
                if 0 <= action.region < len(self._jitter_last_virtual_age_h):
                    self._jitter_last_virtual_age_h[action.region] = 0.0
                if 0 <= action.region < len(self._jitter_last_subject):
                    self._jitter_last_subject[action.region] = "virtual"
                self._structured_logger.add_event(action.region, "probe_success")
            else:
                # Get duration if virtual was running
                duration_h = None
                status = self._current_status_row(action.region)
                if status and status.get("virtual_start_time"):
                    duration_h = (self._now() - status["virtual_start_time"]) / 3600.0

                self._pause_virtual_run(
                    action.region, self.EVENT_FAILURE, source="probe"
                )
                self._probes_active.discard(action.region)
                if 0 <= action.region < len(self._jitter_tick_failures):
                    self._jitter_tick_failures[action.region] += 1
                if 0 <= action.region < len(self._jitter_last_virtual_age_h):
                    self._jitter_last_virtual_age_h[action.region] = None
                if 0 <= action.region < len(self._jitter_last_subject):
                    self._jitter_last_subject[action.region] = "idle"

                if duration_h:
                    self._structured_logger.add_event(
                        action.region, "probe_fail", duration_h=duration_h
                    )
                else:
                    self._structured_logger.add_event(action.region, "probe_fail")
        elif isinstance(action, Terminate):
            # Don't update _probes_active here - termination is only scheduled, not applied yet
            # The actual cleanup will happen in _finalize_active_spot_run_if_ended when instances are gone
            self._log_debug(
                f"Terminate action scheduled for region {action.region} (will clean up _probes_active when actually terminated)"
            )
            if 0 <= action.region < len(self._region_active_since_seconds):
                self._region_active_since_seconds[action.region] = None

    # ------------------------------------------------------------------
    # Overrides from base strategy
    # ------------------------------------------------------------------
    # --- Nelson–Aalen baseline (H0) utilities ---
    def _na_build_knots(self, region: int) -> tuple[list[tuple[float, float]], float]:
        """Build NA baseline from all historical samples in region.
        Returns (knots, tail_h) where knots is list of (tau, step=d/n).
        Per spec: tail_h = max(last_step, ε) where ε≈1e-8.
        """
        epsilon = 1e-8  # Minimum tail hazard

        assert 0 <= region < len(self._historical_logs), (
            f"Invalid region {region}, must be in range [0, {len(self._historical_logs)})"
        )

        samples = [
            (
                float(e.get("runtime_hours", 0.0)),
                1 if e.get("event") == self.EVENT_FAILURE else 0,
            )
            for e in self._historical_logs[region]
        ]
        if not samples:
            return [], epsilon  # No samples -> assume stable

        # Distinct failure times
        failure_times = sorted({rt for rt, ev in samples if ev == 1 and rt >= 0.0})
        if not failure_times:
            return [], epsilon  # No failures observed -> assume stable

        # Pre-compute counts of durations >= tau
        durations = sorted([rt for rt, _ in samples])
        knots: list[tuple[float, float]] = []
        for tau in failure_times:
            d_j = sum(1 for rt, ev in samples if ev == 1 and abs(rt - tau) <= 1e-9)
            n_j = sum(1 for rt in durations if rt + 1e-9 >= tau)
            step = (d_j / max(n_j, 1e-12)) if d_j > 0 else 0.0
            if step > 0:
                knots.append((tau, step))

        # tail_h = max(last_step, ε) as per spec
        if knots:
            tail_h = max(knots[-1][1], epsilon)
        else:
            tail_h = epsilon

        # Log NA baseline construction details (only log periodically to avoid spam)
        if not hasattr(self, "_na_log_counter"):
            self._na_log_counter = {}
        if region not in self._na_log_counter:
            self._na_log_counter[region] = 0
        self._na_log_counter[region] += 1

        # Log every 10th call or when there's interesting data
        if self._na_log_counter[region] % 10 == 1 or len(knots) > 5:
            n_failures = sum(1 for _, ev in samples if ev == 1)
            n_censors = sum(1 for _, ev in samples if ev == 0)
            lines = [
                f"[NA_BASELINE] region={region} samples={len(samples)} (F={n_failures}, C={n_censors})"
            ]
            if knots:
                lines.append(f"  knots: {len(knots)} distinct failure times")
                for i, (tau, step) in enumerate(knots[:3]):
                    lines.append(f"    τ_{i}={tau:.3f}h step={step:.4f}")
                if len(knots) > 3:
                    lines.append(f"    ... ({len(knots) - 3} more)")
                    lines.append(
                        f"    τ_last={knots[-1][0]:.3f}h step={knots[-1][1]:.4f}"
                    )
                lines.append(f"  tail_h={tail_h:.6f}")
            else:
                lines.append(f"  no failures -> tail_h={tail_h:.6f} (epsilon)")

            self._log_debug("\n".join(lines))

        return knots, tail_h

    def _na_H0_value(
        self, knots: list[tuple[float, float]], t: float, tail_h: float = 0.0
    ) -> float:
        """Compute cumulative hazard H0(t) including tail contribution."""
        if t <= 0:
            return 0.0
        if not knots:
            # If no knots, use tail hazard only
            return max(0.0, t * tail_h)

        s = 0.0
        last_tau = 0.0
        for tau, step in knots:
            if tau <= t + 1e-12:
                s += step
                last_tau = tau
            else:
                break

        # Add tail contribution for time beyond last failure point
        if tail_h > 0 and t > last_tau:
            s += tail_h * (t - last_tau)

        return s

    def _baseline_delta_lambda(
        self,
        knots: list[tuple[float, float]],
        tail_h: float,
        start_h: float,
        end_h: float,
    ) -> float:
        """Baseline cumulative hazard increment between two ages."""
        if end_h <= start_h:
            return 0.0
        start_h = max(0.0, float(start_h))
        end_h = max(start_h, float(end_h))
        return max(
            0.0,
            self._na_H0_value(knots, end_h, tail_h)
            - self._na_H0_value(knots, start_h, tail_h),
        )

    def _expected_duration_jitter(
        self, region: int, age_hours: float, *, g_override: Optional[float] = None
    ) -> float:
        """Compute expected remaining lifetime from age using NA baseline scaled by jitter."""
        knots, tail_h = self._na_build_knots(region)
        if g_override is None:
            g = 1.0
            if 0 <= region < len(self._jitter_g_hat):
                g = self._jitter_g_hat[region]
        else:
            g = g_override
        g = max(1e-6, float(g))

        E = 0.0
        S_rel = 1.0
        t = float(max(0.0, age_hours))
        gap_h = self._gap_seconds / 3600.0 if self._gap_seconds else 0.0
        if gap_h <= 0.0:
            gap_h = 1.0

        for tau, step in knots:
            if tau <= t + 1e-12:
                continue
            interval_contrib = S_rel * (tau - t)
            E += interval_contrib
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

    def _jitter_update_once_per_tick(self) -> None:
        menv = typing.cast("env.MultiTraceEnv", self.env)
        tick = menv.tick
        if self._jitter_last_tick == tick:
            return

        self._structured_logger.set_tick(tick, self.env.elapsed_seconds)

        active_instances = menv.get_active_instances()
        now = self._now()
        dt_h = self._gap_seconds / 3600.0 if self._gap_seconds else 0.0
        if dt_h <= 0.0:
            dt_h = 10.0 / 60.0  # default to 10 minutes if gap is unknown

        for r in range(menv.num_regions):
            knots, tail_h = self._na_build_knots(r)
            tail_h = max(tail_h, 1e-8)

            status = self._current_status_row(r)
            virtual_running = (
                status is not None and status.get("virtual_start_time") is not None
            )

            subject = "idle"
            age_now: Optional[float] = None
            age_prev: Optional[float] = None

            is_real = r in active_instances and active_instances[r] == ClusterType.SPOT

            if is_real:
                subject = "real"
                start = None
                if 0 <= r < len(self._region_active_since_seconds):
                    start = self._region_active_since_seconds[r]
                if start is None and getattr(self, "_active_region", None) == r:
                    start = getattr(self, "_active_launch_time_s", None)
                if start is not None:
                    age_now = max(0.0, (now - float(start)) / 3600.0)
                else:
                    age_now = dt_h
                age_prev = self._jitter_last_real_age_h[r]
                if age_prev is None:
                    age_prev = max(age_now - dt_h, 0.0)
                self._jitter_last_real_age_h[r] = age_now
                self._jitter_last_virtual_age_h[r] = None
            elif virtual_running:
                subject = "virtual"
                start = status["virtual_start_time"]
                last = status.get("last_updated_time") or start
                age_now = max(0.0, (float(last) - float(start)) / 3600.0)
                age_prev = self._jitter_last_virtual_age_h[r]
                if age_prev is None:
                    age_prev = max(age_now - dt_h, 0.0)
                self._jitter_last_virtual_age_h[r] = age_now
                self._jitter_last_real_age_h[r] = None
            else:
                self._jitter_last_real_age_h[r] = None
                self._jitter_last_virtual_age_h[r] = None

            delta_lambda = 0.0
            if age_now is not None and age_prev is not None:
                H0_prev = self._na_H0_value(knots, age_prev, tail_h)
                H0_now = self._na_H0_value(knots, age_now, tail_h)
                delta_lambda = max(0.0, H0_now - H0_prev)

            failures = 0
            if 0 <= r < len(self._jitter_tick_failures):
                failures = self._jitter_tick_failures[r]

            if delta_lambda <= 0.0 and (subject != "idle" or failures > 0):
                if not knots:
                    delta_lambda = max(delta_lambda, tail_h * dt_h)
                else:
                    last_tau = knots[-1][0]
                    if (
                        last_tau is not None
                        and age_prev is not None
                        and age_now is not None
                        and age_prev >= last_tau
                    ):
                        delta_lambda = max(
                            delta_lambda,
                            tail_h * max(0.0, age_now - age_prev),
                        )

            event_risk = max(0.0, delta_lambda)
            if failures > 0 and event_risk <= 0.0:
                event_risk = max(event_risk, tail_h * dt_h if tail_h > 0 else 1e-6)

            if event_risk > 0.0 or failures > 0:
                self._burst_append_event(r, risk=event_risk, failures=failures)

            self._jitter_last_subject[r] = subject

            suffix_failures = self._burst_suffix_failures[r]
            suffix_risk = self._burst_suffix_risk[r]
            suffix_len = self._burst_suffix_len[r]

            self._update_logger_history(r)

            # Extract the actual events for logging
            events_for_log = []
            if suffix_len > 0 and 0 <= r < len(self._burst_events):
                recent_events = list(self._burst_events[r])[-suffix_len:]
                for event in recent_events:
                    events_for_log.append(
                        {
                            "risk": event.get("risk", 0.0),
                            "failures": event.get("failures", 0),
                            "tick": event.get("tick", -1),
                        }
                    )

            self._structured_logger.set_jitter_state(
                region=r,
                subject=subject,
                g=self._jitter_g_hat[r],
                g_raw=self._jitter_g_hat_raw[r],
                alpha=0.0,
                beta=0.0,
                n_seg=float(suffix_len),
                f_seg=suffix_risk,
                delta_lambda=event_risk,
                failures=failures,
                mode="burst",
                suffix_failures=suffix_failures,
                suffix_risk=suffix_risk,
                suffix_len=suffix_len,
                events=events_for_log,
            )

            self._jitter_tick_failures[r] = 0

        self._structured_logger.emit_jitter_log()
        self._jitter_last_tick = tick

    def _get_expected_duration_hours(
        self, region_id: int, remaining_time_seconds: float = None
    ) -> float:
        # Ensure jitter state is up to date before querying expectation
        self._jitter_update_once_per_tick()

        # Get knots for logging
        knots, tail_h = self._na_build_knots(region_id)

        # Determine conditional age if region currently has a running SPOT instance
        age_hours = 0.0
        menv = typing.cast("env.MultiTraceEnv", self.env)
        active_instances = menv.get_active_instances()
        if (
            region_id in active_instances
            and active_instances[region_id] == ClusterType.SPOT
        ):
            start = None
            if 0 <= region_id < len(self._region_active_since_seconds):
                start = self._region_active_since_seconds[region_id]
            if start is None and getattr(self, "_active_region", None) == region_id:
                start = getattr(self, "_active_launch_time_s", None)
            if start is not None:
                age_hours = max(0.0, (self._now() - float(start)) / 3600.0)

        # Calculate raw expectation (remaining lifetime)
        raw_pred = self._expected_duration_jitter(region_id, age_hours)

        # Apply caps
        pred = raw_pred
        capped_by_max = False
        clamped_by_deadline = False

        if pred > self.max_prediction_hours:
            pred = self.max_prediction_hours
            capped_by_max = True

        if remaining_time_seconds is None:
            remaining_time_seconds = self.deadline - self.env.elapsed_seconds
        if remaining_time_seconds > 0:
            remaining_h = remaining_time_seconds / 3600.0
            if pred > remaining_h:
                pred = remaining_h
                clamped_by_deadline = True

        g_now = 1.0
        if 0 <= region_id < len(self._jitter_g_hat):
            g_now = self._jitter_g_hat[region_id]

        # Add to structured logger for later batch output
        knots_list = [(t, h) for t, h in knots]
        gap_h = self._gap_seconds / 3600.0 if self._gap_seconds else 0.0
        if gap_h <= 0.0:
            gap_h = 1.0
        self._structured_logger.add_duration_prediction(
            region=region_id,
            knots=knots_list,
            tail=tail_h,
            g=g_now,
            expected=pred,
            mode="jitter",
            gap_hours=gap_h,
        )

        row = self._candidate_debug_row(region_id)
        if row is not None:
            row.update(
                {
                    "mode": "jitter",
                    "predicted_hours": pred,
                    "cap_hours": self.max_prediction_hours,
                }
            )
        return pred

    def _finalize_active_spot_run_if_ended(
        self, active_instances: Dict[int, ClusterType]
    ):
        if self._active_region is not None:
            still_running = (
                self._active_region in active_instances
                and active_instances[self._active_region] == ClusterType.SPOT
            )
            if not still_running:
                elapsed_s = self.env.elapsed_seconds - self._active_launch_time_s
                duration_h = max(0.0, elapsed_s / 3600.0)
                terminated_by_us = getattr(self, "_active_terminated_by_us", False)
                start_time = None
                if 0 <= self._active_region < len(self._region_active_since_seconds):
                    start_time = self._region_active_since_seconds[self._active_region]
                if start_time is None:
                    start_time = self._active_launch_time_s
                self._record_availability_segment(
                    self._active_region, start_time, self.env.elapsed_seconds
                )
                if not terminated_by_us:
                    # Record the actual failure event with full duration
                    self._append_historical_event(
                        self._active_region,
                        duration_h,
                        self.EVENT_FAILURE,
                        source="spot",
                    )
                    # Additionally record a standardized 1-tick preemption failure event
                    # menv = typing.cast("env.MultiTraceEnv", self.env)
                    # tick_duration_hours = menv.gap_seconds / 3600.0
                    # self._append_historical_event(
                    #     self._active_region,
                    #     tick_duration_hours,
                    #     self.EVENT_FAILURE,
                    #     source="preemption",
                    # )
                    if 0 <= self._active_region < len(self._jitter_tick_failures):
                        self._jitter_tick_failures[self._active_region] += 1
                else:
                    self._append_historical_event(
                        self._active_region,
                        duration_h,
                        self.EVENT_CENSOR,
                        source="spot",
                    )
                self._pause_virtual_run(self._active_region)
                if 0 <= self._active_region < len(self._region_active_since_seconds):
                    self._region_active_since_seconds[self._active_region] = None
                if 0 <= self._active_region < len(self._jitter_last_real_age_h):
                    self._jitter_last_real_age_h[self._active_region] = None
                if 0 <= self._active_region < len(self._jitter_last_subject):
                    self._jitter_last_subject[self._active_region] = "idle"

        # Clean up terminated probe instances from _probes_active
        terminated_probes = []
        for probe_region in self._probes_active:
            still_running = (
                probe_region in active_instances
                and active_instances[probe_region] == ClusterType.SPOT
            )
            if not still_running:
                terminated_probes.append(probe_region)
                self._log_debug(
                    f"Probe in region {probe_region} terminated, removing from _probes_active"
                )

        for probe_region in terminated_probes:
            self._probes_active.discard(probe_region)
            if 0 <= probe_region < len(self._jitter_last_virtual_age_h):
                self._jitter_last_virtual_age_h[probe_region] = None
            if 0 <= probe_region < len(self._jitter_last_subject):
                self._jitter_last_subject[probe_region] = "idle"

        super()._finalize_active_spot_run_if_ended(active_instances)

    def _compute_time_value(self) -> float:
        """
        Override parent's time value computation to use dynamic reference cost.
        Instead of fixed C_OD, estimate C_ref based on available resources.
        """
        D = self.task_duration  # Total task (seconds)
        T = self.deadline  # Deadline (seconds)
        p = sum(self.task_done_time)  # Progress (seconds)
        t = self.env.elapsed_seconds  # Elapsed time (seconds)

        # Debug print on first call
        if not hasattr(self, "_first_call_done"):
            self._first_call_done = True
            self._log_debug(
                f"_compute_time_value called: duration={D / 3600:.1f}h deadline={T / 3600:.1f}h"
            )

        remaining_task = D - p
        remaining_time = T - t

        if remaining_time <= 0:
            return float("inf")  # Past deadline

        if remaining_task <= 0:
            return 0  # Task done

        # Estimate dynamic reference cost
        c_ref = self._estimate_reference_cost(remaining_task, remaining_time)

        if not hasattr(self, "_logged_ticks"):
            self._logged_ticks = set()
        tick = int(t / self._gap_seconds) if self._gap_seconds else 0
        self._logged_ticks.add(tick)

        return c_ref * 1.2

    def _compute_od_weight_from_geometry(self) -> float:
        """
        Compute the on-demand weight using a geometric interpretation.

        Geometry:
        - Rectangle: (T, D) compares deadline and task duration.
        - Safety net line: y = x + (D + d - T), where d = 2 * restart_overhead (a more conservative shift).
        - Current trajectory: y = (p / t) x, the ray from the origin through the current point.
        - The ratio of the intersection-to-deadline distance to the remaining deadline gives the weight.

        Weight = |T - x_intersect| / T
        where x_intersect = t * (D + d - T) / (p - t)
        """
        D = self.task_duration  # Total task (seconds)
        T = self.deadline  # Deadline (seconds)
        p = sum(self.task_done_time)  # Accumulated progress (seconds)
        t = self.env.elapsed_seconds  # Elapsed time (seconds)
        d = (
            2 * self.restart_overhead
        )  # Double restart overhead widens the safety margin.

        # Special case handling.
        if t <= 0:
            return 0.0
        if abs(p - t) < 1e-6:  # Avoid divide-by-zero when parallel to the safety net.
            return 0.5  # Neutral weight.

        # Use a sliding-window progress rate rather than the global rate.
        recent_progress_rate = self._estimate_recent_progress_rate()

        # Compute the intersection:
        # Current trajectory: y = recent_rate * (x - t) + p (extended from the current point)
        # Safety net: y = x + (D + d - T) with doubled restart overhead margin.
        # Intersection solves recent_rate * (x - t) + p = x + (D + d - T)
        # => x * (recent_rate - 1) = recent_rate * t - p + (D + d - T)
        # => x = (recent_rate * t - p + (D + d - T)) / (recent_rate - 1)

        if abs(recent_progress_rate - 1.0) < 1e-6:
            # Trajectory nearly parallel to safety net; treat as neutral.
            return 0

        x_intersect = (recent_progress_rate * t - p + (D + d - T)) / (
            recent_progress_rate - 1.0
        )

        # Special cases: intersection outside the deadline window.
        if x_intersect >= T:
            # Intersection after the deadline => aggressively stay on spot.
            od_weight = 0.0
        elif x_intersect <= 0:
            # Intersection before start => already below safety net; go conservative.
            od_weight = 1.0
        else:
            # Intersection within deadline => evaluate ratio normally.
            distance_to_deadline = (
                T - x_intersect
            )  # Intersection occurs before deadline.
            remaining_time = T - t  # Remaining time.
            od_weight = (
                distance_to_deadline / remaining_time
            )  # Fraction of remaining time.
            # Ensure the result stays within [0, 1].
            od_weight = max(0.0, min(1.0, od_weight))

        # Logging.
        if hasattr(self, "_logged_ticks"):
            tick = int(t / self._gap_seconds)
            if tick in self._logged_ticks:
                global_rate = (p / t) if t > 0 else 0.0
                lines = [
                    f"[OD_WEIGHT] tick={tick}",
                    f"  elapsed={t / 3600:.1f}h progress={p / 3600:.1f}h",
                    (
                        f"  recent_rate={recent_progress_rate:.3f} "
                        f"global_rate={global_rate:.3f}"
                    ),
                    f"  intersect_time={x_intersect / 3600:.1f}h",
                ]
                if x_intersect >= T:
                    lines.append(
                        f"  regime=aggressive reason=intersect_after_deadline({T / 3600:.1f}h)"
                    )
                elif x_intersect <= 0:
                    lines.append("  regime=conservative reason=intersect_before_start")
                else:
                    distance = T - x_intersect
                    remaining_window = T - t
                    lines.extend(
                        [
                            f"  distance_to_deadline={distance / 3600:.1f}h",
                            f"  remaining_time={remaining_window / 3600:.1f}h",
                            f"  od_weight={od_weight:.3f}",
                        ]
                    )

                self._log_debug("\n".join(lines))

        return od_weight

    def _estimate_recent_progress_rate(self) -> float:
        """
        Estimate the recent progress rate (work time / wall time) using a configurable
        sliding window.

        Returns:
            float: Recent progress rate. 1.0 means real-time progress, >1 faster, <1 slower.
        """
        window_seconds = self.progress_window_hours * 3600.0

        if (
            self.env.elapsed_seconds < 600
        ):  # Less than 10 minutes of data; use global rate.
            t = self.env.elapsed_seconds
            p = sum(self.task_done_time)
            return (p / t) if t > 0 else 0.0

        # Compute progress inside the window.
        current_time = self.env.elapsed_seconds
        window_start_time = max(0, current_time - window_seconds)

        # Track cumulative progress at window start.
        progress_at_window_start = 0.0
        cumulative_progress = 0.0

        # task_done_time stores work per tick; accumulate until the window start.
        tick_duration = self._gap_seconds
        window_start_tick = int(window_start_time / tick_duration)

        for i, tick_work in enumerate(self.task_done_time):
            if i < window_start_tick:
                progress_at_window_start += tick_work
            cumulative_progress += tick_work

        # Work and wall time inside the window.
        work_in_window = cumulative_progress - progress_at_window_start
        time_in_window = current_time - window_start_time

        # Compute window rate.
        if time_in_window > 0:
            recent_rate = work_in_window / time_in_window
        else:
            recent_rate = 0.0

        # Clamp to a reasonable range (should not exceed 1.0 or drop below 0).
        recent_rate = max(0.0, min(1.0, recent_rate))

        tick = int(current_time / self._gap_seconds)
        total_progress = sum(self.task_done_time)
        global_rate = (total_progress / current_time) if current_time > 0 else 0.0

        delta = abs(global_rate - recent_rate)
        lines = [
            f"[SLIDING_WINDOW] tick={tick}",
            f"  total_time={current_time / 3600:.1f}h window_start={window_start_time / 3600:.1f}h",
            f"  window_duration={time_in_window / 3600:.1f}h target={self.progress_window_hours:.1f}h",
            f"  progress_at_start={progress_at_window_start / 3600:.1f}h",
            f"  cumulative_progress={cumulative_progress / 3600:.1f}h",
            f"  window_work={work_in_window / 3600:.1f}h",
            f"  recent_rate={recent_rate:.3f} global_rate={global_rate:.3f}",
            f"  rate_delta={delta:.3f}",
        ]
        if delta > 0.1:
            lines.append("  note=window_rate_differs_significantly")
        self._log_debug("\n".join(lines))

        return recent_rate

    def _estimate_reference_cost(
        self, remaining_task_seconds: float, remaining_time_seconds: float
    ) -> float:
        """
        Estimate the reference cost based on availability and expected durations.
        Now includes dynamic OD weighting based on geometric position.

        Algorithm:
        1. Sort regions by spot price (cheapest first)
        2. For each region, calculate expected contribution to task completion
        3. Accumulate contributions until we have enough to complete the task
        4. Weighted average of costs based on contribution percentages
        5. Mix with OD price based on geometric weight
        """
        env_t = typing.cast("env.MultiTraceEnv", self.env)
        num_regions = env_t.num_regions

        # Get minimum on-demand price across all regions
        od_prices = [
            env_t.envs[r].get_price()[ClusterType.ON_DEMAND] for r in range(num_regions)
        ]
        c_od = min(od_prices)

        # Collect region information
        region_info = []
        for r in range(num_regions):
            # Get spot price
            spot_price = env_t.envs[r].get_price()[ClusterType.SPOT]

            # Estimate availability (use probe results if available)
            availability = self._estimate_region_availability(r)

            # Get expected duration from risk model (already capped by remaining time)
            expected_duration_hours = self._get_expected_duration_hours(
                r, remaining_time_seconds
            )

            # Calculate effective progress rate (considering restart overhead)
            restart_overhead_hours = self.restart_overhead / 3600.0
            if expected_duration_hours > restart_overhead_hours:
                effective_rate = (
                    expected_duration_hours - restart_overhead_hours
                ) / expected_duration_hours
            else:
                effective_rate = 0.0

            # Expected contribution in seconds
            # = availability * remaining_time * effective_rate
            expected_contribution = (
                availability * remaining_time_seconds * effective_rate
            )

            if expected_contribution > 0:
                effective_duration_hours = max(
                    expected_duration_hours - restart_overhead_hours, 1e-6
                )
                adjusted_price = spot_price
                if effective_duration_hours > 0:
                    adjusted_price = (
                        spot_price * expected_duration_hours / effective_duration_hours
                    )
                adjusted_price += self._transfer_cost_per_hour(r)

                region_info.append(
                    {
                        "region": r,
                        "price": spot_price,
                        "adjusted_price": adjusted_price,
                        "availability": availability,
                        "duration_hours": expected_duration_hours,
                        "effective_rate": effective_rate,
                        "contribution": expected_contribution,
                    }
                )

        # Sort by adjusted price (lowest effective cost first)
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
        contributing_regions: list[dict] = []

        for info in region_info:
            if spot_assigned >= spot_target:
                break
            available = min(info["contribution"], spot_target - spot_assigned)
            if available <= 0:
                continue
            contributing_regions.append(
                {
                    "region": info["region"],
                    "price": info["price"],
                    "adjusted_price": info["adjusted_price"],
                    "availability": info["availability"],
                    "duration_hours": info["duration_hours"],
                    "contribution": available,
                }
            )
            spot_cost += (available / 3600.0) * info["adjusted_price"]
            spot_assigned += available

        deficit = 0.0
        if spot_assigned < spot_target:
            deficit = spot_target - spot_assigned
            od_total_assigned += deficit
            total_cost += (deficit / 3600.0) * c_od
        total_cost += spot_cost

        c_ref = total_cost / max(remaining_task / 3600.0, 1e-6)

        egress_per_hour = 0.0
        assert hasattr(self.task, "checkpoint_size_gb"), (
            f"task missing checkpoint_size_gb: {self.task}"
        )
        checkpoint_size = self.task.checkpoint_size_gb
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

        self._last_c_ref = c_ref

        if hasattr(self, "_logged_ticks") and len(self._logged_ticks) > 0:
            last_tick = max(self._logged_ticks)
            if True:
                if not hasattr(self, "_logged_c_ref_ticks"):
                    self._logged_c_ref_ticks = set()
                if last_tick not in self._logged_c_ref_ticks:
                    self._logged_c_ref_ticks.add(last_tick)
                    lines = [
                        f"[C_REF] tick={last_tick}",
                        (
                            f"  task_remaining={remaining_task_seconds / 3600:.1f}h "
                            f"time_remaining={remaining_time_seconds / 3600:.1f}h"
                        ),
                        f"  contributing_regions={len(contributing_regions)} (spot)",
                    ]
                    for idx, info in enumerate(contributing_regions[:3]):
                        contribution_pct = (
                            info["contribution"] / remaining_task_seconds * 100
                            if remaining_task_seconds > 0
                            else 0.0
                        )
                        lines.append(
                            f"  region_rank={idx + 1} R{info['region']}: "
                            f"price=${info['price']:.3f}/h "
                            f"availability={info['availability']:.1%} "
                            f"duration={info['duration_hours']:.1f}h "
                            f"contribution={contribution_pct:.1f}%"
                        )

                    spot_assigned_hours = sum(
                        info["contribution"] for info in contributing_regions
                    )
                    spot_cost_only = 0.0
                    if spot_assigned_hours > 0:
                        spot_cost_only = sum(
                            (info["contribution"] / spot_assigned_hours)
                            * info["adjusted_price"]
                            for info in contributing_regions
                        )

                    spot_fraction = spot_assigned / max(remaining_task, 1e-6)
                    od_fraction_actual = od_total_assigned / max(remaining_task, 1e-6)
                    total_fraction = max(spot_fraction + od_fraction_actual, 1e-6)
                    spot_fraction /= total_fraction
                    od_fraction_actual /= total_fraction
                    lines.extend(
                        [
                            "  cost_breakdown:",
                            f"    on_demand=${c_od:.3f}/h weight={od_fraction_actual:.1%}",
                            (
                                f"    spot_weighted=${spot_cost_only:.3f}/h "
                                f"weight={spot_fraction:.1%}"
                            ),
                            f"    contributing_counts=spot:{len(contributing_regions)} + od:{1 if od_total_assigned > 0 else 0}",
                            f"    c_ref_pre_egress=${c_ref - egress_per_hour:.3f}/h",
                        ]
                    )
                    if egress_per_hour > 0:
                        lines.append(
                            f"    egress=${egress_per_hour:.3f}/h "
                            f"switches={max(0, (len(contributing_regions) + (1 if od_total_assigned > 0 else 0)) - 1)} "
                            f"size={checkpoint_size:.0f}GB"
                        )
                    lines.append(
                        f"  final_c_ref=${c_ref:.3f}/h ratio_to_od={c_ref / c_od:.2f}x"
                    )
                    self._log_debug("\n".join(lines))

        return c_ref

    def _estimate_region_availability(self, region: int) -> float:
        """
        Estimate availability for a region based on historical alive segments within a
        rolling 12-hour window (includes probe-derived virtual runs).
        """
        assert 0 <= region < len(self._availability_segments), (
            f"Invalid region {region}, must be in range [0, {len(self._availability_segments)})"
        )

        now = float(self.env.elapsed_seconds)
        assert now >= 0.0, f"Invalid elapsed time {now}, must be positive"

        window = self._availability_window_seconds
        window_start = max(0.0, now - window)

        segments = self._availability_segments[region]
        self._trim_availability_segments(region, window_start)

        available_seconds = 0.0
        for start, end in segments:
            seg_start = max(start, window_start)
            seg_end = min(end, now)
            if seg_end > seg_start:
                available_seconds += seg_end - seg_start

        if 0 <= region < len(self._region_active_since_seconds):
            active_start = self._region_active_since_seconds[region]
        else:
            active_start = None
        if active_start is not None:
            seg_start = max(active_start, window_start)
            seg_end = now
            if seg_end > seg_start:
                available_seconds += seg_end - seg_start

        status = self._current_status_row(region)
        virtual_start = status.get("virtual_start_time") if status is not None else None
        if virtual_start is not None:
            seg_start = max(virtual_start, window_start)
            seg_end = now
            if seg_end > seg_start:
                available_seconds += seg_end - seg_start

        denominator = min(window, now)
        if denominator <= 0.0:
            return 0.0

        has_history = (
            available_seconds > 0.0
            or active_start is not None
            or virtual_start is not None
        )
        if not has_history:
            return 0.0

        availability = available_seconds / denominator
        debug_segments = [
            (
                max(start, window_start),
                min(end, now),
            )
            for start, end in segments
            if end > window_start
        ]
        active_display = f"{active_start:.0f}" if active_start is not None else "None"
        virtual_age = (now - virtual_start) if virtual_start is not None else 0.0
        message = (
            "[AVAIL] region={region} window={window:.1f}h available={avail:.3f} "
            "available_s={avail_s:.0f} active_start={active} virtual_age={virt:.0f}s "
            "segments={segments}"
        ).format(
            region=region,
            window=denominator / 3600.0,
            avail=availability,
            avail_s=available_seconds,
            active=active_display,
            virt=virtual_age,
            segments=debug_segments,
        )
        self._log_debug(message)
        return max(0.0, min(1.0, availability))

    def _transfer_cost_per_hour(self, region: int) -> float:
        # This variant prices transfer through the env's egress accounting
        # rather than per-region here.
        return 0.0

    # ------------------------------------------------------------------
    # Structured decision logging
    # ------------------------------------------------------------------
    def _on_emit_candidate_summary(
        self,
        rows: dict[int, dict[str, typing.Any]],
        context: dict[str, typing.Any],
    ) -> None:
        option_details_list = context.get("option_details") or []
        rows = rows or {}
        if not rows and not option_details_list:
            return

        tick = context.get("tick")
        V = context.get("V")
        outcome = context.get("outcome")
        selected_region = context.get("selected_region")
        selected_type = context.get("selected_type")
        current_region = context.get("current_region")
        current_type = context.get("current_type")

        # Set tick context for structured logger
        menv = typing.cast("env.MultiTraceEnv", self.env)
        self._structured_logger.set_tick(tick or menv.tick, self.env.elapsed_seconds)

        # Collect decision options for structured logging
        self._collect_decision_options_for_logger(
            rows,
            option_details_list,
            current_region,
            current_type,
        )

        # Set decision outcome
        if selected_type and selected_region is not None:
            action_str = f"{selected_type.name if hasattr(selected_type, 'name') else selected_type}@R{selected_region}"
        else:
            action_str = outcome or "wait"

        self._structured_logger.set_decision(action_str, V or 0)

        # Emit all structured logs for this decision
        self._structured_logger.emit_all()
