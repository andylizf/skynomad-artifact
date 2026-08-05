"""
UCM variants with different V (time value) computation methods.

Variant 1: Rate Ratio
    V = (remaining_work/remaining_time) / (all_work/all_time)
    = urgency ratio: how much faster we need to work vs planned rate

Variant 2: Rate / Availability
    V = (remaining_work/remaining_time) / avg_spot_availability
    = required rate normalized by available spot capacity
"""

import typing
from typing import Optional

from sky_spot.strategies.unified_cost_model import UnifiedCostModelStrategy
from sky_spot.strategies import ucm_decision_logger as dlog
from sky_spot import env as env_lib
from sky_spot.multi_region_types import Action


# Enable logging for V variants (disabled by default - set UCM_DECISION_LOG=1 to enable)
# dlog.UCM_LOG_ENABLED = True


class OracleDurationMixin:
    """Mixin to add oracle duration and availability estimation capability."""

    # Set to True to use oracle (true) duration instead of estimated
    USE_ORACLE_DURATION: bool = False
    # Set to True to use oracle (true) future availability instead of estimated
    USE_ORACLE_AVAILABILITY: bool = False
    # Window for oracle availability calculation (in hours)
    ORACLE_AVAILABILITY_WINDOW_HOURS: float = 24.0

    def _estimate_region_availability(self, region: int) -> float:
        """Return availability - oracle or estimated based on USE_ORACLE_AVAILABILITY."""
        if not self.USE_ORACLE_AVAILABILITY:
            # Use parent's estimation
            return super()._estimate_region_availability(region)

        # Oracle: calculate true future availability from current tick
        env = typing.cast("env_lib.MultiTraceEnv", self.env)
        sub_env = env.envs[region]
        sub_env.tick = env.tick

        gap_s = sub_env.gap_seconds
        start = sub_env.tick + sub_env._start_index
        trace = sub_env.trace
        n = len(trace)

        # Calculate availability over the next ORACLE_AVAILABILITY_WINDOW_HOURS
        window_ticks = int(self.ORACLE_AVAILABILITY_WINDOW_HOURS * 3600 / gap_s)
        end_idx = min(start + window_ticks, n)

        if end_idx <= start:
            return 0.0

        available_count = 0
        for i in range(start, end_idx):
            if not trace[i]:  # available == not trace[i]
                available_count += 1

        availability = available_count / (end_idx - start)
        return availability

    def _get_expected_duration_hours(
        self, region_id: int, remaining_time_seconds: Optional[float] = None
    ) -> float:
        """Return duration - oracle or estimated based on USE_ORACLE_DURATION."""
        if not self.USE_ORACLE_DURATION:
            # Use parent's estimation
            return super()._get_expected_duration_hours(
                region_id, remaining_time_seconds
            )

        # Oracle: return exact contiguous availability from current tick
        env = typing.cast("env_lib.MultiTraceEnv", self.env)
        sub_env = env.envs[region_id]
        sub_env.tick = env.tick

        try:
            available_now = sub_env.spot_available()
        except Exception:
            return 0.0

        if not available_now:
            return 0.0

        # Count contiguous availability starting at current tick
        gap_s = sub_env.gap_seconds
        start = sub_env.tick + sub_env._start_index
        trace = sub_env.trace
        n = len(trace)
        count = 0
        i = start
        while i < n and (not trace[i]):  # available == not trace[i]
            count += 1
            i += 1

        duration_hours = (count * gap_s) / 3600.0

        # Apply remaining time cap if provided
        if remaining_time_seconds is None:
            remaining_time_seconds = self.deadline - self.env.elapsed_seconds
        if remaining_time_seconds is not None and remaining_time_seconds > 0:
            remaining_h = remaining_time_seconds / 3600.0
            if duration_hours > remaining_h:
                duration_hours = remaining_h

        return duration_hours

    def _post_step_actions(
        self,
        *,
        decision_outcome: str = "",
        primary_region: Optional[int] = None,
        primary_type: typing.Any = None,
        active_instances: typing.Optional[dict] = None,
    ) -> typing.Iterable[Action]:
        """No probing needed when using oracle duration."""
        if self.USE_ORACLE_DURATION:
            return ()
        return super()._post_step_actions(
            decision_outcome=decision_outcome,
            primary_region=primary_region,
            primary_type=primary_type,
            active_instances=active_instances,
        )


class UnifiedCostModelRateRatio(OracleDurationMixin, UnifiedCostModelStrategy):
    """
    V = c_od * (remaining_work/remaining_time) / (all_work/all_time)

    Interpretation:
    - V = c_od: on track
    - V > c_od: behind schedule, need to work faster
    - V < c_od: ahead of schedule

    Lifetime prediction: OracleDurationMixin.USE_ORACLE_DURATION is False in
    the shipped code, so this variant uses the ESTIMATED spot duration, not
    the true one. (An earlier version of this docstring claimed the oracle
    duration was the default; it never was.)
    """

    NAME = "unified_cost_model_rate_ratio"

    def _compute_time_value(self) -> float:
        """V = c_od * required_rate / planned_rate."""
        D = self.task_duration  # total work (seconds)
        T = self.deadline  # total time (seconds)
        p = sum(self.task_done_time)  # work done (seconds)
        t = self.env.elapsed_seconds  # time elapsed (seconds)

        remaining_work = D - p
        remaining_time = T - t

        if remaining_time <= 0:
            return float("inf")
        if remaining_work <= 0:
            return 0.0

        # Get on-demand and spot prices
        env_t = typing.cast("env_lib.MultiTraceEnv", self.env)
        from sky_spot.utils import ClusterType

        od_prices = [
            env_t.envs[r].get_price()[ClusterType.ON_DEMAND]
            for r in range(env_t.num_regions)
        ]
        spot_prices = [
            env_t.envs[r].get_price()[ClusterType.SPOT]
            for r in range(env_t.num_regions)
        ]
        c_od = min(od_prices)
        c_spot_min = min(spot_prices)

        ratio = (remaining_work * T) / (remaining_time * D)
        V = c_od * ratio

        # V floor: at minimum, we must be willing to pay the cheapest SPOT price
        V = max(V, c_spot_min)
        return V

    def _log_time_value(self, V: float) -> None:
        """Log rate_ratio V computation."""
        D = self.task_duration
        T = self.deadline
        p = sum(self.task_done_time)
        t = self.env.elapsed_seconds
        remaining_work = D - p
        remaining_time = T - t

        env_t = typing.cast("env_lib.MultiTraceEnv", self.env)
        from sky_spot.utils import ClusterType

        od_prices = [
            env_t.envs[r].get_price()[ClusterType.ON_DEMAND]
            for r in range(env_t.num_regions)
        ]
        c_od = min(od_prices)

        dlog.log("\n>>> TIME VALUE:")
        dlog.log(
            "# V = c_od × (remaining_work/remaining_time) / (task_duration/deadline)"
        )
        dlog.log(
            f"    remaining: work={remaining_work / 3600:.1f}h, time={remaining_time / 3600:.1f}h"
        )
        dlog.log(f"    planned: work={D / 3600:.1f}h, time={T / 3600:.1f}h")
        dlog.log(f"    c_od: ${c_od:.2f}/h")
        req_rate = remaining_work / remaining_time if remaining_time > 0 else 0
        plan_rate = D / T if T > 0 else 0
        ratio = req_rate / plan_rate if plan_rate > 0 else 0
        dlog.log(f"    V = {c_od:.2f} × {ratio:.3f} = ${V:.2f}/h")


class UnifiedCostModelRatePast(OracleDurationMixin, UnifiedCostModelStrategy):
    """
    V = c_od * (remaining_work/remaining_time) / (work_done/time_elapsed)
      = c_od * required_rate / past_rate

    Interpretation:
    - V = c_od: past rate matches required rate (on track)
    - V > c_od: past rate < required rate (behind schedule, need to speed up)
    - V < c_od: past rate > required rate (ahead of schedule)

    Key difference from RateRatio:
    - RateRatio uses planned_rate = D/T (constant)
    - RatePast uses past_rate = p/t (adapts to actual progress)

    Lifetime prediction: OracleDurationMixin.USE_ORACLE_DURATION is False in
    the shipped code, so this variant uses the ESTIMATED spot duration, not
    the true one. (An earlier version of this docstring claimed the oracle
    duration was the default; it never was.)
    """

    NAME = "unified_cost_model_rate_past"

    def _compute_time_value(self) -> float:
        """V = c_od * required_rate / past_rate."""
        D = self.task_duration  # total work (seconds)
        p = sum(self.task_done_time)  # work done (seconds)
        t = self.env.elapsed_seconds  # time elapsed (seconds)

        remaining_work = D - p
        remaining_time = self.deadline - t

        if remaining_time <= 0:
            return float("inf")
        if remaining_work <= 0:
            return 0.0

        required_rate = remaining_work / remaining_time
        past_rate = p / t if t > 0 else 0.0

        # Early stage: no past rate yet, fall back to planned rate
        if past_rate <= 0:
            past_rate = D / self.deadline

        # Get on-demand and spot prices
        env_t = typing.cast("env_lib.MultiTraceEnv", self.env)
        from sky_spot.utils import ClusterType

        od_prices = [
            env_t.envs[r].get_price()[ClusterType.ON_DEMAND]
            for r in range(env_t.num_regions)
        ]
        spot_prices = [
            env_t.envs[r].get_price()[ClusterType.SPOT]
            for r in range(env_t.num_regions)
        ]
        c_od = min(od_prices)
        c_spot_min = min(spot_prices)

        ratio = required_rate / past_rate
        V = c_od * ratio

        # V floor: at minimum, we must be willing to pay the cheapest SPOT price
        # because the task must be completed regardless of how far ahead we are
        V = max(V, c_spot_min)
        return V

    def _log_time_value(self, V: float) -> None:
        """Log rate_past V computation."""
        D = self.task_duration
        p = sum(self.task_done_time)
        t = self.env.elapsed_seconds
        remaining_work = D - p
        remaining_time = self.deadline - t

        env_t = typing.cast("env_lib.MultiTraceEnv", self.env)
        from sky_spot.utils import ClusterType

        od_prices = [
            env_t.envs[r].get_price()[ClusterType.ON_DEMAND]
            for r in range(env_t.num_regions)
        ]
        c_od = min(od_prices)

        req_rate = remaining_work / remaining_time if remaining_time > 0 else 0
        past_rate = p / t if t > 0 else D / self.deadline

        dlog.log("\n>>> TIME VALUE:")
        dlog.log(
            "# V = c_od × (remaining_work/remaining_time) / (work_done/time_elapsed)"
        )
        dlog.log(
            f"    remaining: work={remaining_work / 3600:.1f}h, time={remaining_time / 3600:.1f}h"
        )
        dlog.log(f"    past: work={p / 3600:.1f}h, time={t / 3600:.1f}h")
        dlog.log(f"    required_rate={req_rate:.4f}, past_rate={past_rate:.4f}")
        dlog.log(f"    c_od: ${c_od:.2f}/h")
        ratio = req_rate / past_rate if past_rate > 0 else 0
        dlog.log(f"    V = {c_od:.2f} × {ratio:.3f} = ${V:.2f}/h")


class UnifiedCostModelRateAvail(OracleDurationMixin, UnifiedCostModelStrategy):
    """
    V = c_od × (remaining_work/remaining_time) / avg_spot_availability

    Interpretation:
    - Higher V when spot availability is low (need to value time more)
    - Lower V when spot availability is high (can afford to be picky)

    Lifetime prediction: OracleDurationMixin.USE_ORACLE_DURATION is False in
    the shipped code, so this variant uses the ESTIMATED spot duration, not
    the true one. (An earlier version of this docstring claimed the oracle
    duration was the default; it never was.)
    """

    NAME = "unified_cost_model_rate_avail"

    def _compute_time_value(self) -> float:
        """V = c_od × required_rate / avg_availability."""
        D = self.task_duration
        T = self.deadline
        p = sum(self.task_done_time)
        t = self.env.elapsed_seconds

        remaining_work = D - p
        remaining_time = T - t

        if remaining_time <= 0:
            return float("inf")
        if remaining_work <= 0:
            return 0.0

        required_rate = remaining_work / remaining_time

        # Compute average spot availability across all regions
        env_t = typing.cast("env_lib.MultiTraceEnv", self.env)
        total_avail = 0.0
        count = 0
        for r in range(env_t.num_regions):
            avail = self._estimate_region_availability(r)
            if avail > 0:
                total_avail += avail
                count += 1

        avg_avail = total_avail / count if count > 0 else 1.0  # Assume 100% if no data

        # Get on-demand and spot prices
        from sky_spot.utils import ClusterType

        od_prices = [
            env_t.envs[r].get_price()[ClusterType.ON_DEMAND]
            for r in range(env_t.num_regions)
        ]
        spot_prices = [
            env_t.envs[r].get_price()[ClusterType.SPOT]
            for r in range(env_t.num_regions)
        ]
        c_od = min(od_prices)
        c_spot_min = min(spot_prices)

        # V = c_od × required_rate / avg_availability
        ratio = required_rate / max(avg_avail, 0.01)
        V = c_od * ratio

        # V floor: at minimum, we must be willing to pay the cheapest SPOT price
        V = max(V, c_spot_min)
        return V

    def _log_time_value(self, V: float) -> None:
        """Log rate_avail V computation."""
        D = self.task_duration
        T = self.deadline
        p = sum(self.task_done_time)
        t = self.env.elapsed_seconds
        remaining_work = D - p
        remaining_time = T - t

        env_t = typing.cast("env_lib.MultiTraceEnv", self.env)
        avails = []
        for r in range(env_t.num_regions):
            avail = self._estimate_region_availability(r)
            if avail > 0:
                avails.append(avail)
        avg_avail = sum(avails) / len(avails) if avails else 0.1

        from sky_spot.utils import ClusterType

        od_prices = [
            env_t.envs[r].get_price()[ClusterType.ON_DEMAND]
            for r in range(env_t.num_regions)
        ]
        c_od = min(od_prices)

        req_rate = remaining_work / remaining_time if remaining_time > 0 else 0
        ratio = req_rate / max(avg_avail, 0.01)

        dlog.log("\n>>> TIME VALUE:")
        dlog.log("# V = c_od × (remaining_work/remaining_time) / avg_availability")
        dlog.log(
            f"    remaining: work={remaining_work / 3600:.1f}h, time={remaining_time / 3600:.1f}h"
        )
        dlog.log(f"    required_rate = {req_rate:.3f}")
        dlog.log(f"    avg_availability = {avg_avail:.2%} (from {len(avails)} regions)")
        dlog.log(f"    c_od: ${c_od:.2f}/h")
        dlog.log(f"    V = {c_od:.2f} × {ratio:.3f} = ${V:.2f}/h")


class UnifiedCostModelRampV(OracleDurationMixin, UnifiedCostModelStrategy):
    """
    V = C_spot + (V_req/φ) × (C_od - C_spot)  if V_req < φ  (linear ramp)
    V = C_od                                   if V_req >= φ (capped)

    Where:
    - V_req = remaining_work / remaining_time (required rate)
    - φ = avg_spot_availability

    Interpretation:
    - When ahead of schedule (V_req << φ): V ≈ C_spot (cheap is fine)
    - As V_req approaches φ: V ramps up towards C_od
    - When behind schedule (V_req >= φ): V = C_od (must use on-demand)

    Lifetime prediction: OracleDurationMixin.USE_ORACLE_DURATION is False in
    the shipped code, so this variant uses the ESTIMATED spot duration, not
    the true one. (An earlier version of this docstring claimed the oracle
    duration was the default; it never was.)
    """

    NAME = "unified_cost_model_ramp_v"

    def _compute_time_value(self) -> float:
        """V = piecewise linear ramp from C_spot to C_od."""
        D = self.task_duration
        T = self.deadline
        p = sum(self.task_done_time)
        t = self.env.elapsed_seconds

        remaining_work = D - p
        remaining_time = T - t

        if remaining_time <= 0:
            return float("inf")
        if remaining_work <= 0:
            return 0.0

        V_req = remaining_work / remaining_time

        # Compute average spot availability (φ)
        env_t = typing.cast("env_lib.MultiTraceEnv", self.env)
        total_avail = 0.0
        count = 0
        for r in range(env_t.num_regions):
            avail = self._estimate_region_availability(r)
            if avail > 0:
                total_avail += avail
                count += 1
        phi = total_avail / count if count > 0 else 0.1

        # Get prices
        from sky_spot.utils import ClusterType

        od_prices = [
            env_t.envs[r].get_price()[ClusterType.ON_DEMAND]
            for r in range(env_t.num_regions)
        ]
        spot_prices = [
            env_t.envs[r].get_price()[ClusterType.SPOT]
            for r in range(env_t.num_regions)
        ]
        c_od = min(od_prices)
        c_spot = min(spot_prices)

        # Piecewise linear V
        if V_req >= phi:
            # Capped at on-demand price
            V = c_od + 0.1
        else:
            # Linear ramp: C_spot + (V_req/φ) × (C_od - C_spot)
            V = c_spot + (V_req / phi) * (c_od - c_spot) + 0.1

        return V

    def _log_time_value(self, V: float) -> None:
        """Log ramp_v computation."""
        D = self.task_duration
        T = self.deadline
        p = sum(self.task_done_time)
        t = self.env.elapsed_seconds
        remaining_work = D - p
        remaining_time = T - t

        V_req = remaining_work / remaining_time if remaining_time > 0 else 0

        env_t = typing.cast("env_lib.MultiTraceEnv", self.env)
        avails = []
        for r in range(env_t.num_regions):
            avail = self._estimate_region_availability(r)
            if avail > 0:
                avails.append(avail)
        phi = sum(avails) / len(avails) if avails else 0.1

        from sky_spot.utils import ClusterType

        od_prices = [
            env_t.envs[r].get_price()[ClusterType.ON_DEMAND]
            for r in range(env_t.num_regions)
        ]
        spot_prices = [
            env_t.envs[r].get_price()[ClusterType.SPOT]
            for r in range(env_t.num_regions)
        ]
        c_od = min(od_prices)
        c_spot = min(spot_prices)

        dlog.log("\n>>> TIME VALUE:")
        dlog.log("# V = C_spot + (V_req/φ)×(C_od - C_spot) if V_req < φ, else C_od")
        dlog.log(
            f"    remaining: work={remaining_work / 3600:.1f}h, time={remaining_time / 3600:.1f}h"
        )
        dlog.log(f"    V_req = {V_req:.3f}")
        dlog.log(f"    φ (avg_avail) = {phi:.2%} (from {len(avails)} regions)")
        dlog.log(f"    C_spot=${c_spot:.2f}/h, C_od=${c_od:.2f}/h")
        if V_req >= phi:
            dlog.log(f"    V_req >= φ → V = C_od = ${V:.2f}/h (capped)")
        else:
            ratio = V_req / phi if phi > 0 else 0
            dlog.log(
                f"    V = {c_spot:.2f} + {ratio:.2f}×({c_od:.2f}-{c_spot:.2f}) = ${V:.2f}/h"
            )
