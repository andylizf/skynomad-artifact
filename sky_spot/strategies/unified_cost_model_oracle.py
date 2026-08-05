"""Oracle variant of Unified Cost Model - true SPOT duration + rate-ratio V."""

import typing
from typing import Optional

from sky_spot.strategies.unified_cost_model import UnifiedCostModelStrategy
from sky_spot.utils import ClusterType
from sky_spot import env as env_lib
from sky_spot.multi_region_types import Action


class UnifiedCostModelOracleDuration(UnifiedCostModelStrategy):
    """
    Multi-region UCM variant with an oracle for the next SPOT run duration,
    scoring with the rate-ratio progress value.

    Differences from the base UCM:
    - _get_expected_duration_hours() returns the exact contiguous availability
      (in hours) for a region starting at the current tick if SPOT is available now;
      otherwise returns 0, effectively filtering out currently unavailable regions.
    - _post_step_actions() returns empty - no probing needed with oracle duration.

    This upper-bounds the benefit of perfect duration estimation at launch time,
      without planning future waits.
    - _compute_time_value() uses the rate-ratio form, V = c_od * (remaining work /
      remaining time) / (P / T), so this arm differs from
      unified_cost_model_rate_ratio in the lifetime input alone. Scoring it with
      the base class's reference-cost V instead moves the V100 result by 13%.

    Everything else (OD margin, migration penalty, VOI gating) is the base UCM's.
    """

    NAME = 'unified_cost_model_oracle'

    @classmethod
    def _from_args(cls, parser, namespace=None):
        group = parser.add_argument_group("UnifiedCostModelOracleDuration")
        group.add_argument(
            "--oracle-v",
            choices=("reference_cost", "rate_ratio"),
            default="reference_cost",
            help="Progress value this arm scores with: 'reference_cost' is the "
                 "base class's, 'rate_ratio' is c_od * T/(T-t) * (P-p)/P. Each "
                 "config sets the one its own panel was measured with.",
        )
        return super()._from_args(parser, namespace)

    def _compute_time_value(self) -> float:
        """V = c_od * required_rate / planned_rate, or the base class's form.

        Which one is set by --oracle-v; see _from_args for why it varies.
        """
        if getattr(self.args, "oracle_v", "reference_cost") != "rate_ratio":
            return super()._compute_time_value()

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

        env_t = typing.cast("env_lib.MultiTraceEnv", self.env)
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
        # Floor at the cheapest spot price: the job has to finish, so the policy
        # is never willing to pay less than that even when far ahead of schedule.
        return max(c_od * ratio, c_spot_min)

    def _get_expected_duration_hours(
        self, region_id: int, remaining_time_seconds: Optional[float] = None
    ) -> float:
        """Return exact contiguous availability from current tick."""
        env = typing.cast('env_lib.MultiTraceEnv', self.env)
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
        """No probing needed for oracle - we have perfect information."""
        return ()
