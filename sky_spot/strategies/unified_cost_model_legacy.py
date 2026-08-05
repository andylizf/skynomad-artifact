"""
Pure Unified Cost Model Strategy

A truly pure implementation of the unified cost framework based on time value theory.
No heuristics, no thresholds, just pure cost-based decisions.
"""

import logging
import typing
import math
import random
import collections
from argparse import BooleanOptionalAction

from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot import env, task
from sky_spot.multi_region_types import (
    TryLaunch,
    Terminate,
    Action,
    LaunchResult,
    ClusterType,
)
from sky_spot.migration_model import (
    get_transfer_time_hours,
    get_transfer_cost_usd,
    get_region_relationship,
)

logger = logging.getLogger(__name__)

# Hardcoded region characteristics from offline analysis (by name).
# These serve as anchors; unknown regions will be estimated from traces at reset.
REGION_CHARACTERISTICS_BY_NAME: dict[str, dict[str, float]] = {
    # Values computed from aligned traces via scripts_multi/compute_region_stats.py
    "us-east-1a_v100_1": {"avg_availability": 0.1851, "avg_duration": 0.7331},
    "us-east-1c_v100_1": {"avg_availability": 0.4684, "avg_duration": 1.3734},
    "us-east-1d_v100_1": {"avg_availability": 0.4768, "avg_duration": 1.6800},
    "us-east-1f_v100_1": {"avg_availability": 0.6145, "avg_duration": 2.2718},
    "us-east-2a_v100_1": {"avg_availability": 0.7523, "avg_duration": 4.7387},
    "us-east-2b_v100_1": {"avg_availability": 0.6763, "avg_duration": 3.9446},
    "us-west-2a_v100_1": {"avg_availability": 0.9130, "avg_duration": 6.4255},
    "us-west-2b_v100_1": {"avg_availability": 0.9324, "avg_duration": 10.2400},
    "us-west-2c_v100_1": {"avg_availability": 0.9221, "avg_duration": 7.2752},
    "asia-east1-c_h100_8": {"avg_availability": 0.0000, "avg_duration": 0.0000},
    "asia-northeast1-b_h100_8": {"avg_availability": 0.0000, "avg_duration": 0.0000},
    "europe-west1-c_h100_8": {"avg_availability": 0.6574, "avg_duration": 3.9500},
    "europe-west4-c_h100_8": {"avg_availability": 0.0000, "avg_duration": 0.0000},
    "us-central1-a_h100_8": {"avg_availability": 0.9417, "avg_duration": 4.0417},
    "us-central1-b_h100_8": {"avg_availability": 0.7601, "avg_duration": 1.9855},
    "us-east4-a_h100_8": {"avg_availability": 0.6976, "avg_duration": 1.7465},
    "us-west1-a_h100_8": {"avg_availability": 0.6033, "avg_duration": 8.0556},
    "asia-south2-b_h100_16": {"avg_availability": 1.0000, "avg_duration": 59.1111},
    "asia-southeast1-b_h100_16": {"avg_availability": 0.2403, "avg_duration": 1.1313},
    "asia-southeast1-c_h100_16": {"avg_availability": 0.8741, "avg_duration": 7.3810},
    "europe-west1-c_h100_16": {"avg_availability": 0.8994, "avg_duration": 6.8357},
    "europe-west3-c_h100_16": {"avg_availability": 0.1883, "avg_duration": 3.5774},
    "us-central1-a_h100_16": {"avg_availability": 0.9803, "avg_duration": 8.8390},
    "us-central1-b_h100_16": {"avg_availability": 0.9342, "avg_duration": 3.1063},
    "us-central1-c_h100_16": {"avg_availability": 0.9220, "avg_duration": 2.7402},
    "us-east4-a_h100_16": {"avg_availability": 0.9784, "avg_duration": 11.0745},
    "us-east4-b_h100_16": {"avg_availability": 0.9088, "avg_duration": 4.2788},
    "us-east5-a_h100_16": {"avg_availability": 0.8590, "avg_duration": 1.7992},
    "us-west1-a_h100_16": {"avg_availability": 0.3772, "avg_duration": 3.7160},
    "us-west1-b_h100_16": {"avg_availability": 0.1291, "avg_duration": 0.3652},
}


class UnifiedCostModelStrategy(MultiRegionStrategy):
    NAME = "unified_cost_model_legacy"

    def __init__(self, args):
        super().__init__(args)

        # Bandit controls layered on unified cost
        self.bandit_mode: str = args.bandit_mode  # 'none' | 'ucb' | 'thompson'
        self.ucb_c: float = args.ucb_c
        self.max_attempts_per_tick: int = args.max_attempts_per_tick
        self.od_margin: float = args.od_margin
        self.thompson_scale: float = args.thompson_scale
        # Whether to include dynamic migration time + transfer USD in decision making
        self.consider_migration_penalty: bool = not args.ignore_migration_penalty

        # Bandit history handling for non-stationarity
        # 'none': original averaging; 'discount': exponential decay; 'window': fixed-size recent window
        self.bandit_history: str = args.bandit_history  # 'none' | 'discount' | 'window'
        self.discount_half_life_hours: float = args.discount_half_life_hours
        self.window_attempts: int = args.window_attempts
        # Bandit blending and migration/exploration guardrails
        self.bandit_alpha: float = args.bandit_alpha  # blend weight for avg_per_hour
        self.only_explore_when_idle: bool = args.only_explore_when_idle
        self.allow_spot_to_spot_migration: bool = args.allow_spot_to_spot_migration
        # VOI gating: explore only if upper-bound beats current best enough to pay migration
        self.voi_gating: bool = args.voi_gating
        self.voi_delta_base: float = args.voi_delta_base  # $/h
        self.voi_k_checkpoint: float = args.voi_k_checkpoint  # $/h per GB
        self.voi_k_v: float = args.voi_k_v  # scales with V
        # Online duration baseline (optional)
        self.use_online_duration: bool = args.use_online_duration
        self.online_duration_min_samples: int = args.online_duration_min_samples
        self.online_duration_min_hours: float = args.online_duration_min_hours
        self.online_duration_max_hours: float = args.online_duration_max_hours
        self.online_duration_mix: float = (
            args.online_duration_mix
        )  # 0..1 weight on online

        # Online stats for bandit adjustments (per region)
        # Raw counters
        self.region_attempts: list[int] = []  # integer count of attempts
        self.total_attempts: int = 0
        # Effective counters for UCB/Thompson (decayed or windowed)
        self.region_eff_attempts: list[float] = []
        self.total_eff_attempts: float = 0.0
        # Reward bookkeeping for avg_per_hour
        # Mode 'none' & 'discount': maintain aggregates
        self.region_sum_rewards: list[float] = []
        self.region_sum_charge_hours: list[float] = []
        # Mode 'window': maintain deques + rolling sums
        self.region_window_rewards: list[collections.deque] = []
        self.region_window_hours: list[collections.deque] = []
        self.region_window_sum_rewards: list[float] = []
        self.region_window_sum_hours: list[float] = []
        # Online duration statistics
        self.region_duration_sum: list[float] = []
        self.region_duration_count: list[float] = []
        self.region_duration_window: list[collections.deque] = []
        self.region_duration_window_sum: list[float] = []

        # Env cadence for discounting
        self._decay_factor_per_tick: float = 1.0
        self._last_decay_tick: int = -1

        # Track active SPOT run to finalize realized reward
        self._active_region: typing.Optional[int] = None
        self._active_launch_time_s: float = 0.0
        self._active_launch_V_per_hour: float = 0.0
        # Tick-local exploration suppression flag
        self._suppress_exploration: bool = False
        # Track the most recent region we ran in (SPOT or OD) to estimate migration when idle
        self._last_region: typing.Optional[int] = None

        # Candidate debug summary (subclasses may enable)
        self.enable_candidate_debug_summary: bool = False
        self._candidate_log_rows: typing.Optional[dict[int, dict[str, typing.Any]]] = (
            None
        )
        self._candidate_summary_context: typing.Optional[dict[str, typing.Any]] = None

        logger.info(f"Initialized Unified Cost Model strategy")

    # ------------------------------------------------------------------
    # Candidate debug helpers (used by subclasses for structured logging)
    # ------------------------------------------------------------------
    def _begin_candidate_capture(self) -> None:
        if getattr(self, "enable_candidate_debug_summary", False):
            self._candidate_log_rows = {}
            self._candidate_summary_context = None
        else:
            self._candidate_log_rows = None
            self._candidate_summary_context = None

    def _candidate_debug_row(self, region_id: int) -> typing.Optional[dict]:
        if self._candidate_log_rows is None:
            return None
        row = self._candidate_log_rows.setdefault(region_id, {"notes": []})
        row["region"] = region_id
        return row

    def _set_candidate_status(
        self,
        region_id: int,
        status: str,
        **extra: typing.Any,
    ) -> None:
        row = self._candidate_debug_row(region_id)
        if row is None:
            return
        row.setdefault("notes", [])
        row["status"] = status
        for key, value in extra.items():
            row[key] = value

    def _emit_candidate_summary(self, **kwargs: typing.Any) -> None:
        if self._candidate_log_rows is None:
            return
        context: dict[str, typing.Any] = dict(self._candidate_summary_context or {})
        context.update(kwargs)
        rows = self._candidate_log_rows
        self._candidate_log_rows = None
        self._candidate_summary_context = None
        self._on_emit_candidate_summary(rows, context)

    def _on_emit_candidate_summary(
        self,
        rows: dict[int, dict[str, typing.Any]],
        context: dict[str, typing.Any],
    ) -> None:
        # Base implementation: no-op. Subclasses may override.
        return

    # ------------------------------------------------------------------
    # Extension hooks
    # ------------------------------------------------------------------
    def _filter_active_instances(
        self,
        active_instances: dict[int, ClusterType],
    ) -> dict[int, ClusterType]:
        """Allow subclasses to mask auxiliary instances (e.g., probes)."""

        return active_instances

    def _pre_step_actions(
        self,
        active_instances: dict[int, ClusterType],
    ) -> typing.Iterable["Action"]:
        """Hook for subclasses to run actions before main decision making."""

        return ()

    def _before_spot_launch(
        self,
        region_id: int,
    ) -> typing.Iterable["Action"]:
        """Hook for subclasses to run cleanup before launching a SPOT instance."""

        return ()

    def _post_step_actions(
        self,
        *,
        decision_outcome: str,
        primary_region: typing.Optional[int],
        primary_type: ClusterType,
        active_instances: dict[int, ClusterType],
    ) -> typing.Iterable["Action"]:
        """Hook for subclasses to schedule additional actions after core decision.

        Subclasses may return TryLaunch/Terminate/ProbeLaunch actions. The default
        implementation returns an empty iterable.
        """

        return ()

    def _handle_post_step_action_result(
        self,
        action: "Action",
        result: typing.Optional["LaunchResult"],
    ) -> None:
        """Hook invoked after each post-step action completes."""

        return None

    def _run_post_step_actions(
        self,
        *,
        decision_outcome: str,
        primary_region: typing.Optional[int],
        primary_type: ClusterType,
        active_instances: dict[int, ClusterType],
    ) -> typing.Generator["Action", typing.Optional["LaunchResult"], None]:
        for action in self._post_step_actions(
            decision_outcome=decision_outcome,
            primary_region=primary_region,
            primary_type=primary_type,
            active_instances=active_instances,
        ):
            result = yield action
            self._handle_post_step_action_result(action, result)

    def reset(self, env: "env.Env", task: "task.Task"):
        super().reset(env, task)
        self._last_duration = 0
        # Initialize bandit state according to current env regions
        env_t = typing.cast("env.MultiTraceEnv", env)
        num_regions = env_t.num_regions
        # Build per-index characteristics by region name (fail fast if unknown)
        region_names: list[str] = []
        for i in range(num_regions):
            region_names.append(env_t.get_region_name(i))
        missing = [
            name for name in region_names if name not in REGION_CHARACTERISTICS_BY_NAME
        ]
        if missing:
            raise ValueError(f"Missing REGION_CHARACTERISTICS for regions: {missing}")
        self._region_characteristics = [
            REGION_CHARACTERISTICS_BY_NAME[name].copy() for name in region_names
        ]
        self._gap_seconds = env_t.gap_seconds
        if self.bandit_history == "discount" and self.discount_half_life_hours > 0:
            half_life_seconds = self.discount_half_life_hours * 3600.0
            # Per-tick multiplicative decay so that over half-life, factor ≈ 0.5
            self._decay_factor_per_tick = 0.5 ** (
                self._gap_seconds / max(1e-9, half_life_seconds)
            )
        else:
            self._decay_factor_per_tick = 1.0
        self._last_decay_tick = -1
        self.region_attempts = [0 for _ in range(num_regions)]
        self.total_attempts = 0
        self.region_eff_attempts = [0.0 for _ in range(num_regions)]
        self.total_eff_attempts = 0.0
        self.region_sum_rewards = [0.0 for _ in range(num_regions)]
        self.region_sum_charge_hours = [0.0 for _ in range(num_regions)]
        self.region_window_rewards = [
            collections.deque(maxlen=self.window_attempts) for _ in range(num_regions)
        ]
        self.region_window_hours = [
            collections.deque(maxlen=self.window_attempts) for _ in range(num_regions)
        ]
        self.region_window_sum_rewards = [0.0 for _ in range(num_regions)]
        self.region_window_sum_hours = [0.0 for _ in range(num_regions)]
        self.region_duration_sum = [0.0 for _ in range(num_regions)]
        self.region_duration_count = [0.0 for _ in range(num_regions)]
        self.region_duration_window = [
            collections.deque(maxlen=self.window_attempts) for _ in range(num_regions)
        ]
        self.region_duration_window_sum = [0.0 for _ in range(num_regions)]
        self._active_region = None
        self._active_launch_time_s = 0.0
        self._active_launch_V_per_hour = 0.0
        logger.info("Unified Cost Model strategy reset")

    def _compute_time_value(self) -> float:
        """
        Compute the time value V(t,p) based on unified theory.

        V(t,p) = C_OD × (D-p)/(T-t) × T/D

        This represents the marginal value of progress per unit time.
        """
        # return 18  # Commented out to allow actual calculation

        D = self.task_duration  # Total task (seconds)
        T = self.deadline  # Deadline (seconds)
        p = sum(self.task_done_time)  # Progress (seconds)
        t = self.env.elapsed_seconds  # Elapsed time (seconds)

        remaining_task = D - p
        remaining_time = T - t

        assert remaining_time > 0

        if remaining_task <= 0:
            # Task done - no value
            return 0

        # On-demand price is the same for all regions, and wont change over time
        env_t = typing.cast("env.MultiTraceEnv", self.env)
        c_od = env_t.envs[0].get_price()[ClusterType.ON_DEMAND]

        # Core formula: V = C_OD × (D-p)/(T-t) × T/D
        # Simplifies to: V = C_OD × T × (D-p) / (D × (T-t))
        V_per_second = c_od * (T / D) * (remaining_task / remaining_time) / 3600
        V_per_hour = V_per_second * 3600

        return V_per_hour

    def _get_expected_duration_hours(self, region_id: int) -> float:
        """Return expected SPOT duration for region in hours, possibly using online estimate.
        When use_online_duration=False, returns offline avg_duration from REGION_CHARACTERISTICS.
        With online enabled, use decayed/windowed average of realized charge hours, mixed with offline.
        """
        offline = self._region_characteristics[region_id]["avg_duration"]
        if not self.use_online_duration:
            return offline
        # Compute online estimate according to history mode
        if self.bandit_history == "window":
            cnt = len(self.region_duration_window[region_id])
            if cnt == 0:
                return offline
            online = self.region_duration_window_sum[region_id] / max(1, cnt)
            eff_cnt = cnt
        else:
            eff_cnt = int(round(self.region_duration_count[region_id]))
            if self.region_duration_count[region_id] <= 1e-9:
                return offline
            online = self.region_duration_sum[region_id] / max(
                1e-9, self.region_duration_count[region_id]
            )
        # Clamp
        online = max(
            self.online_duration_min_hours, min(self.online_duration_max_hours, online)
        )
        # Mix with offline based on sample count
        w = self.online_duration_mix * min(
            1.0, eff_cnt / max(1, self.online_duration_min_samples)
        )
        return (1.0 - w) * offline + w * online

    def _compute_spot_net_value(
        self,
        region_id: int,
        V: float,
        ignore_restart_overhead: bool = False,
    ) -> float:
        """
        Compute expected net value of SPOT in a region.

        Net value = Expected progress value - Expected cost
        """
        row = self._candidate_debug_row(region_id)
        duration = self._get_expected_duration_hours(region_id)
        if row is not None:
            row["duration_hours"] = duration
            row["time_value_per_hour"] = V
        restart_hours = self.restart_overhead / 3600.0

        if ignore_restart_overhead:
            effective_duration = max(0.0, duration)
        else:
            effective_duration = max(0.0, duration - restart_hours)

        if row is not None:
            row["effective_duration_hours"] = effective_duration
            if ignore_restart_overhead:
                notes = row.setdefault("notes", [])
                if "restart_correction" not in notes:
                    notes.append("restart_correction")

        if effective_duration <= 0:
            if row is not None:
                self._set_candidate_status(region_id, "non_positive_duration")
            return -float("inf")

        # Get current SPOT price dynamically
        env_t = typing.cast("env.MultiTraceEnv", self.env)
        c_spot = env_t.envs[region_id].get_price()[ClusterType.SPOT]
        if row is not None:
            row["spot_price_usd_per_hour"] = c_spot

        # Expected net value over the duration
        # Value gained = V × effective_duration
        # Cost incurred = c_spot × duration
        value_gained = V * effective_duration
        cost_incurred = c_spot * duration
        net_value = value_gained - cost_incurred

        # Normalize to per-hour for comparison
        net_value_per_hour = net_value / duration
        if row is not None:
            row["net_value_per_hour"] = net_value_per_hour
            row["value_gained"] = value_gained
            row["cost_incurred"] = cost_incurred
            row["progress_value_total"] = value_gained
            row["spot_cost_total"] = cost_incurred
            row.setdefault("notes", [])
        else:
            logger.debug(
                f"[UCM] region={region_id} net_value_per_hour={net_value_per_hour:.3f} "
                f"(V={V:.3f}, eff_dur={effective_duration:.3f}, spot_cost={c_spot}, dur={duration:.3f})"
            )

        return net_value_per_hour

    def _compute_od_net_value(self, V: float, region_id: int) -> float:
        """
        Compute net value of ON_DEMAND.

        For ON_DEMAND: net value = V - C_OD
        """
        # On-demand price is the same for all regions, and wont change over time
        env_t = typing.cast("env.MultiTraceEnv", self.env)
        c_od = env_t.envs[region_id].get_price()[ClusterType.ON_DEMAND]

        return V - c_od

    # ===== Bandit helpers on top of unified cost values =====
    def _apply_bandit_decay_if_needed(self):
        if self.bandit_history != "discount":
            return
        tick = self.env.tick
        if self._last_decay_tick == tick:
            return  # already decayed this tick
        self._last_decay_tick = tick
        f = self._decay_factor_per_tick
        if abs(f - 1.0) < 1e-9:
            return
        # Decay effective attempts and aggregates
        for i in range(len(self.region_eff_attempts)):
            self.region_eff_attempts[i] *= f
            self.region_sum_rewards[i] *= f
            self.region_sum_charge_hours[i] *= f
            self.region_duration_sum[i] *= f
            self.region_duration_count[i] *= f
        self.total_eff_attempts *= f

    def _update_bandit_stats_on_attempt(self, region: int):
        # Raw attempts (for logging) always increment
        self.total_attempts += 1
        if 0 <= region < len(self.region_attempts):
            self.region_attempts[region] += 1
        # Effective attempts for exploration term
        if self.bandit_history in ("none", "discount"):
            self.total_eff_attempts += 1.0
            if 0 <= region < len(self.region_eff_attempts):
                self.region_eff_attempts[region] += 1.0
        elif self.bandit_history == "window":
            # Use window size as attempt proxy: count of samples in window
            # We still increase a lightweight counter to avoid zero at start
            self.total_eff_attempts = max(self.total_eff_attempts + 1.0, 1.0)
            if 0 <= region < len(self.region_eff_attempts):
                self.region_eff_attempts[region] = min(
                    float(len(self.region_window_rewards[region]) + 1),
                    float(self.window_attempts),
                )

    def _update_bandit_stats_on_reward(
        self, region: int, reward_usd: float, charge_hours: float
    ):
        if region < 0 or region >= len(self.region_sum_rewards):
            return
        if self.bandit_history in ("none", "discount"):
            self.region_sum_rewards[region] += reward_usd
            self.region_sum_charge_hours[region] += charge_hours
        elif self.bandit_history == "window":
            dq_r = self.region_window_rewards[region]
            dq_h = self.region_window_hours[region]
            # Maintain rolling sums
            if len(dq_r) == self.window_attempts:
                # will evict oldest
                self.region_window_sum_rewards[region] -= dq_r[0]
                self.region_window_sum_hours[region] -= dq_h[0]
            dq_r.append(reward_usd)
            dq_h.append(charge_hours)
            self.region_window_sum_rewards[region] += reward_usd
            self.region_window_sum_hours[region] += charge_hours

    def _finalize_active_spot_run_if_ended(
        self, active_instances: dict[int, ClusterType]
    ):
        """When a tracked SPOT run ends (preempted or terminated), compute realized reward
        using V at launch and update per-region stats.
        Reward (per run) = max(0, (duration_h - d_h)) * V_launch - duration_h * c_spot
        We aggregate reward and charge hours for later per-hour averages.
        """
        if self._active_region is None:
            return
        still_running = (
            self._active_region in active_instances
            and active_instances[self._active_region] == ClusterType.SPOT
        )
        if still_running:
            return
        # Compute realized run statistics
        elapsed_s = self.env.elapsed_seconds - self._active_launch_time_s
        duration_h = max(0.0, elapsed_s / 3600.0)
        d_h = self.restart_overhead / 3600.0
        effective_h = max(0.0, duration_h - d_h)

        env_t = typing.cast("env.MultiTraceEnv", self.env)
        c_spot = env_t.envs[self._active_region].get_price()[ClusterType.SPOT]

        reward = effective_h * self._active_launch_V_per_hour - duration_h * c_spot
        region = self._active_region
        # Update stats (mode-dependent)
        self._update_bandit_stats_on_reward(region, reward, duration_h)
        # Censor-aware: only learn duration from non-terminated-by-us runs
        if not getattr(self, "_active_terminated_by_us", False):
            if self.bandit_history == "window":
                dq = self.region_duration_window[region]
                if len(dq) == self.window_attempts:
                    self.region_duration_window_sum[region] -= dq[0]
                dq.append(duration_h)
                self.region_duration_window_sum[region] += duration_h
            else:
                self.region_duration_sum[region] += duration_h
                self.region_duration_count[region] += 1.0
        # Clear tracking
        self._active_region = None
        self._active_launch_time_s = 0.0
        self._active_launch_V_per_hour = 0.0
        self._active_terminated_by_us = False

    def _bandit_adjusted_value(
        self, base_value_per_hour: float, region_id: int
    ) -> float:
        """Return bandit-adjusted per-hour value for a SPOT option in a given region.
        - 'none': return base value
        - 'ucb': base + (avg_per_hour + bonus)
        - 'thompson': base + N(mean=avg_per_hour, std=scale/sqrt(attempts)) sample
        """
        if self.bandit_mode == "none":
            return base_value_per_hour
        # Apply optional decay once per tick
        self._apply_bandit_decay_if_needed()

        # Compute avg_per_hour and attempts according to history mode
        if self.bandit_history == "window":
            sum_h = self.region_window_sum_hours[region_id]
            sum_r = self.region_window_sum_rewards[region_id]
            avg_per_hour = (sum_r / sum_h) if sum_h > 1e-9 else 0.0
            attempts = int(len(self.region_window_rewards[region_id]))
            total = int(sum(len(dq) for dq in self.region_window_rewards))
        else:
            charge_h = (
                self.region_sum_charge_hours[region_id]
                if region_id < len(self.region_sum_charge_hours)
                else 0.0
            )
            avg_per_hour = (
                (self.region_sum_rewards[region_id] / charge_h)
                if charge_h > 1e-9
                else 0.0
            )
            attempts = (
                max(0, int(round(self.region_eff_attempts[region_id])))
                if self.bandit_history == "discount"
                else self.region_attempts[region_id]
            )
            total = (
                int(round(self.total_eff_attempts))
                if self.bandit_history == "discount"
                else max(1, self.total_attempts)
            )
        # Blend base and learned avg to avoid double-counting magnitude
        blended = (
            1.0 - self.bandit_alpha
        ) * base_value_per_hour + self.bandit_alpha * avg_per_hour
        # Guard: suppress proactive exploration while a SPOT is running (unless explicitly allowed)
        if self._suppress_exploration and self.only_explore_when_idle:
            return blended  # no exploration bonus / sampling
        # Ensure attempts>=1 for formulas, but avoid infinity boosts
        eff_attempts = max(1, attempts)
        eff_total = max(eff_attempts + 1, total + 1)
        if self.bandit_mode == "ucb":
            bonus = self.ucb_c * math.sqrt(2.0 * math.log(eff_total) / eff_attempts)
            return blended + bonus
        else:
            # thompson
            std = self.thompson_scale / math.sqrt(eff_attempts)
            sampled = random.gauss(avg_per_hour, std)
            return (
                1.0 - self.bandit_alpha
            ) * base_value_per_hour + self.bandit_alpha * sampled

    def _step_multi(
        self,
    ) -> typing.Generator["Action", typing.Optional["LaunchResult"], None]:
        """Pure unified cost model decision making - simplified version."""

        env = typing.cast("env.MultiTraceEnv", self.env)

        # Get current state and run pre-step cleanup if needed
        active_instances = env.get_active_instances()
        for pre_action in self._pre_step_actions(dict(active_instances)):
            if isinstance(pre_action, TryLaunch):
                pre_result = yield pre_action
                assert pre_result is not None
                self._handle_post_step_action_result(pre_action, pre_result)
            else:
                yield pre_action
                self._handle_post_step_action_result(pre_action, None)
        active_instances = env.get_active_instances()
        logger.debug(f"[UCM] Debug: raw active_instances = {active_instances}")
        # Finalize any ended SPOT run before making new decisions (use raw active_instances)
        self._finalize_active_spot_run_if_ended(dict(active_instances))
        active_instances = self._filter_active_instances(dict(active_instances))
        logger.debug(f"[UCM] Debug: filtered active_instances = {active_instances}")
        logger.debug(
            f"[UCM] Debug: _probes_active = {getattr(self, '_probes_active', 'N/A')}"
        )

        # Check if task is done
        remaining_task = self.task_duration - sum(self.task_done_time)
        if remaining_task <= 1e-3:
            # Task complete - terminate everything
            for region in list(active_instances.keys()):
                action = Terminate(region=region)
                yield action
                self._handle_post_step_action_result(action, None)
            yield from self._run_post_step_actions(
                decision_outcome="task_complete",
                primary_region=None,
                primary_type=ClusterType.NONE,
                active_instances={},
            )
            return

        # Compute current time value
        V = self._compute_time_value()
        logger.debug(f"[UCM] V={V:.3f}")

        od_values = [
            self._compute_od_net_value(V, rid) for rid in range(env.num_regions)
        ]
        gap_seconds = env.gap_seconds
        remaining_time = (
            math.floor((self.deadline - env.elapsed_seconds) / gap_seconds)
            * gap_seconds
        )
        remaining_task_time = remaining_task
        total_task_remaining = (
            math.ceil((remaining_task_time + self.restart_overhead) / gap_seconds)
            * gap_seconds
        )
        in_critical_window = (
            remaining_task_time > 1e-3 and remaining_time <= total_task_remaining
        )

        # Apply migration penalty if we have an active instance
        migration_penalty = 0
        current_region = None
        current_type: ClusterType = ClusterType.NONE
        current_value = 0  # Waiting
        if active_instances:
            logger.debug(
                f"[UCM] Debug: active_instances before filter = {active_instances}"
            )
            logger.debug(
                f"[UCM] Debug: _probes_active = {getattr(self, '_probes_active', 'N/A')}"
            )
            assert len(active_instances) == 1, active_instances
            current_region = list(active_instances.keys())[0]
            current_type = active_instances[current_region]
            # Keep a simple baseline penalty here; per-candidate dynamic penalty is applied below
            migration_penalty = self.restart_overhead / 3600 * V

            # Get current instance value for comparison
            if current_type == ClusterType.SPOT:
                current_value = self._compute_spot_net_value(
                    current_region,
                    V,
                    ignore_restart_overhead=True,
                )
            elif current_type == ClusterType.ON_DEMAND:
                assert current_region is not None
                current_value = od_values[current_region]
            else:
                assert False

            logger.info(
                f"Current: {current_type.name} in R{current_region} (value={current_value:.3f})"
            )
            # Update last-region marker
            self._last_region = current_region

        # Exploration/migration guardrails
        # Suppress exploration if currently on SPOT and not allowing spot->spot migration
        self._suppress_exploration = (
            current_type == ClusterType.SPOT and not self.allow_spot_to_spot_migration
        )

        # Build unified list of all options with their values
        # If we have active instances, stopping means we'll need to restart later
        wait_value = 0.0

        all_options: list[tuple[float, str, typing.Optional[int]]] = [
            (wait_value, "NONE", None)
        ]

        self._begin_candidate_capture()

        option_details: list[dict[str, typing.Any]] = []
        option_detail_map: dict[
            tuple[str, typing.Optional[int]], dict[str, typing.Any]
        ] = {}

        assert hasattr(self.task, "checkpoint_size_gb"), (
            f"task missing checkpoint_size_gb: {self.task}"
        )
        checkpoint_size_gb = self.task.checkpoint_size_gb

        # Add all SPOT options (apply bandit adjustment and optional dynamic migration penalty per candidate)
        for region_id in range(env.num_regions):
            row = self._candidate_debug_row(region_id)
            ignore_restart = (
                current_type == ClusterType.SPOT and region_id == current_region
            )
            base = self._compute_spot_net_value(
                region_id,
                V,
                ignore_restart_overhead=ignore_restart,
            )
            if row is not None:
                row["base_value_per_hour"] = base
            adjusted = self._bandit_adjusted_value(base, region_id)
            if row is not None:
                row["bandit_adjusted_value"] = adjusted

            # Dynamic migration penalty: time value lost + transfer cost (converted to per-hour)
            per_candidate_penalty = 0.0
            penalty_components: typing.Optional[dict[str, float]] = None
            # When idle, use last known region as origin if available
            origin_region = (
                current_region if current_region is not None else self._last_region
            )
            if (
                self.consider_migration_penalty
                and origin_region is not None
                and region_id != origin_region
            ):
                from_region_name = env.get_region_name(origin_region)
                to_region_name = env.get_region_name(region_id)
                instance_startup_hours = self.restart_overheads[0] / 3600.0
                try:
                    transfer_time_hours = get_transfer_time_hours(
                        from_region_name, to_region_name, checkpoint_size_gb
                    )
                except Exception:
                    transfer_time_hours = 0.0
                try:
                    relationship = get_region_relationship(
                        from_region_name, to_region_name
                    )
                except Exception:
                    relationship = None
                count_migration_time = env._count_cross_region_migration_time
                effective_transfer_hours = transfer_time_hours
                if (
                    relationship in {"cross_region", "cross_continent"}
                    and not count_migration_time
                ):
                    effective_transfer_hours = 0.0
                additional_downtime_hours = max(effective_transfer_hours, 0.0)
                transfer_usd = get_transfer_cost_usd(
                    from_region_name, to_region_name, checkpoint_size_gb
                )
                # Convert USD penalty to per-hour by dividing by expected duration of the candidate region
                duration = self._get_expected_duration_hours(region_id)
                duration = max(duration, 1e-6)
                downtime_value_usd = additional_downtime_hours * V
                penalty_usd = downtime_value_usd + transfer_usd
                per_candidate_penalty = penalty_usd / duration
                if per_candidate_penalty > 0:
                    penalty_components = {
                        "restart_hours_accounted_in_base": instance_startup_hours,
                        "transfer_hours": effective_transfer_hours,
                        "raw_transfer_hours": transfer_time_hours,
                        "relationship": relationship,
                        "downtime_value_usd": downtime_value_usd,
                        "transfer_cost_usd": transfer_usd,
                        "penalty_usd": penalty_usd,
                        "duration_hours": duration,
                    }
            if row is not None:
                row["migration_penalty_per_hour"] = per_candidate_penalty
                if penalty_components is not None:
                    row["penalty_breakdown"] = penalty_components

            value = adjusted - per_candidate_penalty
            if row is not None:
                row["value_after_penalty"] = value
            if value < 0:
                self._set_candidate_status(
                    region_id, "negative_value", candidate_value=value
                )
                continue
            # Optional VOI gating for exploration: require margin over current best
            if (
                self.voi_gating
                and current_region is not None
                and region_id != current_region
            ):
                # dynamic delta grows with checkpoint size and V
                delta = (
                    self.voi_delta_base
                    + self.voi_k_checkpoint * checkpoint_size_gb
                    + self.voi_k_v * V
                )
                # if value is not sufficiently better than current value, skip
                if value <= current_value + delta:
                    self._set_candidate_status(
                        region_id,
                        "voi_blocked",
                        candidate_value=value,
                        voi_delta=delta,
                    )
                    continue
            all_options.append((value, "SPOT", region_id))
            if row is not None:
                row.setdefault("status", "candidate")
                row["candidate_value"] = value

        od_bandit_margin = self.od_margin if self.bandit_mode != "none" else 0.0
        origin_region = (
            current_region if current_region is not None else self._last_region
        )
        for od_region in range(env.num_regions):
            c_od = env.envs[od_region].get_price()[ClusterType.ON_DEMAND]
            od_base = od_values[od_region]
            od_penalty_components: typing.Optional[dict[str, typing.Any]] = None
            od_penalty_value = 0.0
            if (
                self.consider_migration_penalty
                and origin_region is not None
                and od_region != origin_region
            ):
                from_region_name = env.get_region_name(origin_region)
                to_region_name = env.get_region_name(od_region)
                instance_startup_hours = self.restart_overheads[0] / 3600.0
                try:
                    transfer_time_hours = get_transfer_time_hours(
                        from_region_name, to_region_name, checkpoint_size_gb
                    )
                except Exception:
                    transfer_time_hours = 0.0
                try:
                    relationship = get_region_relationship(
                        from_region_name, to_region_name
                    )
                except Exception:
                    relationship = None
                count_migration_time = env._count_cross_region_migration_time
                effective_transfer_hours = transfer_time_hours
                if (
                    relationship in {"cross_region", "cross_continent"}
                    and not count_migration_time
                ):
                    effective_transfer_hours = 0.0
                migration_hours = instance_startup_hours + effective_transfer_hours
                transfer_usd = get_transfer_cost_usd(
                    from_region_name, to_region_name, checkpoint_size_gb
                )
                downtime_value_usd = migration_hours * V
                od_penalty_value = downtime_value_usd + transfer_usd
                if od_penalty_value > 0:
                    od_penalty_components = {
                        "restart_hours": instance_startup_hours,
                        "transfer_hours": effective_transfer_hours,
                        "raw_transfer_hours": transfer_time_hours,
                        "relationship": relationship,
                        "downtime_value_usd": downtime_value_usd,
                        "transfer_cost_usd": transfer_usd,
                    }

            od_option_value = od_base - od_penalty_value
            candidate_od_value = od_option_value - od_bandit_margin

            od_detail = {
                "type": "ON_DEMAND",
                "region": od_region,
                "candidate_value": candidate_od_value,
                "net_base": od_base,
                "value_after_penalty": od_option_value,
                "penalty": od_penalty_value,
                "penalty_breakdown": od_penalty_components,
                "bandit_margin": od_bandit_margin,
                "c_od": c_od,
                "status": "candidate",
                "notes": [],
            }
            option_details.append(od_detail)
            option_detail_map[("ON_DEMAND", od_region)] = od_detail

            if od_option_value < 0:
                od_detail["status"] = "negative_value"

            if (
                current_type == ClusterType.ON_DEMAND
                and current_region is not None
                and od_region == current_region
            ):
                od_detail["status"] = "current_instance"

            appended = False
            if (
                self.voi_gating
                and current_region is not None
                and (
                    current_type != ClusterType.ON_DEMAND or current_region != od_region
                )
            ):
                delta = (
                    self.voi_delta_base
                    + self.voi_k_checkpoint * checkpoint_size_gb
                    + self.voi_k_v * V
                )
                od_detail["voi_delta"] = delta
                if candidate_od_value > current_value + delta:
                    all_options.append((candidate_od_value, "ON_DEMAND", od_region))
                    appended = True
                else:
                    od_detail["status"] = "voi_blocked"
            else:
                if od_detail.get("status") == "candidate" and candidate_od_value >= 0:
                    all_options.append((candidate_od_value, "ON_DEMAND", od_region))
                    appended = True

            if appended and candidate_od_value < 0:
                od_detail["status"] = "negative_value"

        # Sort by value (descending)
        all_options.sort(key=lambda x: x[0], reverse=True)

        # Drop all options with negative value
        all_options = [option for option in all_options if option[0] >= 0]
        if self._candidate_log_rows is not None:
            rank = 1
            ordered: list[tuple[float, str, typing.Optional[int]]] = []
            for option_value, option_type, option_region in all_options:
                ordered.append((option_value, option_type, option_region))
                if option_type == "SPOT" and option_region is not None:
                    row = self._candidate_debug_row(option_region)
                    if row is not None:
                        if "status" not in row:
                            row["status"] = "candidate"
                        row["candidate_value"] = option_value
                        row["rank"] = rank
                        rank += 1
            self._candidate_summary_context = {
                "tick": env.tick,
                "V": V,
                "current_region": current_region,
                "current_type": current_type,
                "current_value": current_value,
                "wait_value": wait_value,
                "ordered_options": ordered,
                "option_details": option_details,
            }

        # Try options in order until we find something better than current
        attempts_this_tick = 0
        for value, cluster_type_str, region in all_options:
            assert value >= 0

            # Should be checked first, like current WAITING and still WAITING
            if value <= current_value:
                if region is not None:
                    self._set_candidate_status(
                        region,
                        "not_better_than_current",
                        candidate_value=value,
                        current_value=current_value,
                    )
                    if cluster_type_str == "ON_DEMAND":
                        detail = option_detail_map.get((cluster_type_str, region))
                        if detail is not None:
                            detail["status"] = "not_better_than_current"
                            detail["current_value"] = current_value
                            detail["candidate_value"] = value
                break

            # Skip current instance
            if region == current_region and cluster_type_str == current_type.name:
                if region is not None:
                    self._set_candidate_status(region, "current_instance")
                    if cluster_type_str == "ON_DEMAND":
                        detail = option_detail_map.get((cluster_type_str, region))
                        if detail is not None:
                            detail["status"] = "current_instance"
                continue

            launch_success: bool = False

            # Try to launch this option
            if cluster_type_str == "SPOT":
                assert region is not None
                self._set_candidate_status(region, "launch_attempt")
                for pre_action in self._before_spot_launch(region):
                    if isinstance(pre_action, TryLaunch):
                        pre_result = yield pre_action
                        assert pre_result is not None
                        self._handle_post_step_action_result(pre_action, pre_result)
                    else:
                        yield pre_action
                        self._handle_post_step_action_result(pre_action, None)
                logger.info(f"Trying SPOT in region {region} (value={value:.3f})")
                action = TryLaunch(region=region, cluster_type=ClusterType.SPOT)
                result = yield action
                assert result is not None
                self._handle_post_step_action_result(action, result)
                # Update bandit attempts (raw + effective)
                self._update_bandit_stats_on_attempt(region)
                attempts_this_tick += 1
                launch_success = result.success
                if region is not None:
                    if result.success:
                        self._set_candidate_status(region, "launch_success")
                    else:
                        self._set_candidate_status(region, "launch_failed")
                if result.success:
                    # Track active run for reward finalization later
                    self._active_region = region
                    self._active_launch_time_s = env.elapsed_seconds
                    self._active_launch_V_per_hour = V
            elif cluster_type_str == "ON_DEMAND":
                assert region is not None
                # Check if we need to terminate existing instance in this region first
                active_instances = env.get_active_instances()
                terminated_region = None
                if region in active_instances:
                    logger.debug(
                        f"Terminating {active_instances[region].name} in region {region} before launching ON_DEMAND"
                    )
                    yield Terminate(region=region)
                    terminated_region = region
                logger.debug(f"Launching ON_DEMAND (value={value:.3f})")
                od_detail = option_detail_map.get((cluster_type_str, region))
                if od_detail is not None:
                    od_detail["status"] = "launch_attempt"
                action = TryLaunch(region=region, cluster_type=ClusterType.ON_DEMAND)
                result = yield action
                assert result is not None
                assert result.success  # ON_DEMAND always succeeds
                self._handle_post_step_action_result(action, result)
                launch_success = True
                if od_detail is not None:
                    od_detail["status"] = "launch_success"
                # Mark that we already handled termination if it was the current region
                if terminated_region == current_region:
                    current_region = None  # Clear to avoid double termination
                # Track last region for future migration penalty when idle
                self._last_region = region
            else:
                # We should wait, and terminate current instance
                launch_success = True

            if launch_success:
                if cluster_type_str == "SPOT":
                    post_primary_region = region
                    post_primary_type = ClusterType.SPOT
                    post_active_instances = {region: ClusterType.SPOT}
                    decision_outcome = "launch_spot"
                elif cluster_type_str == "ON_DEMAND":
                    post_primary_region = region
                    post_primary_type = ClusterType.ON_DEMAND
                    post_active_instances = {region: ClusterType.ON_DEMAND}
                    decision_outcome = "launch_on_demand"
                else:
                    post_primary_region = None
                    post_primary_type = ClusterType.NONE
                    post_active_instances = {}
                    decision_outcome = "wait"
                if current_region is not None:
                    logger.info(
                        f"Terminating old instance {current_type.name} in R{current_region}"
                    )
                    self._active_terminated_by_us = True
                    term_action = Terminate(region=current_region)
                    yield term_action
                    self._handle_post_step_action_result(term_action, None)
                self._emit_candidate_summary(
                    outcome=decision_outcome,
                    selected_region=post_primary_region,
                    selected_type=post_primary_type,
                )
                yield from self._run_post_step_actions(
                    decision_outcome=decision_outcome,
                    primary_region=post_primary_region,
                    primary_type=post_primary_type,
                    active_instances=post_active_instances,
                )
                return
            # Respect attempts cap per tick to avoid too many launch attempts
            # If max_attempts_per_tick <= 0, treat as unlimited (no cap)
            if (
                self.max_attempts_per_tick > 0
                and attempts_this_tick >= self.max_attempts_per_tick
            ):
                break

        logger.debug(
            f"No viable options, keeping current {current_type.name} in R{current_region}"
        )
        if current_region is not None:
            post_decision = "keep_current"
            post_primary_region = current_region
            post_primary_type = current_type
            post_active_instances = active_instances
        else:
            post_decision = "idle"
            post_primary_region = None
            post_primary_type = ClusterType.NONE
            post_active_instances = {}
        self._emit_candidate_summary(
            outcome=post_decision,
            selected_region=post_primary_region,
            selected_type=post_primary_type,
        )
        yield from self._run_post_step_actions(
            decision_outcome=post_decision,
            primary_region=post_primary_region,
            primary_type=post_primary_type,
            active_instances=post_active_instances,
        )

    @classmethod
    def _from_args(cls, parser):
        group = parser.add_argument_group("UnifiedCostModelStrategy")
        group.add_argument(
            "--bandit-mode",
            type=str,
            default="none",
            choices=["none", "ucb", "thompson"],
            help="Enable explore/exploit on top of unified costs",
        )
        group.add_argument(
            "--ucb-c", type=float, default=0.2, help="UCB exploration coefficient"
        )
        group.add_argument(
            "--max-attempts-per-tick",
            type=int,
            default=0,
            help="Max number of SPOT regions to try per tick (0 or negative = unlimited)",
        )
        group.add_argument(
            "--od-margin",
            type=float,
            default=0.0,
            help="Require OD per-hour value to exceed best SPOT by this margin",
        )
        group.add_argument(
            "--thompson-scale",
            type=float,
            default=0.2,
            help="Scale of Thompson sampling std (std=scale/sqrt(attempts))",
        )
        group.add_argument(
            "--ignore-migration-penalty",
            action="store_true",
            help="Ignore dynamic migration time and transfer cost in decision scoring",
        )
        group.add_argument(
            "--bandit-history",
            type=str,
            default="none",
            choices=["none", "discount", "window"],
            help="History handling for non-stationarity: none | discount | window",
        )
        group.add_argument(
            "--discount-half-life-hours",
            type=float,
            default=24.0,
            help="Half-life (hours) for exponential discounting (when bandit-history=discount)",
        )
        group.add_argument(
            "--window-attempts",
            type=int,
            default=12,
            help="Number of recent attempts to keep (when bandit-history=window)",
        )
        group.add_argument(
            "--bandit-alpha",
            type=float,
            default=0.0,
            help="Blend weight between base per-hour value and learned avg (0..1)",
        )
        group.add_argument(
            "--only-explore-when-idle",
            action=BooleanOptionalAction,
            default=True,
            help="Disable proactive exploration while a SPOT is running",
        )
        group.add_argument(
            "--allow-spot-to-spot-migration",
            action="store_true",
            help="Permit migrations from SPOT to SPOT while exploring",
        )
        group.add_argument(
            "--voi-gating",
            action=BooleanOptionalAction,
            default=True,
            help="Enable value-of-information gating for exploration",
        )
        group.add_argument(
            "--voi-delta-base",
            type=float,
            default=0.5,
            help="Base margin (USD/h) required to justify exploration",
        )
        group.add_argument(
            "--voi-k-checkpoint",
            type=float,
            default=0.0002,
            help="Additional margin per GB of checkpoint",
        )
        group.add_argument(
            "--voi-k-v",
            type=float,
            default=0.05,
            help="Additional margin proportional to V (USD/h)",
        )
        group.add_argument(
            "--use-online-duration",
            action="store_true",
            help="Use online duration estimate to replace offline avg_duration",
        )
        group.add_argument(
            "--online-duration-min-samples",
            type=int,
            default=2,
            help="Minimum samples before trusting online duration",
        )
        group.add_argument(
            "--online-duration-min-hours",
            type=float,
            default=0.25,
            help="Lower clamp for online duration (hours)",
        )
        group.add_argument(
            "--online-duration-max-hours",
            type=float,
            default=24.0,
            help="Upper clamp for online duration (hours)",
        )
        group.add_argument(
            "--online-duration-mix",
            type=float,
            default=1.0,
            help="Weight of online duration (0..1), modulated by sample count",
        )
        return cls(parser.parse_args())
