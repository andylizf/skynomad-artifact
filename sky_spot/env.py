import math
import json
import logging
import os
import typing
from typing import Dict, List, Tuple, Type, Sequence, Optional, Any, Set
import abc
from collections import defaultdict
from pathlib import Path

from sky_spot import trace
from sky_spot.utils import ClusterType, COSTS, DEVICE_COSTS, COST_K

if typing.TYPE_CHECKING:
    import configargparse
    from sky_spot.strategies.strategy import MultiRegionStrategy

logger = logging.getLogger(__name__)

PROBE_BILLING_SECONDS = 60.0  # Charge probes for a fixed one-minute duration


def _extract_region_and_device(trace_file: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        region_device = Path(trace_file).parent.name
    except Exception:
        return None, None
    for device_key in sorted(DEVICE_COSTS.keys(), key=len, reverse=True):
        suffix = f"_{device_key}"
        if region_device.endswith(suffix):
            region = region_device[: -len(suffix)]
            return region or None, device_key
    return region_device or None, None


def _resolve_region_on_demand_price(
    price_info: Optional[Dict[str, Any]],
) -> Optional[float]:
    if price_info:
        ondemand_meta = (
            price_info.get("on_demand_price")
            or price_info.get("ondemand_price")
            or price_info.get("onDemandPrice")
        )
        if ondemand_meta is not None:
            try:
                return float(ondemand_meta)
            except Exception:
                logger.warning(
                    "Invalid on_demand_price in price_info: %s", ondemand_meta
                )
    return None


class Env(abc.ABC):
    NAME = "abstract"
    SUBCLASSES: Dict[str, Type["Env"]] = {}

    def __init__(self, gap_seconds: float):
        self.gap_seconds = gap_seconds
        self.reset()

    def reset(self):
        # dones not include the cluster_type for the current timestamp - 1 -> timestamp, until observed on timestamp
        self.cluster_type_histroy = []
        self.cluster_type = ClusterType.NONE
        self.tick = 0
        self.observed_tick = -1

    def __init_subclass__(cls) -> None:
        assert cls.NAME not in cls.SUBCLASSES and cls.NAME != "abstract", (
            f"Name {cls.NAME} already exists"
        )
        cls.SUBCLASSES[cls.NAME] = cls

    def spot_available(self) -> bool:
        """
        Returns True if spot is available at the current timestamp -> timestamp + 1
        """
        raise NotImplementedError

    def observe(self) -> Tuple[ClusterType, bool]:
        """
        Returns the cluster type (at last time gap) and whether spot is available
        """
        assert self.observed_tick == self.tick - 1, (self.observed_tick, self.tick)
        self.observed_tick = self.tick
        has_spot = self.spot_available()
        last_cluster_type = self.cluster_type
        self.cluster_type_histroy.append(last_cluster_type)

        if self.cluster_type == ClusterType.SPOT and not has_spot:
            logger.debug(f"Preempted at {self.tick}")
            self.cluster_type = ClusterType.NONE
        return last_cluster_type, has_spot

    def step(self, request_type: ClusterType):
        if self.observed_tick != self.tick:
            self.observe()
        if request_type == ClusterType.SPOT and not self.spot_available():
            raise ValueError("Spot not available")
        new_cluster_type = self._step(request_type)
        self.tick += 1
        return new_cluster_type

    def _step(self, request_type: ClusterType):
        self.cluster_type = request_type
        return self.cluster_type

    def get_trace_before_end(self, end: float) -> trace.Trace:
        # Used for ideal strategy
        raise NotImplementedError

    @property
    def elapsed_seconds(self) -> float:
        return self.tick * self.gap_seconds

    @property
    def accumulated_cost(self) -> float:
        """Accumulated cost of the environment"""
        costs_map = self.get_constant_cost_map()
        return sum(
            costs_map[cluster_type] * self.gap_seconds / 3600
            for cluster_type in self.cluster_type_histroy
        )

    def get_constant_cost_map(self) -> Dict[ClusterType, float]:
        return COSTS

    def info(self) -> dict:
        # Step should have been called
        assert self.tick == self.observed_tick + 1
        return {
            "Timestamp": self.tick - 1,
            "Elapsed": (self.tick - 1) * self.gap_seconds,
            "Cost": self.accumulated_cost,
            "ClusterType": self.cluster_type_histroy[-1].value
            if self.cluster_type_histroy
            else ClusterType.NONE.value,
        }

    def __repr__(self) -> str:
        return f"{self.NAME}({json.dumps(self.config)})"

    @property
    def config(self):
        return dict()

    @classmethod
    def from_args(cls, parser: "configargparse.ArgumentParser") -> Sequence["Env"]:
        # parser.add_argument(f'--env-config', type=str, default=None, is_config_file=True, required=False)
        parser.add_argument(
            "--env", type=str, default="trace", choices=cls.SUBCLASSES.keys()
        )
        args, _ = parser.parse_known_args()
        cls = cls.SUBCLASSES[args.env]
        return cls._from_args(parser)

    @classmethod
    def _from_args(cls, parser: "configargparse.ArgumentParser") -> Sequence["Env"]:
        raise NotImplementedError


class TraceEnv(Env):
    NAME = "trace"

    def __init__(
        self, trace_file: str, env_start_hours: float, window_hours: float | None = None,
        gang_threshold: int | None = None
    ):
        self._trace_file = trace_file
        self.trace: trace.Trace = trace.Trace.from_file(trace_file, gang_threshold=gang_threshold)

        self._start_index = 0
        if env_start_hours > 0:
            self._start_index = int(
                math.ceil(env_start_hours * 3600 / self.trace.gap_seconds)
            )

        # Optional runtime window enforcement (exclusive upper bound in ticks relative to start)
        self._window_max_ticks: int | None = None
        if window_hours is not None and window_hours > 0:
            self._window_max_ticks = int(
                math.ceil(window_hours * 3600 / self.trace.gap_seconds)
            )

        region_name, device_key = _extract_region_and_device(trace_file)
        price_info = (
            self.trace.metadata.get("price_info")
            if isinstance(self.trace.metadata, dict)
            else None
        )
        if price_info and "zone" in price_info:
            region_name = price_info.get("zone")
        elif isinstance(self.trace.metadata, dict) and "zone" in self.trace.metadata:
            region_name = self.trace.metadata.get("zone")

        self.region_name = region_name

        # Get on_demand price - NO FALLBACK, must be explicit
        self._base_price = _resolve_region_on_demand_price(price_info)
        if self._base_price is None:
            raise ValueError(
                f"No on_demand_price in price_info for {trace_file}. "
                f"price_info={price_info}"
            )

        # Get spot price - NO FALLBACK, must be explicit
        self._spot_price = None
        if self.trace.get_price(0) is not None:
            # Price array exists in trace - will use trace.get_price() in get_price()
            pass
        elif price_info and "price" in price_info:
            try:
                self._spot_price = float(price_info["price"])
            except Exception:
                raise ValueError(
                    f"Invalid spot price in price_info for {trace_file}. "
                    f"price_info['price']={price_info.get('price')}"
                )
        else:
            raise ValueError(
                f"No spot price found for {trace_file}. "
                f"Neither trace.price nor price_info['price'] exists."
            )

        super().__init__(self.trace.gap_seconds)

    def spot_available(self) -> bool:
        tick = self.tick + self._start_index
        # Enforce window bound if configured
        if self._window_max_ticks is not None:
            if self.tick > self._window_max_ticks:
                raise ValueError(
                    f"Timestamp {self.tick} reached enforced window bound {self._window_max_ticks}"
                )
            if self.tick == self._window_max_ticks:
                return False
        # Allow a boundary observe at tick == len(trace) to finalize the last interval's progress
        # without going out of bounds; treat availability as False at the boundary.
        if tick > len(self.trace):
            raise ValueError(f"Timestamp {tick} out of range {len(self.trace)}")
        if tick == len(self.trace):
            return False
        return not self.trace[tick]

    def get_available_ticks(self) -> int:
        base_ticks = len(self.trace) - self._start_index
        if self._window_max_ticks is not None:
            return min(base_ticks, self._window_max_ticks)
        return base_ticks

    def get_max_tick_relative(self) -> int:
        return self.get_available_ticks() - 1

    def get_trace_before_end(self, end: float) -> trace.Trace:
        end_index = int(math.ceil(end / self.gap_seconds))
        return self.trace[self._start_index : end_index + self._start_index]  # type: ignore

    def next_wait_spot_length(self) -> Tuple[int, int]:
        wait_length = 0
        spot_length = 0
        start = self.tick + self._start_index
        if not self.spot_available():
            for i in range(start, len(self.trace)):
                if not self.trace[i]:
                    start = i
                    break
                wait_length += 1

        for i in range(start, len(self.trace)):
            if not self.trace[i]:
                spot_length += 1
            else:
                break
        return wait_length, spot_length

    def get_constant_cost_map(self) -> Dict[ClusterType, float]:
        return {
            ClusterType.ON_DEMAND: float(self._base_price),
            ClusterType.SPOT: float(self.trace.get_price(0))  # type: ignore
            if self._spot_price is None
            else float(self._spot_price),
            ClusterType.NONE: 0.0,
        }

    def get_price(self) -> Dict[ClusterType, float]:
        if self._spot_price is not None:
            return {
                ClusterType.ON_DEMAND: float(self._base_price),
                ClusterType.SPOT: float(self._spot_price),
                ClusterType.NONE: 0.0,
            }
        spot_price = self.trace.get_price(self.tick + self._start_index)
        assert spot_price is not None, "Spot price not available"
        return {
            ClusterType.ON_DEMAND: float(self._base_price),
            ClusterType.SPOT: spot_price,
            ClusterType.NONE: 0.0,
        }

    @property
    def accumulated_cost(self) -> float:
        """Accumulate cost with per-tick spot pricing when available.

        Prior behavior used a constant cost map which, when a trace contained a
        `prices` series, effectively charged the first-tick spot price for all
        ticks. This override mirrors MultiTraceEnv's dynamic accounting so that
        single-trace runs (including union traces) are billed using the spot
        price at each tick when provided by the trace.
        """
        total = 0.0
        for i, ct in enumerate(self.cluster_type_histroy):
            if ct == ClusterType.SPOT:
                # Use dynamic per-tick price if available; otherwise fall back
                if self._spot_price is not None:
                    price_per_hour = float(self._spot_price)
                else:
                    price = self.trace.get_price(self._start_index + i)
                    assert price is not None, "Spot price not available"
                    price_per_hour = float(price)
            elif ct == ClusterType.ON_DEMAND:
                price_per_hour = float(self._base_price)
            else:
                price_per_hour = 0.0
            total += price_per_hour * self.gap_seconds / 3600.0
        return total

    @property
    def config(self) -> dict:
        return {
            "name": self.NAME,
            "trace_file": self._trace_file,
            "start_index": self._start_index,
            "metadata": self.trace.metadata,
            "tace_file": self._trace_file,
        }

    @classmethod
    def _from_args(
        cls, parser: "configargparse.ArgumentParser"
    ) -> Sequence["TraceEnv"]:
        group = parser.add_argument_group("TraceEnv")
        group.add_argument(
            "--trace-file", type=str, help="File/folder containing the trace"
        )
        group.add_argument(
            "--env-start-hours", type=float, default=0, help="Start hours of the trace"
        )
        group.add_argument(
            "--enforce-window-bound",
            action="store_true",
            help="If set, enforce runtime window bound = deadline-hours from env_start",
        )
        group.add_argument(
            "--gang-threshold", type=int, default=None,
            help="Gang-scheduling threshold: require count >= threshold to be available (e.g., 16 for full gang)"
        )
        args, _ = parser.parse_known_args()
        window_hours = args.deadline_hours if args.enforce_window_bound else None
        return cls.create_env(args.trace_file, args.env_start_hours, window_hours, args.gang_threshold)

    @classmethod
    def create_env(
        cls,
        trace_file_or_dir: str,
        env_start_hours: float,
        window_hours: float | None = None,
        gang_threshold: int | None = None,
    ) -> Sequence["TraceEnv"]:
        if os.path.isdir(trace_file_or_dir):
            trace_files = []
            for file in sorted(
                os.listdir(trace_file_or_dir), key=lambda x: int(x.split(".")[0])
            ):
                # logger.debug(file)
                if file.endswith(".json"):
                    trace_files.append(os.path.join(trace_file_or_dir, file))
            return [
                cls(trace_file, env_start_hours, window_hours, gang_threshold=gang_threshold)
                for trace_file in trace_files
            ]
        return [cls(trace_file_or_dir, env_start_hours, window_hours, gang_threshold=gang_threshold)]


class MultiTraceEnv(Env):
    NAME = "multi_trace"

    def __init__(
        self,
        trace_files: List[str],
        env_start_hours: float,
        window_hours: float | None = None,
        count_cross_region_migration_time: bool = False,
        gang_threshold: int | None = None,
    ):
        self._trace_files = trace_files
        self.envs = [
            TraceEnv(trace_file, env_start_hours, window_hours, gang_threshold=gang_threshold)
            for trace_file in trace_files
        ]
        self.num_regions = len(trace_files)
        self._count_cross_region_migration_time = count_cross_region_migration_time

        gap_seconds = self.envs[0].trace.gap_seconds
        for env in self.envs:
            assert env.trace.gap_seconds == gap_seconds, (
                "All traces must have the same gap seconds"
            )

        # Multi-region specific state
        # NOTE: Single-instance invariant: len(active_instances) <= 1
        # Structure: (region, cluster_type) -> launch_tick
        self.active_instances: Dict[Tuple[int, ClusterType], int] = {}
        self.cost_history: List[Dict[int, ClusterType]] = []
        self.cost_history_probe_flags: List[Dict[int, bool]] = []
        self.current_tick_probe_flags: Dict[int, bool] = {}
        self.probe_cost_total: float = 0.0
        self.probe_cost_by_region: Dict[int, float] = defaultdict(float)
        self.probe_tick_count: int = 0
        self.current_tick_costs: Dict[int, ClusterType] = {}
        self._launched_this_tick: Set[Tuple[int, ClusterType]] = (
            set()
        )  # Prevent terminating just-launched instances

        # Simple flag to track new launches
        self._new_launch_this_tick = False
        self._had_new_launch_last_tick = False

        # Track the last productive region and cumulative task progress
        self._current_leader_region: Optional[int] = None
        self._current_leader_progress: float = 0.0

        # Launch bookkeeping
        self._launch_info_this_tick: Optional[dict[str, Any]] = None
        self._launch_info_last_tick: Optional[dict[str, Any]] = None

        # Track total number of migrations
        self.migration_count: int = 0
        # Migration breakdown counters
        self.cross_az_migrations_count: int = 0
        self.cross_region_migrations_count: int = 0
        self.cross_continent_migrations_count: int = 0

        # Billing extras: accumulate non-compute costs and timing from migrations
        # - transfer_cost_total: total S3 transfer USD charged on region switches
        # - migration_hours_total: total hours of migration downtime (startup + transfer)
        self.transfer_cost_total: float = 0.0
        self.migration_hours_total: float = 0.0
        # Precise downtime seconds per tick (restart/migration overhead consumed)
        self._downtime_seconds_by_tick: Dict[int, float] = {}

        super().__init__(gap_seconds)
        logger.debug(
            f"MultiTraceEnv initialized with {self.num_regions} regions: {trace_files}"
        )
        # Sticky safety net: once latched, ignore strategy interactions until task completes
        self._safety_net_latched: bool = False

    def reset(self):
        super().reset()
        for env in self.envs:
            env.reset()
        self.active_instances = {}
        self.cost_history = []
        self.cost_history_probe_flags = []
        self.current_tick_probe_flags = {}
        self.probe_cost_total = 0.0
        self.probe_cost_by_region = defaultdict(float)
        self.probe_tick_count = 0
        self.cost_history_ticks = []  # track absolute tick index for each cost snapshot
        self.current_tick_costs = {}
        self._launched_this_tick = set()
        self._new_launch_this_tick = False
        self._had_new_launch_last_tick = False
        # Track the region launched in the previous tick (if any)
        self._last_launch_region: Optional[int] = None
        self.migration_count = 0
        self.cross_az_migrations_count = 0
        self.cross_region_migrations_count = 0
        self.cross_continent_migrations_count = 0
        # Track active regions continuity across ticks
        # Leader snapshot
        self._current_leader_region = None
        self._current_leader_progress = 0.0
        # Launch bookkeeping
        self._launch_info_this_tick = None
        self._launch_info_last_tick = None
        # Reset accumulated extras
        self.transfer_cost_total = 0.0
        self.migration_hours_total = 0.0
        # Reset sticky safety net
        self._safety_net_latched = False
        self._downtime_seconds_by_tick = {}
        logger.debug("MultiTraceEnv reset completed")

    def spot_available(self) -> bool:
        # MultiTraceEnv doesn't have a single spot availability
        raise NotImplementedError(
            "Use _spot_available_in_region() for multi-region environment"
        )

    def _spot_available_in_region(self, region_idx: int) -> bool:
        """Check spot availability in a specific region (internal use only)."""
        assert 0 <= region_idx < self.num_regions, (
            f"Region index {region_idx} out of range"
        )
        # Check if tick is within the valid range for this region
        sub_env = self.envs[region_idx]
        max_tick = sub_env.get_max_tick_relative()
        if self.tick > max_tick:
            if self.tick == max_tick + 1:
                # Boundary observe() after the last valid interval: treat as unavailable
                return False
            # If tick is beyond this region's trace, raise error
            raise ValueError(
                f"Simulation exceeded trace data bounds: tick {self.tick} > max tick {max_tick} "
                f"(region {region_idx}). This could mean: "
                f"1) Task took longer than expected to complete, possibly missing deadline; "
                f"2) Trace file is shorter than the deadline duration ({len(sub_env.trace)} ticks); "
                f"3) Strategy is not terminating properly. "
                f"Check task progress and deadline settings."
            )
        # CRITICAL: Sync the tick of the region with the switcher's tick
        sub_env.tick = self.tick
        return sub_env.spot_available()

    def get_all_regions_spot_prices(self) -> List[Optional[float]]:
        """Return spot prices for all regions, regardless of availability."""
        prices = []
        for i in range(self.num_regions):
            sub_env = self.envs[i]
            max_tick = sub_env.get_max_tick_relative()
            if self.tick > max_tick:
                if self.tick == max_tick + 1:
                    # Boundary tick: reuse last known price (treated as unavailable)
                    sub_env.tick = max_tick
                    price_map = sub_env.get_price()
                    prices.append(price_map.get(ClusterType.SPOT))
                    continue
                # If tick is beyond this region's trace, raise error
                raise ValueError(
                    f"Simulation exceeded trace data bounds: tick {self.tick} > max tick {max_tick} "
                    f"(region {i}). This could mean: "
                    f"1) Task took longer than expected to complete, possibly missing deadline; "
                    f"2) Trace file is shorter than the deadline duration ({len(sub_env.trace)} ticks); "
                    f"3) Strategy is not terminating properly. "
                    f"Check task progress and deadline settings."
                )
            # Sync tick before getting any info from sub-env
            sub_env.tick = self.tick
            # Always get the price, regardless of spot availability
            price_map = sub_env.get_price()
            prices.append(price_map.get(ClusterType.SPOT))
        return prices

    def get_spot_availability_snapshot(self) -> Dict[int, bool]:
        """Return spot availability for all regions at the last completed tick (tick-1).

        This matches `env.info()` which reports `Timestamp = self.tick - 1` and avoids
        querying beyond trace bounds after `self.tick` has been incremented.
        """
        snapshot_tick = max(self.tick - 1, 0)
        availability: Dict[int, bool] = {}
        for region_idx in range(self.num_regions):
            sub_env = self.envs[region_idx]
            max_tick = len(sub_env.trace) - sub_env._start_index - 1
            # Clamp to valid range; if trace is empty, treat as unavailable
            if max_tick < 0:
                availability[region_idx] = False
                continue
            query_tick = min(snapshot_tick, max_tick)
            # Temporarily sync sub-env tick to query_tick
            saved_tick = sub_env.tick
            try:
                sub_env.tick = query_tick
                availability[region_idx] = sub_env.spot_available()
            finally:
                sub_env.tick = saved_tick
        return availability

    def observe(self) -> Tuple[ClusterType, bool]:
        assert self.observed_tick == self.tick - 1, (self.observed_tick, self.tick)
        self.observed_tick = self.tick

        # Check single-instance invariant from previous tick
        assert len(self.active_instances) <= 1, (
            f"Single-instance invariant violated at tick {self.tick}: "
            f"active_instances={self.active_instances}"
        )

        # Finalize costs for the previous tick
        # The instances that need billing are what was in active_instances at the END of the previous tick
        # which is still in active_instances now (before any modifications)
        if self.tick > 0:
            self._finalize_tick_costs()

        # Update launch flags after finalizing costs
        self._had_new_launch_last_tick = self._launch_info_this_tick is not None
        self._launch_info_last_tick = self._launch_info_this_tick
        if self._launch_info_last_tick is not None:
            self._last_launch_region = self._launch_info_last_tick["region"]
        else:
            self._last_launch_region = None
        self._launch_info_this_tick = None
        self._launched_this_tick.clear()  # Clear for new tick
        self._new_launch_this_tick = False

        # Note: _handle_preemptions() is now called in update_strategy_progress()
        # This ensures progress is calculated before instances are preempted

        # Multi-region env doesn't use this interface anymore
        # This method is only here because it's abstract in base class
        return ClusterType.NONE, False

    def _handle_preemptions(self, strategy: "MultiRegionStrategy"):
        """Handle spot instance preemptions at the current tick.

        Semantics: evaluate availability at the current tick. This aligns
        with single-region behavior where preemption is detected at the
        start of the tick (before strategy decisions). Progress for the
        previous interval [tick-1, tick) is calculated using a snapshot
        taken BEFORE this method is called, so preemption doesn't affect
        the previous interval's progress.

        After preemption:
        - Strategy will see no active instance
        - TryLaunch for the preempted region will fail (same tick)
        - Next tick: progress = 0 (no instance was running)
        """
        for (region, cluster_type), launch_tick in list(self.active_instances.items()):
            if cluster_type == ClusterType.SPOT:
                # Check availability at current tick (not tick-1)
                preemption_tick = self.tick
                sub_env = self.envs[region]

                # Temporarily set sub_env tick to check preemption
                original_tick = sub_env.tick
                sub_env.tick = preemption_tick

                # Check if spot is available at current tick
                is_available = sub_env.spot_available()

                # Restore original tick
                sub_env.tick = original_tick

                if not is_available:
                    logger.debug(
                        f"Preempted at tick {self.tick} in region {region} (checked tick {preemption_tick})"
                    )
                    self.active_instances.pop((region, cluster_type))

    def get_trace_before_end(self, end: float) -> trace.Trace:
        # Multi-region env doesn't have a single trace
        raise NotImplementedError(
            "Multi-region environment doesn't have a single trace"
        )

    def get_num_regions(self) -> int:
        """Return the number of available regions."""
        return self.num_regions

    def get_active_instances(self) -> Dict[int, ClusterType]:
        """Get currently active instances across all regions."""
        # Convert back to old format for compatibility
        result = {}
        for (region, cluster_type), _ in self.active_instances.items():
            result[region] = cluster_type
        return result

    def get_region_name(self, region_idx: int) -> str:
        """Get the region name from trace filename."""
        trace_file = self._trace_files[region_idx]
        # Extract region name from path like 'data/.../us-east-1a_v100_1/0.json'
        import os

        region_dir = os.path.basename(os.path.dirname(trace_file))
        return region_dir

    def _safety_net_consolidate_to_single_instance(
        self,
        target_region: int,
        target_type: ClusterType,
        reason: str,
        active_instances: dict[int, ClusterType],
        keep_target_if_exists: bool = False,
    ) -> int:
        """Terminate instances and launch a single instance at target region.

        Args:
            target_region: Region to launch the new instance
            target_type: Type of instance to launch (SPOT or ON_DEMAND)
            reason: Reason for the operation (for logging)
            active_instances: Current active instances
            keep_target_if_exists: If True and target_region has same type, don't restart it

        Returns:
            Number of instances that were terminated
        """
        terminated_count = 0

        # Terminate instances
        for region, ctype in list(active_instances.items()):
            should_terminate = True

            if keep_target_if_exists and region == target_region:
                # Check if target region already has the desired type
                if ctype == target_type:
                    should_terminate = False

            if should_terminate:
                # Pass the cluster type we're terminating
                self._terminate_internal(region, cluster_type=ctype, reason=reason)
                terminated_count += 1

        # Launch new instance if needed
        if not (
            keep_target_if_exists
            and target_region in active_instances
            and active_instances.get(target_region) == target_type
        ):
            launched = self._try_launch_internal(
                target_region,
                target_type,
                reason=reason,
            )
            assert launched, (
                f"Should have launched {target_type.name} in region {target_region}"
            )

        return terminated_count

    def _calculate_cheapest_ondemand_region(
        self, strategy: "MultiRegionStrategy"
    ) -> int:
        """Calculate which region would be cheapest to complete remaining progress with ON_DEMAND.

        Returns the region index that minimizes: completion_cost + migration_cost
        """
        remaining_task_seconds = strategy.task_duration - sum(strategy.task_done_time)

        # Get OD prices for all regions
        # Note: ON_DEMAND price is static (_base_price), but we sync tick for consistency
        od_prices: list[float] = []
        for r in range(self.num_regions):
            self.envs[r].tick = self.tick
            price_map = self.envs[r].get_price()
            od_price = price_map[ClusterType.ON_DEMAND]
            assert od_price is not None, f"ON_DEMAND price not available for region {r}"
            od_prices.append(od_price)

        # Find current leader region (source for migration cost calculation)
        current_leader_region = self._current_leader_region

        min_cost = float("inf")
        best_region = 0

        for r in range(self.num_regions):
            # Compute completion cost in region r
            completion_cost = od_prices[r] * remaining_task_seconds / 3600.0

            # Compute migration cost if moving from current leader
            migration_cost = 0.0
            if current_leader_region is not None and current_leader_region != r:
                from sky_spot.migration_model import get_transfer_cost_usd

                src_name = self.get_region_name(current_leader_region)
                dst_name = self.get_region_name(r)
                assert hasattr(strategy.task, "checkpoint_size_gb"), (
                    f"strategy.task missing checkpoint_size_gb: {strategy.task}"
                )
                checkpoint_size_gb = strategy.task.checkpoint_size_gb
                migration_cost = get_transfer_cost_usd(
                    src_name, dst_name, checkpoint_size_gb
                )
            total_cost = completion_cost + migration_cost

            logger.debug(
                f"[SAFETY NET] Region {r}: OD_price=${od_prices[r]:.3f}/h, "
                f"completion=${completion_cost:.3f}, migration=${migration_cost:.3f}, "
                f"total=${total_cost:.3f}"
            )

            if total_cost < min_cost:
                min_cost = total_cost
                best_region = r

        logger.debug(
            f"[SAFETY NET] Cheapest region for ON_DEMAND completion: {best_region} (cost=${min_cost:.3f})"
        )
        return best_region

    def maybe_trigger_safety_net(self, strategy: "MultiRegionStrategy") -> bool:
        """Returns True if the safety net started to take over from that tick (strategy
        actions should be skipped). Returns False if strategy may proceed normally.

        The purpose of safety net is to 100% ensure the job finishes before deadline,
        no matter how the strategy performs.

        We should trigger the safety net as late as possible. To minimize the impact on
        the specific strategy.
        """
        is_strategy_oracle: bool = getattr(strategy, "IGNORE_SAFETY_NET", False)
        if is_strategy_oracle:
            return False

        # Compute remaining task first (cheap), and allow immediate early returns.
        remaining_task_seconds = strategy.task_duration - sum(strategy.task_done_time)
        if remaining_task_seconds <= 1e-3:
            # Task already effectively done; no safety-net action needed.
            return False

        # Compute remaining time and required work (rounded to tick units)
        # Use math.floor because we can only use complete ticks before deadline
        remaining_time_seconds = (
            math.floor((strategy.deadline - self.elapsed_seconds) / self.gap_seconds)
            * self.gap_seconds
        )
        # If even OD cannot finish (insufficient wall time), safety net is not viable.
        # Panic immediately to surface unrecoverable state.
        # It's a bug. Safety Net should always ensure the task finishes on time.
        #
        # CRITICAL: Restart overhead should NOT be included in this feasibility check.
        # Logic: This check determines if the task is theoretically completable given
        # the remaining work and remaining time. Including restart_overhead here would
        # be overly conservative - we might reject scenarios where the task could
        # actually be completed. The restart_overhead is handled separately in the
        # execution planning below where we calculate the actual ticks needed to
        # complete with overhead included.
        #
        # Example: remaining_time=3600s, remaining_task=3500s, restart_overhead=200s
        # - Without overhead check: 3600 >= 3500 → feasible, proceed to execution planning
        # - With overhead check: 3600 >= 3700 → infeasible, panic unnecessarily
        # - Execution planning: ceil((3500+200)/600)*600 = 3600s needed → exactly fits
        if remaining_time_seconds < remaining_task_seconds:
            raise ValueError(
                f"It's a bug, safety net infeasible: remaining_time_seconds={remaining_time_seconds} < "
                f"remaining_task_seconds={remaining_task_seconds}. Deadline cannot be met even without restart overhead."
            )

        # Safety net always uses full restart_overhead to ensure we can complete
        # even if all instances get preempted (worst case scenario)
        needed = (
            math.ceil(
                (remaining_task_seconds + strategy.restart_overhead) / self.gap_seconds
            )
            * self.gap_seconds
        )

        # If already latched, check if we need emergency recovery
        active_instances = self.get_active_instances()
        if self._safety_net_latched:
            if len(active_instances) == 1:
                logger.debug(
                    "[SAFETY NET] Already latched, ignoring strategy actions "
                    "(remaining time: %.0f s, needed: %.0f s, instances: %s)",
                    remaining_time_seconds,
                    needed,
                    active_instances,
                )
                return True
            if len(active_instances) == 0:
                logger.warning(
                    "[SAFETY NET] Emergency recovery: latched instance was preempted, launching ON_DEMAND immediately "
                    "(remaining time: %.0f s, needed: %.0f s)",
                    remaining_time_seconds,
                    needed,
                )
                # Choose region for emergency OD: prefer pinned override, else cheapest
                pinned_region = getattr(strategy, "SAFETY_NET_PIN_REGION_ID", None)
                if pinned_region is not None:
                    target_region = int(pinned_region)
                    logger.debug(
                        "[SAFETY NET] Emergency recovery pinned to region %d (prev active: %s)",
                        target_region,
                        active_instances,
                    )
                else:
                    target_region = self._calculate_cheapest_ondemand_region(strategy)
                    logger.debug(
                        "[SAFETY NET] Emergency recovery choosing cheapest region %d (prev active: %s)",
                        target_region,
                        active_instances,
                    )
                self._safety_net_consolidate_to_single_instance(
                    target_region=target_region,
                    target_type=ClusterType.ON_DEMAND,
                    reason="safety net emergency recovery",
                    active_instances=active_instances,
                    keep_target_if_exists=False,  # No existing target
                )
                return True
            raise AssertionError(
                f"Safety net latched with multiple instances: {active_instances}"
            )

        # If it is a boundary tick (needed >= remaining_time), we should consider
        # if to initialize the safety net.
        # using >= prevents the strategy from waiting one extra tick and tripping
        # the deadline check.
        is_boundary_tick = needed >= remaining_time_seconds
        logger.debug(
            "[SAFETY NET CHECK] remaining=%ss needed=%ss boundary=%s",
            remaining_time_seconds,
            needed,
            is_boundary_tick,
        )

        if not is_boundary_tick:
            return False

        # Choose target region deterministically
        if active_instances:
            target_region = list(active_instances.keys())[0]
        elif self._current_leader_region is not None:
            target_region = int(self._current_leader_region)
        else:
            target_region = 0

        leader_type = active_instances.get(target_region)

        # Determine action and consolidate instances
        if leader_type is None:
            # No instance running → emergency OD
            pinned_region = getattr(strategy, "SAFETY_NET_PIN_REGION_ID", None)
            chosen_region = (
                int(pinned_region)
                if pinned_region is not None
                else self._calculate_cheapest_ondemand_region(strategy)
            )
            logger.warning(
                "[SAFETY NET LATCHED] No instances running, emergency launch ON_DEMAND at region %d",
                chosen_region,
            )
            self._safety_net_consolidate_to_single_instance(
                target_region=chosen_region,
                target_type=ClusterType.ON_DEMAND,
                reason="safety net emergency launch",
                active_instances=active_instances,
                keep_target_if_exists=False,
            )

        else:
            # Simplified safety net: always switch to ON_DEMAND once boundary is reached
            # No need to handle complex SPOT preemption logic
            logger.warning(
                "[SAFETY NET LATCHED] Boundary reached; switching to ON_DEMAND at region %d (was %s).",
                target_region,
                leader_type.name if leader_type else "NONE",
            )
            self._safety_net_consolidate_to_single_instance(
                target_region=target_region,
                target_type=ClusterType.ON_DEMAND,
                reason="safety net boundary",
                active_instances=active_instances,
                keep_target_if_exists=(leader_type == ClusterType.ON_DEMAND),
            )

        # Latch safety net after taking action
        self._safety_net_latched = True
        return True

    def execute_multi_strategy(self, strategy: "MultiRegionStrategy"):
        """Execute one step of a multi-region strategy using yield/generator pattern."""
        from sky_spot.multi_region_types import (
            TryLaunch,
            Terminate,
            LaunchResult,
            ProbeLaunch,
        )

        # If safety net has already latched in this or previous ticks, strategy
        # actions must be skipped to preserve the override.
        if self._safety_net_latched:
            logger.debug("[SAFETY NET] Latched; skipping strategy execution this tick.")
            return

        gen = strategy._step_multi()

        try:
            action = next(gen)
            while True:
                result: Optional[LaunchResult] = None

                if isinstance(action, TryLaunch):
                    success = self._try_launch_internal(
                        action.region,
                        action.cluster_type,
                    )
                    result = LaunchResult(
                        success=success,
                        region=action.region,
                        cluster_type=action.cluster_type if success else None,
                    )
                elif isinstance(action, Terminate):
                    self._terminate_internal(
                        action.region, cluster_type=action.cluster_type
                    )
                    result = None
                elif isinstance(action, ProbeLaunch):
                    success = self._probe_internal(action.region)
                    result = LaunchResult(
                        success=success,
                        region=action.region,
                        cluster_type=ClusterType.SPOT if success else None,
                    )
                else:
                    raise ValueError(f"Unknown action type: {type(action)}")

                action = gen.send(result)

        except StopIteration:
            pass

        # Enforce single-instance invariant after strategy execution
        assert len(self.active_instances) <= 1, (
            f"Single-instance invariant violated after strategy execution at tick {self.tick}: "
            f"active_instances={self.active_instances}. "
            f"Strategy must ensure at most one instance is active after each tick."
        )

    def _probe_internal(self, region: int) -> bool:
        """Execute a probe launch: check availability and record billing if successful.

        Probes are independent of active instances and don't contribute to progress.
        They're billed for 1 minute (PROBE_BILLING_SECONDS) on success.
        """
        success = self._spot_available_in_region(region)
        if success:
            # Record billing for this tick and flag as probe for 1-minute charging
            self.current_tick_costs[region] = ClusterType.SPOT
            self.current_tick_probe_flags[region] = True
        return success

    def _try_launch_internal(
        self,
        region: int,
        cluster_type: ClusterType,
        *,
        reason: str = "",
    ) -> bool:
        """Launch immediately; returns True if launch succeeds."""
        if cluster_type == ClusterType.SPOT and not self._spot_available_in_region(
            region
        ):
            return False

        key = (region, cluster_type)

        # Check if this exact instance type is already running
        if key in self.active_instances:
            logger.debug(
                f"Duplicate {cluster_type.name} launch in region {region}; treating as no-op"
            )
            return True

        # Log if launching while other instances exist (single-instance invariant)
        if self.active_instances:
            logger.debug(
                f"Launching {cluster_type.name} in region {region} while other instances remain active; "
                f"expecting terminate later in the tick to maintain single-instance invariant"
            )

        self.active_instances[key] = self.tick  # Store launch tick
        self._launched_this_tick.add(key)  # Mark as just launched
        self.current_tick_costs[region] = cluster_type
        self.current_tick_probe_flags[region] = False
        self._new_launch_this_tick = True
        init_progress = float(self._current_leader_progress or 0.0)
        init_source = self._current_leader_region
        self._launch_info_this_tick = {
            "region": region,
            "cluster_type": cluster_type,
            "init_progress": init_progress,
            "init_source": init_source,
            "reason": reason,
        }
        if reason:
            logger.debug(f"Launched {cluster_type.name} in region {region} ({reason})")
        else:
            logger.debug(f"Launched {cluster_type.name} in region {region}")
        return True

    def _terminate_internal(
        self, region: int, cluster_type: Optional[ClusterType] = None, reason: str = ""
    ):
        """Terminate an instance immediately in the given region.

        Args:
            region: Region to terminate in
            cluster_type: Type to terminate (if None, terminates whatever is running)
            reason: Optional reason for termination
        """
        # Find what's running in this region
        if cluster_type is None:
            # Find any instance in this region
            found = False
            for (r, ct), _ in self.active_instances.items():
                if r == region:
                    cluster_type = ct
                    found = True
                    break
            if not found:
                raise ValueError(f"No instance to terminate in region {region}")

        assert cluster_type is not None
        key = (region, cluster_type)

        # Check if this instance was just launched (prevent same-tick termination)
        if key in self._launched_this_tick:
            raise ValueError(
                f"Cannot terminate {cluster_type.name} in region {region} - "
                f"it was just launched in the same tick"
            )

        # Remove from active instances if present
        if key in self.active_instances:
            self.active_instances.pop(key)
            self.current_tick_costs.pop(region, None)
            self.current_tick_probe_flags.pop(region, None)
            message = f"Terminated {cluster_type.name} in region {region}"
        else:
            # Instance not found or already terminated
            message = f"{cluster_type.name} in region {region} not found (may have been replaced)"

        if reason:
            message += f" (reason: {reason})"
        logger.debug(message)

    def _price_for_tick(
        self, sub_env, cluster_type: ClusterType, tick_index: int
    ) -> float:
        if cluster_type == ClusterType.SPOT:
            if getattr(sub_env, "_spot_price", None) is not None:
                return float(sub_env._spot_price)
            price = sub_env.trace.get_price(sub_env._start_index + tick_index)
            assert price is not None, "Spot price not available"
            return float(price)
        if cluster_type == ClusterType.ON_DEMAND:
            return float(sub_env._base_price)
        return 0.0

    def _finalize_tick_costs(self):
        """Finalize cost accounting for the previous tick."""
        tick_index = self.tick - 1

        logger.debug(
            f"Finalizing costs for tick {tick_index}: active={self.active_instances}"
        )

        # Bill for what was running during the previous tick
        # This is what's currently in active_instances (hasn't been modified yet this tick)
        billing_map: Dict[int, ClusterType] = {}
        for (region, cluster_type), launch_tick in self.active_instances.items():
            billing_map[region] = cluster_type

        billing_probe_flags: Dict[int, bool] = {}

        # Add probe charges from the previous tick
        # (probes launched during the tick that just completed)
        for region, ctype in self.current_tick_costs.items():
            if self.current_tick_probe_flags.get(region, False):
                # Probe: add to billing if not already there
                billing_map.setdefault(region, ctype)
                billing_probe_flags[region] = True

        probe_regions = {r for r, flagged in billing_probe_flags.items() if flagged}

        if billing_map:
            self.cost_history.append(billing_map.copy())
            self.cost_history_ticks.append(tick_index)
            self.cost_history_probe_flags.append(billing_probe_flags.copy())

            if probe_regions:
                self.probe_tick_count += 1
                for region in probe_regions:
                    cluster_type = billing_map.get(region)
                    if cluster_type is None:
                        continue
                    sub_env = self.envs[region]
                    price_per_hour = self._price_for_tick(
                        sub_env, cluster_type, tick_index
                    )
                    billed_seconds = (
                        PROBE_BILLING_SECONDS
                        if cluster_type == ClusterType.SPOT
                        else self.gap_seconds
                    )
                    cost = price_per_hour * billed_seconds / 3600
                    self.probe_cost_total += cost
                    self.probe_cost_by_region[region] += cost

        self.current_tick_costs = {}
        self.current_tick_probe_flags = {}

    def get_constant_cost_map(self) -> Dict[ClusterType, float]:
        # Multi-region env doesn't have a single cost map
        raise NotImplementedError(
            "Use envs[region_idx].get_constant_cost_map() for specific region"
        )

    def get_price(self) -> Dict[ClusterType, float]:
        # Multi-region env doesn't have a single price
        raise NotImplementedError(
            "Use envs[region_idx].get_price() for specific region"
        )

    @property
    def accumulated_cost(self) -> float:
        """Calculate total accumulated cost across all regions with per-tick pricing.

        For SPOT, if the trace provides a price series, we charge using the price
        at that tick; otherwise we fall back to the static inferred spot price.
        ON_DEMAND uses the region's base price.
        """
        total_cost = 0.0

        for idx, tick_costs in enumerate(self.cost_history):
            tick_index = (
                self.cost_history_ticks[idx]
                if idx < len(self.cost_history_ticks)
                else idx
            )
            probe_flags = (
                self.cost_history_probe_flags[idx]
                if idx < len(self.cost_history_probe_flags)
                else {}
            )
            for region, cluster_type in tick_costs.items():
                sub_env = self.envs[region]
                price_per_hour = self._price_for_tick(sub_env, cluster_type, tick_index)
                billed_seconds = (
                    PROBE_BILLING_SECONDS
                    if (probe_flags.get(region) and cluster_type == ClusterType.SPOT)
                    else self.gap_seconds
                )
                total_cost += price_per_hour * billed_seconds / 3600

        return total_cost

    def get_cost_breakdown(self) -> Dict[str, typing.Any]:
        """Get detailed cost breakdown by region and type with per-tick pricing.

        Also includes non-compute migration components:
        - transfer_total: accumulated S3 transfer USD for region switches
        - migration_hours_total: accumulated downtime hours (startup + transfer)
        """
        breakdown = {
            "total": self.accumulated_cost,
            "by_region": {},
            "by_type": {
                ClusterType.SPOT: 0.0,
                ClusterType.ON_DEMAND: 0.0,
                ClusterType.NONE: 0.0,
            },
            "tick_count": len(self.cost_history),
            "transfer_total": float(self.transfer_cost_total),
            "migration_hours_total": float(self.migration_hours_total),
            "probe_total": float(self.probe_cost_total),
            "probe_by_region": {
                region: float(cost)
                for region, cost in self.probe_cost_by_region.items()
            },
        }

        # Calculate costs by region using dynamic pricing
        for region in range(self.num_regions):
            region_cost = 0.0
            for idx, tick_costs in enumerate(self.cost_history):
                tick_index = (
                    self.cost_history_ticks[idx]
                    if idx < len(self.cost_history_ticks)
                    else idx
                )
                probe_flags = (
                    self.cost_history_probe_flags[idx]
                    if idx < len(self.cost_history_probe_flags)
                    else {}
                )
                if region in tick_costs:
                    cluster_type = tick_costs[region]
                    sub_env = self.envs[region]
                    price_per_hour = self._price_for_tick(
                        sub_env, cluster_type, tick_index
                    )
                    billed_seconds = (
                        PROBE_BILLING_SECONDS
                        if (
                            probe_flags.get(region) and cluster_type == ClusterType.SPOT
                        )
                        else self.gap_seconds
                    )
                    region_cost += price_per_hour * billed_seconds / 3600
            breakdown["by_region"][region] = region_cost

        # Calculate costs by type using dynamic pricing
        for idx, tick_costs in enumerate(self.cost_history):
            tick_index = (
                self.cost_history_ticks[idx]
                if idx < len(self.cost_history_ticks)
                else idx
            )
            probe_flags = (
                self.cost_history_probe_flags[idx]
                if idx < len(self.cost_history_probe_flags)
                else {}
            )
            for region, cluster_type in tick_costs.items():
                sub_env = self.envs[region]
                price_per_hour = self._price_for_tick(sub_env, cluster_type, tick_index)
                billed_seconds = (
                    PROBE_BILLING_SECONDS
                    if (probe_flags.get(region) and cluster_type == ClusterType.SPOT)
                    else self.gap_seconds
                )
                breakdown["by_type"][cluster_type] += (
                    price_per_hour * billed_seconds / 3600
                )

        return breakdown

    def info(self) -> dict:
        """Override base info to include transfer costs in total Cost and expose components.

        - Cost: compute_cost + transfer_cost_total
        - ComputeCost: compute-only accumulated instance billing
        - TransferCost: accumulated S3 transfer costs from region switches
        - MigrationHours: accumulated migration downtime hours
        """
        base = super().info()
        compute_cost = float(base.get("Cost", 0.0))
        # Compute precise downtime billed cost by aligning downtime seconds with billed instances per tick
        downtime_cost = 0.0
        for idx, tick_costs in enumerate(self.cost_history):
            if idx >= len(self.cost_history_ticks):
                continue
            tick_index = self.cost_history_ticks[idx]
            dt_seconds = float(self._downtime_seconds_by_tick.get(tick_index, 0.0))
            if dt_seconds <= 1e-9:
                continue
            for region, cluster_type in tick_costs.items():
                sub_env = self.envs[region]
                if cluster_type == ClusterType.SPOT:
                    if sub_env._spot_price is not None:
                        price_per_hour = float(sub_env._spot_price)
                    else:
                        price = sub_env.trace.get_price(
                            sub_env._start_index + tick_index
                        )
                        assert price is not None, "Spot price not available"
                        price_per_hour = float(price)
                elif cluster_type == ClusterType.ON_DEMAND:
                    assert sub_env._base_price is not None, (
                        "ON_DEMAND price not available"
                    )
                    price_per_hour = float(sub_env._base_price)
                else:
                    price_per_hour = 0.0
                downtime_cost += price_per_hour * dt_seconds / 3600.0
        base["ComputeCost"] = compute_cost
        base["TransferCost"] = float(self.transfer_cost_total)
        base["MigrationHours"] = float(self.migration_hours_total)
        base["CrossAZMigrations"] = int(self.cross_az_migrations_count)
        base["CrossRegionMigrations"] = int(self.cross_region_migrations_count)
        base["CrossContinentMigrations"] = int(self.cross_continent_migrations_count)
        base["DowntimeCost"] = float(downtime_cost)
        base["DowntimeSecondsTotal"] = float(
            sum(self._downtime_seconds_by_tick.values())
        )
        base["ProbeCost"] = float(self.probe_cost_total)
        base["ProbeCostByRegion"] = {
            str(region): float(cost)
            for region, cost in self.probe_cost_by_region.items()
        }
        base["ProbeTickCount"] = int(self.probe_tick_count)
        base["Cost"] = compute_cost + float(self.transfer_cost_total)
        return base

    @property
    def config(self) -> dict:
        return {
            "name": self.NAME,
            "trace_files": self._trace_files,
            "env_start_hours": self.envs[0]._start_index * self.gap_seconds / 3600
            if self.envs
            else 0,
            "gap_seconds": self.gap_seconds,
            "count_cross_region_migration_time": self._count_cross_region_migration_time,
        }

    @classmethod
    def _from_args(
        cls, parser: "configargparse.ArgumentParser"
    ) -> Sequence["MultiTraceEnv"]:
        """Create MultiTraceEnv instances from command line arguments."""
        group = parser.add_argument_group("MultiTraceEnv")
        group.add_argument(
            "--trace-files",
            type=str,
            nargs="+",
            help="List of trace files for multi-region simulation",
        )
        group.add_argument(
            "--env-start-hours", type=float, default=0, help="Start hours of the trace"
        )
        group.add_argument(
            "--enforce-window-bound",
            action="store_true",
            help="If set, enforce runtime window bound = deadline-hours from env_start for all regions",
        )
        group.add_argument(
            "--gang-threshold", type=int, default=None,
            help="Gang-scheduling threshold: require count >= threshold to be available (e.g., 16 for full gang)"
        )
        args, _ = parser.parse_known_args()
        count_migration_time = getattr(args, "cross_region_migration_time", False)
        gang_threshold = getattr(args, "gang_threshold", None)

        if hasattr(args, "trace_files") and args.trace_files:
            window_hours = args.deadline_hours if args.enforce_window_bound else None
            return cls.create_env(
                args.trace_files,
                args.env_start_hours,
                window_hours,
                count_cross_region_migration_time=count_migration_time,
                gang_threshold=gang_threshold,
            )
        else:
            # If no trace-files specified, fall back to single trace file if available
            if hasattr(args, "trace_file") and args.trace_file:
                # For backward compatibility, treat single trace file as single region
                window_hours = (
                    args.deadline_hours if args.enforce_window_bound else None
                )
                return [
                    cls(
                        [args.trace_file],
                        args.env_start_hours,
                        window_hours,
                        count_cross_region_migration_time=count_migration_time,
                        gang_threshold=gang_threshold,
                    )
                ]
            else:
                raise ValueError(
                    "Either --trace-files or --trace-file must be specified for multi_trace environment"
                )

    @classmethod
    def create_env(
        cls,
        trace_files_or_dirs: List[str],
        env_start_hours: float,
        window_hours: float | None = None,
        count_cross_region_migration_time: bool = False,
        gang_threshold: int | None = None,
    ) -> Sequence["MultiTraceEnv"]:
        """Create MultiTraceEnv instances from trace files or directories.

        If directories are provided, it will create one MultiTraceEnv with all json files from all directories.
        If files are provided, it will create one MultiTraceEnv with those files.
        """
        all_trace_files = []

        for path in trace_files_or_dirs:
            if os.path.isdir(path):
                # If it's a directory, add all json files from it
                for file in sorted(os.listdir(path)):
                    if file.endswith(".json"):
                        all_trace_files.append(os.path.join(path, file))
            else:
                # If it's a file, add it directly
                all_trace_files.append(path)

        if not all_trace_files:
            raise ValueError("No trace files found in the specified paths")

        # Return a single MultiTraceEnv with all trace files
        return [
            cls(
                all_trace_files,
                env_start_hours,
                window_hours,
                count_cross_region_migration_time=count_cross_region_migration_time,
                gang_threshold=gang_threshold,
            )
        ]

    def _apply_launch_overheads(
        self,
        strategy: "MultiRegionStrategy",
        per_overheads: Dict[int, float],
        active_instances: Dict[int, ClusterType],
    ) -> None:
        """Apply restart and migration accounting for launches recorded last tick."""
        info = self._launch_info_last_tick
        if info is None:
            return

        region = info["region"]
        init_source = info.get("init_source")
        restart_seconds = strategy.restart_overheads[0]
        instance_startup_hours = restart_seconds / 3600.0

        from sky_spot.migration_model import (
            get_transfer_time_hours,
            get_transfer_cost_usd,
            get_region_relationship,
        )

        transfer_hours = 0.0
        is_first_instance = init_source is None
        if not is_first_instance:
            src_name = self.get_region_name(init_source)
            dst_name = self.get_region_name(region)
            checkpoint_size_gb = strategy.task.checkpoint_size_gb
            relationship = get_region_relationship(src_name, dst_name)

            if self._count_cross_region_migration_time:
                transfer_hours = get_transfer_time_hours(
                    src_name, dst_name, checkpoint_size_gb
                )

            same_az_migration = init_source in active_instances
            if init_source != region and not same_az_migration:
                self.migration_count += 1
                if relationship == "cross_az":
                    self.cross_az_migrations_count += 1
                elif relationship == "cross_region":
                    self.cross_region_migrations_count += 1
                elif relationship == "cross_continent":
                    self.cross_continent_migrations_count += 1

                self.transfer_cost_total += float(
                    get_transfer_cost_usd(src_name, dst_name, checkpoint_size_gb)
                )

        # STATS-ONLY: report downtime hours for the current launch event
        # - Reporting metric only; not used for scheduling or billing logic
        # - Always include instance startup time
        # - Add transfer time only for migrations (i.e., not the very first instance)
        self.migration_hours_total = float(instance_startup_hours)
        if not is_first_instance:
            self.migration_hours_total += float(transfer_hours)
        per_overheads[region] = restart_seconds + transfer_hours * 3600
        logger.debug(
            f"Init region {region}: init_source={init_source}, overhead={per_overheads[region]:.1f}s"
        )

    def _assert_known_active_region(
        self,
        reference_region: int,
        active_instances: Dict[int, ClusterType],
    ) -> None:
        """Ensure any active region is accounted for in launch bookkeeping."""
        allowed = {reference_region}
        if self._launch_info_last_tick is not None:
            allowed.add(self._launch_info_last_tick["region"])
        if self._launch_info_this_tick is not None:
            allowed.add(self._launch_info_this_tick["region"])
        unknown = set(active_instances.keys()) - allowed
        assert not unknown, (
            f"Detected active regions without recorded launch: {unknown}; "
            f"prev_active={reference_region}, last_launch={self._launch_info_last_tick}, "
            f"current_launch={self._launch_info_this_tick}."
        )

    def _compute_single_tick_progress(
        self,
        strategy: "MultiRegionStrategy",
        region: int,
        per_overheads: Dict[int, float],
        total_before: float,
    ) -> float:
        """Return progress for the region while updating downtime and overhead accounting."""
        available_time = self.gap_seconds
        remaining_restart = per_overheads.get(region, 0.0)
        progress = max(available_time - remaining_restart, 0.0)

        overhead_seconds = max(available_time - float(progress), 0.0)
        if overhead_seconds > 0:
            tick_index = max(self.tick - 1, 0)
            self._downtime_seconds_by_tick[tick_index] = (
                self._downtime_seconds_by_tick.get(tick_index, 0.0)
                + float(overhead_seconds)
            )

        if region in per_overheads:
            per_overheads[region] = max(per_overheads[region] - available_time, 0.0)

        remaining_task_time = strategy.task_duration - total_before
        progress = min(progress, remaining_task_time)

        strategy.remaining_restart_overhead = per_overheads.get(region, 0.0)

        return progress

    def update_strategy_progress(self, strategy: "MultiRegionStrategy"):
        """Update task progress for multi-region strategies.

        This method:
        1. Calculates progress for the previous tick (using active_instances before preemption)
        2. Then handles preemptions for the current tick

        This design ensures that preemption at tick start doesn't lose the previous tick's progress.
        """
        # Enforce call order: observe() must be called before updating progress
        assert self.observed_tick == self.tick, (
            f"update_strategy_progress must be called after observe() for the current tick. "
            f"observed_tick={self.observed_tick}, tick={self.tick}"
        )

        # Use current active_instances for progress calculation (before preemption)
        # This represents what ran during the previous tick [i-1, i)
        current_instances = self.get_active_instances()
        # Use active_instances for progress calculation
        # At this point it still represents what ran in the previous tick
        progress_instances = dict(current_instances)

        # Progress accounting uses the snapshot (state before strategy actions this tick)
        if len(progress_instances) > 1:
            raise AssertionError(
                f"Single-instance invariant violated: active instances={progress_instances}."
            )
        # Get active regions (preemption hasn't happened yet)
        active_regions = set(r for (r, _), _ in self.active_instances.items())

        # Debug: trace progress update context (use logger at debug level)
        logger.debug(
            f"update_strategy_progress tick={self.tick} active={progress_instances} "
            f"had_new={self._had_new_launch_last_tick} last_launch_region={self._last_launch_region}"
        )

        total_before = sum(strategy.task_done_time)

        if not active_regions:
            strategy.task_done_time.append(0)
            logger.debug("No active instances, appending 0 to task_done_time")
            self._current_leader_progress = total_before
            # Handle preemptions after progress calculation
            self._handle_preemptions(strategy)
            # Check single-instance invariant after preemptions
            assert len(self.active_instances) <= 1, (
                f"Single-instance invariant violated after preemptions at tick {self.tick}: "
                f"active_instances={self.active_instances}"
            )
            return

        assert len(active_regions) == 1, (
            f"Invariant violation: multiple active regions {active_regions}"
        )
        region = next(iter(active_regions))

        # Initialize overhead tracking for this tick
        # Use current remaining_restart_overhead as the base
        per_overheads: Dict[int, float] = {region: strategy.remaining_restart_overhead}

        if self._had_new_launch_last_tick:
            self._apply_launch_overheads(strategy, per_overheads, progress_instances)
        else:
            self._assert_known_active_region(region, progress_instances)

        progress = self._compute_single_tick_progress(
            strategy,
            region,
            per_overheads,
            total_before,
        )

        strategy.task_done_time.append(progress)

        # NOW handle preemptions for the current tick (after progress calculation)
        # This ensures that work done in the previous tick is not lost due to preemption at tick start
        self._handle_preemptions(strategy)
        # Check single-instance invariant after preemptions
        assert len(self.active_instances) <= 1, (
            f"Single-instance invariant violated after preemptions at tick {self.tick}: "
            f"active_instances={self.active_instances}"
        )
        self._current_leader_region = region
        self._current_leader_progress = total_before + progress

        logger.debug(
            f"Appending task_done_time={progress}, selected_region={region}, "
            f"remaining_restart_overhead={strategy.remaining_restart_overhead}"
        )
