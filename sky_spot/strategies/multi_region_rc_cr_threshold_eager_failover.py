"""
Uniform Progress with Eager Failover.

Like multi_region_rc_cr_threshold, but after preemption, tries OTHER regions first
before falling back to the original region. Based on SkyPilot's EAGER_NEXT_REGION
strategy observation that preemption is likely to happen again shortly in the same region.

This implementation blocks at the REGION level (not zone level), matching SkyPilot's behavior.
E.g., if preempted from us-west-2a, all zones in us-west-2 (2a, 2b, 2c, 2d) are deprioritized.
"""

import logging
import re
import typing
from collections import defaultdict

from sky_spot.strategies.multi_region_rc_cr_threshold import (
    MultiRegionRCCRThresholdStrategy,
)

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task


def _extract_cloud_region_from_zone(zone_name: str) -> str:
    """Extract cloud region from availability zone (supports AWS and GCP).

    Examples:
        AWS:  us-west-2a      -> us-west-2
              us-east-1b      -> us-east-1
              ap-northeast-1c -> ap-northeast-1
        GCP:  us-central1-a   -> us-central1
              us-west1-b      -> us-west1
              europe-west1-c  -> europe-west1
    """
    # GCP style: region and zone separated by hyphen (us-central1-a -> us-central1)
    # GCP regions don't have hyphen before digit: us-central1, europe-west1
    gcp_match = re.match(r"^(.+\d+)-[a-z]$", zone_name)
    if gcp_match:
        return gcp_match.group(1)

    # AWS style: zone letter directly after digit (us-west-2a -> us-west-2)
    # AWS regions have hyphen before digit: us-west-2, us-east-1
    aws_match = re.match(r"^(.+-\d+)[a-z]$", zone_name)
    if aws_match:
        return aws_match.group(1)

    # Fallback: return as-is
    return zone_name


def _extract_zone_from_trace_path(trace_file: str) -> typing.Optional[str]:
    """Extract zone name from trace file path.

    Example: data/real/ping_based/random_start_time/us-west-2a_k80_1/0.json -> us-west-2a
    """
    from pathlib import Path

    try:
        parent_name = Path(trace_file).parent.name  # e.g., "us-west-2a_k80_1"
        # Split by underscore, first part is zone
        parts = parent_name.split("_")
        if parts:
            return parts[0]  # e.g., "us-west-2a"
    except Exception:
        pass
    return None


class MultiRegionRCCRThresholdEagerFailoverStrategy(MultiRegionRCCRThresholdStrategy):
    NAME = "multi_region_rc_cr_threshold_eager_failover"

    def reset(self, env: "env.Env", task: "task.Task"):
        super().reset(env, task)

        # Build zone index -> AWS region mapping
        self._zone_to_aws_region: typing.Dict[int, str] = {}
        self._zone_idx_to_name: typing.Dict[
            int, str
        ] = {}  # idx -> zone name (e.g., "us-west-2a")
        self._aws_region_to_zones: typing.Dict[str, typing.List[int]] = defaultdict(
            list
        )

        multi_env = typing.cast("env.MultiTraceEnv", self.env)
        if hasattr(multi_env, "_trace_files"):
            for idx, trace_file in enumerate(multi_env._trace_files):
                zone_name = _extract_zone_from_trace_path(trace_file)
                if zone_name:
                    self._zone_idx_to_name[idx] = zone_name
                    aws_region = _extract_cloud_region_from_zone(zone_name)
                    self._zone_to_aws_region[idx] = aws_region
                    self._aws_region_to_zones[aws_region].append(idx)

    def _get_spot_region_order(self, num_regions: int) -> typing.List[int]:
        """Order regions by price, then deprioritize the last preempted AWS region.

        Strategy:
        1. Sort all zones by current spot price (cheapest first)
        2. Move zones in the last preempted AWS region to the end

        Rationale: If we just got preempted from a zone, the entire AWS region
        is likely still under resource pressure. Try cheaper regions first,
        and avoid the preempted region until other options are exhausted.
        """
        # Step 1: Sort by spot price
        multi_env = typing.cast("env.MultiTraceEnv", self.env)
        prices = multi_env.get_all_regions_spot_prices()
        spot_prices = {i: p if p is not None else float("inf") for i, p in enumerate(prices)}
        all_zones = sorted(
            range(num_regions), key=lambda z: spot_prices.get(z, float("inf"))
        )

        # Step 2: Deprioritize last preempted region
        last_zone = self._last_active_region
        if last_zone is None:
            logger.info(
                "[EAGER_FAILOVER] No preemption history. Order by price: %s",
                [self._zone_idx_to_name.get(z, f"zone_{z}") for z in all_zones[:5]],
            )
            return all_zones

        # Get the AWS region of the last active zone
        aws_region = self._zone_to_aws_region.get(last_zone)
        if aws_region is None:
            # Fallback to zone-level if we can't determine region
            if last_zone in all_zones:
                all_zones.remove(last_zone)
                all_zones.append(last_zone)
            return all_zones

        # Get all zones in the same AWS region
        zones_to_deprioritize = set(
            self._aws_region_to_zones.get(aws_region, [last_zone])
        )

        # Reorder: other zones first (still sorted by price), then zones in preempted region
        other_zones = [z for z in all_zones if z not in zones_to_deprioritize]
        deprioritized_zones = [z for z in all_zones if z in zones_to_deprioritize]

        # Helper to get zone name for logging
        def _name(idx: int) -> str:
            return self._zone_idx_to_name.get(idx, f"zone_{idx}")

        def _price(idx: int) -> str:
            p = spot_prices.get(idx, float("inf"))
            return f"${p:.3f}" if p != float("inf") else "N/A"

        last_zone_name = _name(last_zone)
        other_info = [f"{_name(z)}({_price(z)})" for z in other_zones[:5]]
        deprioritized_info = [f"{_name(z)}({_price(z)})" for z in deprioritized_zones]

        logger.info(
            "[EAGER_FAILOVER] Deprioritizing AWS region %s after preemption from %s. "
            "Try first (by price): %s%s, then: %s",
            aws_region,
            last_zone_name,
            other_info,
            "..." if len(other_zones) > 5 else "",
            deprioritized_info,
        )

        return other_zones + deprioritized_zones
