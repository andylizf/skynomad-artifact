"""
Dynamic migration time model based on empirical S3 transfer measurements.

Based on real-world tests conducted across AWS regions and availability zones,
this module provides realistic migration time estimates for ML checkpoints.

IMPORTANT TRANSFER TIME SEMANTICS:
===================================
1. Same-region transfers (same-zone or cross-AZ): Return 0.0
   - The baseline S3 download time is already included in restart_overhead
   - This avoids double-counting the overhead

2. Cross-region/continent transfers: Return ADDITIONAL time only
   - Calculate actual transfer time based on measured speeds
   - Subtract the baseline same-region transfer time (9.72 Gbps)
   - Return only the extra overhead beyond what's in restart_overhead

This design ensures backward compatibility while providing accurate migration costs.
The restart_overhead parameter represents the total cold start time including
baseline S3 download. The transfer_time represents only additional network latency.
"""

__all__ = [
    "parse_region_info",
    "get_region_relationship",
    "get_transfer_time_hours",
    "get_transfer_cost_usd",
]

import re
from typing import Tuple


def parse_region_info(region_name: str) -> Tuple[str, str, str]:
    """
    Parse region string to extract region, zone, and instance type.

    Supports both AWS-style names (e.g., 'us-east-1a_v100_1') and GCP-style names
    where the zone is separated by a hyphen (e.g., 'europe-west1-c_h100_8').
    """
    parts = region_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid region format: {region_name}")
    region_zone = parts[0]
    instance_type = parts[1]

    if not region_zone:
        raise ValueError(f"Invalid region format: {region_name}")
    zone = region_zone[-1]
    if not zone.isalpha():
        raise ValueError(f"Invalid zone in region name: {region_name}")
    region = region_zone[:-1]
    if region.endswith("-"):
        region = region[:-1]
    if not region:
        raise ValueError(f"Invalid region format: {region_name}")
    return region, zone, instance_type


def get_region_relationship(source_region: str, dest_region: str) -> str:
    """
    Determine the relationship between two regions.

    Returns one of:
    - 'same_zone': Same region and zone
    - 'cross_az': Same region, different zone
    - 'cross_region': Different regions in same continent
    - 'cross_continent': Different continents
    """
    src_region, src_zone, _ = parse_region_info(source_region)
    dst_region, dst_zone, _ = parse_region_info(dest_region)

    if src_region == dst_region and src_zone == dst_zone:
        return "same_zone"
    elif src_region == dst_region:
        return "cross_az"
    else:
        # Extract continent from region (first part before hyphen)
        src_continent = src_region.split("-")[0]
        dst_continent = dst_region.split("-")[0]

        if src_continent == dst_continent:
            return "cross_region"
        else:
            return "cross_continent"


def get_transfer_speed_mbps(relationship: str) -> float:
    """
    Get expected transfer speed based on geographic relationship.

    Based on AWS S3 empirical measurements (experiments/s3_transfer_performance.png):
    - Same region (same zone/cross-AZ): 9.72 Gbps
    - Cross region: 8.20 Gbps
    - Cross continent: 3.59 Gbps
    """
    # Speed based on real AWS S3 measurements (convert Gbps to Mbps)
    speeds_mbps = {
        "same_zone": 9720,  # 9.72 Gbps
        "cross_az": 9720,  # Same as same_zone for S3
        "cross_region": 8200,  # 8.20 Gbps
        "cross_continent": 3590,  # 3.59 Gbps
    }

    return speeds_mbps[relationship]


def get_transfer_time_hours(
    source_region: str, dest_region: str, checkpoint_size_gb: float
) -> float:
    """
    Calculate data transfer time between regions.

    Args:
        source_region: Source region string (e.g., 'us-east-1a_v100_1')
        dest_region: Destination region string
        checkpoint_size_gb: Size of checkpoint in GB

    Returns:
        Transfer time in hours (not including instance startup)
    """
    # Determine geographic relationship
    relationship = get_region_relationship(source_region, dest_region)

    # Same zone or cross-AZ within same region: return 0 for backward compatibility
    # Note: The S3 download time within same region is assumed to be
    # included in the restart_overhead (cold start time), not as a
    # separate transfer time. This avoids double-counting the overhead.
    # Since same_zone and cross_az have the same speed (9720 Mbps), they're treated equally.
    if relationship in ["same_zone", "cross_az"]:
        return 0.0

    # Get transfer speed for cross-region/continent transfers
    speed_mbps = get_transfer_speed_mbps(relationship)

    # Calculate transfer time for cross-region/continent transfers
    # These represent the ADDITIONAL time beyond the baseline same-region transfer
    # Convert GB to Mb: GB * 8 * 1024 = Mb
    transfer_time_seconds = (checkpoint_size_gb * 8 * 1024) / speed_mbps

    # Subtract the baseline same-region transfer time (9720 Mbps)
    # since that's already included in restart_overhead
    baseline_seconds = (checkpoint_size_gb * 8 * 1024) / 9720
    additional_seconds = max(0, transfer_time_seconds - baseline_seconds)
    additional_hours = additional_seconds / 3600

    return additional_hours


def get_transfer_cost_usd(
    source_region: str, dest_region: str, checkpoint_size_gb: float
) -> float:
    """
    Calculate S3-to-S3 transfer cost based on real AWS pricing data.

    Uses actual AWS "Data Transfer OUT From Amazon S3 To [Region]" pricing
    verified through AWS Pricing API analysis (November 2024).

    Pricing structure (S3-to-S3):
    - Same region (same zone or cross-AZ): $0.00/GB
    - us-east-1 ↔ us-east-2: $0.01/GB (special discounted rate)
    - Most standard inter-region transfers: $0.02/GB
    - GovCloud regions: $0.03/GB
    - Canada West to Europe: $0.05/GB
    - Asia Pacific inter-region: $0.08/GB
    - Middle East region transfers: $0.085-0.11/GB
    - Australia/New Zealand: $0.098/GB
    - South America: $0.138/GB
    - Africa: $0.147/GB

    Based on comprehensive AWS Pricing API analysis of 10,000+ region pairs.
    """
    relationship = get_region_relationship(source_region, dest_region)

    # Free transfers within same region
    if relationship in ["same_zone", "cross_az"]:
        return 0.0

    # Extract AWS region codes (e.g., 'us-east-1' from 'us-east-1a_v100_1')
    src_region, _, _ = parse_region_info(source_region)
    dst_region, _, _ = parse_region_info(dest_region)

    # US East special pricing (verified via AWS API)
    if (src_region == "us-east-1" and dst_region == "us-east-2") or (
        src_region == "us-east-2" and dst_region == "us-east-1"
    ):
        return checkpoint_size_gb * 0.01

    # GovCloud transfers (higher base rate)
    if "gov" in src_region or "gov" in dst_region:
        return checkpoint_size_gb * 0.03

    # Region-based pricing (verified patterns from AWS Pricing API)

    # North America (US/Canada Central) - standard rate to most destinations
    if src_region in [
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
        "ca-central-1",
    ]:
        return checkpoint_size_gb * 0.02

    # Canada West - higher rates to certain regions
    if src_region == "ca-west-1":
        if dst_region.startswith("eu-"):
            return checkpoint_size_gb * 0.05
        elif dst_region.startswith(("me-", "af-", "il-")):
            return checkpoint_size_gb * 0.14
        else:
            return checkpoint_size_gb * 0.02

    # Mexico Central
    if src_region == "mx-central-1":
        if dst_region.startswith(("me-", "af-", "il-")):
            return checkpoint_size_gb * 0.14
        else:
            return checkpoint_size_gb * 0.02

    # Europe - standard rate with exceptions
    if src_region.startswith("eu-"):
        if dst_region in ["ca-west-1", "mx-central-1"]:
            return checkpoint_size_gb * 0.05
        elif dst_region == "ap-southeast-7":
            return checkpoint_size_gb * 0.08
        else:
            return checkpoint_size_gb * 0.02

    # Asia Pacific regions - higher inter-asia rates
    if src_region.startswith("ap-"):
        if dst_region.startswith("ap-"):
            # Asia-to-Asia pricing: $0.08/GB most common (33.1%), ranges $0.08-0.10/GB
            return checkpoint_size_gb * 0.08
        elif dst_region.startswith(("me-", "af-", "il-")):
            return checkpoint_size_gb * 0.11
        else:
            return checkpoint_size_gb * 0.02

    # Middle East regions
    if src_region.startswith("me-"):
        return checkpoint_size_gb * 0.085

    # South America
    if src_region.startswith("sa-"):
        return checkpoint_size_gb * 0.138

    # Africa
    if src_region.startswith("af-"):
        return checkpoint_size_gb * 0.147

    # Israel Central
    if src_region == "il-central-1":
        if dst_region.startswith("ap-"):
            return checkpoint_size_gb * 0.15
        else:
            return checkpoint_size_gb * 0.02

    # Default fallback for unknown patterns: standard inter-region rate
    return checkpoint_size_gb * 0.02


if __name__ == "__main__":
    # Test the module
    test_cases = [
        ("us-east-1a_v100_1", "us-east-1a_v100_1", 100),  # Same zone
        ("us-east-1a_v100_1", "us-east-1c_v100_1", 100),  # Cross-AZ
        ("us-east-1a_v100_1", "us-west-2a_v100_1", 100),  # Cross-region
        (
            "us-west-2a_v100_1",
            "eu-west-1a_v100_1",
            100,
        ),  # Cross-continent (hypothetical)
    ]

    print("Migration Time Model Test Results:")
    print("=" * 80)

    for src, dst, size_gb in test_cases:
        try:
            relationship = get_region_relationship(src, dst)
            transfer_hours = get_transfer_time_hours(src, dst, size_gb)
            cost_usd = get_transfer_cost_usd(src, dst, size_gb)

            # Example with 0.2 hour (12 minute) instance startup
            instance_startup_hours = 0.2
            total_hours = instance_startup_hours + transfer_hours

            print(f"\nSource: {src}")
            print(f"Dest:   {dst}")
            print(f"Size:   {size_gb} GB")
            print(f"Type:   {relationship}")
            print(
                f"Transfer: {transfer_hours:.2f} hours ({transfer_hours * 60:.1f} minutes)"
            )
            print(
                f"Total (w/ {instance_startup_hours}h startup): {total_hours:.2f} hours ({total_hours * 60:.1f} minutes)"
            )
            print(f"Cost:   ${cost_usd:.2f}")
        except ValueError as e:
            print(f"\nError processing {src} -> {dst}: {e}")
