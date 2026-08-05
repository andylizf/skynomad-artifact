#!/usr/bin/env python3
"""Fetch A100:8 (p4d.24xlarge) spot placement scores and prices across regions."""

import boto3
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# A100:8 on AWS
INSTANCE_TYPE = "p4d.24xlarge"

# Regions with p4d.24xlarge spot availability (from AWS console)
REGIONS = [
    "ap-northeast-2",  # Seoul - score 9
    "us-east-2",       # Ohio - score 9
    "us-east-1",       # N. Virginia - score 9
    "eu-north-1",      # Stockholm - score 9
    "us-west-2",       # Oregon - score 8
    "eu-west-1",       # Ireland - score 6
    "ca-central-1",    # Canada Central - score 4
    "eu-west-2",       # London - score 1
    "ap-northeast-1",  # Tokyo - score 1
    "sa-east-1",       # Sao Paulo - score 1
]


def get_spot_price(region: str) -> dict:
    """Get current spot price for p4d.24xlarge in a region."""
    try:
        ec2 = boto3.client("ec2", region_name=region)
        response = ec2.describe_spot_price_history(
            InstanceTypes=[INSTANCE_TYPE],
            ProductDescriptions=["Linux/UNIX"],
            MaxResults=10,
        )

        if response["SpotPriceHistory"]:
            # Get the most recent price per AZ
            prices = {}
            for item in response["SpotPriceHistory"]:
                az = item["AvailabilityZone"]
                if az not in prices:
                    prices[az] = {
                        "price": float(item["SpotPrice"]),
                        "timestamp": item["Timestamp"].isoformat(),
                    }
            return {"region": region, "status": "ok", "prices": prices}
        else:
            return {"region": region, "status": "no_data", "prices": {}}
    except Exception as e:
        return {"region": region, "status": "error", "error": str(e), "prices": {}}


def get_spot_placement_score(regions: list[str]) -> dict:
    """Get spot placement scores for p4d.24xlarge across regions.

    Note: This API is called from a single region but evaluates all specified regions.
    """
    try:
        # Use us-east-1 as the API endpoint (it evaluates all regions)
        ec2 = boto3.client("ec2", region_name="us-east-1")

        response = ec2.get_spot_placement_scores(
            InstanceTypes=[INSTANCE_TYPE],
            TargetCapacity=1,
            RegionNames=regions,
            SingleAvailabilityZone=False,
        )

        scores = {}
        for item in response.get("SpotPlacementScores", []):
            region = item.get("Region")
            az = item.get("AvailabilityZoneId")
            score = item.get("Score")

            key = az if az else region
            scores[key] = score

        return {"status": "ok", "scores": scores}
    except Exception as e:
        return {"status": "error", "error": str(e), "scores": {}}


def main():
    print(f"Fetching spot info for {INSTANCE_TYPE} (A100:8)")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # Fetch spot prices in parallel
    print("\n[1] Fetching spot prices per region...")
    price_results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_spot_price, r): r for r in REGIONS}
        for future in as_completed(futures):
            result = future.result()
            price_results[result["region"]] = result

    # Fetch placement scores
    print("[2] Fetching spot placement scores...")
    score_result = get_spot_placement_score(REGIONS)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Collect all AZ data for sorting
    all_az_data = []
    for region in REGIONS:
        price_data = price_results.get(region, {})

        if price_data.get("status") in ("error", "no_data"):
            continue

        prices = price_data.get("prices", {})
        for az, info in prices.items():
            score = score_result.get("scores", {}).get(az, score_result.get("scores", {}).get(region, 0))
            all_az_data.append({
                "region": region,
                "az": az,
                "price": info["price"],
                "score": score if isinstance(score, int) else 0,
            })

    # Sort: low score first, then high price first (expensive + unavailable at top)
    all_az_data.sort(key=lambda x: (x["score"], -x["price"]))

    # Table header
    print(f"\n{'Region':<20} {'AZ':<15} {'Price ($/hr)':<15} {'Score':<10}")
    print("-" * 60)

    for item in all_az_data:
        print(f"{item['region']:<20} {item['az']:<15} ${item['price']:<14.4f} {item['score']:<10}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find cheapest
    all_prices = []
    for region, data in price_results.items():
        for az, info in data.get("prices", {}).items():
            all_prices.append((az, info["price"]))

    if all_prices:
        all_prices.sort(key=lambda x: x[1])
        print("\nTop 5 cheapest AZs:")
        for az, price in all_prices[:5]:
            print(f"  {az}: ${price:.4f}/hr")

    # Placement score summary
    if score_result.get("status") == "ok":
        scores = score_result.get("scores", {})
        high_score = [(k, v) for k, v in scores.items() if v >= 7]
        print(f"\nHigh placement score (>=7): {len(high_score)} regions/AZs")
        for k, v in sorted(high_score, key=lambda x: -x[1])[:5]:
            print(f"  {k}: {v}")
    else:
        print(f"\nPlacement score error: {score_result.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
