"""S3 transfer operations module for E2E testing.

This module handles S3 bucket transfers between regions with high-performance
transfer strategies.
"""

import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config

from sky_spot.e2e import config
from sky_spot.e2e.cluster import _get_bucket_name, sim_console

logger = logging.getLogger(__name__)


def _transfer_s3_bucket(src_index: int, dst_index: int, is_recovery: bool = True):
    """Transfer S3 bucket data between regions while preserving file structure.

    Args:
        src_index: Source region index
        dst_index: Destination region index
        is_recovery: True if recovering from preemption, False if proactive migration
    """
    src_zone_name = config.trace_files[src_index]
    dst_zone_name = config.trace_files[dst_index]
    src_bucket = _get_bucket_name(src_index)
    dst_bucket = _get_bucket_name(dst_index)

    if src_bucket == dst_bucket:
        logger.debug("Same bucket, skipping transfer (intra-region migration)")
        return

    if is_recovery:
        config._preemption_displayed_regions.add(src_index)

    sim_console.migration(src_zone_name, dst_zone_name, config.CHECKPOINT_SIZE_GB, is_recovery=is_recovery)

    _transfer_with_boto3_highconcurrency(
        src_bucket, dst_bucket, src_zone=src_zone_name, dst_zone=dst_zone_name, is_recovery=is_recovery
    )


def _transfer_with_boto3_highconcurrency(
    src_bucket: str,
    dst_bucket: str,
    src_zone: str = "",
    dst_zone: str = "",
    is_recovery: bool = True,
):
    """Transfer using AWS CLI with high parallelism while preserving original file paths.

    Uses AWS CLI subprocess calls with P=64 parallel workers.
    Based on experiments/S3_replication/README.md - achieves 7-14 Gbps.
    """
    s3 = boto3.client("s3")

    try:
        s3.delete_object(Bucket=dst_bucket, Key=".skypilot_transfer_complete")
        logger.debug("Deleted old transfer marker from %s", dst_bucket)
    except Exception:
        pass

    subprocess.run(
        ["aws", "configure", "set", "default.s3.max_concurrent_requests", "256"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["aws", "configure", "set", "default.s3.multipart_threshold", "64MB"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["aws", "configure", "set", "default.s3.multipart_chunksize", "256MB"],
        check=True,
        capture_output=True,
    )

    paginator = s3.get_paginator("list_objects_v2")
    src_objects = {}
    for page in paginator.paginate(Bucket=src_bucket):
        for obj in page.get("Contents", []):
            src_objects[obj["Key"]] = {"Size": obj["Size"]}

    if not src_objects:
        return

    dst_objects = {}
    for page in paginator.paginate(Bucket=dst_bucket):
        for obj in page.get("Contents", []):
            dst_objects[obj["Key"]] = {"Size": obj["Size"]}

    objects_to_copy = []
    skipped_count = 0
    for key, src_info in src_objects.items():
        if key == ".skypilot_transfer_complete":
            skipped_count += 1
            continue
        dst_info = dst_objects.get(key)
        if dst_info and dst_info["Size"] == src_info["Size"]:
            skipped_count += 1
        else:
            objects_to_copy.append({"Key": key, "Size": src_info["Size"]})

    if not objects_to_copy:
        if src_zone and dst_zone:
            sim_console.migration_complete(src_zone, dst_zone, 0, 0, 0, skipped_count, is_recovery=is_recovery)
        return

    copy_size = sum(obj["Size"] for obj in objects_to_copy)
    objects_to_copy.sort(key=lambda x: x["Size"], reverse=True)

    def copy_object_cli(obj):
        """Copy single object using AWS CLI."""
        key = obj["Key"]
        size = obj["Size"]
        try:
            result = subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    f"s3://{src_bucket}/{key}",
                    f"s3://{dst_bucket}/{key}",
                    "--only-show-errors",
                ],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            return (result.returncode == 0), key, size
        except Exception:
            return False, key, 0

    start_time = time.time()
    bytes_copied = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(copy_object_cli, obj) for obj in objects_to_copy]
        for future in as_completed(futures):
            success, key, size = future.result()
            if success:
                bytes_copied += size
            else:
                failed += 1
                logger.warning("Failed to copy %s", key)

    elapsed = time.time() - start_time
    speed_gbps = (bytes_copied * 8 / 1e9) / elapsed if elapsed > 0 else 0

    if failed == 0 and bytes_copied > 0:
        for obj in objects_to_copy:
            try:
                dst_head = s3.head_object(Bucket=dst_bucket, Key=obj["Key"])
                if obj["Size"] != dst_head["ContentLength"]:
                    raise RuntimeError(f"Size mismatch for {obj['Key']}")
            except Exception as e:
                raise RuntimeError(f"Validation failed: {e}")

    try:
        s3.put_object(
            Bucket=dst_bucket,
            Key=".skypilot_transfer_complete",
            Body=f"{time.time()}:{bytes_copied}:{elapsed:.1f}s".encode(),
        )
        logger.info("Transfer marker written to %s/.skypilot_transfer_complete", dst_bucket)
    except Exception as e:
        logger.warning("Failed to write transfer marker: %s", e)

    if src_zone and dst_zone:
        sim_console.migration_complete(
            src_zone,
            dst_zone,
            bytes_copied / (1024**3),
            elapsed,
            speed_gbps,
            skipped_count,
            is_recovery=is_recovery,
        )


def _transfer_with_boto3_multiprefix_fanout(
    src_bucket: str, dst_bucket: str, num_prefixes: int = 4
):
    """Transfer using boto3 with multi-prefix fan-out to avoid hot-prefix throttling.

    WARNING: This function CHANGES the file structure by distributing objects across
    prefixes (a/, b/, c/, d/). Only use when the destination application expects this
    structure.
    """
    transfer_config = Config(
        max_pool_connections=512,
        retries={"max_attempts": 10, "mode": "adaptive"},
    )
    s3 = boto3.client("s3", config=transfer_config)

    logger.info("Listing objects in %s...", src_bucket)
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=src_bucket):
        objects.extend(page.get("Contents", []))

    if not objects:
        logger.info("No objects found in %s", src_bucket)
        return

    total_size = sum(obj["Size"] for obj in objects)
    logger.info(
        "Found %d objects, total size: %.2f GB",
        len(objects),
        total_size / (1024**3),
    )

    prefixes = [chr(ord("a") + i) for i in range(num_prefixes)]

    def copy_object(obj, prefix_idx):
        """Copy single object to destination with prefix distribution."""
        src_key = obj["Key"]
        prefix = prefixes[prefix_idx % num_prefixes]
        dst_key = f"{prefix}/{src_key}"

        try:
            s3.copy_object(
                CopySource={"Bucket": src_bucket, "Key": src_key},
                Bucket=dst_bucket,
                Key=dst_key,
            )
            return True, src_key, obj["Size"]
        except Exception as e:
            logger.warning("Failed to copy %s: %s", src_key, e)
            return False, src_key, 0

    logger.info(
        "Copying %d objects with %d prefixes (P=64)...",
        len(objects),
        num_prefixes,
    )
    completed = 0
    failed = 0
    bytes_copied = 0

    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = [
            executor.submit(copy_object, obj, idx) for idx, obj in enumerate(objects)
        ]

        for future in as_completed(futures):
            success, key, size = future.result()
            if success:
                completed += 1
                bytes_copied += size
                if completed % 10 == 0:
                    logger.info(
                        "Progress: %d/%d objects (%.2f GB)",
                        completed,
                        len(objects),
                        bytes_copied / (1024**3),
                    )
            else:
                failed += 1
    logger.info(
        "Transfer summary: %d succeeded, %d failed, %.2f GB total",
        completed,
        failed,
        bytes_copied / (1024**3),
    )
