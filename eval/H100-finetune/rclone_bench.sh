#!/usr/bin/env bash
set -euo pipefail

# Simple benchmark script to measure rclone S3 upload throughput
# for a "checkpoint-style" workload: 8 large shard files.
#
# Usage (on remote VM):
#   bash rclone_bench.sh [TRANSFERS] [BUCKET_PREFIX]
# Example:
#   bash rclone_bench.sh 1
#   bash rclone_bench.sh 4 rclone-bench-qwen
#
# It will:
#   - Create 8×5GiB files under /mnt/rclone-bench/local (once)
#   - Create a unique S3 bucket
#   - rclone copy the directory to s3://BUCKET_PREFIX-<transfers>-<timestamp>/checkpoint-bench
#   - Print timing and list object sizes for a quick consistency check.

TRANSFERS="${1:-1}"
BUCKET_PREFIX="${2:-rclone-bench}"

REGION="${AWS_REGION:-us-east-2}"
TS="$(date +%Y%m%d-%H%M%S)"
BUCKET="${BUCKET_PREFIX}-${TRANSFERS}-${TS}"

echo "=== rclone bench ==="
echo "TRANSFERS = ${TRANSFERS}"
echo "REGION    = ${REGION}"
echo "BUCKET    = ${BUCKET}"
echo

# 1) Prepare local workload directory and test files
sudo mkdir -p /mnt/rclone-bench/local
sudo chown "${USER}:${USER}" -R /mnt/rclone-bench

cd /mnt/rclone-bench/local
echo "[1/3] Preparing local test files in $(pwd)"

if [ ! -f "__0_0.distcp" ]; then
  echo "Creating 8×5GiB files (this may take a bit)..."
  for i in 0 1 2 3 4 5 6 7; do
    name="__${i}_0.distcp"
    echo "  creating ${name}"
    dd if=/dev/zero of="${name}" bs=1G count=5 oflag=direct status=none
  done
else
  echo "Test files already exist, reusing."
fi

ls -lh
echo

# 2) Ensure rclone remote for S3 exists (env_auth = use instance/user creds)
echo "[2/3] Ensuring rclone remote bench-s3 exists"
rclone config create bench-s3 s3 \
  provider AWS \
  env_auth true \
  region "${REGION}" \
  >/dev/null 2>&1 || true

echo "rclone remotes:"
rclone listremotes || true
echo

# 3) Create bucket and run rclone copy with given TRANSFERS
echo "[3/3] Creating bucket and running rclone copy"
if [ "${REGION}" = "us-east-1" ]; then
  aws s3api create-bucket \
    --bucket "${BUCKET}" \
    --region "${REGION}" \
    >/dev/null
else
  aws s3api create-bucket \
    --bucket "${BUCKET}" \
    --region "${REGION}" \
    --create-bucket-configuration LocationConstraint="${REGION}" \
    >/dev/null
fi

echo "start: $(date -Ins)"
/usr/bin/time -f "ELAPSED %E" \
  rclone copy . "bench-s3:${BUCKET}/checkpoint-bench" \
    --transfers "${TRANSFERS}" \
    --s3-upload-concurrency 4 \
    --progress
echo "end:   $(date -Ins)"
echo

echo "Verifying object sizes in s3://${BUCKET}/checkpoint-bench/"
for i in 0 1 2 3 4 5 6 7; do
  aws s3 ls "s3://${BUCKET}/checkpoint-bench/__${i}_0.distcp" || true
done

echo
echo "Done. You can delete the test bucket later with:"
echo "  aws s3 rb s3://${BUCKET} --force"

