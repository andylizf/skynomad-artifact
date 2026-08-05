#!/usr/bin/env bash
# Prepare all trace data needed to reproduce the paper's figures.
#
# Everything this script needs already ships inside the repo; there are no
# network downloads and no cloud credentials involved.
#
# Three steps:
#   1. Unpack the merged 13-zone H100 traces (used by the multi-region figures).
#   2. Regenerate the V100 aligned traces from the raw 2023 availability CSVs.
#      This step matters: it produces the per-region full.json files that the
#      lifetime-distribution analysis reads. Shipping only the numbered window
#      files would silently change the result, because the windows are random
#      overlapping samples of one trace rather than the trace itself.
#
# Usage: bash artifact/prepare_data.sh

set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [ ! -x "$PY" ]; then
  echo "error: $PY not found. Run 'uv sync' first (see README)." >&2
  exit 1
fi

log "Step 1/2: unpacking merged H100 traces (13 GCP zones)"
if [ -d data/converted_multi_region_aligned_h100_16_merged ]; then
  log "  already unpacked, skipping"
else
  tar -xzf data/converted_multi_region_aligned_h100_16_merged.tar.gz -C data/
  log "  unpacked $(ls data/converted_multi_region_aligned_h100_16_merged | wc -l | tr -d ' ') entries"
fi

log "Step 2/2: regenerating V100 aligned traces from raw CSVs"
if [ -f data/converted_multi_region_aligned/us-west-2a_v100_1/full.json ]; then
  log "  already generated, skipping"
else
  "$PY" scripts_multi/trace_sampling/create_aligned_traces.py
fi

n_full=$(find data/converted_multi_region_aligned -name full.json 2>/dev/null | wc -l | tr -d ' ')
log "  V100 regions with full.json: $n_full (expected 9)"

if [ "$n_full" -ne 9 ]; then
  echo "error: expected 9 full.json files, got $n_full" >&2
  exit 1
fi

log "Data preparation complete."
