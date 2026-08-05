#!/usr/bin/env bash
# Reproduce every figure in the paper that is backed by an artifact.
#
# Each entry logs its exit code and wall time so a partial failure is obvious
# rather than silent. Figures land in outputs/ (per-script subdirectories) and
# in the directory given as $1 (default: outputs/figures).
#
# Usage: bash artifact/reproduce_all.sh [output_dir]

set -uo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
OUT="${1:-outputs/figures}"
LOG="outputs/reproduce.log"
TIMEOUT_S="${TIMEOUT_S:-1800}"

mkdir -p "$OUT" outputs
: > "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# `timeout` is GNU coreutils and is absent on stock macOS; `gtimeout` is what
# Homebrew installs it as. Fall back to running without a time limit rather than
# failing every step with rc=127.
TIMEOUT_BIN="$(command -v timeout || command -v gtimeout || true)"
if [ -z "$TIMEOUT_BIN" ]; then
  echo "note: no timeout(1) found; running without per-step time limits." >&2
fi

fail=0
run() {
  local label="$1"; shift
  local start end rc
  start=$(date +%s)
  # shellcheck disable=SC2068
  if [ -n "$TIMEOUT_BIN" ]; then
    "$TIMEOUT_BIN" "$TIMEOUT_S" "$PY" $@ >> "$LOG" 2>&1
  else
    "$PY" $@ >> "$LOG" 2>&1
  fi
  rc=$?
  end=$(date +%s)
  if [ $rc -ne 0 ]; then
    fail=$((fail+1))
    # Surface the reason instead of burying it in the log.
    echo "  --- $label failed (rc=$rc), last lines: ---" >&2
    tail -5 "$LOG" >&2
  fi
  log "$(printf '%-32s rc=%-3s %4ss' "$label" "$rc" "$((end-start))")"
}

if [ ! -d data/converted_multi_region_aligned_h100_16_merged ]; then
  echo "error: data not prepared. Run 'bash artifact/prepare_data.sh' first." >&2
  exit 1
fi

log "=== Section 2: motivation / trace analysis ==="
run "Fig2a availability"      research/plot_availability_filled.py --no-show \
      --price-data data/converted_multi_region_aligned_h100_16_merged \
      --output "$OUT/spot_availability_filled.pdf"
run "Fig2b lifetime boxplot"  research/plot_availability_boxplot.py --no-show \
      --output "$OUT/spot_availability_boxplot.pdf"
run "Fig3a duration H100"     scripts_multi/trace_sampling/spot_duration_prediction_analysis/analyze_duration_distribution.py
run "Fig3b duration V100"     scripts_multi/trace_sampling/spot_duration_prediction_analysis/analyze_duration_distribution_v100.py
run "Fig4b migration cost"    research/migration_costs.py --summary-only

log "=== Section 5.1: end-to-end (replots shipped measurements) ==="
run "Fig7 cost comparison"    research/cost_comparison_bar.py
run "Fig8 migration timeline" research/e2e_timeline.py --output "$OUT/timeline_improved.pdf"

log "=== Section 5.2: simulation ==="
run "Fig9 traces comparison"  research/plot_traces.py --legacy --output "$OUT/traces_comparison.pdf"
run "Fig10a deadline"         research/plot_deadline_sensitivity.py \
      --input research/data/deadline_sensitivity.csv --output "$OUT/deadline_sensitivity.pdf"
run "Fig10b region scaling"   research/plot_region_scaling.py \
      --input research/data/region_scaling.csv --output "$OUT/region_scaling.pdf"

log "=== Appendix ==="
run "Fig11a availability all" research/plot_availability_filled.py --no-show --all-regions \
      --price-data data/converted_multi_region_aligned_h100_16_merged \
      --output "$OUT/spot_availability_filled.pdf"
run "Fig11b boxplot all"      research/plot_availability_boxplot.py --no-show --all-regions \
      --output "$OUT/spot_availability_boxplot.pdf"
run "Fig12 checkpoint"        research/plot_checkpoint_sensitivity.py \
      --input research/data/checkpoint_sensitivity.csv --output "$OUT/checkpoint_sensitivity.pdf"
run "Fig13 model size"        artifact/plot_model_size_vs_cost.py \
      --output "$OUT/model_size_vs_cost.pdf"
run "Fig14 probe interval"    artifact/probe_interval_ablation.py \
      --output "$OUT/probe_interval_ablation.pdf"
run "Fig15 region selection"  research/plot_region_selection.py \
      --input research/data/region_selection.csv --output "$OUT/region_selection.pdf"

# The appendix's one table rather than a figure. 192 simulations, a few minutes.
run "A.7 progress value"      artifact/v_ablation.py

log "=== Done. $fail script(s) failed. Figures in $OUT and outputs/ ==="
[ $fail -eq 0 ] || exit 1
