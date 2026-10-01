#!/bin/bash
# Runs Qwen3-VL on the T+I setting only (image + text, no evidence).
# See run_qwen3_vl.sh for the main T/T+E/T+I+E run.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "===== qwen3-vl / T+I START $(date) ====="
"$PYTHON_BIN" baseline_lvlms/run_baseline_eval.py \
    --model qwen3-vl --setting "T+I"
echo "===== qwen3-vl / T+I END $(date) ====="
