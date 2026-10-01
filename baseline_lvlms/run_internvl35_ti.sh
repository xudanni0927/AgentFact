#!/bin/bash
# Runs InternVL3.5 on the T+I setting only (image + text, no evidence).
# See run_internvl35.sh for the main T/T+E/T+I+E run.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "===== internvl3.5 / T+I START $(date) ====="
"$PYTHON_BIN" baseline_lvlms/run_baseline_eval.py \
    --model internvl3.5 --setting "T+I"
echo "===== internvl3.5 / T+I END $(date) ====="
