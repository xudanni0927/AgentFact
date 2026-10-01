#!/bin/bash
# Runs InternVL3.5 (local GPU inference) across the three main settings
# (T / T+E / T+I+E). Requires transformers>=4.52.1 — see
# baseline_lvlms/README.md. Set CUDA_VISIBLE_DEVICES to a free GPU on
# your machine before running (defaults to GPU 0 below).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for setting in "T" "T+E" "T+I+E"; do
    echo "===== internvl3.5 / $setting START $(date) ====="
    "$PYTHON_BIN" baseline_lvlms/run_baseline_eval.py \
        --model internvl3.5 --setting "$setting"
    echo "===== internvl3.5 / $setting END $(date) ====="
done
