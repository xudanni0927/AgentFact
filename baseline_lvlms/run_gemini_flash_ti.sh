#!/bin/bash
# Runs Gemini-3-Flash on the T+I setting only (image + text, no evidence).
# See run_gemini_flash.sh for the main T/T+E/T+I+E run.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "===== gemini-3-flash / T+I START $(date) ====="
"$PYTHON_BIN" baseline_lvlms/run_baseline_eval.py \
    --model gemini-3-flash --setting "T+I"
echo "===== gemini-3-flash / T+I END $(date) ====="
