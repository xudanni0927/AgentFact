#!/bin/bash
# Runs Gemini-3-Flash across the three main settings (T / T+E / T+I+E).
# Requires: GEMINI_API_KEY set (see baseline_lvlms/README.md) and
# `pip install google-genai` in whichever Python environment PYTHON_BIN
# points at. Gemini billing must be on a paid tier — the free tier's
# daily quota is far too low for a 1459-claim run.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"

for setting in "T" "T+E" "T+I+E"; do
    echo "===== gemini-3-flash / $setting START $(date) ====="
    "$PYTHON_BIN" baseline_lvlms/run_baseline_eval.py \
        --model gemini-3-flash --setting "$setting"
    echo "===== gemini-3-flash / $setting END $(date) ====="
done
