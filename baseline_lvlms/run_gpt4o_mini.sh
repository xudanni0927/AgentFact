#!/bin/bash
# Runs GPT-4o-mini across all four settings (T / T+I / T+E / T+I+E).
# Requires: OPENAI_API_KEY set (see baseline_lvlms/README.md), and the
# "requests"/"python-dotenv" deps from requirements.txt installed in
# whichever Python environment PYTHON_BIN points at.
#
# Note: the T+I numbers this produces come from this framework, not from
# the earlier trial-era scripts used for the published GPT-4o-mini T+I
# baseline — same model/temperature=0, but different prompt wording, so
# expect close but not necessarily identical numbers.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"

for setting in "T" "T+I" "T+E" "T+I+E"; do
    echo "===== gpt-4o-mini / $setting START $(date) ====="
    "$PYTHON_BIN" baseline_lvlms/run_baseline_eval.py \
        --model gpt-4o-mini --setting "$setting"
    echo "===== gpt-4o-mini / $setting END $(date) ====="
done
