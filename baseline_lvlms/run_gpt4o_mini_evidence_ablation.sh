#!/bin/bash
# Evidence-quantity ablation: GPT-4o-mini, T+E setting, K = 1/2/3/5/7/9/full.
# Each K writes to its own results/gpt-4o-mini/TE_k<K>/ (or TE/ for full),
# so this is safe to re-run/resume (skips claims already in output.jsonl).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"

for k in 1 2 3 5 7 9; do
    echo "===== gpt-4o-mini / T+E / k=$k START $(date) ====="
    "$PYTHON_BIN" baseline_lvlms/run_baseline_eval.py \
        --model gpt-4o-mini --setting "T+E" --evidence_k "$k" --evidence_seed 42
    echo "===== gpt-4o-mini / T+E / k=$k END $(date) ====="
done

echo "===== gpt-4o-mini / T+E / full START $(date) ====="
"$PYTHON_BIN" baseline_lvlms/run_baseline_eval.py \
    --model gpt-4o-mini --setting "T+E"
echo "===== gpt-4o-mini / T+E / full END $(date) ====="
