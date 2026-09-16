#!/bin/bash
# Example: fan out main_workflow.py across parallel batches.
# Adjust INPUT_FILE/DATASET/MAX_PARALLEL for your own run; defaults use the
# small bundled demo sample so this script works out of the box.

SCRIPT="main_workflow.py"
PYTHON_PATH="python"
MAX_PARALLEL=16
INPUT_FILE="RW_Post_dataset/demo/demo.jsonl"
DATASET="rwpost"
BATCH_SIZE=20
TOTAL=$(wc -l < "$INPUT_FILE")
LOG_DIR="logs/run_dev"
mkdir -p "$LOG_DIR"

JOBS=0
for ((start_id=0; start_id<TOTAL; start_id+=BATCH_SIZE))
do
    end_id=$((start_id + BATCH_SIZE))
    echo "Launching with start_id = $start_id"
    nohup $PYTHON_PATH "$SCRIPT" --start_id "$start_id" --end_id "$end_id" --dataset "$DATASET" --input_file "$INPUT_FILE" > "${LOG_DIR}/log_${start_id}.txt" 2>&1 &

    ((JOBS+=1))
    if [[ $JOBS -ge $MAX_PARALLEL ]]; then
        wait -n  # wait for any one job to finish before continuing
        ((JOBS-=1))
    fi
done

wait  # wait for all remaining jobs
echo "All tasks completed."
