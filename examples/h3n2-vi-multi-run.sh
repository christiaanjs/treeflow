#!/usr/bin/env bash
# Run the H3N2 variational inference example multiple times with different
# seeds, so the fitted approximations/ELBOs can be compared across runs.
# Wall-clock timing for each run is appended to a CSV file.
set -euo pipefail

cd "$(dirname "$0")"

SEEDS=(1 2 3 4)
NUM_STEPS=60000
OUT_DIR=demo-out
TIMING_FILE="$OUT_DIR/h3n2-multi-run-timing.csv"

mkdir -p "$OUT_DIR"
echo "seed,num_steps,start,end,elapsed_seconds" > "$TIMING_FILE"

for seed in "${SEEDS[@]}"; do
    echo "=== Running seed ${seed} (${NUM_STEPS} iterations) ==="
    start_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    SECONDS=0

    treeflow_vi run -s "$seed" \
        -i demo-data/h3n2.fasta \
        -m h3n2-model.yaml \
        -t demo-data/h3n2.nwk \
        -n "$NUM_STEPS" \
        --learning-rate 0.001 \
        --init-values "clock_rate=0.003" \
        --trace-output "$OUT_DIR/h3n2-trace-seed${seed}.pickle" \
        --samples-output "$OUT_DIR/h3n2-samples-seed${seed}.csv" \
        --tree-samples-output "$OUT_DIR/h3n2-trees-seed${seed}.nexus" \
        --n-output-samples 1000

    elapsed=$SECONDS
    end_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "${seed},${NUM_STEPS},${start_iso},${end_iso},${elapsed}" >> "$TIMING_FILE"
done

echo "Timing written to ${TIMING_FILE}"
