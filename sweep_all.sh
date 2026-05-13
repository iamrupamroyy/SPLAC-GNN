#!/bin/bash

# This script runs all 6 variants (SAGE/GCN/GAT for both Coarsened and Baseline scenarios)
# for a specific dataset.

DATASET=${1:-ogbn-arxiv}
RATIO=${2:-0.5}
EPOCH=${3:-50}

echo "======================================================="
echo "RUNNING FULL COMPARISON FOR DATASET: $DATASET"
echo "Ratio: $RATIO, Epochs: $EPOCH"
echo "======================================================="

MODELS=("sage" "gcn" "gat")
SCENARIOS=("baseline" "coarse")

for scenario in "${SCENARIOS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo ">>> Executing: $model ($scenario)"
        ./run_experiment.sh \
            --model "$model" \
            --scenario "$scenario" \
            --dataset "$DATASET" \
            --ratio "$RATIO" \
            --epoch "$EPOCH" \
            --runs 1 \
            --out_dir "thesis_results"
    done
done

echo "======================================================="
echo "All experiments completed. Check 'thesis_results' folder."
echo "======================================================="
