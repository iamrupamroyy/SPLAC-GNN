#!/bin/bash

# --- CONFIGURATION ---
# Scenario: Baseline (Training on Original Graph)
N=3              # Number of runs per dataset
OUT_DIR="FinalLogs/baseline"
mkdir -p "$OUT_DIR"

# Parameters
EPOCH=100
MODE="puregpu"
NUM_LAYERS=2
FAN_OUT="10,10,10"
BATCH_SIZE=1024

# List of datasets
DATASETS=("pubmed" "ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")

echo "Starting Baseline Training (N=$N runs per dataset)..."
echo "Results will be saved in: $OUT_DIR"

for ds in "${DATASETS[@]}"; do
    echo "------------------------------------------------"
    echo "PROCESSING DATASET: $ds"
    echo "------------------------------------------------"

    # Handle dataset-specific paths
    EXTRA_ARGS=""
    if [ "$ds" == "igb-small" ]; then
        EXTRA_ARGS="--path /data/Dataset/gnn_dataset"
    fi

    for i in $(seq 1 $N); do
        TIMESTAMP=$(date +"%H%M%S")
        FO_NAME=${FAN_OUT//,/_}
        # For baseline, ratio is effectively 100%
        OUTPUT_FILE="${OUT_DIR}/${ds}_Baseline_Ratio100_BS${BATCH_SIZE}_FO${FO_NAME}_Mode${MODE}_run${i}_${TIMESTAMP}.txt"

        echo "--> [Run $i/$N] Baseline Training $ds..."
        
        # Start Log File with Argument Summary
        {
            echo "===================================================="
            echo "SCENARIO: BASELINE (Original Graph)"
            echo "DATASET:           $ds"
            echo "EPOCH:             $EPOCH"
            echo "MODE:              $MODE"
            echo "NUM_LAYERS:        $NUM_LAYERS"
            echo "FAN_OUT:           $FAN_OUT"
            echo "BATCH_SIZE:        $BATCH_SIZE"
            echo "RUN:               $i"
            echo "TIMESTAMP:         $(date)"
            echo "===================================================="
            echo ""
        } > "$OUTPUT_FILE"

        # Run Classification on Original Graph
        TRAIN_CMD="python node_classificationDGL.py --dataset $ds --mode $MODE --epoch $EPOCH --num_layers $NUM_LAYERS --fan_out $FAN_OUT --batch_size $BATCH_SIZE $EXTRA_ARGS"
        
        echo "Executing Baseline Training: $TRAIN_CMD" >> "$OUTPUT_FILE"
        $TRAIN_CMD >> "$OUTPUT_FILE" 2>&1
            
        echo "    Finished $ds run $i. Log: $OUTPUT_FILE"
    done
    echo "Completed $ds."
    echo ""
done

echo "Baseline experiments complete! Results are in '$OUT_DIR'."
