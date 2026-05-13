#!/bin/bash

# --- CONFIGURATION ---
N=3  # Total number of runs per dataset
OUT_DIR="BaselineDir"  # Output directory for results
mkdir -p "$OUT_DIR"

# List of all datasets to process
DATASETS=("ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")

echo "Starting sequential baseline runs (N=$N per dataset)..."
echo "Results will be saved in: $OUT_DIR"

for ds in "${DATASETS[@]}"; do
    echo "------------------------------------------------"
    echo "PROCESSING DATASET: $ds"
    echo "------------------------------------------------"

    # Determine if special path is needed (for igb-small)
    EXTRA_ARGS=""
    if [ "$ds" == "igb-small" ]; then
        EXTRA_ARGS="--path /data/Dataset/gnn_dataset"
    fi

    for i in $(seq 1 $N); do
        echo "--> [Run $i/$N] Running $ds..."
        
        # Construct the output filename using the OUT_DIR variable
        OUTPUT_FILE="${OUT_DIR}/${ds}_baseline_${i}.txt"
        
        python gnn_models/gnn_models/node_classificationDGL.py \
            --dataset "$ds" \
            --mode puregpu \
            --epoch 100 \
            --num_layers 2 \
            --batch_size 1024 \
            --fan_out 10,10,10 \
            $EXTRA_ARGS > "$OUTPUT_FILE" 2>&1
            
        echo "    Finished $ds run $i. Log: $OUTPUT_FILE"
    done
    echo "Completed all $N runs for $ds."
    echo ""
done

echo "All datasets processed! All results are in the '$OUT_DIR' folder."
