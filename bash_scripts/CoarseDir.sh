#!/bin/bash

# --- CONFIGURATION ---
N=3              # Number of runs per dataset
OUT_DIR="CoarseDir"
mkdir -p "$OUT_DIR"

# Parameters
RATIO=0.75
MASK_AGG="majority"
FEAT_AGG="mean"
LABEL_AGG="max"
BOOST_H=0
SW_MAX=10000
EPOCH=100

# List of datasets
DATASETS=("ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")

echo "Starting Coarsening + Training Pipeline (N=$N runs per dataset)..."
echo "Results will be saved in: $OUT_DIR"

for ds in "${DATASETS[@]}"; do
    echo "------------------------------------------------"
    echo "PROCESSING DATASET: $ds"
    echo "------------------------------------------------"

    # Handle IGB-small path and dataset name mapping
    EXTRA_ARGS=""
    if [ "$ds" == "igb-small" ]; then
        EXTRA_ARGS="--path /data/Dataset/gnn_dataset"
    fi

    for i in $(seq 1 $N); do
        echo "--> [Run $i/$N] Coarsening and Training $ds..."
        
        # Construct specific output filename
        # Format: DatasetName_RatioVal_MaskVal_FeatureVal_LabelVal_BoostBoostVal_maxMaxVal_i.txt
        OUTPUT_FILE="${OUT_DIR}/${ds}_${RATIO}_${MASK_AGG}_${FEAT_AGG}_${LABEL_AGG}_Boost${BOOST_H}_max${SW_MAX}_${i}.txt"

        # 1. Run Coarsening (gnn_models/gnn_models/rupam_file.py)
        # 2. Run Classification (gnn_models/gnn_models/node_classification.py)
        # We use '&&' to ensure training only starts if coarsening succeeds.
        (python gnn_models/gnn_models/rupam_file.py \
            --dataset "$ds" \
            -w spectral \
            -r "$RATIO" \
            --mask_agg "$MASK_AGG" \
            --feature_agg "$FEAT_AGG" \
            --label_agg "$LABEL_AGG" \
            --boost_h "$BOOST_H" \
            --sw_max "$SW_MAX" \
            $EXTRA_ARGS && \
         python gnn_models/gnn_models/node_classification.py \
            --dataset "$ds" \
            --mode puregpu \
            --epoch "$EPOCH" \
            --num_layers 2 \
            --fan_out 10,10,10 \
            --batch_size 1024 \
            $EXTRA_ARGS) > "$OUTPUT_FILE" 2>&1
            
        echo "    Finished $ds run $i. Log: $OUTPUT_FILE"
    done
    echo "Completed $ds."
    echo ""
done

echo "Pipeline complete! Results are in '$OUT_DIR'."
