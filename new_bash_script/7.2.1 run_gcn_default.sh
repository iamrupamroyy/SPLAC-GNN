#!/bin/bash

# --- DEFAULT CONFIGURATION (GCN) ---
MODEL_TYPE="GCN"
TRAIN_SCRIPT="node_classification_gcn.py"
N=3
OUT_DIR="FinalLogFiles/gcn_default"
mkdir -p "$OUT_DIR"

# Parameters
RATIO=0.75
BOOST_H=10000.0
USE_FEATURE_SIM=true
BATCH_SIZE=1024
FAN_OUT="10,10,10"
MASK_AGG="majority"
FEAT_AGG="mean"
LABEL_AGG="max"
SW_MAX=10000
MODE="puregpu"

DATASETS=("pubmed" "ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")

echo "Starting $MODEL_TYPE Default Experiments..."

for ds in "${DATASETS[@]}"; do
    EXTRA_ARGS=""
    if [ "$ds" == "igb-small" ]; then EXTRA_ARGS="--path /data/Dataset/gnn_dataset"; fi
    RATIO_PCT=$(echo "$RATIO * 100" | awk '{print int($1)}')
    FO_NAME=${FAN_OUT//,/_}

    for i in $(seq 1 $N); do
        TIMESTAMP=$(date +"%H%M%S")
        OUTPUT_FILE="${OUT_DIR}/${ds}_${MODEL_TYPE}_Ratio${RATIO_PCT}_run${i}_${TIMESTAMP}.txt"
        
        {
            echo "===================================================="
            echo "MODEL:             $MODEL_TYPE"
            echo "DATASET:           $ds"
            echo "RATIO:             $RATIO"
            echo "BOOST_H:           $BOOST_H"
            echo "BATCH_SIZE:        $BATCH_SIZE"
            echo "FAN_OUT:           $FAN_OUT"
            echo "===================================================="
        } > "$OUTPUT_FILE"

        python rupam_file.py --dataset $ds -w spectral -r $RATIO --mask_agg $MASK_AGG --feature_agg $FEAT_AGG --label_agg $LABEL_AGG --boost_h $BOOST_H --sw_max $SW_MAX --use_feature_sim $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            python $TRAIN_SCRIPT --dataset $ds --mode $MODE --epoch 100 --num_layers 2 --fan_out $FAN_OUT --batch_size $BATCH_SIZE $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
        fi
        echo "    Finished $ds run $i."
    done
done
