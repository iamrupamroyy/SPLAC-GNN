#!/bin/bash

# --- DEFAULT CONFIGURATION ---
RATIO=0.75
BOOST_H=10000.0
USE_FEATURE_SIM=true
FT_EPOCH=0
BATCH_SIZE=1024
FAN_OUT="10,10,10"
MASK_AGG="majority"
FEAT_AGG="mean"
LABEL_AGG="max"
SW_MAX=10000

N=3
OUT_DIR_BASE="FinalLogs/sweep_mode"

# Sweep Values
MODES=("puregpu" "cpu" "mixed")
DATASETS=("pubmed" "ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")

for ds in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
        OUT_DIR="${OUT_DIR_BASE}/mode_${mode}"
        mkdir -p "$OUT_DIR"
        
        RATIO_PCT=$(echo "$RATIO * 100" | awk '{print int($1)}')
        FO_NAME=${FAN_OUT//,/_}

        for i in $(seq 1 $N); do
            TIMESTAMP=$(date +"%H%M%S")
            OUTPUT_FILE="${OUT_DIR}/${ds}_Ratio${RATIO_PCT}_M-${MASK_AGG}_F-${FEAT_AGG}_L-${LABEL_AGG}_BS${BATCH_SIZE}_FO${FO_NAME}_Mode${mode}_run${i}_${TIMESTAMP}.txt"

            {
                echo "===================================================="
                echo "EXPERIMENT: MODE SWEEP"
                echo "DATASET:           $ds"
                echo "MODE:              $mode"
                echo "RATIO:             $RATIO"
                echo "RUN:               $i"
                echo "===================================================="
            } > "$OUTPUT_FILE"

            EXTRA_ARGS=""
            if [ "$ds" == "igb-small" ]; then EXTRA_ARGS="--path /data/Dataset/gnn_dataset"; fi

            python rupam_file.py --dataset $ds -w spectral -r $RATIO --mask_agg $MASK_AGG --feature_agg $FEAT_AGG --label_agg $LABEL_AGG --boost_h $BOOST_H --sw_max $SW_MAX --use_feature_sim $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
            
            if [ $? -eq 0 ]; then
                python node_classification.py --dataset $ds --mode "$mode" --epoch 100 --ft_epoch $FT_EPOCH --num_layers 2 --fan_out $FAN_OUT --batch_size $BATCH_SIZE -r $RATIO --boost_h $BOOST_H --use_feature_sim $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
            fi
        done
    done
done
