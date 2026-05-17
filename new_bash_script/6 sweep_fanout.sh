#!/bin/bash

# --- DEFAULT CONFIGURATION ---
RATIO=0.75
BOOST_H=10000.0
USE_FEATURE_SIM=true
FT_EPOCH=0
BATCH_SIZE=1024
MASK_AGG="majority"
FEAT_AGG="mean"
LABEL_AGG="max"
SW_MAX=10000
MODE="puregpu"

N=3
OUT_DIR_BASE="FinalLogFiles/sweep_fanout"

# Sweep Values
FAN_OUTS=("10,10,10" "15,15,15" "20,20,20")
DATASETS=("pubmed" "ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")

for ds in "${DATASETS[@]}"; do
    for fo in "${FAN_OUTS[@]}"; do
        FO_NAME=${fo//,/_}
        OUT_DIR="${OUT_DIR_BASE}/fanout_${FO_NAME}"
        mkdir -p "$OUT_DIR"
        
        RATIO_PCT=$(echo "$RATIO * 100" | awk '{print int($1)}')

        for i in $(seq 1 $N); do
            TIMESTAMP=$(date +"%H%M%S")
            OUTPUT_FILE="${OUT_DIR}/${ds}_Ratio${RATIO_PCT}_M-${MASK_AGG}_F-${FEAT_AGG}_L-${LABEL_AGG}_BS${BATCH_SIZE}_FO${FO_NAME}_Boost${BOOST_H}_Sim${USE_FEATURE_SIM}_run${i}_${TIMESTAMP}.txt"

            {
                echo "===================================================="
                echo "EXPERIMENT: FANOUT SWEEP"
                echo "DATASET:           $ds"
                echo "FAN_OUT:           $fo"
                echo "RATIO:             $RATIO"
                echo "MODE:              $MODE"
                echo "RUN:               $i"
                echo "===================================================="
            } > "$OUTPUT_FILE"

            EXTRA_ARGS=""
            if [ "$ds" == "igb-small" ]; then EXTRA_ARGS="--path /data/Dataset/gnn_dataset"; fi

            python rupam_file.py --dataset $ds -w spectral -r $RATIO --mask_agg $MASK_AGG --feature_agg $FEAT_AGG --label_agg $LABEL_AGG --boost_h $BOOST_H --sw_max $SW_MAX --use_feature_sim $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
            
            if [ $? -eq 0 ]; then
                python node_classification.py --dataset $ds --mode $MODE --epoch 100 --ft_epoch $FT_EPOCH --num_layers 2 --fan_out "$fo" --batch_size $BATCH_SIZE $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
            fi
        done
    done
done
