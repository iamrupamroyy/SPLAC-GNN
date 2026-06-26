#!/bin/bash

# --- DEFAULT CONFIGURATION ---
BOOST_H=10000.0
USE_FEATURE_SIM=true
FT_EPOCH=0
BATCH_SIZE=1024
FAN_OUT="10,10,10"
MASK_AGG="majority"
FEAT_AGG="mean"
LABEL_AGG="max"
SW_MAX=10000
MODE="puregpu"

N=3
OUT_DIR_BASE="FinalLogs/sweep_ratio"

# Sweep Values
RATIOS=(0.75 0.5 0.25)
DATASETS=("pubmed" "ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")

for ds in "${DATASETS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
        OUT_DIR="${OUT_DIR_BASE}/ratio_${ratio}"
        mkdir -p "$OUT_DIR"
        
        RATIO_PCT=$(echo "$ratio * 100" | awk '{print int($1)}')
        FO_NAME=${FAN_OUT//,/_}

        for i in $(seq 1 $N); do
            TIMESTAMP=$(date +"%H%M%S")
            OUTPUT_FILE="${OUT_DIR}/${ds}_Ratio${RATIO_PCT}_M-${MASK_AGG}_F-${FEAT_AGG}_L-${LABEL_AGG}_BS${BATCH_SIZE}_FO${FO_NAME}_Boost${BOOST_H}_Sim${USE_FEATURE_SIM}_run${i}_${TIMESTAMP}.txt"

            {
                echo "===================================================="
                echo "EXPERIMENT: RATIO SWEEP"
                echo "DATASET:           $ds"
                echo "RATIO:             $ratio"
                echo "BOOST_H:           $BOOST_H"
                echo "USE_FEATURE_SIM:   $USE_FEATURE_SIM"
                echo "BATCH_SIZE:        $BATCH_SIZE"
                echo "FAN_OUT:           $FAN_OUT"
                echo "MODE:              $MODE"
                echo "RUN:               $i"
                echo "===================================================="
            } > "$OUTPUT_FILE"

            EXTRA_ARGS=""
            if [ "$ds" == "igb-small" ]; then EXTRA_ARGS="--path /data/Dataset/gnn_dataset"; fi

            python src/gnn_models/rupam_file.py --dataset $ds -w spectral -r $ratio --mask_agg $MASK_AGG --feature_agg $FEAT_AGG --label_agg $LABEL_AGG --boost_h $BOOST_H --sw_max $SW_MAX --use_feature_sim $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
            
            if [ $? -eq 0 ]; then
                python src/gnn_models/node_classification.py --dataset $ds --mode $MODE --epoch $EPOCH --num_layers 2 --fan_out $FAN_OUT --batch_size $BATCH_SIZE -r $ratio --boost_h $BOOST_H --use_feature_sim $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
            fi
        done
    done
done
