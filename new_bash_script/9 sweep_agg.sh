#!/bin/bash

# --- FLEXIBLE AGGREGATION SWEEP ---
# This script iterates through all combinations of Mask, Feature, and Label aggregation methods.

# --- 1. CONFIGURATION ---
N=3
OUT_DIR_BASE="FinalLogFiles/sweep_agg"

# DEFAULT PARAMS
RATIO=0.75
BOOST_H=10000.0
BATCH_SIZE=1024
FAN_OUT="10,10,10"
MODE="puregpu"
EPOCH=100

# --- 2. SWEEP ARRAYS (Add new methods here) ---
MASK_METHODS=("majority" "all" "any")
FEAT_METHODS=("mean")  # Add "sum" or "max" later if needed
LABEL_METHODS=("max") # Add "mode" later if needed

DATASETS=("pubmed" "ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")

echo "Starting Flexible Aggregation Sweep..."

for ds in "${DATASETS[@]}"; do
    # Handle dataset-specific paths
    EXTRA_ARGS=""
    if [ "$ds" == "igb-small" ]; then EXTRA_ARGS="--path /data/Dataset/gnn_dataset"; fi

    for m_agg in "${MASK_METHODS[@]}"; do
        for f_agg in "${FEAT_METHODS[@]}"; do
            for l_agg in "${LABEL_METHODS[@]}"; do
                
                # Define specific output subfolder for this combination
                COMBO_DIR="${OUT_DIR_BASE}/M-${m_agg}_F-${f_agg}_L-${l_agg}"
                mkdir -p "$COMBO_DIR"

                echo "--> [Agg Combo] Mask:$m_agg | Feat:$f_agg | Label:$l_agg | Dataset:$ds"

                for i in $(seq 1 $N); do
                    TIMESTAMP=$(date +"%H%M%S")
                    RATIO_PCT=$(echo "$RATIO * 100" | awk '{print int($1)}')
                    OUTPUT_FILE="${COMBO_DIR}/${ds}_Ratio${RATIO_PCT}_M-${m_agg}_F-${f_agg}_L-${l_agg}_run${i}_${TIMESTAMP}.txt"

                    {
                        echo "===================================================="
                        echo "EXPERIMENT: AGGREGATION SWEEP"
                        echo "DATASET:           $ds"
                        echo "MASK AGGREGATION:  $m_agg"
                        echo "FEATURE AGGREGATION: $f_agg"
                        echo "LABEL AGGREGATION: $l_agg"
                        echo "RATIO:             $RATIO"
                        echo "RUN:               $i"
                        echo "===================================================="
                    } > "$OUTPUT_FILE"

                    # 1. Run Coarsening
                    python rupam_file.py --dataset $ds -w spectral -r $RATIO \
                        --mask_agg $m_agg --feature_agg $f_agg --label_agg $l_agg \
                        --boost_h $BOOST_H --use_feature_sim $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
                    
                    # 2. Run Training (if coarsening succeeded)
                    if [ $? -eq 0 ]; then
                        python node_classification.py --dataset $ds --mode $MODE --epoch $EPOCH \
                            --num_layers 2 --fan_out $FAN_OUT --batch_size $BATCH_SIZE $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
                    fi
                done
            done
        done
    done
done

echo "Aggregation sweep complete. Results in $OUT_DIR_BASE"
