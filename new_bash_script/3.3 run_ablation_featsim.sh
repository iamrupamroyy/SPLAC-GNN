#!/bin/bash

# --- CONFIGURATION ---
# Scenario: Feature Similarity (Feature Sim enabled, No Boost, No Fine-Tuning)
BOOST_H=0.0
USE_FEATURE_SIM=true
FT_EPOCH=0

N=3              # Number of runs per dataset
OUT_DIR="FinalLogFiles/ablation_featsim"
mkdir -p "$OUT_DIR"

# Parameters
RATIO=0.75
MASK_AGG="majority"
FEAT_AGG="mean"
LABEL_AGG="mode"
SW_MAX=1000000
EPOCH=100
BATCH_SIZE=1024
FAN_OUT="10,10,10"
MODE="puregpu"

# List of datasets
DATASETS=("pubmed" "ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")

echo "Starting Ablation Study [FEATSIM] (N=$N runs per dataset)..."
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

    RATIO_PCT=$(echo "$RATIO * 100" | awk '{print int($1)}')

    for i in $(seq 1 $N); do
        TIMESTAMP=$(date +"%H%M%S")
        FO_NAME=${FAN_OUT//,/_}
        OUTPUT_FILE="${OUT_DIR}/${ds}_Ratio${RATIO_PCT}_M-${MASK_AGG}_F-${FEAT_AGG}_L-${LABEL_AGG}_BS${BATCH_SIZE}_FO${FO_NAME}_Boost${BOOST_H}_Sim${USE_FEATURE_SIM}_FT${FT_EPOCH}_run${i}_${TIMESTAMP}.txt"

        echo "--> [Run $i/$N] Feature Sim Coarsening and Training $ds..."
        
        # Start Log File with Argument Summary
        {
            echo "===================================================="
            echo "ABLATION SCENARIO: FEATSIM"
            echo "DATASET:           $ds"
            echo "RATIO:             $RATIO"
            echo "BOOST_H:           $BOOST_H"
            echo "USE_FEATURE_SIM:   $USE_FEATURE_SIM"
            echo "FT_EPOCH:          $FT_EPOCH"
            echo "MASK_AGG:          $MASK_AGG"
            echo "FEAT_AGG:          $FEAT_AGG"
            echo "LABEL_AGG:         $LABEL_AGG"
            echo "SW_MAX:            $SW_MAX"
            echo "EPOCH:             $EPOCH"
            echo "BATCH_SIZE:        $BATCH_SIZE"
            echo "FAN_OUT:           $FAN_OUT"
            echo "MODE:              $MODE"
            echo "RUN:               $i"
            echo "TIMESTAMP:         $(date)"
            echo "===================================================="
            echo ""
        } > "$OUTPUT_FILE"

        # 1. Run Coarsening
        COARSEN_CMD="python rupam_file.py --dataset $ds -w spectral -r $RATIO --mask_agg $MASK_AGG --feature_agg $FEAT_AGG --label_agg $LABEL_AGG --boost_h $BOOST_H --sw_max $SW_MAX $EXTRA_ARGS"
        if [ "$USE_FEATURE_SIM" = true ]; then
            COARSEN_CMD="$COARSEN_CMD --use_feature_sim"
        fi
        
        echo "Executing Coarsening: $COARSEN_CMD" >> "$OUTPUT_FILE"
        $COARSEN_CMD >> "$OUTPUT_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            # 2. Run Classification
            TRAIN_CMD="python node_classification.py --dataset $ds --mode $MODE --epoch $EPOCH --ft_epoch $FT_EPOCH --num_layers 2 --fan_out $FAN_OUT --batch_size $BATCH_SIZE $EXTRA_ARGS"
            echo "Executing Training: $TRAIN_CMD" >> "$OUTPUT_FILE"
            $TRAIN_CMD >> "$OUTPUT_FILE" 2>&1
        else
            echo "ERROR: Coarsening failed for $ds" >> "$OUTPUT_FILE"
        fi
            
        echo "    Finished $ds run $i. Log: $OUTPUT_FILE"
    done
    echo "Completed $ds."
    echo ""
done

echo "Feature Sim ablation complete! Results are in '$OUT_DIR'."
