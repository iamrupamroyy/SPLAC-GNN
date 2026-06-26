#!/bin/bash

# --- BASELINE EXPERIMENTS (GraphSAGE, GCN, GAT) ---
N=3
OUT_DIR="FinalLogs/baseline_models"
mkdir -p "$OUT_DIR"

# Parameters
EPOCH=100
MODE="puregpu"
FAN_OUT="10,10,10"
BATCH_SIZE=1024

DATASETS=("pubmed" "ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")
MODELS=("GraphSAGE" "GCN" "GAT")

echo "Starting Baseline GraphSAGE/GCN/GAT Experiments..."

for ds in "${DATASETS[@]}"; do
    EXTRA_ARGS=""
    if [ "$ds" == "igb-small" ]; then EXTRA_ARGS="--path /data/Dataset/gnn_dataset"; fi
    FO_NAME=${FAN_OUT//,/_}

    for model_name in "${MODELS[@]}"; do
        if [ "$model_name" == "GraphSAGE" ]; then 
            SCRIPT="node_classificationDGL.py"
        elif [ "$model_name" == "GCN" ]; then 
            SCRIPT="node_classificationDGL_gcn.py"
        else 
            SCRIPT="node_classificationDGL_gat.py"
        fi

        for i in $(seq 1 $N); do
            TIMESTAMP=$(date +"%H%M%S")
            # Output naming: dataset_Model_BS_FO_Mode_run_time.txt
            OUTPUT_FILE="${OUT_DIR}/${ds}_${model_name}_BS${BATCH_SIZE}_FO${FO_NAME}_Mode${MODE}_run${i}_${TIMESTAMP}.txt"
            
            {
                echo "===================================================="
                echo "SCENARIO:          BASELINE"
                echo "MODEL:             $model_name"
                echo "DATASET:           $ds"
                echo "BATCH_SIZE:        $BATCH_SIZE"
                echo "FAN_OUT:           $FAN_OUT"
                echo "MODE:              $MODE"
                echo "===================================================="
            } > "$OUTPUT_FILE"

            python $SCRIPT --dataset $ds --mode $MODE --epoch $EPOCH --fan_out $FAN_OUT --batch_size $BATCH_SIZE $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
            echo "    Finished $ds $model_name run $i."
        done
    done
done
