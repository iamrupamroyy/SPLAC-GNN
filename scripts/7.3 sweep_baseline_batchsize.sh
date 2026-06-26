#!/bin/bash

# --- BASELINE BATCH SIZE SWEEP (GraphSAGE, GCN, GAT) ---
N=3
OUT_DIR_BASE="FinalLogs/baseline_batchsize"
mkdir -p "$OUT_DIR_BASE"

# Parameters
EPOCH=100
MODE="puregpu"
FAN_OUT="10,10,10"

BATCH_SIZES=(1024 2048 4096 8192 16384)
DATASETS=("pubmed" "ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")
MODELS=("GraphSAGE")

echo "Starting Baseline Batch Size Sweep..."

for ds in "${DATASETS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        OUT_DIR="${OUT_DIR_BASE}/bs_${bs}"
        mkdir -p "$OUT_DIR"
        
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
                FO_NAME=${FAN_OUT//,/_}
                # Output naming: dataset_Model_BS_FO_Mode_run_time.txt
                OUTPUT_FILE="${OUT_DIR}/${ds}_${model_name}_BS${bs}_FO${FO_NAME}_Mode${MODE}_run${i}_${TIMESTAMP}.txt"
                
                {
                    echo "===================================================="
                    echo "SCENARIO:          BASELINE_BS_SWEEP"
                    echo "MODEL:             $model_name"
                    echo "DATASET:           $ds"
                    echo "BATCH_SIZE:        $bs"
                    echo "FAN_OUT:           $FAN_OUT"
                    echo "MODE:              $MODE"
                    echo "===================================================="
                } > "$OUTPUT_FILE"

                EXTRA_ARGS=""
                if [ "$ds" == "igb-small" ]; then EXTRA_ARGS="--path /data/Dataset/gnn_dataset"; fi

                python $SCRIPT --dataset $ds --mode $MODE --epoch $EPOCH --fan_out $FAN_OUT --batch_size $bs $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
                echo "    Finished $ds $model_name BS=$bs run $i."
            done
        done
    done
done
