#!/bin/bash

# --- BASELINE FAN-OUT SWEEP (GraphSAGE, GAT, GCN) ---
N=3
OUT_DIR_BASE="FinalLogs/baseline_fanout"
mkdir -p "$OUT_DIR_BASE"

# Parameters
EPOCH=100
MODE="puregpu"
BATCH_SIZE=1024

# Sweep Values
FAN_OUTS=("10,10,10" "15,15,15" "20,20,20")
DATASETS=("pubmed" "ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")
MODELS=("GraphSAGE")

echo "Starting Baseline Fan-out Sweep..."

for ds in "${DATASETS[@]}"; do
    for fo in "${FAN_OUTS[@]}"; do
        FO_NAME=${fo//,/_}
        OUT_DIR="${OUT_DIR_BASE}/fo_${FO_NAME}"
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
                # Output naming: dataset_Model_BS_FO_Mode_run_time.txt
                OUTPUT_FILE="${OUT_DIR}/${ds}_${model_name}_BS${BATCH_SIZE}_FO${FO_NAME}_Mode${MODE}_run${i}_${TIMESTAMP}.txt"
                
                {
                    echo "===================================================="
                    echo "SCENARIO:          BASELINE_FO_SWEEP"
                    echo "MODEL:             $model_name"
                    echo "DATASET:           $ds"
                    echo "BATCH_SIZE:        $BATCH_SIZE"
                    echo "FAN_OUT:           $fo"
                    echo "MODE:              $MODE"
                    echo "===================================================="
                } > "$OUTPUT_FILE"

                EXTRA_ARGS=""
                if [ "$ds" == "igb-small" ]; then EXTRA_ARGS="--path /data/Dataset/gnn_dataset"; fi

                python $SCRIPT --dataset $ds --mode $MODE --epoch $EPOCH --fan_out $fo --batch_size $BATCH_SIZE $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
                echo "    Finished $ds $model_name FO=$fo run $i."
            done
        done
    done
done
