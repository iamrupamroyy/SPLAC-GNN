#!/bin/bash

# --- BASELINE MODE SWEEP (GraphSAGE, GCN, GAT) ---
N=3
OUT_DIR_BASE="FinalLogs/baseline_mode"
mkdir -p "$OUT_DIR_BASE"

# Parameters
EPOCH=100
BATCH_SIZE=1024
FAN_OUT="10,10,10"

MODES=("puregpu" "cpu" "mixed")
DATASETS=("pubmed" "ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")
MODELS=("GraphSAGE")

echo "Starting Baseline Mode Sweep..."

for ds in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
        OUT_DIR="${OUT_DIR_BASE}/mode_${mode}"
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
                OUTPUT_FILE="${OUT_DIR}/${ds}_${model_name}_BS${BATCH_SIZE}_FO${FO_NAME}_Mode${mode}_run${i}_${TIMESTAMP}.txt"
                
                {
                    echo "===================================================="
                    echo "SCENARIO:          BASELINE_MODE_SWEEP"
                    echo "MODEL:             $model_name"
                    echo "DATASET:           $ds"
                    echo "BATCH_SIZE:        $BATCH_SIZE"
                    echo "FAN_OUT:           $FAN_OUT"
                    echo "MODE:              $mode"
                    echo "===================================================="
                } > "$OUTPUT_FILE"

                EXTRA_ARGS=""
                if [ "$ds" == "igb-small" ]; then EXTRA_ARGS="--path /data/Dataset/gnn_dataset"; fi

                python $SCRIPT --dataset $ds --mode $mode --epoch $EPOCH --fan_out $FAN_OUT --batch_size $BATCH_SIZE $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
                echo "    Finished $ds $model_name Mode=$mode run $i."
            done
        done
    done
done
