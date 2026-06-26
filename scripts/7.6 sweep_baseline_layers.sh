#!/bin/bash

# --- BASELINE LAYER SWEEP (GraphSAGE, GCN, GAT) ---
N=3
OUT_DIR_BASE="FinalLogs/baseline_layers"
mkdir -p "$OUT_DIR_BASE"

# Parameters
EPOCH=100
MODE="puregpu"
FAN_OUT="10,10,10,10,10" # Sufficiently long for up to 5 layers
BATCH_SIZE=1024

DATASETS=("pubmed" "ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")
MODELS=("GraphSAGE")
LAYERS=(2 3 4 5)

echo "Starting Baseline Layer Sweep (GraphSAGE/GCN/GAT)..."

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

        for nl in "${LAYERS[@]}"; do
            OUT_DIR="${OUT_DIR_BASE}/layers_${nl}"
            mkdir -p "$OUT_DIR"

            for i in $(seq 1 $N); do
                TIMESTAMP=$(date +"%H%M%S")
                # Output naming: dataset_Model_Layers_BS_FO_Mode_run_time.txt
                OUTPUT_FILE="${OUT_DIR}/${ds}_${model_name}_L${nl}_BS${BATCH_SIZE}_FO${FO_NAME}_Mode${MODE}_run${i}_${TIMESTAMP}.txt"
                
                {
                    echo "===================================================="
                    echo "SCENARIO:          BASELINE"
                    echo "MODEL:             $model_name"
                    echo "DATASET:           $ds"
                    echo "NUM_LAYERS:        $nl"
                    echo "BATCH_SIZE:        $BATCH_SIZE"
                    echo "FAN_OUT:           $FAN_OUT"
                    echo "MODE:              $MODE"
                    echo "===================================================="
                } > "$OUTPUT_FILE"

                python $SCRIPT --dataset $ds --mode $MODE --epoch $EPOCH --num_layers $nl --fan_out $FAN_OUT --batch_size $BATCH_SIZE $EXTRA_ARGS >> "$OUTPUT_FILE" 2>&1
                echo "    Finished $ds $model_name Layer $nl run $i."
            done
        done
    done
done

echo "Baseline layer sweep complete! Results are in '$OUT_DIR_BASE'."
