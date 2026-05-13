#!/bin/bash
#SBATCH --job-name=baseline_dgl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=RunOutputs/baseline_%j.out
#SBATCH --error=RunOutputs/baseline_%j.err

module purge
module load cuda-12.1

export PYTHONNOUSERSITE=1

source /data/amitesh/anaconda3/etc/profile.d/conda.sh
conda activate dgl_fresh

# --- 1. MODEL ARCHITECTURE & TRAINING ---
NUM_LAYERS=2
FAN_OUT="10,10,10"
BATCH_SIZE=1024
EPOCH=100
N=3                 # Number of repetitions per dataset

# --- 2. DYNAMIC FOLDER NAMING ---
# Format fan-out for the folder name (e.g., 10,10,10 -> 10x10x10)
FAN_STR=$(echo "$FAN_OUT" | tr ',' 'x')

# Create a meaningful directory name for Baselines
# Format: BASELINE_L[Layers]_F[Fanout]_BS[Batch]
OUT_DIR="BASELINE_L${NUM_LAYERS}_F${FAN_STR}_BS${BATCH_SIZE}"
mkdir -p "$OUT_DIR"

# List of datasets to process
DATASETS=("ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")

echo "=========================================================="
echo " RUNNING BASELINE: $OUT_DIR"
echo "=========================================================="
echo " Model: Layers=${NUM_LAYERS}, Fanout=${FAN_OUT}, Batch=${BATCH_SIZE}"
echo "=========================================================="

for ds in "${DATASETS[@]}"; do
    echo ">>> Processing Dataset: $ds"

    # Specific dataset paths
    EXTRA_ARGS=""
    if [ "$ds" == "igb-small" ]; then
        EXTRA_ARGS="--path /data/Dataset/gnn_dataset"
    fi

    for i in $(seq 1 $N); do
        echo "    Run $i/$N..."
        
        # Log file naming: Dataset_RunNumber.log
        OUTPUT_FILE="${OUT_DIR}/${ds}_run${i}.log"

        # Execute Baseline Training
        python gnn_models/node_classificationDGL.py \
            --dataset "$ds" \
            --mode puregpu \
            --epoch "$EPOCH" \
            --num_layers "$NUM_LAYERS" \
            --fan_out "$FAN_OUT" \
            --batch_size "$BATCH_SIZE" \
            $EXTRA_ARGS > "$OUTPUT_FILE" 2>&1
            
        echo "    Finished $ds run $i. Log: $OUTPUT_FILE"
    done
    echo "----------------------------------------------------------"
done

echo "Baseline Experiment Complete. All logs found in: $OUT_DIR"
