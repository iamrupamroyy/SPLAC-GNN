#!/bin/bash
#SBATCH --job-name=rupamroy_mode
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=RunOutputs/mode_%j.out
#SBATCH --error=RunOutputs/mode_%j.err

module purge
module load cuda-12.1

export PYTHONNOUSERSITE=1

source /data/amitesh/anaconda3/etc/profile.d/conda.sh
conda activate dgl_fresh

# --- 1. CORE EXPERIMENT PARAMETERS ---
RATIO=0.75          # Default coarsening ratio set to 75%
BOOST_H=10000       # Boost parameter
SW_MAX=10000        # Spectral weight max
MASK_AGG="majority"
FEAT_AGG="mean"
LABEL_AGG="max"

# --- MODES TO SWEEP ---
MODES=("cpu" "mixed" "puregpu")

# --- 2. MODEL ARCHITECTURE & TRAINING ---
NUM_LAYERS=2
FAN_OUT="10,10,10"
BATCH_SIZE=1024
EPOCH=100
N=3                 # Number of repetitions per dataset

# --- 3. DYNAMIC FOLDER NAMING & EXECUTION ---
BASE_LOG_DIR="LogFiles/ModeLogs"
RATIO_PCT=$(python -c "print(int($RATIO * 100))")
FAN_STR=$(echo "$FAN_OUT" | tr ',' 'x')

# List of datasets to process
DATASETS=("ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")

for MODE in "${MODES[@]}"; do
    # Create a meaningful directory name for this mode
    # Format: LogFiles/ModeLogs/Exp_Ratio[Ratio]_Mode[Mode]_L[Layers]_F[Fanout]_BS[Batch]
    OUT_DIR="${BASE_LOG_DIR}/EXP_Ratio${RATIO_PCT}_Mode-${MODE}_L${NUM_LAYERS}_F${FAN_STR}_BS${BATCH_SIZE}"
    mkdir -p "$OUT_DIR"

    echo "=========================================================="
    echo " RUNNING EXPERIMENT: $OUT_DIR"
    echo "=========================================================="
    echo " Mode: $MODE"
    echo " Parameters: Ratio=${RATIO_PCT}%, Boost=${BOOST_H}, SW=${SW_MAX}"
    echo " Model: Layers=${NUM_LAYERS}, Fanout=${FAN_OUT}, Batch=${BATCH_SIZE}"
    echo "=========================================================="

    for ds in "${DATASETS[@]}"; do
        echo ">>> Processing Dataset: $ds (Mode: $MODE)"

        # Specific dataset paths
        EXTRA_ARGS=""
        if [ "$ds" == "igb-small" ]; then
            EXTRA_ARGS="--path /data/Dataset/gnn_dataset"
        fi

        for i in $(seq 1 $N); do
            echo "    Run $i/$N..."
            
            # Log file naming
            OUTPUT_FILE="${OUT_DIR}/${ds}_run${i}.log"

            # Execute Pipeline
            (python gnn_models/rupam_file.py \
                --dataset "$ds" \
                -w spectral \
                -r "$RATIO" \
                --mask_agg "$MASK_AGG" \
                --feature_agg "$FEAT_AGG" \
                --label_agg "$LABEL_AGG" \
                --boost_h "$BOOST_H" \
                --sw_max "$SW_MAX" \
                $EXTRA_ARGS && \
             python gnn_models/node_classification.py \
                --dataset "$ds" \
                --mode "$MODE" \
                --epoch "$EPOCH" \
                --num_layers "$NUM_LAYERS" \
                --fan_out "$FAN_OUT" \
                --batch_size "$BATCH_SIZE" \
                $EXTRA_ARGS) > "$OUTPUT_FILE" 2>&1
                
            echo "    Finished $ds run $i. Log: $OUTPUT_FILE"
        done
        echo "----------------------------------------------------------"
    done
done

echo "All Mode Sweep Experiments Complete."
