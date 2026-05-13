#!/bin/bash
#SBATCH --job-name=rupamroy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=RunOutputs/rupam_%j.out
#SBATCH --error=RunOutputs/rupam_%j.err

module purge
module load cuda-12.1

export PYTHONNOUSERSITE=1

source /data/amitesh/anaconda3/etc/profile.d/conda.sh
conda activate dgl_fresh

# --- 1. CORE EXPERIMENT PARAMETERS ---
RATIO=0.50          # Coarsening ratio (0.50 = 50%)
BOOST_H=10000       # Boost parameter
SW_MAX=10000        # Spectral weight max
MASK_AGG="majority"
FEAT_AGG="mean"
LABEL_AGG="max"

# --- 2. MODEL ARCHITECTURE & TRAINING ---
NUM_LAYERS=2
FAN_OUT="10,10,10"
BATCH_SIZES=(512 1024 2048) # List of batch sizes to iterate over
EPOCH=100
N=3                 # Number of repetitions per dataset

# --- 3. DYNAMIC FOLDER NAMING & EXECUTION ---
BASE_LOG_DIR="LogFiles/BatchLogs"
# Convert ratio to percentage (e.g., 0.50 -> 50)
RATIO_PCT=$(python -c "print(int($RATIO * 100))")
# Format fan-out for the folder name (e.g., 10,10,10 -> 10x10x10)
FAN_STR=$(echo "$FAN_OUT" | tr ',' 'x')

# List of datasets to process
DATASETS=("ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr" "igb-small")

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    # Create a meaningful directory name for this batch size inside the specified base log dir
    # Format: LogFiles/BatchLogs/Exp_Ratio[Ratio]_Boost[H]_SW[Max]_L[Layers]_F[Fanout]_BS[Batch]
    OUT_DIR="${BASE_LOG_DIR}/EXP_Ratio${RATIO_PCT}_Boost${BOOST_H}_SW${SW_MAX}_L${NUM_LAYERS}_F${FAN_STR}_BS${BATCH_SIZE}"
    mkdir -p "$OUT_DIR"

    echo "=========================================================="
    echo " RUNNING EXPERIMENT: $OUT_DIR"
    echo "=========================================================="
    echo " Parameters: Ratio=${RATIO_PCT}%, Boost=${BOOST_H}, SW=${SW_MAX}"
    echo " Model: Layers=${NUM_LAYERS}, Fanout=${FAN_OUT}, Batch=${BATCH_SIZE}"
    echo "=========================================================="

    for ds in "${DATASETS[@]}"; do
        echo ">>> Processing Dataset: $ds (Batch Size: $BATCH_SIZE)"

        # Specific dataset paths
        EXTRA_ARGS=""
        if [ "$ds" == "igb-small" ]; then
            EXTRA_ARGS="--path /data/Dataset/gnn_dataset"
        fi

        for i in $(seq 1 $N); do
            echo "    Run $i/$N..."
            
            # Log file naming: Dataset_RunNumber.log (parameters are already in the folder name)
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
                --mode puregpu \
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

echo "All Experiments Complete."
