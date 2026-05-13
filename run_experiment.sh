#!/bin/bash

# Function to display help
show_help() {
    echo "Usage: ./run_experiment.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model MODEL        Model to use: sage, gcn, gat (default: sage)"
    echo "  --scenario SCENARIO  Scenario: coarse, baseline (default: coarse)"
    echo "  --dataset DATASET    Dataset name (default: ogbn-arxiv)"
    echo "  --mode MODE          Execution mode: puregpu, cpu, mixed (default: puregpu)"
    echo "  --epoch EPOCH        Number of training epochs (default: 100)"
    echo "  --ratio RATIO        Retain fraction for coarsening (default: 0.5, coarse only)"
    echo "  --path PATH          Path to dataset directory"
    echo "  --num_layers N       Number of layers (default: 2)"
    echo "  --batch_size B       Batch size (default: 1024)"
    echo "  --fan_out F          Fan-out (default: 10,10,10)"
    echo "  --runs N             Number of runs (default: 1)"
    echo "  --out_dir DIR        Output directory (default: results)"
    echo "  --help               Show this help message"
}

# Default values
MODEL="sage"
SCENARIO="coarse"
DATASET="ogbn-arxiv"
MODE="puregpu"
EPOCH=100
RATIO=0.5
DATA_PATH=""
NUM_LAYERS=2
BATCH_SIZE=1024
FAN_OUT="10,10,10"
RUNS=1
OUT_DIR="results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model) MODEL="$2"; shift 2 ;;
        --scenario) SCENARIO="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --epoch) EPOCH="$2"; shift 2 ;;
        --ratio) RATIO="$2"; shift 2 ;;
        --path) DATA_PATH="$2"; shift 2 ;;
        --num_layers) NUM_LAYERS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --fan_out) FAN_OUT="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --out_dir) OUT_DIR="$2"; shift 2 ;;
        --help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

# Determine which python scripts to use
if [ "$SCENARIO" == "coarse" ]; then
    case $MODEL in
        sage) TRAIN_SCRIPT="gnn_models/node_classification.py" ;;
        gcn)  TRAIN_SCRIPT="gnn_models/node_classification_gcn.py" ;;
        gat)  TRAIN_SCRIPT="gnn_models/node_classification_gat.py" ;;
        *) echo "Invalid model for coarse scenario: $MODEL"; exit 1 ;;
    esac
elif [ "$SCENARIO" == "baseline" ]; then
    case $MODEL in
        sage) TRAIN_SCRIPT="gnn_models/node_classificationDGL.py" ;;
        gcn)  TRAIN_SCRIPT="gnn_models/node_classificationDGL_gcn.py" ;;
        gat)  TRAIN_SCRIPT="gnn_models/node_classificationDGL_gat.py" ;;
        *) echo "Invalid model for baseline scenario: $MODEL"; exit 1 ;;
    esac
else
    echo "Invalid scenario: $SCENARIO. Use 'coarse' or 'baseline'."
    exit 1
fi

# Add path argument if provided
EXTRA_ARGS=""
if [ -n "$DATA_PATH" ]; then
    EXTRA_ARGS="--path $DATA_PATH"
fi

echo "-------------------------------------------------------"
echo "EXPERIMENT CONFIGURATION:"
echo "Model:    $MODEL"
echo "Scenario: $SCENARIO"
echo "Dataset:  $DATASET"
echo "Runs:     $RUNS"
echo "Script:   $TRAIN_SCRIPT"
echo "-------------------------------------------------------"

for i in $(seq 1 $RUNS); do
    echo "--> [Run $i/$RUNS] Starting..."
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${OUT_DIR}/${DATASET}_${SCENARIO}_${MODEL}_run${i}_${TIMESTAMP}.txt"
    
    if [ "$SCENARIO" == "coarse" ]; then
        echo "    Running Coarsening (ratio $RATIO)..."
        python gnn_models/rupam_file.py \
            --dataset "$DATASET" \
            -w spectral \
            -r "$RATIO" \
            $EXTRA_ARGS > "$LOG_FILE" 2>&1
        
        # Check if coarsening succeeded
        if [ $? -ne 0 ]; then
            echo "    [ERROR] Coarsening failed. See $LOG_FILE"
            continue
        fi
        echo "    Coarsening done. Starting Training..."
    else
        echo "    Starting Baseline Training..."
        > "$LOG_FILE" # Clear/create log file
    fi

    # Run the training script
    python "$TRAIN_SCRIPT" \
        --dataset "$DATASET" \
        --mode "$MODE" \
        --epoch "$EPOCH" \
        --num_layers "$NUM_LAYERS" \
        --fan_out "$FAN_OUT" \
        --batch_size "$BATCH_SIZE" \
        $EXTRA_ARGS >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "    Finished Run $i. Log: $LOG_FILE"
        # Extract metrics from log and print to console
        VAL_ACC=$(grep -i "BEST_VAL_ACC" "$LOG_FILE" | awk '{print $NF}')
        TEST_ACC=$(grep -i "FINAL_TEST_ACC" "$LOG_FILE" | awk '{print $NF}')
        # Also check for DGL script format metrics
        if [ -z "$VAL_ACC" ]; then VAL_ACC=$(grep -i "Best Validation Accuracy" "$LOG_FILE" | awk '{print $NF}'); fi
        if [ -z "$TEST_ACC" ]; then TEST_ACC=$(grep -i "Final Test Accuracy" "$LOG_FILE" | awk '{print $NF}'); fi
        
        echo "    Results: Val Acc = $VAL_ACC, Test Acc = $TEST_ACC"
    else
        echo "    [ERROR] Training failed. See $LOG_FILE"
    fi
done

echo "Done! All logs are in '$OUT_DIR'."
