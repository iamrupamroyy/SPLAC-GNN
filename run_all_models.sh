#!/bin/bash

# --- RUN ALL MODELS SCRIPT ---
# Purpose: Execute all GNN models and capture output logs in finalLogs
# Models: 6 classification models + rupam_file.py

# Configuration
DATASET="pubmed"
RATIO=0.75
BOOST_H=10000.0
USE_FEATURE_SIM=true
EPOCH=2
BATCH_SIZE=1024
LOG_DIR="finalLogs"

# Create log directory in root
mkdir -p "$LOG_DIR"

echo "===================================================="
echo "STARTING ALL MODELS EXECUTION (Dataset: $DATASET)"
echo "Log directory: $LOG_DIR"
echo "===================================================="

# Navigate to gnn_models directory
cd gnn_models || { echo "Error: gnn_models directory not found"; exit 1; }

# 1. Run Coarsening Engine (rupam_file.py)
echo "--> Step 0: Running rupam_file.py..."
python rupam_file.py --dataset $DATASET -w spectral -r $RATIO --boost_h $BOOST_H --use_feature_sim --sw_max 10000 > "../$LOG_DIR/rupam_file.log" 2>&1
if [ $? -eq 0 ]; then
    echo "[OK] rupam_file.py finished."
else
    echo "[FAIL] rupam_file.py failed. Check $LOG_DIR/rupam_file.log"
    # We continue even if coarsening fails, though subsequent models might fail too
fi

# 2. List of models to run
COARSED_MODELS=("node_classification.py" "node_classification_gcn.py" "node_classification_gat.py")
BASELINE_MODELS=("node_classificationDGL.py" "node_classificationDGL_gcn.py" "node_classificationDGL_gat.py")

echo ""
echo "--> Running Coarsened Models..."
for script in "${COARSED_MODELS[@]}"; do
    echo "    Running $script..."
    python $script --dataset $DATASET --mode puregpu --epoch $EPOCH --batch_size $BATCH_SIZE > "../$LOG_DIR/${script%.py}.log" 2>&1
    if [ $? -eq 0 ]; then
        echo "    [OK] $script"
    else
        echo "    [FAIL] $script. Check $LOG_DIR/${script%.py}.log"
    fi
done

echo ""
echo "--> Running Baseline Models..."
for script in "${BASELINE_MODELS[@]}"; do
    echo "    Running $script..."
    python $script --dataset $DATASET --mode puregpu --epoch $EPOCH --batch_size $BATCH_SIZE > "../$LOG_DIR/${script%.py}.log" 2>&1
    if [ $? -eq 0 ]; then
        echo "    [OK] $script"
    else
        echo "    [FAIL] $script. Check $LOG_DIR/${script%.py}.log"
    fi
done

echo ""
echo "===================================================="
echo "EXECUTION COMPLETE"
echo "Logs are available in the '$LOG_DIR' directory."
echo "===================================================="
