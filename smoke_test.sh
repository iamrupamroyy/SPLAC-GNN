#!/bin/bash
# --- SMOKE TEST SCRIPT (Robust Version) ---

# Detect Python: Prefer 'python' if it exists, otherwise 'python3'
if command -v python &>/dev/null; then
    PY="python"
elif command -v python3 &>/dev/null; then
    PY="python3"
else
    # Fallback: if we are in a sub-shell, we might have passed $PY
    PY=${PY:-"python"}
fi

echo "Using Python: $(which $PY)"

DATASET="pubmed"
RATIO=0.75
BOOST_H=10000.0
EPOCH=100
OUT_DIR="FinalLogFiles/smoke_test"
mkdir -p "$OUT_DIR"

# Test Coarsened Models
echo "--> Testing Coarsened Pipelines..."
MODELS_COARSE=("node_classification.py" "node_classification_gcn.py" "node_classification_gat.py")
for script in "${MODELS_COARSE[@]}"; do
    LOG_FILE="${OUT_DIR}/smoke_${script%.py}.txt"
    echo "    Running $script..."
    {
        $PY rupam_file.py --dataset $DATASET -w spectral -r $RATIO --boost_h $BOOST_H --use_feature_sim --sw_max 10000 
        $PY $script --dataset $DATASET --mode puregpu --epoch $EPOCH --batch_size 1024
    } > "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then echo "    [OK] $script"; else echo "    [FAIL] $script"; fi
done

# Test Baseline Models
echo "--> Testing Baseline Pipelines..."
MODELS_BASE=("node_classificationDGL.py" "node_classificationDGL_gcn.py" "node_classificationDGL_gat.py")
for script in "${MODELS_BASE[@]}"; do
    LOG_FILE="${OUT_DIR}/smoke_${script%.py}.txt"
    echo "    Running $script..."
    {
        $PY $script --dataset $DATASET --mode puregpu --epoch $EPOCH --batch_size 1024
    } > "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then echo "    [OK] $script"; else echo "    [FAIL] $script"; fi
done
