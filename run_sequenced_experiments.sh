#!/bin/bash

# --- MASTER SEQUENCED RUNNER ---
# This script runs all experiments in the numerical order requested.

echo "===================================================="
echo "STARTING SEQUENCED EXPERIMENTAL SUITE"
echo "===================================================="

# 0. Baseline (GraphSAGE)
echo "[0] Running Baseline..."
"./new_bash_script/0 run_baseline.sh"

# 3. Ablation Studies
echo "[3.1] Running Ablation: Base..."
"./new_bash_script/3.1 run_ablation_base.sh"

echo "[3.2] Running Ablation: Boost..."
"./new_bash_script/3.2 run_ablation_boost.sh"

echo "[3.3] Running Ablation: Feature Sim..."
"./new_bash_script/3.3 run_ablation_featsim.sh"

echo "[3.4] Running Ablation: Fine-Tuning..."
"./new_bash_script/3.4 run_ablation_ft.sh"

# 4-6. Parameter Sweeps
echo "[4] Running Sweep: Ratio..."
"./new_bash_script/4 sweep_ratio.sh"

echo "[5] Running Sweep: Batch Size..."
"./new_bash_script/5 sweep_batchsize.sh"

echo "[6] Running Sweep: Fanout..."
"./new_bash_script/6 sweep_fanout.sh"

# 7. Model Baselines & Defaults
echo "[7.1] Running Baseline Models (GCN/GAT)..."
"./new_bash_script/7.1 run_baseline_models.sh"

echo "[7.2.1] Running GCN Defaults..."
"./new_bash_script/7.2.1 run_gcn_default.sh"

echo "[7.2.2] Running GAT Defaults..."
"./new_bash_script/7.2.2 run_gat_default.sh"

# 9-10. Advanced Sweeps
echo "[9] Running Sweep: Aggregation..."
"./new_bash_script/9 sweep_agg.sh"

echo "[10] Running Sweep: Execution Mode..."
"./new_bash_script/10 sweep_mode.sh"

echo "===================================================="
echo "ALL SEQUENCED EXPERIMENTS COMPLETE"
echo "Check FinalLogFiles/ for results."
echo "===================================================="
