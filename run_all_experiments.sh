#!/bin/bash

# --- Master Experiment Runner ---
# This script executes the entire experimental suite sequentially.
# Total estimated time: Very High (7 datasets x 10+ scenarios x 3 runs)

echo "===================================================="
echo "STARTING ALL EXPERIMENTS"
echo "===================================================="

# 1. Baseline
echo "--> Starting Baseline..."
./new_bash_script/run_baseline.sh

# 2. Ablation Studies
echo "--> Starting Ablation: Base..."
./new_bash_script/run_ablation_base.sh

echo "--> Starting Ablation: Boost..."
./new_bash_script/run_ablation_boost.sh

echo "--> Starting Ablation: Feature Similarity..."
./new_bash_script/run_ablation_featsim.sh

echo "--> Starting Ablation: Fine-Tuning..."
./new_bash_script/run_ablation_ft.sh

# 3. Parameter Sweeps
echo "--> Starting Sweep: Ratio..."
./new_bash_script/sweep_ratio.sh

echo "--> Starting Sweep: Batch Size..."
./new_bash_script/sweep_batchsize.sh

echo "--> Starting Sweep: Fanout..."
./new_bash_script/sweep_fanout.sh

echo "--> Starting Sweep: Aggregation..."
./new_bash_script/sweep_agg.sh

echo "--> Starting Sweep: Execution Mode..."
./new_bash_script/sweep_mode.sh

echo "===================================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "Check FinalLogFiles/ for all results."
echo "===================================================="
