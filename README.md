# SPLAC-GNN

SPLAC-GNN is a pipeline for scalable graph neural network training leveraging spectral graph coarsening. It accelerates GNN training by intelligently coarsening large-scale graphs on the GPU and training models on the reduced graph representations, maintaining accuracy while dramatically reducing computational overhead.

## Project Structure

* **`src/cuda_kernels/`**: Contains the core high-performance GPU C++ (CUDA) kernels used to compute spectral weights and execute the graph coarsening algorithms.
* **`src/gnn_models/`**: Contains the Python pipeline scripts. `rupam_file.py` serves as the driver for the C++ coarsening backend, orchestrating the graph reduction. 
  * `node_classification.py` (and similar) are used for training models on the **coarsened** graphs.
  * `node_classificationDGL.py` (and similar) are used for training **baseline** models on the original, uncoarsened graphs.
* **`scripts/`**: Bash scripts orchestrating the full pipeline, from generating baseline models to running various ablation studies and parameter sweeps.

## How to Run

All scripts and commands **must be executed from the root project directory**. 

The main entry points to the pipeline are the bash scripts in the `scripts/` directory.

### 1. Run Baseline Models
To train the baseline GNN models on the original graphs, execute:
```bash
bash scripts/0\ run_baseline.sh
```

### 2. Run Hyperparameter Sweeps & Ablations
To perform parameter sweeps (e.g., varying batch sizes, layers, fan-outs) or ablation studies on the coarsening mechanism, run the corresponding scripts:
```bash
bash scripts/4\ sweep_ratio.sh
bash scripts/5\ sweep_batchsize.sh
# ... see the scripts/ directory for a full list of available experiments
```

### 3. Manual Execution
If you wish to manually test specific components:

**Coarsening a graph**:
```bash
python src/gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.5
```

**Training on the coarsened graph**:
```bash
python src/gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 100 -r 0.5
```

> **Note**: `rupam_file.py` automatically compiles the required CUDA `.cu` files using `nvcc` before execution, so ensure that the NVIDIA CUDA Toolkit is installed and available in your environment's path.

## Output
* Intermediate processed outputs and `.dgl` graphs are saved directly to the root directory.
* Training logs and experiment results are automatically saved in the `FinalLogs/` directory.
