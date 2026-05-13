#!/bin/bash
#SBATCH --job-name=rupam_roy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=output/rupam_%j.out
#SBATCH --error=output/rupam_%j.err

module purge
module load cuda-12.1

source /data/amitesh/dgl-env/bin/activate

# --- OGBN-Arxiv ---
# Ratio 0.25
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.25 --sw_max 50 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_25_sw50.txt
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.25 --sw_max 100 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_25_sw100.txt
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.25 --sw_max 1000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_25_sw1000.txt
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.25 --sw_max 5000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_25_sw5000.txt
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.25 --sw_max 10000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_25_sw10000.txt

# Ratio 0.5
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.5 --sw_max 50 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_50_sw50.txt
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.5 --sw_max 100 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_50_sw100.txt
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.5 --sw_max 1000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_50_sw1000.txt
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.5 --sw_max 5000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_50_sw5000.txt
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.5 --sw_max 10000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_50_sw10000.txt

# Ratio 0.75
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.75 --sw_max 50 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_75_sw50.txt
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.75 --sw_max 100 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_75_sw100.txt
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.75 --sw_max 1000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_75_sw1000.txt
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.75 --sw_max 5000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_75_sw5000.txt
(python gnn_models/rupam_file.py --dataset ogbn-arxiv -w spectral -r 0.75 --sw_max 10000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-arxiv --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnarxiv_75_sw10000.txt

# --- OGBN-Products ---
# Ratio 0.25
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.25 --sw_max 50 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_25_sw50.txt
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.25 --sw_max 100 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_25_sw100.txt
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.25 --sw_max 1000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_25_sw1000.txt
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.25 --sw_max 5000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_25_sw5000.txt
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.25 --sw_max 10000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_25_sw10000.txt

# Ratio 0.5
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.5 --sw_max 50 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_50_sw50.txt
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.5 --sw_max 100 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_50_sw100.txt
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.5 --sw_max 1000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_50_sw1000.txt
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.5 --sw_max 5000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_50_sw5000.txt
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.5 --sw_max 10000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_50_sw10000.txt

# Ratio 0.75
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.75 --sw_max 50 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_75_sw50.txt
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.75 --sw_max 100 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_75_sw100.txt
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.75 --sw_max 1000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_75_sw1000.txt
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.75 --sw_max 5000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_75_sw5000.txt
(python gnn_models/rupam_file.py --dataset ogbn-products -w spectral -r 0.75 --sw_max 10000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset ogbn-products --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/ogbnproducts_75_sw10000.txt

# --- Reddit ---
# Ratio 0.25
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.25 --sw_max 50 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_25_sw50.txt
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.25 --sw_max 100 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_25_sw100.txt
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.25 --sw_max 1000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_25_sw1000.txt
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.25 --sw_max 5000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_25_sw5000.txt
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.25 --sw_max 10000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_25_sw10000.txt

# Ratio 0.5
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.5 --sw_max 50 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_50_sw50.txt
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.5 --sw_max 100 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_50_sw100.txt
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.5 --sw_max 1000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_50_sw1000.txt
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.5 --sw_max 5000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_50_sw5000.txt
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.5 --sw_max 10000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_50_sw10000.txt

# Ratio 0.75
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.75 --sw_max 50 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_75_sw50.txt
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.75 --sw_max 100 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_75_sw100.txt
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.75 --sw_max 1000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_75_sw1000.txt
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.75 --sw_max 5000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_75_sw5000.txt
(python gnn_models/rupam_file.py --dataset reddit -w spectral -r 0.75 --sw_max 10000 --mask_agg majority --feature_agg mean --label_agg max --boost_h 400 && python gnn_models/node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > output/reddit_75_sw10000.txt
