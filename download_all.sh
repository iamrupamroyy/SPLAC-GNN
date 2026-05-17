#!/bin/bash

# Datasets to pre-download
DATASETS=("pubmed" "ogbn-arxiv" "ogbn-products" "reddit" "yelp" "flickr")

echo "--- Starting Dataset Pre-download ---"

for DATASET in "${DATASETS[@]}"
do
    echo ""
    echo "===================================================="
    echo " Downloading: $DATASET"
    echo "===================================================="
    
    # Run node_classificationDGL.py with --epoch 0 to trigger the download logic
    # We use 'cpu' mode for downloads to avoid needing a GPU just for data fetching
    python node_classificationDGL.py --dataset $DATASET --mode cpu --epoch 0
    
    if [ $? -eq 0 ]; then
        echo "Successfully checked/downloaded $DATASET"
    else
        echo "Error checking/downloading $DATASET"
    fi
done

echo ""
echo "--- All datasets checked/downloaded! ---"
echo "You can now run your experiments safely."
