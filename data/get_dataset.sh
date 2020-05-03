#!/bin/bash
# Activate virtual environment
source .env/bin/activate

# # Download part of the dataset and parse the dataset
gsutil -m cp gs://quickdraw_dataset/full/simplified/a*.ndjson dataset
python data/load_data.py