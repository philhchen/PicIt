#!/bin/bash
# Activate virtual environment
source .env/bin/activate

# # Download part of the dataset and parse the dataset
mkdir dataset
gsutil cp 'gs://quickdraw_dataset/full/simplified/a*.ndjson' dataset
python data/data_utils.py