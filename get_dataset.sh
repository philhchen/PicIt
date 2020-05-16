#!/bin/bash
# Activate virtual environment
source .env/bin/activate

# # Download part of the dataset and parse the dataset
mkdir dataset
mkdir dataset/raw
gsutil -m cp 'gs://quickdraw_dataset/full/simplified/a*.ndjson' dataset/raw
python data/data_utils.py