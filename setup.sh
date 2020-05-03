#!/bin/bash

python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt

# # Download Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# # Download part of the dataset and parse the dataset
gsutil -m cp gs://quickdraw_dataset/full/simplified/a*.ndjson dataset
python data/load_data.py