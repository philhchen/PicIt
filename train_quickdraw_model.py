import torch
import torch.nn as nn

from data.data_load import load_data
from models import model_utils
from train import train

import time

IMG_DIR = 'dataset/img'
batch_size = 128
params = {'num_workers': 8}

data_dict = {}
dataloaders_dict = {} 
for phase in ['train', 'val']:
    data_dict[phase], dataloaders_dict[phase] = load_data(IMG_DIR, batch_size, phase, verbose=True, **params)

# Hyperparameters
model_name = "resnet"
feature_extract = False
num_classes = len(data_dict['train'].classes)

# Initialize the model for this run
model_ft, input_size = model_utils.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Send the model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

# Setup the loss function and optimizer
num_epochs = 2
criterion = nn.CrossEntropyLoss()
optimizer_ft = train.get_optimizer(model_ft, feature_extract)

# Train and evaluate
start = time.time()
# model_ft, hist = train.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
train.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
print(time.time() - start)