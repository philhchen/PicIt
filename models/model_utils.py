import torch
import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, use_checkpoint, checkpoint_path,
                     feature_extract=False, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. 
    # Each of these variables is model specific.
    model_ft = None
    input_size = 224

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    if use_checkpoint:
        checkpoint = torch.load(checkpoint_path)
        model_weights = checkpoint['model_state_dict']
        model_ft.load_state_dict(model_weights)
    return model_ft, input_size