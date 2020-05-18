from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .constants import *

def load_data(data_folder, batch_size, phase, verbose=False, **kwargs):
    """
    @param data_folder - str: path to root of data directory
    @param batch_size - int: batch size
    @param phase - "train" or "test": phase to load data for
    @param verbose - bool: whether to print information about data

    @returns data, dataloader - dataset and data loader pytorch structures.
    """
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([IMG_SIZE, IMG_SIZE]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]),
        'val': transforms.Compose(
            [transforms.Resize([IMG_SIZE, IMG_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([IMG_SIZE, IMG_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        }
    data = ImageFolder(root=data_folder, 
                       transform=transform[phase])
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, 
                             **kwargs, 
                             drop_last = True if phase == 'train' else False)
    if verbose:
        print('Number of training examples: ', len(data))
        print('Number of labels: ', len(data.classes))
        print('Labels: ', data.classes)
    return data, data_loader