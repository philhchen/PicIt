from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .constants import *

class ImageFolderEX(ImageFolder):
    def __getitem__(self, index):
        try:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
        except:
            print('Exception in loading image')
            return None
        return sample, target

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

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
    data = ImageFolderEX(root=data_folder, 
                         transform=transform[phase])
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, **kwargs,
                             drop_last = True if phase == 'train' else False)
    if verbose:
        print('Number of training examples: ', len(data))
        print('Number of labels: ', len(data.classes))
        print('Labels: ', data.classes)
    return data, data_loader