'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet
complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html
On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale
On Tensors
----------
LinearTransformation, Normalize, RandomErasing
Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage
Generic
-------
Use Lambda 
Custom
------
Write own class
Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self, root, transform=None):

        self.transform = transform
        dataset = np.loadtxt(root, skiprows=1, delimiter=',', dtype=np.float32)
        self.n_samples = dataset.shape[0]

        self.data_x = dataset[:, 1:]
        self.data_y = dataset[:, [0]]

    def __getitem__(self, idx):

        samples = self.data_x[idx], self.data_y[idx]
        if self.transform:
            samples = self.transform(samples)

        return samples

    def __len__(self):
        return self.n_samples

class ToTensor:
    # Convvert np arrays to tensor
    def __call__(self, samples):
        inputs, labels = samples
        return torch.from_numpy(inputs), torch.from_numpy(labels)

class MultiTransform:

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, samples):
        inputs, labels = samples
        inputs *= self.factor
        return inputs, labels

# create datasets

root = r'C:\Users\ASUS\Desktop\test_py\ml\ml_torch_general\wine.csv'


print('\nWith Tensor Transform')
dataset = WineDataset(root, transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor and Multiplication Transform')
composed = torchvision.transforms.Compose([ToTensor(), MultiTransform(4)])
dataset = WineDataset(root, transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)