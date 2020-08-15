import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import math


"""
epoch = one forward and backward pass of ALL training samples

batch_size = number of training samples used in one forward/backward pass

number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes

e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

Implement a custom Dataset:
Inherit Dataset
implement __init__ , __getitem__ , and __len__

"""


class WineDataset(Dataset):
    def __init__(self, root):

        dataset = np.loadtxt(root, skiprows=1, delimiter=',', dtype=np.float32)
        self.n_samples = dataset.shape[0]

        self.data_x = torch.from_numpy(
            dataset[:, 1:])  # [n_samples, n_features]
        self.data_y = torch.from_numpy(dataset[:, 0])

    def __getitem__(self, idx):

        return self.data_x[idx], self.data_y[idx]

    def __len__(self):
        return self.n_samples

# create datasets


root = r'C:\Users\ASUS\Desktop\test_py\ml\ml_torch_general\wine.csv'
dataset = WineDataset(root)

# get the firs sample

first_data = dataset[0]
features, label = first_data
print(features, label)

train_loader = DataLoader(dataset,
                          batch_size=4,
                          shuffle=True)

print(train_loader)

# dataiter = iter(train_loader)
# data = dataiter.next()
# features, labels = data
# print(features, labels)

# for j, i in enumerate(train_loader): #178 samples, batch_size = 4 , 178/4 = 44.5 -> 45 iteration --> 1 epoch
#     print(j, i)

batch_size = 4
num_epoch = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/batch_size)
print(total_samples, n_iterations)

for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(train_loader):
        #178 samples, batch_size = 4 , 178/4 = 44.5 -> 45 iteration --> 1 epoch
        print(f'Epoch : {epoch+1}/{num_epoch}, Step {i+1}/{n_iterations}, Inputs {inputs.shape}, Labels {labels.shape}')