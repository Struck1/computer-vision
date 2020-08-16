# MNIST
# DataLoader, Transform
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Traning Loop (batch traning)
# Model evaluation
# GPU support

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch.nn as nn
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else ('cpu'))
print(torch.cuda.is_available(), device)

# Hyper-parameters

input_size = 784  # 28x28
batch_size = 100
hidden_size = 500
num_classes = 10
num_epoch = 2


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)


test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)


# Data loader

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)


test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=True)


example = iter(test_loader)
example_data, example_targets = example.next()
print(example_data.shape, example_targets[0])


for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()

# NN with one hidden layer


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes)
model.to(device)

# Loss and optimizer

lossFunction = nn.CrossEntropyLoss()
lossFunction.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
n_total_step = len(train_loader)
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        # orginal shape [100, 1, 28, 28]
        #resized [100, 784]

        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass

        outputs = model(images)
        loss = lossFunction(outputs, labels)

        #backward and optimzer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                f'Epoch : {epoch +1} / {num_epoch}, Step : {i} / {n_total_step}, Loss: {loss.item():.4f}')

# Test the model

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index

        value, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        # add all true value and convert tensor to number
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images:{acc} %')
