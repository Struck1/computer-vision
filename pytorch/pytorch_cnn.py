import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available(), device)

num_epoch = 4
num_batch = 4
lr = 0.001


transform = T.Compose(
    [T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])


# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_data = torchvision.datasets.CIFAR10(root='./dataa',
                                          train=True,
                                          download=True,
                                          transform=transform)

test_data = torchvision.datasets.CIFAR10(root='./dataa',
                                         train=False,
                                         download=True,
                                         transform=transform)


train_loader = DataLoader(train_data,
                          batch_size=num_batch,
                          shuffle=True)


test_loader = DataLoader(test_data,
                         batch_size=num_batch,
                         shuffle=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    # print(img)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.shape, labels)

img = torchvision.utils.make_grid(images)
print(img.shape)
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # -->cnn_test
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # n , 3, 32, 32

        # n, 6, 14, 14 conv + relu -> pool
        X = self.pool(F.relu(self.conv1(x)))
        X = self.pool(F.relu(self.conv2(X)))  # n, 16, 5, 5
        X = X.view(-1, 16 * 5 * 5)           # n, 400
        X = F.relu(self.fc1(X))              # n, 120
        X = F.relu(self.fc2(X))              # n, 84
        X = self.fc3(X)                      # n, 10
        return X


model = Net()
model.to(device)


# optimize model

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# train the model
n_total_step = len(train_loader)
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size

        images = images.to(device)
        labels = labels.to(device)
        # zero the parameter gradients

        # Forvard pass
        outputs = model(images)
        loss = lossFunction(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(
                f'Epoch : {epoch + 1} / {num_epoch}, Step : {i+1} / {n_total_step}, Loss : {loss.item():.4f}')

print('Finished Traning')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)


with torch.no_grad():
    n_correct = 0
    n_sample = 0
    n_class_correct = [0 for i in range(10)] # [0, 0 ...] (10,)
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _ , predited = torch.max(outputs, 1)
        n_sample += labels.size(0)
        n_correct += (predited == labels).sum().item()

        for i in range(num_batch):
            label = labels[i]
            pred = predited[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    
    acc = 100.0 * n_correct / n_sample
    print(f'Accuracy of the network : {acc} %')      

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i] 
        print(f'Accuracy of {classes[i]} : {acc} %')


