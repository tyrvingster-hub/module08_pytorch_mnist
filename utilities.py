#@title Import necessary libraries
'''
Runtime: GPU $$$
tensors:                    https://pytorch.org/docs/stable/tensors.html
torch.nn:                   https://pytorch.org/docs/stable/nn.html
torch.nn.Linear             https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
torch.nn.functional:        https://pytorch.org/docs/stable/nn.functional.html
torch.utils.data:           https://pytorch.org/docs/stable/data.html
torch.nn.Module:            https://pytorch.org/docs/stable/generated/torch.nn.Module.html
torch.no_grad():            https://pytorch.org/docs/stable/generated/torch.no_grad.html
torch.optim:                https://pytorch.org/docs/stable/optim.html
torchvision:                https://pytorch.org/vision/stable/index.html
torchvision.transforms:     https://pytorch.org/vision/stable/transforms.html
torchvision.datasets:       https://pytorch.org/vision/main/datasets.html
'''

# Import necessary libraries

# Import the PyTorch library
# PyTorch is an open-source machine learning library used for applications such as computer vision & natural language processing
import torch

# Import the neural network module from PyTorch
# torch.nn provides classes & functions to build neural networks
import torch.nn as nn

# Import the optimization module from PyTorch
# torch.optim provides various optimization algorithms such as SGD, Adam, etc.
#import torch.optim as optim

# Import the torchvision library
# torchvision provides tools to handle image & video data
#import torchvision

# Import the transforms module from torchvision
# torchvision.transforms provides common image transformations for data augmentation & preprocessing
#import torchvision.transforms as transforms

# Import the matplotlib library for plotting
# matplotlib.pyplot provides functions to create various types of plots & visualizations
import matplotlib.pyplot as plt

# Import the tqdm library for progress bars
# tqdm provides a fast, extensible progress bar for loops & tasks
from tqdm import tqdm

# Import the numpy library
# numpy provides support for large, multi-dimensional arrays & matrices, along with a collection of mathematical functions to operate on these arrays
#import numpy as np

#@title Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 5x5 kernel
        self.layer1 = nn.Sequential(
            #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            #https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            nn.BatchNorm2d(32),  # Batch normalization for faster convergence
            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),  # Activation function
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Second convolutional layer: 32 input channels, 64 output channels, 5x5 kernel
        self.layer2 = nn.Sequential(
            #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            #https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            nn.BatchNorm2d(64),  # Batch normalization
            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),  # Activation function
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Fully connected layer: input size 7*7*64, output size 1000
        self.fc1 = nn.Linear(7*7*64, 1000)
        # Fully connected layer: input size 1000, output size 10 (number of classes)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # Forward pass through the first convolutional layer
        out = self.layer1(x)
        # Forward pass through the second convolutional layer
        out = self.layer2(out)
        # Flatten the output for the fully connected layer
        out = out.reshape(out.size(0), -1) # also look at https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
        # Forward pass through the first fully connected layer
        out = self.fc1(out)
        # Forward pass through the second fully connected layer
        out = self.fc2(out)
        return out


#@title Track accuracy
def track_accuracy(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)  # Move images to the configured device
            labels = labels.to(device)  # Move labels to the configured device
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)  # Increment total by the number of labels
            correct += (predicted == labels).sum().item()  # Increment correct by the number of correct predictions
        accuracy = 100 * correct / total  # Calculate accuracy
    model.train()  # Set the model back to training mode
    return accuracy

#@title Training
def training(model, num_epochs, train_loader, test_loader, criterion, optimizer, device):
    loss_list = []
    acc_train = []
    acc_test = []

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader,
                    desc=f"Training: {epoch + 1:03d}/{num_epochs}",
                    ncols=125,
                    leave=True)
        running_loss = []

        for i, (images, labels) in enumerate(pbar, 1):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{sum(running_loss) / i:10.6f}")

        loss_list.append(sum(running_loss) / len(running_loss))
        acc_train.append(track_accuracy(model, train_loader, device))
        acc_test.append(track_accuracy(model, test_loader, device))

    return model, loss_list, acc_train, acc_test

#@title Plot
def plot(loss_list, acc_train, acc_test, save_path='./figure.png'):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label='Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc_train, label='Training accuracy')
    plt.plot(acc_test, label='Test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()






