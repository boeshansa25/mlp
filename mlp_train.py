import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from mlp import BaseClassifier

# Load MNIST dataset
train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Hyperparameters
in_dim = 784
out_dim = 10
feature_dim = 256  # Hidden layer size
lr = 1e-3  # Learning rate
epochs = 40  # Number of epochs
target_loss = 0.05  # Target loss to stop training

# Instantiate model, optimizer, and loss function
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
optimizer = optim.SGD(classifier.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# Training function
def train(classifier=classifier, optimizer=optimizer, epochs=epochs, loss_fn=loss_fn, target_loss=target_loss):
    classifier.train()
    loss_lt = []
    epoch = 0
    while True:
        running_loss = 0.0
        for minibatch in train_loader:
            data, target = minibatch
            data = data.flatten(start_dim=1)
            out = classifier(data)
            computed_loss = loss_fn(out, target)
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += computed_loss.item()
        avg_loss = running_loss / len(train_loader)
        loss_lt.append(avg_loss)
        epoch += 1
        print("Epoch: {} train loss: {}".format(epoch, avg_loss))
        if avg_loss <= target_loss:
            print("Target loss reached. Stopping training.")
            break

    # plt.plot([i for i in range(1, len(loss_lt) + 1)], loss_lt)
    # plt.xlabel("Epoch")
    # plt.ylabel("Training Loss")
    # plt.title("MNIST Training Loss: optimizer {}, lr {}".format("SGD", lr))
    # plt.show()

    print("Saving network in mnist.pt")
    torch.save(classifier.state_dict(), 'mnist.pt')

# Train the model
#train()

# Example configuration
feature_dim = 156  # Hidden layer size
lr = 1e-3  # Learning rate
epochs = 40  # Number of epochs
target_loss = 0.4  # Target loss to stop training

# Train the model with the new configuration
train(classifier, optimizer, epochs, loss_fn, target_loss)