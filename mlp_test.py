import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from mlp import BaseClassifier
from torch import nn

# Load MNIST dataset
test_dataset = MNIST(".", train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the trained state dictionary
state_dict = torch.load('mnist.pt')

# Extract input and output dimensions from the state dictionary
in_dim = state_dict['classifier.0.weight'].shape[1]
out_dim = state_dict['classifier.2.weight'].shape[0]
feature_dim = state_dict['classifier.0.weight'].shape[0]

# Instantiate model with extracted dimensions
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
classifier.load_state_dict(state_dict)

# Testing function
def test(classifier=classifier):
    classifier.eval()
    accuracy = 0.0
    computed_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data = data.flatten(start_dim=1)
            out = classifier(data)
            _, preds = out.max(dim=1)
            computed_loss += loss_fn(out, target)
            accuracy += torch.sum(preds == target)

        print("Test loss: {}, test accuracy: {}".format(
            computed_loss.item() / (len(test_loader) * 64), accuracy * 100.0 / (len(test_loader) * 64)))

# Test the model
test()