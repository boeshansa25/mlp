import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from slp import SingleLayerPerceptron
from torch import nn

# Load MNIST test dataset
test_dataset = MNIST(".", train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Hyperparameters
in_dim = 784
out_dim = 10

# Instantiate model
model = SingleLayerPerceptron(in_dim, out_dim)
model.load_state_dict(torch.load('mnist_slp.pt'))
model.eval()

# Testing function
def test(model=model):
    model.eval()
    accuracy = 0.0
    computed_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data = data.flatten(start_dim=1)
            out = model(data)
            _, preds = out.max(dim=1)
            computed_loss += loss_fn(out, target).item()
            accuracy += torch.sum(preds == target).item()

    total_samples = len(test_loader.dataset)
    print("Test loss: {:.4f}, test accuracy: {:.2f}%".format(
        computed_loss / total_samples, (accuracy / total_samples) * 100))

# Test the model
test()