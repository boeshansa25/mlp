import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from slp import SingleLayerPerceptron

# Load MNIST dataset
train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Hyperparameters
in_dim = 784
out_dim = 10
learning_rate = 0.01
num_epochs = 40

# Instantiate model
model = SingleLayerPerceptron(in_dim, out_dim)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for data, target in train_loader:
        data = data.flatten(start_dim=1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'mnist_slp.pt')