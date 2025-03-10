import torch.nn as nn
import torch.nn.functional as F

class SingleLayerPerceptron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SingleLayerPerceptron, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)