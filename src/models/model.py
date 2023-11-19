
import torch
import torch.nn as nn


class FunctionNet(nn.Module):
    def __init__(self, hidden_size=50):
        super(FunctionNet, self).__init__()
        self.hidden_size = hidden_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

