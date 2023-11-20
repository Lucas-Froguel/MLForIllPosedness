
import torch
import torch.nn as nn
from torchquad import Simpson


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
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

    def evaluate(self, k, kernel=None):
        def integral(x, k):
            return self(x) * kernel(x, k)

        integration_domain = [[0, 10]]
        simp = Simpson()
        result = simp.integrate(lambda x: integral(x, k), dim=1, N=10**5, integration_domain=integration_domain)

        return result

