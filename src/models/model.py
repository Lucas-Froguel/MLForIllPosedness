
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

    def evaluate(self, k, kernel=None, M: float = 0):
        def integral(x, k):
            return self(x) * kernel(x, k)

        integration_domain_lower = [[-10, M]]
        integration_domain_upper = [[M, 10]]
        simp = Simpson()
        result_l = simp.integrate(
            lambda x: integral(x, k), dim=1, N=10**5, integration_domain=integration_domain_lower
        )
        result_u = simp.integrate(
            lambda x: integral(x, k), dim=1, N=10 ** 5, integration_domain=integration_domain_upper
        )

        return result_l + result_u

