
import torch
import numpy as np

from torchquad import Simpson

from src.kernels import kernel_1, kernel_2, kernel_3, kernel_4, kernel_5


def cost_function(k: float, y: float, model=None) -> float:
    def integral(x, k):
        return model(x) * kernel_1(x, k)

    integration_domain = [[0, 10**3]]
    simp = Simpson()
    result = simp.integrate(lambda x: integral(x, k), dim=1, N=10**6, integration_domain=integration_domain)

    return torch.mean((result.item() - y)**2)

