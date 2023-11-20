import numpy as np
import torch

from src.decorators import register


def rho(x):
    return np.power(np.sin(x), 2) * np.exp(-x)

@register
def kernel_1(x, k):
    return 1 / (torch.pow(k, 2) + torch.pow(x, 2))

@register
def kernel_2(x, k):
    return 1 / (1 + k**2 * x**2)

@register
def kernel_3(x, k):
    return torch.sin(x) / (k**2 + x**2)

@register
def kernel_4(x, k):
    return k * torch.exp(-x)

@register
def kernel_5(x, k):
    return torch.exp(-(k*x)**2)

