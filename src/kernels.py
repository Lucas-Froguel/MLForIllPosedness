import numpy as np
import torch


def kernel_1(x, k):
    return 1 / (torch.pow(k, 2) + torch.pow(x, 2))

def kernel_2(x, k):
    return 1 / (1 + k**2 * x**2)

def kernel_3(x, k):
    return np.sin(x) / (k**2 + x**2)

def kernel_4(x, k):
    return k * np.exp(-x)

def kernel_5(x, k):
    return np.exp(-(k*x)**2)

