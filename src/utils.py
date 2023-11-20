import torch
from torch import squeeze
import numpy as np
import matplotlib.pyplot as plt

from src.kernels import rho


def plot_function_and_model(i, model=None, epoch: int = 0):
    x = np.linspace(0, 10, num=1000)
    xx = torch.linspace(0, 10, steps=1000).reshape(-1, 1)

    y = rho(x)
    yy = squeeze(model(xx))

    plt.figure()
    plt.plot(x, y, "r-", label="rho")
    plt.plot(x, yy.cpu().detach().numpy(), "b-", label="model")
    plt.legend()
    plt.savefig(f"src/pictures/rho_model_{epoch}_{i}.png")


def plot_kernel_and_model(i, model=None, kernel=None, epoch: int = 0):
    k = torch.tensor(2)

    x = np.linspace(0, 10, num=1000)
    xx = torch.linspace(0, 10, steps=1000).reshape(-1, 1)

    y = rho(x) * kernel(xx, k).cpu().detach().numpy().squeeze()
    yy = squeeze(model(xx)) * squeeze(kernel(xx, k))

    plt.figure()
    plt.plot(x, y, "r-", label="rho")
    plt.plot(x, yy.cpu().detach().numpy(), "b-", label="model")
    plt.legend()
    plt.savefig(f"src/pictures/rho_model_kernel_{epoch}_{i}.png")
