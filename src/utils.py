import torch
import numpy as np
import matplotlib.pyplot as plt

from src.kernels import rho


def plot_function_and_model(i, model=None):
    x = np.linspace(0, 10, num=1000)
    xx = torch.linspace(0, 10, steps=1000).reshape(-1, 1)

    y = rho(x)
    yy = model(xx)

    plt.figure()
    plt.plot(x, y, "r-", label="rho")
    plt.plot(x, yy.cpu().detach().numpy(), "b-", label="model")
    plt.legend()
    plt.savefig(f"src/pictures/rho_model_{i}.png")


def plot_kernel_and_model(i, model=None, kernel=None):
    x = np.linspace(0, 10, num=1000)
    xx = torch.linspace(0, 10, steps=1000).reshape(-1, 1)

    y = rho(x) * kernel(xx).cpu().detach().numpy()
    yy = model(xx) * kernel(xx)

    plt.figure()
    plt.plot(x, y, "r-", label="rho")
    plt.plot(x, yy.cpu().detach().numpy(), "b-", label="model")
    plt.legend()
    plt.savefig(f"src/pictures/rho_model_kernel_{i}.png")
