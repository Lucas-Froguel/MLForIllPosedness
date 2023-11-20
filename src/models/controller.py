import torch
import torch.nn as nn


class NeuralNetController:
    def __init__(self, model: nn.Module = None, model_path: str = None):
        self.model_path: str = model_path
        self.model: nn.Module = model

    def save(self, path: str = None):
        torch.save(self.model, path if path else self.model_path)

    def load(self):
        self.model = torch.load(self.model_path)
