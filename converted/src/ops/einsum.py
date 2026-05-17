import torch
import torch.nn as nn


class Einsum(nn.Module):
    """Applies torch.einsum with stored expression. Receives list of tensors from BFS join."""

    def __init__(self, expr: str = ""):
        super().__init__()
        self.expr = expr

    def forward(self, tensors):
        if not self.expr:
            raise ValueError("Einsum join requires non-empty 'expr' parameter")
        return torch.einsum(self.expr, *tensors)
