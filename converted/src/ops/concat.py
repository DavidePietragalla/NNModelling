import torch
import torch.nn as nn


class Concat(nn.Module):
    """Join node: concatenates all input tensors along dim.

    Receives a list of tensors (one per input handle), returns
    torch.cat(tensors, dim=dim).
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, tensors):
        return torch.cat(tensors, dim=self.dim)
