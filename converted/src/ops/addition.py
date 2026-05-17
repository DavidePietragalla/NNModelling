import torch.nn as nn


class Addition(nn.Module):
    """Sums all input tensors element-wise. Receives list of tensors from BFS join."""

    def __init__(self):
        super().__init__()

    def forward(self, tensors):
        return sum(tensors)
