import torch
import torch.nn as nn


class MatMul(nn.Module):
    """Matrix multiplication join. inputs[0] @ inputs[1]."""

    def forward(self, inputs):
        return torch.matmul(inputs[0], inputs[1])
