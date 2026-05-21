import torch
import torch.nn as nn


class ScaledDotProduct(nn.Module):
    """Scaled dot-product for attention: Q @ K^T / sqrt(d_model)."""

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.scale = d_model ** -0.5

    def forward(self, inputs):
        q, k = inputs[0], inputs[1]
        return torch.matmul(q, k.transpose(-2, -1)) * self.scale
