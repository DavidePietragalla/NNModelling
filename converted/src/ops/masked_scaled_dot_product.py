import torch
import torch.nn as nn


class MaskedScaledDotProduct(nn.Module):
    """Scaled dot-product with causal masking: Q @ K^T / sqrt(d_model) + mask.

    Adds upper-triangular -inf mask so position i attends only to j <= i.
    Downstream Softmax turns -inf to 0 naturally.
    Same interface as ScaledDotProduct — drop-in replacement for autoregressive attention.
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.scale = d_model ** -0.5

    def forward(self, inputs):
        q, k = inputs[0], inputs[1]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
        return scores + mask
