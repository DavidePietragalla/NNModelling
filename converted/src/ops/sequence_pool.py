import torch.nn as nn


class SequencePool(nn.Module):
    """Mean pooling over sequence dimension.

    [batch, seq_len, d_model] → [batch, d_model]
    Passes 2D inputs through unchanged.
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if x.dim() > 2:
            return x.mean(dim=self.dim)
        return x
