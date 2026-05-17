import torch
import torch.nn as nn


class Addition(nn.Module):
    """Sums all input tensors element-wise. Receives list of tensors from BFS join."""

    def __init__(self):
        super().__init__()

    def forward(self, tensors):
        if len(tensors) < 2:
            return tensors[0] if len(tensors) == 1 else tensors
        ref_shape = tensors[0].shape
        for i, t in enumerate(tensors[1:], 1):
            if t.shape != ref_shape:
                raise RuntimeError(
                    f"Addition join: shape mismatch at input {i}: "
                    f"expected {ref_shape}, got {t.shape}. "
                    "All branches feeding into this join must produce same tensor shape."
                )
        return sum(tensors)
