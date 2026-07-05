# NNModelling — DSL for designing neural networks via visual node editor
# Copyright (C) 2026  Luca Sforza
#
# Licensed under the GNU General Public License v3 or later.
# Commercial licenses are available — contact Luca Sforza.
# See the LICENSE file for details.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
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
