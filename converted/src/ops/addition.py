# NNModelling — DSL for designing neural networks via visual node editor
# Copyright (C) 2026  Luca Sforza
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
