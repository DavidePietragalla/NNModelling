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


class Einsum(nn.Module):
    """Applies torch.einsum with stored expression. Receives list of tensors from BFS join."""

    def __init__(self, expr: str = ""):
        super().__init__()
        self.expr = expr

    def forward(self, tensors):
        if not self.expr:
            raise ValueError("Einsum join requires non-empty 'expr' parameter")
        return torch.einsum(self.expr, *tensors)
