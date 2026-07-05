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
