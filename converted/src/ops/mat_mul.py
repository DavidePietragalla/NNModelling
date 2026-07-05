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


class MatMul(nn.Module):
    """Matrix multiplication join. inputs[0] @ inputs[1]."""

    def forward(self, inputs):
        return torch.matmul(inputs[0], inputs[1])
