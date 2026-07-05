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


class ScaledDotProduct(nn.Module):
    """Scaled dot-product for attention: Q @ K^T / sqrt(d_model)."""

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.scale = d_model ** -0.5

    def forward(self, inputs):
        q, k = inputs[0], inputs[1]
        return torch.matmul(q, k.transpose(-2, -1)) * self.scale
