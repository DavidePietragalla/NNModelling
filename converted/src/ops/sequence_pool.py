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
