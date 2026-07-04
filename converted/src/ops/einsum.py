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


class Einsum(nn.Module):
    """Applies torch.einsum with stored expression. Receives list of tensors from BFS join."""

    def __init__(self, expr: str = ""):
        super().__init__()
        self.expr = expr

    def forward(self, tensors):
        if not self.expr:
            raise ValueError("Einsum join requires non-empty 'expr' parameter")
        return torch.einsum(self.expr, *tensors)
