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

import torch.nn as nn

from ops.subflow import Subflow


class Repeat(nn.Module):
    """Repeats subgraph N times with independent module instances.

    Creates N independent Subflow instances chained in nn.Sequential.
    Each instance gets fresh modules (Hydra instantiate creates new objects).
    """

    def __init__(self, entry_node: str, internal_nodes: dict, iterations: int = 1, **kwargs):
        super().__init__()
        self.net = nn.Sequential(*[
            Subflow(entry_node=entry_node, internal_nodes=internal_nodes)
            for _ in range(iterations)
        ])

    def forward(self, x):
        return self.net(x)
