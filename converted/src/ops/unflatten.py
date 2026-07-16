# NNModelling — DSL for designing neural networks via visual node editor
# Copyright (C) 2026  Luca Sforza
#
# Licensed under the GNU General Public License v3 or later.
# Commercial licenses are available — contact Luca Sforza.
# See the LICENSE file for details.

from collections.abc import Sequence

import torch
from torch import nn


class Unflatten(nn.Module):
    """Hydra-compatible adapter for :class:`torch.nn.Unflatten`.

    OmegaConf represents YAML sequences as ``ListConfig`` objects. PyTorch's
    native module requires a tuple, so this adapter normalizes any integer
    sequence before constructing the operation.
    """

    def __init__(self, dim: int, unflattened_size: Sequence[int]) -> None:
        """Initialize an unflatten operation from a Hydra sequence."""
        super().__init__()
        shape = tuple(int(size) for size in unflattened_size)
        self.operation = nn.Unflatten(dim=dim, unflattened_size=shape)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Restore the configured dimensions in ``tensor``."""
        return self.operation(tensor)
