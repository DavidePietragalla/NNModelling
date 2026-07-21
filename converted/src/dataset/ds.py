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
from torch.utils.data import DataLoader

from abc import abstractmethod
from typing import Any


class Dataset(torch.utils.data.Dataset):
    @abstractmethod
    def division(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        raise NotImplementedError("Dataset.division is not implemented by subclasses")

    @classmethod
    def inference_adapter_spec(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Describe a portable inference adapter without constructing a dataset.

        The default accepts only model-ready tensors. Dataset implementations
        may override it with a declarative specification understood by the
        exported model wheel.
        """

        del config
        return {"kind": "tensor", "version": 1}
