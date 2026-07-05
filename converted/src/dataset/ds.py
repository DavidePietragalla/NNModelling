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


class Dataset(torch.utils.data.Dataset):
    @abstractmethod
    def division(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        raise NotImplementedError("Dataset.division is not implemented by subclasses")
