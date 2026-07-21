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
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from typing import Any

from dataset.mnist import MNISTDataset


class _ImageOnly:
    """Wraps MNIST dataset to return (image, image) for autoencoder."""
    def __init__(self, ds):
        self.ds = ds
    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        return img, img
    def __len__(self):
        return len(self.ds)


class AutoencoderMNIST(MNISTDataset):
    """MNIST dataset for autoencoder training. Returns (image, image) instead of (image, label)."""

    @classmethod
    def num_classes(cls, config: dict[str, Any]) -> None:
        """Report no classification cardinality for reconstruction training."""

        del config
        return None

    @classmethod
    def class_names(cls, config: dict[str, Any]) -> None:
        """Report no class labels for reconstruction training."""

        del config
        return None

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image, image

    def division(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_size = int(self.train_size * len(self))
        val_size = len(self) - train_size
        train_dataset, val_dataset = random_split(self, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(_ImageOnly(self.test_dataset), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader
