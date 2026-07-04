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

from torchvision import datasets
from torch.utils.data import DataLoader, random_split

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
