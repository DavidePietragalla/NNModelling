from abc import abstractmethod
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from dataset.ds import Dataset


class MNISTDataset(Dataset):
    def __init__(self, batch_size=32, num_workers=4, train_size=0.8) -> None:
        super().__init__()

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        # Load the MNIST dataset
        self.dataset = datasets.MNIST(
            root="data", train=True, download=True, transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            root="data", train=False, download=True, transform=self.transform
        )

        self.train_size: float = train_size
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def division(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        # Split the dataset into training and validation sets
        train_size = int(self.train_size * len(self.dataset))
        val_size = len(self) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        # Create DataLoaders for training, validation, and test sets
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return train_loader, val_loader, test_loader
