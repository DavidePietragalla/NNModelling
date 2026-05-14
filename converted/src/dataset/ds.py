import torch
from torch.utils.data import DataLoader

from abc import abstractmethod


class Dataset(torch.utils.data.Dataset):
    @abstractmethod
    def division(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        raise NotImplementedError("Dataset.division is not implemented by subclasses")
