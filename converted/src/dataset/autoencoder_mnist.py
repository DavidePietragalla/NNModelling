from dataset.mnist import MNISTDataset
from torch.utils.data import DataLoader


class AutoencoderMNIST(MNISTDataset):
    """MNIST dataset for autoencoder training. Returns (image, image) instead of (image, label)."""

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image, image

    def division(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_loader, val_loader, test_loader = super().division()
        return train_loader, val_loader, test_loader
