"""Deterministic MNIST-shaped datasets used only by the backend E2E tests.

This module is a test fixture: it is imported by the service/E2E test suite
and by the ``main.py`` training subprocesses those tests launch. It generates
a fixed, seeded pool of ``1x28x28`` grayscale images plus 10-class labels so
the full backend job path (config build -> Hydra -> ``main.py`` training ->
wheel export) has no network or filesystem dependency on the real MNIST
download.

Two dataset classes are provided:

- :class:`TinyMNISTDataset`: ``(image, label)`` pairs for 10-class
  MNIST-shaped classification;
- :class:`TinyAutoencoderDataset`: ``(image, image)`` pairs for image
  reconstruction.

A :class:`BrokenDataset` class whose constructor always fails is also defined
so a real job failure state can be exercised through the production executor
without touching product datasets.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from dataset.ds import Dataset


def _make_pool(total: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a deterministic pool of images and labels."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    images = torch.rand(total, 1, 28, 28, generator=generator, dtype=torch.float32)
    labels = torch.randint(0, 10, (total,), generator=generator)
    return images, labels


class _IndexedView(torch.utils.data.Dataset):
    """Read-only view over absolute indices of a parent dataset.

    Delegating to ``parent.__getitem__`` keeps subclass behavior (for example
    the autoencoder's ``(image, image)`` pairs) when splitting the pool.
    """

    def __init__(self, parent: "TinyMNISTDataset", indices: range) -> None:
        self._parent = parent
        self._indices = list(indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._parent[self._indices[index]]

    def __len__(self) -> int:
        return len(self._indices)


class TinyMNISTDataset(Dataset):
    """Deterministic 10-class MNIST-shaped dataset for backend E2E tests.

    Args:
        batch_size: Batch size used by all three returned loaders.
        num_workers: DataLoader worker count. Keep at 0 for determinism.
        train_size: Fraction of the pool used for the training split.
        total_samples: Total deterministic sample pool size.
        seed: Seed for the fixed data pool. Changing it changes the data.
    """

    @classmethod
    def num_classes(cls, config: dict[str, Any]) -> int:
        """Report the ten digit classes supplied by this fixture."""
        del config
        return 10

    @classmethod
    def class_names(cls, config: dict[str, Any]) -> list[str]:
        """Return display names for the ten fixture classes."""
        del config
        return [str(index) for index in range(10)]

    @classmethod
    def inference_adapter_spec(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Export the tensor adapter used by the wheel for this fixture."""
        del config
        return {"kind": "tensor", "version": 1}

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        train_size: float = 0.6,
        total_samples: int = 100,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        if not 0 < train_size < 1:
            raise ValueError("train_size must be strictly between 0 and 1")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self._images, self._labels = _make_pool(total_samples, seed)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._images[index], self._labels[index]

    def __len__(self) -> int:
        return len(self._labels)

    def _loader(self, view: _IndexedView, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            view,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def division(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Return deterministic train, validation, and test loaders."""
        total = len(self._labels)
        train_n = int(self.train_size * total)
        val_n = (total - train_n) // 2
        train = _IndexedView(self, range(train_n))
        val = _IndexedView(self, range(train_n, train_n + val_n))
        test = _IndexedView(self, range(train_n + val_n, total))
        return (
            self._loader(train, shuffle=True),
            self._loader(val, shuffle=False),
            self._loader(test, shuffle=False),
        )


class TinyAutoencoderDataset(TinyMNISTDataset):
    """Deterministic reconstruction dataset returning ``(image, image)``."""

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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, _ = super().__getitem__(index)
        return image, image


class BrokenDataset(Dataset):
    """Dataset whose constructor always fails.

    Used to exercise a real job failure through the production executor: the
    config generation succeeds, but ``main.py`` crashes while instantiating
    the dataset and the manager must record a coherent failed state.
    """

    @classmethod
    def num_classes(cls, config: dict[str, Any]) -> None:
        del config
        return None

    @classmethod
    def class_names(cls, config: dict[str, Any]) -> None:
        del config
        return None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("tiny broken dataset cannot be constructed")

    def division(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        raise RuntimeError("BrokenDataset is never constructed")
