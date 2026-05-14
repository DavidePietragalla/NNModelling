from functorch.dim import Tensor
from omegaconf import DictConfig
import torch
import lightning as lit
from torchmetrics import Accuracy

import torch.nn as nn
import torch.nn.functional as F

from hydra.utils import instantiate


class Net(lit.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(Net, self).__init__()
        self.save_hyperparameters()

        self.residuals: dict[str, Tensor] = dict()

        self.depth = cfg.depth
        self.cfg = cfg

        self.embed = instantiate(cfg.embed)
        self.features = nn.Sequential(
            *[instantiate(cfg.block) for _ in range(self.depth - 1)]
        )

        self.unembed = instantiate(cfg.unembed)

    def forward(self, x):
        x = self.embed(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.unembed(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", accuracy(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", accuracy(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, self.parameters())
        return optimizer


accuracy = Accuracy(task="multiclass", num_classes=10)
