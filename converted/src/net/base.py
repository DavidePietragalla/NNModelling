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
import warnings

import lightning as lit
import torch
import torch.nn as nn
import wandb
from torchmetrics import Accuracy, MeanSquaredError
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


class Net(lit.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(Net, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.module_dict = nn.ModuleDict()
        self.input_order: dict[str, list[str]] = {}

        def instantiate_layer(layer_config):
            layer_dict = OmegaConf.to_container(layer_config, resolve=True)
            layer_dict.pop("stereotype", None)
            layer_dict.pop("taskType", None)
            return instantiate(layer_dict)

        if hasattr(cfg.net, "nodes"):
            for node_id, node in cfg.net.nodes.items():
                node_stereotype = node.get("stereotype", "")
                if self._is_loss(node_stereotype):
                    continue
                if node.type == "sequential":
                    layers = []
                    for layer_cfg in node.layers:
                        ls = layer_cfg.get("stereotype", "")
                        if not self._is_input(ls) and not self._is_loss(ls):
                            layers.append(instantiate_layer(layer_cfg))
                    if layers:
                        self.module_dict[node_id] = nn.Sequential(*layers)

                elif node.type == "module" and hasattr(node, "layer"):
                    ls = node.layer.get("stereotype", "")
                    if not self._is_input(ls) and not self._is_loss(ls):
                        self.module_dict[node_id] = instantiate_layer(node.layer)
                elif node.type == "join" and hasattr(node, "layer"):
                    self.module_dict[node_id] = instantiate_layer(node.layer)
                    inputs_list = node.get("inputs", [])
                    if inputs_list:
                        self.input_order[node_id] = inputs_list
                elif node.type == "subflow":
                    self.module_dict[node_id] = instantiate(node)

        if cfg.net.get("lossNode"):
            self.loss_fn = instantiate_layer(cfg.net.lossNode)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.task_type = self._task_type()
        self.num_classes = getattr(self.cfg.net, "num_classes", 10)
        self.class_names = self._class_names()
        self._test_probabilities: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []
        self.metric = self._build_metric()

    def _is_loss(self, name: str) -> bool:
        return "loss" in name.lower()

    def _is_input(self, name: str) -> bool:
        return name.lower() in ("input", "fork")

    def _task_type(self) -> str:
        """Return the configured task type, defaulting to classification."""

        loss_node = self.cfg.net.get("lossNode")
        return loss_node.get("taskType", "classification") if loss_node is not None else "classification"

    def _is_classification(self) -> bool:
        """Return whether this model should emit classification diagnostics."""

        return self.task_type == "classification"

    def _class_names(self) -> list[str]:
        """Return configured display names or stable generic class labels."""

        configured = self.cfg.net.get("class_names")
        if configured is not None and len(configured) == self.num_classes:
            return [str(name) for name in configured]
        return [f"class_{index}" for index in range(self.num_classes)]

    def _build_metric(self):
        if self.task_type == "classification":
            return Accuracy(task="multiclass", num_classes=self.num_classes)
        elif self.task_type == "regression":
            return MeanSquaredError()
        else:
            return Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        root_id = self.cfg.net.root
        # dict keyed by source node ID (not list) — preserves parent identity
        # for join input ordering via self.input_order.
        node_inputs: dict[str, dict] = {root_id: {"_in": x}}
        in_degrees = self._compute_in_degrees()
        processed_count = {node_id: 0 for node_id in in_degrees}

        queue = [root_id]
        final_output = x

        while queue:
            curr_id = queue.pop(0)
            curr_node = self.cfg.net.nodes[curr_id]
            inputs = node_inputs[curr_id]

            if curr_node.type == "join":
                if curr_id in self.input_order:
                    ordered = [inputs.get(pid) for pid in self.input_order[curr_id]]
                    ordered = [t for t in ordered if t is not None]
                else:
                    ordered = list(inputs.values())
                if curr_id in self.module_dict:
                    out = self.module_dict[curr_id](ordered)
                else:
                    raise NotImplementedError(
                        f"Join node {curr_id} has no instantiated module"
                    )
            else:
                inp = next(iter(inputs.values()))

                if curr_id in self.module_dict:
                    out = self.module_dict[curr_id](inp)
                else:
                    out = inp

            final_output = out

            for child_id in curr_node.children:
                if child_id not in node_inputs:
                    node_inputs[child_id] = {}
                node_inputs[child_id][curr_id] = out
                processed_count[child_id] += 1

                if processed_count[child_id] == in_degrees[child_id]:
                    queue.append(child_id)

        return final_output

    def _compute_in_degrees(self):
        deg = {node_id: 0 for node_id in self.cfg.net.nodes}
        for node in self.cfg.net.nodes.values():
            for child_id in node.children:
                if child_id in deg:
                    deg[child_id] += 1
        return deg

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_metric", self.metric(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_metric", self.metric(y_hat, y), prog_bar=True)
        if self._is_classification():
            self._test_probabilities.append(torch.softmax(y_hat.detach(), dim=1).cpu())
            self._test_targets.append(y.detach().reshape(-1).cpu())

    def on_test_epoch_start(self):
        """Clear test-set classification data before Lightning starts evaluation."""

        if self._is_classification():
            self._test_probabilities = []
            self._test_targets = []

    def on_test_epoch_end(self):
        """Log whole-test-set classification diagnostics to an active W&B run."""

        if not self._is_classification():
            return
        logger = self.logger
        experiment = getattr(logger, "experiment", None)
        if experiment is not None:
            self._log_classification_report(experiment)

    def _classification_report(self) -> dict[str, object] | None:
        """Build scalar metrics and chart inputs from all test predictions."""

        if not self._is_classification() or not self._test_probabilities:
            return None
        probabilities = torch.cat(self._test_probabilities)
        targets = torch.cat(self._test_targets).to(torch.long)
        predictions = probabilities.argmax(dim=1)
        confusion = torch.bincount(
            targets * self.num_classes + predictions,
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes).to(torch.float)
        true_positives = confusion.diag()
        precision = true_positives / confusion.sum(dim=0).clamp_min(1)
        recall = true_positives / confusion.sum(dim=1).clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float).eps)
        scalars: dict[str, float] = {
            "test/accuracy": (predictions == targets).float().mean().item(),
            "test/macro_precision": precision.mean().item(),
            "test/macro_recall": recall.mean().item(),
            "test/macro_f1": f1.mean().item(),
        }
        for index, name in enumerate(self.class_names):
            scalars[f"test/precision/{name}"] = precision[index].item()
            scalars[f"test/recall/{name}"] = recall[index].item()
            scalars[f"test/f1/{name}"] = f1[index].item()
        return {
            "scalars": scalars,
            "targets": targets.tolist(),
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
        }

    def _log_classification_report(self, experiment) -> None:
        """Send metrics and W&B charts without risking a successful test run."""

        report = self._classification_report()
        if report is None:
            return
        payload = dict(report["scalars"])
        try:
            payload["test/confusion_matrix"] = wandb.plot.confusion_matrix(
                y_true=report["targets"],
                preds=report["predictions"],
                class_names=self.class_names,
                title="Test confusion matrix",
            )
            payload["test/roc_curve"] = wandb.plot.roc_curve(
                report["targets"],
                report["probabilities"],
                labels=self.class_names,
                title="Test ROC curve",
            )
            payload["test/precision_recall_curve"] = wandb.plot.pr_curve(
                report["targets"],
                report["probabilities"],
                labels=self.class_names,
                title="Test precision-recall curve",
            )
        except Exception as error:
            warnings.warn(f"Unable to log W&B classification charts: {error}", RuntimeWarning)
        experiment.log(payload)

    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, self.parameters())
