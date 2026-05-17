import torch
import torch.nn as nn
import lightning as lit
from torchmetrics import Accuracy, MeanSquaredError
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


class Net(lit.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(Net, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.module_dict = nn.ModuleDict()

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

        if cfg.net.get("lossNode"):
            self.loss_fn = instantiate_layer(cfg.net.lossNode)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.metric = self._build_metric()

    def _is_loss(self, name: str) -> bool:
        return "loss" in name.lower()

    def _is_input(self, name: str) -> bool:
        return name.lower() == "input"

    def _build_metric(self):
        loss_node = self.cfg.net.get("lossNode")
        if loss_node is not None:
            task_type = loss_node.get("taskType", "")
        else:
            task_type = "classification"

        if task_type == "classification":
            num_classes = getattr(self.cfg.net, "num_classes", 10)
            return Accuracy(task="multiclass", num_classes=num_classes)
        elif task_type == "regression":
            return MeanSquaredError()
        else:
            return Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        root_id = self.cfg.net.root
        node_inputs = {root_id: [x]}
        in_degrees = self._compute_in_degrees()
        processed_count = {node_id: 0 for node_id in in_degrees}

        queue = [root_id]
        final_output = x

        while queue:
            curr_id = queue.pop(0)
            curr_node = self.cfg.net.nodes[curr_id]
            inputs = node_inputs[curr_id]

            if curr_node.type == "join":
                if curr_id in self.module_dict:
                    out = self.module_dict[curr_id](inputs)
                else:
                    raise NotImplementedError(
                        f"Join node {curr_id} has no instantiated module"
                    )
            else:
                inp = inputs[0]

                needs_flatten = False
                if self._is_input(curr_node.get("stereotype", "")):
                    needs_flatten = True
                elif curr_node.type == "sequential" and hasattr(curr_node, "layers"):
                    for l in curr_node.layers:
                        lt = l.get("_target_", "")
                        if "Linear" in lt:
                            needs_flatten = True
                            break

                if needs_flatten and len(inp.shape) > 2:
                    inp = inp.view(inp.size(0), -1)

                if curr_id in self.module_dict:
                    out = self.module_dict[curr_id](inp)
                else:
                    out = inp

            final_output = out

            for child_id in curr_node.children:
                if child_id not in node_inputs:
                    node_inputs[child_id] = []
                node_inputs[child_id].append(out)
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

    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, self.parameters())
