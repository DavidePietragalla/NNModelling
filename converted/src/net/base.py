import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as lit
from torchmetrics import Accuracy
from hydra.utils import instantiate
from omegaconf import DictConfig


class Net(lit.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(Net, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.module_dict = nn.ModuleDict()

        # Istanziazione dinamica tramite Hydra
        if hasattr(cfg.net, "nodes"):
            for node_id, node in cfg.net.nodes.items():
                if node.type == "sequential":
                    layers = []
                    for layer_cfg in node.layers:
                        if layer_cfg.get("stereotype") != "Input":
                            layers.append(instantiate(layer_cfg))
                    if layers:
                        self.module_dict[node_id] = nn.Sequential(*layers)

                elif node.type == "module" and hasattr(node, "layer"):
                    if node.layer.get("stereotype") != "Input":
                        self.module_dict[node_id] = instantiate(node.layer)

        # Istanzia il modulo Loss fornito da Hydra
        self.loss_fn = (
            instantiate(cfg.net.lossNode)
            if cfg.net.get("lossNode")
            else nn.CrossEntropyLoss()
        )

    def forward(self, x):
        # BFS per calcolare il forward pass dell'albero del grafo
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

            # Applica l'operazione in base al tipo di nodo
            if curr_node.type == "join":
                # Gestione dinamica dei join operativi
                if curr_node.stereotype == "Addition":
                    out = sum(inputs)
                else:
                    out = torch.cat(inputs, dim=-1)  # Fallback
            else:
                inp = inputs[0]  # Nodi standard prendono un solo input
                # Flatten automatico se è l'input generico prima dei layer Lineari classici
                if curr_node.get("stereotype") == "Input" or (
                    curr_node.type == "sequential" and "Linear" in str(curr_node.layers)
                ):
                    if len(inp.shape) > 2:
                        inp = inp.view(inp.size(0), -1)

                if curr_id in self.module_dict:
                    out = self.module_dict[curr_id](inp)
                else:
                    out = inp  # Nodi solo per pass-through o Input root

            final_output = out

            # Propagazione ai figli
            for child_id in curr_node.childrens:
                if child_id not in node_inputs:
                    node_inputs[child_id] = []
                node_inputs[child_id].append(out)
                processed_count[child_id] += 1

                # Aggiunge alla coda solo quando tutti gli input per il nodo figlio sono calcolati
                if processed_count[child_id] == in_degrees[child_id]:
                    queue.append(child_id)

        return final_output

    def _compute_in_degrees(self):
        """Calcola i gradi di ingresso per capire quando un nodo Join può essere eseguito."""
        deg = {node_id: 0 for node_id in self.cfg.net.nodes}
        for node in self.cfg.net.nodes.values():
            for child_id in node.childrens:
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
        self.log("val_acc", accuracy(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", accuracy(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, self.parameters())


accuracy = Accuracy(task="multiclass", num_classes=10)
