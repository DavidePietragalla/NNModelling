import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


class Subflow(nn.Module):
    """Executes internal graph via BFS topological sort.

    Receives entry_node + internal_nodes from Hydra with _recursive_: false,
    so internal_nodes is a raw DictConfig. Subflow manually instantiates
    each internal node's module and manages the DAG execution.
    """

    def __init__(self, entry_node: str, internal_nodes: dict, **kwargs):
        super().__init__()
        self.entry_node = entry_node
        self.internal_nodes = internal_nodes
        self.module_dict = nn.ModuleDict()

        for node_id, cfg in internal_nodes.items():
            if isinstance(cfg, DictConfig):
                layer_dict = OmegaConf.to_container(cfg, resolve=True)
            else:
                layer_dict = dict(cfg)
            layer_dict.pop("stereotype", None)
            layer_dict.pop("taskType", None)
            layer_dict.pop("children", None)
            layer_dict.pop("type", None)
            if "_target_" in layer_dict:
                self.module_dict[node_id] = instantiate(layer_dict)

        self.in_degrees: dict[str, int] = {}
        for node_id in internal_nodes:
            self.in_degrees[node_id] = 0
        for node_id, cfg in internal_nodes.items():
            children = cfg.get("children", []) if isinstance(cfg, dict) else cfg.get("children", [])
            for child_id in children:
                self.in_degrees[child_id] = self.in_degrees.get(child_id, 0) + 1

    def forward(self, x):
        node_inputs: dict[str, list] = {self.entry_node: [x]}
        processed: dict[str, int] = {n: 0 for n in self.in_degrees}
        queue = [self.entry_node]
        final = x

        while queue:
            curr = queue.pop(0)

            if curr not in self.internal_nodes:
                continue

            cfg = self.internal_nodes[curr]
            node_type = cfg.get("type", "") if isinstance(cfg, dict) else cfg.type
            children = cfg.get("children", []) if isinstance(cfg, dict) else cfg.get("children", [])

            inputs = node_inputs.get(curr, [x])

            if node_type == "join":
                out = self.module_dict[curr](inputs) if curr in self.module_dict else inputs[0]
            else:
                inp = inputs[0]
                out = self.module_dict[curr](inp) if curr in self.module_dict else inp

            final = out

            for child_id in children:
                if child_id not in node_inputs:
                    node_inputs[child_id] = []
                node_inputs[child_id].append(out)
                processed[child_id] = processed.get(child_id, 0) + 1
                if processed[child_id] == self.in_degrees.get(child_id, 1):
                    queue.append(child_id)

        return final
