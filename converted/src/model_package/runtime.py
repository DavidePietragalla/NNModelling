"""Inference-only graph runtime copied into every exported model wheel."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import ExitStack
from importlib.resources import as_file, files
from typing import Any

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors.torch import load_file

from .adapters import InputAdapter, adapter_from_spec


class GraphNet(nn.Module):
    """Execute a converted NNTree without Lightning training concerns."""

    def __init__(self, net_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.net_config = OmegaConf.create(net_config)
        self.module_dict = nn.ModuleDict()
        self.input_order: dict[str, list[str]] = {}
        nodes = self.net_config.get("nodes", {})
        for node_id, node in nodes.items():
            node_type = node.get("type")
            if node_type == "sequential":
                layers = [
                    self._instantiate_layer(layer)
                    for layer in node.get("layers", [])
                    if self._has_target(layer)
                ]
                if layers:
                    self.module_dict[node_id] = nn.Sequential(*layers)
            elif node_type in {"module", "join"} and self._has_target(node.get("layer", {})):
                self.module_dict[node_id] = self._instantiate_layer(node.layer)
                inputs = node.get("inputs", [])
                if inputs:
                    self.input_order[node_id] = list(inputs)
            elif node_type == "subflow" and self._has_target(node):
                self.module_dict[node_id] = self._instantiate_layer(node)

    @staticmethod
    def _has_target(config: Mapping[str, Any]) -> bool:
        return isinstance(config.get("_target_"), str)

    @staticmethod
    def _instantiate_layer(config: Mapping[str, Any]) -> nn.Module:
        values = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
        assert isinstance(values, dict)
        for key in ("stereotype", "taskType", "children", "type", "inputs"):
            values.pop(key, None)
        return instantiate(values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        root_id = self.net_config.root
        node_inputs: dict[str, dict[str, torch.Tensor]] = {root_id: {"_in": x}}
        in_degrees = self._compute_in_degrees()
        processed = {node_id: 0 for node_id in in_degrees}
        queue = [root_id]
        output = x
        while queue:
            node_id = queue.pop(0)
            node = self.net_config.nodes[node_id]
            inputs = node_inputs[node_id]
            if node.type == "join":
                ordered = [inputs[parent] for parent in self.input_order.get(node_id, inputs) if parent in inputs]
                if node_id not in self.module_dict:
                    raise NotImplementedError(f"Join node {node_id} has no instantiated module")
                output = self.module_dict[node_id](ordered)
            else:
                output = self.module_dict[node_id](next(iter(inputs.values()))) if node_id in self.module_dict else next(iter(inputs.values()))
            for child_id in node.children:
                node_inputs.setdefault(child_id, {})[node_id] = output
                processed[child_id] += 1
                if processed[child_id] == in_degrees[child_id]:
                    queue.append(child_id)
        return output

    def _compute_in_degrees(self) -> dict[str, int]:
        degrees = {node_id: 0 for node_id in self.net_config.nodes}
        for node in self.net_config.nodes.values():
            for child_id in node.children:
                if child_id in degrees:
                    degrees[child_id] += 1
        return degrees


class InferenceModel:
    """Public inference facade with raw-tensor and adapter-aware methods."""

    def __init__(self, network: GraphNet, adapter: InputAdapter, device: torch.device) -> None:
        self.network = network.to(device).eval()
        self.adapter = adapter
        self.device = device

    @torch.inference_mode()
    def predict_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run inference on an already preprocessed batch tensor."""

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("predict_tensor expects a torch.Tensor")
        return self.network(tensor.to(self.device))

    @torch.inference_mode()
    def predict(self, value: object) -> torch.Tensor:
        """Adapt one user-facing input and run inference."""

        return self.predict_tensor(self.adapter.to_tensor(value))


def load_model(device: str | torch.device = "cpu") -> InferenceModel:
    """Load this wheel's architecture, safe weights, and input adapter."""

    package_files = files(__package__)
    architecture = json.loads(package_files.joinpath("architecture.json").read_text(encoding="utf-8"))
    network = GraphNet(architecture["net"])
    with ExitStack() as stack:
        weights_path = stack.enter_context(as_file(package_files.joinpath("weights.safetensors")))
        state_dict = load_file(str(weights_path), device="cpu")
    network.load_state_dict(state_dict, strict=True)
    return InferenceModel(network, adapter_from_spec(architecture["input_adapter"]), torch.device(device))
