"""Tests for portable pip-wheel exports of trained NNModelling models."""

from __future__ import annotations

import importlib
import json
import sys
import zipfile
from pathlib import Path

import torch
from PIL import Image
from omegaconf import OmegaConf
from safetensors.torch import save_file

from convert import build_hydra_configs
from model_package.adapters import adapter_from_spec, adapter_spec_from_dataset_config
from model_package.exporter import build_model_wheel
from model_package.runtime import GraphNet
from net.base import Net


def _write_toy_artifact(root: Path) -> torch.Tensor:
    """Materialize a small resolved architecture and deterministic weights."""

    config = {
        "net": {
            "root": "input",
            "nodes": {
                "input": {
                    "type": "sequential",
                    "children": [],
                    "layers": [{"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 1}],
                }
            },
        },
        "dataset": {"_target_": "dataset.ds.Dataset"},
    }
    (root / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    model = GraphNet(config["net"])
    linear = model.module_dict["input"][0]
    linear.weight.data.copy_(torch.tensor([[2.0, -1.0]]))
    linear.bias.data.copy_(torch.tensor([0.5]))
    save_file(model.state_dict(), root / "weights.safetensors")
    return model(torch.tensor([[3.0, 4.0]]))


def test_build_model_wheel_installs_a_self_contained_inference_package(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    expected = _write_toy_artifact(artifact_dir)

    wheel = build_model_wheel(artifact_dir, package_name="nnm_toy_model", version="0.1.0")

    assert wheel.name == "nnm_toy_model-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert {
        "nnm_toy_model/__init__.py",
        "nnm_toy_model/runtime.py",
        "nnm_toy_model/adapters.py",
        "nnm_toy_model/architecture.json",
        "nnm_toy_model/weights.safetensors",
        "nnm_toy_model-0.1.0.dist-info/METADATA",
    }.issubset(names)

    sys.path.insert(0, str(wheel))
    try:
        package = importlib.import_module("nnm_toy_model")
        model = package.load_model()
        assert torch.equal(model.predict_tensor(torch.tensor([[3.0, 4.0]])), expected)
    finally:
        sys.path.remove(str(wheel))
        for module_name in list(sys.modules):
            if module_name == "nnm_toy_model" or module_name.startswith("nnm_toy_model."):
                del sys.modules[module_name]


def test_export_uses_dataset_adapter_specs_without_instantiating_the_dataset():
    spec = adapter_spec_from_dataset_config({"_target_": "dataset.mnist.MNISTDataset"})

    assert spec == {
        "kind": "image",
        "version": 1,
        "channels": 1,
        "size": [28, 28],
        "mean": [0.1307],
        "std": [0.3081],
    }


def test_enron_export_declares_a_text_adapter():
    spec = adapter_spec_from_dataset_config(
        {
            "_target_": "dataset.enron_spam.EnronSpamDataset",
            "model_name": "bert-base-uncased",
            "max_length": 64,
        }
    )

    assert spec == {
        "kind": "text",
        "version": 1,
        "model_name": "bert-base-uncased",
        "max_length": 64,
    }


def test_text_adapter_tokenizes_one_email(monkeypatch):
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Sale ends today"
            assert kwargs == {
                "return_tensors": "pt",
                "truncation": True,
                "max_length": 4,
                "padding": "max_length",
            }
            return {"input_ids": torch.tensor([[101, 1, 2, 102]])}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name):
            assert model_name == "bert-base-uncased"
            return Tokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer", AutoTokenizer)
    adapter = adapter_from_spec(
        {"kind": "text", "version": 1, "model_name": "bert-base-uncased", "max_length": 4}
    )

    assert torch.equal(adapter.to_tensor("Sale ends today"), torch.tensor([[101, 1, 2, 102]]))


def test_mnist_image_adapter_converts_one_image_to_a_normalized_batch_tensor():
    adapter = adapter_from_spec(adapter_spec_from_dataset_config({"_target_": "dataset.mnist.MNISTDataset"}))

    tensor = adapter.to_tensor(Image.new("L", (40, 30), color=255))

    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor.mean(), torch.tensor((1 - 0.1307) / 0.3081), atol=1e-5)


def test_export_rejects_an_invalid_distribution_name(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_toy_artifact(artifact_dir)

    try:
        build_model_wheel(artifact_dir, package_name="classifier", version="0.1.0")
    except ValueError as error:
        assert "package_name" in str(error)
    else:
        raise AssertionError("unsafe package name was accepted")


def test_graph_runtime_executes_a_converted_skip_connection_graph(tmp_path):
    """The wheel runtime accepts the graph format emitted by the editor converter."""

    fixture = Path(__file__).parents[3] / "examples" / "nntrees" / "mninst_skip.json"
    output_dir = tmp_path / "config"
    build_hydra_configs(str(fixture), output_dir=str(output_dir), num_classes=10)
    net_config = OmegaConf.to_container(
        OmegaConf.load(output_dir / "net" / "custom_sequence.yaml"), resolve=True
    )
    assert isinstance(net_config, dict)

    output = GraphNet(net_config)(torch.randn(2, 1, 28, 28))

    assert output.shape == (2, 10)


def test_exported_wheel_executes_a_converted_skip_connection_graph(tmp_path):
    """Copied custom operations remain importable from an installed wheel."""

    fixture = Path(__file__).parents[3] / "examples" / "nntrees" / "mninst_skip.json"
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    config_dir = tmp_path / "config"
    build_hydra_configs(str(fixture), output_dir=str(config_dir), num_classes=10)
    net_config = OmegaConf.to_container(
        OmegaConf.load(config_dir / "net" / "custom_sequence.yaml"), resolve=True
    )
    assert isinstance(net_config, dict)
    config = {
        "net": net_config,
        "dataset": {"_target_": "dataset.mnist.MNISTDataset"},
    }
    (artifact_dir / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    source_model = Net(
        OmegaConf.create({"net": net_config, "optimizer": {"_target_": "torch.optim.Adam"}})
    )
    save_file(source_model.state_dict(), artifact_dir / "weights.safetensors")
    inputs = torch.randn(2, 1, 28, 28)
    expected = source_model(inputs)

    wheel = build_model_wheel(artifact_dir, package_name="nnm_skip_model")

    sys.path.insert(0, str(wheel))
    try:
        package = importlib.import_module("nnm_skip_model")
        actual = package.load_model().predict_tensor(inputs)
        assert torch.equal(actual, expected)
    finally:
        sys.path.remove(str(wheel))
        for module_name in list(sys.modules):
            if module_name == "nnm_skip_model" or module_name.startswith("nnm_skip_model."):
                del sys.modules[module_name]
