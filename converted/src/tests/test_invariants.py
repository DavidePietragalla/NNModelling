"""Numerical invariants for the backend-owned training and runtime paths.

These tests are fast: they build Hydra configs from the ``mninst`` (MNIST
classifier) and ``autoencoder_mnist`` fixtures, compose them, and run
forward/backward/optimizer steps on synthetic batches. They assert finite
losses and gradients, actual parameter updates after an optimizer step, safe
weights reload equivalence, and prediction cardinality/schema — without
training a full epoch, so no convergence assertions are needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from convert import build_hydra_configs
from model_package.runtime import GraphNet
from net.base import Net

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "examples" / "nntrees"
INPUT_SHAPE = (2, 1, 28, 28)


@pytest.fixture(scope="module")
def classifier_net(tmp_path_factory) -> Net:
    config_dir = tmp_path_factory.mktemp("cfg-classifier")
    build_hydra_configs(
        str(FIXTURES_DIR / "mninst_skip.json"),
        output_dir=str(config_dir),
        num_classes=10,
    )
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="base")
    return Net(cfg)


@pytest.fixture(scope="module")
def autoencoder_net(tmp_path_factory) -> Net:
    config_dir = tmp_path_factory.mktemp("cfg-autoencoder")
    build_hydra_configs(
        str(FIXTURES_DIR / "auto_encoder.json"),
        output_dir=str(config_dir),
        num_classes=None,
    )
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="base")
    return Net(cfg)


def _run_backward_step(net: Net, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """One forward/backward/optimizer step with invariant checks."""
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    optimizer.zero_grad()
    output = net(x)
    loss = net.loss_fn(output, y)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    loss.backward()
    gradients = [parameter.grad for parameter in net.parameters() if parameter.grad is not None]
    assert gradients, "backward produced no gradients"
    for gradient in gradients:
        assert torch.isfinite(gradient).all(), "non-finite gradient found"
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients), "all gradients are zero"
    before = {id(parameter): parameter.detach().clone() for parameter in net.parameters()}
    optimizer.step()
    changed = [
        not torch.equal(parameter.detach(), before[id(parameter)])
        for parameter in net.parameters()
    ]
    assert any(changed), "optimizer step did not change any parameter"
    return {"loss": loss.item()}


def test_classifier_forward_backward_updates_parameters(classifier_net):
    """The mninst classifier backpropagates finite gradients and updates weights."""
    x = torch.randn(*INPUT_SHAPE)
    y = torch.randint(0, 10, (INPUT_SHAPE[0],))
    metrics = _run_backward_step(classifier_net, x, y)
    assert metrics["loss"] >= 0


def test_autoencoder_forward_backward_updates_parameters(autoencoder_net):
    """The autoencoder backpropagates finite gradients and updates weights."""
    x = torch.randn(*INPUT_SHAPE)
    metrics = _run_backward_step(autoencoder_net, x, y=x)
    assert metrics["loss"] >= 0


def test_classifier_prediction_cardinality_and_schema(classifier_net):
    """Classification predictions are (B, 10) float32 and finite."""
    classifier_net.eval()
    x = torch.randn(*INPUT_SHAPE)
    with torch.inference_mode():
        output = classifier_net(x)
    assert tuple(output.shape) == (2, 10)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_autoencoder_prediction_cardinality_and_schema(autoencoder_net):
    """Reconstruction predictions are (B, 1, 28, 28) float32 and finite."""
    autoencoder_net.eval()
    x = torch.randn(*INPUT_SHAPE)
    with torch.inference_mode():
        output = autoencoder_net(x)
    assert tuple(output.shape) == INPUT_SHAPE
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_safe_weights_reload_equivalence(classifier_net, tmp_path):
    """A safetensors round-trip reproduces identical eval outputs."""
    inputs = torch.randn(*INPUT_SHAPE, generator=torch.Generator().manual_seed(11))
    classifier_net.eval()
    with torch.inference_mode():
        before = classifier_net(inputs)

    weights_path = tmp_path / "weights.safetensors"
    save_file(classifier_net.state_dict(), str(weights_path))
    state = load_file(str(weights_path))
    reloaded = Net(classifier_net.cfg)
    reloaded.load_state_dict(state, strict=True)
    reloaded.eval()
    with torch.inference_mode():
        after = reloaded(inputs)

    assert torch.equal(before, after)


def test_safetensors_matches_graph_runtime_architecture(classifier_net, tmp_path):
    """The wheel runtime loads the same safetensors with identical output."""
    build_hydra_configs(
        str(FIXTURES_DIR / "mninst_skip.json"),
        output_dir=str(tmp_path / "config"),
        num_classes=10,
    )
    net_config = OmegaConf.to_container(
        OmegaConf.load(tmp_path / "config" / "net" / "custom_sequence.yaml"),
        resolve=True,
    )
    assert isinstance(net_config, dict)
    weights_path = tmp_path / "weights.safetensors"
    save_file(classifier_net.state_dict(), str(weights_path))
    state = load_file(str(weights_path))
    network = GraphNet(net_config)
    network.load_state_dict(state, strict=True)

    inputs = torch.randn(*INPUT_SHAPE, generator=torch.Generator().manual_seed(13))
    classifier_net.eval()
    network.eval()
    with torch.inference_mode():
        expected = classifier_net(inputs)
        actual = network(inputs)
    assert torch.equal(expected, actual)
