from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

from convert import build_hydra_configs
from net.base import Net

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "examples" / "nntrees"


class TestTransformerClassifierPipeline:
    """Full pipeline: NNTree JSON → convert.py → Hydra config → Net → forward."""

    @pytest.fixture
    def cfg_dir(self, tmp_path):
        out = tmp_path / "cfg"
        json_path = FIXTURES_DIR / "transformer_classifier.json"
        build_hydra_configs(str(json_path), output_dir=str(out), num_classes=2)
        return out

    def test_yaml_files_created(self, cfg_dir):
        yamls = [
            "base.yaml",
            "net/custom_sequence.yaml",
            "optimizer/adam.yaml",
            "trainer/default.yaml",
            "wandb/wandb.yaml",
            "dataset/dataset.yaml",
            "early_stopping/default.yaml",
        ]
        for y in yamls:
            assert (cfg_dir / y).exists(), f"Missing {y}"

    def test_net_yaml_structure(self, cfg_dir):
        net = OmegaConf.load(cfg_dir / "net" / "custom_sequence.yaml")
        assert net.root == "input"
        assert net.lossNode._target_ == "torch.nn.CrossEntropyLoss"
        assert net.lossNode.taskType == "classification"
        assert net.num_classes == 2
        assert "input" in net.nodes
        assert "encoder" in net.nodes
        assert "pool" in net.nodes

    def test_forward_pass(self, cfg_dir):
        """Load config with Hydra compose, instantiate Net, run forward."""
        with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
            cfg = compose(config_name="base")

        net = Net(cfg)

        # input: Embedding(30522, 128) expects token indices
        x = torch.randint(0, 100, (2, 16))
        out = net(x)
        assert out.shape == (2, 2)  # batch=2, num_classes=2

    def test_forward_preserves_grad(self, cfg_dir):
        """Output should require grad for training."""
        with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
            cfg = compose(config_name="base")

        net = Net(cfg)
        x = torch.randint(0, 100, (2, 16))
        out = net(x)
        assert out.requires_grad


class TestAutoEncoderPipeline:
    @pytest.fixture
    def cfg_dir(self, tmp_path):
        out = tmp_path / "cfg"
        json_path = FIXTURES_DIR / "auto_encoder.json"
        build_hydra_configs(str(json_path), output_dir=str(out))
        return out

    def test_net_yaml_regression(self, cfg_dir):
        net = OmegaConf.load(cfg_dir / "net" / "custom_sequence.yaml")
        assert net.lossNode._target_ == "torch.nn.MSELoss"
        assert net.lossNode.taskType == "regression"
        assert "num_classes" not in net

    def test_forward_pass(self, cfg_dir):
        with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
            cfg = compose(config_name="base")

        net = Net(cfg)
        x = torch.randn(2, 1, 28, 28)
        out = net(x)
        # autoencoder: output same shape as input
        assert out.shape == (2, 1, 28, 28)


class TestSkipConnectionsPipeline:
    @pytest.fixture
    def cfg_dir(self, tmp_path):
        out = tmp_path / "cfg"
        json_path = FIXTURES_DIR / "skip_connections_with_repetition.json"
        build_hydra_configs(str(json_path), output_dir=str(out), num_classes=10)
        return out

    def test_net_yaml(self, cfg_dir):
        net = OmegaConf.load(cfg_dir / "net" / "custom_sequence.yaml")
        assert net.lossNode._target_ == "torch.nn.CrossEntropyLoss"
        assert net.num_classes == 10
        # join nodes are inside subflow internal_nodes, not top-level
        # has subflow nodes
        sfs = [n for n in net.nodes.values() if n.type == "subflow"]
        assert len(sfs) >= 1

    def test_forward_pass(self, cfg_dir):
        with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
            cfg = compose(config_name="base")

        net = Net(cfg)
        x = torch.randn(2, 1, 28, 28)
        out = net(x)
        assert out.shape == (2, 10)
