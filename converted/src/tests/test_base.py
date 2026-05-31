import torch
import pytest
from omegaconf import OmegaConf

from net.base import Net


def _make_cfg(nodes: dict, root: str = "a", loss_node: dict | None = None,
              num_classes: int | None = None) -> OmegaConf:
    loss_node = loss_node or {
        "stereotype": "CrossEntropyLoss",
        "_target_": "torch.nn.CrossEntropyLoss",
        "taskType": "classification",
    }
    d = {
        "net": {
            "root": root,
            "nodes": nodes,
            "lossNode": loss_node,
        },
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
    }
    if num_classes is not None:
        d["net"]["num_classes"] = num_classes
    return OmegaConf.create(d)


# -- _compute_in_degrees ------------------------------------------------------------

class TestComputeInDegrees:
    def test_simple_chain(self):
        cfg = _make_cfg({
            "a": {"type": "module", "stereotype": "Input", "children": ["b"]},
            "b": {"type": "module", "stereotype": "Linear", "children": ["c"]},
            "c": {"type": "module", "stereotype": "ReLU", "children": []},
        }, root="a")
        net = Net(cfg)
        deg = net._compute_in_degrees()
        assert deg == {"a": 0, "b": 1, "c": 1}

    def test_fork(self):
        cfg = _make_cfg({
            "a": {"type": "module", "stereotype": "Input", "children": ["b", "c"]},
            "b": {"type": "module", "stereotype": "ReLU", "children": []},
            "c": {"type": "module", "stereotype": "Tanh", "children": []},
        }, root="a")
        net = Net(cfg)
        deg = net._compute_in_degrees()
        assert deg == {"a": 0, "b": 1, "c": 1}

    def test_join(self):
        cfg = _make_cfg({
            "a": {"type": "module", "stereotype": "Input", "children": ["c"]},
            "b": {"type": "module", "stereotype": "Linear", "children": ["c"]},
            "c": {"type": "join", "stereotype": "Addition", "children": []},
        }, root="a")
        net = Net(cfg)
        assert net._compute_in_degrees() == {"a": 0, "b": 0, "c": 2}

    def test_diamond(self):
        cfg = _make_cfg({
            "a": {"type": "module", "stereotype": "Input", "children": ["b", "c"]},
            "b": {"type": "module", "stereotype": "ReLU", "children": ["d"]},
            "c": {"type": "module", "stereotype": "Tanh", "children": ["d"]},
            "d": {"type": "join", "stereotype": "Addition", "children": []},
        }, root="a")
        net = Net(cfg)
        assert net._compute_in_degrees() == {"a": 0, "b": 1, "c": 1, "d": 2}

    def test_child_not_in_nodes(self):
        """Edge case: children referencing nodes not in the nodes dict."""
        cfg = _make_cfg({
            "a": {"type": "module", "stereotype": "Input", "children": ["missing"]},
        }, root="a")
        net = Net(cfg)
        deg = net._compute_in_degrees()
        assert deg == {"a": 0}  # missing not in deg, no crash


# -- _is_loss / _is_input / _build_metric ------------------------------------------

class TestClassifierHelpers:
    def test_is_loss_positive(self):
        net = Net(_make_cfg({"a": {"type": "module", "stereotype": "Input", "children": []}}, root="a"))
        assert net._is_loss("CrossEntropyLoss")
        assert net._is_loss("MSELoss")
        assert net._is_loss("BCELoss")

    def test_is_loss_negative(self):
        net = Net(_make_cfg({"a": {"type": "module", "stereotype": "Input", "children": []}}, root="a"))
        assert not net._is_loss("Linear")
        assert not net._is_loss("ReLU")

    def test_is_input(self):
        net = Net(_make_cfg({"a": {"type": "module", "stereotype": "Input", "children": []}}, root="a"))
        assert net._is_input("Input")
        assert net._is_input("Fork")
        assert not net._is_input("Linear")

    def test_build_metric_classification(self):
        cfg = _make_cfg({"a": {"type": "module", "stereotype": "Input", "children": []}}, root="a",
                        num_classes=10)
        net = Net(cfg)
        from torchmetrics.classification import MulticlassAccuracy
        assert isinstance(net.metric, MulticlassAccuracy)

    def test_build_metric_regression(self):
        loss_node = {"stereotype": "MSELoss", "_target_": "torch.nn.MSELoss", "taskType": "regression"}
        cfg = _make_cfg({"a": {"type": "module", "stereotype": "Input", "children": []}}, root="a",
                        loss_node=loss_node)
        net = Net(cfg)
        from torchmetrics import MeanSquaredError
        assert isinstance(net.metric, MeanSquaredError)


# -- Net.__init__ dispatch ----------------------------------------------------------

class TestNetInit:
    def test_sequential_node(self):
        cfg = _make_cfg({
            "a": {
                "type": "sequential",
                "stereotype": "Input",
                "children": [],
                "layers": [
                    {"stereotype": "Linear", "_target_": "torch.nn.Linear",
                     "in_features": 4, "out_features": 8, "bias": False},
                    {"stereotype": "ReLU", "_target_": "torch.nn.ReLU"},
                ],
            },
        }, root="a")
        net = Net(cfg)
        assert "a" in net.module_dict
        assert isinstance(net.module_dict["a"], torch.nn.Sequential)

    def test_module_node(self):
        cfg = _make_cfg({
            "a": {"type": "module", "stereotype": "Input", "children": []},
            "b": {
                "type": "module", "stereotype": "Linear", "children": [],
                "layer": {"stereotype": "Linear", "_target_": "torch.nn.Linear",
                          "in_features": 4, "out_features": 8, "bias": False},
            },
        }, root="a")
        net = Net(cfg)
        assert "b" in net.module_dict
        assert isinstance(net.module_dict["b"], torch.nn.Linear)

    def test_join_node_with_inputs(self):
        cfg = _make_cfg({
            "a": {"type": "module", "stereotype": "Input", "children": ["j"]},
            "j": {
                "type": "join", "stereotype": "Addition", "children": [],
                "layer": {"stereotype": "Addition", "_target_": "ops.Addition"},
                "inputs": ["a"],
            },
        }, root="a")
        net = Net(cfg)
        assert "j" in net.module_dict
        assert net.input_order["j"] == ["a"]

    def test_subflow_node(self):
        cfg = _make_cfg({
            "a": {"type": "module", "stereotype": "Input", "children": ["sf"]},
            "sf": {
                "type": "subflow", "stereotype": "Repeat", "children": [],
                "_target_": "ops.Repeat", "_recursive_": False,
                "entry_node": "lin1",
                "internal_nodes": {
                    "lin1": {
                        "type": "module", "children": [],
                        "stereotype": "Linear", "_target_": "torch.nn.Linear",
                        "in_features": 4, "out_features": 8, "bias": False,
                    },
                },
                "iterations": 1,
            },
        }, root="a")
        net = Net(cfg)
        assert "sf" in net.module_dict

    def test_loss_nodes_excluded(self):
        cfg = _make_cfg({
            "a": {"type": "module", "stereotype": "Input", "children": ["loss"]},
            "loss": {"type": "module", "stereotype": "CrossEntropyLoss", "children": [],
                     "layer": {"stereotype": "CrossEntropyLoss", "_target_": "torch.nn.CrossEntropyLoss"}},
        }, root="a")
        net = Net(cfg)
        assert "loss" not in net.module_dict


# -- Net.forward BFS ---------------------------------------------------------------

class TestNetForward:
    def test_chain_identity(self):
        cfg = _make_cfg({
            "a": {"type": "module", "stereotype": "Input", "children": []},
        }, root="a")
        net = Net(cfg)
        x = torch.randn(2, 10)
        out = net(x)
        assert out.shape == (2, 10)

    def test_chain_linear(self):
        cfg = _make_cfg({
            "a": {"type": "module", "stereotype": "Input", "children": ["lin"]},
            "lin": {
                "type": "module", "stereotype": "Linear", "children": [],
                "layer": {"stereotype": "Linear", "_target_": "torch.nn.Linear",
                          "in_features": 4, "out_features": 8, "bias": False},
            },
        }, root="a")
        net = Net(cfg)
        x = torch.randn(2, 4)
        out = net(x)
        assert out.shape == (2, 8)

    def test_fork_addition_join(self):
        """x -> fork -> [lin_a, lin_b] -> add -> out."""
        cfg = _make_cfg({
            "fork": {"type": "module", "stereotype": "Input", "children": ["lin_a", "lin_b"]},
            "lin_a": {
                "type": "module", "stereotype": "Linear", "children": ["add"],
                "layer": {"stereotype": "Linear", "_target_": "torch.nn.Linear",
                          "in_features": 4, "out_features": 8, "bias": False},
            },
            "lin_b": {
                "type": "module", "stereotype": "Linear", "children": ["add"],
                "layer": {"stereotype": "Linear", "_target_": "torch.nn.Linear",
                          "in_features": 4, "out_features": 8, "bias": False},
            },
            "add": {
                "type": "join", "stereotype": "Addition", "children": [],
                "layer": {"stereotype": "Addition", "_target_": "ops.Addition"},
                "inputs": ["lin_a", "lin_b"],
            },
        }, root="fork")
        net = Net(cfg)
        x = torch.randn(2, 4)
        out = net(x)
        assert out.shape == (2, 8)

    def test_diamond(self):
        """x -> fork -> [a, b] -> join -> out.
        a: Linear(4,8), b: Linear(4,8), join: Addition -> [2,8].
        Then join -> Linear(8,16)."""
        cfg = _make_cfg({
            "fork": {"type": "module", "stereotype": "Input", "children": ["lin_a", "lin_b"]},
            "lin_a": {
                "type": "module", "stereotype": "Linear", "children": ["add"],
                "layer": {"stereotype": "Linear", "_target_": "torch.nn.Linear",
                          "in_features": 4, "out_features": 8, "bias": False},
            },
            "lin_b": {
                "type": "module", "stereotype": "Linear", "children": ["add"],
                "layer": {"stereotype": "Linear", "_target_": "torch.nn.Linear",
                          "in_features": 4, "out_features": 8, "bias": False},
            },
            "add": {
                "type": "join", "stereotype": "Addition", "children": ["out"],
                "layer": {"stereotype": "Addition", "_target_": "ops.Addition"},
                "inputs": ["lin_a", "lin_b"],
            },
            "out": {
                "type": "module", "stereotype": "Linear", "children": [],
                "layer": {"stereotype": "Linear", "_target_": "torch.nn.Linear",
                          "in_features": 8, "out_features": 16, "bias": False},
            },
        }, root="fork")
        net = Net(cfg)
        x = torch.randn(2, 4)
        out = net(x)
        assert out.shape == (2, 16)

    def test_subflow_in_net(self):
        """x -> subflow(lin: 4->8) -> out."""
        cfg = _make_cfg({
            "a": {"type": "module", "stereotype": "Input", "children": ["sf"]},
            "sf": {
                "type": "subflow", "stereotype": "Repeat", "children": [],
                "_target_": "ops.Repeat", "_recursive_": False,
                "entry_node": "lin1",
                "internal_nodes": {
                    "lin1": {
                        "type": "module", "children": [],
                        "stereotype": "Linear", "_target_": "torch.nn.Linear",
                        "in_features": 4, "out_features": 8, "bias": False,
                    },
                },
                "iterations": 1,
            },
        }, root="a")
        net = Net(cfg)
        x = torch.randn(2, 4)
        out = net(x)
        assert out.shape == (2, 8)

    def test_three_way_fork(self):
        """Fork passthrough: Input -> 3 children, no join (just verify traversal)."""
        cfg = _make_cfg({
            "a": {"type": "module", "stereotype": "Input", "children": ["b", "c", "d"]},
            "b": {"type": "module", "stereotype": "ReLU", "children": []},
            "c": {"type": "module", "stereotype": "Tanh", "children": []},
            "d": {"type": "module", "stereotype": "Sigmoid", "children": []},
        }, root="a")
        net = Net(cfg)
        x = torch.randn(2, 10)
        out = net(x)
        # fork returns last processed output (d)
        assert out.shape == (2, 10)
