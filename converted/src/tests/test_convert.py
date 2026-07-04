# NNModelling — DSL for designing neural networks via visual node editor
# Copyright (C) 2026  Luca Sforza
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from convert import (
    parse_params,
    build_layer_config,
    _build_nested_subflow_config,
    build_hydra_configs,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "examples" / "nntrees"


# -- parse_params -------------------------------------------------------------------

class TestParseParams:
    def test_int(self):
        assert parse_params({"x": {"value": "42"}}) == {"x": 42}

    def test_float(self):
        assert parse_params({"lr": {"value": "0.001"}}) == {"lr": 0.001}

    def test_bool_true(self):
        assert parse_params({"bias": {"value": "True"}}) == {"bias": True}

    def test_bool_false(self):
        assert parse_params({"bias": {"value": "False"}}) == {"bias": False}

    def test_bool_lowercase_true(self):
        assert parse_params({"flag": {"value": "true"}}) == {"flag": True}

    def test_bool_lowercase_false(self):
        assert parse_params({"flag": {"value": "false"}}) == {"flag": False}

    def test_none(self):
        assert parse_params({"x": {"value": "None"}}) == {"x": None}

    def test_empty_string(self):
        assert parse_params({"x": {"value": ""}}) == {"x": None}

    def test_null_skipped(self):
        assert parse_params({"x": {"value": "null"}}) == {}

    def test_undefined_skipped(self):
        assert parse_params({"x": {"value": "undefined"}}) == {}

    def test_string_fallback(self):
        assert parse_params({"name": {"value": "relu"}}) == {"name": "relu"}

    def test_list(self):
        assert parse_params({"shape": {"value": "[1, 2, 3]"}}) == {"shape": [1, 2, 3]}

    def test_dict(self):
        result = parse_params({"cfg": {"value": "{'a': 1}"}})
        assert result == {"cfg": {"a": 1}}


# -- build_layer_config -------------------------------------------------------------

class TestBuildLayerConfig:
    def test_input_skipped(self):
        cfg = build_layer_config({"stereotype": "Input", "pythonClassName": "nn.Input"})
        assert "_target_" not in cfg
        assert cfg["stereotype"] == "Input"

    def test_fork_skipped(self):
        cfg = build_layer_config({"stereotype": "Fork", "pythonClassName": "nn.Fork"})
        assert "_target_" not in cfg
        assert cfg["stereotype"] == "Fork"

    def test_nn_to_torch_nn(self):
        cfg = build_layer_config({"stereotype": "Linear", "pythonClassName": "nn.Linear"})
        assert cfg["_target_"] == "torch.nn.Linear"

    def test_torch_prefix_unchanged(self):
        cfg = build_layer_config({"stereotype": "ReLU", "pythonClassName": "torch.nn.ReLU"})
        assert cfg["_target_"] == "torch.nn.ReLU"

    def test_dotted_path_as_is(self):
        cfg = build_layer_config({"stereotype": "Addition", "pythonClassName": "ops.Addition"})
        assert cfg["_target_"] == "ops.Addition"

    def test_plain_fallback(self):
        cfg = build_layer_config({"stereotype": "Tanh"})  # no pythonClassName -> falls back to stereotype
        assert cfg["_target_"] == "torch.nn.Tanh"

    def test_stereotype_as_class_when_pcn_empty(self):
        cfg = build_layer_config({"stereotype": "GELU"})
        assert cfg["_target_"] == "torch.nn.GELU"

    def test_task_type_preserved(self):
        cfg = build_layer_config({"stereotype": "Linear", "taskType": "classification"})
        assert cfg["taskType"] == "classification"

    def test_params_merged(self):
        cfg = build_layer_config({
            "stereotype": "Linear",
            "pythonClassName": "nn.Linear",
            "params": {"in_features": {"value": "784"}, "out_features": {"value": "10"}},
        })
        assert cfg["in_features"] == 784
        assert cfg["out_features"] == 10

    def test_empty_params(self):
        cfg = build_layer_config({"stereotype": "ReLU"})
        assert cfg["_target_"] == "torch.nn.ReLU"


# -- _build_nested_subflow_config --------------------------------------------------

class TestBuildNestedSubflowConfig:
    def test_plain_subflow(self):
        data = {
            "entryNode": "lin1",
            "nodes": {
                "lin1": {
                    "type": "module",
                    "stereotype": "Linear",
                    "pythonClassName": "nn.Linear",
                    "children": [],
                    "params": {"in_features": {"value": "4"}, "out_features": {"value": "8"}},
                }
            },
            "params": {},
        }
        cfg = _build_nested_subflow_config(data)
        assert cfg["entry_node"] == "lin1"
        assert cfg["_target_"] == "ops.Subflow"
        assert cfg["_recursive_"] is False
        assert "lin1" in cfg["internal_nodes"]
        assert cfg["internal_nodes"]["lin1"]["_target_"] == "torch.nn.Linear"

    def test_dotted_python_class(self):
        data = {
            "entryNode": "a",
            "pythonClassName": "ops.Repeat",
            "nodes": {"a": {"type": "module", "stereotype": "Identity", "children": []}},
            "params": {"iterations": {"value": "3"}},
        }
        cfg = _build_nested_subflow_config(data)
        assert cfg["_target_"] == "ops.Repeat"
        assert cfg["iterations"] == 3

    def test_nested_subflow_inside(self):
        data = {
            "entryNode": "inner",
            "nodes": {
                "inner": {
                    "type": "subflow",
                    "stereotype": "Repeat",
                    "pythonClassName": "ops.Repeat",
                    "entryNode": "a",
                    "nodes": {
                        "a": {"type": "module", "stereotype": "Linear", "pythonClassName": "nn.Linear",
                               "children": [], "params": {"in_features": {"value": "4"}, "out_features": {"value": "8"}}}
                    },
                    "params": {"iterations": {"value": "2"}},
                }
            },
            "params": {},
        }
        cfg = _build_nested_subflow_config(data)
        inner = cfg["internal_nodes"]["inner"]
        assert inner["type"] == "subflow"
        assert inner["_target_"] == "ops.Repeat"
        assert inner["entry_node"] == "a"
        assert "a" in inner["internal_nodes"]

    def test_join_node_inside(self):
        data = {
            "entryNode": "add",
            "nodes": {
                "add": {
                    "type": "join",
                    "stereotype": "Addition",
                    "pythonClassName": "ops.Addition",
                    "children": [],
                    "inputs": ["a", "b"],
                    "params": {},
                }
            },
            "params": {},
        }
        cfg = _build_nested_subflow_config(data)
        add = cfg["internal_nodes"]["add"]
        assert add["type"] == "join"
        assert add["inputs"] == ["a", "b"]
        assert add["_target_"] == "ops.Addition"


# -- build_hydra_configs -----------------------------------------------------------

class TestBuildHydraConfigs:
    def _assert_yaml_structure(self, output_dir: Path):
        """Verify all expected YAML files exist."""
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
            assert (output_dir / y).exists(), f"Missing {y}"

    def test_transformer_classifier(self, tmp_path, transformer_classifier_json):
        out = tmp_path / "cfg"
        json_path = FIXTURES_DIR / "transformer_classifier.json"
        build_hydra_configs(str(json_path), output_dir=str(out), num_classes=2)
        self._assert_yaml_structure(out)

        net = OmegaConf.load(out / "net" / "custom_sequence.yaml")
        assert net.root == "input"
        assert "lossNode" in net
        assert net.lossNode._target_ == "torch.nn.CrossEntropyLoss"
        assert net.lossNode.taskType == "classification"
        assert net.num_classes == 2

        # input node: sequential
        assert net.nodes.input.type == "sequential"
        assert len(net.nodes.input.layers) >= 2

        # encoder node: subflow (Repeat)
        assert net.nodes.encoder.type == "subflow"
        assert net.nodes.encoder._target_ == "ops.Repeat"
        assert net.nodes.encoder._recursive_ is False
        assert "internal_nodes" in net.nodes.encoder

        # pool node: sequential
        assert net.nodes.pool.type == "sequential"

    def test_auto_encoder(self, tmp_path, auto_encoder_json):
        out = tmp_path / "cfg"
        json_path = FIXTURES_DIR / "auto_encoder.json"
        build_hydra_configs(str(json_path), output_dir=str(out))
        self._assert_yaml_structure(out)

        net = OmegaConf.load(out / "net" / "custom_sequence.yaml")
        assert net.lossNode._target_ == "torch.nn.MSELoss"
        assert net.lossNode.taskType == "regression"
        assert "num_classes" not in net

        # subflow nodes exist
        subflow_nodes = [n for n in net.nodes.values() if n.type == "subflow"]
        assert len(subflow_nodes) >= 2
        for sf in subflow_nodes:
            assert sf._recursive_ is False
            assert "internal_nodes" in sf
            assert sf.entry_node

    def test_auto_encoder_nested(self, tmp_path, auto_encoder_nested_json):
        out = tmp_path / "cfg"
        json_path = FIXTURES_DIR / "auto_encoder_nested_submodels.json"
        build_hydra_configs(str(json_path), output_dir=str(out))
        self._assert_yaml_structure(out)

        net = OmegaConf.load(out / "net" / "custom_sequence.yaml")
        subflow_nodes = [n for n in net.nodes.values() if n.type == "subflow"]
        assert len(subflow_nodes) >= 2
        # check at least one has a nested subflow inside
        has_nested = any(
            any(v.type == "subflow" for v in sf.internal_nodes.values())
            for sf in subflow_nodes
        )
        assert has_nested, "Expected at least one subflow with nested subflow"

    def test_skip_connections(self, tmp_path, skip_connections_json):
        out = tmp_path / "cfg"
        json_path = FIXTURES_DIR / "skip_connections_with_repetition.json"
        build_hydra_configs(str(json_path), output_dir=str(out), num_classes=10)
        self._assert_yaml_structure(out)

        net = OmegaConf.load(out / "net" / "custom_sequence.yaml")
        # joins are inside subflow internal_nodes, not at top level
        sf_nodes = [n for n in net.nodes.values() if n.type == "subflow"]
        assert len(sf_nodes) >= 1
        joins_in_subflows = sum(
            1 for sf in sf_nodes
            for n in sf.internal_nodes.values()
            if n.type == "join"
        )
        assert joins_in_subflows >= 1

        # should have subflow nodes (Repeat)
        sf_nodes = [n for n in net.nodes.values() if n.type == "subflow"]
        assert len(sf_nodes) >= 1

    def test_mninst_skip(self, tmp_path, mninst_skip_json):
        out = tmp_path / "cfg"
        json_path = FIXTURES_DIR / "mninst_skip.json"
        build_hydra_configs(str(json_path), output_dir=str(out), num_classes=10)
        self._assert_yaml_structure(out)

        net = OmegaConf.load(out / "net" / "custom_sequence.yaml")
        assert net.lossNode._target_ == "torch.nn.CrossEntropyLoss"
        # simple joins at top level
        join_nodes = [n for n in net.nodes.values() if n.type == "join"]
        assert len(join_nodes) >= 1

    def test_classification_warning_no_num_classes(self, tmp_path, transformer_classifier_json, capsys):
        out = tmp_path / "cfg"
        json_path = FIXTURES_DIR / "transformer_classifier.json"
        build_hydra_configs(str(json_path), output_dir=str(out))
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Defaulting to 10" in captured.out
        net = OmegaConf.load(out / "net" / "custom_sequence.yaml")
        assert net.num_classes == 10

    def test_regression_no_num_classes(self, tmp_path, auto_encoder_json, capsys):
        out = tmp_path / "cfg"
        json_path = FIXTURES_DIR / "auto_encoder.json"
        build_hydra_configs(str(json_path), output_dir=str(out))
        net = OmegaConf.load(out / "net" / "custom_sequence.yaml")
        assert "num_classes" not in net
        # output should mention regression, not num_classes warning
        captured = capsys.readouterr()
        assert "regression" in captured.out

    def test_directory_structure(self, tmp_path, transformer_classifier_json):
        out = tmp_path / "cfg"
        json_path = FIXTURES_DIR / "transformer_classifier.json"
        build_hydra_configs(str(json_path), output_dir=str(out))
        subdirs = ["net", "optimizer", "trainer", "wandb", "dataset", "early_stopping"]
        for sd in subdirs:
            assert (out / sd).is_dir(), f"Missing subdir {sd}"

    def test_dataset_config(self, tmp_path, transformer_classifier_json):
        out = tmp_path / "cfg"
        json_path = FIXTURES_DIR / "transformer_classifier.json"
        build_hydra_configs(str(json_path), output_dir=str(out),
                            dataset="dataset.enron_spam.EnronSpamDataset")
        ds = OmegaConf.load(out / "dataset" / "dataset.yaml")
        assert ds._target_ == "dataset.enron_spam.EnronSpamDataset"
        assert ds.batch_size == 1024
