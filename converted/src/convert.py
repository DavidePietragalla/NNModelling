from typing import Any
import json
import os
from omegaconf import OmegaConf

import ast


def parse_params(params_dict: dict[str, Any]) -> dict[str, Any]:
    """Converte i parametri dal JSON in valori Python."""
    result: dict[str, Any] = {}
    for param_name, param_data in params_dict.items():
        val_str = param_data.get("value", "")

        if str(val_str).lower() in ["null", "undefined"]:
            continue

        if str(val_str).lower() in ["none", ""]:
            result[param_name] = None
            continue

        if str(val_str).lower() == "true":
            result[param_name] = True
            continue
        if str(val_str).lower() == "false":
            result[param_name] = False
            continue

        try:
            result[param_name] = ast.literal_eval(val_str)
        except (ValueError, SyntaxError):
            result[param_name] = val_str
    return result


def build_layer_config(layer_data: dict[str, Any]) -> dict[str, Any]:
    """Costruisce il dizionario compatibile con Hydra per l'instanziazione."""
    stereotype = layer_data.get("stereotype", "")
    config = {"stereotype": stereotype}

    if stereotype not in ["Input", "Addition", "Einsum", ""]:
        python_class_name = layer_data.get("pythonClassName", stereotype)
        if python_class_name.startswith("nn."):
            config["_target_"] = "torch." + python_class_name
        elif python_class_name.startswith("torch."):
            config["_target_"] = python_class_name
        else:
            config["_target_"] = f"torch.nn.{python_class_name}"

    config.update(parse_params(layer_data.get("params", {})))
    return config


def build_hydra_configs(json_path: str, output_dir: str = "cfg"):
    with open(json_path, "r") as f:
        diagram: dict[str, Any] = json.load(f)

    for d in ["net", "optimizer", "trainer", "wandb", "dataset"]:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    nntree = diagram.get("NNTree", diagram)

    net_config_dict = {
        "root": nntree.get("root", ""),
        "nodes": {},
        "lossNode": build_layer_config(nntree.get("lossNode", {}))
        if "lossNode" in nntree
        else None,
    }

    for node_id, node_info in nntree.get("nodes", {}).items():
        node_type = node_info["data"].get("type", "")
        node_config = {
            "children": node_info.get("children", []),
            "type": node_type,
            "stereotype": node_info["data"].get("stereotype", ""),
        }

        if node_type == "sequential":
            node_config["layers"] = [
                build_layer_config(l) for l in node_info["data"].get("layers", [])
            ]
        elif node_type == "module":
            node_config["layer"] = build_layer_config(node_info["data"])

        net_config_dict["nodes"][node_id] = node_config

    net_config = OmegaConf.create(net_config_dict)
    OmegaConf.save(
        config=net_config, f=os.path.join(output_dir, "net", "custom_sequence.yaml")
    )

    OmegaConf.save(
        config=OmegaConf.create({"_target_": "torch.optim.Adam", "lr": 0.001}),
        f=os.path.join(output_dir, "optimizer", "adam.yaml"),
    )
    OmegaConf.save(
        config=OmegaConf.create({"max_epochs": 20, "accelerator": "auto"}),
        f=os.path.join(output_dir, "trainer", "default.yaml"),
    )
    OmegaConf.save(
        config=OmegaConf.create({"project": "NeuralNetworks", "name": "Dynamic_Model"}),
        f=os.path.join(output_dir, "wandb", "wandb.yaml"),
    )
    OmegaConf.save(
        config=OmegaConf.create(
            {
                "import": "dataset.mnist",
                "name": "MNISTDataset",
                "params": {"batch_size": 1024, "train_size": 0.8},
            }
        ),
        f=os.path.join(output_dir, "dataset", "dataset.yaml"),
    )

    base_config = OmegaConf.create(
        {
            "defaults": [
                {"net": "custom_sequence"},
                {"optimizer": "adam"},
                {"trainer": "default"},
                {"wandb": "wandb"},
                {"dataset": "dataset"},
                "_self_",
            ],
            "seed": 42,
        }
    )
    OmegaConf.save(config=base_config, f=os.path.join(output_dir, "base.yaml"))
    print(f"\nConfigurazione salvata con successo in '{output_dir}/'!")


if __name__ == "__main__":
    import sys

    json_path = sys.argv[1] if len(sys.argv) > 1 else "../converted_minst.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "cfg"
    build_hydra_configs(json_path, output_dir=output_dir)
