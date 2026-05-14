from typing import Any
import json
import os
from omegaconf import OmegaConf

import ast


# 1. Definiamo un dizionario che mappa il nome del tipo alla funzione di conversione
TYPE_MAP = {
    "int": int,
    "integer": int,
    "float": float,
    "str": str,
    "string": str,
    # Funzione lambda (anonima) per gestire correttamente i booleani
    "bool": lambda x: str(x).lower() in ["true", "1", "t", "y", "yes"],
    "boolean": lambda x: str(x).lower() in ["true", "1", "t", "y", "yes"],
    # ast.literal_eval per valutare stringhe come "[1, 2]" in modo sicuro
    "list": ast.literal_eval,
    "tuple": ast.literal_eval,
    "dict": ast.literal_eval,
}


def parse_value(val_str: str, type_name: str):
    # Gestione dei nulli
    if str(val_str).lower() in ["none", "null", "undefined", ""]:
        return None

    # 2. Recuperiamo la funzione corretta dal dizionario.
    # Se il tipo non esiste, usiamo 'str' come fallback.
    converter_function = TYPE_MAP.get(type_name.lower(), str)

    try:
        # 3. Eseguiamo la funzione passandogli la stringa
        return converter_function(val_str)
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Impossibile convertire '{val_str}' in '{type_name}'. Errore: {e}")
        return val_str


def process_params(params: dict[str, Any]) -> dict[str, Any]:
    """Process params from the JSON format to raw values."""
    result: dict[str, Any] = {}
    for param_name, param_data in params.items():
        if isinstance(param_data, dict):
            result[param_name] = parse_value(param_data.get("value", ""), param_data.get("type", "str"))
        else:
            # Already a raw value
            result[param_name] = param_data
    return result


def build_hydra_configs(json_path: str, output_dir: str = "cfg"):
    # Carica il JSON generato da Svelte
    with open(json_path, "r") as f:
        diagram: dict[str, Any] = json.load(f)

    network: list[Any] = diagram.get("network", [])

    # Create directory structure
    layer_dir = os.path.join(output_dir, "layer")
    net_dir = os.path.join(output_dir, "net")
    optimizer_dir = os.path.join(output_dir, "optimizer")
    trainer_dir = os.path.join(output_dir, "trainer")
    wandb_dir = os.path.join(output_dir, "wandb")
    dataset_dir = os.path.join(output_dir, "dataset")
    os.makedirs(layer_dir, exist_ok=True)
    os.makedirs(net_dir, exist_ok=True)
    os.makedirs(optimizer_dir, exist_ok=True)
    os.makedirs(trainer_dir, exist_ok=True)
    os.makedirs(wandb_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    net_defaults: list[dict[str, str]] = []
    sequence_list: list[str] = []

    # Process each layer in the network (flat structure)
    for layer in network:
        layer_id: str = layer["id"]
        target = layer.get("target", "")
        
        # Skip input marker layers (target=None or "None")
        if target is None or target == "None":
            continue
            
        sequence_list.append(layer_id)

        # Build the layer configuration
        layer_conf_dict = {"_target_": target}
        
        # Fix nn. prefix to torch.nn.
        if layer_conf_dict.get("_target_", "").startswith("nn."):
            layer_conf_dict["_target_"] = "torch." + layer_conf_dict["_target_"]

        params = layer.get("params", {})
        if isinstance(params, dict):
            for param_name, param in params.items():
                layer_conf_dict[param_name] = parse_value(param.get("value"), param.get("type"))

        # Remove None values
        layer_conf_dict = {k: v for k, v in layer_conf_dict.items() if v is not None}

        # Save the layer YAML file
        layer_yaml_path = os.path.join(layer_dir, f"{layer_id}.yaml")
        OmegaConf.save(config=layer_conf_dict, f=layer_yaml_path)

        # Add to defaults
        net_defaults.append({f"layer/{layer_id}": layer_id})

    # Handle loss function if present
    if "loss" in diagram:
        loss_data = diagram["loss"]
        loss_layer_id = loss_data.get("id", "loss_final")

        # Build loss configuration
        loss_conf_dict = {"_target_": loss_data["target"]}

        params = loss_data.get("params", {})
        if isinstance(params, dict):
            for param_name, param in params.items():
                loss_conf_dict[param_name] = parse_value(param.get("value"), param.get("type"))

        # Fix nn. prefix to torch.nn.
        if loss_conf_dict.get("_target_", "").startswith("nn."):
            loss_conf_dict["_target_"] = "torch." + loss_conf_dict["_target_"]

        # Remove None values
        loss_conf_dict = {k: v for k, v in loss_conf_dict.items() if v is not None}

        # Save the loss YAML file
        loss_yaml_path = os.path.join(layer_dir, f"{loss_layer_id}.yaml")
        OmegaConf.save(config=loss_conf_dict, f=loss_yaml_path)

        # Add to defaults only (loss is NOT in sequence - it's used separately)
        net_defaults.append({f"layer/{loss_layer_id}": loss_layer_id})

    # Build the full net config with all layer configs
    net_config_dict = {
        "sequence": sequence_list,
        "name": "SequentialModel_from_UI",
    }
    
    # Add layer configs
    for layer_id in sequence_list:
        layer_config = OmegaConf.load(os.path.join(layer_dir, f"{layer_id}.yaml"))
        net_config_dict[layer_id] = layer_config
    
    # Add loss layer if present
    if "loss" in diagram:
        loss_layer_id = diagram["loss"].get("id", "loss_final")
        loss_config = OmegaConf.load(os.path.join(layer_dir, f"{loss_layer_id}.yaml"))
        net_config_dict[loss_layer_id] = loss_config
        net_config_dict["loss"] = loss_config  # For compatibility with SequentialNet
    
    net_config = OmegaConf.create(net_config_dict)
    # Disable struct mode to allow dynamic keys
    OmegaConf.set_struct(net_config, False)
    
    net_yaml_path = os.path.join(net_dir, "custom_sequence.yaml")
    OmegaConf.save(config=net_config, f=net_yaml_path)

    # Generate a default optimizer configuration
    optimizer_config = OmegaConf.create({
        "_target_": "torch.optim.Adam",
        "lr": 0.001,
        "weight_decay": 0.0005,
    })
    optimizer_yaml_path = os.path.join(optimizer_dir, "adam.yaml")
    OmegaConf.save(config=optimizer_config, f=optimizer_yaml_path)

    print(f"Configurazione salvata con successo in '{output_dir}/'!")

    # Generate trainer configuration with defaults
    trainer_config = OmegaConf.create({
        "max_epochs": 20,
        "log_every_n_steps": 10,
        "accelerator": "auto",
    })
    trainer_yaml_path = os.path.join(trainer_dir, "default.yaml")
    OmegaConf.save(config=trainer_config, f=trainer_yaml_path)

    # Generate wandb configuration
    wandb_config = OmegaConf.create({
        "project": "NeuralNetwors",
        "name": "${net.name}",
    })
    # wandb.yaml is saved in wandb/ directory
    wandb_yaml_path = os.path.join(wandb_dir, "wandb.yaml")
    OmegaConf.save(config=wandb_config, f=wandb_yaml_path)

    # dataset.yaml is saved in dataset/ directory
    dataset_yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    dataset_config = OmegaConf.create({
        "import": "dataset.mnist",
        "name": "MNISTDataset",
        "params": {
            "batch_size": 1024,
            "train_size": 0.8,
        }
    })
    OmegaConf.save(config=dataset_config, f=dataset_yaml_path)

    # Generate seed config
    seed = 42
    
    # Generate the main base.yaml that combines all defaults
    base_defaults = [
        {"net": "custom_sequence"},
        {"optimizer": "adam"},
        {"trainer": "default"},
        {"wandb": "wandb"},
        {"dataset": "dataset"},
    ]
    
    # Load the custom_sequence to get sequence and name
    custom_seq_config = OmegaConf.load(os.path.join(net_dir, "custom_sequence.yaml"))
    
    # Build layer defaults list
    layer_defaults = []
    for layer_id in custom_seq_config.sequence:
        # Hydra looks for layer/{layer_id}.yaml when we specify just {layer_id}
        layer_defaults.append({layer_id: layer_id})
    
    # Add loss layer if present
    if "loss" in diagram:
        loss_layer_id = diagram["loss"].get("id", "loss_final")
        layer_defaults.append({loss_layer_id: loss_layer_id})
    
    # Generate the main base.yaml that combines all defaults
    # Note: dataset, wandb, trainer are loaded from root level files
    # Note: defaults must be a list of dicts
    # Added _self_ to ensure proper composition order
    base_defaults = [
        {"net": "custom_sequence"},
        {"optimizer": "adam"},
        {"trainer": "default"},
        {"wandb": "wandb"},
        {"dataset": "dataset"},
        "_self_",
    ]
    
    base_config = OmegaConf.create({
        "defaults": base_defaults,
        "seed": seed,
        # trainer, wandb, dataset are loaded from defaults
    })
    
    base_yaml_path = os.path.join(output_dir, "base.yaml")
    OmegaConf.save(config=base_config, f=base_yaml_path)

    # Print summary
    print(f"\nConfigurazione salvata con successo in '{output_dir}/'!")
    print(f"   - Layer totali: {len(network)}")
    if "loss" in diagram:
        print(f"   - Loss: {diagram['loss'].get('name', 'Unknown')}")


if __name__ == "__main__":
    import sys
    # Accept JSON path and output directory as command line arguments
    json_path = "../converted_minst.json"
    output_dir = "cfg"
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    build_hydra_configs(json_path, output_dir=output_dir)
