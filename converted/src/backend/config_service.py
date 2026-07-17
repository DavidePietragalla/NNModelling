"""Normalize job training data and generate Hydra configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from convert import build_hydra_configs


def normalize_training_config(training: dict[str, Any]) -> dict[str, Any]:
    """Normalize the JSON training document to Hydra-compatible mappings.

    The dataset may be supplied as a shorthand import path or as a complete
    Hydra mapping. All other sections are copied without dropping unknown
    Hydra fields.
    """

    normalized = json.loads(json.dumps(training))
    dataset = normalized.get("dataset")
    if isinstance(dataset, str):
        normalized["dataset"] = {"_target_": dataset}
    elif not isinstance(dataset, dict) or not dataset.get("_target_"):
        raise ValueError("training.dataset must be a Python target string or Hydra mapping")
    return normalized


def _section(training: dict[str, Any], name: str, default: dict[str, Any]) -> dict[str, Any]:
    """Return a copied Hydra section, preserving arbitrary fields."""

    value = training.get(name, default)
    if not isinstance(value, dict):
        raise ValueError(f"training.{name} must be a JSON object")
    return json.loads(json.dumps(value))


def _resolve_config(config_dir: Path, overrides: list[str]) -> DictConfig:
    """Compose a generated Hydra config with user-provided overrides."""

    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        return compose(config_name="base", overrides=overrides)


def build_job_hydra_configs(job: dict[str, Any], output_dir: str | Path) -> Path:
    """Generate Hydra files for a complete remote-training job.

    The generated directory is returned. The original job is written by the
    job manager so that conversion remains a pure filesystem operation.
    """

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    network = job.get("network", {})
    if network.get("format") != "nntree":
        raise ValueError("network.format must be 'nntree'")
    nntree = network.get("value")
    if not isinstance(nntree, dict):
        raise ValueError("network.value must be an NNTree object")

    training = normalize_training_config(job.get("training", {}))
    num_classes = training.get("num_classes")
    if num_classes is not None and not isinstance(num_classes, int):
        raise ValueError("training.num_classes must be an integer")

    # The existing converter owns the network-specific transformation. The
    # optional training_config argument lets it write all Hydra groups from
    # this job rather than from narrow CLI flags.
    build_hydra_configs(
        nntree,
        output_dir=str(output_path / "cfg"),
        num_classes=num_classes,
        training_config={
            "dataset": _section(training, "dataset", {}),
            "optimizer": _section(
                training,
                "optimizer",
                {"_target_": "torch.optim.Adam", "lr": 0.001},
            ),
            "trainer": _section(
                training,
                "trainer",
                {"max_epochs": 20, "accelerator": "auto"},
            ),
            "wandb": _section(
                training,
                "wandb",
                {"project": "NeuralNetworks", "name": "Dynamic_Model"},
            ),
            "early_stopping": _section(
                training,
                "early_stopping",
                {"patience": 3, "min_delta": 0.0},
            ),
        },
    )

    overrides = training.get("overrides", [])
    if overrides is None:
        overrides = []
    if not isinstance(overrides, list) or not all(isinstance(item, str) for item in overrides):
        raise ValueError("training.overrides must be a list of Hydra override strings")

    resolved = _resolve_config(output_path / "cfg", overrides)
    resolved_path = output_path / "resolved_config.yaml"
    OmegaConf.save(config=resolved, f=str(resolved_path))
    return output_path / "cfg"

