"""Create deterministic, self-contained pip wheels from training artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from model_package.adapters import adapter_spec_from_dataset_config


PACKAGE_NAME = re.compile(r"nnm_[A-Za-z][A-Za-z0-9_]*\Z")
VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)*(?:[A-Za-z0-9.+-]*)\Z")
RUNTIME_FILES = ("runtime.py", "adapters.py")


def build_model_wheel(
    artifact_dir: str | Path,
    *,
    package_name: str,
    version: str = "0.1.0",
) -> Path:
    """Build a pure-Python inference wheel from one successful job artifact."""

    if not PACKAGE_NAME.fullmatch(package_name):
        raise ValueError("package_name must match nnm_<name> using letters, digits, and underscores")
    if not VERSION.fullmatch(version):
        raise ValueError("version is not a valid package version")
    artifact_path = Path(artifact_dir).resolve()
    weights_path = artifact_path / "weights.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"model weights not found: {weights_path}")
    config = _load_resolved_config(artifact_path)
    module_name = package_name
    architecture = {
        "schema_version": 1,
        "net": _rewrite_targets(_mapping(config["net"]), module_name),
        "input_adapter": adapter_spec_from_dataset_config(_mapping(config.get("dataset", {}))),
    }
    dist_dir = artifact_path / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    wheel_name = f"{module_name}-{version}-py3-none-any.whl"
    wheel_path = dist_dir / wheel_name
    with tempfile.TemporaryDirectory(prefix="nnm-wheel-") as temporary:
        staging = Path(temporary)
        package_dir = staging / module_name
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text(
            "from .runtime import InferenceModel, load_model\n\n__all__ = ['InferenceModel', 'load_model']\n",
            encoding="utf-8",
        )
        (package_dir / "architecture.json").write_text(json.dumps(architecture, sort_keys=True), encoding="utf-8")
        shutil.copy2(weights_path, package_dir / "weights.safetensors")
        _copy_runtime(package_dir)
        dist_info = staging / f"{module_name}-{version}.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(_metadata(package_name, version), encoding="utf-8")
        (dist_info / "WHEEL").write_text(
            "Wheel-Version: 1.0\nGenerator: NNModelling\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            encoding="utf-8",
        )
        _write_wheel(staging, wheel_path)
    manifest = {
        "schema_version": 1,
        "package_name": package_name,
        "version": version,
        "wheel": str(Path("dist") / wheel_name),
        "sha256": _sha256(wheel_path),
        "input_adapter": architecture["input_adapter"],
    }
    (artifact_path / "model-package.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return wheel_path


def _load_resolved_config(artifact_path: Path) -> dict[str, Any]:
    json_path = artifact_path / "resolved_config.json"
    if json_path.is_file():
        value = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        yaml_path = artifact_path / "resolved_config.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError("resolved_config.yaml is required to export a model package")
        value = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    return _mapping(value)


def _copy_runtime(package_dir: Path) -> None:
    source_dir = Path(__file__).resolve().parent
    for filename in RUNTIME_FILES:
        shutil.copy2(source_dir / filename, package_dir / filename)
    source_ops = source_dir.parent / "ops"
    target_ops = package_dir / "_ops"
    target_ops.mkdir()
    for source in source_ops.glob("*.py"):
        content = source.read_text(encoding="utf-8").replace("from ops.", "from .")
        (target_ops / source.name).write_text(content, encoding="utf-8")


def _rewrite_targets(value: Any, module_name: str) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_targets(item, module_name) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_targets(item, module_name) for item in value]
    if isinstance(value, str) and value.startswith("ops."):
        return f"{module_name}._ops.{value.removeprefix('ops.')}"
    return value


def _metadata(package_name: str, version: str) -> str:
    return (
        "Metadata-Version: 2.1\n"
        f"Name: {package_name}\n"
        f"Version: {version}\n"
        "Summary: Exported NNModelling inference model\n"
        "Requires-Python: >=3.12\n"
        "Requires-Dist: torch\n"
        "Requires-Dist: hydra-core\n"
        "Requires-Dist: omegaconf\n"
        "Requires-Dist: safetensors\n"
        "Requires-Dist: torchvision\n"
        "Requires-Dist: Pillow\n"
    )


def _write_wheel(staging: Path, destination: Path) -> None:
    files = sorted(path for path in staging.rglob("*") if path.is_file())
    record_path = next(path for path in staging.rglob("*.dist-info") if path.is_dir()) / "RECORD"
    records = [f"{path.relative_to(staging).as_posix()},{_record_hash(path)},{path.stat().st_size}" for path in files]
    records.append(f"{record_path.relative_to(staging).as_posix()},,")
    record_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(path for path in staging.rglob("*") if path.is_file()):
            archive.write(path, path.relative_to(staging).as_posix())


def _record_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("resolved model configuration must be a mapping")
    return dict(value)
