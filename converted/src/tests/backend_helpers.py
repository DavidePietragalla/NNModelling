"""Shared helpers for the backend service and E2E test suites.

These helpers are test infrastructure only. They build tiny deterministic
``JobSubmission`` documents from the repository's ``mninst`` (MNIST
classifier) and ``autoencoder_mnist`` NNTree fixtures, wait for job
transitions through a real :class:`JobManager`, and run the exported wheel in
an isolated subprocess for ``load_model().predict()`` validation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from backend.manager import JobManager
from backend.models import JobStatus, JobSubmission, ResourceRequest

# converted/src/tests/test_backend_e2e.py -> parents[3] = repo root
EXAMPLES_NNTREES = Path(__file__).resolve().parents[3] / "examples" / "nntrees"
CONVERTED_DIR = Path(__file__).resolve().parents[2]

# Dedicated Valkey logical database for tests so service/E2E runs never touch
# the development database 0. Override with NNM_VALKEY_URL in CI.
DEFAULT_TEST_VALKEY_URL = "valkey://127.0.0.1:6379/15"

TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


def valkey_required() -> bool:
    """Return whether the current run demands a real Valkey instance."""
    return os.getenv("NNM_REQUIRE_VALKEY", "0").lower() in {"1", "true", "yes"}


def get_test_valkey_url() -> str:
    """Return the Valkey URL used by service/E2E tests."""
    return os.getenv("NNM_VALKEY_URL", DEFAULT_TEST_VALKEY_URL)


def mninst_nntree() -> dict[str, Any]:
    """Load the repository's verified MNIST classifier NNTree fixture."""
    return json.loads((EXAMPLES_NNTREES / "mninst_skip.json").read_text(encoding="utf-8"))


def autoencoder_nntree() -> dict[str, Any]:
    """Load the repository's verified autoencoder NNTree fixture."""
    return json.loads((EXAMPLES_NNTREES / "auto_encoder.json").read_text(encoding="utf-8"))


def tiny_training(dataset_target: str, *, max_epochs: int = 1) -> dict[str, Any]:
    """Build the training document for a tiny deterministic job."""
    return {
        "dataset": {
            "_target_": dataset_target,
            "batch_size": 8,
            "num_workers": 0,
            "train_size": 0.6,
            "total_samples": 100,
            "seed": 1234,
        },
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
        "trainer": {"max_epochs": max_epochs, "accelerator": "cpu"},
        "wandb": {"project": "nnm-e2e", "mode": "disabled"},
        "early_stopping": {"patience": 1, "min_delta": 0.0},
    }


def classification_submission(*, package_name: str = "nnm_mnist_classifier") -> JobSubmission:
    """Build a valid tiny MNIST-classifier job document."""
    return JobSubmission(
        network={"format": "nntree", "value": mninst_nntree()},
        training=tiny_training("tests.tiny_mnist_dataset.TinyMNISTDataset"),
        resources=ResourceRequest(cpu=1, memory_gb=1, gpu=0),
        priority=10,
        package_name=package_name,
    )


def autoencoder_submission(*, package_name: str = "nnm_autoencoder_tiny") -> JobSubmission:
    """Build a valid tiny image-reconstruction job document."""
    return JobSubmission(
        network={"format": "nntree", "value": autoencoder_nntree()},
        training=tiny_training("tests.tiny_mnist_dataset.TinyAutoencoderDataset"),
        resources=ResourceRequest(cpu=1, memory_gb=1, gpu=0),
        priority=10,
        package_name=package_name,
    )


def broken_submission(*, package_name: str = "nnm_broken_job") -> JobSubmission:
    """Build a job whose dataset fails at instantiation inside main.py."""
    return JobSubmission(
        network={"format": "nntree", "value": mninst_nntree()},
        training=tiny_training("tests.tiny_mnist_dataset.BrokenDataset"),
        resources=ResourceRequest(cpu=1, memory_gb=1, gpu=0),
        priority=10,
        package_name=package_name,
    )


def wait_for_terminal(
    manager: JobManager,
    job_id: str,
    owner_connection_id: str,
    *,
    timeout: float = 600.0,
    interval: float = 0.5,
) -> JobStatus:
    """Poll a real manager until a job reaches a terminal state."""
    deadline = time.monotonic() + timeout
    last: JobStatus | None = None
    while time.monotonic() < deadline:
        last = manager.status(job_id, owner_connection_id=owner_connection_id)
        if last.status in TERMINAL_STATES:
            return last
        time.sleep(interval)
    assert last is not None
    raise TimeoutError(
        f"job {job_id} did not reach a terminal state within {timeout}s "
        f"(last status: {last.status})"
    )


def wait_for_package(
    manager: JobManager,
    job_id: str,
    owner_connection_id: str,
    *,
    timeout: float = 120.0,
    interval: float = 0.5,
) -> JobStatus:
    """Wait until a succeeded job exported its wheel and emitted its terminal event.

    The manager marks the job ``succeeded`` before exporting the wheel and
    appends the ``succeeded`` stream event only after the export, so waiting
    for both guarantees the model package is fully materialized.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.status(job_id, owner_connection_id=owner_connection_id)
        events = manager.events(job_id, owner_connection_id=owner_connection_id)
        if status.model_package is not None and events and events[-1]["type"] == "succeeded":
            return status
        time.sleep(interval)
    raise TimeoutError(f"job {job_id} did not export a model package within {timeout}s")


def deterministic_input(batch: int = 2, seed: int = 7) -> torch.Tensor:
    """Return a reproducible MNIST-shaped input batch."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(batch, 1, 28, 28, generator=generator, dtype=torch.float32)


# Script executed by the isolated venv interpreter with the package installed
# from the *downloaded* wheel bytes. The prediction is cross-checked against
# an independent safetensors + architecture reload from the installed package.
_INSTALLED_PREDICT_SCRIPT = r"""
import json
import sys
from importlib.resources import as_file, files

import torch
from safetensors.torch import load_file

package_name, input_path, output_path = sys.argv[1:4]
module = __import__(package_name)

data = torch.load(input_path, weights_only=True)
with torch.inference_mode():
    wheel_output = module.load_model(device="cpu").predict_tensor(data)

architecture = json.loads(
    files(package_name).joinpath("architecture.json").read_text(encoding="utf-8")
)
network = module.runtime.GraphNet(architecture["net"])
with as_file(files(package_name).joinpath("weights.safetensors")) as weights_path:
    state = load_file(str(weights_path), device="cpu")
network.load_state_dict(state, strict=True)
with torch.inference_mode():
    direct_output = network(data)

json.dump(
    {
        "shape": list(wheel_output.shape),
        "dtype": str(wheel_output.dtype),
        "finite": bool(torch.isfinite(wheel_output).all()),
        "reload_equivalent": bool(torch.equal(wheel_output, direct_output)),
        "sample": wheel_output.flatten()[:5].tolist(),
    },
    open(output_path, "w"),
)
"""


def install_and_predict(
    downloaded_wheel: Path,
    package_name: str,
    inputs: torch.Tensor,
    work_dir: Path,
    *,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Install a *downloaded* wheel into an isolated venv and predict with it.

    The wheel passed here is the client-owned file written from the bytes
    served by ``GET /jobs/{id}/package`` — never the server artifact path.
    It is installed with pip into a fresh isolated virtual environment that
    inherits the already-installed heavy dependencies (torch, hydra,
    omegaconf, safetensors, ...) from the parent environment via a ``.pth``
    file, so no network fetch is needed. The NNModelling package itself is
    installed from the downloaded bytes and is never added directly to
    ``sys.path``. Prediction runs with the venv interpreter, so imports
    resolve exclusively to the installed wheel, and the result is
    cross-checked against an independent safetensors + architecture reload
    from the installed package.
    """
    venv_dir = work_dir / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )
    venv_python = venv_dir / "bin" / "python"
    # The parent interpreter is itself a virtual environment (uv), and a
    # nested venv's --system-site-packages would only expose the *base*
    # Python's packages, not the parent venv's (torch, hydra, ...). Inherit
    # the parent venv's site-packages explicitly with a .pth file so heavy
    # dependencies resolve without a network fetch while the wheel itself is
    # installed by pip into this isolated venv.
    parent_site = subprocess.check_output(
        [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"],
        cwd=work_dir,
        text=True,
        timeout=timeout,
    ).strip()
    venv_site = subprocess.check_output(
        [str(venv_python), "-c", "import site; print(site.getsitepackages()[0])"],
        cwd=work_dir,
        text=True,
        timeout=timeout,
    ).strip()
    Path(venv_site, "nnm-parent-site.pth").write_text(f"{parent_site}\n", encoding="utf-8")
    install = subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--no-deps", "--quiet", str(downloaded_wheel)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert install.returncode == 0, (
        f"pip install of the downloaded wheel failed:\n"
        f"STDOUT:\n{install.stdout}\nSTDERR:\n{install.stderr}"
    )
    input_path = work_dir / "input.pt"
    output_path = work_dir / "predict.json"
    torch.save(inputs, input_path)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [
            str(venv_python),
            "-c",
            _INSTALLED_PREDICT_SCRIPT,
            package_name,
            str(input_path),
            str(output_path),
        ],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"installed-wheel predict failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return json.loads(output_path.read_text(encoding="utf-8"))
