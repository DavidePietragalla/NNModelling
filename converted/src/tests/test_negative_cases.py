"""Negative characterization: unsupported schema versions and artifact failures.

These tests document and lock the backend's rejection behavior for documents
and artifacts that must never be silently accepted: unsupported request
schema versions, missing model packages, and wheel exports blocked by missing
safe weights, missing or corrupt configuration, or exporter exceptions. A
packaging failure is a terminal job failure whose events and logs stay
visible. They use in-memory stores and executor doubles so they stay fast;
the *real* job failure path is covered by the E2E suite.
"""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path
from typing import Any

import httpx
import pytest

from backend.app import create_app
from backend.auth import AuthService, InMemoryAuthStore
from backend.manager import JobManager
from backend.models import JobSubmission, ResourceRequest
from backend.store import InMemoryJobStore
from tests.backend_helpers import mninst_nntree

OWNER = "negative-connection"


def _write_minimal_safetensors(path: Path) -> None:
    """Write a valid single-tensor safetensors container without torch."""
    header = json.dumps(
        {"tensor0": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + struct.pack("<f", 0.0))


def _payload(**changes: Any) -> dict[str, Any]:
    """Build a minimal valid job document payload with field overrides."""
    payload: dict[str, Any] = {
        "network": {"format": "nntree", "value": mninst_nntree()},
        "training": {
            "dataset": {"_target_": "dataset.mnist.MNISTDataset"},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.01},
            "trainer": {"max_epochs": 1, "accelerator": "cpu"},
            "wandb": {"project": "tests", "mode": "disabled"},
            "early_stopping": {"patience": 1, "min_delta": 0.0},
        },
        "resources": {"cpu": 1, "memory_gb": 1, "gpu": 0},
        "priority": 10,
    }
    payload.update(changes)
    return payload


class _SucceedingExecutor:
    """Executor double that finishes successfully without training."""

    name = "fake"
    kind = "local"

    def can_run(self, resources: dict[str, Any]) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {"id": self.name, "kind": self.kind, "capacity": {}, "enabled": True}

    def submit(self, job, artifact_dir, on_heartbeat, on_finished):
        root = Path(artifact_dir)
        (root / "stdout.log").write_text("ok\n", encoding="utf-8")
        (root / "stderr.log").write_text("", encoding="utf-8")
        on_finished(0, {"stdout": str(root / "stdout.log"), "stderr": str(root / "stderr.log")})
        return {"worker": "fake"}

    def cancel(self, job_id: str) -> bool:
        del job_id
        return True


def test_schema_version_one_is_accepted():
    """The current supported schema version is accepted explicitly."""
    submission = JobSubmission(**_payload(schema_version=1))
    assert submission.schema_version == 1


def test_unsupported_schema_version_is_rejected():
    """An unsupported request schema version fails loudly, not silently."""
    with pytest.raises(ValueError, match="schema_version"):
        JobSubmission(**_payload(schema_version=2))


def test_unsupported_schema_version_rejected_via_api(tmp_path):
    """The API surfaces the schema rejection as a 422."""
    manager = JobManager(InMemoryJobStore(), tmp_path, [_SucceedingExecutor()])
    auth = AuthService(InMemoryAuthStore())
    pairing = auth.create_pairing("Browser", client_host="127.0.0.1")
    auth.approve(pairing.request_id)
    app = create_app(manager, auth_service=auth, admin_token="admin-secret")

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                headers = {"authorization": f"Bearer {pairing.token}"}
                payload = _payload(schema_version=999)
                response = await client.post("/jobs", headers=headers, json=payload)
                assert response.status_code == 422
                assert "schema_version" in response.text


    asyncio.run(exercise_api())


def test_missing_model_package_has_no_download(tmp_path):
    """A job without an exported wheel has no downloadable package."""
    manager = JobManager(InMemoryJobStore(), tmp_path, [_SucceedingExecutor()])
    auth = AuthService(InMemoryAuthStore())
    pairing = auth.create_pairing("Browser", client_host="127.0.0.1")
    auth.approve(pairing.request_id)
    job = manager.submit(JobSubmission(**_payload()), owner_connection_id=pairing.connection_id)
    manager.run_once()
    app = create_app(manager, auth_service=auth, admin_token="admin-secret")

    # Without safe weights the job is a packaging failure, so a package can
    # never be downloaded from a job that did not export one.
    status = manager.status(job.id, owner_connection_id=pairing.connection_id)
    assert status.status == "failed"

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                headers = {"authorization": f"Bearer {pairing.token}"}
                response = await client.get(f"/jobs/{job.id}/package", headers=headers)
                assert response.status_code == 404


    asyncio.run(exercise_api())


def test_wheel_export_failure_records_package_error_and_fails_job(tmp_path):
    """A missing resolved config blocks export and fails the job coherently."""
    store = InMemoryJobStore()
    manager = JobManager(store, tmp_path, [_SucceedingExecutor()])
    queued = manager.submit(JobSubmission(**_payload()), owner_connection_id=OWNER)
    artifact = Path(queued.artifact_dir)
    # The fake executor never writes weights; simulate a job that trained.
    _write_minimal_safetensors(artifact / "weights.safetensors")
    (artifact / "resolved_config.yaml").unlink(missing_ok=True)

    manager.run_once()

    status = manager.status(queued.id, owner_connection_id=OWNER)
    assert status.status == "failed"
    assert status.finished_at is not None
    assert status.model_package is None
    assert status.package_error is not None
    assert "resolved_config" in (status.package_error or "").lower()
    assert "resolved_config" in (status.error or "").lower()
    events = manager.events(queued.id, owner_connection_id=OWNER)
    types = [event["type"] for event in events]
    assert any(event["type"] == "package_failed" for event in events)
    assert "failed" in types
    assert types.index("package_failed") < types.index("failed")
    # Training logs and the artifact root remain visible after the failure.
    assert manager.logs(queued.id, owner_connection_id=OWNER)["stdout"] == "ok\n"
    assert Path(status.artifact_dir).is_dir()


def test_corrupt_resolved_config_blocks_wheel_export_and_fails_job(tmp_path):
    """Invalid configuration YAML is a coherent export failure, not a crash."""
    store = InMemoryJobStore()
    manager = JobManager(store, tmp_path, [_SucceedingExecutor()])
    queued = manager.submit(JobSubmission(**_payload()), owner_connection_id=OWNER)
    artifact = Path(queued.artifact_dir)
    _write_minimal_safetensors(artifact / "weights.safetensors")
    (artifact / "resolved_config.yaml").write_text("net: [unclosed", encoding="utf-8")

    manager.run_once()

    status = manager.status(queued.id, owner_connection_id=OWNER)
    assert status.status == "failed"
    assert status.finished_at is not None
    assert status.package_error is not None
    assert status.model_package is None
    events = manager.events(queued.id, owner_connection_id=OWNER)
    types = [event["type"] for event in events]
    assert any(event["type"] == "package_failed" for event in events)
    assert "failed" in types
    assert types.index("package_failed") < types.index("failed")
    assert manager.logs(queued.id, owner_connection_id=OWNER)["stdout"] == "ok\n"


def test_corrupt_safe_weights_fail_job_with_package_error(tmp_path):
    """Garbage in weights.safetensors is a packaging failure, not a success."""
    store = InMemoryJobStore()
    manager = JobManager(store, tmp_path, [_SucceedingExecutor()])
    queued = manager.submit(JobSubmission(**_payload()), owner_connection_id=OWNER)
    (Path(queued.artifact_dir) / "weights.safetensors").write_bytes(b"not-a-safetensors-file")

    manager.run_once()

    status = manager.status(queued.id, owner_connection_id=OWNER)
    assert status.status == "failed"
    assert status.finished_at is not None
    assert status.model_package is None
    assert status.package_error is not None
    events = manager.events(queued.id, owner_connection_id=OWNER)
    types = [event["type"] for event in events]
    assert "failed" in types
    assert any(event["type"] == "package_failed" for event in events)
    assert types.index("package_failed") < types.index("failed")
    assert manager.logs(queued.id, owner_connection_id=OWNER)["stdout"] == "ok\n"


def test_missing_safe_weights_fails_job_with_package_error(tmp_path):
    """A job without safe weights is a packaging failure, not a silent success."""
    store = InMemoryJobStore()
    manager = JobManager(store, tmp_path, [_SucceedingExecutor()])
    queued = manager.submit(JobSubmission(**_payload()), owner_connection_id=OWNER)

    manager.run_once()

    status = manager.status(queued.id, owner_connection_id=OWNER)
    assert status.status == "failed"
    assert status.finished_at is not None
    assert status.model_package is None
    assert "weights.safetensors" in (status.package_error or "")
    events = manager.events(queued.id, owner_connection_id=OWNER)
    types = [event["type"] for event in events]
    assert "failed" in types
    assert any(event["type"] == "package_failed" for event in events)
    assert types.index("package_failed") < types.index("failed")
    assert manager.logs(queued.id, owner_connection_id=OWNER)["stdout"] == "ok\n"


def test_exporter_exception_fails_job_and_preserves_logs(tmp_path, monkeypatch):
    """An exporter exception is a terminal packaging failure with logs intact."""
    def exploding_export(artifact_dir, *, package_name, version):
        del artifact_dir, package_name, version
        raise ValueError("unsupported input adapter kind: 'weird'")

    monkeypatch.setattr("backend.manager.build_model_wheel", exploding_export)
    store = InMemoryJobStore()
    manager = JobManager(store, tmp_path, [_SucceedingExecutor()])
    queued = manager.submit(JobSubmission(**_payload()), owner_connection_id=OWNER)
    # A current successful job has safe weights; the exporter still fails.
    _write_minimal_safetensors(Path(queued.artifact_dir) / "weights.safetensors")

    manager.run_once()

    status = manager.status(queued.id, owner_connection_id=OWNER)
    assert status.status == "failed"
    assert status.finished_at is not None
    assert status.model_package is None
    assert "unsupported input adapter" in (status.package_error or "")
    assert "unsupported input adapter" in (status.error or "")
    events = manager.events(queued.id, owner_connection_id=OWNER)
    types = [event["type"] for event in events]
    assert any(event["type"] == "package_failed" for event in events)
    assert "failed" in types
    assert types.index("package_failed") < types.index("failed")
    assert manager.logs(queued.id, owner_connection_id=OWNER)["stdout"] == "ok\n"


def test_api_surfaces_package_failure_as_failed_job_with_events_and_logs(tmp_path):
    """The API exposes failed status, package_error, events, and preserved logs."""
    manager = JobManager(InMemoryJobStore(), tmp_path, [_SucceedingExecutor()])
    auth = AuthService(InMemoryAuthStore())
    pairing = auth.create_pairing("Browser", client_host="127.0.0.1")
    auth.approve(pairing.request_id)
    job = manager.submit(JobSubmission(**_payload()), owner_connection_id=pairing.connection_id)
    manager.run_once()
    app = create_app(manager, auth_service=auth, admin_token="admin-secret")

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                headers = {"authorization": f"Bearer {pairing.token}"}
                status_response = await client.get(f"/jobs/{job.id}", headers=headers)
                assert status_response.status_code == 200
                body = status_response.json()
                assert body["status"] == "failed"
                assert body["model_package"] is None
                assert body["package_error"] is not None
                assert "weights.safetensors" in (body["package_error"] or "")
                assert (await client.get(f"/jobs/{job.id}/package", headers=headers)).status_code == 404

                events_response = await client.get(f"/jobs/{job.id}/events", headers=headers)
                assert events_response.status_code == 200
                types = [
                    json.loads(line.removeprefix("data: "))["type"]
                    for line in events_response.text.splitlines()
                    if line.startswith("data: ")
                ]
                assert "failed" in types
                assert types.index("package_failed") < types.index("failed")

                logs = await client.get(f"/jobs/{job.id}/logs", headers=headers)
                assert logs.status_code == 200
                assert logs.json()["stdout"] == "ok\n"

    asyncio.run(exercise_api())


def test_manifest_digest_verifies_wheel_bytes_when_exported(tmp_path):
    """The exporter's manifest digest matches the wheel it actually built.

    This is a positive control for the E2E digest check: the exporter must
    never claim a digest for different bytes.
    """
    from model_package.exporter import build_model_wheel

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    net_config = {
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
    (artifact / "resolved_config.json").write_text(json.dumps(net_config), encoding="utf-8")
    from model_package.runtime import GraphNet
    from safetensors.torch import save_file

    import torch

    network = GraphNet(net_config["net"])
    save_file(network.state_dict(), artifact / "weights.safetensors")

    wheel = build_model_wheel(artifact, package_name="nnm_digest_check")
    manifest = json.loads((artifact / "model-package.json").read_text(encoding="utf-8"))

    import hashlib

    assert manifest["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert manifest["wheel"] == str(Path("dist") / wheel.name)
