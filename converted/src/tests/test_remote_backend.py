"""Tests for remote job configuration, storage, scheduling and API wiring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from backend.app import create_app
from backend.config_service import normalize_training_config
from backend.dataset_registry import discover_datasets
from backend.executors import SlurmExecutor
from backend.manager import JobManager
from backend.models import JobSubmission, ResourceRequest
from backend.store import InMemoryJobStore, ValkeyJobStore


TRANSFORMER_NNTREE_PATH = Path(__file__).resolve().parents[3] / "examples" / "nntrees" / "transformer_classifier.json"


def transformer_nntree() -> dict[str, Any]:
    """Load the repository's verified transformer NNTree fixture."""

    return json.loads(TRANSFORMER_NNTREE_PATH.read_text(encoding="utf-8"))


def submission() -> JobSubmission:
    """Build a minimal valid job request."""

    return JobSubmission(
        network={"format": "nntree", "value": transformer_nntree()},
        training={
            "dataset": "dataset.enron_spam.EnronSpamDataset",
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.01},
            "trainer": {"max_epochs": 1, "accelerator": "cpu"},
            "wandb": {"project": "tests", "mode": "disabled"},
            "early_stopping": {"patience": 1, "min_delta": 0.0},
            "overrides": ["trainer.max_epochs=2"],
        },
        resources=ResourceRequest(cpu=1, memory_gb=1, gpu=0),
        priority=10,
    )


class ImmediateExecutor:
    """Executor double that finishes successfully in the callback."""

    name = "fake"
    kind = "local"

    def can_run(self, resources: dict[str, Any]) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.name,
            "kind": self.kind,
            "capacity": ResourceRequest(cpu=8, memory_gb=16).model_dump(mode="json"),
            "enabled": True,
        }

    def submit(self, job, artifact_dir, on_heartbeat, on_finished):
        Path(artifact_dir, "stdout.log").write_text("ok\n", encoding="utf-8")
        Path(artifact_dir, "stderr.log").write_text("", encoding="utf-8")
        on_heartbeat({"worker": "test"})
        on_finished(0, {"stdout": str(Path(artifact_dir, "stdout.log"))})
        return {"worker": "test"}

    def cancel(self, job_id: str) -> bool:
        return True


def test_normalize_training_config_accepts_dataset_target_string():
    normalized = normalize_training_config({"dataset": "dataset.mnist.MNISTDataset"})
    assert normalized["dataset"] == {"_target_": "dataset.mnist.MNISTDataset"}


def test_dataset_registry_discovers_installed_classes():
    targets = {item.target for item in discover_datasets()}
    assert "dataset.mnist.MNISTDataset" in targets
    assert "dataset.autoencoder_mnist.AutoencoderMNIST" in targets


def test_in_memory_store_orders_priority_then_fifo():
    store = InMemoryJobStore()
    store.enqueue("low", priority=1, created_at="2026-01-01T00:00:00+00:00")
    store.enqueue("high-late", priority=10, created_at="2026-01-01T00:02:00+00:00")
    store.enqueue("high-early", priority=10, created_at="2026-01-01T00:01:00+00:00")
    assert store.claim_next() == "high-early"
    assert store.claim_next() == "high-late"
    assert store.claim_next() == "low"


def test_valkey_event_cursor_continues_after_retention_limit():
    """A stream cursor must keep its native identity after 1,000 events."""

    class FakeStreamClient:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, str]]] = []

        def xadd(self, _key, fields, maxlen):
            event_id = f"{len(self.events) + 1}-0"
            self.events.append((event_id, fields))
            self.events = self.events[-maxlen:]
            return event_id

        def xrange(self, _key, min="-", max="+", count=None):
            del max
            events = self.events
            if min.startswith("("):
                after = tuple(int(part) for part in min[1:].split("-"))
                events = [
                    event
                    for event in events
                    if tuple(int(part) for part in event[0].split("-")) > after
                ]
            return events[:count]

    store = object.__new__(ValkeyJobStore)
    store.client = FakeStreamClient()
    for sequence in range(1_000):
        store.append_event("job-1", {"sequence": sequence})

    first_batch = store.get_events("job-1")
    assert first_batch[-1]["id"] == "1000-0"

    store.append_event("job-1", {"sequence": 1_000})
    following_batch = store.get_events("job-1", after=first_batch[-1]["id"])
    assert [event["sequence"] for event in following_batch] == [1_000]


def test_manager_builds_job_artifacts_and_finishes(tmp_path):
    manager = JobManager(InMemoryJobStore(), tmp_path, [ImmediateExecutor()])
    queued = manager.submit(submission())
    assert queued.status == "queued"
    assert manager.run_once() is True
    finished = manager.status(queued.id)
    assert finished.status == "succeeded"
    assert Path(finished.artifact_dir, "requested_config.json").exists()
    assert Path(finished.artifact_dir, "resolved_config.yaml").exists()
    resolved = Path(finished.artifact_dir, "resolved_config.yaml").read_text(encoding="utf-8")
    assert "max_epochs: 2" in resolved
    assert "dataset.enron_spam.EnronSpamDataset" in resolved
    assert manager.logs(queued.id)["stdout"] == "ok\n"


def test_manager_skips_incompatible_high_priority_job(tmp_path):
    """A blocked high-priority job must not starve a runnable lower-priority job."""

    class CpuOnlyExecutor(ImmediateExecutor):
        def can_run(self, resources: dict[str, Any]) -> bool:
            return ResourceRequest.model_validate(resources).gpu == 0

    manager = JobManager(InMemoryJobStore(), tmp_path, [CpuOnlyExecutor()])
    blocked = manager.submit(
        submission().model_copy(
            update={"priority": 10, "resources": ResourceRequest(cpu=1, memory_gb=1, gpu=1)}
        )
    )
    runnable = manager.submit(
        submission().model_copy(
            update={"priority": 1, "resources": ResourceRequest(cpu=1, memory_gb=1, gpu=0)}
        )
    )

    assert manager.run_once() is True
    assert manager.status(blocked.id).status == "queued"
    assert manager.status(runnable.id).status == "succeeded"


def test_failed_job_keeps_complete_executor_logs(tmp_path):
    class FailingExecutor(ImmediateExecutor):
        name = "failing"

        def submit(self, job, artifact_dir, on_heartbeat, on_finished):
            stdout = Path(artifact_dir, "stdout.log")
            stderr = Path(artifact_dir, "stderr.log")
            stdout.write_text("started\n", encoding="utf-8")
            stderr.write_text("Traceback\nall details\n", encoding="utf-8")
            on_finished(1, {"stdout": str(stdout), "stderr": str(stderr)})
            return {"stdout": str(stdout), "stderr": str(stderr)}

    manager = JobManager(InMemoryJobStore(), tmp_path, [FailingExecutor()])
    queued = manager.submit(submission())
    manager.run_once()

    failed = manager.status(queued.id)
    assert failed.status == "failed"
    assert "all details" in (failed.error or "")
    assert manager.logs(queued.id) == {
        "stdout": "started\n",
        "stderr": "Traceback\nall details\n",
    }


def test_slurm_script_maps_resources_without_client_commands(tmp_path):
    executor = SlurmExecutor(
        tmp_path,
        partition="gpu",
        account="project",
        capacity=ResourceRequest(cpu=32, memory_gb=128, gpu=2, gpu_type="A100"),
    )
    job = {
        "id": "job-123456789",
        "resources": ResourceRequest(
            cpu=8, memory_gb=32, gpu=1, gpu_type="A100", node="node-01"
        ).model_dump(mode="json"),
    }

    script = executor.build_batch_script(job, str(tmp_path / "job"))

    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=32G" in script
    assert "#SBATCH --gres=gpu:A100:1" in script
    assert "#SBATCH --nodelist=node-01" in script
    assert "#SBATCH --output=" in script and "stdout.log" in script
    assert "src/main.py" in script
    assert "network" not in script


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_type", "A100\nid > /tmp/owned"),
        ("node", "node-01\nid > /tmp/owned"),
    ],
)
def test_resource_request_rejects_unsafe_slurm_selectors(field, value):
    """Resource selectors must not inject a second line into a batch script."""

    with pytest.raises(ValueError, match="selector"):
        ResourceRequest(**{field: value})


def test_api_exposes_health_datasets_and_jobs(tmp_path):
    manager = JobManager(InMemoryJobStore(), tmp_path, [ImmediateExecutor()])
    app = create_app(manager)

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                assert (await client.get("/health")).json() == {"status": "ok"}
                datasets = await client.get("/datasets")
                assert datasets.status_code == 200
                assert any(item["name"] == "MNISTDataset" for item in datasets.json())
                response = await client.post("/jobs", json=submission().model_dump(mode="json"))
                assert response.status_code == 202
                assert response.json()["status"] == "queued"
                assert (await client.get("/jobs")).json()[0]["status"] == "queued"

    asyncio.run(exercise_api())


def test_job_submission_rejects_missing_dataset():
    with pytest.raises(ValueError, match="training.dataset"):
        normalize_training_config({"trainer": {"max_epochs": 1}})
