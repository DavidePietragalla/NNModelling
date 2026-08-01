"""Tests for remote job configuration, storage, scheduling and API wiring."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from backend.app import create_app
from backend.auth import AuthService, InMemoryAuthStore, PairingGrant
from backend.config_service import build_job_hydra_configs, normalize_training_config
from backend.dataset_registry import discover_datasets
from backend.executors import SlurmExecutor
from backend.manager import JobManager
from backend.models import JobStatus, JobSubmission, ResourceRequest
from backend.store import InMemoryJobStore, ValkeyJobStore


TRANSFORMER_NNTREE_PATH = Path(__file__).resolve().parents[3] / "examples" / "nntrees" / "transformer_classifier.json"
OWNER = "test-connection"


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


def test_job_submission_requires_an_nnm_prefixed_package_name():
    payload = submission().model_dump(mode="json")
    payload["package_name"] = "classifier"

    with pytest.raises(ValueError, match="nnm_"):
        JobSubmission.model_validate(payload)


def test_dataset_registry_discovers_installed_classes():
    datasets = {item.target: item for item in discover_datasets()}
    assert datasets["dataset.mnist.MNISTDataset"].num_classes == 10
    assert datasets["dataset.enron_spam.EnronSpamDataset"].num_classes == 2
    assert datasets["dataset.autoencoder_mnist.AutoencoderMNIST"].num_classes is None


def test_job_config_uses_dataset_class_count_when_request_omits_it(tmp_path):
    job = submission().model_dump(mode="json")

    config_dir = build_job_hydra_configs(job, tmp_path)

    net = (config_dir / "net" / "custom_sequence.yaml").read_text(encoding="utf-8")
    assert "num_classes: 2" in net
    assert "class_names:" in net
    assert "- ham" in net
    assert "- spam" in net


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
    queued = manager.submit(submission(), owner_connection_id=OWNER)
    assert queued.status == "queued"
    assert manager.run_once() is True
    finished = manager.status(queued.id, owner_connection_id=OWNER)
    assert finished.status == "succeeded"
    assert Path(finished.artifact_dir, "requested_config.json").exists()
    assert Path(finished.artifact_dir, "resolved_config.yaml").exists()
    resolved = Path(finished.artifact_dir, "resolved_config.yaml").read_text(encoding="utf-8")
    assert "max_epochs: 2" in resolved
    assert "dataset.enron_spam.EnronSpamDataset" in resolved
    assert manager.logs(queued.id, owner_connection_id=OWNER)["stdout"] == "ok\n"


def test_manager_exports_a_model_wheel_after_a_successful_job(tmp_path, monkeypatch):
    def fake_export(artifact_dir, *, package_name, version):
        wheel = Path(artifact_dir) / "dist" / "model.whl"
        wheel.parent.mkdir()
        wheel.write_bytes(b"wheel")
        (Path(artifact_dir) / "model-package.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "package_name": package_name,
                    "version": version,
                    "wheel": "dist/model.whl",
                    "sha256": hashlib.sha256(b"wheel").hexdigest(),
                    "input_adapter": {"kind": "tensor", "version": 1},
                }
            ),
            encoding="utf-8",
        )
        return wheel

    monkeypatch.setattr("backend.manager.build_model_wheel", fake_export)
    manager = JobManager(InMemoryJobStore(), tmp_path, [ImmediateExecutor()])
    queued = manager.submit(
        submission().model_copy(update={"package_name": "nnm_mnist_classifier"}),
        owner_connection_id=OWNER,
    )
    Path(queued.artifact_dir, "weights.safetensors").write_bytes(b"safe-weights")

    manager.run_once()

    status = manager.status(queued.id, owner_connection_id=OWNER)
    assert status.model_package is not None
    assert status.model_package.package_name == "nnm_mnist_classifier"
    assert any(event["type"] == "package_ready" for event in manager.events(queued.id, owner_connection_id=OWNER))


def test_manager_tails_only_new_log_bytes_and_resets_a_stale_offset(tmp_path):
    """Log viewers receive bounded incremental output rather than whole files."""

    manager = JobManager(InMemoryJobStore(), tmp_path, [ImmediateExecutor()])
    queued = manager.submit(submission(), owner_connection_id=OWNER)
    root = Path(manager.status(queued.id, owner_connection_id=OWNER).artifact_dir)
    (root / "stdout.log").write_text("first\nsecond\n", encoding="utf-8")
    (root / "stderr.log").write_text("warning\n", encoding="utf-8")

    first = manager.tail_logs(queued.id, owner_connection_id=OWNER, stdout_after=0, stderr_after=0)
    assert first == {
        "stdout": {"text": "first\nsecond\n", "offset": 13, "reset": False},
        "stderr": {"text": "warning\n", "offset": 8, "reset": False},
    }

    (root / "stdout.log").write_text("first\nsecond\nthird\n", encoding="utf-8")
    appended = manager.tail_logs(
        queued.id,
        owner_connection_id=OWNER,
        stdout_after=first["stdout"]["offset"],
        stderr_after=first["stderr"]["offset"],
    )
    assert appended["stdout"] == {"text": "third\n", "offset": 19, "reset": False}
    assert appended["stderr"] == {"text": "", "offset": 8, "reset": False}

    reset = manager.tail_logs(queued.id, owner_connection_id=OWNER, stdout_after=99, stderr_after=99)
    assert reset["stdout"]["text"] == "first\nsecond\nthird\n"
    assert reset["stdout"]["reset"] is True
    assert reset["stderr"]["reset"] is True


def test_manager_publishes_wandb_url_as_soon_as_a_heartbeat_sees_it(tmp_path):
    manager = JobManager(InMemoryJobStore(), tmp_path, [ImmediateExecutor()])
    queued = manager.submit(submission(), owner_connection_id=OWNER)
    root = Path(manager.status(queued.id, owner_connection_id=OWNER).artifact_dir)
    (root / "stdout.log").write_text("W&B URL: https://wandb.ai/team/project/runs/live-run\n", encoding="utf-8")

    manager._heartbeat(queued.id, {"worker": "test"})

    status = manager.status(queued.id, owner_connection_id=OWNER)
    assert status.wandb_url == "https://wandb.ai/team/project/runs/live-run"
    assert manager.events(queued.id, owner_connection_id=OWNER)[-1]["type"] == "heartbeat"
    assert any(
        event["type"] == "wandb_ready" and event["wandb_url"] == status.wandb_url
        for event in manager.events(queued.id, owner_connection_id=OWNER)
    )


def test_manager_publishes_wandb_url_when_a_short_job_finishes_before_heartbeat(tmp_path):
    class WandbImmediateExecutor(ImmediateExecutor):
        def submit(self, job, artifact_dir, on_heartbeat, on_finished):
            stdout = Path(artifact_dir, "stdout.log")
            stdout.write_text("W&B URL: https://wandb.ai/team/project/runs/quick-run\n", encoding="utf-8")
            on_finished(0, {"stdout": str(stdout)})
            return {"worker": "test"}

    manager = JobManager(InMemoryJobStore(), tmp_path, [WandbImmediateExecutor()])
    queued = manager.submit(submission(), owner_connection_id=OWNER)

    manager.run_once()

    assert any(
        event["type"] == "wandb_ready"
        and event["wandb_url"] == "https://wandb.ai/team/project/runs/quick-run"
        for event in manager.events(queued.id, owner_connection_id=OWNER)
    )


def test_manager_skips_incompatible_high_priority_job(tmp_path):
    """A blocked high-priority job must not starve a runnable lower-priority job."""

    class CpuOnlyExecutor(ImmediateExecutor):
        def can_run(self, resources: dict[str, Any]) -> bool:
            return ResourceRequest.model_validate(resources).gpu == 0

    manager = JobManager(InMemoryJobStore(), tmp_path, [CpuOnlyExecutor()])
    blocked = manager.submit(
        submission().model_copy(
            update={"priority": 10, "resources": ResourceRequest(cpu=1, memory_gb=1, gpu=1)}
        ),
        owner_connection_id=OWNER,
    )
    runnable = manager.submit(
        submission().model_copy(
            update={"priority": 1, "resources": ResourceRequest(cpu=1, memory_gb=1, gpu=0)}
        ),
        owner_connection_id=OWNER,
    )

    assert manager.run_once() is True
    assert manager.status(blocked.id, owner_connection_id=OWNER).status == "queued"
    assert manager.status(runnable.id, owner_connection_id=OWNER).status == "succeeded"


def test_manager_stop_cancels_active_job_and_marks_it_failed(tmp_path):
    """Graceful shutdown must not leave an executor process orphaned."""

    class BlockingExecutor(ImmediateExecutor):
        def __init__(self) -> None:
            self.cancelled_job_ids: list[str] = []

        def submit(self, job, artifact_dir, on_heartbeat, on_finished):
            del artifact_dir, on_heartbeat, on_finished
            return {"job_id": job["id"]}

        def cancel(self, job_id: str) -> bool:
            self.cancelled_job_ids.append(job_id)
            return True

    executor = BlockingExecutor()
    manager = JobManager(InMemoryJobStore(), tmp_path, [executor])
    queued = manager.submit(submission(), owner_connection_id=OWNER)
    assert manager.run_once() is True
    assert manager.status(queued.id, owner_connection_id=OWNER).status == "running"

    manager.stop()

    stopped = manager.status(queued.id, owner_connection_id=OWNER)
    assert executor.cancelled_job_ids == [queued.id]
    assert stopped.status == "failed"
    assert stopped.finished_at is not None


def test_manager_recovery_records_terminal_event_for_interrupted_job(tmp_path):
    """Restart recovery must leave a complete terminal audit trail."""

    store = InMemoryJobStore()
    manager = JobManager(store, tmp_path, [ImmediateExecutor()])
    queued = manager.submit(submission(), owner_connection_id=OWNER)
    record = store.get_job(queued.id)
    assert record is not None
    record["status"] = "running"
    store.save_job(queued.id, record)

    manager._recover()

    recovered = manager.status(queued.id, owner_connection_id=OWNER)
    assert recovered.status == "failed"
    assert recovered.finished_at is not None
    assert manager.events(queued.id, owner_connection_id=OWNER)[-1]["type"] == "failed"


def test_recovery_failed_job_is_removed_from_queue_and_contract_preserved(tmp_path):
    """D5: a failed job must not remain claimable; metadata/events/logs stay.

    A crash after enqueue but before the claim persisted leaves the record
    marked ``running`` with its queue entry still present. Recovery fails the
    job and must clean the queue entry without deleting the job or its audit
    trail.
    """

    store = InMemoryJobStore()
    manager = JobManager(store, tmp_path, [ImmediateExecutor()])
    queued = manager.submit(submission(), owner_connection_id=OWNER)
    record = store.get_job(queued.id)
    assert record is not None
    record["status"] = "running"
    store.save_job(queued.id, record)
    assert queued.id in store.queue

    manager._recover()

    recovered = manager.status(queued.id, owner_connection_id=OWNER)
    assert recovered.status == "failed"
    assert recovered.finished_at is not None
    # Queue invariants: entry removed, not claimable, nothing left to drain.
    assert store.queue == {}
    assert store.claim_next() is None
    # Visibility contracts: list/get/events/logs remain available.
    assert [job.id for job in manager.list_status(owner_connection_id=OWNER)] == [queued.id]
    assert manager.events(queued.id, owner_connection_id=OWNER)[-1]["type"] == "failed"
    assert manager.logs(queued.id, owner_connection_id=OWNER) == {"stdout": "", "stderr": ""}


def test_recovery_keeps_queued_jobs_claimable(tmp_path):
    """D5: recovery re-enqueues queued jobs; only failed jobs leave the queue."""

    store = InMemoryJobStore()
    manager = JobManager(store, tmp_path, [ImmediateExecutor()])
    queued = manager.submit(submission(), owner_connection_id=OWNER)
    assert queued.id in store.queue

    manager._recover()

    assert manager.status(queued.id, owner_connection_id=OWNER).status == "queued"
    assert store.claim_next() == queued.id


def test_failed_job_is_not_claimable_after_executor_failure(tmp_path):
    """D5: the real execution failure path leaves no claimable queue entry."""

    class FailingExecutor(ImmediateExecutor):
        name = "failing"

        def submit(self, job, artifact_dir, on_heartbeat, on_finished):
            stdout = Path(artifact_dir, "stdout.log")
            stderr = Path(artifact_dir, "stderr.log")
            stdout.write_text("started\n", encoding="utf-8")
            stderr.write_text("Traceback\nboom\n", encoding="utf-8")
            on_finished(1, {"stdout": str(stdout), "stderr": str(stderr)})
            return {"stdout": str(stdout), "stderr": str(stderr)}

    store = InMemoryJobStore()
    manager = JobManager(store, tmp_path, [FailingExecutor()])
    queued = manager.submit(submission(), owner_connection_id=OWNER)
    assert queued.id in store.queue

    assert manager.run_once() is True

    failed = manager.status(queued.id, owner_connection_id=OWNER)
    assert failed.status == "failed"
    assert store.queue == {}
    assert store.claim_next() is None
    assert manager.logs(queued.id, owner_connection_id=OWNER)["stderr"] == "Traceback\nboom\n"


class _FailingMarkFailedStore(InMemoryJobStore):
    """In-memory store whose atomic failed transition always raises."""

    def mark_failed(self, job_id, changes):
        del job_id, changes
        raise RuntimeError("injected persistence failure")


class _FlakyMarkFailedStore(InMemoryJobStore):
    """In-memory store whose atomic failed transition fails N times first."""

    def __init__(self, failures_before_success: int) -> None:
        super().__init__()
        self.remaining = failures_before_success

    def mark_failed(self, job_id, changes):
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError("injected transient persistence failure")
        return super().mark_failed(job_id, changes)


class _PassiveExecutor:
    """Executor double that never reports completion on its own."""

    name = "passive"
    kind = "local"

    def can_run(self, resources):
        del resources
        return True

    def describe(self):
        return {"id": self.name, "kind": self.kind, "capacity": {}, "enabled": True}

    def submit(self, job, artifact_dir, on_heartbeat, on_finished):
        del job, artifact_dir, on_heartbeat, on_finished
        return {"passive": True}

    def cancel(self, job_id):
        del job_id
        return True


def _recover_with_working_store(record, tmp_path, *, owner=OWNER):
    """Simulate a restart: a fresh manager over a fresh working store."""
    working = InMemoryJobStore()
    working.save_job(record["id"], record)
    working.enqueue(record["id"], int(record["priority"]), record["created_at"])
    manager = JobManager(working, tmp_path, [ImmediateExecutor()])
    manager._recover()
    return manager, working


def test_recovery_store_failure_leaves_no_partial_state(tmp_path, monkeypatch):
    """H1: a failed atomic transition must leave neither failed+queued nor
    dequeued+running state, and the manager must surface the failure."""

    monkeypatch.setattr("backend.manager.FAILED_TRANSITION_BACKOFF_SECONDS", 0)
    store = _FailingMarkFailedStore()
    manager = JobManager(store, tmp_path, [ImmediateExecutor()])
    queued = manager.submit(submission(), owner_connection_id=OWNER)
    record = store.get_job(queued.id)
    assert record is not None
    record["status"] = "running"
    store.save_job(queued.id, record)
    assert queued.id in store.queue

    with pytest.raises(RuntimeError, match="injected persistence failure"):
        manager._recover()

    # Atomic contract: neither the record nor the queue changed.
    assert store.get_job(queued.id)["status"] == "running"
    assert queued.id in store.queue
    # No failed event was emitted (the transition never committed).
    assert all(event["type"] != "failed" for event in store.get_events(queued.id))

    # Restart recovery over the same persisted state reconciles cleanly.
    reconciled, working = _recover_with_working_store(record, tmp_path)
    assert reconciled.status(queued.id, owner_connection_id=OWNER).status == "failed"
    assert working.queue == {}
    assert working.claim_next() is None


def test_failed_transition_retries_and_heals_transient_store_failure(tmp_path, monkeypatch):
    """H1: a transient store failure on the failed transition is retried."""

    monkeypatch.setattr("backend.manager.FAILED_TRANSITION_BACKOFF_SECONDS", 0)
    store = _FlakyMarkFailedStore(failures_before_success=1)
    manager = JobManager(store, tmp_path, [ImmediateExecutor()])
    queued = manager.submit(submission(), owner_connection_id=OWNER)
    record = store.get_job(queued.id)
    assert record is not None
    record["status"] = "running"
    store.save_job(queued.id, record)
    assert queued.id in store.queue

    manager._recover()

    recovered = manager.status(queued.id, owner_connection_id=OWNER)
    assert recovered.status == "failed"
    assert store.queue == {}
    assert store.claim_next() is None
    assert manager.events(queued.id, owner_connection_id=OWNER)[-1]["type"] == "failed"


def test_terminal_persistence_failure_keeps_recoverable_running_state(tmp_path, monkeypatch):
    """H1: a store failure after executor completion must not be masked.

    The job stays a tracked ``running`` job (active entry retained) with no
    partial failed transition, so stop()/restart can reconcile it.
    """

    monkeypatch.setattr("backend.manager.FAILED_TRANSITION_BACKOFF_SECONDS", 0)
    store = _FailingMarkFailedStore()
    manager = JobManager(store, tmp_path, [_PassiveExecutor()])
    queued = manager.submit(submission(), owner_connection_id=OWNER)
    assert manager.run_once() is True
    assert manager.status(queued.id, owner_connection_id=OWNER).status == "running"
    assert queued.id in manager._active

    with pytest.raises(RuntimeError, match="injected persistence failure"):
        manager._finished(queued.id, 1, {"stdout": "x", "stderr": "y"})

    # No partial transition: still running, still tracked, no failed event.
    assert manager.status(queued.id, owner_connection_id=OWNER).status == "running"
    assert queued.id in manager._active
    assert all(event["type"] != "failed" for event in store.get_events(queued.id))

    # Restart recovery reconciles the same persisted record.
    record = store.get_job(queued.id)
    assert record is not None
    reconciled, working = _recover_with_working_store(record, tmp_path)
    assert reconciled.status(queued.id, owner_connection_id=OWNER).status == "failed"
    assert working.queue == {}
    assert reconciled.events(queued.id, owner_connection_id=OWNER)[-1]["type"] == "failed"


def test_manager_cancel_queued_job_emits_cancelled_event(tmp_path):
    """Queued and running cancellations must have the same observable transition."""

    manager = JobManager(InMemoryJobStore(), tmp_path, [ImmediateExecutor()])
    queued = manager.submit(submission(), owner_connection_id=OWNER)

    cancelled = manager.cancel(queued.id, owner_connection_id=OWNER)

    assert cancelled.status == "cancelled"
    assert manager.events(queued.id, owner_connection_id=OWNER)[-1]["type"] == "cancelled"


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
    queued = manager.submit(submission(), owner_connection_id=OWNER)
    manager.run_once()

    failed = manager.status(queued.id, owner_connection_id=OWNER)
    assert failed.status == "failed"
    assert "all details" in (failed.error or "")
    assert manager.logs(queued.id, owner_connection_id=OWNER) == {
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


def test_api_requires_pairing_and_scopes_jobs_to_their_connection(tmp_path):
    manager = JobManager(InMemoryJobStore(), tmp_path, [ImmediateExecutor()])
    auth = AuthService(InMemoryAuthStore())
    first = auth.create_pairing("First", client_host="127.0.0.1")
    second = auth.create_pairing("Second", client_host="127.0.0.2")
    auth.approve(first.request_id)
    auth.approve(second.request_id)
    app = create_app(manager, auth_service=auth, admin_token="admin-secret")

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                assert (await client.get("/health")).json() == {"status": "ok"}
                assert (await client.get("/datasets")).status_code == 401
                first_headers = {"authorization": f"Bearer {first.token}"}
                second_headers = {"authorization": f"Bearer {second.token}"}
                datasets = await client.get("/datasets", headers=first_headers)
                assert datasets.status_code == 200
                assert any(item["name"] == "MNISTDataset" for item in datasets.json())
                response = await client.post(
                    "/jobs",
                    headers=first_headers,
                    json=submission().model_dump(mode="json"),
                )
                assert response.status_code == 202
                assert response.json()["status"] == "queued"
                job_id = response.json()["id"]
                assert len((await client.get("/jobs", headers=first_headers)).json()) == 1
                assert (await client.get("/jobs", headers=second_headers)).json() == []
                assert (await client.get(f"/jobs/{job_id}", headers=second_headers)).status_code == 404
                tail = await client.get(
                    f"/jobs/{job_id}/logs/tail?stdout_after=0&stderr_after=0",
                    headers=first_headers,
                )
                assert tail.status_code == 200
                assert set(tail.json()["stdout"]) == {"text", "offset", "reset"}
                assert (await client.get(f"/jobs/{job_id}/logs/tail", headers=second_headers)).status_code == 404

    asyncio.run(exercise_api())


def _materialize_package(
    store: InMemoryJobStore,
    job: JobStatus,
    *,
    wheel_bytes: bytes = b"wheel-content",
    wheel_rel: str = "dist/nnm-model.whl",
    sha256: str | None = None,
) -> dict[str, Any]:
    """Write a wheel and a matching manifest into a job's artifact directory."""
    record = store.get_job(job.id)
    assert record is not None
    artifact = Path(record["artifact_dir"])
    wheel = artifact / wheel_rel
    wheel.parent.mkdir(parents=True, exist_ok=True)
    wheel.write_bytes(wheel_bytes)
    record["model_package"] = {
        "schema_version": 1,
        "package_name": "nnm-model",
        "version": "0.1.0",
        "wheel": wheel_rel,
        "sha256": sha256 if sha256 is not None else hashlib.sha256(wheel_bytes).hexdigest(),
        "input_adapter": {"kind": "tensor", "version": 1},
    }
    store.save_job(job.id, record)
    return record


def _download_context(
    tmp_path: Path,
) -> tuple[Any, InMemoryJobStore, JobStatus, PairingGrant, PairingGrant]:
    """Build an authenticated app with one owner, one other, and a packaged job."""
    store = InMemoryJobStore()
    manager = JobManager(store, tmp_path, [ImmediateExecutor()])
    auth = AuthService(InMemoryAuthStore())
    owner = auth.create_pairing("Owner", client_host="127.0.0.1")
    other = auth.create_pairing("Other", client_host="127.0.0.2")
    auth.approve(owner.request_id)
    auth.approve(other.request_id)
    job = manager.submit(submission(), owner_connection_id=owner.connection_id)
    app = create_app(manager, auth_service=auth, admin_token="admin-secret")
    return app, store, job, owner, other


def test_api_downloads_only_the_owner_model_wheel(tmp_path):
    app, store, job, owner, other = _download_context(tmp_path)
    _materialize_package(store, job)
    expected = hashlib.sha256(b"wheel-content").hexdigest()

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                owner_headers = {"authorization": f"Bearer {owner.token}"}
                other_headers = {"authorization": f"Bearer {other.token}"}
                response = await client.get(f"/jobs/{job.id}/package", headers=owner_headers)
                assert response.status_code == 200
                assert response.content == b"wheel-content"
                assert response.headers["x-nnm-sha256"] == expected
                assert "attachment" in response.headers["content-disposition"]
                assert (await client.get(f"/jobs/{job.id}/package", headers=other_headers)).status_code == 404

    asyncio.run(exercise_api())


def test_api_rejects_a_wheel_replaced_after_export(tmp_path):
    """D3: bytes that no longer match the manifest digest are never served."""
    app, store, job, owner, _other = _download_context(tmp_path)
    _materialize_package(store, job, wheel_bytes=b"original")
    record = store.get_job(job.id)
    assert record is not None
    wheel = Path(record["artifact_dir"]) / record["model_package"]["wheel"]
    wheel.write_bytes(b"tampered-bytes")

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                headers = {"authorization": f"Bearer {owner.token}"}
                response = await client.get(f"/jobs/{job.id}/package", headers=headers)
                assert response.status_code == 409
                assert response.json()["detail"] == {
                    "code": "package_integrity_error",
                    "message": "Model package integrity check failed",
                }
                # A conflicted download must not leak where artifacts live.
                assert str(tmp_path) not in response.text

    asyncio.run(exercise_api())


def test_api_rejects_a_missing_declared_digest(tmp_path):
    """A manifest without a sha256 cannot be verified, so nothing is served."""
    app, store, job, owner, _other = _download_context(tmp_path)
    _materialize_package(store, job)
    record = store.get_job(job.id)
    assert record is not None
    del record["model_package"]["sha256"]
    store.save_job(job.id, record)

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                headers = {"authorization": f"Bearer {owner.token}"}
                response = await client.get(f"/jobs/{job.id}/package", headers=headers)
                assert response.status_code == 409
                assert response.json()["detail"] == {
                    "code": "package_integrity_error",
                    "message": "Model package integrity cannot be verified",
                }
                assert str(tmp_path) not in response.text

    asyncio.run(exercise_api())


def test_api_rejects_a_malformed_declared_digest(tmp_path):
    """A non-hex manifest digest is corrupt state, not a downloadable package."""
    app, store, job, owner, _other = _download_context(tmp_path)
    _materialize_package(store, job, sha256="not-a-hex-digest")

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                headers = {"authorization": f"Bearer {owner.token}"}
                response = await client.get(f"/jobs/{job.id}/package", headers=headers)
                assert response.status_code == 409
                assert response.json()["detail"]["code"] == "package_integrity_error"
                assert str(tmp_path) not in response.text

    asyncio.run(exercise_api())


def test_api_exposes_the_package_digest_header_via_cors(tmp_path):
    """Browsers must be allowed to read X-NNM-SHA256 on the download response."""
    app, store, job, owner, _other = _download_context(tmp_path)
    _materialize_package(store, job)

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                headers = {
                    "authorization": f"Bearer {owner.token}",
                    "origin": "http://127.0.0.1:5173",
                }
                response = await client.get(f"/jobs/{job.id}/package", headers=headers)
                assert response.status_code == 200
                assert response.headers["x-nnm-sha256"] == hashlib.sha256(b"wheel-content").hexdigest()
                assert "X-NNM-SHA256" in response.headers.get("access-control-expose-headers", "")

    asyncio.run(exercise_api())


def test_public_pairing_flow_waits_for_administrator_approval(tmp_path):
    manager = JobManager(InMemoryJobStore(), tmp_path, [ImmediateExecutor()])
    auth = AuthService(InMemoryAuthStore())
    app = create_app(manager, auth_service=auth, admin_token="admin-secret")

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                created = await client.post("/pairing/requests", json={"device_name": "Lab browser"})
                assert created.status_code == 201
                body = created.json()
                headers = {"authorization": f"Bearer {body['token']}"}
                pending = await client.get(f"/pairing/requests/{body['request_id']}", headers=headers)
                assert pending.json()["status"] == "pending"
                assert (await client.get("/session", headers=headers)).status_code == 401

                auth.approve(body["request_id"])

                assert (await client.get("/session", headers=headers)).status_code == 200

    asyncio.run(exercise_api())


def test_admin_api_approves_sessions_and_manages_all_jobs(tmp_path):
    store = InMemoryJobStore()
    manager = JobManager(store, tmp_path, [ImmediateExecutor()])
    auth = AuthService(InMemoryAuthStore())
    pairing = auth.create_pairing("Admin test", client_host="127.0.0.1")
    app = create_app(manager, auth_service=auth, admin_token="admin-secret")

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                assert (await client.get("/admin/pairing/requests")).status_code == 401
                wrong = {"x-nnm-admin-token": "wrong"}
                assert (await client.get("/admin/pairing/requests", headers=wrong)).status_code == 401
                admin = {"x-nnm-admin-token": "admin-secret"}
                pending = await client.get("/admin/pairing/requests", headers=admin)
                assert [item["id"] for item in pending.json()] == [pairing.request_id]

                approved = await client.post(
                    f"/admin/pairing/requests/{pairing.request_id}/approve",
                    headers=admin,
                    json={"ttl": "8h"},
                )
                assert approved.status_code == 200
                assert approved.json()["status"] == "active"
                session_headers = {"authorization": f"Bearer {pairing.token}"}
                submitted = await client.post(
                    "/jobs",
                    headers=session_headers,
                    json=submission().model_dump(mode="json"),
                )
                assert submitted.status_code == 202
                job_id = submitted.json()["id"]
                assert any(item["id"] == job_id for item in (await client.get("/admin/jobs", headers=admin)).json())
                assert (await client.delete(f"/admin/jobs/{job_id}", headers=admin)).status_code == 200

                sessions = await client.get("/admin/sessions", headers=admin)
                assert sessions.json()[0]["id"] == pairing.connection_id
                assert "token_hash" not in sessions.text
                revoked = await client.delete(
                    f"/admin/sessions/{pairing.connection_id}",
                    headers=admin,
                )
                assert revoked.json()["status"] == "revoked"
                assert (await client.get("/jobs", headers=session_headers)).status_code == 401

    asyncio.run(exercise_api())


def test_job_submission_rejects_missing_dataset():
    with pytest.raises(ValueError, match="training.dataset"):
        normalize_training_config({"trainer": {"max_epochs": 1}})
