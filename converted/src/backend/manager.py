"""Job lifecycle manager and single-queue scheduler."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from backend.config_service import build_job_hydra_configs
from backend.executors import Executor, LocalExecutor, SlurmExecutor
from backend.models import JobStatus, JobSubmission, ResourceRequest
from backend.store import JobStore, ValkeyJobStore, utc_now


TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


class JobManager:
    """Persist jobs, schedule them by priority, and run one at a time."""

    def __init__(
        self,
        store: JobStore,
        artifact_root: str | Path,
        executors: list[Executor],
        max_running_jobs: int = 1,
        poll_interval: float = 0.25,
    ) -> None:
        if not executors:
            raise ValueError("At least one executor is required")
        self.store = store
        self.artifact_root = Path(artifact_root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.executors = executors
        self.max_running_jobs = max_running_jobs
        self.poll_interval = poll_interval
        self._active: dict[str, tuple[Executor, dict[str, Any]]] = {}
        self._round_robin_cursor = 0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_environment(cls) -> "JobManager":
        """Build a production manager from backend environment variables."""

        converted_dir = Path(
            os.getenv("NNM_CONVERTED_DIR", Path(__file__).resolve().parents[2])
        ).expanduser().resolve()
        default_artifact_root = converted_dir / "jobs"
        artifact_root = Path(
            os.getenv("NNM_BACKEND_ARTIFACT_ROOT", str(default_artifact_root))
        ).expanduser().resolve()
        store = ValkeyJobStore(os.getenv("NNM_VALKEY_URL", "valkey://127.0.0.1:6379/0"))
        executors: list[Executor] = [LocalExecutor(converted_dir)]
        if os.getenv("NNM_ENABLE_SLURM", "0").lower() in {"1", "true", "yes"}:
            slurm_gpu_type = os.getenv("NNM_SLURM_GPU_TYPE") or None
            executors.append(
                SlurmExecutor(
                    converted_dir,
                    unit_id=os.getenv("NNM_SLURM_UNIT_ID", "slurm-main"),
                    partition=os.getenv("NNM_SLURM_PARTITION"),
                    account=os.getenv("NNM_SLURM_ACCOUNT"),
                    ssh_host=os.getenv("NNM_SLURM_SSH_HOST"),
                    project_dir=os.getenv("NNM_SLURM_PROJECT_DIR", str(converted_dir)),
                    capacity=ResourceRequest(
                        cpu=int(os.getenv("NNM_SLURM_CPU", "1")),
                        memory_gb=float(os.getenv("NNM_SLURM_MEMORY_GB", "1")),
                        gpu=int(os.getenv("NNM_SLURM_GPU", "1")),
                        gpu_type=slurm_gpu_type,
                    ),
                )
            )
        return cls(store, artifact_root, executors)

    def start(self) -> None:
        """Start the scheduler thread and recover persisted queue metadata."""

        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._recover()
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="nnm-scheduler", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop scheduling and terminate executions owned by this manager."""

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            active_jobs = list(self._active.items())
            self._active.clear()
        for job_id, (executor, _handle) in active_jobs:
            job = self.store.get_job(job_id)
            if job is None or job.get("status") in TERMINAL_STATES:
                continue
            try:
                executor.cancel(job_id)
                error = "Backend stopped and cancelled the active execution; job must be resubmitted."
            except Exception as exc:
                error = f"Backend stopped before active execution could be cancelled: {exc}"
            self._set_status(job_id, "failed", finished_at=utc_now(), error=error)
            self._event(job_id, "failed", {"error": error})

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                # A scheduler loop must remain alive; the job-specific error
                # is recorded by submit/finish paths whenever possible.
                pass
            self._stop_event.wait(self.poll_interval)

    def _recover(self) -> None:
        """Restore queued jobs and fail unverifiable pre-restart executions."""

        for job in self.store.list_jobs():
            status = job.get("status")
            if status == "queued":
                self.store.enqueue(job["id"], int(job["priority"]), job["created_at"])
            elif status == "running":
                error = "Backend restarted while the execution was running; job must be resubmitted."
                self._set_status(
                    job["id"],
                    "failed",
                    finished_at=utc_now(),
                    error=error,
                )
                self._event(job["id"], "failed", {"error": error})

    def submit(self, submission: JobSubmission) -> JobStatus:
        """Validate, materialize and enqueue a complete job document."""

        job_id = str(uuid.uuid4())
        created_at = utc_now()
        artifact_dir = self.artifact_root / job_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = submission.model_dump(mode="json")
        payload["id"] = job_id
        payload["created_at"] = created_at
        payload["artifact_dir"] = str(artifact_dir)
        requested_path = artifact_dir / "requested_config.json"
        requested_path.write_text(json.dumps(submission.model_dump(mode="json"), indent=2), encoding="utf-8")
        build_job_hydra_configs(payload, artifact_dir)
        record = {
            "id": job_id,
            "status": "queued",
            "priority": submission.priority,
            "created_at": created_at,
            "started_at": None,
            "finished_at": None,
            "executor": None,
            "compute_unit": None,
            "error": None,
            "heartbeat_at": None,
            "wandb_url": None,
            "artifact_dir": str(artifact_dir),
            "resources": submission.resources.model_dump(mode="json"),
            "submission": payload,
        }
        self.store.save_job(job_id, record)
        self.store.enqueue(job_id, submission.priority, created_at)
        self._event(job_id, "queued", {"priority": submission.priority})
        return self.status(job_id)

    def status(self, job_id: str) -> JobStatus:
        """Return public job metadata."""

        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return JobStatus.model_validate({key: job.get(key) for key in JobStatus.model_fields})

    def list_status(self) -> list[JobStatus]:
        """Return jobs newest first."""

        return [
            JobStatus.model_validate({key: job.get(key) for key in JobStatus.model_fields})
            for job in sorted(self.store.list_jobs(), key=lambda item: item["created_at"], reverse=True)
        ]

    def run_once(self) -> bool:
        """Claim and start the highest-priority compatible job, if any."""

        with self._lock:
            if len(self._active) >= self.max_running_jobs:
                return False
            deferred_job_ids: list[str] = []
            try:
                while True:
                    job_id = self.store.claim_next()
                    if job_id is None:
                        return False
                    job = self.store.get_job(job_id)
                    if job is None or job.get("status") != "queued":
                        continue
                    executor = self._select_executor(job["resources"])
                    if executor is None:
                        deferred_job_ids.append(job_id)
                        continue
                    self._set_status(
                        job_id,
                        "running",
                        started_at=utc_now(),
                        executor=executor.kind,
                        compute_unit=executor.name,
                    )
                    try:
                        handle = executor.submit(
                            job,
                            job["artifact_dir"],
                            lambda details: self._heartbeat(job_id, details),
                            lambda return_code, details: self._finished(job_id, return_code, details),
                        )
                    except Exception as exc:
                        self._set_status(job_id, "failed", finished_at=utc_now(), error=str(exc))
                        self._event(job_id, "failed", {"error": str(exc)})
                        return False
                    current = self.store.get_job(job_id) or job
                    current["executor_details"] = handle
                    self.store.save_job(job_id, current)
                    if current.get("status") not in TERMINAL_STATES:
                        self._active[job_id] = (executor, handle)
                    self._event(job_id, "running", {"executor": executor.name, **handle})
                    return True
            finally:
                for deferred_job_id in deferred_job_ids:
                    deferred_job = self.store.get_job(deferred_job_id)
                    if deferred_job is not None and deferred_job.get("status") == "queued":
                        self.store.enqueue(
                            deferred_job_id,
                            int(deferred_job["priority"]),
                            deferred_job["created_at"],
                        )

    def _select_executor(self, resources: dict[str, Any]) -> Executor | None:
        """Select a compatible executor with a round-robin cursor."""

        count = len(self.executors)
        for offset in range(count):
            index = (self._round_robin_cursor + offset) % count
            candidate = self.executors[index]
            if candidate.can_run(resources):
                self._round_robin_cursor = (index + 1) % count
                return candidate
        return None

    def _heartbeat(self, job_id: str, details: dict[str, Any]) -> None:
        job = self.store.get_job(job_id)
        if job is None or job.get("status") in TERMINAL_STATES:
            return
        timestamp = utc_now()
        job["heartbeat_at"] = timestamp
        job["heartbeat"] = details
        self.store.save_job(job_id, job)
        self._event(job_id, "heartbeat", details)

    def _finished(self, job_id: str, return_code: int, details: dict[str, Any]) -> None:
        with self._lock:
            self._active.pop(job_id, None)
        job = self.store.get_job(job_id)
        if job is None or job.get("status") in TERMINAL_STATES:
            return
        if return_code == 0:
            self._set_status(
                job_id,
                "succeeded",
                finished_at=utc_now(),
                error=None,
                wandb_url=_find_wandb_url(job),
            )
            self._event(job_id, "succeeded", details)
        else:
            error = self._failure_text(job, details)
            self._set_status(
                job_id,
                "failed",
                finished_at=utc_now(),
                error=error,
                wandb_url=_find_wandb_url(job),
            )
            self._event(job_id, "failed", {"error": error, **details})

    def _failure_text(self, job: dict[str, Any], details: dict[str, Any]) -> str:
        """Build a compact error while retaining full logs on disk."""

        stderr_path = details.get("stderr")
        if stderr_path and Path(stderr_path).exists():
            content = Path(stderr_path).read_text(encoding="utf-8", errors="replace").strip()
            if content:
                return content[-4000:]
        return f"Training executor failed: {details}"

    def cancel(self, job_id: str) -> JobStatus:
        """Cancel a queued or active job."""

        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] == "queued":
            self.store.remove_from_queue(job_id, int(job["priority"]))
            self._set_status(job_id, "cancelled", finished_at=utc_now())
        elif job["status"] == "running":
            with self._lock:
                active = self._active.get(job_id)
            if active:
                active[0].cancel(job_id)
            self._set_status(job_id, "cancelled", finished_at=utc_now())
            self._event(job_id, "cancelled", {})
        return self.status(job_id)

    def logs(self, job_id: str) -> dict[str, str]:
        """Read complete stdout/stderr logs from the job artifact directory."""

        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        root = Path(job["artifact_dir"])
        return {
            "stdout": _read_text(root / "stdout.log"),
            "stderr": _read_text(root / "stderr.log"),
        }

    def events(self, job_id: str, after: str | None = None) -> list[dict[str, Any]]:
        """Return events after a stream sequence number."""

        if self.store.get_job(job_id) is None:
            raise KeyError(job_id)
        return self.store.get_events(job_id, after)

    def _set_status(self, job_id: str, status: str, **changes: Any) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return
        job["status"] = status
        job.update(changes)
        self.store.save_job(job_id, job)

    def _event(self, job_id: str, event_type: str, details: dict[str, Any]) -> None:
        self.store.append_event(job_id, {"type": event_type, "at": utc_now(), **details})


def _read_text(path: Path) -> str:
    """Read a log file if it exists."""

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _find_wandb_url(job: dict[str, Any]) -> str | None:
    """Extract the W&B run URL printed by the known training entry point."""

    root = Path(job["artifact_dir"])
    content = "\n".join(
        _read_text(root / filename) for filename in ("stdout.log", "stderr.log")
    )
    match = re.search(r"https?://wandb\.ai/[A-Za-z0-9._/-]+", content)
    return match.group(0).rstrip(".,)") if match else None
