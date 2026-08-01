"""Job lifecycle manager and single-queue scheduler."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from backend.config_service import build_job_hydra_configs
from backend.executors import Executor, LocalExecutor, SlurmExecutor
from backend.models import JobStatus, JobSubmission, ResourceRequest
from backend.store import JobStore, ValkeyJobStore, utc_now
from model_package.exporter import build_model_wheel


TERMINAL_STATES = {"succeeded", "failed", "cancelled"}

# The failed transition is persisted atomically by the store (record update +
# queue removal in one operation). A bounded retry heals transient store
# failures; a persistent failure raises so the job keeps a recoverable running
# state instead of a partial transition.
FAILED_TRANSITION_ATTEMPTS = 3
FAILED_TRANSITION_BACKOFF_SECONDS = 0.2

# Authoritative wheel digests are lowercase hex SHA-256 strings. Anything else
# is a corrupt manifest that must never be served.
PACKAGE_SHA256_HEX = re.compile(r"[0-9a-fA-F]{64}\Z")


class PackageIntegrityError(Exception):
    """Raised when an exported wheel no longer matches its declared digest.

    The message is deliberately generic: it never includes filesystem paths,
    so the download endpoint can surface it to clients without leaking where
    job artifacts are stored.
    """


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

    def submit(self, submission: JobSubmission, *, owner_connection_id: str) -> JobStatus:
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
            "model_package": None,
            "package_error": None,
            "artifact_dir": str(artifact_dir),
            "owner_connection_id": owner_connection_id,
            "resources": submission.resources.model_dump(mode="json"),
            "submission": payload,
        }
        self.store.save_job(job_id, record)
        self.store.enqueue(job_id, submission.priority, created_at)
        self._event(job_id, "queued", {"priority": submission.priority})
        return self.status(job_id, owner_connection_id=owner_connection_id)

    def status(self, job_id: str, *, owner_connection_id: str) -> JobStatus:
        """Return public metadata for a job owned by one connection."""

        return self._public_status(self._owned_job(job_id, owner_connection_id))

    def admin_status(self, job_id: str) -> JobStatus:
        """Return job metadata without applying a browser ownership filter."""

        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return self._public_status(job)

    def list_status(self, *, owner_connection_id: str) -> list[JobStatus]:
        """Return one connection's jobs newest first."""

        return [
            self._public_status(job)
            for job in self._sorted_jobs()
            if job.get("owner_connection_id") == owner_connection_id
        ]

    def admin_list_status(self) -> list[JobStatus]:
        """Return every job, including records created before ownership existed."""

        return [self._public_status(job) for job in self._sorted_jobs()]

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
        wandb_url = _find_wandb_url(job)
        if wandb_url and wandb_url != job.get("wandb_url"):
            job["wandb_url"] = wandb_url
            self.store.save_job(job_id, job)
            self._event(job_id, "wandb_ready", {"wandb_url": wandb_url})
        else:
            self.store.save_job(job_id, job)
        self._event(job_id, "heartbeat", details)

    def _finished(self, job_id: str, return_code: int, details: dict[str, Any]) -> None:
        job = self.store.get_job(job_id)
        if job is None or job.get("status") in TERMINAL_STATES:
            with self._lock:
                self._active.pop(job_id, None)
            return
        wandb_url = _find_wandb_url(job)
        publish_wandb_url = wandb_url is not None and wandb_url != job.get("wandb_url")
        if return_code == 0:
            if publish_wandb_url:
                self._event(job_id, "wandb_ready", {"wandb_url": wandb_url})
            # The wheel is part of the promised output of a successful job:
            # the job is never persisted as ``succeeded`` before the package
            # export committed. A packaging failure (missing/corrupt safe
            # weights, unsupported adapter, exporter exception) transitions
            # the job to terminal ``failed`` through the same atomic failed
            # transition as any other failure, keeping artifacts and logs.
            if self._export_model_package(job_id):
                self._set_status(
                    job_id,
                    "succeeded",
                    finished_at=utc_now(),
                    error=None,
                    wandb_url=wandb_url,
                )
                self._drop_active(job_id)
                self._event(job_id, "succeeded", details)
            else:
                error = self._package_failure_text(job_id)
                self._set_status(
                    job_id,
                    "failed",
                    finished_at=utc_now(),
                    error=error,
                    wandb_url=wandb_url,
                )
                self._drop_active(job_id)
                self._event(job_id, "failed", {"error": error, **details})
        else:
            error = self._failure_text(job, details)
            self._set_status(
                job_id,
                "failed",
                finished_at=utc_now(),
                error=error,
                wandb_url=wandb_url,
            )
            self._drop_active(job_id)
            if publish_wandb_url:
                self._event(job_id, "wandb_ready", {"wandb_url": wandb_url})
            self._event(job_id, "failed", {"error": error, **details})

    def _drop_active(self, job_id: str) -> None:
        """Remove a finished job from the active set.

        Called only after the terminal state was persisted: keeping the entry
        on a store failure leaves the job as a tracked, recoverable ``running``
        state that stop()/cancellation or restart recovery can reconcile.
        """
        with self._lock:
            self._active.pop(job_id, None)

    def _failure_text(self, job: dict[str, Any], details: dict[str, Any]) -> str:
        """Build a compact error while retaining full logs on disk."""

        stderr_path = details.get("stderr")
        if stderr_path and Path(stderr_path).exists():
            content = Path(stderr_path).read_text(encoding="utf-8", errors="replace").strip()
            if content:
                return content[-4000:]
        return f"Training executor failed: {details}"

    def cancel(self, job_id: str, *, owner_connection_id: str) -> JobStatus:
        """Cancel a queued or active job owned by one connection."""

        self._owned_job(job_id, owner_connection_id)
        return self.admin_cancel(job_id)

    def admin_cancel(self, job_id: str) -> JobStatus:
        """Cancel any queued or active job as the backend administrator."""

        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] == "queued":
            self.store.remove_from_queue(job_id, int(job["priority"]))
            self._set_status(job_id, "cancelled", finished_at=utc_now())
            self._event(job_id, "cancelled", {})
        elif job["status"] == "running":
            with self._lock:
                active = self._active.get(job_id)
            if active:
                active[0].cancel(job_id)
            self._set_status(job_id, "cancelled", finished_at=utc_now())
            self._event(job_id, "cancelled", {})
        return self.admin_status(job_id)

    def logs(self, job_id: str, *, owner_connection_id: str) -> dict[str, str]:
        """Read logs from a job owned by one connection."""

        self._owned_job(job_id, owner_connection_id)
        return self.admin_logs(job_id)

    def admin_logs(self, job_id: str) -> dict[str, str]:
        """Read complete stdout/stderr logs for any job."""

        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        root = Path(job["artifact_dir"])
        return {
            "stdout": _read_text(root / "stdout.log"),
            "stderr": _read_text(root / "stderr.log"),
        }

    def package_download(self, job_id: str, *, owner_connection_id: str) -> tuple[Path, str, str]:
        """Resolve and verify the owned job's wheel for download.

        Returns the wheel path, its safe download filename, and the SHA-256
        digest of the bytes currently on disk. The digest is recomputed with a
        streaming read on every call and compared in constant time against the
        authoritative ``model_package.sha256`` recorded at export time, so a
        wheel that was corrupted or replaced after export is never served.

        Raises:
            KeyError: The job does not exist or is not owned by the connection.
            FileNotFoundError: The job has no exported wheel or its declared
                wheel path escapes the artifact root.
            PackageIntegrityError: The declared manifest digest is missing or
                malformed, or the wheel bytes no longer match it.
        """

        job = self._owned_job(job_id, owner_connection_id)
        package = job.get("model_package")
        if not isinstance(package, dict) or not isinstance(package.get("wheel"), str):
            raise FileNotFoundError("Model package is not available")
        root = Path(job["artifact_dir"]).resolve()
        wheel = (root / package["wheel"]).resolve()
        if root not in wheel.parents or not wheel.is_file() or wheel.suffix != ".whl":
            raise FileNotFoundError("Model package is not available")
        declared = package.get("sha256")
        if not isinstance(declared, str) or not PACKAGE_SHA256_HEX.fullmatch(declared):
            raise PackageIntegrityError("Model package integrity cannot be verified")
        computed = _sha256_hex(wheel)
        if not hmac.compare_digest(computed, declared.lower()):
            raise PackageIntegrityError("Model package integrity check failed")
        return wheel, wheel.name, computed

    def tail_logs(
        self,
        job_id: str,
        *,
        owner_connection_id: str,
        stdout_after: int = 0,
        stderr_after: int = 0,
    ) -> dict[str, dict[str, str | int | bool]]:
        """Return the appended bytes for an owned job's stdout and stderr files.

        Offsets are byte positions so a browser can continuously follow a file
        without forcing the server to load the full artifact into memory.
        """

        job = self._owned_job(job_id, owner_connection_id)
        root = Path(job["artifact_dir"])
        return {
            "stdout": _tail_text(root / "stdout.log", stdout_after),
            "stderr": _tail_text(root / "stderr.log", stderr_after),
        }

    def events(
        self,
        job_id: str,
        after: str | None = None,
        *,
        owner_connection_id: str,
    ) -> list[dict[str, Any]]:
        """Return events for a job owned by one connection."""

        self._owned_job(job_id, owner_connection_id)
        return self.admin_events(job_id, after)

    def admin_events(self, job_id: str, after: str | None = None) -> list[dict[str, Any]]:
        """Return events for any job after a stream sequence number."""

        if self.store.get_job(job_id) is None:
            raise KeyError(job_id)
        return self.store.get_events(job_id, after)

    def _owned_job(self, job_id: str, owner_connection_id: str) -> dict[str, Any]:
        """Load a job only when its persisted owner matches exactly."""

        job = self.store.get_job(job_id)
        if job is None or job.get("owner_connection_id") != owner_connection_id:
            raise KeyError(job_id)
        return job

    def _sorted_jobs(self) -> list[dict[str, Any]]:
        return sorted(self.store.list_jobs(), key=lambda item: item["created_at"], reverse=True)

    @staticmethod
    def _public_status(job: dict[str, Any]) -> JobStatus:
        return JobStatus.model_validate({key: job.get(key) for key in JobStatus.model_fields})

    def _set_status(self, job_id: str, status: str, **changes: Any) -> None:
        if status == "failed":
            self._persist_failed(job_id, changes)
            return
        job = self.store.get_job(job_id)
        if job is None:
            return
        job["status"] = status
        job.update(changes)
        self.store.save_job(job_id, job)

    def _persist_failed(self, job_id: str, changes: dict[str, Any]) -> None:
        """Atomically persist the failed transition, retrying transient failures.

        Uses the store's ``mark_failed`` so the record update and the queue
        removal are one atomic operation. A persistence failure is never
        swallowed: after a bounded number of attempts the exception propagates,
        leaving the job in its previous recoverable state rather than a partial
        transition (``failed``+queued or dequeued+``running``).
        """
        for attempt in range(FAILED_TRANSITION_ATTEMPTS):
            try:
                self.store.mark_failed(job_id, changes)
                return
            except Exception:
                if attempt == FAILED_TRANSITION_ATTEMPTS - 1:
                    raise
                time.sleep(FAILED_TRANSITION_BACKOFF_SECONDS)

    def _event(self, job_id: str, event_type: str, details: dict[str, Any]) -> None:
        self.store.append_event(job_id, {"type": event_type, "at": utc_now(), **details})

    def _package_failure_text(self, job_id: str) -> str:
        """Build the client-visible error for a job whose package export failed."""

        job = self.store.get_job(job_id)
        package_error = job.get("package_error") if job is not None else None
        if package_error:
            return f"Model package export failed: {package_error}"
        return "Model package export failed"

    def _export_model_package(self, job_id: str) -> bool:
        """Export the portable wheel for a finished job.

        Returns True when the wheel was built and its manifest persisted.
        On any failure — missing or corrupt safe weights, an unsupported
        input adapter, or an exporter exception — the job record keeps a
        client-visible ``package_error``, the ``package_failed`` event is
        emitted, and False is returned so the caller can transition the job
        to the terminal ``failed`` state. A job without safe weights is a
        packaging failure, never a silent success.
        """

        job = self.store.get_job(job_id)
        if job is None:
            return False
        artifact_dir = Path(job["artifact_dir"])
        package_name = job["submission"].get("package_name") or f"nnm_job_{job_id.replace('-', '')}"
        try:
            build_model_wheel(artifact_dir, package_name=package_name, version="0.1.0")
            manifest_path = artifact_dir / "model-package.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            job["package_error"] = str(exc)
            self.store.save_job(job_id, job)
            self._event(job_id, "package_failed", {"error": str(exc)})
            return False
        job["model_package"] = manifest
        job["package_error"] = None
        self.store.save_job(job_id, job)
        self._event(job_id, "package_ready", manifest)
        return True


def _read_text(path: Path) -> str:
    """Read a log file if it exists."""

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _sha256_hex(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hex digest of a file without loading it whole."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail_text(path: Path, offset: int, *, limit: int = 64 * 1024) -> dict[str, str | int | bool]:
    """Read at most ``limit`` new bytes, restarting when a file was replaced."""

    safe_offset = max(offset, 0)
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return {"text": "", "offset": safe_offset, "reset": False}
    reset = safe_offset > size
    start = 0 if reset else safe_offset
    with path.open("rb") as stream:
        stream.seek(start)
        chunk = stream.read(limit)
    return {
        "text": chunk.decode("utf-8", errors="replace"),
        "offset": start + len(chunk),
        "reset": reset,
    }


def _find_wandb_url(job: dict[str, Any]) -> str | None:
    """Extract the W&B run URL printed by the known training entry point."""

    root = Path(job["artifact_dir"])
    content = "\n".join(
        _read_text(root / filename) for filename in ("stdout.log", "stderr.log")
    )
    match = re.search(r"https?://wandb\.ai/[A-Za-z0-9._/-]+", content)
    return match.group(0).rstrip(".,)") if match else None
