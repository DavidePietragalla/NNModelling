"""Valkey persistence and an in-memory test double for job state."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from typing import Any, Protocol


EventCursor = str | None


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat()


class JobStore(Protocol):
    """Storage operations required by the scheduler."""

    def save_job(self, job_id: str, data: dict[str, Any]) -> None: ...

    def get_job(self, job_id: str) -> dict[str, Any] | None: ...

    def list_jobs(self) -> list[dict[str, Any]]: ...

    def enqueue(self, job_id: str, priority: int, created_at: str) -> None: ...

    def remove_from_queue(self, job_id: str, priority: int) -> None: ...

    def claim_next(self) -> str | None: ...

    def append_event(self, job_id: str, event: dict[str, Any]) -> None: ...

    def get_events(self, job_id: str, after: EventCursor = None) -> list[dict[str, Any]]: ...


class InMemoryJobStore:
    """Thread-safe store used by unit and API tests."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.queue: dict[str, tuple[int, str, str]] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def save_job(self, job_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            self.jobs[job_id] = json.loads(json.dumps(data))

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self.jobs.get(job_id)
            return json.loads(json.dumps(data)) if data is not None else None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [json.loads(json.dumps(item)) for item in self.jobs.values()]

    def enqueue(self, job_id: str, priority: int, created_at: str) -> None:
        with self._lock:
            self.queue[job_id] = (-priority, created_at, job_id)

    def remove_from_queue(self, job_id: str, priority: int) -> None:
        with self._lock:
            self.queue.pop(job_id, None)

    def claim_next(self) -> str | None:
        with self._lock:
            if not self.queue:
                return None
            job_id = min(self.queue, key=self.queue.__getitem__)
            self.queue.pop(job_id)
            return job_id

    def append_event(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            stream = self.events.setdefault(job_id, [])
            stream.append({"id": f"{len(stream) + 1}-0", **json.loads(json.dumps(event))})

    def get_events(self, job_id: str, after: EventCursor = None) -> list[dict[str, Any]]:
        with self._lock:
            return [
                json.loads(json.dumps(event))
                for event in self.events.get(job_id, [])
                if after is None or _stream_id_key(event["id"]) > _stream_id_key(after)
            ]


class ValkeyJobStore:
    """Persistent Valkey implementation of the job store.

    The queue is split into a priority index and one timestamp-sorted set per
    priority. This preserves priority then FIFO ordering without encoding both
    values into a floating-point score.
    """

    def __init__(self, url: str = "valkey://127.0.0.1:6379/0") -> None:
        try:
            import valkey
        except ImportError as exc:  # pragma: no cover - exercised in deployment
            raise RuntimeError("Install the 'valkey' Python package to use ValkeyJobStore") from exc
        self.client = valkey.from_url(url, decode_responses=True)
        self.queue_priorities = "queue:priorities"
        self.queue_prefix = "queue:priority:"

    def save_job(self, job_id: str, data: dict[str, Any]) -> None:
        self.client.set(f"job:{job_id}", json.dumps(data))

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        value = self.client.get(f"job:{job_id}")
        return json.loads(value) if value else None

    def list_jobs(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for key in self.client.scan_iter(match="job:*"):
            if ":" in key[4:]:
                continue
            value = self.client.get(key)
            if value:
                result.append(json.loads(value))
        return result

    def enqueue(self, job_id: str, priority: int, created_at: str) -> None:
        score = datetime_from_iso(created_at)
        pipe = self.client.pipeline()
        pipe.zadd(self.queue_priorities, {str(priority): -priority})
        pipe.zadd(f"{self.queue_prefix}{priority}", {job_id: score})
        pipe.execute()

    def remove_from_queue(self, job_id: str, priority: int) -> None:
        queue_key = f"{self.queue_prefix}{priority}"
        pipe = self.client.pipeline()
        pipe.zrem(queue_key, job_id)
        pipe.zcard(queue_key)
        remaining = pipe.execute()[1]
        if remaining == 0:
            self.client.zrem(self.queue_priorities, str(priority))

    def claim_next(self) -> str | None:
        # Queue selection and removal must be one Valkey operation. This is
        # important even while the manager has one worker: the data model is
        # intentionally ready for concurrent schedulers later.
        result = self.client.eval(
            """
            local priorities = redis.call('ZRANGE', KEYS[1], 0, 0)
            if #priorities == 0 then return false end
            local priority = priorities[1]
            local queue = ARGV[1] .. priority
            local ids = redis.call('ZRANGE', queue, 0, 0)
            if #ids == 0 then
                redis.call('ZREM', KEYS[1], priority)
                return false
            end
            local job_id = ids[1]
            redis.call('ZREM', queue, job_id)
            if redis.call('ZCARD', queue) == 0 then
                redis.call('ZREM', KEYS[1], priority)
            end
            return job_id
            """,
            1,
            self.queue_priorities,
            self.queue_prefix,
        )
        return str(result) if result else None

    def append_event(self, job_id: str, event: dict[str, Any]) -> None:
        self.client.xadd(f"job:{job_id}:events", {"event": json.dumps(event)}, maxlen=1000)

    def get_events(self, job_id: str, after: EventCursor = None) -> list[dict[str, Any]]:
        minimum = f"({after}" if after else "-"
        events = self.client.xrange(f"job:{job_id}:events", min=minimum, max="+", count=1000)
        return [
            {"id": str(event_id), **json.loads(fields["event"])}
            for event_id, fields in events
        ]


def _stream_id_key(value: str) -> tuple[int, int]:
    """Return a comparable pair for Valkey stream IDs and the in-memory double."""

    milliseconds, separator, sequence = value.partition("-")
    return int(milliseconds), int(sequence) if separator else 0


def datetime_from_iso(value: str) -> float:
    """Convert an ISO timestamp to a monotonic-enough queue score."""

    from datetime import datetime

    return datetime.fromisoformat(value).timestamp()
