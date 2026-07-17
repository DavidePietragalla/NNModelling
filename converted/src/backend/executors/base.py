"""Common executor protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


HeartbeatCallback = Callable[[dict[str, Any]], None]
FinishedCallback = Callable[[int, dict[str, Any]], None]


class Executor(Protocol):
    """Interface implemented by local and Slurm job executors."""

    name: str
    kind: str

    def can_run(self, resources: dict[str, Any]) -> bool: ...

    def submit(
        self,
        job: dict[str, Any],
        artifact_dir: str,
        on_heartbeat: HeartbeatCallback,
        on_finished: FinishedCallback,
    ) -> dict[str, Any]: ...

    def cancel(self, job_id: str) -> bool: ...

    def describe(self) -> dict[str, Any]: ...

