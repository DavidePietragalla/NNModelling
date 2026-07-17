"""FastAPI application for remote NNModelling training."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.dataset_registry import discover_datasets
from backend.manager import JobManager
from backend.models import JobStatus, JobSubmission


def create_app(manager: JobManager | None = None) -> FastAPI:
    """Create the API application, optionally with an injected manager."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.manager.start()
        try:
            yield
        finally:
            app.state.manager.stop()

    app = FastAPI(
        title="NNModelling Training Backend",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.manager = manager or JobManager.from_environment()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/datasets")
    async def datasets() -> list[dict[str, Any]]:
        return [dataset.model_dump(mode="json") for dataset in discover_datasets()]

    @app.get("/compute-units")
    async def compute_units() -> list[dict[str, Any]]:
        return [executor.describe() for executor in app.state.manager.executors]

    @app.post("/jobs", response_model=JobStatus, status_code=202)
    async def submit_job(submission: JobSubmission) -> JobStatus:
        try:
            return app.state.manager.submit(submission)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/jobs", response_model=list[JobStatus])
    async def list_jobs() -> list[JobStatus]:
        return app.state.manager.list_status()

    @app.get("/jobs/{job_id}", response_model=JobStatus)
    async def get_job(job_id: str) -> JobStatus:
        try:
            return app.state.manager.status(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown job") from exc

    @app.get("/jobs/{job_id}/logs")
    async def get_logs(job_id: str) -> dict[str, str]:
        try:
            return app.state.manager.logs(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown job") from exc

    @app.get("/jobs/{job_id}/events")
    def get_events(job_id: str, after: int = 0) -> StreamingResponse:
        try:
            app.state.manager.status(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown job") from exc

        def stream() -> Iterator[str]:
            cursor = after
            idle_cycles = 0
            while idle_cycles < 120:
                events = app.state.manager.events(job_id, cursor)
                if events:
                    idle_cycles = 0
                    for event in events:
                        cursor = int(event["id"])
                        yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
                else:
                    idle_cycles += 1
                    status = app.state.manager.status(job_id)
                    if status.status in {"succeeded", "failed", "cancelled"}:
                        break
                    yield ": keep-alive\n\n"
                time.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.delete("/jobs/{job_id}", response_model=JobStatus)
    async def cancel_job(job_id: str) -> JobStatus:
        try:
            return app.state.manager.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown job") from exc

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
