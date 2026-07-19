"""FastAPI application for authenticated remote NNModelling training."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.auth import (
    AuthError,
    AuthService,
    InMemoryAuthStore,
    PairingLimitError,
    ValkeyAuthStore,
    parse_duration,
)
from backend.dataset_registry import discover_datasets
from backend.manager import JobManager
from backend.models import (
    JobStatus,
    JobSubmission,
    PairingGrantResponse,
    PairingRequestInput,
    PairingStatusResponse,
    SessionInfo,
)


DEFAULT_ALLOWED_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://localhost:5173",
    "http://localhost:5174",
]


def create_app(
    manager: JobManager | None = None,
    *,
    auth_service: AuthService | None = None,
    admin_token: str | None = None,
    allowed_origins: list[str] | None = None,
) -> FastAPI:
    """Create the API application with injectable services for tests."""

    injected_manager = manager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.manager.start()
        try:
            yield
        finally:
            app.state.manager.stop()

    app = FastAPI(
        title="NNModelling Training Backend",
        version="0.2.0",
        lifespan=lifespan,
    )
    origins = allowed_origins if allowed_origins is not None else _allowed_origins_from_environment()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Last-Event-ID", "X-NNM-Admin-Token"],
    )
    app.state.manager = manager or JobManager.from_environment()
    app.state.auth = auth_service or _auth_from_environment(in_memory=injected_manager is not None)
    app.state.admin_token = admin_token if admin_token is not None else _read_admin_token()

    async def bearer_token(authorization: str | None = Header(default=None)) -> str:
        if authorization is None or not authorization.startswith("Bearer "):
            raise _auth_http_error(AuthError("missing_token"))
        token = authorization.removeprefix("Bearer ").strip()
        if not token:
            raise _auth_http_error(AuthError("missing_token"))
        return token

    async def current_connection(token: str = Depends(bearer_token)) -> dict[str, Any]:
        try:
            return app.state.auth.authenticate(token)
        except AuthError as exc:
            raise _auth_http_error(exc) from exc

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/pairing/requests", response_model=PairingGrantResponse, status_code=201)
    async def create_pairing(body: PairingRequestInput, request: Request) -> PairingGrantResponse:
        try:
            grant = app.state.auth.create_pairing(
                body.device_name,
                client_host=_client_host(request),
                origin=request.headers.get("origin"),
                user_agent=request.headers.get("user-agent"),
            )
            return PairingGrantResponse(**grant.__dict__)
        except PairingLimitError as exc:
            raise HTTPException(
                status_code=429,
                detail={"code": exc.code, "message": str(exc)},
                headers={"Retry-After": "60"},
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/pairing/requests/{request_id}", response_model=PairingStatusResponse)
    async def pairing_status(
        request_id: str,
        token: str = Depends(bearer_token),
    ) -> PairingStatusResponse:
        try:
            return PairingStatusResponse.model_validate(app.state.auth.pairing_status(request_id, token))
        except AuthError as exc:
            raise _auth_http_error(exc, not_found_codes={"pairing_not_found"}) from exc

    @app.post("/pairing/renewals", response_model=PairingGrantResponse, status_code=201)
    async def create_renewal(
        request: Request,
        token: str = Depends(bearer_token),
    ) -> PairingGrantResponse:
        try:
            grant = app.state.auth.create_renewal(
                token,
                client_host=_client_host(request),
                origin=request.headers.get("origin"),
                user_agent=request.headers.get("user-agent"),
            )
            return PairingGrantResponse(**grant.__dict__)
        except PairingLimitError as exc:
            raise HTTPException(
                status_code=429,
                detail={"code": exc.code, "message": str(exc)},
                headers={"Retry-After": "60"},
            ) from exc
        except AuthError as exc:
            raise _auth_http_error(exc) from exc

    @app.get("/session", response_model=SessionInfo)
    async def session(connection: dict[str, Any] = Depends(current_connection)) -> SessionInfo:
        return SessionInfo.model_validate(app.state.auth.public_session(connection))

    @app.delete("/session", response_model=SessionInfo)
    async def revoke_session(connection: dict[str, Any] = Depends(current_connection)) -> SessionInfo:
        return SessionInfo.model_validate(app.state.auth.revoke(connection["id"]))

    @app.get("/datasets")
    async def datasets(
        _connection: dict[str, Any] = Depends(current_connection),
    ) -> list[dict[str, Any]]:
        return [dataset.model_dump(mode="json") for dataset in discover_datasets()]

    @app.get("/compute-units")
    async def compute_units(
        _connection: dict[str, Any] = Depends(current_connection),
    ) -> list[dict[str, Any]]:
        return [executor.describe() for executor in app.state.manager.executors]

    @app.post("/jobs", response_model=JobStatus, status_code=202)
    async def submit_job(
        submission: JobSubmission,
        connection: dict[str, Any] = Depends(current_connection),
    ) -> JobStatus:
        try:
            return app.state.manager.submit(submission, owner_connection_id=connection["id"])
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/jobs", response_model=list[JobStatus])
    async def list_jobs(
        connection: dict[str, Any] = Depends(current_connection),
    ) -> list[JobStatus]:
        return app.state.manager.list_status(owner_connection_id=connection["id"])

    @app.get("/jobs/{job_id}", response_model=JobStatus)
    async def get_job(
        job_id: str,
        connection: dict[str, Any] = Depends(current_connection),
    ) -> JobStatus:
        try:
            return app.state.manager.status(job_id, owner_connection_id=connection["id"])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown job") from exc

    @app.get("/jobs/{job_id}/logs")
    async def get_logs(
        job_id: str,
        connection: dict[str, Any] = Depends(current_connection),
    ) -> dict[str, str]:
        try:
            return app.state.manager.logs(job_id, owner_connection_id=connection["id"])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown job") from exc

    @app.get("/jobs/{job_id}/events")
    def get_events(
        job_id: str,
        connection: dict[str, Any] = Depends(current_connection),
        after: str | None = None,
        last_event_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        owner_connection_id = connection["id"]
        try:
            app.state.manager.status(job_id, owner_connection_id=owner_connection_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown job") from exc

        def stream() -> Iterator[str]:
            cursor = after or last_event_id
            idle_cycles = 0
            while idle_cycles < 120:
                events = app.state.manager.events(
                    job_id,
                    cursor,
                    owner_connection_id=owner_connection_id,
                )
                if events:
                    idle_cycles = 0
                    for event in events:
                        cursor = str(event["id"])
                        yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
                else:
                    idle_cycles += 1
                    status = app.state.manager.status(job_id, owner_connection_id=owner_connection_id)
                    if status.status in {"succeeded", "failed", "cancelled"}:
                        break
                    yield ": keep-alive\n\n"
                time.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.delete("/jobs/{job_id}", response_model=JobStatus)
    async def cancel_job(
        job_id: str,
        connection: dict[str, Any] = Depends(current_connection),
    ) -> JobStatus:
        try:
            return app.state.manager.cancel(job_id, owner_connection_id=connection["id"])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown job") from exc

    return app


def _auth_from_environment(*, in_memory: bool = False) -> AuthService:
    store = InMemoryAuthStore() if in_memory else ValkeyAuthStore(os.getenv("NNM_VALKEY_URL", "valkey://127.0.0.1:6379/0"))
    return AuthService(
        store,
        session_ttl=parse_duration(os.getenv("NNM_SESSION_TTL", "24h")),
        request_ttl=parse_duration(os.getenv("NNM_PAIRING_REQUEST_TTL", "10m")),
        max_pending_per_ip=int(os.getenv("NNM_PAIRING_MAX_PER_IP", "5")),
        max_pending_global=int(os.getenv("NNM_PAIRING_MAX_GLOBAL", "100")),
    )


def _allowed_origins_from_environment() -> list[str]:
    configured = os.getenv("NNM_ALLOWED_ORIGINS")
    if configured is None:
        return DEFAULT_ALLOWED_ORIGINS
    return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]


def _read_admin_token() -> str | None:
    direct = os.getenv("NNM_ADMIN_TOKEN")
    if direct:
        return direct.strip()
    default_path = Path(__file__).resolve().parents[2] / "valkey-data" / "admin.token"
    path = Path(os.getenv("NNM_ADMIN_TOKEN_FILE", str(default_path))).expanduser()
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


def _client_host(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


def _auth_http_error(exc: AuthError, *, not_found_codes: set[str] | None = None) -> HTTPException:
    status_code = 404 if not_found_codes and exc.code in not_found_codes else 401
    return HTTPException(status_code=status_code, detail={"code": exc.code, "message": str(exc)})


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
