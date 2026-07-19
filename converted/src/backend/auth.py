"""Browser pairing, persistent sessions, and authentication stores."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol


Clock = Callable[[], datetime]
Duration = timedelta
MAX_SESSION_TTL = timedelta(days=365)
DURATION_PATTERN = re.compile(r"([1-9][0-9]*)([mhd])")


def utc_datetime() -> datetime:
    """Return the current timezone-aware UTC datetime."""

    return datetime.now(UTC)


def parse_duration(value: str) -> Duration:
    """Parse a positive duration expressed as minutes, hours, or days."""

    match = DURATION_PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError("duration must use a positive value followed by m, h, or d")
    amount = int(match.group(1))
    unit = match.group(2)
    duration = timedelta(**{"m": {"minutes": amount}, "h": {"hours": amount}, "d": {"days": amount}}[unit])
    if duration > MAX_SESSION_TTL:
        raise ValueError("duration cannot exceed 365 days")
    return duration


class AuthError(ValueError):
    """Authentication or pairing lifecycle error with a stable API code."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code.replace("_", " "))


class PairingLimitError(AuthError):
    """Raised when a client creates too many pending pairing requests."""

    def __init__(self) -> None:
        super().__init__("pairing_rate_limited", "too many pending pairing requests")


@dataclass(frozen=True)
class PairingGrant:
    """One-time response returned when a pairing request is created."""

    request_id: str
    connection_id: str
    token: str
    verification_code: str
    expires_at: str


class AuthStore(Protocol):
    """Persistence operations required by :class:`AuthService`."""

    def save_connection(self, connection_id: str, data: dict[str, Any]) -> None: ...

    def get_connection(self, connection_id: str) -> dict[str, Any] | None: ...

    def list_connections(self) -> list[dict[str, Any]]: ...

    def save_request(self, request_id: str, data: dict[str, Any]) -> None: ...

    def get_request(self, request_id: str) -> dict[str, Any] | None: ...

    def list_requests(self) -> list[dict[str, Any]]: ...

    def append_audit(self, event: dict[str, Any]) -> None: ...


class InMemoryAuthStore:
    """Thread-safe authentication store used by unit and API tests."""

    def __init__(self) -> None:
        self.connections: dict[str, dict[str, Any]] = {}
        self.requests: dict[str, dict[str, Any]] = {}
        self.audit: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    @staticmethod
    def _copy(data: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(data))

    def save_connection(self, connection_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            self.connections[connection_id] = self._copy(data)

    def get_connection(self, connection_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self.connections.get(connection_id)
            return self._copy(data) if data is not None else None

    def list_connections(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._copy(item) for item in self.connections.values()]

    def save_request(self, request_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            self.requests[request_id] = self._copy(data)

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self.requests.get(request_id)
            return self._copy(data) if data is not None else None

    def list_requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._copy(item) for item in self.requests.values()]

    def append_audit(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.audit.append(self._copy(event))


class ValkeyAuthStore:
    """Valkey persistence for pairing requests, sessions, and audit events."""

    def __init__(self, url: str = "valkey://127.0.0.1:6379/0") -> None:
        try:
            import valkey
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError("Install the 'valkey' package to use ValkeyAuthStore") from exc
        self.client = valkey.from_url(url, decode_responses=True)

    def save_connection(self, connection_id: str, data: dict[str, Any]) -> None:
        self.client.set(f"auth:connection:{connection_id}", json.dumps(data))

    def get_connection(self, connection_id: str) -> dict[str, Any] | None:
        value = self.client.get(f"auth:connection:{connection_id}")
        return json.loads(value) if value else None

    def list_connections(self) -> list[dict[str, Any]]:
        return self._list_json("auth:connection:*")

    def save_request(self, request_id: str, data: dict[str, Any]) -> None:
        self.client.set(f"auth:request:{request_id}", json.dumps(data))

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        value = self.client.get(f"auth:request:{request_id}")
        return json.loads(value) if value else None

    def list_requests(self) -> list[dict[str, Any]]:
        return self._list_json("auth:request:*")

    def append_audit(self, event: dict[str, Any]) -> None:
        self.client.xadd("auth:audit", {"event": json.dumps(event)}, maxlen=10_000)

    def _list_json(self, pattern: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for key in self.client.scan_iter(match=pattern):
            value = self.client.get(key)
            if value:
                result.append(json.loads(value))
        return result


class AuthService:
    """Manage browser pairing requests and bearer-token sessions."""

    def __init__(
        self,
        store: AuthStore,
        *,
        session_ttl: Duration = timedelta(hours=24),
        request_ttl: Duration = timedelta(minutes=10),
        max_pending_per_ip: int = 5,
        max_pending_global: int = 100,
        clock: Clock = utc_datetime,
        secret_factory: Callable[[], str] | None = None,
        code_factory: Callable[[], str] | None = None,
    ) -> None:
        self.store = store
        self.session_ttl = session_ttl
        self.request_ttl = request_ttl
        self.max_pending_per_ip = max_pending_per_ip
        self.max_pending_global = max_pending_global
        self.clock = clock
        self.secret_factory = secret_factory or (lambda: secrets.token_urlsafe(32))
        self.code_factory = code_factory or (lambda: f"{secrets.randbelow(1_000_000):06d}")

    def create_pairing(
        self,
        device_name: str | None,
        *,
        client_host: str,
        origin: str | None = None,
        user_agent: str | None = None,
    ) -> PairingGrant:
        """Create a pending connection and return its bearer token once."""

        self._check_pending_limits(client_host)
        now = self.clock()
        connection_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        token = f"nnm_v1.{connection_id}.{self.secret_factory()}"
        expires_at = now + self.request_ttl
        connection = {
            "id": connection_id,
            "token_hash": self._token_hash(token),
            "device_name": self._normalize_device_name(device_name),
            "status": "pending",
            "created_at": now.isoformat(),
            "approved_at": None,
            "expires_at": None,
            "last_seen_at": None,
            "revoked_at": None,
            "last_client_host": client_host,
            "last_origin": origin,
            "last_user_agent": user_agent,
        }
        request = self._request_record(
            request_id,
            connection,
            "new",
            self.code_factory(),
            now,
            expires_at,
            client_host,
            origin,
            user_agent,
        )
        self.store.save_connection(connection_id, connection)
        self.store.save_request(request_id, request)
        self._audit("pairing_requested", connection_id, request_id=request_id)
        return PairingGrant(
            request_id=request_id,
            connection_id=connection_id,
            token=token,
            verification_code=request["verification_code"],
            expires_at=expires_at.isoformat(),
        )

    def create_renewal(
        self,
        token: str,
        *,
        client_host: str,
        origin: str | None = None,
        user_agent: str | None = None,
    ) -> PairingGrant:
        """Create a pending renewal for an expired, non-revoked connection."""

        connection = self._connection_for_token(token)
        connection = self._refresh_connection_state(connection)
        if connection["status"] == "revoked":
            raise AuthError("session_revoked")
        if connection["status"] != "expired":
            raise AuthError("session_not_expired", "only an expired session can be renewed")
        self._check_pending_limits(client_host)
        now = self.clock()
        request_id = str(uuid.uuid4())
        expires_at = now + self.request_ttl
        request = self._request_record(
            request_id,
            connection,
            "renewal",
            self.code_factory(),
            now,
            expires_at,
            client_host,
            origin,
            user_agent,
        )
        connection.update(
            status="pending",
            last_client_host=client_host,
            last_origin=origin,
            last_user_agent=user_agent,
        )
        self.store.save_connection(connection["id"], connection)
        self.store.save_request(request_id, request)
        self._audit("renewal_requested", connection["id"], request_id=request_id)
        return PairingGrant(
            request_id=request_id,
            connection_id=connection["id"],
            token=token,
            verification_code=request["verification_code"],
            expires_at=expires_at.isoformat(),
        )

    def pairing_status(self, request_id: str, token: str) -> dict[str, Any]:
        """Return the status of one pairing request owned by the token."""

        connection = self._connection_for_token(token)
        request = self._request(request_id)
        if request["connection_id"] != connection["id"]:
            raise AuthError("pairing_not_found")
        request = self._refresh_request_state(request)
        return {
            "request_id": request["id"],
            "connection_id": request["connection_id"],
            "status": request["status"],
            "verification_code": request["verification_code"],
            "expires_at": request["expires_at"],
            "session_expires_at": connection.get("expires_at") if request["status"] == "approved" else None,
        }

    def approve(self, request_id: str, ttl: Duration | None = None) -> dict[str, Any]:
        """Approve a pending request using the default or supplied session TTL."""

        request = self._refresh_request_state(self._request(request_id))
        if request["status"] != "pending":
            raise AuthError(f"pairing_{request['status']}")
        duration = ttl or self.session_ttl
        if duration <= timedelta(0) or duration > MAX_SESSION_TTL:
            raise ValueError("session TTL must be positive and no longer than 365 days")
        now = self.clock()
        connection = self._connection(request["connection_id"])
        connection.update(
            status="active",
            approved_at=now.isoformat(),
            expires_at=(now + duration).isoformat(),
            revoked_at=None,
        )
        request.update(status="approved", decided_at=now.isoformat())
        self.store.save_connection(connection["id"], connection)
        self.store.save_request(request_id, request)
        self._audit("pairing_approved", connection["id"], request_id=request_id, ttl_seconds=int(duration.total_seconds()))
        return self.public_session(connection)

    def reject(self, request_id: str) -> dict[str, Any]:
        """Reject a pending pairing or renewal request."""

        request = self._refresh_request_state(self._request(request_id))
        if request["status"] != "pending":
            raise AuthError(f"pairing_{request['status']}")
        now = self.clock().isoformat()
        request.update(status="rejected", decided_at=now)
        connection = self._connection(request["connection_id"])
        connection.update(status="rejected" if request["kind"] == "new" else "expired")
        self.store.save_request(request_id, request)
        self.store.save_connection(connection["id"], connection)
        self._audit("pairing_rejected", connection["id"], request_id=request_id)
        return request

    def authenticate(self, token: str) -> dict[str, Any]:
        """Authenticate an active bearer token and update its last-seen time."""

        connection = self._refresh_connection_state(self._connection_for_token(token))
        if connection["status"] != "active":
            raise AuthError(f"session_{connection['status']}")
        connection["last_seen_at"] = self.clock().isoformat()
        self.store.save_connection(connection["id"], connection)
        return connection

    def revoke(self, connection_id: str) -> dict[str, Any]:
        """Revoke a connection immediately."""

        connection = self._connection(connection_id)
        now = self.clock().isoformat()
        connection.update(status="revoked", revoked_at=now)
        self.store.save_connection(connection_id, connection)
        self._audit("session_revoked", connection_id)
        return self.public_session(connection)

    def list_requests(self, *, pending_only: bool = False) -> list[dict[str, Any]]:
        """List pairing requests for administrative commands."""

        requests = [self._refresh_request_state(item) for item in self.store.list_requests()]
        if pending_only:
            requests = [item for item in requests if item["status"] == "pending"]
        return sorted(requests, key=lambda item: item["created_at"], reverse=True)

    def list_sessions(self) -> list[dict[str, Any]]:
        """List public session metadata for administrative commands."""

        connections = [self._refresh_connection_state(item) for item in self.store.list_connections()]
        return [self.public_session(item) for item in sorted(connections, key=lambda item: item["created_at"], reverse=True)]

    @staticmethod
    def public_session(connection: dict[str, Any]) -> dict[str, Any]:
        """Remove token material from a connection record."""

        return {key: value for key, value in connection.items() if key != "token_hash"}

    def _connection_for_token(self, token: str) -> dict[str, Any]:
        parts = token.split(".", 2)
        if len(parts) != 3 or parts[0] != "nnm_v1":
            raise AuthError("invalid_token")
        connection = self.store.get_connection(parts[1])
        if connection is None or not hmac.compare_digest(connection["token_hash"], self._token_hash(token)):
            raise AuthError("invalid_token")
        return connection

    def _connection(self, connection_id: str) -> dict[str, Any]:
        connection = self.store.get_connection(connection_id)
        if connection is None:
            raise AuthError("session_not_found")
        return connection

    def _request(self, request_id: str) -> dict[str, Any]:
        request = self.store.get_request(request_id)
        if request is None:
            raise AuthError("pairing_not_found")
        return request

    def _refresh_connection_state(self, connection: dict[str, Any]) -> dict[str, Any]:
        expires_at = connection.get("expires_at")
        if connection["status"] == "active" and expires_at and datetime.fromisoformat(expires_at) <= self.clock():
            connection["status"] = "expired"
            self.store.save_connection(connection["id"], connection)
        return connection

    def _refresh_request_state(self, request: dict[str, Any]) -> dict[str, Any]:
        if request["status"] == "pending" and datetime.fromisoformat(request["expires_at"]) <= self.clock():
            request["status"] = "expired"
            self.store.save_request(request["id"], request)
            connection = self._connection(request["connection_id"])
            connection["status"] = "expired" if request["kind"] == "renewal" else "rejected"
            self.store.save_connection(connection["id"], connection)
        return request

    def _check_pending_limits(self, client_host: str) -> None:
        pending = self.list_requests(pending_only=True)
        per_ip = sum(item["client_host"] == client_host for item in pending)
        if per_ip >= self.max_pending_per_ip or len(pending) >= self.max_pending_global:
            raise PairingLimitError()

    @staticmethod
    def _normalize_device_name(device_name: str | None) -> str | None:
        if device_name is None:
            return None
        normalized = " ".join(device_name.strip().split())
        if not normalized:
            return None
        if len(normalized) > 80:
            raise ValueError("device name cannot exceed 80 characters")
        return normalized

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _request_record(
        request_id: str,
        connection: dict[str, Any],
        kind: str,
        code: str,
        now: datetime,
        expires_at: datetime,
        client_host: str,
        origin: str | None,
        user_agent: str | None,
    ) -> dict[str, Any]:
        return {
            "id": request_id,
            "connection_id": connection["id"],
            "kind": kind,
            "status": "pending",
            "verification_code": code,
            "device_name": connection["device_name"],
            "client_host": client_host,
            "origin": origin,
            "user_agent": user_agent,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "decided_at": None,
        }

    def _audit(self, event_type: str, connection_id: str, **details: Any) -> None:
        self.store.append_audit(
            {
                "type": event_type,
                "connection_id": connection_id,
                "at": self.clock().isoformat(),
                **details,
            }
        )
