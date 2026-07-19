"""Tests for browser pairing and session authentication."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from backend.auth import (
    AuthError,
    AuthService,
    InMemoryAuthStore,
    PairingLimitError,
    parse_duration,
)


class MutableClock:
    """Deterministic UTC clock for expiry tests."""

    def __init__(self) -> None:
        self.current = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, **changes: float) -> None:
        self.current += timedelta(**changes)


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def auth(clock: MutableClock) -> AuthService:
    secrets = iter(["a" * 43, "b" * 43, "c" * 43])
    codes = iter(["123456", "654321", "111222"])
    return AuthService(
        InMemoryAuthStore(),
        session_ttl=timedelta(hours=24),
        request_ttl=timedelta(minutes=10),
        clock=clock,
        secret_factory=lambda: next(secrets),
        code_factory=lambda: next(codes),
    )


def test_pairing_approval_uses_hashed_token_and_default_ttl(
    auth: AuthService,
    clock: MutableClock,
) -> None:
    pairing = auth.create_pairing("Laptop", client_host="192.168.1.5")

    stored = auth.store.get_connection(pairing.connection_id)
    assert stored is not None
    assert pairing.token not in str(stored)
    assert stored["token_hash"] != pairing.token
    assert auth.pairing_status(pairing.request_id, pairing.token)["status"] == "pending"

    session = auth.approve(pairing.request_id)

    assert session["status"] == "active"
    assert session["expires_at"] == (clock() + timedelta(hours=24)).isoformat()
    assert auth.authenticate(pairing.token)["id"] == pairing.connection_id


def test_pairing_approval_accepts_per_request_ttl(auth: AuthService, clock: MutableClock) -> None:
    pairing = auth.create_pairing(None, client_host="192.168.1.5")

    session = auth.approve(pairing.request_id, timedelta(hours=8))

    assert session["expires_at"] == (clock() + timedelta(hours=8)).isoformat()


def test_expired_session_can_renew_with_same_connection_and_jobs_owner(
    auth: AuthService,
    clock: MutableClock,
) -> None:
    pairing = auth.create_pairing("Browser", client_host="192.168.1.8")
    auth.approve(pairing.request_id)
    clock.advance(hours=25)

    with pytest.raises(AuthError, match="expired"):
        auth.authenticate(pairing.token)

    renewal = auth.create_renewal(pairing.token, client_host="192.168.1.8")
    auth.approve(renewal.request_id, timedelta(days=2))

    assert renewal.connection_id == pairing.connection_id
    assert auth.authenticate(pairing.token)["id"] == pairing.connection_id


def test_rejected_and_revoked_connections_never_authenticate(auth: AuthService) -> None:
    rejected = auth.create_pairing("Rejected", client_host="192.168.1.9")
    auth.reject(rejected.request_id)
    assert auth.pairing_status(rejected.request_id, rejected.token)["status"] == "rejected"
    with pytest.raises(AuthError, match="rejected"):
        auth.authenticate(rejected.token)

    approved = auth.create_pairing("Approved", client_host="192.168.1.10")
    auth.approve(approved.request_id)
    auth.revoke(approved.connection_id)
    with pytest.raises(AuthError, match="revoked"):
        auth.authenticate(approved.token)
    with pytest.raises(AuthError, match="revoked"):
        auth.create_renewal(approved.token, client_host="192.168.1.10")


def test_pairing_request_expires_independently(auth: AuthService, clock: MutableClock) -> None:
    pairing = auth.create_pairing("Slow browser", client_host="192.168.1.11")
    clock.advance(minutes=11)

    assert auth.pairing_status(pairing.request_id, pairing.token)["status"] == "expired"
    with pytest.raises(AuthError, match="expired"):
        auth.approve(pairing.request_id)


def test_pairing_limits_pending_requests_per_ip(clock: MutableClock) -> None:
    auth = AuthService(
        InMemoryAuthStore(),
        max_pending_per_ip=1,
        clock=clock,
        secret_factory=lambda: "x" * 43,
        code_factory=lambda: "123456",
    )
    auth.create_pairing(None, client_host="192.168.1.12")

    with pytest.raises(PairingLimitError):
        auth.create_pairing(None, client_host="192.168.1.12")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("30m", 1800), ("24h", 86400), ("7d", 604800)],
)
def test_parse_duration_accepts_documented_units(value: str, expected: int) -> None:
    assert parse_duration(value).total_seconds() == expected


@pytest.mark.parametrize("value", ["", "0h", "2 hours", "-1d", "9999d"])
def test_parse_duration_rejects_ambiguous_or_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_duration(value)
