"""Tests for machine-local backend administration helpers."""

from __future__ import annotations

import stat

from backend.admin_cli import initialize_admin_token


def test_initialize_admin_token_creates_private_file_and_preserves_existing(tmp_path) -> None:
    path = tmp_path / "admin.token"

    first = initialize_admin_token(path)
    second = initialize_admin_token(path)

    assert first == second
    assert len(first) >= 43
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
