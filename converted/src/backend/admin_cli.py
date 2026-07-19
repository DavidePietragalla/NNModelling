"""Machine-local command line client for backend administration."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Any

import httpx


def default_admin_token_path() -> Path:
    """Return the default untracked administrator token path."""

    return Path(__file__).resolve().parents[2] / "valkey-data" / "admin.token"


def configured_admin_token_path() -> Path:
    """Return the administrator token path selected by the environment."""

    return Path(os.getenv("NNM_ADMIN_TOKEN_FILE", str(default_admin_token_path()))).expanduser()


def initialize_admin_token(path: Path) -> str:
    """Create a private administrator capability or preserve the existing one."""

    if path.exists():
        path.chmod(0o600)
        token = path.read_text(encoding="utf-8").strip()
        if not token:
            raise RuntimeError(f"administrator token file is empty: {path}")
        return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(f"{token}\n")
    return token


def read_admin_token(path: Path) -> str:
    """Read a private administrator capability after checking permissions."""

    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise RuntimeError(f"administrator token must have mode 0600: {path}")
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"administrator token file is empty: {path}")
    return token


class AdminClient:
    """Small HTTP client used exclusively by the backend justfile."""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"x-nnm-admin-token": token}

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        """Call an administrator endpoint and return its JSON document."""

        response = httpx.request(
            method,
            f"{self.base_url}{path}",
            headers=self.headers,
            json=body,
            params=params,
            timeout=10,
        )
        if response.is_error:
            raise RuntimeError(f"backend returned {response.status_code}: {response.text}")
        return response.json()


def build_parser() -> argparse.ArgumentParser:
    """Build the administration command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("pairing-pending")
    approve = commands.add_parser("pairing-approve")
    approve.add_argument("request_id")
    approve.add_argument("--ttl")
    reject = commands.add_parser("pairing-reject")
    reject.add_argument("request_id")
    commands.add_parser("sessions")
    revoke = commands.add_parser("session-revoke")
    revoke.add_argument("connection_id")
    commands.add_parser("jobs")
    job = commands.add_parser("job")
    job.add_argument("job_id")
    logs = commands.add_parser("job-logs")
    logs.add_argument("job_id")
    events = commands.add_parser("job-events")
    events.add_argument("job_id")
    events.add_argument("--after")
    cancel = commands.add_parser("job-cancel")
    cancel.add_argument("job_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one administration command and print JSON output."""

    args = build_parser().parse_args(argv)
    token_path = configured_admin_token_path()
    if args.command == "init":
        initialize_admin_token(token_path)
        print(f"Administrator token ready at {token_path}")
        return 0
    token = read_admin_token(token_path)
    base_url = os.getenv("NNM_ADMIN_URL", "http://127.0.0.1:8000")
    client = AdminClient(base_url, token)
    routes: dict[str, tuple[str, str]] = {
        "pairing-pending": ("GET", "/admin/pairing/requests"),
        "sessions": ("GET", "/admin/sessions"),
        "jobs": ("GET", "/admin/jobs"),
    }
    if args.command in routes:
        method, path = routes[args.command]
        result = client.request(method, path)
    elif args.command == "pairing-approve":
        result = client.request(
            "POST",
            f"/admin/pairing/requests/{args.request_id}/approve",
            body={"ttl": args.ttl},
        )
    elif args.command == "pairing-reject":
        result = client.request("POST", f"/admin/pairing/requests/{args.request_id}/reject")
    elif args.command == "session-revoke":
        result = client.request("DELETE", f"/admin/sessions/{args.connection_id}")
    elif args.command == "job":
        result = client.request("GET", f"/admin/jobs/{args.job_id}")
    elif args.command == "job-logs":
        result = client.request("GET", f"/admin/jobs/{args.job_id}/logs")
    elif args.command == "job-events":
        params = {"after": args.after} if args.after else None
        result = client.request("GET", f"/admin/jobs/{args.job_id}/events", params=params)
    elif args.command == "job-cancel":
        result = client.request("DELETE", f"/admin/jobs/{args.job_id}")
    else:  # pragma: no cover - argparse guarantees a known command
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
