#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
FRONTEND_PORT="${NNM_FRONTEND_PORT:-5174}"
FRONTEND_URL="${NNM_FRONTEND_URL:-http://127.0.0.1:${FRONTEND_PORT}}"
CDP_PORT="${NNM_CDP_PORT:-9223}"
CDP_URL="http://127.0.0.1:${CDP_PORT}"

status() {
  echo "Frontend: ${FRONTEND_URL}"
  curl --silent --show-error --max-time 2 "${FRONTEND_URL}/" >/dev/null \
    && echo "  reachable" || echo "  unavailable"
  echo "Chromium DevTools: ${CDP_URL}"
  curl --silent --show-error --max-time 2 "${CDP_URL}/json/version" >/dev/null \
    && echo "  reachable" || echo "  unavailable"
  echo "Listening ports:"
  lsof -nP -iTCP:"${FRONTEND_PORT}" -iTCP:"${CDP_PORT}" -iTCP:9339 \
    -sTCP:LISTEN || true
}

case "${1:-status}" in
  status)
    status
    ;;
  frontend)
    cd "${ROOT_DIR}"
    exec pnpm --dir front-end dev --host 127.0.0.1 --port "${FRONTEND_PORT}"
    ;;
  browser)
    if curl --silent --max-time 2 "${CDP_URL}/json/version" >/dev/null; then
      echo "Chromium DevTools is already available at ${CDP_URL}"
      exit 0
    fi
    if [[ -n "${NNM_BROWSER_COMMAND:-}" ]]; then
      exec "${NNM_BROWSER_COMMAND}" \
        --user-data-dir=/tmp/nnmodelling-codex-chromium \
        --remote-debugging-port="${CDP_PORT}" \
        --no-first-run --no-default-browser-check \
        --new-window "${FRONTEND_URL}"
    fi
    exec flatpak run org.chromium.Chromium \
      --user-data-dir=/tmp/nnmodelling-codex-chromium \
      --remote-debugging-port="${CDP_PORT}" \
      --no-first-run --no-default-browser-check \
      --disable-session-crashed-bubble \
      --new-window "${FRONTEND_URL}"
    ;;
  mcp)
    cd "${ROOT_DIR}"
    pnpm --dir mcp-server build
    exec pnpm --dir mcp-server start
    ;;
  *)
    echo "Usage: $0 {status|frontend|browser|mcp}" >&2
    exit 2
    ;;
esac
