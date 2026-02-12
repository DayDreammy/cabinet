#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="8002"
HOST="0.0.0.0"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/uvicorn_8002.log"
PID_FILE="${LOG_DIR}/uvicorn_8002.pid"

PROXY=""
UNSET_PROXY="0"
TAIL="1"
TOKEN="cabinet-dev-token"
WAIT_SEC="1.2"

usage() {
  cat <<EOF
Usage:
  scripts/start_backend_8002.sh [--proxy http://127.0.0.1:7890] [--token <token>] [--unset-proxy] [--no-tail]

Notes:
  - Proxy only affects the backend process (and spawned codex subprocesses), not the frontend.
  - If --token is omitted, a fixed token is used: ${TOKEN}
  - Logs: ${LOG_FILE}
  - PID:  ${PID_FILE}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --proxy)
      PROXY="${2:-}"
      shift 2
      ;;
    --token)
      TOKEN="${2:-}"
      shift 2
      ;;
    --unset-proxy)
      UNSET_PROXY="1"
      shift 1
      ;;
    --no-tail)
      TAIL="0"
      shift 1
      ;;
    --wait-sec)
      WAIT_SEC="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${LOG_DIR}"

export CABINET_API_TOKEN="${TOKEN}"

if ! python3 -c "import uvicorn, fastapi" >/dev/null 2>&1; then
  echo "ERROR: python3 is missing required deps (uvicorn/fastapi). Use the right Python env." >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Stopping existing uvicorn (pid=${old_pid})"
    kill "${old_pid}" 2>/dev/null || true
    sleep 0.5
    if kill -0 "${old_pid}" 2>/dev/null; then
      echo "Still running, sending SIGKILL (pid=${old_pid})"
      kill -9 "${old_pid}" 2>/dev/null || true
    fi
  fi
fi

if [[ "${UNSET_PROXY}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
elif [[ -n "${PROXY}" ]]; then
  export HTTP_PROXY="${PROXY}"
  export HTTPS_PROXY="${PROXY}"
  export ALL_PROXY="${PROXY}"
  export http_proxy="${PROXY}"
  export https_proxy="${PROXY}"
  export all_proxy="${PROXY}"
  export NO_PROXY="127.0.0.1,localhost"
  export no_proxy="127.0.0.1,localhost"
fi

echo "Starting backend: http://${HOST}:${PORT}/"
echo "Logging to: ${LOG_FILE}"
echo "CABINET_API_TOKEN: ${CABINET_API_TOKEN}"
echo "Example call:"
echo "  curl -sS -X POST http://127.0.0.1:${PORT}/api/deep_research -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\":\"...\",\"context\":\"\",\"token\":\"${CABINET_API_TOKEN}\"}' | head"

nohup python3 -m uvicorn main:app --host "${HOST}" --port "${PORT}" > "${LOG_FILE}" 2>&1 & echo $! > "${PID_FILE}"

new_pid="$(cat "${PID_FILE}")"
echo "PID: ${new_pid}"

sleep "${WAIT_SEC}"
if command -v ss >/dev/null 2>&1; then
  if ! ss -ltnp 2>/dev/null | grep -qE ":${PORT}\\b"; then
    echo "ERROR: backend failed to listen on ${HOST}:${PORT} (pid=${new_pid}). Recent log:" >&2
    tail -n 120 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  ss -ltnp 2>/dev/null | grep -E ":${PORT}\\b" || true
fi

if [[ "${TAIL}" == "1" ]]; then
  echo ""
  echo "Tailing log (Ctrl-C to stop tail):"
  tail -f "${LOG_FILE}"
fi
