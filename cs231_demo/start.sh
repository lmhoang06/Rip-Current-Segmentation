#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

cd "$ROOT_DIR"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  python -m pip install -r cs231_demo/requirements-demo.txt
fi

exec python -m uvicorn cs231_demo.app:app --host "$HOST" --port "$PORT"

