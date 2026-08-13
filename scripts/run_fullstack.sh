#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Starting ASAG fullstack server at http://127.0.0.1:8000"
if [ -x ".venv/bin/python" ]; then
  .venv/bin/python -m uvicorn web.backend.app.main:app --host 0.0.0.0 --port 8000 --reload
else
  python3 -m uvicorn web.backend.app.main:app --host 0.0.0.0 --port 8000 --reload
fi
