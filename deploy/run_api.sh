#!/usr/bin/env bash
# Start the FastAPI application with Uvicorn on the public port (honours proxy headers and WebSocket pings).
set -euo pipefail
cd "${APP_DIR:-/app}"
exec python -m neural_forge.main
