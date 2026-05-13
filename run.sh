#!/usr/bin/env bash
# run.sh
# -------
# Convenience script to launch the FastAPI development server.
# Usage:  bash run.sh
#         bash run.sh --port 8080

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-true}"   # set RELOAD=false in production

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AI Recommendation System — FastAPI"
echo "  http://${HOST}:${PORT}"
echo "  Docs → http://${HOST}:${PORT}/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

uvicorn app.main:app \
  --host "$HOST" \
  --port "$PORT" \
  $( [ "$RELOAD" = "true" ] && echo "--reload" )
