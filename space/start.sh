#!/bin/bash
set -e

echo "=== CDS Agent — Starting services ==="

# ── 1. Start FastAPI backend ────────────────────────────────────
echo "[1/3] Starting FastAPI backend on :8002 ..."
cd /app/backend
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8002 \
    --workers 1 \
    --timeout-keep-alive 300 \
    &

# ── 2. Start Next.js frontend (standalone mode) ────────────────
echo "[2/3] Starting Next.js frontend on :3000 ..."
cd /app/frontend
PORT=3000 HOSTNAME=0.0.0.0 node server.js &

# ── 3. Start nginx reverse proxy ───────────────────────────────
echo "[3/3] Starting nginx on :7860 ..."
sleep 3  # Give backend/frontend a moment to bind
nginx -g 'daemon off;'
