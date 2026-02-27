# ── Stage 1: Build the Next.js frontend ──────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY src/frontend/package.json src/frontend/package-lock.json* ./
RUN npm ci 2>/dev/null || npm install

COPY src/frontend/ ./

# Build-time env: WebSocket and API go through the same origin via nginx
ENV NEXT_PUBLIC_WS_URL=""
ENV NEXT_PUBLIC_API_URL=""
ENV NEXT_TELEMETRY_DISABLED=1

# Limit Node memory for constrained HF Space build environment
RUN NODE_OPTIONS=--max-old-space-size=1536 npm run build


# ── Stage 2: Production image ────────────────────────────────────
FROM python:3.10-slim

# System deps: nginx + node (for Next.js standalone server)
RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx \
        curl \
        && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
        && apt-get install -y --no-install-recommends nodejs \
        && rm -rf /var/lib/apt/lists/*

# ── Python backend ───────────────────────────────────────────────
WORKDIR /app/backend
COPY src/backend/requirements.txt ./

# Install Python deps (skip torch/transformers — we use API mode only)
RUN pip install --no-cache-dir \
        fastapi==0.115.0 \
        "uvicorn[standard]==0.30.6" \
        websockets==12.0 \
        pydantic-settings==2.5.2 \
        python-dotenv==1.0.1 \
        openai==1.51.0 \
        httpx==0.27.2 \
        chromadb==0.5.7 \
        sentence-transformers==3.1.1 \
        python-multipart==0.0.10

# Copy backend source (ChromaDB will auto-build from app/data/clinical_guidelines.json on first run)
COPY src/backend/app/ ./app/

# Enable privacy mode for open-web deployment (disables data retention)
# Set to "false" or omit for hospital/production environments with full traceability
ENV PRIVACY_MODE=true

# ── Frontend standalone build ────────────────────────────────────
# Next.js standalone output: self-contained server, no node_modules needed
WORKDIR /app/frontend
COPY --from=frontend-build /app/frontend/.next/standalone ./
COPY --from=frontend-build /app/frontend/.next/static ./.next/static
RUN mkdir -p ./public

# ── Nginx config ─────────────────────────────────────────────────
COPY space/nginx.conf /etc/nginx/nginx.conf

# ── Startup script ───────────────────────────────────────────────
COPY space/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# HF Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

CMD ["/app/start.sh"]
