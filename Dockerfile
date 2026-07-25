# Single-container image for Hugging Face Spaces (Docker SDK). Distinct
# from backend/Dockerfile + frontend/Dockerfile, which stay as they are for
# the two-container docker-compose setup (compose builds from those
# subdirectories directly and never looks at this file). HF Spaces runs
# exactly one container on one port (7860 by default), so this combines
# frontend + backend + Ollama into one image instead of shipping
# nginx/supervisord to replicate the two-container split inside it — see
# deploy/start.sh for how the two processes (Ollama, FastAPI) are run, and
# app/main.py for how FastAPI serves the built frontend directly.

FROM node:20-alpine AS frontend-build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

# build-essential: same reasoning as backend/Dockerfile — needed to build
# a couple of sentence-transformers' dependencies' wheels on some platforms.
# curl: installs Ollama and is used by deploy/start.sh's readiness check.
# zstd: Ollama's own install.sh requires it to extract its release archive
# — not present by default on this slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl zstd \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV OLLAMA_HOST=0.0.0.0:11434

# Bake the LLM into the image so the Space's first boot doesn't need a
# ~2.3GB download (and doesn't risk a health-check timeout while it
# happens): start Ollama just long enough to pull the model into this
# build layer, then stop it — deploy/start.sh starts the real long-running
# instance at container start. Deliberately placed BEFORE `COPY backend/app`
# below (and needs nothing from it) — an application code change shouldn't
# invalidate Docker's cache for this multi-minute, multi-GB step, only a
# requirements.txt or Dockerfile change should.
RUN ollama serve & \
    OLLAMA_PID=$! && \
    until curl -sf http://127.0.0.1:11434/api/tags >/dev/null; do sleep 1; done && \
    ollama pull phi3:mini && \
    kill $OLLAMA_PID

# Same idea for the embedding + reranker models (~220MB combined, small,
# but this is cheap insurance against a slow or flaky first request pulling
# them from the HF Hub at runtime instead of build time) — also placed
# before the app code copy for the same cache-locality reason.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY backend/app ./app

# Same reasoning as backend/Dockerfile's identical line: core/config.py
# derives data/uploads/chroma paths from this file's own location, which
# assumes a directory depth that only holds if EKA_BASE_DIR is pinned
# explicitly once just `app/` is copied into the image.
ENV EKA_BASE_DIR=/app

# Where app/main.py's StaticFiles mount looks for the built frontend.
COPY --from=frontend-build /app/dist ./static

COPY deploy/start.sh ./start.sh
RUN chmod +x ./start.sh

# HF Spaces' default expected port for Docker SDK Spaces.
EXPOSE 7860

CMD ["./start.sh"]
