FROM python:3.11-slim

WORKDIR /app

# build-essential is needed to build a couple of scientific-Python wheels
# (e.g. for sentence-transformers' dependencies) on some platforms; slim
# base images don't include a compiler by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# EKA_BASE_DIR anchors where uploads/chroma/documents.json live (see
# core/config.py's `base_dir`, normally derived from this file's own path
# four directories up — i.e. the repo root). That math assumes the
# `backend/` wrapper directory from the source repo still exists, which it
# doesn't once only `app/` is copied into the image. Pinning it explicitly
# here is simpler than preserving an identical directory depth just to
# keep a path-derivation trick working across two different layouts.
ENV EKA_BASE_DIR=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
