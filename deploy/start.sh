#!/bin/sh
# Runs the two processes a single HF Spaces container needs: Ollama (as a
# background daemon) and FastAPI (as the foreground process the container
# lives or dies by). A plain shell script rather than supervisord — only
# two processes, so this matches the project's existing preference for a
# hand-rolled solution over a heavier tool when the simple case is this
# small (same reasoning as the MCP server and eval harness).
set -e

ollama serve &

# Wait for Ollama's API to actually be reachable before starting FastAPI —
# EmbeddingService/RerankerService load at import time (see api/deps.py),
# but ChatService's first real generate() call would otherwise race
# Ollama's own startup.
until curl -sf http://127.0.0.1:11434/api/tags >/dev/null; do
  sleep 1
done

exec uvicorn app.main:app --host 0.0.0.0 --port 7860
