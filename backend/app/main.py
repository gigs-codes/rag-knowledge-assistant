"""
Application entrypoint.
main.py's only job is to assemble the app
(middleware, routers) and start it. Anything more than that belongs in
services/routes, not here — this file should be readable top-to-bottom
in ten seconds to answer "what does this API expose?"
"""
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import agent, auth, documents, feedback, query
from app.core.config import settings
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger("app.request")

app = FastAPI(
    title=settings.app_name,
    description="Upload company documents and ask questions in natural language, "
    "answered with citations from a local RAG pipeline.",
    version="0.1.0",
)

# Dev-only CORS: the React dev server (Vite, port 5173) runs on a different
# origin than the API (port 8000), so the browser blocks requests without
# this. Tightened to a real allowlist before any production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Deliberately hand-rolled (one uuid4 + one log line) rather than
# OpenTelemetry — same "explainable, dependency-light over a heavy
# library for a small local project" philosophy documented in
# mcp_server.py and eval/judge.py. Gives every request a correlation ID
# (echoed back so a client/log-search can tie a response to its request)
# and a structured method/path/status/latency log line, without a new
# dependency or an external collector to run.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s -> %d (%dms) [%s]",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        request_id,
    )
    return response


app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(agent.router)
app.include_router(feedback.router)


@app.get("/health")
def health_check():
    return {"status": "ok", "app": settings.app_name}
