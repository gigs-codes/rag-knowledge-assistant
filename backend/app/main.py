"""
Application entrypoint.
main.py's only job is to assemble the app
(middleware, routers) and start it. Anything more than that belongs in
services/routes, not here — this file should be readable top-to-bottom
in ten seconds to answer "what does this API expose?"
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import agent, documents, query
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()

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

app.include_router(documents.router)
app.include_router(query.router)
app.include_router(agent.router)


@app.get("/health")
def health_check():
    return {"status": "ok", "app": settings.app_name}
