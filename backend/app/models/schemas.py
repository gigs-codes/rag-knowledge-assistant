"""
Pydantic schemas: the data contracts for the API.

Why separate from services/vectorstore code: these describe the *shape of
data crossing the HTTP boundary*. Internal services can use plain dataclasses
or dicts; only what the client sends/receives needs this validation layer.
Keeping them in one file makes the API surface easy to review in one glance
(useful when someone asks "what does this API actually expose?").
"""
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    filename: str
    num_chunks: int
    uploaded_at: datetime


class UploadResponse(BaseModel):
    document: DocumentOut
    message: str


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_id: str | None = Field(
        default=None,
        description="Optional: restrict retrieval to a single document.",
    )


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: int
