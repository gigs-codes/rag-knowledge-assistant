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


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    thread_id: str = Field(
        default="default",
        description="Conversation identifier — reused across calls to give the agent memory "
        "of prior turns (see agent/graph.py's MemorySaver checkpointer).",
    )


class AgentQueryResponse(BaseModel):
    answer: str
    trace: list[str] = Field(description="Thought/Action/Observation steps the agent took.")
    latency_ms: int


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=200)


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    username: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut
