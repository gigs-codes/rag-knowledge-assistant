"""
Retrieval service: question -> ranked relevant chunks.

Kept deliberately thin — it composes EmbeddingService (embed the query)
and ChromaStore (nearest-neighbor search) without knowing how either
works internally. This is the seam where "hybrid retrieval" (combining
this semantic search with keyword/BM25 search) would plug in later
without changing chat_service.py's contract.
"""
from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore


class RetrievalService:
    def __init__(self, embedding_service: EmbeddingService, vector_store: ChromaStore):
        self._embeddings = embedding_service
        self._store = vector_store

    def retrieve(
        self, question: str, top_k: int = settings.top_k, document_id: str | None = None
    ) -> list[dict]:
        query_vector = self._embeddings.embed_query(question)
        return self._store.query(query_vector, top_k=top_k, document_id=document_id)
