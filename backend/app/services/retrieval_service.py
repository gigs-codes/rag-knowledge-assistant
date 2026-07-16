"""
Retrieval service: question -> ranked relevant chunks.

Kept deliberately thin — it composes EmbeddingService (embed the query)
and ChromaStore (nearest-neighbor search) without knowing how either
works internally. This is the seam where "hybrid retrieval" (combining
this semantic search with keyword/BM25 search) would plug in later
without changing chat_service.py's contract.

Why the relevance-score filter lives HERE and not in ChromaStore: Chroma
is a generic nearest-neighbor adapter — it has no concept of "relevant
enough," it just returns the closest k vectors, always, even if the
closest one is still a poor match (this was a known, explicitly-documented
gap in the original build: a single-chunk store will return that chunk
for literally any query, however unrelated). "What counts as relevant" is
a business-logic decision, which belongs in this service, not the
adapter — ChromaStore should stay swappable for FAISS/pgvector without
carrying retrieval policy along with it.
"""
from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore


class RetrievalService:
    def __init__(self, embedding_service: EmbeddingService, vector_store: ChromaStore):
        self._embeddings = embedding_service
        self._store = vector_store

    def retrieve(
        self,
        question: str,
        top_k: int = settings.top_k,
        document_id: str | None = None,
        min_score: float = settings.min_relevance_score,
    ) -> list[dict]:
        query_vector = self._embeddings.embed_query(question)
        hits = self._store.query(query_vector, top_k=top_k, document_id=document_id)
        return [hit for hit in hits if hit["score"] >= min_score]
