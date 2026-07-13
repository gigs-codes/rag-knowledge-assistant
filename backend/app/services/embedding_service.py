"""
Embedding generation — turns text into vectors for semantic search.

Why a dedicated service instead of calling sentence-transformers inline
wherever needed: the model is loaded ONCE at process start (loading it is
slow — a few seconds and ~130MB) and reused for every request. If this
lived inside a route handler, every request would reload the model.

Model choice: BAAI/bge-small-en-v1.5 — runs on CPU, no API key/cost,
strong quality-for-size on retrieval benchmarks (MTEB), used widely in
production RAG systems when teams want to avoid embedding-API cost/latency.
"""
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = settings.embedding_model):
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunks (for ingestion)."""
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query. bge models recommend a query instruction
        prefix for better retrieval quality vs. raw passage embeddings."""
        instructed = f"Represent this sentence for searching relevant passages: {text}"
        return self._model.encode(instructed, normalize_embeddings=True).tolist()
