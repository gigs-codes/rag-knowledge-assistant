"""
Cross-encoder reranking — the second stage of "retrieve wide, rerank
narrow." Vector/BM25 search (bi-encoders: query and chunk embedded
independently, then compared) is fast enough to search the whole corpus,
but less precise than a cross-encoder, which reads the query and a
candidate chunk TOGETHER through one transformer pass and scores their
actual relevance jointly. Cross-encoders don't scale to searching a whole
corpus (one forward pass per candidate), so the standard pattern — used
here — is: cheap bi-encoder search to pull a wide candidate set, then this
slower-but-more-accurate model to reorder just those few candidates down
to the final top_k.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 — free, local, small (~90MB),
trained specifically for query-passage relevance ranking (MS MARCO), and
uses the same sentence-transformers library already a dependency
(EmbeddingService), so no new ML framework enters the stack.
"""
from sentence_transformers import CrossEncoder

from app.core.logging import get_logger

logger = get_logger(__name__)


class RerankerService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info("Loading reranker model: %s", model_name)
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, hit["text"]) for hit in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [hit for hit, _score in ranked[:top_k]]
