"""
Retrieval service: question -> ranked relevant chunks.

Hybrid search: composes vector search (ChromaStore, via EmbeddingService)
with keyword search (BM25Index) — this is the seam the original module
docstring flagged for exactly this extension. The two are fused with
Reciprocal Rank Fusion (RRF) rather than by combining raw scores, because
cosine similarity (0-1, dense) and BM25 term-frequency scores (unbounded,
sparse) aren't on comparable scales; RRF only needs each list's RANK
order, which is comparable regardless of how the underlying score was
computed. A wide candidate set (candidate_k) is fused, then narrowed to
top_k by RerankerService's cross-encoder — see reranker_service.py for
why that two-stage "retrieve wide, rerank narrow" shape is used.

Why the relevance-score filter lives HERE and not in ChromaStore/BM25Index:
both adapters are generic search primitives with no concept of "relevant
enough" — that's a business-logic decision that belongs in this service.
The filter only applies to hits that carry a vector cosine score; a
BM25-only hit (found by keyword match but never returned by the vector
search) has no comparable score to gate on, and BM25Index already drops
zero-score (no term overlap) hits itself, so keeping it is the correct
default rather than a silent gap.
"""
from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import RerankerService
from app.vectorstore.bm25_index import BM25Index
from app.vectorstore.chroma_store import ChromaStore

_RRF_K = 60  # standard RRF damping constant (Cormack et al., 2009)


def _chunk_key(hit: dict) -> tuple:
    return (hit["metadata"]["document_id"], hit["metadata"]["chunk_index"])


def _reciprocal_rank_fusion(*ranked_lists: list[dict]) -> list[dict]:
    rrf_scores: dict[tuple, float] = {}
    fused: dict[tuple, dict] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked):
            key = _chunk_key(hit)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            # Prefer a version of this chunk that carries a vector cosine
            # score (needed for the min_relevance_score gate below and for
            # Citation.score) over a BM25-only version that has none.
            if key not in fused or (not fused[key]["_is_vector_scored"] and hit["_is_vector_scored"]):
                fused[key] = hit
    ordered_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    return [fused[k] for k in ordered_keys]


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaStore,
        bm25_index: BM25Index,
        reranker: RerankerService,
    ):
        self._embeddings = embedding_service
        self._store = vector_store
        self._bm25 = bm25_index
        self._reranker = reranker

    def retrieve(
        self,
        question: str,
        top_k: int = settings.top_k,
        document_id: str | None = None,
        min_score: float = settings.min_relevance_score,
    ) -> list[dict]:
        # Cast a wider net than top_k before fusing/reranking — the whole
        # point of reranking is to have more real candidates to choose
        # from than the final answer needs.
        candidate_k = max(top_k * 3, 15)

        query_vector = self._embeddings.embed_query(question)
        vector_hits = self._store.query(query_vector, top_k=candidate_k, document_id=document_id)
        for hit in vector_hits:
            hit["_is_vector_scored"] = True

        bm25_hits = self._bm25.search(question, top_k=candidate_k, document_id=document_id)
        for hit in bm25_hits:
            hit["_is_vector_scored"] = False

        fused = _reciprocal_rank_fusion(vector_hits, bm25_hits)
        reranked = self._reranker.rerank(question, fused, top_k=top_k)

        return [
            hit
            for hit in reranked
            if not hit["_is_vector_scored"] or hit["score"] >= min_score
        ]
