"""Unit test for RerankerService against the REAL cross-encoder model —
worth doing for real (not mocked) since the whole point is verifying the
model actually reorders candidates by relevance, which a mock can't prove.
Downloads the small (~90MB) model on first run, same as EmbeddingService's
tests implicitly do for the embedding model."""
from app.services.reranker_service import RerankerService


def test_rerank_orders_by_relevance_to_query():
    reranker = RerankerService()
    candidates = [
        {"text": "The Eiffel Tower is located in Paris, France.", "metadata": {}, "score": 0.5},
        {"text": "Remote employees get a $500 annual home office stipend.", "metadata": {}, "score": 0.5},
    ]

    ranked = reranker.rerank("How much is the home office stipend?", candidates, top_k=2)

    assert "stipend" in ranked[0]["text"]


def test_rerank_respects_top_k():
    reranker = RerankerService()
    candidates = [
        {"text": f"Unrelated sentence number {i}.", "metadata": {}, "score": 0.1} for i in range(5)
    ]

    ranked = reranker.rerank("some query", candidates, top_k=2)

    assert len(ranked) == 2


def test_rerank_empty_candidates_returns_empty():
    reranker = RerankerService()
    assert reranker.rerank("query", [], top_k=3) == []
