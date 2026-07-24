"""
Unit tests for RetrievalService with all collaborators mocked — this
service's own logic is sequencing (embed -> vector search + BM25 search
-> RRF fusion -> rerank -> relevance filter), which is cheap to verify
with mocks/fakes rather than real models.
"""
from unittest.mock import MagicMock

from app.services.retrieval_service import RetrievalService


def _hit(document_id, chunk_index, text, score):
    return {
        "text": text,
        "metadata": {"document_id": document_id, "filename": f"{document_id}.pdf", "chunk_index": chunk_index},
        "score": score,
    }


def _service(vector_hits, bm25_hits, reranked=None):
    embedding_service = MagicMock()
    embedding_service.embed_query.return_value = [0.1, 0.2, 0.3]
    vector_store = MagicMock()
    vector_store.query.return_value = vector_hits
    bm25_index = MagicMock()
    bm25_index.search.return_value = bm25_hits
    reranker = MagicMock()
    # By default, the reranker is a passthrough that just truncates to
    # top_k — tests that care about actual reordering pass `reranked`.
    reranker.rerank.side_effect = (
        (lambda query, candidates, top_k: reranked)
        if reranked is not None
        else (lambda query, candidates, top_k: candidates[:top_k])
    )
    service = RetrievalService(embedding_service, vector_store, bm25_index, reranker)
    return service, embedding_service, vector_store, bm25_index, reranker


def test_retrieve_embeds_query_then_searches_both_stores():
    hit = _hit("doc-1", 0, "chunk", 0.9)
    service, embedding_service, vector_store, bm25_index, _ = _service([hit], [])

    hits = service.retrieve("what is the policy?", top_k=3, document_id="doc-1")

    embedding_service.embed_query.assert_called_once_with("what is the policy?")
    vector_store.query.assert_called_once_with([0.1, 0.2, 0.3], top_k=15, document_id="doc-1")
    bm25_index.search.assert_called_once_with("what is the policy?", top_k=15, document_id="doc-1")
    assert hits[0]["text"] == "chunk"


def test_retrieve_filters_out_vector_hits_below_min_score():
    hits = [
        _hit("doc-1", 0, "relevant chunk", 0.8),
        _hit("doc-1", 1, "borderline chunk", 0.2),
        _hit("doc-1", 2, "noise chunk", 0.05),
    ]
    service, *_ = _service(hits, [])

    result = service.retrieve("question", min_score=0.2)

    assert [h["text"] for h in result] == ["relevant chunk", "borderline chunk"]


def test_retrieve_keeps_bm25_only_hits_regardless_of_min_score():
    # A chunk found only via BM25 (never returned by the vector search)
    # has no comparable cosine score to gate on — see retrieval_service.py's
    # docstring for why it's kept rather than dropped.
    bm25_hit = _hit("doc-2", 0, "keyword match only", 4.2)
    service, *_ = _service([], [bm25_hit], reranked=[bm25_hit])

    result = service.retrieve("question", min_score=0.9)

    assert result == [bm25_hit]


def test_retrieve_deduplicates_hits_found_by_both_searches():
    shared = _hit("doc-1", 0, "found by both", 0.7)
    service, *_ = _service([shared], [dict(shared)], reranked=[shared])

    result = service.retrieve("question", min_score=0.0)

    assert len(result) == 1


def test_retrieve_passes_fused_candidates_to_reranker():
    vector_hit = _hit("doc-1", 0, "vector match", 0.9)
    bm25_hit = _hit("doc-2", 0, "bm25 match", 3.0)
    service, *_, reranker = _service([vector_hit], [bm25_hit], reranked=[vector_hit, bm25_hit])

    service.retrieve("question", top_k=2, min_score=0.0)

    call_args = reranker.rerank.call_args
    assert call_args.args[0] == "question"
    assert call_args.kwargs["top_k"] == 2
    fused_keys = {(h["metadata"]["document_id"], h["metadata"]["chunk_index"]) for h in call_args.args[1]}
    assert fused_keys == {("doc-1", 0), ("doc-2", 0)}
