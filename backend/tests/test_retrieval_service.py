"""
Unit tests for RetrievalService with both collaborators mocked — this
service's own logic is just sequencing (embed the query, then query the
store) plus the relevance-score filter, both cheap to verify with mocks
rather than a real vector store.
"""
from unittest.mock import MagicMock

from app.services.retrieval_service import RetrievalService


def test_retrieve_embeds_query_then_queries_store():
    embedding_service = MagicMock()
    embedding_service.embed_query.return_value = [0.1, 0.2, 0.3]
    vector_store = MagicMock()
    vector_store.query.return_value = [{"text": "chunk", "metadata": {}, "score": 0.9}]

    service = RetrievalService(embedding_service, vector_store)
    hits = service.retrieve("what is the policy?", top_k=3, document_id="doc-1")

    embedding_service.embed_query.assert_called_once_with("what is the policy?")
    vector_store.query.assert_called_once_with([0.1, 0.2, 0.3], top_k=3, document_id="doc-1")
    assert hits == [{"text": "chunk", "metadata": {}, "score": 0.9}]


def test_retrieve_filters_out_hits_below_min_score():
    embedding_service = MagicMock()
    embedding_service.embed_query.return_value = [0.1, 0.2, 0.3]
    vector_store = MagicMock()
    vector_store.query.return_value = [
        {"text": "relevant chunk", "metadata": {}, "score": 0.8},
        {"text": "borderline chunk", "metadata": {}, "score": 0.2},
        {"text": "noise chunk", "metadata": {}, "score": 0.05},
    ]

    service = RetrievalService(embedding_service, vector_store)
    hits = service.retrieve("question", min_score=0.2)

    assert [h["text"] for h in hits] == ["relevant chunk", "borderline chunk"]
