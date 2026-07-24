"""Unit tests for BM25Index — real rank_bm25 index against a tmp_path
pickle file, since the point of this class is search/persistence
correctness, not something worth mocking."""
from app.vectorstore.bm25_index import BM25Index


def _index(tmp_path):
    return BM25Index(persist_path=tmp_path / "bm25.pkl")


def test_add_and_search_returns_matching_chunk(tmp_path):
    index = _index(tmp_path)
    index.add_document("doc-1", "policy.pdf", ["Remote work is allowed three days a week."])

    hits = index.search("remote work policy", top_k=3)

    assert len(hits) == 1
    assert hits[0]["metadata"]["document_id"] == "doc-1"
    assert hits[0]["score"] > 0


def test_search_returns_empty_for_unrelated_query(tmp_path):
    index = _index(tmp_path)
    index.add_document("doc-1", "policy.pdf", ["Remote work is allowed three days a week."])

    hits = index.search("quarterly revenue projections", top_k=3)

    assert hits == []


def test_search_with_no_documents_returns_empty(tmp_path):
    index = _index(tmp_path)
    assert index.search("anything", top_k=3) == []


def test_remove_document_excludes_it_from_future_searches(tmp_path):
    index = _index(tmp_path)
    index.add_document("doc-1", "policy.pdf", ["Remote work is allowed three days a week."])

    index.remove_document("doc-1")

    assert index.search("remote work", top_k=3) == []


def test_document_id_filter_scopes_results(tmp_path):
    index = _index(tmp_path)
    index.add_document("doc-1", "a.pdf", ["Remote work policy details."])
    index.add_document("doc-2", "b.pdf", ["Remote work policy details."])

    hits = index.search("remote work policy", top_k=5, document_id="doc-1")

    assert all(h["metadata"]["document_id"] == "doc-1" for h in hits)


def test_persists_and_reloads_from_disk(tmp_path):
    path = tmp_path / "bm25.pkl"
    index = BM25Index(persist_path=path)
    index.add_document("doc-1", "policy.pdf", ["Remote work is allowed three days a week."])

    reloaded = BM25Index(persist_path=path)
    hits = reloaded.search("remote work", top_k=3)

    assert len(hits) == 1
    assert hits[0]["metadata"]["filename"] == "policy.pdf"
