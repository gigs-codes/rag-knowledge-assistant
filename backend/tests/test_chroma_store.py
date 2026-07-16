"""
Unit tests for ChromaStore against a real (temp-directory) Chroma instance
— not mocked. Chroma itself is fast and has no external dependency (it's
an embedded DB), so there's no reason to fake it; we supply our own small
hand-picked vectors so nearest-neighbor results are deterministic without
needing the real embedding model loaded.
"""
from app.vectorstore.chroma_store import ChromaStore, new_document_id


def test_add_and_query_returns_nearest_chunk(tmp_path):
    store = ChromaStore(persist_dir=str(tmp_path))
    doc_id = new_document_id()
    store.add_chunks(
        doc_id,
        "animals.pdf",
        ["chunk about cats", "chunk about dogs"],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    hits = store.query([1.0, 0.0], top_k=1)

    assert len(hits) == 1
    assert hits[0]["text"] == "chunk about cats"
    assert hits[0]["metadata"]["filename"] == "animals.pdf"
    assert hits[0]["metadata"]["chunk_index"] == 0
    assert hits[0]["score"] > 0.9  # near-identical vector -> near-1.0 similarity


def test_query_filters_by_document_id(tmp_path):
    store = ChromaStore(persist_dir=str(tmp_path))
    doc_a, doc_b = new_document_id(), new_document_id()
    store.add_chunks(doc_a, "a.pdf", ["a text"], [[1.0, 0.0]])
    store.add_chunks(doc_b, "b.pdf", ["b text"], [[1.0, 0.0]])

    hits = store.query([1.0, 0.0], top_k=5, document_id=doc_a)

    assert len(hits) == 1
    assert hits[0]["metadata"]["document_id"] == doc_a


def test_query_on_empty_collection_returns_empty_list(tmp_path):
    store = ChromaStore(persist_dir=str(tmp_path))
    assert store.query([1.0, 0.0], top_k=5) == []


def test_delete_document_removes_its_chunks_only(tmp_path):
    store = ChromaStore(persist_dir=str(tmp_path))
    doc_a, doc_b = new_document_id(), new_document_id()
    store.add_chunks(doc_a, "a.pdf", ["a text"], [[1.0, 0.0]])
    store.add_chunks(doc_b, "b.pdf", ["b text"], [[0.0, 1.0]])

    store.delete_document(doc_a)

    assert store.count_chunks(doc_a) == 0
    assert store.count_chunks(doc_b) == 1
