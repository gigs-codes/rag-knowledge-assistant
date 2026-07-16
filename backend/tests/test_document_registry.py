"""
Unit tests for DocumentRegistry — real file I/O against a pytest tmp_path
rather than mocking the filesystem. Worth doing for real here: the whole
point of this class is JSON read/write correctness, so mocking `open()`
would test nothing real.
"""
from app.services.document_registry import DocumentRegistry


def test_add_and_list(tmp_path):
    registry = DocumentRegistry(path=tmp_path / "documents.json")

    record = registry.add("doc-1", "policy.pdf", 5)

    assert record["id"] == "doc-1"
    assert record["filename"] == "policy.pdf"
    assert record["num_chunks"] == 5
    assert "uploaded_at" in record
    assert registry.list() == [record]


def test_get_missing_returns_none(tmp_path):
    registry = DocumentRegistry(path=tmp_path / "documents.json")
    assert registry.get("does-not-exist") is None


def test_get_existing(tmp_path):
    registry = DocumentRegistry(path=tmp_path / "documents.json")
    registry.add("doc-1", "policy.pdf", 5)
    assert registry.get("doc-1")["filename"] == "policy.pdf"


def test_delete_removes_record(tmp_path):
    registry = DocumentRegistry(path=tmp_path / "documents.json")
    registry.add("doc-1", "policy.pdf", 5)

    registry.delete("doc-1")

    assert registry.list() == []
    assert registry.get("doc-1") is None


def test_delete_missing_is_a_noop(tmp_path):
    registry = DocumentRegistry(path=tmp_path / "documents.json")
    registry.delete("never-existed")  # should not raise
    assert registry.list() == []


def test_new_registry_creates_file_if_missing(tmp_path):
    path = tmp_path / "nested" / "documents.json"
    path.parent.mkdir()
    DocumentRegistry(path=path)
    assert path.exists()
    assert path.read_text() == "{}"
