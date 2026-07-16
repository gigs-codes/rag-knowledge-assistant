"""
Unit tests for IngestionService — using real file I/O (tmp_path) and a
mocked embedding service/vector store/registry, since what's actually
worth verifying here is the extraction-format dispatch and text cleaning,
not re-proving embedding/storage (those have their own tests).
"""
from unittest.mock import MagicMock

import docx
import pytest

from app.services.document_registry import DocumentRegistry
from app.services.ingestion_service import IngestionService, _clean_text


def _service(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.ingestion_service.settings.upload_dir", tmp_path)
    embedding_service = MagicMock()
    embedding_service.embed_texts.return_value = [[0.1, 0.2]]
    vector_store = MagicMock()
    registry = DocumentRegistry(path=tmp_path / "documents.json")
    return IngestionService(embedding_service, vector_store, registry), vector_store


def test_clean_text_collapses_whitespace_artifacts():
    messy = "Hello   world\n\n\n\nnext   paragraph"
    assert _clean_text(messy) == "Hello world\n\nnext paragraph"


def test_ingest_txt_document(tmp_path, monkeypatch):
    service, vector_store = _service(tmp_path, monkeypatch)
    record = service.ingest_document(b"Plain text content about a policy.", "notes.txt")

    assert record["filename"] == "notes.txt"
    assert record["num_chunks"] >= 1
    vector_store.add_chunks.assert_called_once()


def test_ingest_md_document(tmp_path, monkeypatch):
    service, vector_store = _service(tmp_path, monkeypatch)
    record = service.ingest_document(b"# Heading\n\nSome markdown content.", "readme.md")

    assert record["filename"] == "readme.md"
    vector_store.add_chunks.assert_called_once()


def test_ingest_docx_document(tmp_path, monkeypatch):
    # Build a real, minimal .docx in memory rather than a hand-rolled
    # fixture — python-docx already gives us a correct writer, and using
    # it here proves ingestion against an actually valid .docx file.
    import io

    document = docx.Document()
    document.add_paragraph("This is a Word document about a remote work policy.")
    buffer = io.BytesIO()
    document.save(buffer)

    service, vector_store = _service(tmp_path, monkeypatch)
    record = service.ingest_document(buffer.getvalue(), "policy.docx")

    assert record["filename"] == "policy.docx"
    vector_store.add_chunks.assert_called_once()
    chunks_arg = vector_store.add_chunks.call_args.args[2]
    assert "remote work policy" in chunks_arg[0]


def test_ingest_document_rejects_unsupported_extension(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Unsupported file type"):
        service.ingest_document(b"whatever", "archive.zip")


def test_ingest_document_rejects_empty_extracted_text(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="No extractable text"):
        service.ingest_document(b"   \n\n  ", "empty.txt")


def test_delete_document_removes_file_regardless_of_extension(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, monkeypatch)
    record = service.ingest_document(b"some content here", "notes.txt")
    document_id = record["id"]
    assert list(tmp_path.glob(f"{document_id}.*"))

    service.delete_document(document_id)

    assert list(tmp_path.glob(f"{document_id}.*")) == []
