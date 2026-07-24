"""
Unit tests for IngestionService — using real file I/O (tmp_path) and a
mocked embedding service/vector store/registry, since what's actually
worth verifying here is the extraction-format dispatch and text cleaning,
not re-proving embedding/storage (those have their own tests).
"""
from unittest.mock import MagicMock, patch

import docx
import openpyxl
import pytest

from app.services.document_registry import DocumentRegistry
from app.services.ingestion_service import (
    IngestionService,
    _clean_text,
    _extract_pdf_tables,
    _rows_to_markdown_table,
)


def _service(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.ingestion_service.settings.upload_dir", tmp_path)
    embedding_service = MagicMock()
    embedding_service.embed_texts.return_value = [[0.1, 0.2]]
    vector_store = MagicMock()
    registry = DocumentRegistry(path=tmp_path / "documents.json")
    bm25_index = MagicMock()
    return IngestionService(embedding_service, vector_store, registry, bm25_index), vector_store, bm25_index


def test_clean_text_collapses_whitespace_artifacts():
    messy = "Hello   world\n\n\n\nnext   paragraph"
    assert _clean_text(messy) == "Hello world\n\nnext paragraph"


def test_ingest_txt_document(tmp_path, monkeypatch):
    service, vector_store, bm25_index = _service(tmp_path, monkeypatch)
    record = service.ingest_document(b"Plain text content about a policy.", "notes.txt")

    assert record["filename"] == "notes.txt"
    assert record["num_chunks"] >= 1
    vector_store.add_chunks.assert_called_once()
    bm25_index.add_document.assert_called_once()


def test_ingest_md_document(tmp_path, monkeypatch):
    service, vector_store, _ = _service(tmp_path, monkeypatch)
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

    service, vector_store, _ = _service(tmp_path, monkeypatch)
    record = service.ingest_document(buffer.getvalue(), "policy.docx")

    assert record["filename"] == "policy.docx"
    vector_store.add_chunks.assert_called_once()
    chunks_arg = vector_store.add_chunks.call_args.args[2]
    assert "remote work policy" in chunks_arg[0]


def test_ingest_document_rejects_unsupported_extension(tmp_path, monkeypatch):
    service, _, _bm25 = _service(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Unsupported file type"):
        service.ingest_document(b"whatever", "archive.zip")


def test_ingest_document_rejects_empty_extracted_text(tmp_path, monkeypatch):
    service, _, _bm25 = _service(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="No extractable text"):
        service.ingest_document(b"   \n\n  ", "empty.txt")


def test_ingest_csv_document(tmp_path, monkeypatch):
    service, vector_store, _ = _service(tmp_path, monkeypatch)
    csv_bytes = b"name,department\nAlice,Engineering\nBob,Sales\n"

    record = service.ingest_document(csv_bytes, "employees.csv")

    assert record["filename"] == "employees.csv"
    chunks_arg = vector_store.add_chunks.call_args.args[2]
    combined = "\n".join(chunks_arg)
    assert "name: Alice" in combined
    assert "department: Engineering" in combined


def test_ingest_xlsx_document(tmp_path, monkeypatch):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Staff"
    sheet.append(["name", "role"])
    sheet.append(["Alice", "Engineer"])
    buffer_path = tmp_path / "src.xlsx"
    workbook.save(buffer_path)
    xlsx_bytes = buffer_path.read_bytes()

    service, vector_store, _ = _service(tmp_path, monkeypatch)
    record = service.ingest_document(xlsx_bytes, "staff.xlsx")

    assert record["filename"] == "staff.xlsx"
    chunks_arg = vector_store.add_chunks.call_args.args[2]
    combined = "\n".join(chunks_arg)
    assert "Sheet: Staff" in combined
    assert "name: Alice" in combined
    assert "role: Engineer" in combined


def test_rows_to_markdown_table_formats_header_and_rows():
    rows = [["Name", "Role"], ["Alice", "Engineer"], ["Bob", "Sales"]]
    markdown = _rows_to_markdown_table(rows)
    lines = markdown.splitlines()
    assert lines[0] == "| Name | Role |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| Alice | Engineer |"


def test_extract_pdf_tables_skips_tables_with_only_a_header_row():
    fake_page = MagicMock()
    fake_page.extract_tables.return_value = [[["Name", "Role"]]]  # one row only
    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = False

    with patch("app.services.ingestion_service.pdfplumber.open", return_value=fake_pdf):
        assert _extract_pdf_tables("fake.pdf") == ""


def test_extract_pdf_tables_renders_detected_table_as_markdown():
    fake_page = MagicMock()
    fake_page.extract_tables.return_value = [[["Name", "Role"], ["Alice", "Engineer"]]]
    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = False

    with patch("app.services.ingestion_service.pdfplumber.open", return_value=fake_pdf):
        result = _extract_pdf_tables("fake.pdf")

    assert "Table (page 1):" in result
    assert "| Alice | Engineer |" in result


def test_delete_document_removes_file_regardless_of_extension(tmp_path, monkeypatch):
    service, _, bm25_index = _service(tmp_path, monkeypatch)
    record = service.ingest_document(b"some content here", "notes.txt")
    document_id = record["id"]
    assert list(tmp_path.glob(f"{document_id}.*"))

    service.delete_document(document_id)

    assert list(tmp_path.glob(f"{document_id}.*")) == []
    bm25_index.remove_document.assert_called_once_with(document_id)
