"""
Integration tests for /documents endpoints: real FastAPI app + real
routing/validation, with the service layer swapped for test doubles via
`app.dependency_overrides`. This tests the HTTP layer's actual job —
status codes, content-type validation, request/response shape — without
needing a real PDF, embedding model, or vector store.

Note: importing `app.main` still pays the one-time cost of constructing
the real singletons in deps.py (including loading the embedding model),
because those are built eagerly at import time — see deps.py's docstring
for why that's the deliberate choice. `dependency_overrides` swaps which
instances the routes receive; it doesn't prevent the module-level ones
from being constructed in the first place. A lazy-DI refactor (e.g.
`@lru_cache`-wrapped provider functions) would avoid paying that cost at
import time, at the cost of moving it to the first real request instead —
worth revisiting if this test suite's startup time becomes a real pain
point, not worth it purely to make this file marginally faster today.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_ingestion_service, get_registry
from app.api.security import get_current_user
from app.main import app
from app.models.schemas import UserOut

client = TestClient(app)


@pytest.fixture(autouse=True)
def _authenticated_as_admin():
    # These tests exercise upload/delete, which now require the admin
    # role (see documents.py) — default every test in this file to an
    # admin identity so existing behavior stays covered without each test
    # wiring up a real JWT. Role-gating itself is verified separately
    # below (test_upload_requires_admin_role, test_delete_requires_admin_role).
    app.dependency_overrides[get_current_user] = lambda: UserOut(username="test-admin", role="admin")
    yield
    app.dependency_overrides.pop(get_current_user, None)


def test_upload_rejects_unsupported_extension():
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.xyz", b"hello", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_accepts_txt_and_md_and_docx_extensions():
    # These are validated by extension at the route layer (see
    # documents.py) before ever reaching IngestionService — this test
    # proves that gate lets the new formats through, not that ingestion
    # itself works (that's ingestion_service's own tests' job).
    fake_ingestion = MagicMock()
    fake_ingestion.ingest_document.return_value = {
        "id": "doc-1",
        "filename": "x",
        "num_chunks": 1,
        "uploaded_at": "2026-01-01T00:00:00Z",
    }
    app.dependency_overrides[get_ingestion_service] = lambda: fake_ingestion
    try:
        for filename, content_type in [
            ("notes.txt", "text/plain"),
            ("notes.md", "text/markdown"),
            ("report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ]:
            response = client.post(
                "/documents/upload", files={"file": (filename, b"content", content_type)}
            )
            assert response.status_code == 200, f"{filename} should be accepted"
    finally:
        app.dependency_overrides.pop(get_ingestion_service, None)


def test_upload_success_returns_document_record():
    fake_ingestion = MagicMock()
    fake_ingestion.ingest_document.return_value = {
        "id": "doc-1",
        "filename": "policy.pdf",
        "num_chunks": 3,
        "uploaded_at": "2026-01-01T00:00:00Z",
    }
    app.dependency_overrides[get_ingestion_service] = lambda: fake_ingestion
    try:
        response = client.post(
            "/documents/upload",
            files={"file": ("policy.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["document"]["id"] == "doc-1"
        assert body["document"]["num_chunks"] == 3
    finally:
        app.dependency_overrides.pop(get_ingestion_service, None)


def test_upload_translates_value_error_to_422():
    fake_ingestion = MagicMock()
    fake_ingestion.ingest_document.side_effect = ValueError("No extractable text found in PDF.")
    app.dependency_overrides[get_ingestion_service] = lambda: fake_ingestion
    try:
        response = client.post(
            "/documents/upload",
            files={"file": ("scanned.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_ingestion_service, None)


def test_delete_missing_document_returns_404():
    fake_registry = MagicMock()
    fake_registry.get.return_value = None
    app.dependency_overrides[get_registry] = lambda: fake_registry
    try:
        response = client.delete("/documents/does-not-exist")
        assert response.status_code == 404
    finally:
        app.dependency_overrides.pop(get_registry, None)


def test_delete_existing_document_succeeds():
    fake_registry = MagicMock()
    fake_registry.get.return_value = {"id": "doc-1", "filename": "policy.pdf"}
    fake_ingestion = MagicMock()
    app.dependency_overrides[get_registry] = lambda: fake_registry
    app.dependency_overrides[get_ingestion_service] = lambda: fake_ingestion
    try:
        response = client.delete("/documents/doc-1")
        assert response.status_code == 200
        fake_ingestion.delete_document.assert_called_once_with("doc-1")
    finally:
        app.dependency_overrides.pop(get_registry, None)
        app.dependency_overrides.pop(get_ingestion_service, None)


def test_upload_requires_admin_role():
    app.dependency_overrides[get_current_user] = lambda: UserOut(username="viewer1", role="viewer")
    try:
        response = client.post(
            "/documents/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
        assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_upload_without_auth_returns_401():
    app.dependency_overrides.pop(get_current_user, None)
    response = client.post("/documents/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 401


def test_delete_requires_admin_role():
    app.dependency_overrides[get_current_user] = lambda: UserOut(username="viewer1", role="viewer")
    try:
        response = client.delete("/documents/doc-1")
        assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_list_documents_allows_viewer_role():
    fake_registry = MagicMock()
    fake_registry.list.return_value = []
    app.dependency_overrides[get_registry] = lambda: fake_registry
    app.dependency_overrides[get_current_user] = lambda: UserOut(username="viewer1", role="viewer")
    try:
        response = client.get("/documents")
        assert response.status_code == 200
    finally:
        app.dependency_overrides.pop(get_registry, None)
        app.dependency_overrides.pop(get_current_user, None)
