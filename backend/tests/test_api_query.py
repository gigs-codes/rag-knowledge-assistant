"""Integration tests for the /query endpoint. See test_api_documents.py's
docstring for why importing app.main has a one-time startup cost."""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.deps import get_chat_service
from app.main import app
from app.models.schemas import QueryResponse

client = TestClient(app)


def test_query_delegates_to_chat_service_and_returns_its_response():
    fake_chat = MagicMock()
    fake_chat.answer.return_value = QueryResponse(answer="42", citations=[], latency_ms=10)
    app.dependency_overrides[get_chat_service] = lambda: fake_chat
    try:
        response = client.post("/query", json={"question": "what is the answer?"})
        assert response.status_code == 200
        assert response.json()["answer"] == "42"
        fake_chat.answer.assert_called_once_with("what is the answer?", document_id=None)
    finally:
        app.dependency_overrides.pop(get_chat_service, None)


def test_query_passes_document_id_through():
    fake_chat = MagicMock()
    fake_chat.answer.return_value = QueryResponse(answer="x", citations=[], latency_ms=1)
    app.dependency_overrides[get_chat_service] = lambda: fake_chat
    try:
        client.post("/query", json={"question": "q", "document_id": "doc-1"})
        fake_chat.answer.assert_called_once_with("q", document_id="doc-1")
    finally:
        app.dependency_overrides.pop(get_chat_service, None)


def test_query_rejects_empty_question():
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_rejects_missing_question_field():
    response = client.post("/query", json={})
    assert response.status_code == 422
