"""Integration tests for /feedback endpoints. See test_api_documents.py's
docstring for why importing app.main has a one-time startup cost."""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_feedback_store
from app.api.security import get_current_user
from app.main import app
from app.models.schemas import UserOut

client = TestClient(app)


@pytest.fixture(autouse=True)
def _authenticated_as_viewer():
    app.dependency_overrides[get_current_user] = lambda: UserOut(username="test-user", role="viewer")
    yield
    app.dependency_overrides.pop(get_current_user, None)


def test_submit_feedback_returns_record():
    fake_store = MagicMock()
    fake_store.add.return_value = {
        "id": 1,
        "question": "Q",
        "answer": "A",
        "rating": "up",
        "source": "query",
        "created_at": "2026-01-01T00:00:00Z",
    }
    app.dependency_overrides[get_feedback_store] = lambda: fake_store
    try:
        response = client.post("/feedback", json={"question": "Q", "answer": "A", "rating": "up"})
        assert response.status_code == 200
        assert response.json()["rating"] == "up"
        fake_store.add.assert_called_once_with("Q", "A", "up", "query")
    finally:
        app.dependency_overrides.pop(get_feedback_store, None)


def test_submit_feedback_rejects_invalid_rating():
    response = client.post("/feedback", json={"question": "Q", "answer": "A", "rating": "sideways"})
    assert response.status_code == 422


def test_submit_feedback_without_auth_returns_401():
    app.dependency_overrides.pop(get_current_user, None)
    response = client.post("/feedback", json={"question": "Q", "answer": "A", "rating": "up"})
    assert response.status_code == 401


def test_list_feedback_requires_admin_role():
    response = client.get("/feedback")  # authenticated as viewer via the autouse fixture
    assert response.status_code == 403


def test_list_feedback_allows_admin_role():
    app.dependency_overrides[get_current_user] = lambda: UserOut(username="admin1", role="admin")
    fake_store = MagicMock()
    fake_store.list.return_value = []
    app.dependency_overrides[get_feedback_store] = lambda: fake_store
    try:
        response = client.get("/feedback")
        assert response.status_code == 200
        assert response.json() == []
    finally:
        app.dependency_overrides.pop(get_feedback_store, None)
