"""Regression test for the single-container deploy (see Dockerfile,
deploy/start.sh): main.py mounts every router a second time under /api,
in addition to the existing unprefixed mounts, so FastAPI can serve both
the API and the built frontend from one process/port with no nginx or
Vite dev-proxy to strip the prefix. This only checks that both mount
points resolve to the same behavior — the routes' actual logic is already
covered by the unprefixed tests (test_api_documents.py, test_api_query.py,
etc.), so this deliberately doesn't re-test business logic, just the
routing-table wiring itself."""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_registry
from app.api.security import get_current_user
from app.main import app
from app.models.schemas import UserOut

client = TestClient(app)


@pytest.fixture(autouse=True)
def _authenticated():
    app.dependency_overrides[get_current_user] = lambda: UserOut(username="test-user", role="viewer")
    yield
    app.dependency_overrides.pop(get_current_user, None)


def test_documents_route_resolves_both_unprefixed_and_under_api():
    fake_registry = MagicMock()
    fake_registry.list.return_value = []
    app.dependency_overrides[get_registry] = lambda: fake_registry
    try:
        assert client.get("/documents").status_code == 200
        assert client.get("/api/documents").status_code == 200
    finally:
        app.dependency_overrides.pop(get_registry, None)


def test_api_prefixed_route_still_enforces_auth():
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/api/documents")
    assert response.status_code == 401
