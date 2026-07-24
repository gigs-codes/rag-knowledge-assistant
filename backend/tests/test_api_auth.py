"""Integration tests for /auth endpoints. See test_api_documents.py's
docstring for why importing app.main has a one-time startup cost."""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.deps import get_auth_service
from app.api.security import get_current_user
from app.main import app
from app.models.schemas import UserOut
from app.services.auth_service import AuthError

client = TestClient(app)


def test_register_returns_token_and_user():
    fake_auth = MagicMock()
    fake_auth.register.return_value = UserOut(username="alice", role="admin")
    fake_auth.create_access_token.return_value = "fake-token"
    app.dependency_overrides[get_auth_service] = lambda: fake_auth
    try:
        response = client.post("/auth/register", json={"username": "alice", "password": "password123"})
        assert response.status_code == 200
        body = response.json()
        assert body["access_token"] == "fake-token"
        assert body["user"]["username"] == "alice"
    finally:
        app.dependency_overrides.pop(get_auth_service, None)


def test_register_duplicate_username_returns_409():
    fake_auth = MagicMock()
    fake_auth.register.side_effect = AuthError("Username 'alice' is already taken.")
    app.dependency_overrides[get_auth_service] = lambda: fake_auth
    try:
        response = client.post("/auth/register", json={"username": "alice", "password": "password123"})
        assert response.status_code == 409
    finally:
        app.dependency_overrides.pop(get_auth_service, None)


def test_register_rejects_short_password():
    response = client.post("/auth/register", json={"username": "alice", "password": "short"})
    assert response.status_code == 422


def test_login_returns_token_on_success():
    fake_auth = MagicMock()
    fake_auth.authenticate.return_value = UserOut(username="alice", role="viewer")
    fake_auth.create_access_token.return_value = "fake-token"
    app.dependency_overrides[get_auth_service] = lambda: fake_auth
    try:
        response = client.post("/auth/login", json={"username": "alice", "password": "password123"})
        assert response.status_code == 200
        assert response.json()["access_token"] == "fake-token"
    finally:
        app.dependency_overrides.pop(get_auth_service, None)


def test_login_wrong_password_returns_401():
    fake_auth = MagicMock()
    fake_auth.authenticate.side_effect = AuthError("Invalid username or password.")
    app.dependency_overrides[get_auth_service] = lambda: fake_auth
    try:
        response = client.post("/auth/login", json={"username": "alice", "password": "wrong"})
        assert response.status_code == 401
    finally:
        app.dependency_overrides.pop(get_auth_service, None)


def test_me_returns_current_user():
    app.dependency_overrides[get_current_user] = lambda: UserOut(username="alice", role="admin")
    try:
        response = client.get("/auth/me")
        assert response.status_code == 200
        assert response.json() == {"username": "alice", "role": "admin"}
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_me_without_token_returns_401():
    response = client.get("/auth/me")
    assert response.status_code == 401
