"""Unit tests for AuthService — real UserStore against a temp SQLite file
(bcrypt/JWT correctness is exactly what's being tested, not worth mocking)."""
import pytest

from app.services.auth_service import AuthError, AuthService
from app.services.user_store import UserStore


def _service(tmp_path):
    store = UserStore(f"sqlite:///{tmp_path / 'users.db'}")
    return AuthService(store)


def test_first_registered_user_becomes_admin(tmp_path):
    auth = _service(tmp_path)
    user = auth.register("alice", "password123")
    assert user.role == "admin"


def test_second_registered_user_becomes_viewer(tmp_path):
    auth = _service(tmp_path)
    auth.register("alice", "password123")
    second = auth.register("bob", "password456")
    assert second.role == "viewer"


def test_register_duplicate_username_raises(tmp_path):
    auth = _service(tmp_path)
    auth.register("alice", "password123")
    with pytest.raises(AuthError):
        auth.register("alice", "different-password")


def test_authenticate_correct_password_succeeds(tmp_path):
    auth = _service(tmp_path)
    auth.register("alice", "password123")
    user = auth.authenticate("alice", "password123")
    assert user.username == "alice"


def test_authenticate_wrong_password_raises(tmp_path):
    auth = _service(tmp_path)
    auth.register("alice", "password123")
    with pytest.raises(AuthError):
        auth.authenticate("alice", "wrong-password")


def test_authenticate_unknown_username_raises(tmp_path):
    auth = _service(tmp_path)
    with pytest.raises(AuthError):
        auth.authenticate("nobody", "password123")


def test_token_roundtrip_recovers_username_and_role(tmp_path):
    auth = _service(tmp_path)
    user = auth.register("alice", "password123")

    token = auth.create_access_token(user)
    decoded = auth.decode_token(token)

    assert decoded.username == "alice"
    assert decoded.role == "admin"


def test_decode_garbage_token_raises(tmp_path):
    auth = _service(tmp_path)
    with pytest.raises(AuthError):
        auth.decode_token("not-a-real-token")
