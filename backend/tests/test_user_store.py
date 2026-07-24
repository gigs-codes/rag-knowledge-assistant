"""Unit tests for UserStore — real SQLite file via pytest tmp_path, same
reasoning as test_document_registry.py: the point of this class is SQL
read/write correctness, so a real (temp) database is what's worth testing."""
from app.services.user_store import UserStore


def _store(tmp_path):
    return UserStore(f"sqlite:///{tmp_path / 'users.db'}")


def test_create_and_get_by_username(tmp_path):
    store = _store(tmp_path)

    record = store.create("alice", "hashed-pw", "admin")

    assert record["username"] == "alice"
    assert record["role"] == "admin"
    fetched = store.get_by_username("alice")
    assert fetched["password_hash"] == "hashed-pw"


def test_get_by_username_missing_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.get_by_username("nobody") is None


def test_count_reflects_number_of_users(tmp_path):
    store = _store(tmp_path)
    assert store.count() == 0

    store.create("alice", "hash1", "admin")
    store.create("bob", "hash2", "viewer")

    assert store.count() == 2
