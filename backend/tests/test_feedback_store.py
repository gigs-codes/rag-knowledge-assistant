"""Unit tests for FeedbackStore — real SQLite file via pytest tmp_path,
same reasoning as test_user_store.py."""
from app.services.feedback_store import FeedbackStore


def _store(tmp_path):
    return FeedbackStore(f"sqlite:///{tmp_path / 'feedback.db'}")


def test_add_returns_persisted_record(tmp_path):
    store = _store(tmp_path)

    record = store.add("What is X?", "X is Y.", "up", "query")

    assert record["question"] == "What is X?"
    assert record["answer"] == "X is Y."
    assert record["rating"] == "up"
    assert record["source"] == "query"
    assert "created_at" in record


def test_list_returns_all_feedback_newest_first(tmp_path):
    store = _store(tmp_path)
    store.add("Q1", "A1", "up", "query")
    store.add("Q2", "A2", "down", "agent")

    records = store.list()

    assert len(records) == 2
    assert records[0]["question"] == "Q2"  # newest first
    assert records[1]["question"] == "Q1"


def test_list_empty_store_returns_empty_list(tmp_path):
    store = _store(tmp_path)
    assert store.list() == []
