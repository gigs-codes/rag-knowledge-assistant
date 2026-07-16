"""
Tests for PostgresDocumentRegistry against a REAL Postgres instance —
not mocked, deliberately, since the whole point of this class is SQL
correctness (schema creation, commits, row deletion), which a mock can't
verify. Skips automatically if no test database is reachable (set
EKA_TEST_DATABASE_URL to point at one, e.g. a local `docker run
postgres:16-alpine`), rather than failing the whole suite when Postgres
isn't set up — the JSON registry stays the tested default path either way.
"""
import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from app.services.postgres_document_registry import Base, PostgresDocumentRegistry

TEST_DATABASE_URL = os.environ.get(
    "EKA_TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/eka_test"
)


def _postgres_available() -> bool:
    try:
        create_engine(TEST_DATABASE_URL).connect().close()
        return True
    except OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(), reason="No test Postgres reachable at EKA_TEST_DATABASE_URL"
)


@pytest.fixture
def registry():
    reg = PostgresDocumentRegistry(TEST_DATABASE_URL)
    yield reg
    # Clean slate between tests rather than a fresh container per test —
    # cheaper, and fine since these tests don't run concurrently.
    Base.metadata.drop_all(reg._engine)


def test_add_and_get(registry):
    doc_id = str(uuid.uuid4())
    record = registry.add(doc_id, "policy.pdf", 5)

    assert record["id"] == doc_id
    assert record["num_chunks"] == 5
    fetched = registry.get(doc_id)
    assert fetched["filename"] == "policy.pdf"


def test_list_returns_all_documents(registry):
    registry.add(str(uuid.uuid4()), "a.pdf", 1)
    registry.add(str(uuid.uuid4()), "b.pdf", 2)

    assert len(registry.list()) == 2


def test_delete_removes_row(registry):
    doc_id = str(uuid.uuid4())
    registry.add(doc_id, "policy.pdf", 5)

    registry.delete(doc_id)

    assert registry.get(doc_id) is None
    assert registry.list() == []


def test_get_missing_returns_none(registry):
    assert registry.get(str(uuid.uuid4())) is None
