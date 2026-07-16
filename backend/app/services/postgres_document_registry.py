"""
Postgres-backed implementation of DocumentRegistryBase — same interface
as the JSON registry (document_registry.py), different storage engine.
Activated by setting EKA_DATABASE_URL (see api/deps.py); the JSON
registry stays the zero-setup default otherwise.

Why this becomes worth it eventually, when JSON stops being enough:
- Concurrent writes: the JSON registry holds a Python `threading.Lock`,
  which only protects against races WITHIN a single process. Run two
  backend instances behind a load balancer (needed once traffic outgrows
  one process) and they'd each hold their own lock, guarding nothing —
  two simultaneous uploads could corrupt the file. Postgres's row-level
  locking and transactions handle this correctly across any number of
  connections/processes without extra application code.
- Query flexibility: "list" always means "read and parse the entire file"
  for the JSON version, even to answer something like "documents uploaded
  this week" — a real database answers that with an indexed WHERE clause
  instead of loading everything into memory to filter in Python.
- This is also exactly the kind of decision worth NOT making prematurely:
  a single JSON file genuinely is simpler and sufficient at this
  project's actual scale, which is why it's still the default.
"""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.services.document_registry_base import DocumentRegistryBase

Base = declarative_base()


class DocumentRow(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    num_chunks = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), nullable=False)


def _to_dict(row: DocumentRow) -> dict:
    return {
        "id": row.id,
        "filename": row.filename,
        "num_chunks": row.num_chunks,
        "uploaded_at": row.uploaded_at.isoformat(),
    }


class PostgresDocumentRegistry(DocumentRegistryBase):
    def __init__(self, database_url: str):
        self._engine = create_engine(database_url)
        # Table creation belongs here rather than in a separate migration
        # step for THIS project's scope — a single `documents` table with
        # no foreign keys or evolving schema doesn't yet justify a real
        # migration tool (Alembic). Worth revisiting the moment the schema
        # needs a second table or its first ALTER.
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    def add(self, document_id: str, filename: str, num_chunks: int) -> dict:
        with self._Session() as session:
            row = DocumentRow(
                id=document_id,
                filename=filename,
                num_chunks=num_chunks,
                uploaded_at=datetime.now(timezone.utc),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return _to_dict(row)

    def list(self) -> list[dict]:
        with self._Session() as session:
            return [_to_dict(row) for row in session.query(DocumentRow).all()]

    def get(self, document_id: str) -> dict | None:
        with self._Session() as session:
            row = session.get(DocumentRow, document_id)
            return _to_dict(row) if row else None

    def delete(self, document_id: str) -> None:
        with self._Session() as session:
            row = session.get(DocumentRow, document_id)
            if row is not None:
                session.delete(row)
                session.commit()
