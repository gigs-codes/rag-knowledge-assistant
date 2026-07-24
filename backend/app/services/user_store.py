"""
SQLite-backed user store for auth. Same per-call session pattern as
PostgresDocumentRegistry (services/postgres_document_registry.py) — that
class documents why Postgres eventually wins for the *documents* table
(concurrent multi-process writes, query flexibility at scale); neither
concern applies here. User accounts are low-write, single-process, and
never need ad-hoc querying beyond "look up by username" — SQLite via
SQLAlchemy gives the same declarative-model/session ergonomics with zero
setup, in a file dedicated to this one concern (settings.users_db_path).
"""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class UserRow(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, nullable=False, unique=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


def _to_dict(row: UserRow) -> dict:
    return {
        "id": row.id,
        "username": row.username,
        "password_hash": row.password_hash,
        "role": row.role,
        "created_at": row.created_at.isoformat(),
    }


class UserStore:
    def __init__(self, database_url: str):
        self._engine = create_engine(database_url)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    def create(self, username: str, password_hash: str, role: str) -> dict:
        with self._Session() as session:
            row = UserRow(
                username=username,
                password_hash=password_hash,
                role=role,
                created_at=datetime.now(timezone.utc),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return _to_dict(row)

    def get_by_username(self, username: str) -> dict | None:
        with self._Session() as session:
            row = session.query(UserRow).filter(UserRow.username == username).one_or_none()
            return _to_dict(row) if row else None

    def count(self) -> int:
        with self._Session() as session:
            return session.query(func.count(UserRow.id)).scalar()
