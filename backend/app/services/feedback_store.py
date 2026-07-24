"""
SQLite-backed feedback log — thumbs up/down on answers. Same reasoning
and pattern as user_store.py: low-write, single-process, no ad-hoc
querying beyond "list everything," so SQLite via SQLAlchemy is a
zero-setup fit; no swappable-backend abstraction needed for something
this simple.
"""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class FeedbackRow(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    rating = Column(String, nullable=False)  # "up" or "down"
    source = Column(String, nullable=False)  # "query" or "agent"
    created_at = Column(DateTime(timezone=True), nullable=False)


def _to_dict(row: FeedbackRow) -> dict:
    return {
        "id": row.id,
        "question": row.question,
        "answer": row.answer,
        "rating": row.rating,
        "source": row.source,
        "created_at": row.created_at.isoformat(),
    }


class FeedbackStore:
    def __init__(self, database_url: str):
        self._engine = create_engine(database_url)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    def add(self, question: str, answer: str, rating: str, source: str) -> dict:
        with self._Session() as session:
            row = FeedbackRow(
                question=question,
                answer=answer,
                rating=rating,
                source=source,
                created_at=datetime.now(timezone.utc),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return _to_dict(row)

    def list(self) -> list[dict]:
        with self._Session() as session:
            # Order by id, not created_at: two inserts in the same call
            # can land in the same microsecond depending on platform clock
            # resolution, making created_at ties non-deterministic. id is
            # a strictly increasing autoincrement, so it's both a reliable
            # tie-breaker and — since it only ever increases with each
            # insert — already a correct proxy for insertion order on its own.
            rows = session.query(FeedbackRow).order_by(FeedbackRow.id.desc()).all()
            return [_to_dict(row) for row in rows]
