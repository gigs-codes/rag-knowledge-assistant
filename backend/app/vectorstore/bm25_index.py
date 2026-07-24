"""
BM25 keyword index — the keyword half of hybrid search (see
retrieval_service.py, which documents this as the seam where hybrid
retrieval plugs in). Vector search (ChromaStore) is strong at semantic
similarity but weak at exact-term matches (product codes, names, IDs) that
share little embedding-space structure with paraphrases; BM25 is the
inverse. Combining both (via Reciprocal Rank Fusion in retrieval_service.py)
covers each other's blind spot.

Why `rank_bm25` and a full-corpus rebuild per write, rather than an
incremental index: BM25's variants have no incremental-update API — they
compute term/document statistics (IDF, average doc length) over the whole
corpus up front. At this project's scale (a handful of uploaded documents,
not a production-sized corpus), rebuilding on every add/delete is fast
enough that reaching for a real inverted-index library (e.g. Whoosh,
Elasticsearch) would be solving a problem this project doesn't have yet.

Why BM25Plus specifically, not the more common BM25Okapi: classic BM25's
IDF term goes NEGATIVE for a word that appears in every document in the
corpus (df == N) — and with only a handful of uploaded documents (this
project's actual scale), that's common, not an edge case: upload one PDF
and every one of its distinctive terms has df == N == 1. A negative score
for a genuinely relevant match would be indistinguishable from — and this
class's own zero-score filter would have discarded it same as — an
irrelevant one. BM25Plus adds a small positive floor (delta) specifically
to fix this small-corpus degenerate case; verified empirically that
BM25Okapi does return a negative score here while BM25Plus returns a
real positive one.

Why persisted via pickle rather than rebuilt from Chroma at every startup:
Chroma doesn't expose "give me every chunk back" as a first-class query
(it's built for nearest-neighbor search, not full scans), so BM25Index
keeps its own copy of chunk records — pickling that copy is simpler than
reconstructing it by re-reading every document from disk on each boot.
"""
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Plus

from app.core.config import settings


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, persist_path: Path = settings.base_dir / "data" / "bm25_index.pkl"):
        self._persist_path = persist_path
        self._records: list[dict] = []  # each: {document_id, filename, chunk_index, text}
        self._bm25: BM25Plus | None = None
        self._load()

    def _load(self) -> None:
        if self._persist_path.exists():
            with open(self._persist_path, "rb") as f:
                self._records = pickle.load(f)
            self._rebuild()

    def _save(self) -> None:
        with open(self._persist_path, "wb") as f:
            pickle.dump(self._records, f)

    def _rebuild(self) -> None:
        self._bm25 = BM25Plus([_tokenize(r["text"]) for r in self._records]) if self._records else None

    def add_document(self, document_id: str, filename: str, chunks: list[str]) -> None:
        for i, chunk in enumerate(chunks):
            self._records.append(
                {"document_id": document_id, "filename": filename, "chunk_index": i, "text": chunk}
            )
        self._rebuild()
        self._save()

    def remove_document(self, document_id: str) -> None:
        self._records = [r for r in self._records if r["document_id"] != document_id]
        self._rebuild()
        self._save()

    def search(self, query: str, top_k: int, document_id: str | None = None) -> list[dict]:
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(self._records)), key=lambda i: scores[i], reverse=True)

        hits = []
        for i in ranked:
            record = self._records[i]
            if document_id is not None and record["document_id"] != document_id:
                continue
            if scores[i] <= 0:
                # BM25 gives a 0 score to documents sharing no terms with
                # the query at all — not a "weak match", a NON-match. Same
                # reasoning as ChromaStore always returning top_k regardless
                # of relevance (see retrieval_service.py): don't let padding
                # with irrelevant hits masquerade as a real result.
                continue
            hits.append(
                {
                    "text": record["text"],
                    "metadata": {
                        "document_id": record["document_id"],
                        "filename": record["filename"],
                        "chunk_index": record["chunk_index"],
                    },
                    "score": round(float(scores[i]), 4),
                }
            )
            if len(hits) >= top_k:
                break
        return hits
