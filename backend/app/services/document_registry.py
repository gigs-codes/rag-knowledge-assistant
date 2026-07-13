"""
Document metadata registry.

Why JSON-backed and not the vector store: Chroma stores *chunk* vectors,
not document-level bookkeeping (original filename, upload time, chunk
count). We need something to answer "what documents exist?" without
scanning the vector index.

Why JSON and not Postgres for this MVP: zero setup cost, zero extra
service to run, and the access pattern is trivial (list/get/delete by id)
— exactly what the original spec calls out as "PostgreSQL for metadata
(optional, designed for future scaling)". The registry is written behind
a small class with the same shape a Postgres-backed repository would have
(add/get/list/delete), so swapping the storage engine later means
rewriting this one file, not the callers.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings


class DocumentRegistry:
    def __init__(self, path: Path = settings.base_dir / "data" / "documents.json"):
        self._path = path
        self._lock = threading.Lock()
        if not self._path.exists():
            self._path.write_text("{}")

    def _read(self) -> dict:
        return json.loads(self._path.read_text())

    def _write(self, data: dict) -> None:
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def add(self, document_id: str, filename: str, num_chunks: int) -> dict:
        with self._lock:
            data = self._read()
            record = {
                "id": document_id,
                "filename": filename,
                "num_chunks": num_chunks,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            data[document_id] = record
            self._write(data)
            return record

    def list(self) -> list[dict]:
        return list(self._read().values())

    def get(self, document_id: str) -> dict | None:
        return self._read().get(document_id)

    def delete(self, document_id: str) -> None:
        with self._lock:
            data = self._read()
            data.pop(document_id, None)
            self._write(data)
