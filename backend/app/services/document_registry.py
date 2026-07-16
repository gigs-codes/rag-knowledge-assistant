"""
Document metadata registry — JSON-backed implementation of
DocumentRegistryBase.

Why JSON-backed and not the vector store: Chroma stores *chunk* vectors,
not document-level bookkeeping (original filename, upload time, chunk
count). We need something to answer "what documents exist?" without
scanning the vector index.

Why this is still here at all now that Postgres support exists
(postgres_document_registry.py): zero setup cost, zero extra service to
run — this remains the default (see api/deps.py) precisely so the app
keeps working out of the box with no database to install, and Postgres
becomes an opt-in upgrade via EKA_DATABASE_URL once someone actually
needs multi-instance/concurrent-write guarantees a single JSON file can't
give them.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.services.document_registry_base import DocumentRegistryBase


class DocumentRegistry(DocumentRegistryBase):
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
