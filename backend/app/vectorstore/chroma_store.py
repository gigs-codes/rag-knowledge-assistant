"""
ChromaDB adapter — the vector store "port" implementation.

Why isolate this behind a small class instead of calling chromadb directly
from services: retrieval_service.py should only know "add these vectors"
and "give me the k nearest vectors to this one" — not Chroma's specific
collection API. Swapping to FAISS/pgvector/Pinecone later means rewriting
this one file, not every place that does retrieval.

Why we compute embeddings ourselves (via EmbeddingService) instead of
letting Chroma embed internally: keeps the embedding model choice explicit
and swappable independent of the vector store choice — two separate
decisions, two separate adapters.
"""
import uuid

import chromadb

from app.core.config import settings


class ChromaStore:
    def __init__(self, persist_dir: str = str(settings.chroma_dir)):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        ids = [f"{document_id}-{i}" for i in range(len(chunks))]
        metadatas = [
            {"document_id": document_id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        document_id: str | None = None,
    ) -> list[dict]:
        where = {"document_id": document_id} if document_id else None
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        if not results["ids"][0]:
            return []

        hits = []
        for i in range(len(results["ids"][0])):
            # Chroma returns cosine *distance* (0 = identical); convert to
            # an intuitive similarity score (1 = identical) for the API/UI.
            distance = results["distances"][0][i]
            hits.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": round(1 - distance, 4),
                }
            )
        return hits

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def count_chunks(self, document_id: str) -> int:
        return len(self._collection.get(where={"document_id": document_id})["ids"])


def new_document_id() -> str:
    return str(uuid.uuid4())
