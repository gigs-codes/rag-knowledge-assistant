"""
Document ingestion pipeline: PDF bytes -> clean text -> chunks -> vectors.

This is the core "document processing" flow from the spec:
extract -> clean -> chunk -> embed -> store metadata -> store vectors.

Why chunk at all instead of embedding whole documents: embedding models
have a limited context window and lose precision over long inputs, and an
LLM answering a question needs *focused* context, not an entire PDF. We
split into overlapping chunks so an answer near a chunk boundary isn't cut
off from its surrounding context (chunk_overlap).

Why RecursiveCharacterTextSplitter (LangChain): it tries to split on
paragraph breaks first, then sentences, then words — only falling back to
a hard character cut if nothing else fits. That keeps chunks semantically
coherent instead of slicing mid-sentence, which measurably improves
retrieval quality.
"""
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.config import settings
from app.core.logging import get_logger
from app.services.document_registry import DocumentRegistry
from app.services.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore, new_document_id

logger = get_logger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _clean_text(text: str) -> str:
    # Collapse repeated whitespace left over from PDF extraction artifacts
    # (multiple spaces, stray form-feed/newline runs) without altering words.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class IngestionService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaStore,
        registry: DocumentRegistry,
    ):
        self._embeddings = embedding_service
        self._store = vector_store
        self._registry = registry

    def ingest_pdf(self, file_bytes: bytes, filename: str) -> dict:
        document_id = new_document_id()
        pdf_path = settings.upload_dir / f"{document_id}.pdf"
        pdf_path.write_bytes(file_bytes)

        raw_text = _extract_text(pdf_path)
        clean_text = _clean_text(raw_text)
        if not clean_text:
            raise ValueError("No extractable text found in PDF (is it a scanned image?).")

        chunks = _splitter.split_text(clean_text)
        logger.info("Ingesting %s: %d chunks", filename, len(chunks))

        vectors = self._embeddings.embed_texts(chunks)
        self._store.add_chunks(document_id, filename, chunks, vectors)

        return self._registry.add(document_id, filename, len(chunks))

    def delete_document(self, document_id: str) -> None:
        self._store.delete_document(document_id)
        self._registry.delete(document_id)
        pdf_path = settings.upload_dir / f"{document_id}.pdf"
        pdf_path.unlink(missing_ok=True)
