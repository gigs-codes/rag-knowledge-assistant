"""
Document ingestion pipeline: file bytes -> extracted text -> clean text ->
chunks -> vectors.

This is the core "document processing" flow from the spec:
extract -> clean -> chunk -> embed -> store metadata -> store vectors.

Why chunk at all instead of embedding whole documents: embedding models
have a limited context window and lose precision over long inputs, and an
LLM answering a question needs *focused* context, not an entire document.
We split into overlapping chunks so an answer near a chunk boundary isn't
cut off from its surrounding context (chunk_overlap).

Why RecursiveCharacterTextSplitter (LangChain): it tries to split on
paragraph breaks first, then sentences, then words — only falling back to
a hard character cut if nothing else fits. That keeps chunks semantically
coherent instead of slicing mid-sentence, which measurably improves
retrieval quality.

Multi-format support: extraction is dispatched by file extension via
`_EXTRACTORS` below. Everything after extraction (cleaning, chunking,
embedding, storing) is format-agnostic — adding a new format means adding
one function and one dict entry here, nothing else in the pipeline
changes. That's the same "isolate the part that varies" idea as the
LLMProvider/ChromaStore interfaces elsewhere in this codebase, just
applied to input formats instead of providers.
"""
import re
from pathlib import Path

import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.config import settings
from app.core.logging import get_logger
from app.services.document_registry_base import DocumentRegistryBase
from app.services.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore, new_document_id

logger = get_logger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    document = docx.Document(str(path))
    return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())


def _extract_plain_text(path: Path) -> str:
    # Covers .txt and .md alike — Markdown syntax (#, *, etc.) is left as
    # literal characters rather than parsed/stripped. That's a deliberate
    # simplification: the raw markup doesn't meaningfully hurt embedding
    # or retrieval quality at this scale, and stripping it would need a
    # markdown parser dependency for marginal benefit.
    return path.read_bytes().decode("utf-8", errors="replace")


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_plain_text,
    ".md": _extract_plain_text,
}

# Exposed for the route layer to do a cheap, fast upfront rejection (see
# api/routes/documents.py) before reading a potentially large file into
# memory — IngestionService.ingest_document() below is still the actual
# source of truth and re-checks this itself, so nothing breaks if a
# caller skips this and calls the service directly (as eval/run_eval.py
# and the tests do).
SUPPORTED_EXTENSIONS = frozenset(_EXTRACTORS)


def _clean_text(text: str) -> str:
    # Collapse repeated whitespace left over from extraction artifacts
    # (multiple spaces, stray form-feed/newline runs) without altering words.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class IngestionService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaStore,
        registry: DocumentRegistryBase,
    ):
        self._embeddings = embedding_service
        self._store = vector_store
        self._registry = registry

    def ingest_document(self, file_bytes: bytes, filename: str) -> dict:
        extension = Path(filename).suffix.lower()
        extractor = _EXTRACTORS.get(extension)
        if extractor is None:
            raise ValueError(
                f"Unsupported file type '{extension}'. Supported: {', '.join(_EXTRACTORS)}."
            )

        document_id = new_document_id()
        file_path = settings.upload_dir / f"{document_id}{extension}"
        file_path.write_bytes(file_bytes)

        raw_text = extractor(file_path)
        clean_text = _clean_text(raw_text)
        if not clean_text:
            raise ValueError(
                f"No extractable text found in '{filename}' (is it empty, or a scanned image?)."
            )

        chunks = _splitter.split_text(clean_text)
        logger.info("Ingesting %s: %d chunks", filename, len(chunks))

        vectors = self._embeddings.embed_texts(chunks)
        self._store.add_chunks(document_id, filename, chunks, vectors)

        return self._registry.add(document_id, filename, len(chunks))

    def delete_document(self, document_id: str) -> None:
        self._store.delete_document(document_id)
        self._registry.delete(document_id)
        # Glob rather than a fixed extension: the file on disk could be
        # .pdf/.docx/.txt/.md depending on what was originally uploaded,
        # and the registry doesn't track that separately from the
        # (human-facing) filename — matching by document_id prefix finds
        # whichever one it actually is.
        for path in settings.upload_dir.glob(f"{document_id}.*"):
            path.unlink(missing_ok=True)
