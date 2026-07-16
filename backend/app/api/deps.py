"""
Dependency injection wiring.

Why this file exists: routes shouldn't construct their own services (that
would mean re-loading the embedding model on every request, and would
hardcode which LLM/vector-store implementation is used). Instead, routes
declare `service: X = Depends(get_x_service)` and FastAPI resolves it.

Singletons are created once at import time (module-level) because the
embedding model and Chroma client are expensive/stateful — they should
live for the lifetime of the process, not per-request. FastAPI's
`Depends()` then just hands back the same instance each time.

This single file is also *the* place you'd change to swap providers:
e.g. `llm_provider = OpenAIProvider()` instead of `OllamaProvider()`.
"""
from app.agent.graph import build_agent_graph
from app.core.config import settings
from app.llm.base import LLMProvider
from app.llm.ollama_provider import OllamaProvider
from app.services.chat_service import ChatService
from app.services.document_registry import DocumentRegistry
from app.services.document_registry_base import DocumentRegistryBase
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService
from app.vectorstore.chroma_store import ChromaStore

_embedding_service = EmbeddingService()
_vector_store = ChromaStore()

# The one place this choice is made: Postgres if EKA_DATABASE_URL is set,
# JSON otherwise. Both satisfy DocumentRegistryBase, so nothing downstream
# (IngestionService, the routes) needs to know or care which one it got —
# same pattern as swapping OllamaProvider for a future OpenAIProvider.
if settings.database_url:
    from app.services.postgres_document_registry import PostgresDocumentRegistry

    _registry: DocumentRegistryBase = PostgresDocumentRegistry(settings.database_url)
else:
    _registry: DocumentRegistryBase = DocumentRegistry()

_llm_provider: LLMProvider = OllamaProvider()

_ingestion_service = IngestionService(_embedding_service, _vector_store, _registry)
_retrieval_service = RetrievalService(_embedding_service, _vector_store)
_chat_service = ChatService(_retrieval_service, _llm_provider)

# Compiled once at startup, same reasoning as everything else above: the
# graph and its tool bindings are stateless/reusable across requests, only
# the per-conversation checkpointed state (keyed by thread_id) varies.
_agent_graph = build_agent_graph(_llm_provider, _retrieval_service)


def get_ingestion_service() -> IngestionService:
    return _ingestion_service


def get_registry() -> DocumentRegistryBase:
    return _registry


def get_chat_service() -> ChatService:
    return _chat_service


def get_agent_graph():
    return _agent_graph
