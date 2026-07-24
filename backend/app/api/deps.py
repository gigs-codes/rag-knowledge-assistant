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
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from app.agent.graph import build_agent_graph
from app.core.config import settings
from app.llm.base import LLMProvider
from app.llm.ollama_provider import OllamaProvider
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.document_registry import DocumentRegistry
from app.services.document_registry_base import DocumentRegistryBase
from app.services.embedding_service import EmbeddingService
from app.services.feedback_store import FeedbackStore
from app.services.ingestion_service import IngestionService
from app.services.reranker_service import RerankerService
from app.services.retrieval_service import RetrievalService
from app.services.user_store import UserStore
from app.vectorstore.bm25_index import BM25Index
from app.vectorstore.chroma_store import ChromaStore

_embedding_service = EmbeddingService()
_vector_store = ChromaStore()
_bm25_index = BM25Index()
_reranker_service = RerankerService()

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

_ingestion_service = IngestionService(_embedding_service, _vector_store, _registry, _bm25_index)
_retrieval_service = RetrievalService(_embedding_service, _vector_store, _bm25_index, _reranker_service)
_chat_service = ChatService(_retrieval_service, _llm_provider)

# SQLite always, regardless of EKA_DATABASE_URL — user accounts are a
# separate concern from the document registry and don't need the
# Postgres upgrade path (see user_store.py's docstring for why).
_user_store = UserStore(f"sqlite:///{settings.users_db_path}")
_auth_service = AuthService(_user_store)

_feedback_store = FeedbackStore(f"sqlite:///{settings.feedback_db_path}")

# check_same_thread=False: FastAPI's sync route handlers (and uvicorn's
# worker threadpool) can call into this from a different thread than the
# one that opened the connection — sqlite3's default same-thread check
# would otherwise raise. SqliteSaver itself doesn't do its own
# thread-safety beyond what the underlying connection provides, but this
# app's agent requests are one-at-a-time per thread_id in practice, so
# the shared connection is fine here.
_agent_checkpoint_conn = sqlite3.connect(str(settings.agent_checkpoints_db_path), check_same_thread=False)
_agent_checkpointer = SqliteSaver(_agent_checkpoint_conn)
_agent_checkpointer.setup()

# Compiled once at startup, same reasoning as everything else above: the
# graph and its tool bindings are stateless/reusable across requests, only
# the per-conversation checkpointed state (keyed by thread_id) varies —
# and now persists across restarts via _agent_checkpointer (see agent/graph.py).
_agent_graph = build_agent_graph(_llm_provider, _retrieval_service, _agent_checkpointer)


def get_ingestion_service() -> IngestionService:
    return _ingestion_service


def get_registry() -> DocumentRegistryBase:
    return _registry


def get_chat_service() -> ChatService:
    return _chat_service


def get_agent_graph():
    return _agent_graph


def get_user_store() -> UserStore:
    return _user_store


def get_auth_service() -> AuthService:
    return _auth_service


def get_feedback_store() -> FeedbackStore:
    return _feedback_store
