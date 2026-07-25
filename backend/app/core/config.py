"""
Centralized application configuration.

Why this exists: every module that needs a setting (upload dir, model name,
chunk size...) imports `settings` from here instead of reading env vars
directly. That means:
  1. One place to see every configurable value in the system.
  2. Swapping environments (dev -> staging -> prod) is an env var change,
     not a code change.
  3. Testable: tests can override `Settings` without touching real env vars.
"""
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App ---
    app_name: str = "Enterprise Knowledge Assistant"
    log_level: str = "INFO"

    # --- Storage ---
    base_dir: Path = Path(__file__).resolve().parent.parent.parent.parent
    upload_dir: Path = base_dir / "data" / "uploads"
    chroma_dir: Path = base_dir / "data" / "chroma"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Embeddings ---
    # Local, free, no API key. Small (~130MB) and fast on CPU.
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- LLM (Ollama — local, free, no API key) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    # --- Retrieval ---
    top_k: int = 4
    # Cosine similarity floor below which a chunk is dropped rather than
    # shown as a "relevant" citation. Deliberately conservative (low) —
    # observed scores for genuinely UNRELATED content with bge-small still
    # land around 0.4-0.55 (embedding models have a real similarity floor
    # from shared language statistics alone), so a threshold precise
    # enough to reliably separate "relevant" from "not" would also risk
    # cutting real matches phrased differently from the question. This is
    # a coarse backstop against degenerate near-zero matches, not a
    # substitute for the LLM's own grounding instruction — that instruction
    # (see chat_service.py's SYSTEM_PROMPT) remains the primary defense
    # against answering from irrelevant context.
    min_relevance_score: float = 0.2

    # --- Metadata storage ---
    # None (default) means "use the JSON registry" — zero setup, works
    # immediately. Set EKA_DATABASE_URL (e.g.
    # postgresql://user:pass@host:5432/eka) to switch to Postgres instead;
    # see api/deps.py for where that choice is actually made.
    database_url: str | None = None

    # --- Auth ---
    # Dev-only default so the app still boots with zero setup; docker-compose.yml
    # already overrides this via EKA_JWT_SECRET_KEY for anything beyond local dev.
    jwt_secret_key: str = "dev-only-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480

    # Where the SQLite-backed auth/feedback/agent-memory stores live —
    # separate small files rather than one shared DB, so each concern
    # (users, feedback, agent checkpoints) can be inspected/reset independently.
    users_db_path: Path = base_dir / "data" / "users.db"
    feedback_db_path: Path = base_dir / "data" / "feedback.db"
    agent_checkpoints_db_path: Path = base_dir / "data" / "agent_checkpoints.db"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="EKA_")

    # Pydantic evaluates a field's default expression (e.g. `base_dir /
    # "data" / "uploads"` above) once, at class-definition time, using
    # whatever `base_dir` resolves to AT THAT LINE — the bare class
    # default, not the actual runtime value after EKA_BASE_DIR's env
    # override is applied. Every path derived from `base_dir` above was
    # silently ignoring EKA_BASE_DIR as a result (only `base_dir` itself
    # picked up the override; everything derived from it stayed pinned to
    # the class-body-time value). A model_validator running AFTER all
    # fields (including the real, possibly-overridden base_dir) are
    # resolved is the correct place to re-derive them.
    @model_validator(mode="after")
    def _rederive_paths_from_base_dir(self) -> "Settings":
        self.upload_dir = self.base_dir / "data" / "uploads"
        self.chroma_dir = self.base_dir / "data" / "chroma"
        self.users_db_path = self.base_dir / "data" / "users.db"
        self.feedback_db_path = self.base_dir / "data" / "feedback.db"
        self.agent_checkpoints_db_path = self.base_dir / "data" / "agent_checkpoints.db"
        return self


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
