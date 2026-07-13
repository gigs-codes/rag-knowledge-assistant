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

    model_config = SettingsConfigDict(env_file=".env", env_prefix="EKA_")


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
