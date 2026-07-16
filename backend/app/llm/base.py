"""
LLM provider interface (the "port" in ports-and-adapters).

Why: services/chat_service.py should ask "generate an answer for this
prompt" without knowing or caring whether that's Ollama running locally,
OpenAI's API, or anything else. Any class that implements `generate()`
can be swapped in via app/api/deps.py with a one-line change — no
changes to business logic, routes, or tests that mock this interface.
"""
from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return a complete text response for the given prompts."""
        raise NotImplementedError

    @abstractmethod
    def generate_stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Yield the response incrementally, one text delta at a time —
        for the streaming /query/stream endpoint, so the user sees tokens
        arrive instead of waiting for the full 3-60s generation to finish
        before anything appears."""
        raise NotImplementedError
