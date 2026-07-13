"""
LLM provider interface (the "port" in ports-and-adapters).

Why: services/chat_service.py should ask "generate an answer for this
prompt" without knowing or caring whether that's Ollama running locally,
OpenAI's API, or anything else. Any class that implements `generate()`
can be swapped in via app/api/deps.py with a one-line change — no
changes to business logic, routes, or tests that mock this interface.
"""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return a complete text response for the given prompts."""
        raise NotImplementedError
