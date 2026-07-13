"""
Ollama adapter — implements LLMProvider against a local Ollama server.

Why Ollama for this build: it's free, runs fully offline, and needs no API
key, which matters given the zero-cost constraint. It talks to Ollama's
REST API (localhost:11434), which must be running with the configured
model pulled (`ollama pull phi3:mini`) before this will work.

To add OpenAI later: create `openai_provider.py` implementing the same
`LLMProvider` interface, then change one line in app/api/deps.py.
"""
import requests

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.base import LLMProvider

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = settings.ollama_base_url, model: str = settings.ollama_model):
        self.base_url = base_url
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError as exc:
            logger.error("Could not reach Ollama at %s — is `ollama serve` running?", self.base_url)
            raise RuntimeError(
                f"Ollama is not reachable at {self.base_url}. "
                f"Start it with `ollama serve` and ensure `{self.model}` is pulled."
            ) from exc
