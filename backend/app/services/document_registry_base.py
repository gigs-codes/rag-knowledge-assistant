"""
Document registry interface — the same "port" idea as LLMProvider
(app/llm/base.py) and the implicit ChromaStore contract, applied to
metadata storage. This formalizes what was previously just an informal
convention (DocumentRegistry's method shapes) into an actual ABC, because
now there are TWO implementations (JSON, Postgres) and callers
(ingestion_service.py, the routes) should depend on this interface, not
on which storage engine happens to be configured.
"""
from abc import ABC, abstractmethod


class DocumentRegistryBase(ABC):
    @abstractmethod
    def add(self, document_id: str, filename: str, num_chunks: int) -> dict:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def get(self, document_id: str) -> dict | None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, document_id: str) -> None:
        raise NotImplementedError
