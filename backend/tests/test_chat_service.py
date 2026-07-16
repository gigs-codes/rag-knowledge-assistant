"""
Unit tests for ChatService — the highest-value tests in this suite,
because they exercise the exact seam the codebase was designed around:
`ChatService` only depends on the `LLMProvider` interface, never on
Ollama specifically. `FakeLLM` below is a second, real implementation of
that interface (alongside `OllamaProvider`) written purely for tests —
if this weren't a genuine interface, it wouldn't be substitutable here
without changes to ChatService itself.
"""
from collections.abc import Iterator
from unittest.mock import MagicMock

from app.llm.base import LLMProvider
from app.services.chat_service import ChatService


class FakeLLM(LLMProvider):
    def __init__(self, response: str = "mock answer"):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response

    def generate_stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        self.calls.append((system_prompt, user_prompt))
        yield self.response


def _retrieval_returning(hits):
    retrieval = MagicMock()
    retrieval.retrieve.return_value = hits
    return retrieval


def test_answer_skips_the_llm_entirely_when_no_hits():
    """Hallucination-prevention isn't just the system prompt — when there's
    nothing to ground an answer in, we never even call the LLM."""
    retrieval = _retrieval_returning([])
    llm = FakeLLM()
    service = ChatService(retrieval, llm)

    response = service.answer("some unanswerable question")

    assert llm.calls == []
    assert "don't have enough information" in response.answer
    assert response.citations == []


def test_answer_builds_numbered_context_and_returns_citations():
    hits = [
        {
            "text": "Remote work is allowed 3 days a week.",
            "metadata": {"document_id": "d1", "filename": "policy.pdf", "chunk_index": 0},
            "score": 0.91,
        }
    ]
    retrieval = _retrieval_returning(hits)
    llm = FakeLLM(response="You can work remotely 3 days a week [1].")
    service = ChatService(retrieval, llm)

    response = service.answer("how many remote days are allowed?")

    assert len(llm.calls) == 1
    system_prompt, user_prompt = llm.calls[0]
    assert "ONLY the numbered context" in system_prompt
    assert "[1]" in user_prompt
    assert "Remote work is allowed 3 days a week." in user_prompt

    assert response.answer == "You can work remotely 3 days a week [1]."
    assert len(response.citations) == 1
    assert response.citations[0].filename == "policy.pdf"
    assert response.citations[0].score == 0.91


def test_answer_passes_document_id_through_to_retrieval():
    retrieval = _retrieval_returning([])
    service = ChatService(retrieval, FakeLLM())

    service.answer("question", document_id="doc-42")

    retrieval.retrieve.assert_called_once_with("question", document_id="doc-42")


def test_answer_always_reports_nonnegative_latency():
    service = ChatService(_retrieval_returning([]), FakeLLM())
    response = service.answer("question")
    assert response.latency_ms >= 0


class StreamingFakeLLM(LLMProvider):
    """A second FakeLLM whose generate_stream() yields multiple separate
    chunks (rather than one), to prove answer_stream() actually forwards
    each chunk as its own event instead of buffering them."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "".join(self.chunks)

    def generate_stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        yield from self.chunks


def test_answer_stream_yields_citations_before_tokens_then_done():
    hits = [
        {
            "text": "Remote work is allowed 3 days a week.",
            "metadata": {"document_id": "d1", "filename": "policy.pdf", "chunk_index": 0},
            "score": 0.91,
        }
    ]
    service = ChatService(_retrieval_returning(hits), StreamingFakeLLM(["You can ", "work remotely."]))

    events = list(service.answer_stream("how many remote days?"))

    assert events[0]["type"] == "citations"
    assert events[0]["citations"][0]["filename"] == "policy.pdf"
    assert [e["text"] for e in events[1:-1]] == ["You can ", "work remotely."]
    assert events[-1]["type"] == "done"
    assert events[-1]["latency_ms"] >= 0


def test_answer_stream_yields_refusal_without_calling_llm_when_no_hits():
    llm = StreamingFakeLLM(["should not be used"])
    service = ChatService(_retrieval_returning([]), llm)

    events = list(service.answer_stream("unanswerable question"))

    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 1
    assert "don't have enough information" in token_events[0]["text"]
