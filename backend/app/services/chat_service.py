"""
Chat service: orchestrates retrieval -> prompt construction -> generation.

This is where "prompt engineering" and "hallucination prevention" from the
spec actually live. Two concrete techniques used here:

1. Context injection with numbered sources: retrieved chunks are labeled
   [1], [2], ... in the prompt so the model can (and is instructed to)
   reference them, and so citations returned to the client map 1:1 to
   what the model actually saw.

2. Explicit grounding instruction: the system prompt tells the model to
   answer ONLY from the provided context and to say so plainly if the
   context doesn't contain the answer, rather than guessing. This doesn't
   make hallucination impossible (nothing does, with a small local model
   like phi3:mini especially) — it materially reduces it, and is the
   standard first line of defense before reaching for a separate
   fact-checking/eval pass (see the deferred Evaluation phase).

Design note on citations: rather than parsing which [n] markers the model
actually used in its text (fragile — models are inconsistent about
citing), we return *all* chunks that were retrieved and fed into the
prompt as citations. This is a deliberate precision/recall trade-off:
we may show a source the model didn't end up using, but we never hide
a source that influenced the answer.
"""
import time
from collections.abc import Iterator

from app.core.config import settings
from app.llm.base import LLMProvider
from app.models.schemas import Citation, QueryResponse
from app.services.guardrails import detect_prompt_injection
from app.services.retrieval_service import RetrievalService

SYSTEM_PROMPT = """You are an enterprise knowledge assistant. Answer the user's \
question using ONLY the numbered context sources below. \
Rules:
- If the answer is not contained in the context, say "I don't have enough \
information in the uploaded documents to answer that." Do not guess.
- When you use a fact from a source, reference it inline like [1], [2].
- Be concise and factual. Do not invent details not present in the context."""

DECOMPOSITION_SYSTEM_PROMPT = "You split compound questions into simple standalone sub-questions."

_DECOMPOSITION_USER_PROMPT = """Split the question below into 2-3 standalone sub-questions that \
together cover everything it's asking, one per line, no numbering or extra commentary. If it's \
already a single simple question, just repeat it unchanged on one line.

Question: {question}"""


_NO_INFO_MESSAGE = (
    "I don't have enough information in the uploaded documents "
    "to answer that. Try uploading a relevant document first."
)


def _looks_compound(question: str) -> bool:
    # Cheap heuristic gate, deliberately conservative: only pay for the
    # extra LLM call (decomposition) when the question actually looks like
    # it's asking more than one thing. The agent already gets multi-step
    # decomposition for free via its ReAct loop's repeated tool calls (see
    # agent/graph.py) — this only benefits the plain, single-shot /query
    # endpoint.
    return question.count("?") > 1 or " and " in question.lower()


def _dedupe_key(hit: dict) -> tuple:
    return (hit["metadata"]["document_id"], hit["metadata"]["chunk_index"])


def _build_user_prompt(question: str, hits: list[dict]) -> str:
    context_blocks = [
        f"[{i + 1}] (source: {hit['metadata']['filename']})\n{hit['text']}"
        for i, hit in enumerate(hits)
    ]
    context = "\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"
    return f"Context sources:\n{context}\n\nQuestion: {question}"


def _build_citations(hits: list[dict]) -> list[Citation]:
    return [
        Citation(
            document_id=hit["metadata"]["document_id"],
            filename=hit["metadata"]["filename"],
            chunk_index=hit["metadata"]["chunk_index"],
            text=hit["text"],
            score=hit["score"],
        )
        for hit in hits
    ]


class ChatService:
    def __init__(self, retrieval_service: RetrievalService, llm_provider: LLMProvider):
        self._retrieval = retrieval_service
        self._llm = llm_provider

    def _decompose(self, question: str) -> list[str]:
        raw = self._llm.generate(
            DECOMPOSITION_SYSTEM_PROMPT, _DECOMPOSITION_USER_PROMPT.format(question=question)
        )
        sub_questions = [line.strip("-•*0123456789. \t") for line in raw.strip().splitlines()]
        sub_questions = [q for q in sub_questions if q]
        return sub_questions or [question]

    def _retrieve_hits(self, question: str, document_id: str | None) -> list[dict]:
        hits = self._retrieve_hits_uncheck(question, document_id)
        # Flag (not block — see guardrails.py's docstring) retrieved
        # content that looks like it's trying to hijack the assistant's
        # instructions via indirect prompt injection.
        for hit in hits:
            detect_prompt_injection(hit["text"])
        return hits

    def _retrieve_hits_uncheck(self, question: str, document_id: str | None) -> list[dict]:
        if not _looks_compound(question):
            return self._retrieval.retrieve(question, document_id=document_id)

        sub_questions = self._decompose(question)
        if len(sub_questions) <= 1:
            return self._retrieval.retrieve(question, document_id=document_id)

        merged: list[dict] = []
        seen: set[tuple] = set()
        for sub_question in sub_questions:
            for hit in self._retrieval.retrieve(sub_question, document_id=document_id):
                key = _dedupe_key(hit)
                if key not in seen:
                    seen.add(key)
                    merged.append(hit)
        # Cap total context size: merging per-sub-question hits can produce
        # more chunks than a single retrieval call would, and phi3:mini's
        # context window (and answer quality) degrades with too much
        # unfocused context stuffed into one prompt.
        return merged[: settings.top_k * 2]

    def answer(
        self, question: str, document_id: str | None = None
    ) -> QueryResponse:
        start = time.perf_counter()

        hits = self._retrieve_hits(question, document_id)

        if not hits:
            answer_text = _NO_INFO_MESSAGE
        else:
            user_prompt = _build_user_prompt(question, hits)
            answer_text = self._llm.generate(SYSTEM_PROMPT, user_prompt)

        latency_ms = int((time.perf_counter() - start) * 1000)
        return QueryResponse(
            answer=answer_text, citations=_build_citations(hits), latency_ms=latency_ms
        )

    def answer_stream(self, question: str, document_id: str | None = None) -> Iterator[dict]:
        """Same retrieval + grounding logic as answer(), but yields
        incremental events instead of building one final QueryResponse —
        for the /query/stream endpoint. Citations are known and sent
        BEFORE any token, since they come from retrieval (which happens
        before generation starts either way); the client can render them
        immediately rather than waiting for the full answer."""
        start = time.perf_counter()

        hits = self._retrieve_hits(question, document_id)
        yield {"type": "citations", "citations": [c.model_dump(mode="json") for c in _build_citations(hits)]}

        if not hits:
            yield {"type": "token", "text": _NO_INFO_MESSAGE}
        else:
            user_prompt = _build_user_prompt(question, hits)
            for token in self._llm.generate_stream(SYSTEM_PROMPT, user_prompt):
                yield {"type": "token", "text": token}

        latency_ms = int((time.perf_counter() - start) * 1000)
        yield {"type": "done", "latency_ms": latency_ms}
