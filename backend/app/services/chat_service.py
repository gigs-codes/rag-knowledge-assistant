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

from app.llm.base import LLMProvider
from app.models.schemas import Citation, QueryResponse
from app.services.retrieval_service import RetrievalService

SYSTEM_PROMPT = """You are an enterprise knowledge assistant. Answer the user's \
question using ONLY the numbered context sources below. \
Rules:
- If the answer is not contained in the context, say "I don't have enough \
information in the uploaded documents to answer that." Do not guess.
- When you use a fact from a source, reference it inline like [1], [2].
- Be concise and factual. Do not invent details not present in the context."""


def _build_user_prompt(question: str, hits: list[dict]) -> str:
    context_blocks = [
        f"[{i + 1}] (source: {hit['metadata']['filename']})\n{hit['text']}"
        for i, hit in enumerate(hits)
    ]
    context = "\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"
    return f"Context sources:\n{context}\n\nQuestion: {question}"


class ChatService:
    def __init__(self, retrieval_service: RetrievalService, llm_provider: LLMProvider):
        self._retrieval = retrieval_service
        self._llm = llm_provider

    def answer(
        self, question: str, document_id: str | None = None
    ) -> QueryResponse:
        start = time.perf_counter()

        hits = self._retrieval.retrieve(question, document_id=document_id)

        if not hits:
            answer_text = (
                "I don't have enough information in the uploaded documents "
                "to answer that. Try uploading a relevant document first."
            )
        else:
            user_prompt = _build_user_prompt(question, hits)
            answer_text = self._llm.generate(SYSTEM_PROMPT, user_prompt)

        citations = [
            Citation(
                document_id=hit["metadata"]["document_id"],
                filename=hit["metadata"]["filename"],
                chunk_index=hit["metadata"]["chunk_index"],
                text=hit["text"],
                score=hit["score"],
            )
            for hit in hits
        ]

        latency_ms = int((time.perf_counter() - start) * 1000)
        return QueryResponse(answer=answer_text, citations=citations, latency_ms=latency_ms)
