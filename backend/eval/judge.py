"""
LLM-as-judge scoring functions.

Why an LLM judges the LLM: faithfulness and answer relevancy aren't
checkable with a regex or exact-match — "the stipend is $500" and "remote
workers get a five-hundred dollar stipend" are both correct but share no
substring. The standard technique (what Ragas/DeepEval do internally) is
to ask a second LLM call to compare meaning, not text, and return a score.
We reuse the *same* `LLMProvider` interface the app already has — the
judge is just another call through OllamaProvider, not a new integration.

Known limitation, stated plainly: using phi3:mini (a small local model)
to judge phi3:mini's own answers is weaker than using a stronger judge
model (e.g. GPT-4) the way Ragas usually recommends — a small model is
less reliable at nuanced scoring. This is the honest trade-off of staying
100% free/local. If judge quality becomes the bottleneck, the fix is a
one-line change: pass a different LLMProvider instance as the judge.
"""
import json
import re

from app.llm.base import LLMProvider

FAITHFULNESS_PROMPT = """You are grading whether an AI-generated answer is faithful to the \
given source context — i.e. every factual claim in the answer is actually supported by the \
context, with no invented details.

Context:
{context}

Answer to grade:
{answer}

Respond with ONLY a JSON object, no other text: {{"score": <0.0 to 1.0>, "reasoning": "<one sentence>"}}
A score of 1.0 means every claim in the answer is fully supported by the context.
A score of 0.0 means the answer contains claims not supported by the context at all."""

ANSWER_RELEVANCY_PROMPT = """You are grading whether an AI-generated answer actually addresses \
the question asked, regardless of whether it's factually correct.

Question:
{question}

Answer to grade:
{answer}

Respond with ONLY a JSON object, no other text: {{"score": <0.0 to 1.0>, "reasoning": "<one sentence>"}}
A score of 1.0 means the answer directly and completely addresses the question.
A score of 0.0 means the answer is off-topic or doesn't address the question at all."""

REFUSAL_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "cannot answer",
    "can't answer",
    "not contained in the context",
    "no relevant context",
    "i don't know",
]


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def _parse_score(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return {
                "score": max(0.0, min(1.0, float(parsed.get("score", 0.0)))),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Fallback for a judge model that didn't return clean JSON: grab the
    # first standalone decimal between 0 and 1 in the raw text.
    number_match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", raw)
    score = float(number_match.group(1)) if number_match else 0.0
    return {"score": score, "reasoning": f"(unparsed judge output) {raw[:150]}"}


def score_faithfulness(judge: LLMProvider, context: str, answer: str) -> dict:
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    raw = judge.generate("You are a strict, precise grading assistant.", prompt)
    return _parse_score(raw)


def score_answer_relevancy(judge: LLMProvider, question: str, answer: str) -> dict:
    prompt = ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)
    raw = judge.generate("You are a strict, precise grading assistant.", prompt)
    return _parse_score(raw)
