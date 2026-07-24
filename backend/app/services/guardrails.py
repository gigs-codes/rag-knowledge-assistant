"""
Lightweight, regex-based guardrails — deliberately not a heavy framework
(e.g. NeMo Guardrails, Presidio): same "explainable, dependency-light"
philosophy as mcp_server.py and eval/judge.py. These are coarse backstops,
not a substitute for not uploading sensitive documents in the first place.

redact_pii: applied at ingestion time, before text is chunked/embedded —
scrubbing after the fact would mean the original PII already made it into
the vector store and is unrecoverable without re-ingesting.

detect_prompt_injection: applied to retrieved chunks before they're fed to
the LLM as context — a malicious/compromised document could contain text
like "ignore previous instructions" aimed at hijacking the assistant's
behavior via its own retrieved context (indirect prompt injection). This
only logs a warning (doesn't block) — the grounding system prompt
(chat_service.SYSTEM_PROMPT) remains the primary defense, same relationship
the relevance-score filter has to retrieval quality: a coarse signal, not
a precise classifier.
"""
import re

from app.core.logging import get_logger

logger = get_logger(__name__)

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |the )?(previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"disregard (all |the )?(previous|prior|above) (instructions|rules)", re.IGNORECASE),
    re.compile(r"you are now (in |a )?(developer|admin|jailbreak|dan)\s*mode", re.IGNORECASE),
    re.compile(r"reveal (your |the )?(system prompt|instructions)", re.IGNORECASE),
]


def redact_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _SSN_RE.sub("[REDACTED_SSN]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def detect_prompt_injection(text: str) -> bool:
    hit = any(pattern.search(text) for pattern in _INJECTION_PATTERNS)
    if hit:
        logger.warning("Possible prompt injection detected in retrieved/uploaded content.")
    return hit
