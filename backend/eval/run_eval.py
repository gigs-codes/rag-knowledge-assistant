"""
Evaluation runner: ingest the fixture document through the REAL pipeline
(not a shortcut), ask every golden question through the REAL ChatService,
score each answer with an LLM judge, print a report.

Run from backend/ with the venv active:
    python -m eval.run_eval

Why this goes through the actual ingestion_service.ingest_document() and
ChatService.answer() — the same code paths the API uses — instead of
calling internal pieces directly: an eval harness that bypasses the code
it's supposed to be evaluating would give you false confidence. This is
as close to "hit the real app" as you can get without spinning up uvicorn.

Why separate Chroma/registry paths instead of the app's real ones (see
`_build_isolated_services` below): the eval fixture document is not
something the user uploaded — it shouldn't appear in the real document
list or get mixed into real retrieval. Same reasoning you'd use for a
test database vs. a production database. One known gap: `IngestionService`
still writes the fixture's raw PDF bytes into the shared `data/uploads/`
dir, because `upload_dir` comes from global settings rather than being
injected per-instance — harmless (never referenced by the real registry)
but a clean follow-up would thread storage paths through the constructor
the same way we did for the vector store and registry.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.llm.ollama_provider import OllamaProvider  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.document_registry import DocumentRegistry  # noqa: E402
from app.services.embedding_service import EmbeddingService  # noqa: E402
from app.services.ingestion_service import IngestionService  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402
from app.vectorstore.chroma_store import ChromaStore  # noqa: E402
from eval.golden_set import FIXTURE_DOCUMENT_TEXT, FIXTURE_FILENAME, GOLDEN_SET  # noqa: E402
from eval.judge import is_refusal, score_answer_relevancy, score_faithfulness  # noqa: E402


def _build_isolated_services():
    embedding_service = EmbeddingService()
    vector_store = ChromaStore(persist_dir=str(settings.base_dir / "data" / "eval_chroma"))
    registry = DocumentRegistry(path=settings.base_dir / "data" / "eval_documents.json")
    llm_provider = OllamaProvider()

    ingestion_service = IngestionService(embedding_service, vector_store, registry)
    retrieval_service = RetrievalService(embedding_service, vector_store)
    chat_service = ChatService(retrieval_service, llm_provider)
    return ingestion_service, registry, chat_service, llm_provider


def _build_fixture_pdf_bytes(text: str) -> bytes:
    """Hand-build a minimal valid single-page PDF containing `text`, so the
    eval fixture goes through the real PDF-extraction code path rather than
    injecting pre-chunked text. No external PDF-writing library needed for
    a single page of plain text."""
    lines = text.split("\n")
    content_lines = ["BT", "/F1 12 Tf", "72 750 Td", "14 TL"]
    for i, line in enumerate(lines):
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content_lines.append(("" if i == 0 else "T*"))
        content_lines.append(f"({escaped}) Tj")
    content_lines.append("ET")
    content_stream = "\n".join(l for l in content_lines if l).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (f"<< /Length {len(content_stream)} >>\nstream\n".encode("latin-1")
         + content_stream + b"\nendstream"),
    ]
    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n".encode("latin-1") + obj + b"\nendobj\n"
    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    pdf += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n".encode("latin-1")
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF"
    ).encode("latin-1")
    return pdf


def _ensure_fixture_ingested(ingestion_service: IngestionService, registry: DocumentRegistry) -> str:
    for doc in registry.list():
        if doc["filename"] == FIXTURE_FILENAME:
            return doc["id"]
    record = ingestion_service.ingest_document(_build_fixture_pdf_bytes(FIXTURE_DOCUMENT_TEXT), FIXTURE_FILENAME)
    return record["id"]


def run() -> dict:
    ingestion_service, registry, chat_service, judge = _build_isolated_services()
    fixture_doc_id = _ensure_fixture_ingested(ingestion_service, registry)

    results = []
    for item in GOLDEN_SET:
        response = chat_service.answer(item["question"], document_id=fixture_doc_id)
        refused = is_refusal(response.answer)
        context = "\n\n".join(c.text for c in response.citations)

        row = {
            "question": item["question"],
            "expect_answerable": item["expect_answerable"],
            "answer": response.answer,
            "refused": refused,
            "latency_ms": response.latency_ms,
        }

        if item["expect_answerable"]:
            row["grounding_correct"] = not refused
            if not refused:
                row["faithfulness"] = score_faithfulness(judge, context, response.answer)
                row["answer_relevancy"] = score_answer_relevancy(judge, item["question"], response.answer)
        else:
            # For a question with no answer in the document, "correct" means
            # the model refused rather than hallucinated — that IS the metric.
            row["grounding_correct"] = refused

        results.append(row)
        status = "OK" if row["grounding_correct"] else "FAIL"
        print(f"[{status}] {item['question']}")

    answerable = [r for r in results if r["expect_answerable"] and not r["refused"]]
    refusal_rows = [r for r in results if not r["expect_answerable"]]
    grounding_rows = [r for r in results]

    summary = {
        "total_questions": len(results),
        "grounding_accuracy": sum(r["grounding_correct"] for r in grounding_rows) / len(grounding_rows),
        "avg_faithfulness": (
            sum(r["faithfulness"]["score"] for r in answerable) / len(answerable) if answerable else None
        ),
        "avg_answer_relevancy": (
            sum(r["answer_relevancy"]["score"] for r in answerable) / len(answerable) if answerable else None
        ),
        "refusal_accuracy": (
            sum(r["grounding_correct"] for r in refusal_rows) / len(refusal_rows) if refusal_rows else None
        ),
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results),
    }

    print("\n--- Summary ---")
    for key, value in summary.items():
        print(f"{key}: {value}")

    output = {"summary": summary, "results": results}
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nFull results written to {out_path}")
    return output


if __name__ == "__main__":
    run()
