# Enterprise Knowledge Assistant

Upload company PDFs, ask questions in natural language, get answers grounded
in your documents with citations back to the source chunks. Runs entirely
locally and free — no API keys required.

## What this is (and isn't) — read this first

This is an **MVP core slice** of a larger planned system, built to
demonstrate a real, working RAG pipeline end-to-end: **upload → extract →
chunk → embed → store → retrieve → generate → cite**. It was built under a
hard deadline, so scope was deliberately narrowed to the pipeline itself.

**Not yet built** (see [Roadmap](#roadmap) below for why and what's next):
authentication/RBAC, the LangGraph agent + tool layer, MCP integration, the
Ragas/DeepEval evaluation suite, a Pytest test suite, Docker packaging, and
Postgres-backed metadata (a JSON file stands in for now — see
`backend/app/services/document_registry.py` for why that's a reasonable
interim choice).

## Architecture

```
frontend/  React + TypeScript + Tailwind — upload UI, chat UI
backend/
  app/
    api/routes/     HTTP layer only (FastAPI routers)
    api/deps.py      dependency injection — swap providers here
    services/        business logic (ingestion, retrieval, chat orchestration)
    llm/             LLM provider interface + Ollama adapter
    vectorstore/     vector store interface + ChromaDB adapter
    core/            config + logging
    models/          Pydantic request/response schemas
data/
  uploads/    saved PDFs (gitignored)
  chroma/     persisted vector index (gitignored)
  documents.json   document metadata registry
```

Every module has a docstring explaining *why* it exists and *why* it was
built this way — that's intentional, treat them as the design-decision log.

**Why layered like this:** routes never talk to Chroma or Ollama directly —
they go through `services`, which go through small adapter interfaces
(`LLMProvider`, `ChromaStore`). Swapping Ollama for OpenAI, or Chroma for
FAISS/pgvector, means changing `app/api/deps.py` and adding one new adapter
file — not touching routes, business logic, or tests.

## Stack (all free / local)

| Layer | Choice | Why |
|---|---|---|
| LLM | Ollama, `phi3:mini` | Free, offline, no API key. Swappable via `LLMProvider`. |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) | Free, CPU-friendly, strong MTEB retrieval scores for its size. |
| Vector store | ChromaDB (persistent, local) | Zero-ops embedded vector DB, no separate service to run. |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Splits on paragraph/sentence boundaries first, avoids mid-sentence cuts. |
| Backend | FastAPI + Pydantic | Async, auto-validated request/response schemas, auto OpenAPI docs. |
| Frontend | React + TypeScript + Tailwind (Vite) | Fast dev loop, typed API contract via `src/api.ts`. |

## Running it

### Prerequisites
- Python 3.11+, Node 18+
- [Ollama](https://ollama.com) installed and running, with the model pulled:
  ```
  ollama pull phi3:mini
  ollama serve
  ```

### Backend
```bash
cd backend
python -m venv .venv
./.venv/Scripts/activate      # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

### Frontend
```bash
cd frontend
npm install
npm run dev
```
App: http://localhost:5173 (proxies `/api/*` to the backend on :8000)

## Using it

1. Upload a PDF (drag-and-drop area in the left panel). Wait for embedding
   to finish — you'll see the chunk count appear.
2. Ask a question in the chat panel. Optionally click a specific document
   in the sidebar first to scope retrieval to just that file.
3. Expand a citation under any answer to see the exact source chunk, its
   filename, chunk index, and similarity score that the model was shown.

## Roadmap

These were in the original spec and are the natural next phases, each
deserving its own design discussion before implementation (not bolted on
under deadline pressure):

- **Auth** — JWT-based login, employee/admin roles, protecting upload/delete
  behind role checks.
- **Agent layer** — LangGraph planner with retriever/calculator/Python tools,
  memory across a conversation.
- **MCP integration** — expose the retriever as an MCP tool server so any
  MCP-compatible client (not just this frontend) can use it.
- **Evaluation** — Ragas/DeepEval for faithfulness, answer relevance, context
  precision/recall; LangSmith tracing; latency and token/cost tracking (the
  scaffolding for latency is already in `QueryResponse.latency_ms`).
- **Testing** — Pytest unit tests per service (mock `LLMProvider`/
  `ChromaStore`), integration tests hitting the FastAPI test client, RAG
  quality tests using the eval suite above.
- **Deployment** — Dockerfiles per service + docker-compose, environment
  promotion (dev/staging/prod config via `Settings`).
- **Metadata storage** — migrate `document_registry.py` from JSON to
  Postgres once multi-instance/concurrent-write needs justify it.
