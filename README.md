# Enterprise Knowledge Assistant

Upload company documents (PDF, DOCX, TXT, MD) and ask questions in natural
language. Answers are grounded in your documents, streamed token-by-token,
and cited back to the exact source chunk. Runs entirely locally and free —
no API keys required. Includes a LangGraph agent, an MCP tool server, an
LLM-as-judge evaluation harness, and a Pytest suite. No authentication —
open access, by design for this deployment (see "What's deliberately not
built" below).

## Architecture

```
frontend/  React + TypeScript + Tailwind
  src/
    api.ts            typed API client (incl. SSE streaming)
    components/         DocumentPanel, AskPanel

backend/
  app/
    api/routes/          HTTP layer (documents, query, agent)
    api/deps.py          dependency injection — swap providers here
    agent/               hand-built LangGraph ReAct agent + tools
    services/            business logic (ingestion, retrieval, chat orchestration)
    llm/                 LLM provider interface + Ollama adapter (streaming + non-streaming)
    vectorstore/         vector store interface + ChromaDB adapter
    core/                config + logging
    models/              Pydantic request/response schemas
  eval/                  LLM-as-judge evaluation harness (faithfulness + answer relevancy)
  mcp_server/            hand-built MCP server (JSON-RPC over stdio)
  tests/                 Pytest tests (unit + integration + subprocess)

data/
  uploads/          saved source files (gitignored)
  chroma/           persisted vector index (gitignored)
  documents.json    document metadata registry (JSON, or Postgres — see below)
```

Every module has a docstring explaining *why* it exists and *why* it was built
that way — treat them as the design-decision log, not just comments.

**Why layered like this:** routes never talk to Chroma, Ollama, or the
database directly — they go through `services`, which go through small
adapter interfaces (`LLMProvider`, `ChromaStore`, `DocumentRegistryBase`).
Swapping Ollama for OpenAI, Chroma for FAISS/pgvector, or JSON for Postgres
means changing `app/api/deps.py` and adding one new adapter file — not
touching routes, business logic, or tests.

## Stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Ollama, `phi3:mini` | Free, offline, no API key. Swappable via `LLMProvider`. |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) | Free, CPU-friendly, strong MTEB retrieval scores for its size. |
| Vector store | ChromaDB (persistent, local) | Zero-ops embedded vector DB, no separate service to run. |
| Metadata store | JSON (default) or Postgres | JSON needs zero setup; Postgres (`EKA_DATABASE_URL`) adds real concurrent-write safety once you need it. |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Splits on paragraph/sentence boundaries first, avoids mid-sentence cuts. |
| Agent | Hand-built LangGraph ReAct loop | `phi3:mini` isn't reliable at native tool-calling; plain-text Thought/Action/Observation works with any instruction-following model. |
| Eval | Hand-built LLM-as-judge (not Ragas) | Same technique Ragas uses internally, zero extra dependency surface. |
| MCP | Hand-built JSON-RPC/stdio server (not the `mcp` SDK) | The SDK's HTTP transport deps conflicted with our pinned FastAPI/Starlette. |
| Backend | FastAPI + Pydantic | Async, auto-validated request/response schemas, auto OpenAPI docs. |
| Frontend | React + TypeScript + Tailwind (Vite) | Fast dev loop, typed API contract via `src/api.ts`. |
| Testing | Pytest | Unit tests via mocked `LLMProvider`/`ChromaStore`, integration tests via FastAPI `TestClient`, MCP tests via a real subprocess. |
| Deployment | Docker + docker-compose | Backend + frontend containerized; Ollama stays on the host for GPU access. |

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

### Docker (backend + frontend; Ollama stays on the host)
```bash
docker compose up --build
```
Frontend: http://localhost:5173 · Backend: http://localhost:8000

### Tests
```bash
cd backend
pytest -q
```
Unit tests (mocked collaborators), FastAPI integration tests (`TestClient`
+ `dependency_overrides`), and MCP tests (real subprocess speaking
JSON-RPC over stdio). Postgres-specific tests skip automatically unless
`EKA_TEST_DATABASE_URL` points at a reachable database.

### Evaluation
```bash
cd backend
python -m eval.run_eval
```
Ingests a fixture document through the real pipeline, asks a golden set of
8 questions through the real `ChatService`, scores faithfulness and answer
relevancy with an LLM judge, prints a report, writes `eval/results.json`.

### MCP server
```bash
cd backend
python -m mcp_server.server
```
Speaks JSON-RPC 2.0 over stdio (`initialize`, `tools/list`, `tools/call`)
— point any MCP-compatible client at this command to expose document
search as a tool outside this app entirely.

## Using it

No login — the app opens straight to the document panel and chat.

1. **Upload** a document (PDF, DOCX, TXT, or MD). Wait for embedding to
   finish — the chunk count appears once it's searchable.
2. **Ask** a question in the chat panel — the answer streams in
   token-by-token, with citations appearing as soon as retrieval
   completes (before generation even starts). Optionally select a
   specific document first to scope retrieval to just that file.
3. **Expand a citation** to see the exact source chunk, filename, chunk
   index, and similarity score the model was shown.
4. Conversation history persists across page refreshes (via
   `localStorage`) — use "Clear conversation" to reset it.
5. **Delete** any document from the sidebar — irreversible, guarded only
   by a confirmation dialog, no login or role required.

## What's deliberately not built, and why

- **Authentication** — was built earlier (JWT + bcrypt, employee/admin
  roles, protecting upload/delete) and then deliberately removed at the
  user's request to keep this deployment open-access. The code pattern
  (a `Depends()`-based auth dependency wrapping specific routes, backed
  by a swappable interface) is straightforward to reintroduce if a future
  deployment needs it — it isn't gone because it was hard, it's gone
  because this deployment doesn't want a login wall.
- **Python code-execution agent tool** — giving an LLM agent arbitrary code
  execution is a real security risk (prompt injection → RCE) without heavy
  sandboxing, disproportionate to this project's scope. The calculator
  tool covers safe arithmetic via a restricted AST evaluator instead.
- **Web search agent tool** — would require an external, typically paid,
  rate-limited API, conflicting with the zero-cost constraint.
- **LangSmith tracing, token/cost tracking** — `latency_ms` is tracked
  throughout (`QueryResponse`, `AgentQueryResponse`), but there's no LLM
  provider metering here since Ollama is free/local; wiring in tracing is
  a natural next step if this moves to a hosted LLM.
- **Context precision/recall metrics** in the eval harness — meaningful
  measurement needs a multi-chunk golden document; the current fixture is
  short enough to fit in one chunk, so only faithfulness and answer
  relevancy (generation-quality metrics) are measured.

## Known limitations, stated plainly

- `phi3:mini` (3.8B, CPU) is the latency bottleneck — expect 3-60s per
  answer depending on whether the model is already warm, and it struggles
  with true multi-hop agent reasoning (e.g., "look up X, then compute Y
  from it" across two different tools in one trajectory) — observed and
  documented in `app/agent/graph.py`.
- The relevance-score filter on retrieval (`min_relevance_score`) is a
  coarse backstop, not a precise relevance classifier — bge-small's
  similarity floor for genuinely unrelated content still sits around
  0.4-0.55, so the LLM's own grounding instruction remains the real
  defense against answering off-topic.
- The JSON document registry uses a process-local lock — safe for one
  backend instance, not for multiple instances behind a load balancer.
  The Postgres migration (`EKA_DATABASE_URL`) exists for exactly that
  case; a real multi-instance deployment would need it turned on.
- With no authentication, every endpoint — including delete — is open to
  anyone who can reach the server. Fine for a local/trusted deployment;
  not something to expose on the open internet as-is.
