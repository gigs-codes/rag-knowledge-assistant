"""
MCP server, hand-built over stdio — JSON-RPC 2.0 messages, one per line,
read from stdin and written to stdout. No `mcp` SDK dependency.

Why hand-built instead of the official Python SDK: a real, current
version conflict — the SDK's HTTP/SSE transport now pulls in a Starlette
version incompatible with this project's pinned FastAPI. Rather than
fight dependency pins for a transport we don't even use (we only need
stdio), this implements the small set of JSON-RPC methods MCP actually
requires for a tool server, directly. Same philosophy as the eval harness
(hand-rolled instead of Ragas) and the agent (hand-rolled ReAct instead
of a prebuilt tool-calling agent): prefer an explainable, dependency-light
implementation of the real technique over fighting a heavy library for a
small local project. If the SDK's transport dependencies loosen later,
swapping to `mcp.server.fastmcp.FastMCP` is a like-for-like replacement —
the tool logic in `_call_tool` wouldn't change at all.

Why stdio, and what that means about this server's shape: stdio is the
standard MCP transport for local tools — the CLIENT (Claude Desktop,
another agent, a test harness) launches this script as a subprocess and
talks to it over its stdin/stdout. No network port, no auth, no CORS —
just a process that reads a line, does something, writes a line back.
That's why this file has none of the machinery the FastAPI app has.

What JSON-RPC 2.0 actually is, briefly: a convention for structuring
request/response messages so either side can tell requests from
notifications (a notification has no `id` and gets no reply), match a
response back to the request that triggered it (via `id`), and represent
errors uniformly (`{"code": ..., "message": ...}`). It says nothing about
transport — you could run the exact same message format over HTTP or
WebSockets instead of stdio. MCP chose it as its message format; this
file is a minimal, from-scratch implementation of that format for the
handful of methods a tool server needs: `initialize`, the
`notifications/initialized` handshake-complete signal, `tools/list`, and
`tools/call`.

Run directly for manual testing:
    python -m mcp_server.server
    (then paste a JSON-RPC message and press Enter)

See tests/test_mcp_server.py for an automated version of the same thing,
driven as a real subprocess.
"""
import json
import sys

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import RerankerService
from app.services.retrieval_service import RetrievalService
from app.vectorstore.bm25_index import BM25Index
from app.vectorstore.chroma_store import ChromaStore

_embedding_service = EmbeddingService()
_vector_store = ChromaStore(persist_dir=str(settings.chroma_dir))
_bm25_index = BM25Index()
_reranker_service = RerankerService()
_retrieval_service = RetrievalService(_embedding_service, _vector_store, _bm25_index, _reranker_service)

SERVER_INFO = {"name": "enterprise-knowledge-assistant", "version": "0.1.0"}
PROTOCOL_VERSION = "2024-11-05"

# Why this shape (name/description/inputSchema): this is exactly what an
# MCP client shows the LLM driving it, so the LLM can decide whether and
# how to call the tool — same purpose `Tool.description` serves for our
# own hand-built ReAct agent in app/agent/tools.py, just a different
# wire format (JSON Schema here, since that's what the protocol expects).
TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Search the enterprise knowledge base (uploaded company PDFs) for "
            "passages relevant to a query. Returns matched passages with their "
            "source filename, chunk index, and similarity score."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of passages to return.",
                    "default": 4,
                },
            },
            "required": ["query"],
        },
    }
]


def _call_tool(name: str, arguments: dict) -> dict:
    if name != "search_documents":
        return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}

    query = arguments.get("query", "")
    top_k = arguments.get("top_k", 4)
    hits = _retrieval_service.retrieve(query, top_k=top_k)

    if not hits:
        text = "No relevant documents found."
    else:
        text = "\n\n".join(
            f"[score={hit['score']}] (source: {hit['metadata']['filename']}, "
            f"chunk {hit['metadata']['chunk_index']})\n{hit['text']}"
            for hit in hits
        )
    return {"content": [{"type": "text", "text": text}], "isError": False}


def handle_message(message: dict) -> dict | None:
    """Pure dispatch function, deliberately separated from the stdin/stdout
    loop below — this is what tests/test_mcp_server.py could call directly
    for a fast unit test, though we instead test through the real
    subprocess/stdio path for the same reason run_eval.py goes through the
    real ingestion pipeline: the protocol framing is exactly what's being
    verified, not just the tool logic underneath it."""
    method = message.get("method")
    msg_id = message.get("id")

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        }
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    if method == "notifications/initialized":
        return None  # notification: no id was sent, so no response is sent back

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = message.get("params", {})
        result = _call_tool(params.get("name"), params.get("arguments", {}))
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    if msg_id is None:
        return None  # unrecognized notification — ignore rather than error

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle_message(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
