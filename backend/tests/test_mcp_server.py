"""
Integration test for the MCP server, driven as a REAL subprocess talking
JSON-RPC over its actual stdin/stdout — not a direct function call into
handle_message(). The protocol framing (one JSON object per line, request
IDs matching responses, notifications getting no reply) is exactly what
this needs to prove works; calling the dispatch function directly would
skip the part most likely to have a bug.

This uses the app's REAL vector store (same as the running app would),
not an isolated fixture — a real client connecting to this server should
see the real knowledge base. Because we don't control what's actually in
that store at test time (whatever the user has uploaded), assertions
check response SHAPE, not specific retrieved content.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def mcp_process():
    # Module-scoped deliberately: a real MCP client opens ONE connection
    # and sends many requests over it (that's the whole point of a
    # long-lived subprocess transport) — spawning a fresh process per test
    # would both misrepresent that and pay the embedding-model-load cost
    # (plus, as observed once during development, a slow one-time Chroma
    # tombstone cleanup after heavy add/delete churn) on every single test.
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        cwd=str(BACKEND_DIR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        # NOT subprocess.PIPE: a pipe has a small, fixed OS buffer, and
        # nothing in this fixture ever reads it mid-test. If the child logs
        # enough there (as chromadb's HNSW cleanup did once, thousands of
        # lines, before app/vectorstore/chroma_store.py capped its log
        # level), the child blocks writing to a full, undrained pipe —
        # and this test, waiting on stdout, blocks right along with it.
        # Discarding it here removes that failure mode entirely; we don't
        # need stderr content for these tests.
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    yield proc
    proc.terminate()
    proc.wait(timeout=10)


def _send(proc, message: dict) -> None:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def _recv(proc) -> dict:
    line = proc.stdout.readline()
    assert line, "no response from server (process may have exited — check its exit code)"
    return json.loads(line)


def test_initialize_handshake(mcp_process):
    _send(mcp_process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    response = _recv(mcp_process)
    assert response["id"] == 1
    assert response["result"]["serverInfo"]["name"] == "enterprise-knowledge-assistant"
    assert "tools" in response["result"]["capabilities"]


def test_initialized_notification_gets_no_response(mcp_process):
    # A notification has no "id" and must not produce a reply. We prove
    # that by sending it immediately followed by a real request and
    # checking the FIRST line we read back is that request's response,
    # not something echoed for the notification.
    _send(mcp_process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    _send(mcp_process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    response = _recv(mcp_process)
    assert response["id"] == 2


def test_tools_list_returns_search_documents_tool(mcp_process):
    _send(mcp_process, {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}})
    response = _recv(mcp_process)
    tools = response["result"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "search_documents"
    assert "query" in tools[0]["inputSchema"]["properties"]
    assert tools[0]["inputSchema"]["required"] == ["query"]


def test_tools_call_returns_well_formed_result(mcp_process):
    _send(
        mcp_process,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "search_documents", "arguments": {"query": "remote work policy"}},
        },
    )
    response = _recv(mcp_process)
    result = response["result"]
    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"
    assert isinstance(result["content"][0]["text"], str)
    assert len(result["content"][0]["text"]) > 0


def test_tools_call_unknown_tool_returns_error_result(mcp_process):
    _send(
        mcp_process,
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "not_a_real_tool", "arguments": {}},
        },
    )
    response = _recv(mcp_process)
    assert response["result"]["isError"] is True


def test_unknown_method_returns_json_rpc_error(mcp_process):
    _send(mcp_process, {"jsonrpc": "2.0", "id": 6, "method": "not/a/real/method", "params": {}})
    response = _recv(mcp_process)
    assert response["error"]["code"] == -32601
