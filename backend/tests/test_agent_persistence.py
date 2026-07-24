"""Integration test: agent conversation memory survives a rebuilt graph
pointed at the same SQLite checkpoint file — this is exactly what happens
across a real backend restart, since api/deps.py builds a fresh SqliteSaver
+ compiled graph at process startup, pointed at the same on-disk file
(data/agent_checkpoints.db). A MemorySaver (the in-process default) could
never demonstrate this — its whole state lives in RAM and is gone the
instant the process exits."""
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from app.agent.graph import build_agent_graph, run_agent
from app.llm.base import LLMProvider


class FinalAnswerLLM(LLMProvider):
    """Always answers directly on the first turn — never calls a tool —
    so this test doesn't need a real RetrievalService."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "Thought: I know this.\nFinal Answer: 42"

    def generate_stream(self, system_prompt: str, user_prompt: str):
        yield self.generate(system_prompt, user_prompt)


def _build_graph(db_path):
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return build_agent_graph(FinalAnswerLLM(), retrieval_service=None, checkpointer=checkpointer)


def test_agent_memory_persists_across_a_rebuilt_graph(tmp_path):
    db_path = tmp_path / "checkpoints.db"

    graph1 = _build_graph(db_path)
    run_agent(graph1, "My name is Alice.", thread_id="t1")

    # Simulate a backend restart: a brand new checkpointer and a brand new
    # compiled graph, pointed at the same on-disk file and the same
    # thread_id — nothing in-process carries over except the file.
    graph2 = _build_graph(db_path)
    run_agent(graph2, "What is my name?", thread_id="t1")

    state = graph2.get_state({"configurable": {"thread_id": "t1"}})
    history = state.values.get("conversation_history", [])

    assert len(history) == 2
    assert history[0]["question"] == "My name is Alice."
    assert history[1]["question"] == "What is my name?"


def test_different_thread_ids_do_not_share_memory(tmp_path):
    db_path = tmp_path / "checkpoints.db"
    graph = _build_graph(db_path)

    run_agent(graph, "My name is Alice.", thread_id="thread-a")
    run_agent(graph, "My name is Bob.", thread_id="thread-b")

    state_a = graph.get_state({"configurable": {"thread_id": "thread-a"}})
    state_b = graph.get_state({"configurable": {"thread_id": "thread-b"}})

    assert len(state_a.values.get("conversation_history", [])) == 1
    assert len(state_b.values.get("conversation_history", [])) == 1
    assert state_a.values["conversation_history"][0]["question"] == "My name is Alice."
    assert state_b.values["conversation_history"][0]["question"] == "My name is Bob."
