"""
LangGraph ReAct agent.

Why hand-build the graph instead of `langgraph.prebuilt.create_react_agent`:
that helper binds tools to the model via `.bind_tools()`, which relies on
the model emitting OpenAI-style structured tool-call output. phi3:mini
(our free, local, 3.8B model) isn't reliably good at that — small models
trained mostly for chat, not function-calling, often emit malformed or
missing tool calls through that path. The classic ReAct pattern (Yao et
al., 2022) sidesteps this: the model just writes plain text in a
Thought/Action/Action Input format, which is nothing more than
"follow this text pattern" — something an instruction-following model of
any size can do noticeably more reliably than emitting a specific JSON
tool-call schema. The trade-off: we're now responsible for parsing that
text ourselves (see `_parse_step`), which is more fragile than a
schema-validated tool call IF the model breaks format. We handle that by
treating unparseable output as a final answer rather than looping forever.

Why LangGraph at all, then, if we're not using its tool-binding: LangGraph
gives us the explicit graph (nodes, edges, conditional routing) and —
more importantly — the checkpointer, which is what gives this agent
memory across turns without us hand-rolling session storage.
"""
import re
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agent.tools import build_tools
from app.llm.base import LLMProvider
from app.services.retrieval_service import RetrievalService

MAX_ITERATIONS = 5

REACT_SYSTEM_PROMPT = """You are an AI agent that answers questions by reasoning step by step \
and using tools when they're useful. You do not have to use a tool for every question — if you \
already know the answer or the question is conversational, you may answer directly.

Available tools:
{tool_descriptions}

Respond using EXACTLY one of these two formats, nothing else:

Thought: <your reasoning about what to do next>
Action: <one tool name from [{tool_names}]>
Action Input: <the input to give that tool>

OR, when you have enough information to answer the user:

Thought: <your final reasoning>
Final Answer: <the answer to the user's question>

Never write "Observation:" yourself — it will be provided to you after a tool runs. \
Output only ONE Thought, and either one Action+Action Input OR one Final Answer. Nothing after that."""


class AgentState(TypedDict, total=False):
    question: str
    scratchpad: str
    trace: list[str]
    iterations: int
    final_answer: str | None
    pending_tool: str | None
    pending_input: str | None
    conversation_history: list[dict]


def _parse_step(text: str) -> dict:
    # Bounded capture: stop at the first blank line or the next
    # Question:/Thought: marker, so a small model's tendency to keep
    # generating past its answer (observed with phi3:mini — it will
    # sometimes hallucinate an entirely new, unrelated question right
    # after answering) doesn't get swept into the "final answer" text.
    final_match = re.search(
        r"Final Answer:\s*(.+?)(?:\n\n|\nQuestion:|\nThought:|\Z)", text, re.DOTALL
    )
    action_match = re.search(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    input_match = re.search(r"Action Input:\s*(.+?)(?:\n(?=\S)|\n*$)", text, re.DOTALL)
    has_action = bool(action_match and input_match)

    if has_action and final_match:
        # Small-model artifact: it wrote both an Action and a Final Answer
        # in one turn instead of stopping after the Action. Trust whichever
        # came first in the text — if it's the Action, run the real tool
        # for a grounded Observation instead of the model's self-computed
        # (unverified) guess at what that tool would have returned.
        if action_match.start() < final_match.start():
            return {
                "type": "action",
                "tool": action_match.group(1).strip(),
                "input": input_match.group(1).strip(),
                # Everything up to the end of the matched Action Input line
                # only — NOT the full raw text. A small model will often
                # keep generating past this point (observed: inventing an
                # entirely new, unrelated question). If that runaway text
                # got written into the scratchpad, the next turn's prompt
                # would include it as if it were legitimate prior
                # reasoning, actively steering the model off the original
                # question. Truncating here is what keeps the loop on-task.
                "clean_text": text[: input_match.end()].strip(),
            }
        return {"type": "final", "answer": final_match.group(1).strip()}
    if final_match:
        return {"type": "final", "answer": final_match.group(1).strip()}
    if has_action:
        return {
            "type": "action",
            "tool": action_match.group(1).strip(),
            "input": input_match.group(1).strip(),
            "clean_text": text[: input_match.end()].strip(),
        }
    return {"type": "unparseable", "raw": text.strip()}


def _build_user_prompt(state: AgentState) -> str:
    history = state.get("conversation_history") or []
    parts = []
    if history:
        recent = history[-3:]
        history_text = "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in recent)
        parts.append(f"Previous conversation:\n{history_text}\n")
    parts.append(f"Question: {state['question']}")
    if state.get("scratchpad"):
        parts.append(state["scratchpad"])
        parts.append(
            "You now have the Observation(s) above for the ORIGINAL question. If they "
            "already contain enough information to answer it, you MUST respond now with "
            "'Thought: <brief reasoning>' followed by 'Final Answer: <answer>' — do NOT call "
            "the same tool again with the same input, and do NOT invent a new question."
        )
    return "\n".join(parts)


def _planner_node(state: AgentState, llm: LLMProvider, system_prompt: str) -> AgentState:
    raw = llm.generate(system_prompt, _build_user_prompt(state))
    step = _parse_step(raw)

    trace = state.get("trace", [])
    trace.append(raw.strip())
    iterations = state.get("iterations", 0) + 1

    updates: AgentState = {"trace": trace, "iterations": iterations}

    if step["type"] == "final":
        history = state.get("conversation_history") or []
        history = history + [{"question": state["question"], "answer": step["answer"]}]
        updates["final_answer"] = step["answer"]
        updates["conversation_history"] = history
        updates["pending_tool"] = None
    elif step["type"] == "action":
        updates["pending_tool"] = step["tool"]
        updates["pending_input"] = step["input"]
        updates["scratchpad"] = state.get("scratchpad", "") + f"\n{step['clean_text']}\n"
        trace[-1] = step["clean_text"]  # keep the trace clean too, same reasoning
    else:
        # Model didn't follow the format. Rather than loop forever hoping
        # it self-corrects, treat its raw text as the answer — a worse
        # answer beats a hung request.
        updates["final_answer"] = step["raw"]
        updates["pending_tool"] = None

    return updates


def _tool_node(state: AgentState, tools_by_name: dict) -> AgentState:
    tool_name = state.get("pending_tool")
    tool_input = state.get("pending_input", "")
    tool = tools_by_name.get(tool_name)

    if tool is None:
        observation = f"Error: no such tool '{tool_name}'. Available: {list(tools_by_name)}"
    else:
        observation = tool.invoke(tool_input)

    trace = state.get("trace", [])
    trace.append(f"Observation: {observation}")

    return {
        "trace": trace,
        "scratchpad": state.get("scratchpad", "") + f"Observation: {observation}\n",
        "pending_tool": None,
        "pending_input": None,
    }


def _route_after_planner(state: AgentState) -> str:
    if state.get("final_answer") is not None:
        return "end"
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return "end"
    if state.get("pending_tool") is not None:
        return "tool"
    return "end"


def build_agent_graph(llm: LLMProvider, retrieval_service: RetrievalService):
    tools = build_tools(retrieval_service)
    tools_by_name = {t.name: t for t in tools}
    tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    tool_names = ", ".join(tools_by_name)
    system_prompt = REACT_SYSTEM_PROMPT.format(
        tool_descriptions=tool_descriptions, tool_names=tool_names
    )

    graph = StateGraph(AgentState)
    graph.add_node("planner", lambda state: _planner_node(state, llm, system_prompt))
    graph.add_node("tool", lambda state: _tool_node(state, tools_by_name))
    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", _route_after_planner, {"tool": "tool", "end": END})
    graph.add_edge("tool", "planner")

    return graph.compile(checkpointer=MemorySaver())


def run_agent(compiled_graph, question: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = compiled_graph.invoke(
        {
            "question": question,
            "scratchpad": "",
            "trace": [],
            "iterations": 0,
            "final_answer": None,
            "pending_tool": None,
            "pending_input": None,
        },
        config=config,
    )
    answer = result.get("final_answer") or "I wasn't able to reach a final answer."
    if result.get("iterations", 0) >= MAX_ITERATIONS and result.get("final_answer") is None:
        answer = "I wasn't able to reach a final answer within the allowed reasoning steps."
    return {"answer": answer, "trace": result.get("trace", [])}
