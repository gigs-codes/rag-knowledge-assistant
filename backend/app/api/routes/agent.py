"""
Agent endpoint: a ReAct agent that can choose to search documents,
do arithmetic, or answer directly — as opposed to /query, which always
retrieves then always generates. See app/agent/graph.py for why this is
a hand-built LangGraph ReAct loop rather than a prebuilt tool-calling agent.
"""
import time

from fastapi import APIRouter, Depends

from app.api.deps import get_agent_graph
from app.agent.graph import run_agent
from app.models.schemas import AgentQueryRequest, AgentQueryResponse

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/ask", response_model=AgentQueryResponse)
def ask_agent(request: AgentQueryRequest, graph=Depends(get_agent_graph)):
    start = time.perf_counter()
    result = run_agent(graph, request.question, request.thread_id)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return AgentQueryResponse(answer=result["answer"], trace=result["trace"], latency_ms=latency_ms)
