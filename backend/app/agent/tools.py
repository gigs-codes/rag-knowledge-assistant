"""
Agent tools, defined as real LangChain `Tool` objects (via the `@tool`
decorator) — not because we use LangChain's native tool-calling here (we
don't; see graph.py for why), but because `Tool.name`/`Tool.description`
are exactly what we need to render the tool list into the ReAct prompt,
and `Tool.invoke()` gives a consistent call interface regardless of how
each tool is implemented underneath. Reuse the abstraction, skip the
native binding it's usually paired with.

Why a factory function (`build_tools`) instead of module-level tools: the
retriever tool needs a `RetrievalService` instance, and that instance is
constructed once at app startup (see api/deps.py) — a factory lets us
close over it without reaching for a global.
"""
import ast
import operator

from langchain_core.tools import BaseTool, tool

from app.services.retrieval_service import RetrievalService

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval_arithmetic(expression: str) -> float:
    """Evaluate a numeric expression using Python's `ast` module rather
    than `eval()` — the AST walker below only recognizes numeric literals
    and a fixed set of arithmetic operators, so there is no way to reach
    name lookups, attribute access, or function calls. `eval()` on
    LLM-controlled input would be a direct code-execution vulnerability;
    this restricted walker structurally cannot execute arbitrary code."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression near: {ast.dump(node)}")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree.body)


def build_tools(retrieval_service: RetrievalService) -> list[BaseTool]:
    @tool
    def search_documents(query: str) -> str:
        """Search the uploaded company documents for information relevant to the query.
        Use this for any question that might be answered by document content."""
        hits = retrieval_service.retrieve(query, top_k=3)
        if not hits:
            return "No relevant documents found."
        return "\n\n".join(
            f"(source: {hit['metadata']['filename']}) {hit['text']}" for hit in hits
        )

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.
        Supports + - * / ** and parentheses only. Use this instead of doing
        math yourself whenever a question requires a numeric calculation."""
        try:
            return str(_safe_eval_arithmetic(expression))
        except Exception as exc:
            return f"Error: could not evaluate '{expression}' ({exc})"

    return [search_documents, calculator]
