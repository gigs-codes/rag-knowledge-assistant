"""Query endpoints: natural-language question -> grounded answer + citations,
as a single JSON response (`POST /query`) or as a token stream
(`POST /query/stream`)."""
import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_chat_service
from app.api.security import get_current_user
from app.models.schemas import QueryRequest, QueryResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse, dependencies=[Depends(get_current_user)])
def ask_question(
    request: QueryRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    return chat_service.answer(request.question, document_id=request.document_id)


def _sse_format(events: Iterator[dict]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event)}\n\n"


@router.post("/stream", dependencies=[Depends(get_current_user)])
def ask_question_stream(
    request: QueryRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    # Server-Sent Events: each `data: <json>\n\n` line is one event the
    # browser's fetch stream reader can parse as it arrives, rather than
    # waiting for the whole response body — see api.ts's askQuestionStream
    # for the client side of this same framing. Event shape matches what
    # ChatService.answer_stream() yields: one "citations" event, then many
    # "token" events, then one "done" event carrying latency_ms.
    return StreamingResponse(
        _sse_format(chat_service.answer_stream(request.question, document_id=request.document_id)),
        media_type="text/event-stream",
    )
