"""Query endpoint: natural-language question -> grounded answer + citations."""
from fastapi import APIRouter, Depends

from app.api.deps import get_chat_service
from app.models.schemas import QueryRequest, QueryResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def ask_question(
    request: QueryRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    return chat_service.answer(request.question, document_id=request.document_id)
