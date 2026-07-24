"""
Thumbs up/down on answers. Any authenticated user can submit feedback;
only admins can review the log — same role split as documents (everyone
can use the app, only admins see/manage the aggregate data).
"""
from fastapi import APIRouter, Depends

from app.api.deps import get_feedback_store
from app.api.security import get_current_user, require_role
from app.models.schemas import FeedbackOut, FeedbackRequest
from app.services.feedback_store import FeedbackStore

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackOut, dependencies=[Depends(get_current_user)])
def submit_feedback(request: FeedbackRequest, store: FeedbackStore = Depends(get_feedback_store)):
    return store.add(request.question, request.answer, request.rating, request.source)


@router.get("", response_model=list[FeedbackOut], dependencies=[Depends(require_role("admin"))])
def list_feedback(store: FeedbackStore = Depends(get_feedback_store)):
    return store.list()