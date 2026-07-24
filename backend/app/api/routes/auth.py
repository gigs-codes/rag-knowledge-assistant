"""
Registration, login, and "who am I" endpoints. See auth_service.py for the
role-bootstrap rule (first registered account becomes admin).
"""
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_auth_service
from app.api.security import get_current_user
from app.models.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut
from app.services.auth_service import AuthError, AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
def register(request: RegisterRequest, auth_service: AuthService = Depends(get_auth_service)):
    try:
        user = auth_service.register(request.username, request.password)
    except AuthError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    token = auth_service.create_access_token(user)
    return TokenResponse(access_token=token, user=user)


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, auth_service: AuthService = Depends(get_auth_service)):
    try:
        user = auth_service.authenticate(request.username, request.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    token = auth_service.create_access_token(user)
    return TokenResponse(access_token=token, user=user)


@router.get("/me", response_model=UserOut)
def me(user: UserOut = Depends(get_current_user)):
    return user
