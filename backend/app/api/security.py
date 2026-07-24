"""
Auth dependencies: who is making this request, and are they allowed to.

Same DI shape as the rest of api/deps.py — routes declare
`Depends(get_current_user)` and FastAPI resolves it, rather than each
route parsing the Authorization header itself. Split into its own module
(not deps.py) because these are request-scoped (depend on the incoming
Authorization header) rather than process-lifetime singletons, which is
what everything in deps.py is.
"""
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.deps import get_auth_service
from app.models.schemas import UserOut
from app.services.auth_service import AuthError, AuthService

_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserOut:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    try:
        return auth_service.decode_token(credentials.credentials)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def require_role(*allowed_roles: str):
    # Factory, not a plain dependency, so a route can parameterize which
    # roles are allowed (e.g. require_role("admin")) while still fitting
    # FastAPI's Depends(...) call shape.
    def _check(user: UserOut = Depends(get_current_user)) -> UserOut:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{user.role}' is not permitted to perform this action.",
            )
        return user

    return _check
