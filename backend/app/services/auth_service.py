"""
Registration, login, and JWT issuance/verification.

Role model, deliberately simple: two roles, "admin" and "viewer". There's
no self-service way to become admin (a client could otherwise just POST
role="admin" and grant itself upload/delete rights) — instead the FIRST
account ever registered becomes admin, and every account after that
defaults to "viewer". This is a common bootstrap pattern for small/internal
tools with no separate admin-provisioning workflow: whoever sets the app up
first is trusted to be the owner. Promoting additional users to admin later
is a direct UserStore/DB edit, not an API surface — deliberately, so it's
not something a compromised viewer token can ever reach.
"""
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.config import settings
from app.models.schemas import UserOut
from app.services.user_store import UserStore


class AuthError(Exception):
    """Raised for any auth failure the route layer should turn into a 401/409."""


class AuthService:
    def __init__(self, user_store: UserStore):
        self._store = user_store

    def register(self, username: str, password: str) -> UserOut:
        if self._store.get_by_username(username) is not None:
            raise AuthError(f"Username '{username}' is already taken.")

        role = "admin" if self._store.count() == 0 else "viewer"
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        record = self._store.create(username, password_hash, role)
        return UserOut(username=record["username"], role=record["role"])

    def authenticate(self, username: str, password: str) -> UserOut:
        record = self._store.get_by_username(username)
        if record is None:
            raise AuthError("Invalid username or password.")
        if not bcrypt.checkpw(password.encode("utf-8"), record["password_hash"].encode("utf-8")):
            raise AuthError("Invalid username or password.")
        return UserOut(username=record["username"], role=record["role"])

    def create_access_token(self, user: UserOut) -> str:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
        payload = {"sub": user.username, "role": user.role, "exp": expire}
        return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    def decode_token(self, token: str) -> UserOut:
        try:
            payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        except jwt.PyJWTError as exc:
            raise AuthError("Invalid or expired token.") from exc
        return UserOut(username=payload["sub"], role=payload["role"])
