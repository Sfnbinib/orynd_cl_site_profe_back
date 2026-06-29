"""FastAPI dependencies for resolving the current authenticated user."""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from .supabase_jwt import JWTError, verify_supabase_jwt


@dataclass(frozen=True)
class UserContext:
    id: str
    email: str | None
    role: str


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


async def current_user(authorization: str | None = Header(default=None)) -> UserContext:
    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    try:
        claims = verify_supabase_jwt(token)
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e)) from e
    return UserContext(id=claims.sub, email=claims.email, role=claims.role)


async def optional_user(authorization: str | None = Header(default=None)) -> UserContext | None:
    """Same as current_user but returns None for unauthenticated requests.

    Use on endpoints that should work for guests but enrich for signed-in users.
    """
    token = _extract_bearer(authorization)
    if not token:
        return None
    try:
        claims = verify_supabase_jwt(token)
    except JWTError:
        return None
    return UserContext(id=claims.sub, email=claims.email, role=claims.role)


def require_role(*allowed_roles: str):
    """Dependency factory: enforce that user.role is in allowed_roles."""

    async def _dep(user: UserContext = Depends(current_user)) -> UserContext:
        if user.role not in allowed_roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "insufficient role")
        return user

    return _dep
