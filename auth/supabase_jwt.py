"""Supabase JWT verification (HS256).

Supabase issues JWTs signed with a project-wide secret (SUPABASE_JWT_SECRET).
We verify locally — no network call per request.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import jwt


class JWTError(Exception):
    pass


@dataclass(frozen=True)
class Claims:
    sub: str           # user UUID
    email: str | None
    role: str          # 'authenticated' | 'anon' | 'service_role'
    exp: int
    aud: str


def _secret() -> str:
    s = os.getenv("SUPABASE_JWT_SECRET")
    if not s:
        raise JWTError("SUPABASE_JWT_SECRET not configured")
    return s


def verify_supabase_jwt(token: str) -> Claims:
    """Decode and validate a Supabase access token. Raises JWTError on any failure."""
    if not token:
        raise JWTError("empty token")

    try:
        payload = jwt.decode(
            token,
            _secret(),
            algorithms=["HS256"],
            audience="authenticated",
            options={"require": ["exp", "sub", "aud"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise JWTError("token expired") from e
    except jwt.InvalidAudienceError as e:
        raise JWTError("invalid audience") from e
    except jwt.InvalidTokenError as e:
        raise JWTError(f"invalid token: {e}") from e

    # Extra defence-in-depth.
    if payload.get("exp", 0) < int(time.time()):
        raise JWTError("token expired")
    if not payload.get("sub"):
        raise JWTError("missing sub")

    return Claims(
        sub=payload["sub"],
        email=payload.get("email"),
        role=payload.get("role", "authenticated"),
        exp=int(payload["exp"]),
        aud=payload["aud"],
    )
