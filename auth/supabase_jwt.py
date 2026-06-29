"""Supabase JWT verification.

Supabase projects can sign access tokens with either:
  - ES256 / RS256 (asymmetric) — current default. Verified via the project's
    public JWKS endpoint ({SUPABASE_URL}/auth/v1/.well-known/jwks.json).
  - HS256 (legacy shared secret) — verified locally with SUPABASE_JWT_SECRET.

We branch on the token header `alg` and verify accordingly. No network call for
HS256; JWKS keys are fetched once and cached by PyJWKClient.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import jwt
from jwt import PyJWKClient


class JWTError(Exception):
    pass


@dataclass(frozen=True)
class Claims:
    sub: str           # user UUID
    email: str | None
    role: str          # 'authenticated' | 'anon' | 'service_role'
    exp: int
    aud: str


_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        base = os.getenv("SUPABASE_URL", "").rstrip("/")
        if not base:
            raise JWTError("SUPABASE_URL not configured")
        _jwks_client = PyJWKClient(f"{base}/auth/v1/.well-known/jwks.json")
    return _jwks_client


def _hs256_secret() -> str:
    s = os.getenv("SUPABASE_JWT_SECRET")
    if not s:
        raise JWTError("SUPABASE_JWT_SECRET not configured")
    return s


def verify_supabase_jwt(token: str) -> Claims:
    """Decode and validate a Supabase access token. Raises JWTError on any failure."""
    if not token:
        raise JWTError("empty token")

    try:
        alg = jwt.get_unverified_header(token).get("alg", "")
    except jwt.InvalidTokenError as e:
        raise JWTError(f"malformed token: {e}") from e

    decode_kwargs = dict(
        audience="authenticated",
        options={"require": ["exp", "sub", "aud"]},
    )

    try:
        if alg == "HS256":
            payload = jwt.decode(
                token,
                _hs256_secret(),
                algorithms=["HS256"],
                **decode_kwargs,
            )
        else:
            # ES256 / RS256 — asymmetric. Resolve signing key from JWKS by `kid`.
            signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256", "RS256"],
                **decode_kwargs,
            )
    except jwt.ExpiredSignatureError as e:
        raise JWTError("token expired") from e
    except jwt.InvalidAudienceError as e:
        raise JWTError("invalid audience") from e
    except jwt.PyJWKClientError as e:
        raise JWTError(f"jwks error: {e}") from e
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
