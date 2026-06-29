"""JWT license verification.

Production: server signs licenses with RSA-2048 (or Ed25519) private key.
Public key is **compiled into the binary** (`_LICENSE_PUBLIC_KEY_PEM`) so
attackers can't simply replace it with their own pubkey.

Demo period: when no public key is configured AND ``ORYND_LICENSE_DEMO_OK=1``,
we accept demo-tier unsigned tokens so dev/onboarding flow works without
a running license server.

Token shape (claims)::

    {
      "sub":  "<user_id>",
      "tier": "free" | "pro" | "max",
      "hwid": "<sha256 hex>",
      "max_devices": 2,
      "iat":  <int>,
      "exp":  <int>,            # +24h refresh
      "iss":  "oryndai.com",
      "jti":  "<uuid>"
    }

Verification:
    1. signature (RS256 / EdDSA against compiled-in pub key)
    2. exp not past
    3. iss == "oryndai.com"
    4. hwid matches local compute_hwid()
    5. tier ∈ Tier enum
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import jwt as pyjwt
from jwt import InvalidTokenError, PyJWTError

from orynd_core.errors import OryndError
from orynd_core.services.license.hwid import compute_hwid
from orynd_core.services.license.tiers import Tier
from orynd_core.services.logging import get_logger

log = get_logger("orynd.license.jwt")

# Demo/trial period — founder decision 2026-06-09: 3 days (was 14)
DEMO_PERIOD_DAYS = 3


# Public key gets compiled in via the build pipeline (Nuitka). For now we
# read it from env / file so dev iteration is possible.
_LICENSE_PUBLIC_KEY_PEM = os.environ.get("ORYND_LICENSE_PUBLIC_KEY_PEM")
_LICENSE_PUBLIC_KEY_PATH = os.environ.get("ORYND_LICENSE_PUBLIC_KEY_PATH")
_LICENSE_ISSUER = "oryndai.com"
_LICENSE_DEMO_OK = os.environ.get("ORYND_LICENSE_DEMO_OK") == "1"


class LicenseVerificationError(OryndError):
    code = "license.invalid"
    http_status = 403
    user_message = "License could not be verified"


@dataclass
class VerifiedClaims:
    sub: str
    tier: Tier
    hwid: str
    exp: int
    iat: int
    iss: str
    jti: Optional[str] = None
    max_devices: int = 1
    raw: dict[str, Any] | None = None


def _load_public_key() -> Optional[str]:
    if _LICENSE_PUBLIC_KEY_PEM:
        return _LICENSE_PUBLIC_KEY_PEM
    if _LICENSE_PUBLIC_KEY_PATH:
        try:
            with open(_LICENSE_PUBLIC_KEY_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except OSError as exc:
            log.warning("license.pubkey_read_failed", path=_LICENSE_PUBLIC_KEY_PATH, error=str(exc))
    return None


def verify_jwt(token: str, *, expected_hwid: Optional[str] = None) -> VerifiedClaims:
    """Verify a license JWT. Raises :class:`LicenseVerificationError` on any failure."""
    pubkey = _load_public_key()

    if pubkey is None:
        if _LICENSE_DEMO_OK:
            # Demo path — decode without verification, force DEMO tier.
            try:
                claims = pyjwt.decode(token, options={"verify_signature": False})
            except PyJWTError as exc:
                raise LicenseVerificationError(
                    "demo token unreadable",
                    details={"error": str(exc)},
                ) from exc
            return VerifiedClaims(
                sub=str(claims.get("sub", "demo-user")),
                tier=Tier.DEMO,
                hwid=str(claims.get("hwid", expected_hwid or "")),
                exp=int(claims.get("exp", time.time() + DEMO_PERIOD_DAYS * 86400)),
                iat=int(claims.get("iat", time.time())),
                iss=str(claims.get("iss", _LICENSE_ISSUER)),
                jti=claims.get("jti"),
                max_devices=int(claims.get("max_devices", 1)),
                raw=claims,
            )
        raise LicenseVerificationError(
            "license public key not configured",
            details={"hint": "set ORYND_LICENSE_PUBLIC_KEY_PEM or ORYND_LICENSE_DEMO_OK=1"},
        )

    try:
        claims = pyjwt.decode(
            token,
            pubkey,
            algorithms=["RS256", "EdDSA"],
            issuer=_LICENSE_ISSUER,
            options={"require": ["exp", "iat", "iss", "sub", "tier", "hwid"]},
        )
    except InvalidTokenError as exc:
        raise LicenseVerificationError("signature invalid", details={"error": str(exc)}) from exc
    except PyJWTError as exc:
        raise LicenseVerificationError("decode failed", details={"error": str(exc)}) from exc

    raw_tier = str(claims.get("tier", ""))
    try:
        tier = Tier(raw_tier)
    except ValueError as exc:
        raise LicenseVerificationError(
            f"unknown tier: {raw_tier}",
            details={"received": raw_tier},
        ) from exc

    hwid_in_token = str(claims.get("hwid", ""))
    expected = expected_hwid if expected_hwid is not None else compute_hwid()
    if hwid_in_token != expected:
        raise LicenseVerificationError(
            "hwid mismatch",
            details={"expected_prefix": expected[:8], "received_prefix": hwid_in_token[:8]},
        )

    return VerifiedClaims(
        sub=str(claims["sub"]),
        tier=tier,
        hwid=hwid_in_token,
        exp=int(claims["exp"]),
        iat=int(claims["iat"]),
        iss=str(claims["iss"]),
        jti=claims.get("jti"),
        max_devices=int(claims.get("max_devices", 1)),
        raw=claims,
    )


__all__ = ["LicenseVerificationError", "VerifiedClaims", "verify_jwt"]
