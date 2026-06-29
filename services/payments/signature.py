"""NOWPayments IPN signature verification (HMAC-SHA512 over sorted JSON)."""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def _canonical_json(payload: dict[str, Any]) -> str:
    """NOWPayments signs the JSON body with keys sorted alphabetically."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def verify_ipn_signature(*, payload: dict[str, Any], header_sig: str, secret: str) -> bool:
    """Constant-time HMAC-SHA512 verification of an IPN payload."""
    if not header_sig or not secret:
        return False
    body = _canonical_json(payload).encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha512).hexdigest()
    return hmac.compare_digest(expected, header_sig.strip())
