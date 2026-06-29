"""Process-wide license state + grace-period accounting.

Lifecycle:
    1. App starts → ``get_license_state()`` returns DEMO state by default.
    2. User signs in → license server returns JWT → ``load_license_jwt()``
       verifies and caches.
    3. Periodic refresh updates JWT; if offline, grace period applies.
    4. Grace expired → ``state.is_locked`` becomes True, Pro features deny.

The JWT itself is the source of truth — we don't trust local storage to
mutate the tier. Disk cache is **encrypted** (Electron safeStorage) in the
real client; this Python side only keeps in-memory + reads from `~/.orynd/license.jwt`
when present (for headless dev / CI).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from orynd_core.services.license.hwid import compute_hwid
from orynd_core.services.license.jwt_verify import (
    LicenseVerificationError,
    VerifiedClaims,
    verify_jwt,
)
from orynd_core.services.license.tiers import Tier
from orynd_core.services.logging import get_logger

log = get_logger("orynd.license.state")

GRACE_PERIOD_SECONDS = 14 * 24 * 60 * 60  # 14 days offline


@dataclass
class LicenseState:
    tier: Tier = Tier.DEMO
    user_id: str = ""
    hwid: str = ""
    license_id: Optional[str] = None
    issued_at: int = 0
    expires_at: int = 0
    last_refresh_at: int = field(default_factory=lambda: int(time.time()))
    is_locked: bool = False
    raw_claims: Optional[dict] = None

    def seconds_until_expiry(self) -> int:
        return max(0, self.expires_at - int(time.time()))

    def is_expired(self) -> bool:
        return self.expires_at > 0 and int(time.time()) >= self.expires_at

    def grace_remaining(self) -> int:
        elapsed = int(time.time()) - self.last_refresh_at
        return max(0, GRACE_PERIOD_SECONDS - elapsed)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "user_id": self.user_id,
            "license_id": self.license_id,
            "hwid_prefix": (self.hwid[:8] if self.hwid else ""),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "last_refresh_at": self.last_refresh_at,
            "is_locked": self.is_locked,
            "is_expired": self.is_expired(),
            "grace_remaining_s": self.grace_remaining(),
        }


_lock = threading.Lock()
_state: Optional[LicenseState] = None


def _disk_jwt_path() -> Path:
    return Path(os.environ.get("ORYND_INSTALL_DIR", "~/.orynd")).expanduser() / "license.jwt"


def _try_load_from_disk() -> Optional[str]:
    p = _disk_jwt_path()
    try:
        return p.read_text().strip() if p.exists() else None
    except OSError:
        return None


def get_license_state() -> LicenseState:
    """Return current state. Loads from disk on first call."""
    global _state
    if _state is not None:
        # Re-evaluate locked status as time passes.
        if not _state.is_locked and _state.is_expired() and _state.grace_remaining() == 0:
            _state.is_locked = True
        return _state
    with _lock:
        if _state is not None:
            return _state
        token = _try_load_from_disk()
        if token:
            try:
                claims = verify_jwt(token)
                _state = _state_from_claims(claims)
                return _state
            except LicenseVerificationError as exc:
                log.warning("license.disk_jwt_invalid", error=str(exc))
        _state = _default_demo_state()
        return _state


def _state_from_claims(claims: VerifiedClaims) -> LicenseState:
    return LicenseState(
        tier=claims.tier,
        user_id=claims.sub,
        hwid=claims.hwid,
        license_id=claims.jti,
        issued_at=claims.iat,
        expires_at=claims.exp,
        last_refresh_at=int(time.time()),
        is_locked=False,
        raw_claims=claims.raw,
    )


def _default_demo_state() -> LicenseState:
    hwid = compute_hwid()
    return LicenseState(
        tier=Tier.DEMO,
        user_id="demo",
        hwid=hwid,
        license_id=None,
        issued_at=int(time.time()),
        expires_at=0,
        last_refresh_at=int(time.time()),
        is_locked=False,
    )


def load_license_jwt(token: str, *, persist: bool = True) -> LicenseState:
    """Verify ``token`` and replace the cached state."""
    global _state
    claims = verify_jwt(token)
    new_state = _state_from_claims(claims)
    with _lock:
        _state = new_state
    if persist:
        try:
            path = _disk_jwt_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(token)
        except OSError as exc:
            log.warning("license.disk_persist_failed", error=str(exc))
    log.info("license.loaded", tier=new_state.tier.value, user=new_state.user_id)
    return new_state


def clear_license_state() -> None:
    """Drop the cached state. For sign-out + tests."""
    global _state
    with _lock:
        _state = None
    try:
        p = _disk_jwt_path()
        if p.exists():
            p.unlink()
    except OSError:
        pass


__all__ = [
    "LicenseState",
    "GRACE_PERIOD_SECONDS",
    "get_license_state",
    "load_license_jwt",
    "clear_license_state",
]
