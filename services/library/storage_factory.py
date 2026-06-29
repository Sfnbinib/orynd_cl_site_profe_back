"""Single entry point for getting the active library StorageBackend.

Env contract::

    LIBRARY_BACKEND=supabase|mock    # default: supabase
    SUPABASE_LIBRARY_URL=...
    SUPABASE_LIBRARY_KEY=...

If ``LIBRARY_BACKEND=supabase`` but either:
  * the ``supabase`` package is not installed, or
  * the env vars are missing,

we log a warning and return ``MockStorageBackend`` — the app keeps running
with in-memory state so dev / Phase 0 demo continues without external infra.

A process-wide singleton is cached so all routers see the same instance.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from orynd_core.services.library.storage_abstract import StorageBackend
from orynd_core.services.library.storage_mock import MockStorageBackend
from orynd_core.services.logging import get_logger

log = get_logger("orynd.library.factory")

_lock = threading.Lock()
_backend: Optional[StorageBackend] = None


def get_storage_backend() -> StorageBackend:
    global _backend
    if _backend is not None:
        return _backend
    with _lock:
        if _backend is not None:
            return _backend
        _backend = _build_backend()
        return _backend


def reset_storage_backend() -> None:
    """Clear the cached singleton — for tests only."""
    global _backend
    with _lock:
        _backend = None


def _build_backend() -> StorageBackend:
    requested = os.environ.get("LIBRARY_BACKEND", "supabase").lower()

    if requested == "mock":
        log.info("library.backend.selected", backend="mock", reason="explicit")
        return MockStorageBackend()

    if requested == "supabase":
        try:
            from orynd_core.services.library.storage_supabase import SupabaseBackend
            backend = SupabaseBackend.from_env()
            log.info("library.backend.selected", backend="supabase")
            return backend
        except Exception as exc:
            log.warning(
                "library.backend.fallback_to_mock",
                requested="supabase",
                reason=str(exc),
            )
            return MockStorageBackend()

    log.warning("library.backend.unknown_falling_back_to_mock", requested=requested)
    return MockStorageBackend()


__all__ = ["get_storage_backend", "reset_storage_backend"]
