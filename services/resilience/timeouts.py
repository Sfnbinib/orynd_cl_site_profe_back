"""Timeout matrix — CONNECTIONS_AND_INTEGRATION.md § Timeout matrix.

Use ``get_timeout(name)`` to build an ``httpx.Timeout`` for any external call.
"""

from __future__ import annotations

import httpx

# (connect_s, total_s) per logical connection.
TIMEOUTS: dict[str, tuple[float, float]] = {
    "local_backend": (1.0, 30.0),
    "library_db": (5.0, 30.0),
    "supabase": (5.0, 10.0),
    "anthropic": (10.0, 300.0),
    "printables": (5.0, 15.0),
    "thingiverse": (5.0, 15.0),
    "makerworld": (5.0, 15.0),
    "github": (5.0, 15.0),
    "arxiv": (5.0, 20.0),
    "ollama": (1.0, 60.0),
}


def get_timeout(name: str) -> httpx.Timeout:
    connect, total = TIMEOUTS.get(name, (5.0, 30.0))
    return httpx.Timeout(connect=connect, read=total, write=total, pool=connect)


__all__ = ["TIMEOUTS", "get_timeout"]
