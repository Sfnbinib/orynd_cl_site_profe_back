"""Shared HTTP clients (connection-pooled).

Reuse these instead of creating ad-hoc ``httpx.AsyncClient`` instances — fresh
clients leak connections and hide circuit-breaker state behind separate pools.

Clients are lazy: created on first access so the module can be imported in test
contexts without forcing a network stack init.
"""

from __future__ import annotations

import os
import threading

import httpx

from orynd_core.services.resilience.timeouts import get_timeout

_lock = threading.Lock()
_clients: dict[str, httpx.AsyncClient] = {}


def _build_client(name: str, **overrides) -> httpx.AsyncClient:
    base_kwargs: dict = {
        "timeout": get_timeout(name),
        "limits": httpx.Limits(max_connections=20, max_keepalive_connections=10),
    }
    base_kwargs.update(overrides)
    return httpx.AsyncClient(**base_kwargs)


def get_client(name: str, **overrides) -> httpx.AsyncClient:
    """Return a shared client for the named external service."""
    if name in _clients:
        return _clients[name]
    with _lock:
        if name in _clients:
            return _clients[name]
        _clients[name] = _build_client(name, **overrides)
        return _clients[name]


def get_printables_client() -> httpx.AsyncClient:
    return get_client("printables", base_url="https://api.printables.com")


def get_thingiverse_client() -> httpx.AsyncClient:
    return get_client("thingiverse", base_url="https://api.thingiverse.com")


def get_github_client() -> httpx.AsyncClient:
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return get_client("github", base_url="https://api.github.com", headers=headers)


def get_ollama_client() -> httpx.AsyncClient:
    return get_client(
        "ollama",
        base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
    )


async def close_all() -> None:
    """Call on process shutdown."""
    for client in list(_clients.values()):
        try:
            await client.aclose()
        except Exception:
            pass
    _clients.clear()


__all__ = [
    "get_client",
    "get_printables_client",
    "get_thingiverse_client",
    "get_github_client",
    "get_ollama_client",
    "close_all",
]
