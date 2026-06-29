"""In-process async pub/sub.

Used by background tasks, agents, and the lifecycle manager to notify each
other without hard coupling. Frontend gets a mirror via the WebSocket bridge
that will be wired in Phase 7.

Design:
* topic-keyed listener list (callable or async callable)
* fire-and-forget publish — never blocks the publisher
* slow listeners run in a background task so one bad subscriber doesn't stall
* every event also emits a structlog line for replay/debug
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections import defaultdict
from typing import Any, Awaitable, Callable

from orynd_core.services.logging import get_logger

log = get_logger("orynd.event_bus")

Listener = Callable[[dict[str, Any]], Any]


class EventBus:
    def __init__(self) -> None:
        self._listeners: dict[str, list[Listener]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def subscribe(self, topic: str, listener: Listener) -> Callable[[], None]:
        """Register a listener. Returns an unsubscribe callable."""
        self._listeners[topic].append(listener)

        def _unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._listeners[topic].remove(listener)

        return _unsubscribe

    async def publish(self, topic: str, payload: dict[str, Any] | None = None) -> int:
        """Dispatch to every listener of ``topic``. Returns number invoked."""
        payload = payload or {}
        safe = {f"p_{k}" if k == "topic" else k: v for k, v in payload.items() if not _is_secret(k)}
        log.info("event.publish", event_topic=topic, **safe)
        listeners = list(self._listeners.get(topic, ()))
        for listener in listeners:
            asyncio.create_task(_safe_invoke(topic, listener, payload))
        return len(listeners)

    def clear(self) -> None:
        self._listeners.clear()


async def _safe_invoke(topic: str, listener: Listener, payload: dict[str, Any]) -> None:
    try:
        result = listener(payload)
        if inspect.isawaitable(result):
            await result
    except Exception:
        log.error("event.listener_failed", topic=topic, exc_info=True)


_SECRET_KEYS = {"password", "token", "secret", "api_key", "jwt"}


def _is_secret(key: str) -> bool:
    return any(part in key.lower() for part in _SECRET_KEYS)


# Process-wide singleton — services import this directly.
bus = EventBus()

__all__ = ["EventBus", "bus"]
