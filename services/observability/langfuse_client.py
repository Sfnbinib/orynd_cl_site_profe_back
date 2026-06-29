"""Optional Langfuse wrapping.

If the ``langfuse`` package is installed and ``LANGFUSE_PUBLIC_KEY`` /
``LANGFUSE_SECRET_KEY`` are set, LLM calls are traced. Otherwise this module
provides a no-op shim so callers don't need ``if enabled`` branches everywhere.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

try:
    from langfuse import Langfuse  # type: ignore
except Exception:  # pragma: no cover — langfuse is optional
    Langfuse = None  # type: ignore


_client: Any | None = None


def _init() -> Any | None:
    global _client
    if _client is not None:
        return _client
    if Langfuse is None:
        return None
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    if not (public and secret):
        return None
    try:
        _client = Langfuse(
            public_key=public,
            secret_key=secret,
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception:
        _client = None
    return _client


def langfuse_enabled() -> bool:
    return _init() is not None


@asynccontextmanager
async def traced_llm_call(name: str, **metadata: Any) -> AsyncIterator[Any]:
    """Async ctx manager that records an LLM call span when Langfuse is on.

    Usage::

        async with traced_llm_call("anthropic.claude-haiku", model="claude-haiku-4-5") as span:
            response = await anthropic_client.messages.create(...)
            if span:
                span.update_trace(metadata={"output_tokens": response.usage.output_tokens})
    """
    client = _init()
    if client is None:
        yield None
        return
    try:
        span_cm = client.start_as_current_span(name=name)  # type: ignore[attr-defined]
    except Exception:
        yield None
        return
    with span_cm as span:
        if metadata:
            try:
                span.update_trace(metadata=metadata)
            except Exception:
                pass
        yield span


__all__ = ["langfuse_enabled", "traced_llm_call"]
