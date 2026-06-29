"""Standard tenacity retry policies.

Wrap any awaitable that talks to the network with the matching policy.
Spec: CONNECTIONS_AND_INTEGRATION.md § Retry policy.
"""

from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Transient errors that are worth retrying for every external network call.
_TRANSIENT_HTTP = (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError)

network_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_TRANSIENT_HTTP),
    reraise=True,
)
"""3 attempts, 1-10s exponential backoff. Use for Printables, Thingiverse, etc."""


expensive_api_retry = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(_TRANSIENT_HTTP),
    reraise=True,
)
"""2 attempts only — for paid APIs (Anthropic) where wasted tokens cost money."""


__all__ = ["network_retry", "expensive_api_retry"]
