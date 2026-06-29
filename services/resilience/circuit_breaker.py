"""Async circuit breaker registry.

Spec: CONNECTIONS_AND_INTEGRATION.md § Circuit breaker.

Each external service gets a named breaker. The shared :data:`breakers` dict
keeps state per-process and is also surfaced via ``/system/health/deep``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Awaitable, Callable, TypeVar

from orynd_core.errors import CircuitOpenError

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Half-open after recovery_timeout; closes on first success, opens again on failure."""

    def __init__(
        self,
        name: str,
        *,
        fail_threshold: int = 5,
        recovery_timeout: int = 60,
    ) -> None:
        self.name = name
        self.fail_threshold = fail_threshold
        self.recovery_timeout = recovery_timeout
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: datetime | None = None

    async def call(self, fn: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        if self.state is CircuitState.OPEN:
            assert self.last_failure_time is not None
            elapsed = datetime.now(timezone.utc) - self.last_failure_time
            if elapsed > timedelta(seconds=self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError(
                    f"{self.name} circuit open",
                    details={"name": self.name, "recovery_in_s": self.recovery_timeout - elapsed.total_seconds()},
                )

        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        if self.failure_count >= self.fail_threshold:
            self.state = CircuitState.OPEN

    def _record_success(self) -> None:
        if self.state is CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
        self.failure_count = 0

    def reset(self) -> None:
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None


# Default registry — extend in service init code as new externals are added.
breakers: dict[str, CircuitBreaker] = {
    "anthropic": CircuitBreaker("anthropic", fail_threshold=5, recovery_timeout=60),
    "printables": CircuitBreaker("printables", fail_threshold=10, recovery_timeout=120),
    "thingiverse": CircuitBreaker("thingiverse", fail_threshold=10, recovery_timeout=120),
    "makerworld": CircuitBreaker("makerworld", fail_threshold=10, recovery_timeout=120),
    "github": CircuitBreaker("github", fail_threshold=10, recovery_timeout=120),
    "supabase": CircuitBreaker("supabase", fail_threshold=3, recovery_timeout=30),
    "ollama": CircuitBreaker("ollama", fail_threshold=5, recovery_timeout=60),
}


def get_breaker(name: str) -> CircuitBreaker:
    if name not in breakers:
        breakers[name] = CircuitBreaker(name)
    return breakers[name]


__all__ = ["CircuitBreaker", "CircuitState", "breakers", "get_breaker"]
