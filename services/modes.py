"""Operation Modes — #87.

Founder voice: *"Plan mode, bypass mode, auto mode, ask permission. Мы будем
работать через терминал, у нас такой полноценной возможности нету."*

The session mode is a single process-wide value (per backend instance) plus
a per-session override stored in memory. The ActionGate consults both to
decide whether an action proceeds, is paused for plan preview, or requires
explicit approval.

Phase 5 = state + gate. Frontend ModeSelector + WebSocket-pushed mode
changes land alongside Multi-Context UI (Phase 7).
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Optional

from orynd_core.errors import ForbiddenError
from orynd_core.services.event_bus import bus
from orynd_core.services.logging import get_logger

log = get_logger("orynd.modes")


class OperationMode(str, Enum):
    PLAN = "plan"                      # preview every action, no execution
    AUTO = "auto"                      # current default
    ASK_PERMISSION = "ask_permission"  # confirm every action
    BYPASS = "bypass"                  # autonomous, no approval


_lock = threading.Lock()
_global_mode: OperationMode = OperationMode.AUTO
_session_modes: dict[str, OperationMode] = {}


def get_mode(session_id: Optional[str] = None) -> OperationMode:
    if session_id and session_id in _session_modes:
        return _session_modes[session_id]
    return _global_mode


async def set_mode(mode: OperationMode, *, session_id: Optional[str] = None) -> OperationMode:
    global _global_mode
    with _lock:
        if session_id:
            _session_modes[session_id] = mode
        else:
            _global_mode = mode
    log.info("modes.changed", mode=mode.value, session_id=session_id)
    await bus.publish("mode.changed", {"mode": mode.value, "session_id": session_id})
    return mode


def clear_session_mode(session_id: str) -> None:
    with _lock:
        _session_modes.pop(session_id, None)


async def gate(
    action: str,
    *,
    permission_category: str = "low",
    session_id: Optional[str] = None,
    approval_resolver=None,
) -> bool:
    """Decide whether an action proceeds under the current mode.

    Returns:
        ``True`` — proceed
        ``False`` — silently skipped (e.g. plan mode)
    Raises:
        ForbiddenError — explicit denial (user declined in ask_permission)
    """
    mode = get_mode(session_id)

    if mode is OperationMode.BYPASS:
        return True

    if mode is OperationMode.PLAN:
        # Preview only — caller must surface the plan and call gate again
        # under a different mode to actually run.
        log.info("modes.gate.plan_blocked", action=action)
        return False

    if mode is OperationMode.ASK_PERMISSION:
        if approval_resolver is None:
            # Headless/test context: default-deny to keep behaviour predictable.
            return False
        approved = bool(await approval_resolver(action, permission_category))
        if not approved:
            raise ForbiddenError(
                f"action {action!r} declined by user",
                details={"action": action, "mode": mode.value},
            )
        return True

    # AUTO: only stop for high/critical
    if permission_category in {"high", "critical"}:
        if approval_resolver is None:
            return False
        approved = bool(await approval_resolver(action, permission_category))
        if not approved:
            raise ForbiddenError(
                f"action {action!r} ({permission_category}) declined by user",
                details={"action": action, "mode": mode.value},
            )
        return True

    return True


__all__ = ["OperationMode", "get_mode", "set_mode", "clear_session_mode", "gate"]
