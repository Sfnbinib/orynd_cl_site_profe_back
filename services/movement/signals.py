"""Movement signal schema — one observable user action."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MovementSignal(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=_utcnow)
    user_id: Optional[UUID] = None
    session_id: UUID

    action_type: str
    object_ids: list[str] = Field(default_factory=list)
    parameters: dict = Field(default_factory=dict)

    scene_state_hash: str = ""
    camera_position: Optional[tuple[float, float, float]] = None
    selection_count_before: int = 0

    time_since_prev_action_ms: int = 0
    last_actions: list[str] = Field(default_factory=list)


__all__ = ["MovementSignal"]
