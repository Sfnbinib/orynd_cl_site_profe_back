"""Background task manager.

Lightweight wrapper over ``asyncio.create_task`` that gives tasks an id,
progress reporting, and a queryable registry. Used by long-running flows
(Drop 2 install, Deep Research, AI Model 4 verification).

Phase 0: in-memory only. Phase 14 may persist to ``~/.orynd/tasks.db``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable
from uuid import uuid4

from orynd_core.services.event_bus import bus
from orynd_core.services.logging import get_logger
from orynd_core.services.observability.metrics import (
    background_tasks_active,
    background_tasks_completed_total,
)

log = get_logger("orynd.background_tasks")


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    id: str
    type: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result if self.status is TaskStatus.COMPLETED else None,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class BackgroundTaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._initialised = False

    async def initialize(self) -> None:
        self._initialised = True

    async def shutdown(self) -> None:
        for task in self._tasks.values():
            if task._task and not task._task.done():
                task._task.cancel()
        for task in list(self._tasks.values()):
            if task._task:
                try:
                    await task._task
                except (asyncio.CancelledError, Exception):
                    pass
        self._initialised = False

    def submit(
        self,
        task_type: str,
        coro_factory: Callable[["BackgroundTask"], Awaitable[Any]],
    ) -> BackgroundTask:
        task = BackgroundTask(id=uuid4().hex, type=task_type)
        self._tasks[task.id] = task
        background_tasks_active.labels(type=task_type).inc()
        task._task = asyncio.create_task(self._run(task, coro_factory))
        return task

    async def _run(
        self,
        task: BackgroundTask,
        coro_factory: Callable[["BackgroundTask"], Awaitable[Any]],
    ) -> None:
        task.status = TaskStatus.RUNNING
        await bus.publish("task.started", {"task_id": task.id, "type": task.type})
        try:
            task.result = await coro_factory(task)
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            await bus.publish("task.complete", {"task_id": task.id, "result_summary": _summary(task.result)})
            background_tasks_completed_total.labels(type=task.type, status="completed").inc()
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            background_tasks_completed_total.labels(type=task.type, status="cancelled").inc()
            raise
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = f"{type(exc).__name__}: {exc}"
            log.error("background_task.failed", task_id=task.id, type=task.type, exc_info=True)
            await bus.publish("task.failed", {"task_id": task.id, "error": task.error})
            background_tasks_completed_total.labels(type=task.type, status="failed").inc()
        finally:
            task.completed_at = time.time()
            background_tasks_active.labels(type=task.type).dec()

    def get(self, task_id: str) -> BackgroundTask | None:
        return self._tasks.get(task_id)

    def list(self) -> list[BackgroundTask]:
        return list(self._tasks.values())

    async def update_progress(self, task: BackgroundTask, progress: float, message: str | None = None) -> None:
        task.progress = max(0.0, min(1.0, progress))
        await bus.publish(
            "task.progress",
            {"task_id": task.id, "progress": task.progress, "message": message},
        )


def _summary(value: Any) -> Any:
    """Coerce to JSON-safe shape for event payloads (avoid huge blobs)."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {k: _summary(v) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [_summary(v) for v in list(value)[:20]]
    return repr(value)[:200]


# Process-wide singleton.
manager = BackgroundTaskManager()

__all__ = ["BackgroundTask", "BackgroundTaskManager", "TaskStatus", "manager"]
