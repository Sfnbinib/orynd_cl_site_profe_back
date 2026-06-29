"""Composition planner + executor.

Phase 1 keeps planning **heuristic** (capability search + simple ranking).
An LLM-driven planner lands in Phase 10 once Deep Research is wired.

The executor supports three on-error policies (abort / skip / retry) and
mode-aware approval gating that integrates with #87 Operation Modes
(coming in Phase 5).
"""

from __future__ import annotations

import time
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from orynd_core.errors import OryndError
from orynd_core.services.harness.capabilities import (
    Capability,
    CapabilityRegistry,
    get_capability_registry,
)
from orynd_core.services.logging import get_logger

log = get_logger("orynd.harness.composer")

OperationMode = Literal["plan", "auto", "ask_permission", "bypass"]
OnError = Literal["abort", "skip", "retry"]


class PlanStep(BaseModel):
    """One step in a composition plan."""

    step_id: str
    capability_id: str
    args: dict[str, Any] = Field(default_factory=dict)
    on_error: OnError = "abort"
    requires_user_approval: bool = False
    rationale: str = ""


class CompositionPlan(BaseModel):
    task: str
    steps: list[PlanStep] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class StepResult(BaseModel):
    step_id: str
    capability_id: str
    status: Literal["completed", "skipped", "failed"]
    duration_ms: int = 0
    result: Optional[Any] = None
    error: Optional[str] = None


class ExecutionResult(BaseModel):
    task: str
    steps: list[StepResult] = Field(default_factory=list)
    total_duration_ms: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0


class CompositionPlanner:
    """Heuristic planner — keyword search + cost-aware ranking."""

    def __init__(self, registry: Optional[CapabilityRegistry] = None) -> None:
        self.registry = registry or get_capability_registry()

    async def plan(
        self,
        task: str,
        *,
        max_steps: int = 3,
        budget_tokens: Optional[int] = None,
    ) -> CompositionPlan:
        candidates = self.registry.search(task, k=max(10, max_steps * 3))

        chosen: list[Capability] = []
        running_cost = 0
        for cap in candidates:
            if budget_tokens is not None and running_cost + cap.estimated_cost_tokens > budget_tokens:
                continue
            chosen.append(cap)
            running_cost += cap.estimated_cost_tokens
            if len(chosen) >= max_steps:
                break

        steps = [
            PlanStep(
                step_id=f"s{idx + 1}",
                capability_id=cap.id,
                args={},
                requires_user_approval=cap.requires_permission,
                rationale=f"matched task keywords: {task!r}",
            )
            for idx, cap in enumerate(chosen)
        ]

        confidence = min(1.0, 0.3 + 0.2 * len(steps))
        return CompositionPlan(
            task=task,
            steps=steps,
            candidates=[c.id for c in candidates],
            confidence=confidence,
        )

    async def execute(
        self,
        plan: CompositionPlan,
        *,
        mode: OperationMode = "auto",
        approval_resolver=None,
    ) -> ExecutionResult:
        total_started = time.time()
        result = ExecutionResult(task=plan.task)

        for step in plan.steps:
            cap = self.registry.get(step.capability_id)
            if cap is None:
                result.steps.append(
                    StepResult(
                        step_id=step.step_id,
                        capability_id=step.capability_id,
                        status="failed",
                        error="capability not found",
                    )
                )
                result.failed += 1
                if step.on_error == "abort":
                    break
                continue

            if _needs_approval(step, cap, mode):
                approved = await _resolve_approval(step, cap, approval_resolver, mode)
                if not approved:
                    result.steps.append(
                        StepResult(
                            step_id=step.step_id,
                            capability_id=step.capability_id,
                            status="skipped",
                            error="user_declined",
                        )
                    )
                    result.skipped += 1
                    continue

            step_started = time.time()
            try:
                payload = await cap.handler(**step.args)
                duration_ms = int((time.time() - step_started) * 1000)
                result.steps.append(
                    StepResult(
                        step_id=step.step_id,
                        capability_id=cap.id,
                        status="completed",
                        duration_ms=duration_ms,
                        result=_safe_payload(payload),
                    )
                )
                result.succeeded += 1
            except Exception as exc:
                duration_ms = int((time.time() - step_started) * 1000)
                err_msg = _stringify_error(exc)
                log.warning(
                    "harness.step_failed",
                    step_id=step.step_id,
                    capability_id=cap.id,
                    on_error=step.on_error,
                    error=err_msg,
                )
                if step.on_error == "abort":
                    result.steps.append(
                        StepResult(
                            step_id=step.step_id,
                            capability_id=cap.id,
                            status="failed",
                            duration_ms=duration_ms,
                            error=err_msg,
                        )
                    )
                    result.failed += 1
                    break
                if step.on_error == "retry":
                    try:
                        payload = await cap.handler(**step.args)
                        result.steps.append(
                            StepResult(
                                step_id=step.step_id,
                                capability_id=cap.id,
                                status="completed",
                                duration_ms=int((time.time() - step_started) * 1000),
                                result=_safe_payload(payload),
                            )
                        )
                        result.succeeded += 1
                        continue
                    except Exception as exc_retry:
                        err_msg = _stringify_error(exc_retry)
                # "skip" or retry-also-failed:
                result.steps.append(
                    StepResult(
                        step_id=step.step_id,
                        capability_id=cap.id,
                        status="failed",
                        duration_ms=duration_ms,
                        error=err_msg,
                    )
                )
                result.failed += 1

        result.total_duration_ms = int((time.time() - total_started) * 1000)
        return result


# ---- helpers -------------------------------------------------------------


def _needs_approval(step: PlanStep, cap: Capability, mode: OperationMode) -> bool:
    if mode == "bypass":
        return False
    if mode == "ask_permission":
        return True
    if mode == "plan":
        return True  # plan mode previews before any execution
    # auto: only approve permission-marked + medium/high/critical
    return step.requires_user_approval or cap.permission_category in {"high", "critical"}


async def _resolve_approval(
    step: PlanStep,
    cap: Capability,
    approval_resolver,
    mode: OperationMode,
) -> bool:
    if approval_resolver is None:
        # No resolver in tests/headless contexts:
        # bypass-equivalent behavior so the executor remains predictable.
        return mode != "ask_permission"
    return bool(await approval_resolver(step, cap, mode))


def _safe_payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _stringify_error(exc: Exception) -> str:
    if isinstance(exc, OryndError):
        return f"{exc.code}: {exc}"
    return f"{type(exc).__name__}: {exc}"


__all__ = [
    "CompositionPlan",
    "CompositionPlanner",
    "ExecutionResult",
    "PlanStep",
    "StepResult",
]
