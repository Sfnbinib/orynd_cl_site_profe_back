"""/harness/* — capability discovery, planning, execution."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from orynd_core.services.harness import (
    CompositionPlan,
    CompositionPlanner,
    get_capability_registry,
)
from orynd_core.services.harness.composer import (
    ExecutionResult,
    OperationMode,
)

router = APIRouter(prefix="/harness", tags=["harness"])


@router.get("/capabilities")
async def list_capabilities(category: Optional[str] = None) -> list[dict[str, Any]]:
    registry = get_capability_registry()
    items = (
        registry.list_by_category(category) if category else registry.list_all()
    )
    return [cap.public_manifest() for cap in items]


@router.get("/capabilities/{capability_id}")
async def get_capability(capability_id: str) -> dict[str, Any]:
    cap = get_capability_registry().get(capability_id)
    if cap is None:
        raise HTTPException(status_code=404, detail="capability not found")
    return cap.public_manifest()


@router.get("/capabilities/search/{query}")
async def search_capabilities(
    query: str, k: int = Query(10, ge=1, le=50)
) -> list[dict[str, Any]]:
    matches = get_capability_registry().search(query, k=k)
    return [cap.public_manifest() for cap in matches]


@router.post("/plan", response_model=CompositionPlan)
async def plan(payload: dict[str, Any] = Body(...)):
    task = payload.get("task")
    if not task or not isinstance(task, str):
        raise HTTPException(status_code=422, detail="missing 'task' string")
    max_steps = int(payload.get("max_steps", 3))
    budget_tokens = payload.get("budget_tokens")
    planner = CompositionPlanner()
    return await planner.plan(
        task,
        max_steps=max_steps,
        budget_tokens=budget_tokens,
    )


@router.post("/execute", response_model=ExecutionResult)
async def execute(payload: dict[str, Any] = Body(...)):
    plan_payload = payload.get("plan")
    if not isinstance(plan_payload, dict):
        raise HTTPException(status_code=422, detail="missing 'plan' object")
    try:
        comp_plan = CompositionPlan(**plan_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"bad plan: {exc}")
    mode: OperationMode = payload.get("mode", "auto")
    planner = CompositionPlanner()
    return await planner.execute(comp_plan, mode=mode)
