"""/macro/* — natural language → CoreOps preview.

Used by bottom chat — юзер пишет "create 20mm cube and drill 5mm hole",
видит preview операций, applies them через harness if happy.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from orynd_core.services.macro import parse_text_to_coreops

router = APIRouter(prefix="/macro", tags=["macro"])


@router.post("/parse")
async def parse(payload: dict = Body(...)) -> dict:
    text = payload.get("text") or payload.get("input") or ""
    if not isinstance(text, str):
        raise HTTPException(status_code=422, detail="'text' must be string")
    use_llm = bool(payload.get("use_llm_fallback", False))
    result = parse_text_to_coreops(text, use_llm_fallback=use_llm)
    return result.to_dict()
