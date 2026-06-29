"""/mesh/manual/* — manual mesh decomposition (HYBRID workflow).

Founder ask: ORYND ≠ agent-first. Manual workflow + AI hints.

Flow:
  POST /mesh/manual/suggest_primitive — user submits region → AI hints type
  POST /mesh/manual/assemble — user confirms regions → CoreOps document
  POST /mesh/manual/build_cad — assemble + run CADAgent → STEP/STL/OBJ
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from orynd_core.services.mesh.manual_decompose import (
    PrimitiveSuggestion,
    RegionSelection,
    assemble_coreops_from_manual_regions,
    suggest_primitive_for_region,
)

router = APIRouter(prefix="/mesh/manual", tags=["mesh-manual"])


def _region_from_payload(p: dict) -> RegionSelection:
    try:
        bbox_min = tuple(float(v) for v in p["bbox_min"])
        bbox_max = tuple(float(v) for v in p["bbox_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"bad bbox: {exc}")
    if len(bbox_min) != 3 or len(bbox_max) != 3:
        raise HTTPException(status_code=422, detail="bbox must be 3-tuple")
    return RegionSelection(
        region_id=str(p.get("region_id", "r0")),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        face_ids=list(p.get("face_ids", []) or []),
        vertex_ids=list(p.get("vertex_ids", []) or []),
        user_hint=p.get("user_hint"),
    )


@router.post("/suggest_primitive")
async def suggest_primitive(payload: dict = Body(...)) -> dict[str, Any]:
    """Single-region primitive type suggestion. Used live as user drags bbox."""
    region = _region_from_payload(payload.get("region", {}))
    mesh_path = payload.get("mesh_path")
    suggestion = suggest_primitive_for_region(region, mesh_path=mesh_path)
    return {
        "region_id": region.region_id,
        "primitive_type": suggestion.primitive_type,
        "confidence": suggestion.confidence,
        "parameters": suggestion.parameters,
        "rationale": suggestion.rationale,
    }


@router.post("/assemble")
async def assemble(payload: dict = Body(...)) -> dict[str, Any]:
    """User finalised set of (region, primitive_type) → CoreOps doc.

    Body:
        {
            "regions": [
                {
                    "region": {bbox_min, bbox_max, region_id, user_hint?},
                    "primitive": {
                        "primitive_type": "box",
                        "confidence": 1.0,
                        "parameters": {"sx": 10, "sy": 10, "sz": 10}
                    }
                },
                ...
            ]
        }
    """
    raw = payload.get("regions") or []
    if not isinstance(raw, list):
        raise HTTPException(status_code=422, detail="'regions' must be a list")
    pairs: list[tuple[RegionSelection, PrimitiveSuggestion]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail="region entry must be object")
        region = _region_from_payload(item.get("region", {}))
        prim_raw = item.get("primitive", {}) or {}
        suggestion = PrimitiveSuggestion(
            primitive_type=prim_raw.get("primitive_type", "unknown"),
            confidence=float(prim_raw.get("confidence", 0.5)),
            parameters=dict(prim_raw.get("parameters", {}) or {}),
            rationale=str(prim_raw.get("rationale", "user-assembled")),
        )
        pairs.append((region, suggestion))
    doc = assemble_coreops_from_manual_regions(pairs)
    return doc


@router.post("/build_cad")
async def build_cad(payload: dict = Body(...)) -> dict[str, Any]:
    """Assemble manual regions → CoreOps → CADAgent (STEP/STL/OBJ).

    Body shape == /assemble + ``session_id``.
    """
    raw = payload.get("regions") or []
    if not raw:
        raise HTTPException(status_code=422, detail="no regions")
    session_id = str(payload.get("session_id", "manual-cad"))

    # Assemble first
    pairs: list[tuple[RegionSelection, PrimitiveSuggestion]] = []
    for item in raw:
        region = _region_from_payload(item.get("region", {}))
        prim_raw = item.get("primitive", {}) or {}
        suggestion = PrimitiveSuggestion(
            primitive_type=prim_raw.get("primitive_type", "unknown"),
            confidence=float(prim_raw.get("confidence", 0.5)),
            parameters=dict(prim_raw.get("parameters", {}) or {}),
            rationale=str(prim_raw.get("rationale", "user-assembled")),
        )
        pairs.append((region, suggestion))
    coreops = assemble_coreops_from_manual_regions(pairs)

    # Bridge — same path as AI Model 4 dual-pass CAD bridge
    try:
        from orynd_core.agents.ai_model_4.cad_translator import translate_to_cad_coreops
        from orynd_core.agents.base import AgentContext
        from orynd_core.agents.cad import CADAgent
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"cad pipeline unavailable: {exc}")

    cad_doc = translate_to_cad_coreops(
        {
            "operations": coreops["operations"],
            "summary": coreops["summary"],
        }
    )
    if not cad_doc.get("operations"):
        return {
            "manual_coreops": coreops,
            "cad_coreops": cad_doc,
            "cad_ok": False,
            "error": "no_supported_primitives",
        }

    ctx = AgentContext(session_id=session_id)
    ctx.extra["coreops"] = cad_doc
    cad_agent = CADAgent()
    cad_res = await cad_agent.run(ctx)

    cad_output = ctx.extra.get("cad", {}) or {}
    return {
        "manual_coreops": coreops,
        "cad_coreops": cad_doc,
        "cad_ok": bool(getattr(cad_res, "ok", False)),
        "cad_output": cad_output,
        "error": getattr(cad_res, "error", None),
    }
