"""
CAD router — execute CoreOps, validate, and serve model files.

POST /cad/execute  — run CoreOps → get STL/STEP/OBJ paths
POST /cad/validate — validate CoreOps JSON without executing
GET  /cad/model/{session_id}/{filename} — download model file
"""
from __future__ import annotations
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from orynd_core.agents.base import AgentContext
from orynd_core.agents.cad import CADAgent
from orynd_core.agents.orchestrator import Pipeline
from orynd_core.services.cad.engine import CAD_OUTPUT_DIR
from orynd_core.services.cad.schemas import CoreOpsDocument

router = APIRouter(prefix="/cad", tags=["cad"])


class CadExecuteRequest(BaseModel):
    session_id: str = "default"
    operations: list[dict]
    units: str = "mm"


def _adapt_operations(operations: list[dict]) -> tuple[list[dict], list[str], list[dict]]:
    """Accept all three op dialects and normalise to CADAgent CoreOps.

    * native CoreOps  — {"op": "CreateSketch"/"Extrude"/…}      → pass through
    * AI Model 4      — {"op": "box"/"cylinder", "params": {…}} → translator
    * macro parser    — {"type": "box", "parameters": {…}}      → → translator

    Returns (coreops_operations, notes, viewport_primitives). The third item is
    the AI-Model-4-dialect primitive list ({op, params}) the UI viewport / Tree
    render as selectable objects — empty for pure-native CoreOps builds.
    """
    from orynd_core.services.cad.schemas import OP_REGISTRY

    if all(op.get("op") in OP_REGISTRY for op in operations):
        return operations, [], []

    direct_coreops: list[dict] = []      # native CoreOps that can be mixed with adapted ops
    primitives: list[dict] = []          # shape-building primitives → translator
    hole_ops: list[dict] = []            # native CutHole ops, applied AFTER bodies exist
    notes: list[str] = []
    for op in operations:
        if op.get("op") in OP_REGISTRY:
            direct_coreops.append(op)
            continue
        if "params" in op and "op" in op:
            primitives.append(op)  # already AI Model 4 dialect
            continue
        kind = op.get("type") or op.get("primitive_type")
        p = op.get("parameters", {}) or {}
        ox = float(p.get("_offset_x", 0.0))  # spacing for multi-object (model_session.place_next)
        if kind in ("box", "cube"):
            sx, sy, sz = (float(p.get("sx", 20)), float(p.get("sy", 20)), float(p.get("sz", 20)))
            primitives.append({
                "op": "box",
                "params": {"size": [sx, sy, sz]},
                "transform": {
                    "center": [ox, 0.0, sz / 2.0],
                    "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                },
            })
        elif kind in ("cylinder", "attach", "bolt", "shaft"):
            radius = float(p.get("radius", p.get("outer_radius", 5)))
            height = float(p.get("height", p.get("length", 20)))
            primitives.append({
                "op": "cylinder",
                "params": {"radius": radius, "height": height},
                "transform": {"center": [ox, 0.0, height / 2.0], "axis": [0.0, 0.0, 1.0]},
            })
        elif kind in ("drill_hole", "hole", "cut_hole"):
            # Real subtractive feature — engine supports native CutHole. Drilled into the
            # final solid from its top face (Z+). center defaults to the object axis.
            radius = float(p.get("radius", p.get("r", 2.5)))
            cx = float(p.get("x", p.get("cx", ox)))
            cy = float(p.get("y", p.get("cy", 0.0)))
            depth = float(p.get("depth", 0.0))
            hole_ops.append({
                "op": "CutHole",
                "id": f"hole{len(hole_ops) + 1}",
                "center": {"x": cx, "y": cy},
                "radius": radius,
                "depth": depth,
                "through": depth <= 0.0,
                "on_face": "top",
            })
        else:
            notes.append(f"{kind}: no CAD mapping yet (scene-only op)")

    coreops: list[dict] = list(direct_coreops)
    if primitives:
        from orynd_core.agents.ai_model_4.cad_translator import translate_to_cad_coreops

        translated = translate_to_cad_coreops({"operations": primitives})
        notes.extend(translated.get("meta", {}).get("translation_notes", []))
        coreops.extend(translated.get("operations", []))

    # Holes need an existing body. Append them after the shape ops so they cut the
    # final solid; if there's no body to drill, surface an honest note (don't fail silently).
    if hole_ops:
        if coreops:
            coreops.extend(hole_ops)
        else:
            notes.append("drill_hole: no body to drill — build a shape first")

    return coreops, notes, primitives


class CadValidateRequest(BaseModel):
    operations: list[dict]
    units: str = "mm"


@router.post("/execute")
async def cad_execute(req: CadExecuteRequest) -> dict:
    operations, notes, primitives = _adapt_operations(req.operations)
    if not operations:
        raise HTTPException(
            422,
            detail={"message": "no executable CAD operations after adaptation", "notes": notes},
        )
    try:
        from orynd_core.services.cad import model_session

        model_session.clear(req.session_id)
        model_session.set_ops(req.session_id, operations)
    except Exception:
        pass

    ctx = AgentContext(
        session_id=req.session_id,
        extra={
            "coreops": {
                "operations": operations,
                "units": req.units,
            }
        },
    )

    result = await Pipeline([CADAgent()]).run(ctx)

    cad = ctx.extra.get("cad", {})
    if not cad:
        detail = "CAD execution returned no result"
        last = result.last
        if not result.ok and last is not None and last.error:
            detail = f"{detail}: {last.error}"
        raise HTTPException(500, detail=detail)

    stl_path = cad.get("stl_path", "")
    step_path = cad.get("step_path", "")
    obj_path = cad.get("obj_path", "")

    files = {
        "stl": f"/cad/model/{req.session_id}/part.stl" if stl_path else None,
        "step": f"/cad/model/{req.session_id}/part.step" if step_path else None,
        "obj": f"/cad/model/{req.session_id}/part.obj" if obj_path else None,
    }

    # Real-time: tell any connected UI a model is ready (incl. MCP-driven builds)
    from orynd_core.services.event_bus import bus
    await bus.publish("model.ready", {
        "session_id": req.session_id,
        "stl_url": files["stl"],
        "step_url": files["step"],
        "properties": cad.get("properties", {}),
        "operations_executed": cad.get("operations_executed", 0),
        "primitives": primitives,  # viewport / Tree render these as selectable objects
    })

    return {
        "ok": True,
        "session_id": req.session_id,
        "dry_run": cad.get("dry_run", False),
        "properties": cad.get("properties", {}),
        "operations_executed": cad.get("operations_executed", 0),
        "adaptation_notes": notes,
        "files": files,
    }


class CadAppendRequest(BaseModel):
    session_id: str = "default"
    operations: list[dict]          # NEW ops to add onto the current model
    units: str = "mm"
    place_beside: bool = True       # auto-offset new primitives so they don't overlap


@router.post("/append")
async def cad_append(req: CadAppendRequest) -> dict:
    """Build ops ONTO the session's existing model (multi-object / tools / sketch).

    Unlike /execute (which builds a fresh model from exactly the ops given), this
    accumulates: the new ops are appended to the session's model document and the
    WHOLE document is rebuilt, so previous objects stay in the scene.
    """
    from orynd_core.services.cad import model_session

    new_ops = list(req.operations)
    if req.place_beside:
        # offset each new primitive so 'build another one' sits beside, not inside
        placed = []
        for op in new_ops:
            placed.append(model_session.place_next(req.session_id, op))
            model_session.append_ops(req.session_id, [placed[-1]])
        full = model_session.get_ops(req.session_id)
    else:
        full = model_session.append_ops(req.session_id, new_ops)

    operations, notes, primitives = _adapt_operations(full)
    if not operations:
        raise HTTPException(422, detail={"message": "no executable ops after adaptation", "notes": notes})

    # Subtractive features (holes/cuts) live only in the built STL, not in the primitive
    # shapes — so the viewport must render the STL, not the pickable primitive boxes.
    has_cut = any((o.get("type") or o.get("op")) in ("drill_hole", "hole", "cut_hole", "cut") for o in full)

    # Re-apply any session modifiers (fillet/chamfer) so adding an object keeps them.
    mods = model_session.get_mods(req.session_id)
    if mods:
        operations = operations + mods

    ctx = AgentContext(
        session_id=req.session_id,
        extra={"coreops": {"operations": operations, "units": req.units}},
    )
    await Pipeline([CADAgent()]).run(ctx)
    cad = ctx.extra.get("cad", {})
    if not cad:
        raise HTTPException(500, detail="CAD execution returned no result")

    stl_path = cad.get("stl_path", "")
    files = {
        "stl": f"/cad/model/{req.session_id}/part.stl" if stl_path else None,
        "step": f"/cad/model/{req.session_id}/part.step" if cad.get("step_path") else None,
        "obj": f"/cad/model/{req.session_id}/part.obj" if cad.get("obj_path") else None,
    }

    from orynd_core.services.event_bus import bus
    await bus.publish("model.ready", {
        "session_id": req.session_id,
        "stl_url": files["stl"],
        "step_url": files["step"],
        "properties": cad.get("properties", {}),
        "operations_executed": cad.get("operations_executed", 0),
        # modifiers/cuts fuse into one solid → render the STL, not pickable prims
        "primitives": [] if (mods or has_cut) else primitives,
        "source": "append",
    })

    return {
        "ok": True,
        "session_id": req.session_id,
        "object_count": len(primitives),
        "properties": cad.get("properties", {}),
        "adaptation_notes": notes,
        "files": files,
    }


class CadModifyRequest(BaseModel):
    session_id: str = "default"
    kind: str                       # "fillet" | "chamfer"
    value: float = 2.0              # radius (fillet) or distance (chamfer), mm
    units: str = "mm"


@router.post("/modify")
async def cad_modify(req: CadModifyRequest) -> dict:
    """Apply a finishing modifier (fillet / chamfer) to the current model and rebuild.

    Lets the user select a built object and round/bevel it — the 'real CAD' edit.
    The modifier is stored on the session so it survives further appends.
    """
    from orynd_core.services.cad import model_session

    kind = (req.kind or "").lower()
    if kind == "fillet":
        model_session.append_mod(req.session_id, {"op": "Fillet", "id": f"fil{model_session.count(req.session_id)}", "radius": float(req.value), "edges": ["all"]})
    elif kind == "chamfer":
        model_session.append_mod(req.session_id, {"op": "Chamfer", "id": f"cha{model_session.count(req.session_id)}", "distance": float(req.value), "edges": ["all"]})
    else:
        raise HTTPException(400, detail=f"unknown modifier: {req.kind}")

    full = model_session.get_ops(req.session_id)
    operations, notes, _prims = _adapt_operations(full)
    if not operations:
        raise HTTPException(422, detail={"message": "no model to modify — build something first", "notes": notes})
    operations = operations + model_session.get_mods(req.session_id)

    ctx = AgentContext(session_id=req.session_id, extra={"coreops": {"operations": operations, "units": req.units}})
    await Pipeline([CADAgent()]).run(ctx)
    cad = ctx.extra.get("cad", {})
    if not cad or not cad.get("stl_path"):
        raise HTTPException(500, detail="modify build failed")

    files = {
        "stl": f"/cad/model/{req.session_id}/part.stl",
        "step": f"/cad/model/{req.session_id}/part.step" if cad.get("step_path") else None,
    }
    from orynd_core.services.event_bus import bus
    await bus.publish("model.ready", {
        "session_id": req.session_id,
        "stl_url": files["stl"],
        "step_url": files["step"],
        "properties": cad.get("properties", {}),
        "primitives": [],            # modified → one fused solid, render STL
        "source": "modify",
    })
    return {"ok": True, "session_id": req.session_id, "kind": kind, "value": req.value, "properties": cad.get("properties", {}), "files": files}


@router.post("/clear")
async def cad_clear(session_id: str = "default") -> dict:
    """Reset the session's model document (new part)."""
    from orynd_core.services.cad import model_session
    model_session.clear(session_id)
    return {"ok": True, "session_id": session_id}


@router.post("/validate")
async def cad_validate(req: CadValidateRequest) -> dict:
    try:
        doc = CoreOpsDocument(units=req.units, operations=req.operations)
        ops = doc.parse_operations()
        return {
            "valid": True,
            "operations_count": len(ops),
            "operations": [op.op for op in ops],
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
        }


@router.get("/model/{session_id}/{filename}")
async def cad_model_file(session_id: str, filename: str):
    allowed_extensions = {".stl", ".step", ".obj"}
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(400, detail=f"Invalid file type: {ext}")

    file_path = CAD_OUTPUT_DIR / session_id / filename
    if not file_path.exists():
        raise HTTPException(404, detail=f"File not found: {filename}")

    media_types = {
        ".stl": "application/sla",
        ".step": "application/step",
        ".obj": "text/plain",
    }

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=media_types.get(ext, "application/octet-stream"),
    )
