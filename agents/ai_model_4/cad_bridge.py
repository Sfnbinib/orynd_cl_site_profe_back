"""
End-to-end bridge: STL → AI Model 4 → CADAgent → STEP/STL/OBJ.

This is the one-call helper the plan describes:

    from orynd_core.agents.ai_model_4 import run_dual_pass_to_cad
    result = await run_dual_pass_to_cad("part.stl", session_id="demo")

`result` includes both the AI Model 4 dual-pass report and the CADAgent
output (STL/STEP/OBJ paths + computed properties). Useful for the
verification CLI, demos, and integration tests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from orynd_core.agents.base import AgentContext
from orynd_core.agents.cad import CADAgent

from .cad_translator import translate_to_cad_coreops
from .orchestrator import DualPassOrchestrator, DualPassResult

log = logging.getLogger(__name__)


@dataclass
class DualPassToCadResult:
    dual_pass: DualPassResult
    cad_coreops: dict = field(default_factory=dict)
    cad_output: dict = field(default_factory=dict)
    translation_notes: list[str] = field(default_factory=list)
    ok: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "dual_pass": self.dual_pass.to_dict(),
            "translation_notes": self.translation_notes,
            "cad_coreops": self.cad_coreops,
            "cad_output": self.cad_output,
        }


async def run_dual_pass_to_cad(
    mesh_path: str,
    *,
    session_id: str = "ai_model_4_cad",
    user_id: str | None = None,
) -> DualPassToCadResult:
    """
    Run the full STL → CAD pipeline.

    Steps:
      1. DualPassOrchestrator on the mesh.
      2. Translate primitive CoreOps to CADAgent CoreOps.
      3. Execute via CADAgent (or dry-run if CadQuery unavailable).
    """
    orch = DualPassOrchestrator()
    dual = orch.run(mesh_path=mesh_path)

    cad_doc = translate_to_cad_coreops(dual.final_coreops)
    notes = cad_doc.get("meta", {}).get("translation_notes", [])

    if not cad_doc.get("operations"):
        log.warning("[cad_bridge] no executable ops after translation — skipping CADAgent")
        return DualPassToCadResult(
            dual_pass=dual,
            cad_coreops=cad_doc,
            translation_notes=notes,
            ok=False,
            error="no_supported_primitives",
        )

    ctx = AgentContext(session_id=session_id, user_id=user_id)
    ctx.extra["coreops"] = cad_doc

    cad_agent = CADAgent()
    cad_res = await cad_agent.run(ctx)
    cad_output = ctx.extra.get("cad", {})

    # Real-time: notify any connected UI that a model is ready (incl. MCP-driven)
    try:
        from orynd_core.services.event_bus import bus
        summary = dual.to_dict().get("pass2", {}).get("primitive_summary", {})
        # Per-primitive geometry → UI renders each as a selectable object (.orynd native)
        fc = getattr(dual, "final_coreops", {}) or {}
        prims = (fc.get("operations", []) if isinstance(fc, dict) else [])[:200]
        await bus.publish("model.ready", {
            "session_id": session_id,
            "stl_url": f"/cad/model/{session_id}/part.stl" if cad_output.get("stl_path") else None,
            "step_url": f"/cad/model/{session_id}/part.step" if cad_output.get("step_path") else None,
            "source": "mesh_decompose",
            "primitive_summary": summary,
            "primitives": prims,
            "quality_score": dual.to_dict().get("quality_score", 0.0),
        })
    except Exception:
        log.warning("[cad_bridge] model.ready publish failed", exc_info=True)

    return DualPassToCadResult(
        dual_pass=dual,
        cad_coreops=cad_doc,
        cad_output=cad_output,
        translation_notes=notes,
        ok=cad_res.ok,
        error=cad_res.error,
    )
