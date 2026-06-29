"""
CADAgent — builds 3D models from CoreOps JSON.

Input:  ctx.extra["coreops"] — CoreOps operations list (from LLM or CoreOpsAgent)
Output: ctx.extra["cad"] — paths to STL/STEP/OBJ + properties

Safety: LLM NEVER calls CadQuery directly. All geometry goes through
CoreOps JSON schema validation before execution.
"""
from __future__ import annotations
import asyncio
import logging

from orynd_core.agents.base import AgentContext, AgentResult, BaseAgent
from orynd_core.services.cad.schemas import CoreOpsDocument, OP_REGISTRY
from orynd_core.services.cad.engine import CadEngine

log = logging.getLogger(__name__)


class CADAgent(BaseAgent):
    name = "cad_agent"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.engine = CadEngine()

    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        coreops_data = ctx.extra.get("coreops")
        if not coreops_data:
            return AgentResult.failure(self.name, "No CoreOps data in ctx.extra['coreops']")

        if isinstance(coreops_data, dict):
            operations = coreops_data.get("operations", [])
            units = coreops_data.get("units", "mm")
        elif isinstance(coreops_data, list):
            operations = coreops_data
            units = "mm"
        else:
            return AgentResult.failure(self.name, f"Invalid coreops format: {type(coreops_data)}")

        # Per-op validation: validate each operation individually so ONE malformed
        # op (common with small local LLMs) doesn't sink the whole build. Keep the
        # valid ops, skip the rest with a reason — partial build beats total failure.
        valid_ops, skipped = [], []
        for i, raw in enumerate(operations):
            op_type = raw.get("op") if isinstance(raw, dict) else None
            cls = OP_REGISTRY.get(op_type)
            if cls is None:
                skipped.append({"index": i, "op": op_type, "reason": "unknown op"})
                continue
            try:
                cls.model_validate(raw)
                valid_ops.append(raw)
            except Exception as e:
                skipped.append({"index": i, "op": op_type, "reason": str(e).splitlines()[0][:140]})
        if skipped:
            log.warning("[cad_agent] skipped %d invalid op(s): %s", len(skipped), skipped)
        if not valid_ops:
            return AgentResult.failure(self.name, f"No valid CoreOps to build (all {len(skipped)} op(s) invalid)")
        doc = CoreOpsDocument(units=units, operations=valid_ops)

        # Run synchronous CadQuery in thread pool — don't block the async event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.engine.execute, doc, ctx.session_id)

        if not result.ok and not result.dry_run:
            return AgentResult.failure(self.name, result.error or "CAD execution failed")

        cad_output = {
            "stl_path": result.stl_path or "",
            "step_path": result.step_path or "",
            "obj_path": result.obj_path or "",
            "properties": result.properties,
            "dry_run": result.dry_run,
            "operations_executed": result.operations_executed,
            # Merge validation-level skips (above) with geometry-level skips
            # (degenerate primitives that OCCT rejected during the build).
            "skipped_ops": skipped + list(result.skipped_ops or []),
            "_source": "cadquery" if not result.dry_run else "dry_run",
        }

        ctx.extra["cad"] = cad_output
        return AgentResult.success(self.name, cad_output)
