"""AI Model 4 dual-pass mesh decomposition wrapped as a Skill.

* Pass 1 — MeshDecomposer (existing) splits the mesh into rough parts.
* Pass 2 — Engineering-clean primitive fitting.
* CAD bridge translates fitted primitives to CoreOpsDocument.

This skill is the harness-callable entry point. Direct programmatic access
still lives at ``orynd_core.agents.ai_model_4.run_dual_pass`` /
``run_dual_pass_to_cad``.
"""

from __future__ import annotations

from typing import Any

from orynd_core.skills.base import Skill, SkillSignature


class MeshDecomposeSkill(Skill):
    slug = "mesh_decompose"
    name = "Mesh Decomposition (AI Model 4)"
    description = (
        "Decompose an STL/OBJ mesh into engineering-clean primitives via "
        "AI Model 4 dual-pass: rough → engineering filter → primitive fit."
    )
    signature = SkillSignature(
        inputs={
            "mesh_path": "str — absolute path to STL/OBJ on disk",
            "build_cad": "bool — also translate to CoreOps + STEP (default False)",
            "session_id": "str — session id for telemetry (default 'skill')",
        },
        outputs={
            "primitive_summary": "dict — count by primitive type",
            "quality_score": "float — 0..1",
            "cad_paths": "list[str] — STEP/STL files when build_cad=True",
            "errors": "list[str]",
        },
        instructions=(
            "Run dual-pass decomposition on the mesh and optionally build CAD. "
            "Returns the primitive summary + quality + any output file paths."
        ),
    )
    tools = ["mesh_loader", "ai_model_4", "primitive_fitter"]
    version = "1.0.0"

    async def invoke(
        self,
        mesh_path: str,
        build_cad: bool = False,
        session_id: str = "skill",
        **_: Any,
    ) -> dict[str, Any]:
        if build_cad:
            from orynd_core.agents.ai_model_4 import run_dual_pass_to_cad

            outcome = await run_dual_pass_to_cad(mesh_path, session_id=session_id)
            return _summarise_cad_outcome(outcome)

        import asyncio
        from orynd_core.agents.ai_model_4 import run_dual_pass

        # run_dual_pass is sync (trimesh + numpy heavy) — push to thread so
        # the FastAPI event loop stays free for other requests.
        result = await asyncio.to_thread(run_dual_pass, mesh_path=mesh_path)
        return _summarise_dual_pass(result)


def _summarise_dual_pass(result: Any) -> dict[str, Any]:
    """Adapt DualPassResult dataclass into a JSON-safe skill response."""
    if hasattr(result, "to_dict"):
        # Native DualPassResult — trust its serializer.
        d = result.to_dict()
        primitive_summary = d.get("pass2", {}).get("primitive_summary", {}) or {}
        return {
            "primitive_summary": primitive_summary,
            "primitive_total": sum(primitive_summary.values()) if primitive_summary else 0,
            "primitive_fits_count": d.get("pass2", {}).get("primitive_fits_count", 0),
            "pass1_regions": d.get("pass1", {}).get("regions_count", 0),
            "filter_parts": d.get("filter", {}).get("parts_count", 0),
            "success": d.get("success", False),
            "quality_score": float(d.get("quality_score", 0.0) or 0.0),
            "duration_ms": d.get("duration_ms", {}),
            "errors": list(d.get("notes", []) or []),
        }
    # Fallback for unknown shapes — keep router responsive.
    return {
        "primitive_summary": getattr(result, "primitive_summary", {}) or {},
        "primitive_total": 0,
        "quality_score": float(getattr(result, "quality_score", 0.0) or 0.0),
        "errors": list(getattr(result, "notes", []) or []),
    }


def _summarise_cad_outcome(outcome: Any) -> dict[str, Any]:
    """Adapt DualPassToCadResult into a JSON-safe skill response."""
    dual = getattr(outcome, "dual_pass", outcome)
    base = _summarise_dual_pass(dual)

    cad_output: dict = getattr(outcome, "cad_output", {}) or {}
    # Common shape: cad_output may have step_path / stl_path / obj_path /
    # files / paths. Be liberal so the front-end always gets a flat list.
    cad_paths: list[str] = []
    for key in ("step_path", "stl_path", "obj_path"):
        value = cad_output.get(key)
        if value:
            cad_paths.append(str(value))
    extra = cad_output.get("paths") or cad_output.get("files") or []
    if isinstance(extra, list):
        cad_paths.extend(str(p) for p in extra if p)

    base["cad_paths"] = cad_paths
    base["cad_ok"] = bool(getattr(outcome, "ok", False))
    base["cad_output"] = cad_output
    base["translation_notes"] = list(getattr(outcome, "translation_notes", []) or [])
    if getattr(outcome, "error", None):
        base["errors"].append(str(outcome.error))
    return base
