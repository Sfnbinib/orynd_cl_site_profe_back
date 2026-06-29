"""
Per-part assembly exporter for AI Model 4.

Founder principle (2026-06-05 voice dump §4 "native ORYND format"):

  > "Лучше тогда данные кидать напрямую, иначе мы можем и дерево элементов
  >  терять — а это важно."

STEP is a terminal export — it collapses the assembly tree, loses region
attribution, fit error, rejected alternatives, transform parameters in
editable form. This module makes the rich JSON the primary output and
treats STEP/STL as on-demand exports.

Outputs per dual-pass run:

    <run>/<stl_name>/
        assembly.orynd.json          ← PRIMARY: full tree, all fits, all metadata
        assembly/
            full.step                ← all fits unioned (for "open one file" workflows)
            full.stl
        parts/
            part_01_box.step         ← each PrimitiveFit as its own file
            part_01_box.stl
            part_02_cylinder.step
            ...

When the `.orynd` container format ships, `assembly.orynd.json` becomes its
JSON sidecar and the directory becomes the container's internal layout.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cad_translator import (
    _translate_box,
    _translate_cylinder,
    _translate_plane,
    UNSUPPORTED_PRIMITIVES,
    _next_id,
)

log = logging.getLogger(__name__)


@dataclass
class PartExport:
    """One emitted part (one PrimitiveFit → one file pair)."""
    index: int
    primitive_type: str
    region_id: int
    step_path: Path | None
    stl_path: Path | None
    skipped_reason: str | None
    fit_error_mm: float | None
    parameters: dict
    transform: dict


def _build_one_primitive_doc(op: dict) -> tuple[dict | None, str | None]:
    """Translate a single AI-Model-4 op into a one-primitive CADAgent doc."""
    counter: dict[str, int] = {}
    sketch_id = _next_id("sketch", counter)
    body_id = _next_id("body", counter)

    kind = op.get("op")
    if kind in UNSUPPORTED_PRIMITIVES:
        return None, UNSUPPORTED_PRIMITIVES[kind]
    if kind == "box":
        new_ops, err = _translate_box(op, sketch_id, body_id)
    elif kind == "cylinder":
        new_ops, err = _translate_cylinder(op, sketch_id, body_id)
    elif kind == "plane":
        new_ops, err = _translate_plane(op, sketch_id, body_id)
    else:
        return None, f"{kind}: unknown primitive"

    if err:
        return None, err

    return {
        "units": "mm",
        "operations": new_ops,
        "meta": {
            "translation_notes": [],
            "bodies_built": 1,
            "source_op_count": 1,
            "skipped_count": 0,
        },
    }, None


async def export_per_part(
    final_coreops: dict,
    out_dir: Path,
    session_id: str,
) -> list[PartExport]:
    """
    Export each primitive op from `final_coreops` as its own STEP+STL pair.

    Returns one PartExport per input op (including skipped ones — so the user
    can see what the pipeline tried but couldn't compile).
    """
    from orynd_core.agents.base import AgentContext
    from orynd_core.agents.cad import CADAgent

    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    exports: list[PartExport] = []
    cad_agent = CADAgent()

    for i, op in enumerate(final_coreops.get("operations") or [], start=1):
        kind = op.get("op", "unknown")
        region_id = op.get("region_id", -1)
        params = op.get("params") or {}
        transform = op.get("transform") or {}
        fit_error = op.get("fit_error_mm")

        cad_doc, err = _build_one_primitive_doc(op)
        if cad_doc is None:
            exports.append(PartExport(
                index=i, primitive_type=kind, region_id=region_id,
                step_path=None, stl_path=None,
                skipped_reason=err,
                fit_error_mm=fit_error,
                parameters=params, transform=transform,
            ))
            continue

        # Build this single part
        ctx = AgentContext(session_id=f"{session_id}_part{i:02d}")
        ctx.extra["coreops"] = cad_doc
        res = await cad_agent.run(ctx)
        cad_out = ctx.extra.get("cad", {})

        step_dst = parts_dir / f"part_{i:02d}_{kind}.step"
        stl_dst = parts_dir / f"part_{i:02d}_{kind}.stl"

        if res.ok and cad_out.get("step_path"):
            import shutil
            try:
                shutil.copyfile(cad_out["step_path"], step_dst)
            except Exception as e:  # noqa: BLE001
                log.warning("[assembly] copy step failed for part %d: %s", i, e)
                step_dst = None
            if cad_out.get("stl_path"):
                try:
                    shutil.copyfile(cad_out["stl_path"], stl_dst)
                except Exception as e:  # noqa: BLE001
                    log.warning("[assembly] copy stl failed for part %d: %s", i, e)
                    stl_dst = None
            exports.append(PartExport(
                index=i, primitive_type=kind, region_id=region_id,
                step_path=step_dst if step_dst and step_dst.exists() else None,
                stl_path=stl_dst if stl_dst and stl_dst.exists() else None,
                skipped_reason=None,
                fit_error_mm=fit_error,
                parameters=params, transform=transform,
            ))
        else:
            exports.append(PartExport(
                index=i, primitive_type=kind, region_id=region_id,
                step_path=None, stl_path=None,
                skipped_reason=res.error or "cad_build_failed",
                fit_error_mm=fit_error,
                parameters=params, transform=transform,
            ))

    return exports


def write_orynd_json(
    out_path: Path,
    *,
    mesh_info: dict,
    pass1_summary: dict,
    filter_summary: dict,
    primitive_summary: dict,
    final_coreops: dict,
    parts: list[PartExport],
    cad_doc: dict | None = None,
) -> None:
    """
    Write the proto-`.orynd` container as JSON.

    This is the PRIMARY output. STEP/STL are exports derived from this.
    When the binary container ships later, this dict becomes its JSON layer.
    """
    payload = {
        "schema": "orynd.assembly/v0.1-proto",
        "source": {
            "kind": "stl_dual_pass",
            "mesh_info": mesh_info,
        },
        "pipeline": {
            "pass1": pass1_summary,
            "engineering_filter": filter_summary,
            "pass2_primitives": primitive_summary,
        },
        "assembly": {
            "tree_kind": "flat_union",  # future: hierarchical
            "parts": [
                {
                    "index": p.index,
                    "primitive_type": p.primitive_type,
                    "region_id": p.region_id,
                    "fit_error_mm": p.fit_error_mm,
                    "parameters": p.parameters,
                    "transform": p.transform,
                    "exports": {
                        "step": str(p.step_path) if p.step_path else None,
                        "stl": str(p.stl_path) if p.stl_path else None,
                    },
                    "skipped_reason": p.skipped_reason,
                }
                for p in parts
            ],
            "part_count_total": len(parts),
            "part_count_exported": sum(1 for p in parts if p.step_path),
            "part_count_skipped": sum(1 for p in parts if p.skipped_reason),
        },
        "exports": {
            "assembly_step": "assembly/full.step",
            "assembly_stl": "assembly/full.stl",
            "parts_dir": "parts/",
            "primary_doc": "assembly.orynd.json",
        },
        "raw": {
            "final_coreops_from_ai_model_4": final_coreops,
            "cad_translation_doc": cad_doc,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
