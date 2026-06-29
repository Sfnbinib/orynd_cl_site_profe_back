"""
Dual-Pass Orchestrator — chains Pass 1 → Engineering Filter → Pass 2 → CoreOps.

This is the entry point for "AI Model 4" as founder defined:
the name applies only when the full pipeline successfully outputs engineering primitives.

Usage:
    result = await run_dual_pass(mesh_path="part.stl")
    print(result.to_dict())
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from orynd_core.services.mesh.loader import load_mesh, load_mesh_from_bytes, MeshData
from orynd_core.services.mesh.decomposer import decompose_mesh, DecompositionResult
from orynd_core.services.mesh.feature_extractor import extract_features, FeatureExtractionResult

from .engineering_filter import EngineeringFilter, FilteredPart, BuildabilityTag
from .pass2_rebuild import Pass2Rebuilder, PrimitiveFit
from .fitters import PrimitiveType

log = logging.getLogger(__name__)


@dataclass
class DualPassResult:
    """Full output of dual-pass pipeline."""

    # Input metadata
    mesh_info: dict = field(default_factory=dict)

    # Pass 1 output
    pass1_regions_count: int = 0
    pass1_features_count: int = 0
    pass1_coreops: dict = field(default_factory=dict)

    # Engineering filter output
    filtered_summary: dict = field(default_factory=dict)
    filtered_parts: list[dict] = field(default_factory=list)

    # Pass 2 output
    primitive_fits: list[dict] = field(default_factory=list)
    primitive_summary: dict = field(default_factory=dict)

    # Final CoreOps
    final_coreops: dict = field(default_factory=dict)

    # Quality verdict
    success: bool = False
    quality_score: float = 0.0  # 0..1
    notes: list[str] = field(default_factory=list)

    # Timing
    duration_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mesh_info": self.mesh_info,
            "pass1": {
                "regions_count": self.pass1_regions_count,
                "features_count": self.pass1_features_count,
                "coreops_summary": self.pass1_coreops.get("summary", {}),
            },
            "filter": {
                "summary": self.filtered_summary,
                "parts_count": len(self.filtered_parts),
            },
            "pass2": {
                "primitive_fits_count": len(self.primitive_fits),
                "primitive_summary": self.primitive_summary,
                "fits": self.primitive_fits,
            },
            "final_coreops": self.final_coreops,
            "success": self.success,
            "quality_score": round(self.quality_score, 3),
            "notes": self.notes,
            "duration_ms": self.duration_ms,
        }


class DualPassOrchestrator:
    """Orchestrates Pass 1 → Filter → Pass 2 → CoreOps."""

    def __init__(
        self,
        filter_thresholds: Optional[dict] = None,
        pass2_thresholds: Optional[dict] = None,
    ):
        self.engineering_filter = EngineeringFilter()
        if filter_thresholds:
            for k, v in filter_thresholds.items():
                setattr(self.engineering_filter, k, v)

        self.rebuilder = Pass2Rebuilder(**(pass2_thresholds or {}))

    def run(
        self,
        mesh_path: Optional[str] = None,
        mesh_bytes: Optional[bytes] = None,
        mesh_format: str = "stl",
        scale: float = 1.0,
        decompose_angle: float = 15.0,
        decompose_min_faces: int = 5,
    ) -> DualPassResult:
        """Run full dual-pass pipeline."""
        import time

        result = DualPassResult()

        # ── Load mesh ──
        t0 = time.time()
        try:
            if mesh_path:
                mesh = load_mesh(mesh_path, scale=scale)
            elif mesh_bytes:
                mesh = load_mesh_from_bytes(mesh_bytes, file_type=mesh_format, scale=scale)
            else:
                result.notes.append("no_input_provided")
                return result
        except Exception as e:
            result.notes.append(f"mesh_load_failed: {e}")
            return result

        result.mesh_info = {
            "vertices": mesh.vertex_count,
            "triangles": mesh.triangle_count,
            "bbox_size_mm": mesh.size_mm().tolist(),
            "watertight": mesh.is_watertight,
            "source": mesh.source_path,
        }
        result.duration_ms["load"] = int((time.time() - t0) * 1000)

        # ── Pass 1: Decompose + extract features ──
        t1 = time.time()
        try:
            decomposition = decompose_mesh(
                mesh,
                angle_threshold_deg=decompose_angle,
                min_region_faces=decompose_min_faces,
            )
            extraction = extract_features(mesh, decomposition)
        except Exception as e:
            result.notes.append(f"pass1_failed: {e}")
            return result

        result.pass1_regions_count = len(decomposition.regions)
        result.pass1_features_count = extraction.total_features
        result.pass1_coreops = extraction.to_coreops_json()
        result.duration_ms["pass1"] = int((time.time() - t1) * 1000)

        log.info(
            f"[dual_pass] Pass 1: {len(decomposition.regions)} regions, "
            f"{extraction.total_features} features"
        )

        # ── Engineering Filter ──
        t2 = time.time()
        try:
            filtered = self.engineering_filter.filter(decomposition.regions)
        except Exception as e:
            result.notes.append(f"filter_failed: {e}")
            return result

        result.filtered_parts = [p.to_dict() for p in filtered]
        result.filtered_summary = {
            "buildable": sum(1 for p in filtered if p.tag == BuildabilityTag.BUILDABLE),
            "complex": sum(1 for p in filtered if p.tag == BuildabilityTag.COMPLEX),
            "noise": sum(1 for p in filtered if p.tag == BuildabilityTag.NOISE),
        }
        result.duration_ms["filter"] = int((time.time() - t2) * 1000)

        # ── Pass 2: Primitive fitting ──
        t3 = time.time()
        try:
            fits = self.rebuilder.rebuild(mesh, filtered)
        except Exception as e:
            result.notes.append(f"pass2_failed: {e}")
            return result

        result.primitive_fits = [f.to_dict() for f in fits]
        prim_counts = {}
        for fit in fits:
            key = fit.chosen_primitive.value
            prim_counts[key] = prim_counts.get(key, 0) + 1
        result.primitive_summary = prim_counts
        result.duration_ms["pass2"] = int((time.time() - t3) * 1000)

        log.info(f"[dual_pass] Pass 2: {prim_counts}")

        # ── Generate final CoreOps ──
        result.final_coreops = self._generate_final_coreops(mesh, fits, filtered)

        # ── Quality verdict ──
        result.quality_score, result.notes_extra = self._compute_quality(result)
        result.notes.extend(getattr(result, 'notes_extra', []) or [])
        result.success = result.quality_score >= 0.3 and len(fits) > 0

        result.duration_ms["total"] = sum(v for v in result.duration_ms.values() if isinstance(v, int))

        return result

    def _generate_final_coreops(
        self,
        mesh: MeshData,
        fits: list[PrimitiveFit],
        filtered: list[FilteredPart],
    ) -> dict:
        """Convert PrimitiveFits to CoreOps JSON for CADAgent execution."""
        ops = []
        for fit in fits:
            if fit.chosen_primitive == PrimitiveType.MESH:
                ops.append({
                    "op": "mesh",
                    "params": {"note": "primitive_fit_failed"},
                    "region_id": fit.region_id,
                })
                continue

            params = fit.fit_result.parameters
            placement = fit.fit_result.placement

            if fit.chosen_primitive == PrimitiveType.CYLINDER:
                ops.append({
                    "op": "cylinder",
                    "params": {
                        "radius": params.get("radius_mm"),
                        "height": params.get("height_mm"),
                    },
                    "transform": {
                        "center": placement.get("center"),
                        "axis": params.get("axis"),
                    },
                    "region_id": fit.region_id,
                    "fit_error_mm": round(fit.fit_result.rms_error, 3),
                })
            elif fit.chosen_primitive == PrimitiveType.BOX:
                ops.append({
                    "op": "box",
                    "params": {"size": params.get("size_mm")},
                    "transform": {
                        "center": placement.get("center"),
                        "axes": params.get("axes"),
                    },
                    "region_id": fit.region_id,
                    "fit_error_mm": round(fit.fit_result.rms_error, 3),
                })
            elif fit.chosen_primitive == PrimitiveType.SPHERE:
                ops.append({
                    "op": "sphere",
                    "params": {"radius": params.get("radius_mm")},
                    "transform": {"center": placement.get("center")},
                    "region_id": fit.region_id,
                    "fit_error_mm": round(fit.fit_result.rms_error, 3),
                })
            elif fit.chosen_primitive == PrimitiveType.PLANE:
                ops.append({
                    "op": "plane",
                    "params": {
                        "normal": params.get("normal"),
                        "extent_uv": params.get("extent_uv_mm"),
                    },
                    "transform": {"center": placement.get("center")},
                    "region_id": fit.region_id,
                    "fit_error_mm": round(fit.fit_result.rms_error, 3),
                })
            elif fit.chosen_primitive == PrimitiveType.CONE:
                ops.append({
                    "op": "cone",
                    "params": params,
                    "transform": placement,
                    "region_id": fit.region_id,
                    "fit_error_mm": round(fit.fit_result.rms_error, 3),
                })
            elif fit.chosen_primitive == PrimitiveType.TORUS:
                ops.append({
                    "op": "torus",
                    "params": params,
                    "transform": placement,
                    "region_id": fit.region_id,
                    "fit_error_mm": round(fit.fit_result.rms_error, 3),
                })

        return {
            "schema_version": "1.0",
            "source_mesh": mesh.source_path,
            "operations": ops,
            "summary": {
                "operation_count": len(ops),
                "primitives_used": list(set(o["op"] for o in ops)),
            },
        }

    def _compute_quality(self, result: DualPassResult) -> tuple[float, list[str]]:
        """Compute overall quality score 0..1."""
        notes = []
        score = 1.0

        if result.pass1_regions_count == 0:
            return 0.0, ["no_regions_decomposed"]

        # Penalty: high noise ratio
        total = sum(result.filtered_summary.values())
        if total > 0:
            noise_ratio = result.filtered_summary.get("noise", 0) / total
            if noise_ratio > 0.5:
                score *= 0.6
                notes.append(f"high_noise_ratio={noise_ratio:.0%}")

        # Bonus: primitive fits dominate
        prim_total = sum(v for k, v in result.primitive_summary.items() if k != "mesh")
        mesh_count = result.primitive_summary.get("mesh", 0)
        if prim_total + mesh_count > 0:
            primitive_ratio = prim_total / (prim_total + mesh_count)
            score *= (0.5 + 0.5 * primitive_ratio)  # 50% → 100% scaling
            if primitive_ratio < 0.3:
                notes.append(f"low_primitive_ratio={primitive_ratio:.0%}")

        # Penalty: failed pass 1
        if result.pass1_features_count == 0:
            score *= 0.3
            notes.append("zero_features_extracted")

        return max(0.0, min(1.0, score)), notes


def run_dual_pass(
    mesh_path: Optional[str] = None,
    mesh_bytes: Optional[bytes] = None,
    **kwargs
) -> DualPassResult:
    """Convenience wrapper."""
    orch = DualPassOrchestrator()
    return orch.run(mesh_path=mesh_path, mesh_bytes=mesh_bytes, **kwargs)
