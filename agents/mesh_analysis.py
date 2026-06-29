"""
MeshAnalysisAgent — Pipeline B of AI Model 4.

Takes a mesh file (STL/OBJ) or mesh bytes, runs:
  1. Load mesh → MeshData
  2. Decompose → surface regions
  3. Extract features → CoreOps JSON
  4. (Optional) LLM enrichment — describe features in natural language

This agent works WITHOUT LLM (algorithm-only path).
LLM is used only for optional enrichment / ambiguity resolution.

Usage:
    agent = MeshAnalysisAgent()
    result = await agent.run(ctx)
    # result.data["coreops_json"] → CoreOps-compatible feature set
"""
from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from .base import BaseAgent, AgentContext, AgentResult
from orynd_core.services.mesh.loader import load_mesh, load_mesh_from_bytes, MeshData
from orynd_core.services.mesh.decomposer import decompose_mesh, DecompositionResult
from orynd_core.services.mesh.feature_extractor import extract_features, FeatureExtractionResult

log = logging.getLogger(__name__)

# Local action log (B11) — used until Supabase ingestion is wired up.
# Override via env to send elsewhere or to /dev/null to disable.
_ACTION_LOG_PATH = os.environ.get("ORYND_ACTION_LOG", "/tmp/orynd_actions.jsonl")


def _log_action(record: dict) -> None:
    """Append-only JSONL log of mesh-pipeline actions for future training."""
    if _ACTION_LOG_PATH in ("", "/dev/null"):
        return
    try:
        record = {"ts": time.time(), **record}
        with open(_ACTION_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.debug("[mesh_analysis] action log write failed: %s", e)


def _quality_marker(extraction: FeatureExtractionResult) -> dict:
    """B12 — summarize confidence distribution across detected features."""
    confs = [float(f.confidence) for f in extraction.features]
    if not confs:
        return {
            "feature_count": 0,
            "confident_count": 0,
            "uncertain_count": 0,
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "overall": "empty",
        }
    confident = sum(1 for c in confs if c >= 0.75)
    uncertain = sum(1 for c in confs if c < 0.5)
    mean = sum(confs) / len(confs)

    if mean >= 0.75 and uncertain == 0:
        overall = "high"
    elif mean >= 0.6:
        overall = "medium"
    else:
        overall = "low"

    return {
        "feature_count": len(confs),
        "confident_count": confident,
        "uncertain_count": uncertain,
        "mean_confidence": round(mean, 3),
        "min_confidence": round(min(confs), 3),
        "max_confidence": round(max(confs), 3),
        "overall": overall,
    }


class MeshAnalysisAgent(BaseAgent):
    """
    AI Model 4 — Mesh Pipeline (Pipeline B).

    Input (via ctx.extra):
        mesh_path: str           — path to STL/OBJ/PLY file
        OR mesh_bytes: bytes     — raw mesh data
        mesh_format: str         — file format hint (default "stl")
        mesh_scale: float        — coordinate scale (1.0 = mm, 25.4 = inch→mm)
        decompose_angle: float   — region growing threshold (default 15°)
        decompose_min_faces: int — min faces per region (default 5)

    Output (in result.data):
        mesh_info: dict          — vertex count, face count, bbox, watertight, etc.
        regions: list[dict]      — decomposed surface regions
        features: list[dict]     — extracted manufacturing features
        coreops_json: dict       — full CoreOps-compatible output
        decomposition_stats: dict
    """

    name = "mesh_analysis"

    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        # ── Step 0: Get mesh input ──
        mesh_path = ctx.extra.get("mesh_path")
        mesh_bytes = ctx.extra.get("mesh_bytes")
        mesh_format = ctx.extra.get("mesh_format", "stl")
        scale = ctx.extra.get("mesh_scale", 1.0)
        angle = ctx.extra.get("decompose_angle", 15.0)
        min_faces = ctx.extra.get("decompose_min_faces", 5)
        repair = bool(ctx.extra.get("repair", False))
        auto_scale = bool(ctx.extra.get("auto_scale", False))
        denoise = bool(ctx.extra.get("denoise", False))
        smooth_iters = int(ctx.extra.get("denoise_smooth_iters", 0))

        # Denoise bumps min_faces unless caller explicitly set it higher.
        # Skip the bump on tiny meshes — would force aggressive region merging
        # and can hit pre-existing cycle-in-chain bugs in _merge_small_regions.

        if not mesh_path and not mesh_bytes:
            return AgentResult.failure(
                self.name,
                "No mesh input: set ctx.extra['mesh_path'] or ctx.extra['mesh_bytes']"
            )

        # ── Step 1: Load mesh ──
        log.info(f"[{self.name}] Loading mesh (repair={repair}, auto_scale={auto_scale})...")
        try:
            if mesh_path:
                mesh = load_mesh(mesh_path, scale=scale, repair=repair, auto_scale=auto_scale)
            else:
                mesh = load_mesh_from_bytes(
                    mesh_bytes, file_type=mesh_format, scale=scale,
                    repair=repair, auto_scale=auto_scale,
                )
        except Exception as e:
            return AgentResult.failure(self.name, f"Mesh load failed: {e}")

        # Now that mesh is loaded, conditionally bump min_faces.
        # Need enough headroom: bump only if the mesh has at least min_faces * 6
        # triangles (so the box-of-12 case doesn't force every region to merge).
        if denoise and min_faces < 10 and mesh.triangle_count >= 60:
            min_faces = 10

        # ── Step 1b: Optional smoothing for noisy scans ──
        if denoise and smooth_iters > 0 and mesh._trimesh is not None:
            try:
                import trimesh.smoothing as _sm
                _sm.filter_taubin(mesh._trimesh, iterations=smooth_iters)
                # Re-sync arrays after in-place smoothing
                import numpy as _np
                mesh.vertices = _np.asarray(mesh._trimesh.vertices, dtype=_np.float64)
                mesh.face_normals = _np.asarray(mesh._trimesh.face_normals, dtype=_np.float64)
                log.info(f"[{self.name}] Applied Taubin smoothing × {smooth_iters}")
            except Exception as e:
                log.warning(f"[{self.name}] smoothing failed: {e}")

        mesh_info = {
            "source": mesh.source_path,
            "format": mesh.source_format,
            "vertices": mesh.vertex_count,
            "triangles": mesh.triangle_count,
            "bbox_min": mesh.bbox_min.tolist(),
            "bbox_max": mesh.bbox_max.tolist(),
            "size_mm": mesh.size_mm().tolist(),
            "diagonal_mm": round(mesh.diagonal_mm(), 2),
            "is_watertight": mesh.is_watertight,
            "volume_mm3": round(mesh.volume_mm3, 2),
            "surface_area_mm2": round(mesh.surface_area_mm2, 2),
        }
        log.info(
            f"[{self.name}] Loaded: {mesh.vertex_count} verts, "
            f"{mesh.triangle_count} tris, size={mesh.size_mm().round(1)}"
        )

        # ── Step 2: Decompose ──
        log.info(f"[{self.name}] Decomposing (angle={angle}°, min_faces={min_faces})...")
        try:
            decomposition = decompose_mesh(
                mesh,
                angle_threshold_deg=angle,
                min_region_faces=min_faces,
            )
        except Exception as e:
            return AgentResult.failure(self.name, f"Decomposition failed: {e}")

        regions_data = [r.to_dict() for r in decomposition.regions]
        log.info(f"[{self.name}] Decomposed into {len(decomposition.regions)} regions")

        # ── Step 3: Extract features ──
        log.info(f"[{self.name}] Extracting features...")
        try:
            extraction = extract_features(mesh, decomposition)
        except Exception as e:
            return AgentResult.failure(self.name, f"Feature extraction failed: {e}")

        # ── Step 3b: Denoise — drop low-confidence noise features ──
        if denoise:
            mesh_diag = mesh.diagonal_mm()
            noise_area_threshold = max(1.0, mesh_diag * 0.005) ** 2  # ~0.5% of diag squared
            before = len(extraction.features)
            extraction.features = [
                f for f in extraction.features
                if not (f.confidence < 0.45 and f.area_mm2 < noise_area_threshold)
            ]
            extraction.total_features = len(extraction.features)
            extraction.feature_summary = {}
            for f in extraction.features:
                t = f.feature_type.value
                extraction.feature_summary[t] = extraction.feature_summary.get(t, 0) + 1
            log.info(f"[{self.name}] Denoise: {before} → {extraction.total_features} features")

        coreops_json = extraction.to_coreops_json()
        quality = _quality_marker(extraction)
        coreops_json["quality"] = quality
        log.info(
            f"[{self.name}] Extracted {extraction.total_features} features "
            f"(quality={quality['overall']}, mean_conf={quality['mean_confidence']}): "
            f"{extraction.feature_summary}"
        )

        # B11: action log for future Movement Engine training
        _log_action({
            "session_id": ctx.session_id,
            "user_id": ctx.user_id,
            "agent": self.name,
            "source": mesh_info["source"],
            "format": mesh_info["format"],
            "triangles": mesh_info["triangles"],
            "regions": len(regions_data),
            "features": extraction.total_features,
            "feature_summary": extraction.feature_summary,
            "quality": quality,
            "scale_hint": mesh.scale_hint,
            "denoise": denoise,
            "repair": repair,
        })

        # ── Step 4: Optional LLM enrichment ──
        llm_description = None
        if self.provider and ctx.extra.get("enrich_with_llm", False):
            llm_description = await self._llm_enrich(mesh_info, coreops_json)
            if llm_description:
                coreops_json["llm_description"] = llm_description

        # ── Store in context for downstream agents ──
        ctx.extra["mesh_data"] = mesh
        ctx.extra["mesh_decomposition"] = decomposition
        ctx.extra["mesh_features"] = extraction
        ctx.extra["coreops_json"] = coreops_json

        return AgentResult.success(
            self.name,
            data={
                "mesh_info": mesh_info,
                "regions": regions_data,
                "regions_count": len(regions_data),
                "features": [f.to_dict() for f in extraction.features],
                "features_count": extraction.total_features,
                "feature_summary": extraction.feature_summary,
                "coreops_json": coreops_json,
                "decomposition_stats": decomposition.stats,
                "llm_description": llm_description,
                "quality": quality,
                "scale_hint": mesh.scale_hint,
            },
        )

    async def _llm_enrich(self, mesh_info: dict, coreops: dict) -> Optional[str]:
        """Use LLM to describe the part in natural language."""
        if not self.provider:
            return None

        features_text = []
        for f in coreops.get("features", []):
            features_text.append(
                f"  - {f['feature_type']}: {f.get('dimensions_mm', {})} "
                f"(confidence: {f.get('confidence', 0):.0%})"
            )

        prompt = f"""Analyze this 3D part from its extracted features.

Mesh info:
  Size: {mesh_info.get('size_mm')} mm
  Triangles: {mesh_info.get('triangles')}
  Watertight: {mesh_info.get('is_watertight')}
  Volume: {mesh_info.get('volume_mm3')} mm³

Detected features ({len(features_text)}):
{chr(10).join(features_text)}

Provide:
1. What this part likely is (bracket, housing, adapter, etc.)
2. Key manufacturing considerations
3. Suggested manufacturing method (FDM, SLA, CNC, injection molding)

Be concise — 3-5 sentences max."""

        try:
            response = await self.provider.generate(prompt)
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            log.warning(f"[{self.name}] LLM enrichment failed: {e}")
            return None
