"""
Mesh Analysis Router — API endpoint for Pipeline B (AI Model 4).

POST /mesh/analyze — analyze STL/OBJ mesh → features → CoreOps JSON
"""
from __future__ import annotations
import logging
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body

from orynd_core.models.schemas import MeshAnalysisRequest, MeshAnalysisResponse, MeshFeatureOut
from orynd_core.agents.mesh_analysis import MeshAnalysisAgent
from orynd_core.agents.base import AgentContext

log = logging.getLogger(__name__)
router = APIRouter(prefix="/mesh", tags=["mesh"])

# Mesh download limits
_MESH_DOWNLOAD_TIMEOUT_S = 30
_MESH_DOWNLOAD_MAX_BYTES = 100 * 1024 * 1024  # 100 MB
_MESH_DOWNLOAD_ALLOWED_EXT = {".stl", ".obj", ".ply", ".3mf", ".step", ".stp"}


def _download_mesh_to_tmp(url: str) -> tuple[str, str]:
    """
    Download a mesh from HTTPS URL to a temp file.
    Returns (tmp_path, extension). Caller must delete the file.

    Raises HTTPException on validation/network errors.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise HTTPException(400, f"mesh_url must use https scheme (got: {parsed.scheme!r})")
    if not parsed.netloc:
        raise HTTPException(400, "mesh_url is missing host")

    # Pick extension from URL path; default to .stl
    ext = Path(parsed.path).suffix.lower()
    if ext and ext not in _MESH_DOWNLOAD_ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported mesh extension: {ext}")
    if not ext:
        ext = ".stl"

    # Streamed download with size cap
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "ORYND/1.0 (+mesh-analyze)"})
        with urllib.request.urlopen(req, timeout=_MESH_DOWNLOAD_TIMEOUT_S) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > _MESH_DOWNLOAD_MAX_BYTES:
                raise HTTPException(413, f"Mesh too large: {content_length} bytes (max {_MESH_DOWNLOAD_MAX_BYTES})")

            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            try:
                total = 0
                chunk_size = 64 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _MESH_DOWNLOAD_MAX_BYTES:
                        tmp.close()
                        Path(tmp.name).unlink(missing_ok=True)
                        raise HTTPException(413, f"Mesh exceeds max size {_MESH_DOWNLOAD_MAX_BYTES} bytes during download")
                    tmp.write(chunk)
                tmp.close()
                return tmp.name, ext.lstrip(".")
            except Exception:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise
    except HTTPException:
        raise
    except TimeoutError:
        raise HTTPException(504, f"Mesh download timed out after {_MESH_DOWNLOAD_TIMEOUT_S}s")
    except Exception as e:
        log.warning("[mesh] download failed: %s", e)
        raise HTTPException(502, f"Mesh download failed: {e}")


@router.post("/analyze", response_model=MeshAnalysisResponse)
async def analyze_mesh(req: MeshAnalysisRequest):
    """
    Analyze a mesh file through Pipeline B.

    Provide either mesh_path (local file) or mesh_url (to download).
    Returns decomposed regions, extracted features, and CoreOps JSON.
    """
    if not req.mesh_path and not req.mesh_url:
        raise HTTPException(400, "Provide mesh_path or mesh_url")

    mesh_path = req.mesh_path
    mesh_format = req.mesh_format
    downloaded_tmp: str | None = None

    if req.mesh_url:
        downloaded_tmp, ext = _download_mesh_to_tmp(req.mesh_url)
        mesh_path = downloaded_tmp
        mesh_format = mesh_format or ext

    if mesh_path and not Path(mesh_path).exists():
        raise HTTPException(404, f"Mesh file not found: {mesh_path}")

    # Build agent context
    ctx = AgentContext(session_id=req.session_id)
    ctx.extra = {
        "mesh_path": mesh_path,
        "mesh_format": mesh_format,
        "mesh_scale": req.scale,
        "decompose_angle": req.decompose_angle,
        "decompose_min_faces": req.decompose_min_faces,
        "enrich_with_llm": req.enrich_with_llm,
    }

    try:
        agent = MeshAnalysisAgent()
        result = await agent.run(ctx)
    finally:
        if downloaded_tmp:
            Path(downloaded_tmp).unlink(missing_ok=True)

    if not result.ok:
        raise HTTPException(500, f"Mesh analysis failed: {result.error}")

    data = result.data

    # Build response
    features_out = []
    for f in data.get("features", []):
        features_out.append(MeshFeatureOut(
            feature_id=f["feature_id"],
            feature_type=f["feature_type"],
            surface_type=f["surface_type"],
            region_ids=f.get("region_ids", []),
            position=f.get("position", [0, 0, 0]),
            direction=f.get("direction", [0, 0, 0]),
            dimensions_mm=f.get("dimensions_mm", {}),
            area_mm2=f.get("area_mm2", 0),
            confidence=f.get("confidence", 0),
            is_through=f.get("is_through", False),
            is_blind=f.get("is_blind", False),
            depth_mm=f.get("depth_mm", 0),
        ))

    return MeshAnalysisResponse(
        session_id=req.session_id,
        mesh_info=data.get("mesh_info", {}),
        regions_count=data.get("regions_count", 0),
        regions=data.get("regions", []),
        features_count=data.get("features_count", 0),
        features=features_out,
        feature_summary=data.get("feature_summary", {}),
        coreops_json=data.get("coreops_json", {}),
        llm_description=data.get("llm_description"),
        decomposition_stats=data.get("decomposition_stats", {}),
    )


@router.post("/analyze/upload", response_model=MeshAnalysisResponse)
async def analyze_mesh_upload(
    file: UploadFile = File(...),
    session_id: str = Form("anonymous"),
    scale: float = Form(1.0),
    decompose_angle: float = Form(15.0),
    decompose_min_faces: int = Form(5),
):
    """
    Upload a mesh file directly for analysis.
    Accepts STL, OBJ, PLY files.
    """
    allowed = {".stl", ".obj", ".ply", ".3mf", ".step", ".stp"}
    ext = Path(file.filename or "model.stl").suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {allowed}")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Empty file")

    # Save to temp file (mesh loader needs a path for manual STL fallback)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    ctx = AgentContext(session_id=session_id)
    ctx.extra = {
        "mesh_path": tmp_path,
        "mesh_format": ext.lstrip("."),
        "mesh_scale": scale,
        "decompose_angle": decompose_angle,
        "decompose_min_faces": decompose_min_faces,
    }

    agent = MeshAnalysisAgent()
    result = await agent.run(ctx)

    # Cleanup temp
    try:
        Path(tmp_path).unlink()
    except Exception:
        pass

    if not result.ok:
        raise HTTPException(500, f"Mesh analysis failed: {result.error}")

    data = result.data
    features_out = []
    for f in data.get("features", []):
        features_out.append(MeshFeatureOut(
            feature_id=f["feature_id"],
            feature_type=f["feature_type"],
            surface_type=f["surface_type"],
            region_ids=f.get("region_ids", []),
            position=f.get("position", [0, 0, 0]),
            direction=f.get("direction", [0, 0, 0]),
            dimensions_mm=f.get("dimensions_mm", {}),
            area_mm2=f.get("area_mm2", 0),
            confidence=f.get("confidence", 0),
            is_through=f.get("is_through", False),
            is_blind=f.get("is_blind", False),
            depth_mm=f.get("depth_mm", 0),
        ))

    return MeshAnalysisResponse(
        session_id=session_id,
        mesh_info=data.get("mesh_info", {}),
        regions_count=data.get("regions_count", 0),
        regions=data.get("regions", []),
        features_count=data.get("features_count", 0),
        features=features_out,
        feature_summary=data.get("feature_summary", {}),
        coreops_json=data.get("coreops_json", {}),
        decomposition_stats=data.get("decomposition_stats", {}),
    )


@router.post("/rebuild")
async def rebuild_mesh(payload: dict = Body(...)) -> dict:
    """
    Full STL → clean CAD rebuild: download mesh → AI Model 4 dual-pass →
    CADAgent → downloadable STEP/STL/OBJ.

    Body: {"mesh_url": "https://...stl", "session_id": "..."} or
          {"mesh_path": "/local.stl", "session_id": "..."}.

    Returns step_url/stl_url (served by GET /cad/model/{session_id}/{file}).
    """
    from orynd_core.agents.ai_model_4.cad_bridge import run_dual_pass_to_cad

    mesh_url = payload.get("mesh_url")
    mesh_path = payload.get("mesh_path")
    if not mesh_url and not mesh_path:
        raise HTTPException(400, "Provide mesh_url or mesh_path")
    session_id = str(payload.get("session_id", "ai_model_4_cad"))

    downloaded_tmp: str | None = None
    if mesh_url:
        downloaded_tmp, _ext = _download_mesh_to_tmp(mesh_url)
        mesh_path = downloaded_tmp

    try:
        result = await run_dual_pass_to_cad(mesh_path, session_id=session_id)
        rd = result.to_dict() if hasattr(result, "to_dict") else {}
        cad_out = getattr(result, "cad_output", {}) or {}
        return {
            "ok": getattr(result, "ok", False),
            "session_id": session_id,
            "error": getattr(result, "error", None),
            "translation_notes": getattr(result, "translation_notes", []),
            "files": {
                "stl": f"/cad/model/{session_id}/part.stl" if cad_out.get("stl_path") else None,
                "step": f"/cad/model/{session_id}/part.step" if cad_out.get("step_path") else None,
                "obj": f"/cad/model/{session_id}/part.obj" if cad_out.get("obj_path") else None,
            },
            "dual_pass": rd.get("dual_pass", {}),
        }
    finally:
        if downloaded_tmp:
            Path(downloaded_tmp).unlink(missing_ok=True)
