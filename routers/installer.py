"""/installer/* — Drop 2 lifecycle + Ollama detection for the Electron shell.

Used by Electron main process via HTTP (not via IPC directly) so the same
endpoints can serve dev (browser inspector) and prod (Electron renderer).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, HTTPException

from pathlib import Path

from orynd_core.installer.dependency_drop_2 import (
    Artifact,
    DependencyDrop2Manager,
    load_manifest,
)
from orynd_core.installer.ollama_check import (
    check_ollama,
    plan_download_skip,
)
from orynd_core.services.background_tasks import manager as task_manager
from orynd_core.services.logging import get_logger

router = APIRouter(prefix="/installer", tags=["installer"])
log = get_logger("orynd.installer.router")

# Process-wide manager + last task handle so we can query progress.
_drop2: Optional[DependencyDrop2Manager] = None
_current_task_id: Optional[str] = None


def _get_drop2() -> DependencyDrop2Manager:
    global _drop2
    if _drop2 is None:
        _drop2 = DependencyDrop2Manager()
    return _drop2


@router.get("/ollama/check")
async def check_ollama_endpoint() -> dict:
    """Detect local Ollama install + list installed models. Never raises.

    Used by onboarding to skip downloading models the user already has
    (founder explicit ask FINAL_DECISIONS_2026-06-02 § H3).
    """
    status = await check_ollama()
    return status.to_dict()


@router.post("/ollama/plan-skip")
async def plan_ollama_skip(
    payload: dict = Body(default_factory=dict),
) -> dict:
    """Given list of required model names → decide skip vs download.

    Body: ``{"required": ["llama3.2:3b", "nomic-embed-text"]}``
    Returns ``{"skip": [...], "download": [...], "bytes_saved": N}``.
    """
    required = payload.get("required") or []
    if not isinstance(required, list):
        raise HTTPException(status_code=422, detail="'required' must be a list of model names")
    status = await check_ollama()
    plan = plan_download_skip(status, [str(r) for r in required])
    return {"ollama_reachable": status.reachable, **plan}


@router.get("/drop2/manifest")
async def drop2_manifest() -> dict:
    """Current artifact manifest (remote when ORYND_DROP2_MANIFEST_URL is set,
    bundled fallback otherwise)."""
    artifacts, source = await load_manifest()
    return {
        "source": source,
        "artifacts": [
            {
                "name": a.name,
                "url": a.url,
                "sha256": a.sha256,
                "size_bytes": a.size_bytes,
                "install_path": str(a.install_path) if a.install_path else None,
            }
            for a in artifacts
        ],
    }


@router.post("/drop2/start")
async def start_drop2(payload: dict = Body(default_factory=dict)) -> dict:
    """Kick off Drop 2 background install.

    Body: ``{"artifacts": [{"name":"...","url":"...","sha256":"...","size_bytes":N}]}``
    Empty/omitted ``artifacts`` → install from the manifest (the normal path;
    explicit list is for dev/testing).
    Returns ``{"task_id": "..."}`` — poll via ``GET /installer/drop2/status``.
    """
    global _current_task_id
    raw_artifacts = payload.get("artifacts") or []
    if not isinstance(raw_artifacts, list):
        raise HTTPException(status_code=422, detail="'artifacts' must be a list")

    source = "request"
    artifacts: list[Artifact] = []
    if raw_artifacts:
        for item in raw_artifacts:
            try:
                artifacts.append(
                    Artifact(
                        name=str(item["name"]),
                        url=str(item["url"]),
                        sha256=str(item["sha256"]),
                        install_path=Path(item["install_path"]).expanduser()
                        if item.get("install_path") else None,
                        size_bytes=int(item.get("size_bytes", 0) or 0),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"bad artifact: {exc}")
    else:
        artifacts, source = await load_manifest()

    drop2 = _get_drop2()
    task = await drop2.start_background(artifacts=artifacts)
    _current_task_id = task.id
    return {"task_id": task.id, "artifact_count": len(artifacts), "manifest_source": source}


@router.get("/drop2/status")
async def drop2_status() -> dict:
    """Return progress + per-artifact state of the currently running Drop 2."""
    if _current_task_id is None:
        return {"task_id": None, "status": "idle"}
    task = task_manager.get(_current_task_id)
    if task is None:
        return {"task_id": _current_task_id, "status": "unknown"}
    drop2 = _get_drop2()
    return {
        "task_id": task.id,
        "status": task.status.value,
        "progress": task.progress,
        "result": task.result if task.status.value == "completed" else None,
        "error": task.error,
        "per_artifact": [
            {
                "artifact": p.artifact,
                "status": p.status.value,
                "bytes_downloaded": p.bytes_downloaded,
                "bytes_total": p.bytes_total,
                "attempt": p.attempt,
            }
            for p in drop2._progress.values()  # noqa: SLF001 — diagnostic readout
        ],
    }
