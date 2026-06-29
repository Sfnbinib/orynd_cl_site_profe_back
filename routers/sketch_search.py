"""/search/sketch — sketch/photo → 3D model search (founder pick).

Local-first vision: uses an Ollama vision model (llava / llama3.2-vision /
moondream / bakllava) to describe the sketch, then runs the standard
retrieval search on the description.

Honest stub when no local vision model is installed: responds with
vision_available=false and install instructions instead of pretending.

* GET  /search/sketch/status — is a local vision model available?
* POST /search/sketch        — multipart image upload → candidates
"""

from __future__ import annotations

import base64
import logging
import os
import uuid
from typing import Any, Optional

import httpx
from fastapi import APIRouter, File, Form, UploadFile

from orynd_core.agents.base import AgentContext
from orynd_core.agents.orchestrator import Pipeline
from orynd_core.agents.retrieval import RetrievalAgent

log = logging.getLogger(__name__)
router = APIRouter(prefix="/search/sketch", tags=["sketch-search"])

VISION_MODEL_PREFIXES = ("llava", "llama3.2-vision", "moondream", "bakllava", "minicpm-v")

_DESCRIBE_PROMPT = (
    "This is an engineering sketch or photo of a mechanical part. "
    "Describe the part in one short search query (max 8 words): part name, "
    "key features, mounting style. Answer with the query only."
)


def _ollama_url() -> str:
    return os.getenv("OLLAMA_URL", "http://localhost:11434")


async def _find_vision_model() -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{_ollama_url()}/api/tags")
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return None
    for name in models:
        if any(name.startswith(p) for p in VISION_MODEL_PREFIXES):
            return name
    return None


async def _describe_image(model: str, image_b64: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{_ollama_url()}/api/generate",
            json={
                "model": model,
                "prompt": _DESCRIBE_PROMPT,
                "images": [image_b64],
                "stream": False,
            },
        )
        r.raise_for_status()
        return str(r.json().get("response", "")).strip()


@router.get("/status")
async def status() -> dict[str, Any]:
    model = await _find_vision_model()
    return {
        "vision_available": model is not None,
        "model": model,
        "hint": None if model else
            "Install a local vision model: `ollama pull llava` (or llama3.2-vision)",
    }


@router.post("")
async def sketch_search(
    image: UploadFile = File(...),
    session_id: str = Form("sketch"),
    limit: int = Form(5),
) -> dict[str, Any]:
    model = await _find_vision_model()
    if model is None:
        return {
            "vision_available": False,
            "candidates": [],
            "query": None,
            "message": "No local vision model installed. Run `ollama pull llava` "
                       "to enable sketch search — images never leave this machine.",
        }

    raw = await image.read()
    image_b64 = base64.b64encode(raw).decode()

    try:
        query = await _describe_image(model, image_b64)
    except Exception as exc:
        log.warning("[sketch] vision describe failed: %s", exc)
        return {
            "vision_available": True,
            "candidates": [],
            "query": None,
            "message": f"Vision model error: {exc}",
        }

    ctx = AgentContext(session_id=session_id or str(uuid.uuid4()), raw_text=query)
    ctx.intent = {"keywords": query.split(), "object_name": query, "_source": "sketch"}
    await Pipeline([RetrievalAgent(limit=limit)]).run(ctx)

    candidates = [
        c.model_dump() if hasattr(c, "model_dump") else c
        for c in ctx.candidates[:limit]
    ]
    return {
        "vision_available": True,
        "model": model,
        "query": query,
        "candidates": candidates,
    }
