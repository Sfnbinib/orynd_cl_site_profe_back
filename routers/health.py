"""GET /health — liveness check."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0", "core": "orynd_core"}
