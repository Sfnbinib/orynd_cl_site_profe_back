"""System endpoints — health, deep health, metrics, version, telemetry sink.

Spec: CONNECTIONS_AND_INTEGRATION.md § Health checks +
MONITORING_OBSERVABILITY.md § Metrics endpoint.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Body, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from orynd_core.services.connection_pool import get_ollama_client
from orynd_core.services.logging import get_logger
from orynd_core.services.observability.metrics import refresh_circuit_metrics
from orynd_core.services.resilience.circuit_breaker import breakers

router = APIRouter(prefix="/system", tags=["system"])
log = get_logger("orynd.system")

_BOOT_TS = time.time()
_APP_VERSION = "0.2.0"


@router.get("/health")
async def health() -> dict[str, Any]:
    """Liveness — process is up. Cheap, called every 30s by the renderer."""
    return {
        "status": "ok",
        "version": _APP_VERSION,
        "ts": datetime.now(timezone.utc).isoformat(),
        "uptime_s": int(time.time() - _BOOT_TS),
    }


@router.get("/health/deep")
async def deep_health() -> dict[str, Any]:
    """Readiness — checks each dependency. Slower; used by the dev dashboard."""
    checks: dict[str, Any] = {
        "backend": "ok",
        "ollama": await _check_ollama(),
        "supabase": _check_supabase(),
        "circuits": {name: cb.state.value for name, cb in breakers.items()},
    }
    flat_values: list[str] = []
    for v in checks.values():
        if isinstance(v, dict):
            flat_values.extend(v.values())
        else:
            flat_values.append(v)
    healthy_set = {"ok", "closed", "skipped"}
    overall = "ok" if all(v in healthy_set for v in flat_values) else "degraded"
    return {"status": overall, "checks": checks}


@router.get("/version")
async def version() -> dict[str, Any]:
    return {
        "app": _APP_VERSION,
        "schema_version": 1,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/metrics")
async def metrics() -> Response:
    refresh_circuit_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/telemetry")
async def telemetry_sink(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Receives batched frontend telemetry events.

    Phase 0: just logs the batch (PII strip happens client-side before send).
    Phase 14: forward to analytics sink.
    """
    events = payload.get("events", [])
    log.info("telemetry.received", count=len(events))
    return {"accepted": len(events)}


# ---- check helpers ---------------------------------------------------------


async def _check_ollama() -> str:
    if os.environ.get("ORYND_SKIP_OLLAMA_CHECK"):
        return "skipped"
    try:
        client = get_ollama_client()
        resp = await client.get("/api/tags", timeout=2.0)
        return "ok" if resp.status_code == 200 else f"unhealthy:{resp.status_code}"
    except (httpx.ConnectError, httpx.TimeoutException):
        return "unreachable"
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def _check_supabase() -> str:
    if not os.environ.get("SUPABASE_URL"):
        return "skipped"
    return "ok"
