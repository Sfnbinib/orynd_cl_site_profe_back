"""FastAPI middleware + exception handlers.

Hooks attached by :func:`install_middleware` (called from ``api/main.py``):
* trace_id propagation (``X-Trace-Id`` in/out, structlog contextvar)
* request metrics (count + duration histogram)
* :class:`OryndError` → structured JSON envelope
* Unhandled :class:`Exception` → 500 envelope (no stack to client)
"""

from __future__ import annotations

import time
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from orynd_core.errors import OryndError
from orynd_core.services.logging import get_logger
from orynd_core.services.observability.metrics import (
    http_request_duration_seconds,
    http_requests_total,
)

log = get_logger("orynd.middleware")


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return route.path
    return request.url.path


def install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def trace_and_metrics_middleware(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or uuid4().hex
        token = structlog.contextvars.bind_contextvars(trace_id=trace_id)
        start = time.time()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Trace-Id"] = trace_id
            return response
        finally:
            duration = time.time() - start
            route = _route_path(request)
            try:
                http_requests_total.labels(
                    method=request.method,
                    route=route,
                    status=str(status_code),
                ).inc()
                http_request_duration_seconds.labels(route=route).observe(duration)
            except Exception:
                pass
            structlog.contextvars.unbind_contextvars("trace_id")

    app.add_exception_handler(OryndError, orynd_exception_handler)
    app.add_exception_handler(Exception, unexpected_exception_handler)


async def orynd_exception_handler(request: Request, exc: OryndError) -> JSONResponse:
    trace_id = structlog.contextvars.get_contextvars().get("trace_id")
    log.error(
        exc.code,
        message=str(exc),
        details=exc.details,
        path=request.url.path,
        exc_info=True,
    )
    return JSONResponse(
        status_code=exc.http_status,
        content={
            "error": {
                "code": exc.code,
                "message": exc.user_message,
                "details": exc.details,
                "trace_id": trace_id,
            }
        },
        headers={"X-Trace-Id": trace_id} if trace_id else {},
    )


async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    trace_id = structlog.contextvars.get_contextvars().get("trace_id")
    log.critical(
        "unexpected_error",
        path=request.url.path,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "orynd.unexpected",
                "message": "An unexpected error occurred",
                "trace_id": trace_id,
            }
        },
        headers={"X-Trace-Id": trace_id} if trace_id else {},
    )


__all__ = [
    "install_middleware",
    "orynd_exception_handler",
    "unexpected_exception_handler",
]
