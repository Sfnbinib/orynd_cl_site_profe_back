"""
ORYND Core — FastAPI entry point.

Run:
  uvicorn orynd_core.api.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load workspace-root .env before routers read os.getenv at import/request time
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from orynd_core.routers import (
    attachment,
    auth,
    billing,
    cad,
    chat,
    credits,
    events,
    fabricate,
    harness,
    health,
    installer,
    learning,
    library,
    macro,
    mcp,
    mesh,
    mesh_manual,
    modes,
    movement,
    multi_context,
    research,
    search,
    select,
    sketch_search,
    skills,
    sources,
    standards,
    system,
)
from orynd_core.services.logging import configure_logging
from orynd_core.services.observability import install_middleware

# Configure structured logging before anything else binds the stdlib root logger.
configure_logging()


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if not origins:
        # Dev fallback only — production must set CORS_ORIGINS.
        origins = ["http://localhost:5173", "http://localhost:8000"]
    # Allow Electron file:// pages — browser sends Origin: null for file:// requests.
    if "null" not in origins:
        origins.append("null")
    return origins


# Rate limiter — applied selectively per-route via @limiter.limit(...).
# Default budget for unauthenticated callers; authenticated routes can override.
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Wire event-bus subscriptions on startup so all blocks are connected."""
    from orynd_core.services.event_bus import bus
    from orynd_core.services.learning.engine import on_action_recorded

    # Wire #3: action.recorded → Learning Engine (theory-vs-practice loop)
    _unsub_learning = bus.subscribe("action.recorded", on_action_recorded)

    # Wire #7: Supabase sink (only when SUPABASE_URL is configured)
    _unsub_sink_action = None
    _unsub_sink_credits = None
    import os as _os
    if _os.getenv("SUPABASE_URL", "").startswith("https://") and "your-project" not in _os.getenv("SUPABASE_URL", ""):
        from orynd_core.services.supabase_sink import on_action_recorded as _sink_action
        from orynd_core.services.supabase_sink import on_credits_consumed as _sink_credits
        _unsub_sink_action = bus.subscribe("action.recorded", _sink_action)
        _unsub_sink_credits = bus.subscribe("credits.consumed", _sink_credits)

    yield  # app running

    # Cleanup
    _unsub_learning()
    if _unsub_sink_action:
        _unsub_sink_action()
    if _unsub_sink_credits:
        _unsub_sink_credits()


app = FastAPI(
    title="ORYND Core",
    description="AI-powered 3D engineering workspace backend",
    version="0.2.0",
    lifespan=lifespan,
)

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(_request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limited", "detail": str(exc.detail)},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With", "X-Trace-Id"],
    expose_headers=["X-Trace-Id"],
)

# Cross-cutting: trace_id middleware + OryndError handler + metrics middleware.
install_middleware(app)

# Public routers (no auth required — health for k8s/electron probe).
app.include_router(health.router)
app.include_router(system.router)

# Auth router — exposes /api/auth/*, internal guards via Depends(current_user).
app.include_router(auth.router)

# Billing + credits — auth-gated via Depends(current_user) inside each router.
app.include_router(billing.router)
app.include_router(credits.router)

# Domain routers — TODO before launch (A6.2): add Depends(current_user) inside each.
app.include_router(search.router)
app.include_router(select.router)
app.include_router(fabricate.router)
app.include_router(chat.router)
app.include_router(cad.router)
app.include_router(mesh.router)
app.include_router(library.router)
app.include_router(sources.router)
app.include_router(skills.router)
app.include_router(harness.router)
app.include_router(modes.router)
app.include_router(movement.router)
app.include_router(installer.router)
app.include_router(mesh_manual.router)
app.include_router(macro.router)
app.include_router(attachment.router)
app.include_router(multi_context.router)
app.include_router(research.router)
app.include_router(learning.router)
app.include_router(standards.router)
app.include_router(sketch_search.router)
app.include_router(mcp.router)
app.include_router(events.router)


@app.get("/")
async def root():
    return {
        "name": "orynd_core",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
    }
