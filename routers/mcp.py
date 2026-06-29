"""/mcp — MCP server over Streamable HTTP (feature 48, ФАЗА B).

JSON-RPC 2.0 on POST /mcp per the MCP spec (stateless variant: every
response is a single application/json body — valid per spec, no SSE
needed for request/response tools).

Supported methods: initialize, notifications/initialized, ping,
tools/list, tools/call.

Auth: when ORYND_MCP_TOKEN env is set, requests MUST carry
``Authorization: Bearer <token>``; unset = open (local dev / same machine).
Full JWT (web-client tokens) lands with the license server phase.

Connect from Claude Code:
    claude mcp add orynd --transport http http://127.0.0.1:8765/mcp
GET /mcp/tools is a plain-REST discovery convenience (curl-friendly).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from orynd_core.mcp.tools import TOOLS, execute_tool

log = logging.getLogger(__name__)
router = APIRouter(prefix="/mcp", tags=["mcp"])

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "orynd-core", "title": "ORYND Workspace", "version": "0.2.0"}

# Cap tool output so a chatty endpoint can't blow up the client's context.
MAX_RESULT_CHARS = 60_000


def _auth_error(request: Request) -> JSONResponse | None:
    expected = os.getenv("ORYND_MCP_TOKEN", "")
    if not expected:
        return None
    got = request.headers.get("Authorization", "")
    if got == f"Bearer {expected}":
        return None
    return JSONResponse(status_code=401, content={
        "jsonrpc": "2.0", "id": None,
        "error": {"code": -32001, "message": "unauthorized: bad or missing bearer token"},
    })


def _result(rpc_id: Any, result: dict) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": result})


def _error(rpc_id: Any, code: int, message: str) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}})


@router.post("")
async def mcp_rpc(request: Request) -> Response:
    denied = _auth_error(request)
    if denied is not None:
        return denied

    try:
        payload = await request.json()
    except Exception:
        return _error(None, -32700, "parse error: body is not JSON")
    if not isinstance(payload, dict):
        return _error(None, -32600, "batch requests not supported")

    method = payload.get("method", "")
    rpc_id = payload.get("id")
    params = payload.get("params") or {}

    # Notifications (no id) → acknowledge with 202, no body
    if rpc_id is None and method.startswith("notifications/"):
        return Response(status_code=202)

    if method == "initialize":
        client_version = str(params.get("protocolVersion", PROTOCOL_VERSION))
        return _result(rpc_id, {
            "protocolVersion": client_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": (
                "ORYND engineering workspace tools: search 3D models, decompose "
                "meshes to CAD, execute CoreOps, run deep research, check standards. "
                "Typical chain: macro_parse → cad_execute → standards_check."
            ),
        })

    if method == "ping":
        return _result(rpc_id, {})

    if method == "tools/list":
        return _result(rpc_id, {"tools": [t.manifest() for t in TOOLS]})

    if method == "tools/call":
        name = str(params.get("name", ""))
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            return _error(rpc_id, -32602, "'arguments' must be an object")
        try:
            is_error, data = await execute_tool(name, args)
        except Exception as exc:  # tool crashed — report inside the protocol
            log.exception("[mcp] tool %s crashed", name)
            is_error, data = True, {"error": str(exc)}
        text = json.dumps(data, ensure_ascii=False, default=str)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"… [truncated, {len(text)} chars total]"
        return _result(rpc_id, {
            "content": [{"type": "text", "text": text}],
            "isError": bool(is_error),
        })

    return _error(rpc_id, -32601, f"method not found: {method}")


@router.get("/tools")
async def discovery() -> dict[str, Any]:
    """Plain-REST tool discovery (curl-friendly; the MCP way is tools/list)."""
    return {"server": SERVER_INFO, "count": len(TOOLS), "tools": [t.manifest() for t in TOOLS]}
