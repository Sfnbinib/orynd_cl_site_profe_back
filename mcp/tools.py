"""MCP tool registry — thin wraps over existing orynd_core HTTP endpoints.

Each tool maps 1:1 onto a REST endpoint and is executed in-process via an
ASGI transport (no network hop, no logic duplication). Adding a tool =
adding a ToolSpec; the REST API stays the single source of truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import httpx

# (method, path, json_body, query_params)
CallPlan = tuple[str, str, dict | None, dict | None]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    build: Callable[[dict], CallPlan]

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def _obj(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }

_AXIS_SCHEMA = _obj(
    {
        "origin": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "direction": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "diameter": {"type": "number"},
        "length": {"type": "number"},
    },
    required=["origin", "direction"],
)

_S = {"type": "string"}
_N = {"type": "number"}
_I = {"type": "integer"}
_OPS = {"type": "array", "items": {"type": "object"}}


_UI_CHAT_SURFACES = {
    "bottom": {"session_id": "design", "workspace_id": "ws-design"},
    "side": {"session_id": "design-side", "workspace_id": "ws-design"},
    "agent": {"session_id": "design-agent", "workspace_id": "ws-design"},
}


def _ui_chat_body(args: dict) -> dict:
    surface = args.get("surface", "bottom")
    config = _UI_CHAT_SURFACES.get(surface)
    if config is None:
        raise ValueError(f"unknown UI chat surface: {surface}")
    workspace_id = args.get("workspace_id", config["workspace_id"])
    session_id = args.get("session_id", config["session_id"])
    context = {
        "workspace_id": workspace_id,
        "surface": surface,
        "mode": args.get("mode", "auto"),
    }
    if "selection" in args:
        context["selection"] = args["selection"]
    return {
        "message": args["message"],
        "session_id": session_id,
        "platform": "desktop",
        "context": context,
    }


TOOLS: list[ToolSpec] = [
    ToolSpec(
        "search_models",
        "Search 3D models across Printables/Thingiverse/MakerWorld/GitHub + web. "
        "Returns ranked candidates with printability scores.",
        _obj({"query": _S, "session_id": _S}, ["query"]),
        lambda a: ("POST", "/search", {"query": a["query"], "session_id": a.get("session_id", "mcp")}, None),
    ),
    ToolSpec(
        "mesh_decompose",
        "Decompose an STL/OBJ mesh (absolute path on this machine) into engineering "
        "primitives via AI Model 4; build_cad=true also rebuilds STEP/STL.",
        _obj({"mesh_path": _S, "build_cad": {"type": "boolean"}, "session_id": _S}, ["mesh_path"]),
        lambda a: ("POST", "/skills/mesh_decompose/invoke",
                   {"mesh_path": a["mesh_path"], "build_cad": a.get("build_cad", True),
                    "session_id": a.get("session_id", "mcp")}, None),
    ),
    ToolSpec(
        "cad_execute",
        "Execute validated CoreOps operations → STEP/STL/OBJ files. "
        "Returns served file URLs under /cad/model/{session_id}/.",
        _obj({"operations": _OPS, "units": _S, "session_id": _S}, ["operations"]),
        lambda a: ("POST", "/cad/execute",
                   {"operations": a["operations"], "units": a.get("units", "mm"),
                    "session_id": a.get("session_id", "mcp")}, None),
    ),
    ToolSpec(
        "cad_append",
        "Build primitives ONTO the session's existing model (multi-object). New "
        "primitives are placed beside existing ones, the whole scene is rebuilt. "
        "Use instead of cad_execute when adding objects rather than starting fresh.",
        _obj({"operations": _OPS, "units": _S, "session_id": _S}, ["operations"]),
        lambda a: ("POST", "/cad/append",
                   {"operations": a["operations"], "units": a.get("units", "mm"),
                    "session_id": a.get("session_id", "mcp")}, None),
    ),
    ToolSpec(
        "cad_modify",
        "Apply a finishing modifier to the current session model and rebuild: "
        "kind='fillet' (round, value=radius mm) or kind='chamfer' (bevel, value=distance mm).",
        _obj({"kind": {"type": "string", "enum": ["fillet", "chamfer"]},
              "value": {"type": "number"}, "session_id": _S}, ["kind"]),
        lambda a: ("POST", "/cad/modify",
                   {"kind": a["kind"], "value": a.get("value", 2.0),
                    "session_id": a.get("session_id", "mcp")}, None),
    ),
    ToolSpec(
        "deep_research",
        "5-phase parallel research (3D models + web + GitHub → filter → synthesis). "
        "Pushes the result article into the Knowledge Library. depth 1=quick 3=deep.",
        _obj({"topic": _S, "depth": _I}, ["topic"]),
        lambda a: ("POST", "/research", {"topic": a["topic"], "depth": a.get("depth", 1)}, None),
    ),
    ToolSpec(
        "fabricate",
        "Recommend fabrication method + material + parameters for a search candidate "
        "(by index) from a prior search_models session.",
        _obj({"session_id": _S, "index": _I}, ["session_id"]),
        lambda a: ("POST", "/fabricate", {"session_id": a["session_id"], "index": a.get("index", 0)}, None),
    ),
    ToolSpec(
        "macro_parse",
        "Natural language → CoreOps operations preview (e.g. 'create 20mm cube; "
        "drill 5mm hole'). Use cad_execute to actually build the result.",
        _obj({"text": _S}, ["text"]),
        lambda a: ("POST", "/macro/parse", {"text": a["text"]}, None),
    ),
    ToolSpec(
        "attach_suggest",
        "Axes attachment: given an axis (origin+direction, optional diameter mm), "
        "return top-9 catalog parts that fit (bolts, bearings, mounts...).",
        _obj({"axis": _AXIS_SCHEMA, "intent": _S, "category": _S}, ["axis"]),
        lambda a: ("POST", "/attachment/suggest",
                   {"axis": a["axis"], "intent": a.get("intent"), "category": a.get("category")}, None),
    ),
    ToolSpec(
        "attach_apply",
        "Apply a catalog part to an axis → CoreOps attach operation (adaptive fit "
        "via adaptation.scale / rotation_deg).",
        _obj({"axis": _AXIS_SCHEMA, "part_id": _S, "session_id": _S,
              "adaptation": {"type": "object"}}, ["axis", "part_id"]),
        lambda a: ("POST", "/attachment/apply",
                   {"session_id": a.get("session_id", "mcp"), "axis": a["axis"],
                    "part_id": a["part_id"], "adaptation": a.get("adaptation", {})}, None),
    ),
    ToolSpec(
        "library_search",
        "Full-text search over Knowledge Library articles.",
        _obj({"query": _S, "k": _I}, ["query"]),
        lambda a: ("GET", "/library/articles/search", None, {"q": a["query"], "k": a.get("k", 10)}),
    ),
    ToolSpec(
        "skill_invoke",
        "Invoke any registered skill by slug with keyword args. "
        "Use the skills_list tool to discover slugs and signatures.",
        _obj({"slug": _S, "args": {"type": "object"}}, ["slug"]),
        lambda a: ("POST", f"/skills/{a['slug']}/invoke", a.get("args", {}), None),
    ),
    ToolSpec(
        "skills_list",
        "List all registered skills with their manifests (inputs/outputs).",
        _obj({}),
        lambda a: ("GET", "/skills", None, None),
    ),
    ToolSpec(
        "standards_check",
        "Check CoreOps operation dimensions against standard sizes "
        "(ISO/DIN/GOST metric fasteners, drills, bearing bores).",
        _obj({"operations": _OPS, "system": _S}, ["operations"]),
        lambda a: ("POST", "/standards/check",
                   {"operations": a["operations"], "system": a.get("system", "ISO")}, None),
    ),
    ToolSpec(
        "learning_suggest",
        "Theory-vs-practice comparison for an action; returns match score, gaps "
        "and rate-limited improvement hints.",
        _obj({"action_type": _S, "action_params": {"type": "object"}, "text": _S}, ["action_type"]),
        lambda a: ("POST", "/learning/suggest",
                   {"action_type": a["action_type"], "action_params": a.get("action_params", {}),
                    "text": a.get("text", "")}, None),
    ),
    ToolSpec(
        "workspace_chat",
        "Send a message to the WorkspaceAgent (the app's main brain) and get the "
        "aggregated event stream back (text, agent calls, candidates).",
        _obj({"message": _S, "session_id": _S, "workspace_id": _S}, ["message"]),
        lambda a: ("POST", "/chat", {
            "message": a["message"],
            "session_id": a.get("session_id", "mcp"),
            "context": {"workspace_id": a.get("workspace_id", a.get("session_id", "mcp"))},
        }, None),
    ),
    ToolSpec(
        "ui_chat_send",
        "Send a message to a named ORYND desktop chat surface. "
        "surface=bottom targets the CAD workspace composer, side targets the left "
        "agent panel, and agent targets the full-screen agent chat.",
        _obj({
            "surface": {"type": "string", "enum": ["bottom", "side", "agent"]},
            "message": _S,
            "workspace_id": _S,
            "session_id": _S,
            "mode": _S,
            "selection": {"type": "object"},
        }, ["message"]),
        lambda a: ("POST", "/chat", _ui_chat_body(a), None),
    ),
    ToolSpec(
        "harness_plan",
        "Plan a multi-skill composition for a task (returns an executable plan).",
        _obj({"task": _S, "max_steps": _I}, ["task"]),
        lambda a: ("POST", "/harness/plan", {"task": a["task"], "max_steps": a.get("max_steps", 3)}, None),
    ),
    ToolSpec(
        "harness_execute",
        "Execute a composition plan produced by harness_plan.",
        _obj({"plan": {"type": "object"}}, ["plan"]),
        lambda a: ("POST", "/harness/execute", {"plan": a["plan"]}, None),
    ),
]

TOOL_INDEX: dict[str, ToolSpec] = {t.name: t for t in TOOLS}


async def execute_tool(name: str, args: dict) -> tuple[bool, Any]:
    """Run a tool in-process against the FastAPI app. Returns (is_error, data)."""
    spec = TOOL_INDEX.get(name)
    if spec is None:
        return True, {"error": f"unknown tool: {name}", "available": sorted(TOOL_INDEX)}
    try:
        method, path, body, params = spec.build(args or {})
    except KeyError as exc:
        return True, {"error": f"missing required argument: {exc}"}
    except ValueError as exc:
        return True, {"error": str(exc)}

    from orynd_core.api.main import app  # late import — avoids circular import

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://mcp.internal", timeout=600.0
    ) as client:
        response = await client.request(method, path, json=body, params=params)

    content_type = response.headers.get("content-type", "")
    if "ndjson" in content_type:
        events = []
        for line in response.text.splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return response.status_code >= 400, {"events": events}
    try:
        data = response.json()
    except json.JSONDecodeError:
        data = {"raw": response.text[:2000]}
    return response.status_code >= 400, data
