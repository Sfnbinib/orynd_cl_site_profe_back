"""
WorkspaceAgent — central brain of ORYND Desktop.

Uses Claude's native tool use (function calling) to orchestrate agents.
Claude decides which agents to call, in what order, how many times.
Not a router — a real agentic loop.

Algorithm fallback when no API key: simple keyword routing.

Input:  ctx.raw_text, ctx.image_b64, ctx.extra["history"], ctx.extra["user_profile"]
Output: ctx.extra["workspace_response"]  — final text to user
        ctx.extra["tool_calls"]          — list of agents that were activated
        ctx.candidates                   — if search was run
        ctx.extra["fabrication"]         — if fabrication was run
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import re
from typing import AsyncIterator

from orynd_core.agents.base import AgentContext, AgentResult, BaseAgent
from orynd_core.services.llm.base import LLMProvider

log = logging.getLogger(__name__)

# ── Tool definitions (sent to Claude) ────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "search_models",
        "description": (
            "Search for 3D models across Printables, Thingiverse, and other sources. "
            "Use this when the user wants to find, print, or download a 3D model."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Clean English search terms, 2-5 words",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "analyze_image",
        "description": (
            "Analyze an image to identify the object, infer what 3D model is needed, "
            "or assess a part for fabrication. Use when the user uploads a photo."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "What to focus on: 'identify' | 'broken_part' | 'fabrication' | 'fit_analysis'",
                    "default": "identify",
                },
            },
        },
    },
    {
        "name": "get_fabrication",
        "description": (
            "Get fabrication recommendations for a selected or described 3D model: "
            "material, infill %, orientation, supports, print time estimate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name or description of the model",
                },
                "use_case": {
                    "type": "string",
                    "description": "What it will be used for (mechanical, decorative, outdoor, etc.)",
                },
            },
            "required": ["model_name"],
        },
    },
    {
        "name": "select_model",
        "description": "Select a specific model from search results by index (0-based).",
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Index of the model in current results (0 = best match)",
                    "default": 0,
                },
            },
        },
    },
    {
        "name": "deep_research",
        "description": (
            "Run deep multi-source research on a topic: searches 3D model databases, "
            "GitHub repositories, web articles, and open-source projects in parallel. "
            "Use for complex questions like 'how to build X', 'what exists for Y', "
            "'find open-source solutions for Z'. Returns knowledge map + recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Research topic — describe what the user wants to understand or build",
                },
                "depth": {
                    "type": "integer",
                    "description": "Research depth: 1=quick, 2=standard, 3=deep (default 2)",
                    "default": 2,
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "analyze_mesh",
        "description": (
            "Analyze a 3D mesh file (STL/OBJ/PLY) — decompose into surface regions, "
            "extract manufacturing features (holes, pockets, bosses, fillets), and generate "
            "CoreOps JSON. Use when user uploads or provides a mesh file for reverse engineering, "
            "analysis, or fabrication planning. Pipeline B of AI Model 4."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mesh_path": {
                    "type": "string",
                    "description": "Local file path to STL/OBJ/PLY mesh",
                },
                "scale": {
                    "type": "number",
                    "description": "Coordinate scale: 1.0 = mm, 25.4 = inch→mm",
                    "default": 1.0,
                },
                "angle_threshold": {
                    "type": "number",
                    "description": "Region decomposition angle threshold in degrees (default 15)",
                    "default": 15.0,
                },
            },
            "required": ["mesh_path"],
        },
    },
    {
        "name": "build_3d_model",
        "description": (
            "Build a 3D model from scratch using CoreOps operations. "
            "Generate a sequence of CAD operations (CreateSketch, Extrude, CutHole, Fillet, etc.) "
            "that will be executed by the CAD engine. Returns STL/STEP/OBJ files. "
            "Use when the user asks to CREATE, BUILD, or DESIGN a part from dimensions or description."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "description": (
                        "List of CoreOps operations. Each operation is an object with 'op', 'id', and operation-specific params. "
                        "Available ops: CreateSketch (plane, shapes[rect/circle/polygon]), "
                        "Extrude (sketch_ref, height), Cut (sketch_ref, depth), "
                        "CutHole (center {x,y}, radius, through), CutSlot (start, end, width, depth), "
                        "Fillet (radius, edges), Chamfer (distance, edges), "
                        "Revolve (sketch_ref, axis, angle), Loft (sketch_refs), "
                        "Boolean (operation: union/subtract/intersect, body_refs), Mirror (body_ref, plane)"
                    ),
                    "items": {"type": "object"},
                },
                "units": {
                    "type": "string",
                    "description": "Units: mm (default) or inch",
                    "default": "mm",
                },
            },
            "required": ["operations"],
        },
    },
]

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are ORYND — an AI engineering workspace for 3D design, fabrication, and part creation.
You are a senior mechanical engineer colleague. Direct, technical, no fluff.

This is a WORKSPACE with real CAD capabilities — not just a search tool or chatbot.

WHEN **NOT** TO USE TOOLS (important — do not call any tool in these cases):
- Greetings, thanks, small talk ("hi", "привет", "who are you", "what can you do") → reply in 1-2 sentences, NO tool calls.
- Vague or unclear requests with no concrete object/dimensions/topic → ask ONE clarifying question, NO tool calls.
- Only call a tool when the user clearly wants something found, built, analyzed, or researched. When in doubt, ask — don't fabricate a build.

Available tools (when there IS a clear request, act — don't just talk):
1. search_models — find existing 3D models across Printables, Thingiverse, GitHub
2. analyze_image — analyze photos: identify objects, extract geometry hints, detect broken parts
3. analyze_mesh — REVERSE ENGINEER a mesh file (STL/OBJ): decompose → features → CoreOps JSON
4. build_3d_model — BUILD a real 3D model from CoreOps → produces STL/STEP/OBJ files
5. get_fabrication — recommend manufacturing method, material, print settings
6. select_model — pick a specific model from search results
7. deep_research — multi-source parallel research on engineering topics

ROUTING RULES (follow strictly):
- "find/search/download/I need a [object]" → search_models
- STL/OBJ file provided or "analyze this mesh/model" → analyze_mesh (Pipeline B — reverse engineering)
- DIMENSIONS given (50x30x10, 2 inch) or "create/build/design/make me" → build_3d_model
  ALWAYS: CreateSketch (rect/circle) → Extrude → add features (CutHole, Fillet, etc.)
  Example for "box 50x30 height 10 with 4 corner holes r=2.5":
    CreateSketch(rect 50x30) → Extrude(10) → CutHole(-20,-10,r=2.5) → CutHole(20,-10) → CutHole(-20,10) → CutHole(20,10) → Fillet(r=1)
- PHOTO uploaded → analyze_image FIRST, then:
  - Broken part → analyze → build_3d_model (reconstruct)
  - "Find similar" → analyze → search_models
  - "What is this" → analyze only
- Printing/material/settings questions → get_fabrication
- "select #2" / "take this one" → select_model
- "how to build X" / "compare options" → deep_research
- CHAIN tools: analyze_image → build_3d_model → get_fabrication (all in one turn)

CoreOps for build_3d_model:
- CreateSketch: plane (XY/XZ/YZ), shapes [{type:"rect", width, height} | {type:"circle", radius}]
- Extrude: sketch_ref, height
- CutHole: center {x,y}, radius, through:true
- CutSlot: start, end, width, depth
- Fillet: radius, edges:["all"]
- Chamfer: distance
- Revolve: sketch_ref, axis, angle
- Boolean: union/subtract/intersect, body_refs
- Mirror: body_ref, plane

Context:
- Reference previous results by name
- After build → offer fabrication recommendations
- After search → suggest selecting best match

Language: MATCH user's language (Russian → Russian).
Tone: senior engineer. 2-4 sentences when no tools needed.
"""

# Answer-FORMATION criteria — the orchestration runs a fast model (3b) to route &
# execute tools, then a STRONGER local model rewrites the result into the final
# user-facing answer using these rules. This is what makes the reply read like an
# engineer wrote it instead of a weak model dumping raw text/JSON.
_FORMATION_CRITERIA = """\
You are ORYND, a senior mechanical-engineering CAD copilot. Turn the draft + actions
below into the FINAL answer for the user.

Rules:
- Match the user's language (Russian request → Russian answer).
- Senior engineer talking to a peer: direct, concrete, zero filler. No "as an AI",
  no apologies, no restating the question.
- NEVER print raw JSON, tool names, or internal field names — describe results in words.
- Built a model → one line: what it is + key dims + volume, then ONE concrete next step
  (fabrication / fillet / attach).
- Found candidates/sources → 1-line summary + top 2-3 only, never a wall of text.
- Use short markdown bullets only when listing 3+ items; otherwise plain sentences.
- ≤4 sentences unless listing results.
- If the draft is empty or garbled, answer the request directly from the actions taken."""

# ── Tool executors ────────────────────────────────────────────────────────────

async def _run_search(query: str, limit: int, ctx: AgentContext) -> dict:
    from orynd_core.agents.retrieval import RetrievalAgent
    from orynd_core.agents.orchestrator import Pipeline

    ctx.raw_text = query
    ctx.intent = {
        "keywords": query.split(),
        "object_name": query,
        "_source": "workspace",
    }
    ctx.extra["search_query"] = query

    await Pipeline([RetrievalAgent(limit=limit)]).run(ctx)

    results = ctx.candidates[:limit]
    if not results:
        return {"found": 0, "message": "No models found"}

    return {
        "found": len(results),
        "models": [
            {
                "index": i,
                "name": c.get("name", "") if isinstance(c, dict) else c.name,
                "description": (c.get("description", "") if isinstance(c, dict) else c.description)[:100],
                "source": c.get("source", "") if isinstance(c, dict) else c.source,
                "printability": c.get("printability", 5) if isinstance(c, dict) else c.printability,
            }
            for i, c in enumerate(results)
        ],
    }


async def _run_analyze_image(focus: str, ctx: AgentContext) -> dict:
    from orynd_core.agents.vision import VisionAgent
    from orynd_core.agents.orchestrator import Pipeline

    provider = ctx.extra.get("_provider")
    await Pipeline([VisionAgent(provider=provider)]).run(ctx)

    vision = ctx.extra.get("vision", {})
    return {
        "object_name": vision.get("object_name", "unknown"),
        "description": vision.get("description", ""),
        "search_query": vision.get("search_query", ""),
        "is_broken": vision.get("is_broken", False),
        "tags": vision.get("tags", []),
        "confidence": vision.get("confidence", 0),
    }


async def _run_fabrication(model_name: str, use_case: str, ctx: AgentContext) -> dict:
    from orynd_core.agents.fabrication import FabricationAgent
    from orynd_core.agents.orchestrator import Pipeline

    provider = ctx.extra.get("_provider")

    # inject model info into context
    if not ctx.selected:
        ctx.selected = {"name": model_name, "description": use_case}

    await Pipeline([FabricationAgent(provider=provider)]).run(ctx)
    fab = ctx.extra.get("fabrication", {})

    return {
        "method": fab.get("recommended_method", "fdm"),
        "material": fab.get("material", "PLA"),
        "infill": fab.get("infill_pct", 20),
        "orientation": fab.get("orientation_hint", ""),
        "supports": fab.get("support_needed", False),
        "notes": fab.get("notes", ""),
    }


async def _run_select(index: int, ctx: AgentContext) -> dict:
    from orynd_core.agents.selector import SelectorAgent
    from orynd_core.agents.orchestrator import Pipeline

    ctx.extra["select_index"] = index
    await Pipeline([SelectorAgent()]).run(ctx)

    if ctx.selected:
        return {
            "selected": ctx.selected.get("name", ""),
            "stl_url": ctx.stl_url or ctx.selected.get("source_url", ""),
            "verified": bool(ctx.stl_url),
        }
    return {"error": "No model at that index"}


async def _run_deep_research(topic: str, depth: int, ctx: AgentContext) -> dict:
    from orynd_core.agents.research import DeepResearchAgent
    from orynd_core.agents.orchestrator import Pipeline

    research_ctx = AgentContext(
        session_id=ctx.session_id,
        user_id=ctx.user_id,
        raw_text=topic,
        extra={"research_topic": topic},
    )
    # SAFETY: deep research fans out parallel HTTP + multiple local-LLM calls. On a
    # constrained machine an unbounded run can exhaust RAM and freeze the OS. Cap
    # depth and enforce a hard wall-clock budget — on timeout we return whatever
    # partial result exists instead of blocking the event loop indefinitely.
    # The agent self-bounds at 110s (covers every caller); this outer net (130s)
    # is just a backstop in case the agent-level guard is ever bypassed.
    agent = DeepResearchAgent(depth=max(1, min(int(depth or 1), 2)))
    try:
        await asyncio.wait_for(Pipeline([agent]).run(research_ctx), timeout=130)
    except asyncio.TimeoutError:
        log.warning("[research] outer backstop timeout (130s)")
    except Exception as e:
        log.warning("[research] failed: %s", e)

    research = research_ctx.extra.get("research", {})
    # Copy 3D candidates back to main ctx if found
    if research_ctx.candidates:
        ctx.candidates = research_ctx.candidates

    # Wire #6: persist research → Library (topic + article + hypotheses + session)
    try:
        from orynd_core.services.library.research_writer import save_research
        await save_research(topic, research, ctx.session_id)
    except Exception:
        pass  # never block the agent loop

    return {
        "topic": topic,
        "sources_found": research.get("sources_total", 0),
        "open_source": research.get("open_source", [])[:3],
        "gaps": research.get("gaps", []),
        "recommendations": research.get("recommendations", ""),
        "confidence": research.get("confidence", 0),
        "iterations": research.get("iterations", 1),
    }


_COREOPS_REF_KEYS = ("id", "sketch_ref", "body_ref")
_COREOPS_REF_LIST_KEYS = ("sketch_refs", "body_refs")


def _normalize_coreops_ids(operations: list) -> list:
    """LLMs emit numeric ids (`"id": 1`); the CoreOps schema requires string
    ids/refs. Stringify those fields so validation passes — purely a type
    adaptation, no change to operation semantics or the schema itself.
    """
    if not isinstance(operations, list):
        return operations
    out = []
    for op in operations:
        if not isinstance(op, dict):
            out.append(op)
            continue
        o = dict(op)
        for k in _COREOPS_REF_KEYS:
            if isinstance(o.get(k), (int, float)) and not isinstance(o.get(k), bool):
                o[k] = str(o[k])
        for k in _COREOPS_REF_LIST_KEYS:
            if isinstance(o.get(k), list):
                o[k] = [str(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x for x in o[k]]
            elif isinstance(o.get(k), (str, int, float)):  # single ref → list
                o[k] = [str(o[k])]
        # `edges` is list[str] in the schema; LLMs often emit the bare string "all".
        if "edges" in o and isinstance(o["edges"], str):
            o["edges"] = [o["edges"]]
        out.append(o)
    return out


def _ensure_extrude_after_sketch(operations: list[dict]) -> list[dict]:
    """llama3.2:3b often emits CreateSketch without a following Extrude, producing
    flat (z=0) geometry.  Walk the op list and inject a default Extrude immediately
    after any CreateSketch that isn't already followed by one."""
    result: list[dict] = []
    for i, op in enumerate(operations):
        result.append(op)
        if op.get("op") != "CreateSketch":
            continue
        next_op = operations[i + 1] if i + 1 < len(operations) else None
        if next_op and next_op.get("op") == "Extrude":
            continue
        # Infer a sensible extrusion height from the sketch shapes.
        h = 20.0  # default mm
        for s in (op.get("shapes") or []):
            if isinstance(s, dict):
                if s.get("type") == "rect":
                    h = max(5.0, min(float(s.get("height", s.get("width", 40))), 60.0))
                elif s.get("type") == "circle":
                    h = max(5.0, min(float(s.get("radius", 10)) * 2, 60.0))
                break
        sketch_id = str(op.get("id", "s0"))
        result.append({
            "op": "Extrude",
            "id": sketch_id + "_ext",
            "sketch_ref": sketch_id,
            "height": h,
        })
    return result


async def _run_build_3d_model(operations: list[dict], units: str, ctx: AgentContext) -> dict:
    from orynd_core.agents.cad import CADAgent
    from orynd_core.agents.orchestrator import Pipeline

    operations = _normalize_coreops_ids(operations)
    operations = _ensure_extrude_after_sketch(operations)
    ctx.extra["coreops"] = {"operations": operations, "units": units}
    await Pipeline([CADAgent()]).run(ctx)

    cad = ctx.extra.get("cad", {})
    if not cad:
        return {"error": "CAD execution returned no result"}

    return {
        "built": True,
        "dry_run": cad.get("dry_run", False),
        "operations_executed": cad.get("operations_executed", 0),
        "properties": cad.get("properties", {}),
        "stl_url": f"/cad/model/{ctx.session_id}/part.stl" if cad.get("stl_path") else None,
        "step_url": f"/cad/model/{ctx.session_id}/part.step" if cad.get("step_path") else None,
        "obj_url": f"/cad/model/{ctx.session_id}/part.obj" if cad.get("obj_path") else None,
    }


async def _run_analyze_mesh(mesh_path: str, scale: float, angle: float, ctx: AgentContext) -> dict:
    from orynd_core.agents.mesh_analysis import MeshAnalysisAgent

    mesh_ctx = AgentContext(
        session_id=ctx.session_id,
        user_id=ctx.user_id,
    )
    mesh_ctx.extra = {
        "mesh_path": mesh_path,
        "mesh_scale": scale,
        "decompose_angle": angle,
    }

    agent = MeshAnalysisAgent()
    result = await agent.run(mesh_ctx)

    if not result.ok:
        return {"error": result.error}

    # Copy results back to main ctx
    ctx.extra["mesh_features"] = mesh_ctx.extra.get("mesh_features")
    ctx.extra["coreops_json"] = mesh_ctx.extra.get("coreops_json")

    data = result.data
    return {
        "mesh_info": data.get("mesh_info", {}),
        "regions_count": data.get("regions_count", 0),
        "features_count": data.get("features_count", 0),
        "feature_summary": data.get("feature_summary", {}),
        "coreops_json": data.get("coreops_json", {}),
    }


async def _execute_tool(name: str, tool_input: dict, ctx: AgentContext) -> str:
    """Run a tool and return result as JSON string."""
    try:
        if name == "search_models":
            result = await _run_search(
                tool_input.get("query", ""),
                tool_input.get("limit", 5),
                ctx,
            )
        elif name == "analyze_image":
            result = await _run_analyze_image(
                tool_input.get("focus", "identify"),
                ctx,
            )
        elif name == "get_fabrication":
            result = await _run_fabrication(
                tool_input.get("model_name", ""),
                tool_input.get("use_case", ""),
                ctx,
            )
        elif name == "select_model":
            result = await _run_select(tool_input.get("index", 0), ctx)
        elif name == "deep_research":
            result = await _run_deep_research(
                tool_input.get("topic", ctx.raw_text or ""),
                tool_input.get("depth", 1),
                ctx,
            )
        elif name == "analyze_mesh":
            result = await _run_analyze_mesh(
                tool_input.get("mesh_path", ""),
                tool_input.get("scale", 1.0),
                tool_input.get("angle_threshold", 15.0),
                ctx,
            )
        elif name == "build_3d_model":
            result = await _run_build_3d_model(
                tool_input.get("operations", []),
                tool_input.get("units", "mm"),
                ctx,
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        log.exception("[workspace] tool %s failed", name)
        result = {"error": str(e)}

    # Feed action to Learning Engine (#26 → #42 wire)
    try:
        from orynd_core.services import action_log
        await action_log.write(name, tool_input, result, ctx.session_id)
    except Exception:
        pass  # never break the main loop

    # Track credits consumed per session (#credits wire)
    try:
        from orynd_core.services.credits import session_tracker
        await session_tracker.record(name, tool_input, ctx.session_id)
    except Exception:
        pass

    return json.dumps(result, ensure_ascii=False)


# ── Algorithm fallback ────────────────────────────────────────────────────────

async def _algorithm_fallback(ctx: AgentContext) -> AsyncIterator[dict]:
    """When no LLM — simple keyword routing, yield events."""
    from orynd_core.agents.chat import _algorithm_route
    text = ctx.raw_text or ""
    routing = _algorithm_route(text)
    action = routing.get("action", "search")

    if action == "search":
        query = routing.get("search_query") or text
        yield {"type": "agent_call", "agent": "search_models", "input": {"query": query}}
        result_json = await _execute_tool("search_models", {"query": query, "limit": 5}, ctx)
        result = json.loads(result_json)
        yield {"type": "agent_result", "agent": "search_models", "found": result.get("found", 0)}
        if result.get("found", 0) > 0:
            yield {"type": "text", "content": f'Searching for "{query}"…\n'}
        else:
            yield {"type": "text", "content": f'Nothing found for "{query}". Try different terms.'}
    else:
        yield {"type": "text", "content": routing.get("response") or "How can I help?"}


# ── WorkspaceAgent ────────────────────────────────────────────────────────────

class WorkspaceAgent(BaseAgent):
    """
    Central workspace brain.
    Runs an agentic loop: Claude calls tools → execute → feed back → continue.
    Yields events for streaming to UI.
    """

    name = "workspace_agent"

    def __init__(self, provider: LLMProvider | None = None) -> None:
        super().__init__(provider=provider)

    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        """Non-streaming run — collects all events, returns result."""
        events = []
        async for event in self.stream(ctx):
            events.append(event)
        return AgentResult.success(self.name, {"events": events})

    async def stream(self, ctx: AgentContext) -> AsyncIterator[dict]:
        """Yield events as they happen. Call from router for streaming."""
        ctx.extra["_provider"] = self.provider
        ctx.extra["tool_calls"] = []

        if not self.provider:
            async for event in _algorithm_fallback(ctx):
                yield event
            return

        # Local Ollama provider → run the free local tool-use loop.
        from orynd_core.services.llm.local import LocalProvider
        if isinstance(self.provider, LocalProvider):
            async for event in self._ollama_stream(ctx):
                yield event
            return

        # Wire #8: Library RAG — inject known articles into context
        try:
            from orynd_core.services.library.storage_factory import get_storage_backend
            _lib_hits = await get_storage_backend().search_articles_fts(
                ctx.raw_text or "", k=3
            )
            if _lib_hits:
                _snippets = [
                    f"- {h.article.title}: {(h.article.body_md or '')[:300]}"
                    for h in _lib_hits[:3]
                ]
                ctx.extra["_library_context"] = "Known from library:\n" + "\n".join(_snippets)
        except Exception:
            pass

        # Build initial messages
        history = ctx.extra.get("history", [])
        messages = _build_messages(ctx, history)

        max_rounds = 6  # prevent infinite loops
        round_num = 0

        import anthropic as _anthropic
        client = _anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", "")
        )
        model = getattr(self.provider, "model", "claude-haiku-4-5-20251001")

        while round_num < max_rounds:
            round_num += 1

            # Call Claude with tools
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=_SYSTEM + _context_note(ctx),
                    tools=TOOLS,
                    messages=messages,
                )
            except Exception as e:
                log.error("[workspace] Claude API error: %s", e)
                yield {"type": "error", "message": str(e)}
                return

            # Process response blocks
            tool_uses = []
            text_chunks = []

            for block in response.content:
                if block.type == "text" and block.text:
                    text_chunks.append(block.text)
                    yield {"type": "text", "content": block.text}
                elif block.type == "tool_use":
                    tool_uses.append(block)

            # If no tool calls → Claude is done
            if response.stop_reason == "end_turn" or not tool_uses:
                final_text = "".join(text_chunks)
                ctx.extra["workspace_response"] = final_text
                return

            # Execute all tool calls (parallel where possible)
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_input = tool_use.input

                # Notify UI: agent is being called
                ctx.extra["tool_calls"].append(tool_name)
                yield {
                    "type": "agent_call",
                    "agent": tool_name,
                    "input": tool_input,
                }

                # Execute
                result_str = await _execute_tool(tool_name, tool_input, ctx)
                result_data = json.loads(result_str)

                yield {
                    "type": "agent_result",
                    "agent": tool_name,
                    "result": result_data,
                }

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result_str,
                })

            # If search ran and found candidates → emit them
            if ctx.candidates:
                yield {"type": "candidates", "candidates": [
                    c if isinstance(c, dict) else c.model_dump()
                    for c in ctx.candidates
                ]}

            # If CAD built a model → emit model_ready for frontend viewer
            cad_result = ctx.extra.get("cad")
            if cad_result and cad_result.get("stl_path"):
                yield {
                    "type": "model_ready",
                    "stl_url": f"/cad/model/{ctx.session_id}/part.stl",
                    "step_url": f"/cad/model/{ctx.session_id}/part.step",
                    "obj_url": f"/cad/model/{ctx.session_id}/part.obj",
                    "properties": cad_result.get("properties", {}),
                    "dry_run": cad_result.get("dry_run", False),
                }

            # Feed tool results back to Claude
            messages.append({"role": "user", "content": tool_results})

        # Max rounds reached
        ctx.extra["workspace_response"] = ctx.extra.get("workspace_response", "")

    async def _ollama_stream(self, ctx: AgentContext) -> AsyncIterator[dict]:
        """Agentic loop over a local Ollama model (free, no API key).

        Ollama exposes OpenAI-style tool calling for tool-capable models
        (e.g. llama3.2). We convert our Anthropic-style TOOLS to that schema and
        run the same execute → feed-result-back loop, emitting identical events.
        Small local models are weaker at multi-step tool use than Claude — this
        is the free path, not the strongest one.
        """
        import httpx

        base_url = getattr(self.provider, "base_url", "http://localhost:11434").rstrip("/")
        model = getattr(self.provider, "model", "llama3.2:3b")
        tools = _ollama_tools()

        # Wire #8: Library RAG (Ollama path)
        try:
            from orynd_core.services.library.storage_factory import get_storage_backend
            _lib_hits = await get_storage_backend().search_articles_fts(
                ctx.raw_text or "", k=3
            )
            if _lib_hits:
                _snippets = [
                    f"- {h.article.title}: {(h.article.body_md or '')[:300]}"
                    for h in _lib_hits[:3]
                ]
                ctx.extra["_library_context"] = "Known from library:\n" + "\n".join(_snippets)
        except Exception:
            pass

        msgs: list[dict] = [{"role": "system", "content": _SYSTEM + _context_note(ctx)}]
        for h in ctx.extra.get("history", []) or []:
            if isinstance(h, dict) and isinstance(h.get("content"), str) and h.get("role") in ("user", "assistant"):
                msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": ctx.raw_text})

        max_rounds = 6
        text_chunks: list[str] = []
        tool_summary: list[str] = []
        form = _formation_enabled()

        async with httpx.AsyncClient(timeout=180) as client:
            for _ in range(max_rounds):
                try:
                    resp = await client.post(
                        f"{base_url}/api/chat",
                        json={
                            "model": model,
                            "messages": msgs,
                            "tools": tools,
                            "stream": False,
                            "options": {"temperature": 0.2},
                        },
                    )
                    data = resp.json()
                except Exception as e:
                    log.error("[workspace/ollama] request error: %s", e)
                    yield {"type": "error", "message": f"Ollama error: {e}"}
                    return

                if isinstance(data, dict) and data.get("error"):
                    yield {"type": "error", "message": f"Ollama: {data['error']}"}
                    return

                message = (data or {}).get("message", {}) or {}
                content = message.get("content") or ""
                tool_calls = message.get("tool_calls") or []

                if content:
                    content = _strip_leaked_tool_json(content)
                if content:
                    text_chunks.append(content)
                    # When formation is on, the strong model writes the final answer
                    # after the loop — don't stream the weak 3b draft to the user.
                    if not form:
                        yield {"type": "text", "content": content}

                if not tool_calls:
                    break

                # Echo the assistant turn (with its tool calls) back into context.
                msgs.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                for tc in tool_calls:
                    fn = (tc.get("function") or {})
                    tool_name = fn.get("name", "")
                    tool_input = fn.get("arguments", {})
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except Exception:
                            tool_input = {}
                    tool_input = _coerce_args(tool_name, tool_input)

                    ctx.extra["tool_calls"].append(tool_name)
                    yield {"type": "agent_call", "agent": tool_name, "input": tool_input}

                    result_str = await _execute_tool(tool_name, tool_input, ctx)
                    tool_summary.append(f"{tool_name}: {result_str[:300]}")
                    try:
                        result_data = json.loads(result_str)
                    except Exception:
                        result_data = {"raw": result_str}
                    yield {"type": "agent_result", "agent": tool_name, "result": result_data}

                    msgs.append({"role": "tool", "content": result_str})

                if ctx.candidates:
                    yield {"type": "candidates", "candidates": [
                        c if isinstance(c, dict) else c.model_dump() for c in ctx.candidates
                    ]}

                cad_result = ctx.extra.get("cad")
                if cad_result and cad_result.get("stl_path"):
                    yield {
                        "type": "model_ready",
                        "stl_url": f"/cad/model/{ctx.session_id}/part.stl",
                        "step_url": f"/cad/model/{ctx.session_id}/part.step",
                        "obj_url": f"/cad/model/{ctx.session_id}/part.obj",
                        "properties": cad_result.get("properties", {}),
                        "dry_run": cad_result.get("dry_run", False),
                    }

        # ── FORMATION: strong local model writes the final answer (orchestrator
        # pattern — 3b routed & executed above, this model forms the reply). ──
        draft = "".join(text_chunks).strip()
        if form:
            formed_any = False
            async for chunk in _form_answer(ctx, draft, tool_summary):
                formed_any = True
                yield {"type": "text", "content": chunk}
            if not formed_any and draft:
                yield {"type": "text", "content": draft}  # fallback: model unavailable
        ctx.extra["workspace_response"] = draft


# ── Helpers ───────────────────────────────────────────────────────────────────


def _context_note(ctx: AgentContext) -> str:
    """Сквозной контекст: turn workspace state (ctx.extra['workspace_context'])
    into a note the orchestrator sees automatically — so the agent is aware of the
    current model / selection / running tasks without being told each time."""
    wc = ctx.extra.get("workspace_context") or {}
    if not isinstance(wc, dict) or not wc:
        return ""
    label = {"surface": "active surface", "model": "current model", "dims": "dimensions",
             "selected": "selected", "selection": "user-selected geometry (the part the user means by 'this'/'эта грань')",
             "active_tasks": "running tasks", "model_url": "loaded model"}
    parts = []
    for k in ("surface", "model", "dims", "selected", "selection", "active_tasks", "model_url"):
        if wc.get(k):
            parts.append(f"{label[k]}: {wc[k]}")
    extra = {k: v for k, v in wc.items() if k not in label}
    if extra:
        parts.append("other: " + json.dumps(extra, ensure_ascii=False)[:300])
    if not parts:
        return ""
    return ("\n\nWORKSPACE CONTEXT (you see this automatically — use it; the user may "
            "refer to it as 'this', 'the model', 'selected'):\n- " + "\n- ".join(parts))


def _ollama_tools() -> list[dict]:
    """Convert Anthropic-style TOOLS → Ollama/OpenAI function-calling schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in TOOLS
    ]


def _formation_enabled() -> bool:
    return os.getenv("ORYND_FORMATION", "on").strip().lower() not in ("0", "off", "false", "no")


def _formation_provider():
    """The ANSWER model — writes the final reply (founder principle: this is where a
    strong model pays off, NOT orchestration).

    Priority: Claude (if key) → local Ollama formation model. Claude is the right
    home for a paid key — it answers messy/long requests well with full context.
    Local default is llama3.2:3b (~17s, good format via criteria); qwen2.5-coder:7b
    is cleaner but ~60-75s + memory thrash on this hardware, so it's opt-in via
    ORYND_FORMATION_MODEL. None if nothing reachable → caller falls back to draft."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if key:
        from orynd_core.services.llm.claude import ClaudeProvider
        return ClaudeProvider(api_key=key, model=os.getenv("ORYND_FORMATION_CLAUDE_MODEL", "claude-sonnet-4-5"))
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("ORYND_FORMATION_MODEL", "llama3.2:3b")
    try:
        import httpx
        if httpx.get(f"{base_url}/api/tags", timeout=1.0).status_code != 200:
            return None
    except Exception:
        return None
    from orynd_core.services.llm.local import LocalProvider
    return LocalProvider(base_url=base_url, model=model)


async def _form_answer(ctx: AgentContext, draft: str, tool_summary: list[str]) -> AsyncIterator[str]:
    """Stream the final answer from the formation model. Yields nothing on failure
    (caller then falls back to the draft)."""
    provider = _formation_provider()
    if provider is None:
        return
    from orynd_core.services.llm.base import LLMMessage
    # FULL CONTEXT (founder principle): a swapped-in external model must SEE the
    # whole situation — request + workspace state + recent history + tool results —
    # not an isolated task, or it answers blind.
    parts = [f"User request (verbatim): {ctx.raw_text or ''}"]
    note = _context_note(ctx)
    if note:
        parts.append("Current workspace:" + note)
    history = ctx.extra.get("history", []) or []
    if history:
        recent = []
        for h in history[-3:]:
            if not isinstance(h, dict):
                continue
            q = h.get("query")
            r = h.get("workspace_response") or h.get("chat_response")
            if q:
                recent.append(f"User: {q}")
            if r:
                recent.append(f"You: {str(r)[:200]}")
        if recent:
            parts.append("Recent conversation:\n" + "\n".join(recent))
    if tool_summary:
        parts.append("Actions taken this turn (with results):\n" + "\n".join(tool_summary))
    if draft:
        parts.append("Rough draft (may be messy/incomplete):\n" + draft)
    parts.append("Write the final answer now.")
    prompt = "\n\n".join(parts)
    try:
        async for chunk in provider.stream(
            [LLMMessage(role="user", content=prompt)],
            system=_FORMATION_CRITERIA,
            max_tokens=600,
            temperature=0.3,
        ):
            if chunk:
                yield chunk
    except Exception as e:
        log.warning("[formation] stream failed: %s", e)
        return


def _strip_leaked_tool_json(text: str) -> str:
    """Small local models sometimes emit a tool call as plain text instead of a
    structured tool_call (e.g. trailing `{"name": "get_fabrication", ...}`). That
    raw JSON leaks into the chat bubble. Strip any tool-call-shaped JSON object so
    the user only sees prose."""
    if not text or "{" not in text:
        return text
    # Drop from the first tool-call-looking JSON object to the end of the string.
    cleaned = re.sub(
        r'\{\s*"(?:name|type|function|tool|parameters|arguments)"\s*:.*$',
        "",
        text,
        flags=re.DOTALL,
    )
    return cleaned.strip()


def _coerce_args(tool_name: str, args: dict) -> dict:
    """Ollama returns every tool argument as a string ("5", "3", "true").

    Cast each value to the type declared in the tool's input_schema so
    downstream code (list slicing `[:limit]`, `depth < x`, etc.) doesn't blow
    up. Without this, search_models and deep_research crash on string args.
    """
    if not isinstance(args, dict):
        return args
    schema = next((t["input_schema"] for t in TOOLS if t["name"] == tool_name), None)
    if not schema:
        return args
    props = schema.get("properties", {})
    out = dict(args)
    for k, v in list(out.items()):
        if not isinstance(v, str):
            continue
        typ = (props.get(k) or {}).get("type")
        try:
            if typ == "integer":
                out[k] = int(float(v))
            elif typ == "number":
                out[k] = float(v)
            elif typ == "boolean":
                out[k] = v.strip().lower() in ("true", "1", "yes", "y")
            elif typ in ("array", "object"):
                # Ollama often serialises complex args as a JSON string.
                out[k] = json.loads(v)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    return out

def _build_messages(ctx: AgentContext, history: list[dict]) -> list[dict]:
    """Build Anthropic messages list from history + current input."""
    messages: list[dict] = []

    # Last 6 turns of history
    for turn in history[-6:]:
        if turn.get("query"):
            content: list | str = turn["query"]
            messages.append({"role": "user", "content": content})
        if turn.get("workspace_response") or turn.get("chat_response"):
            resp = turn.get("workspace_response") or turn.get("chat_response")
            messages.append({"role": "assistant", "content": resp})

    # Current user message
    if ctx.image_b64:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": ctx.image_b64,
                },
            },
            {"type": "text", "text": ctx.raw_text or "What is this?"},
        ]
        messages.append({"role": "user", "content": content})
    else:
        # Inject workspace context into first message
        workspace_ctx = _workspace_context(ctx)
        user_text = ctx.raw_text or ""
        if workspace_ctx:
            user_text = f"{workspace_ctx}\n\n{user_text}"
        messages.append({"role": "user", "content": user_text})

    return messages


def _workspace_context(ctx: AgentContext) -> str:
    """Build a context string about current workspace state."""
    parts = []

    profile = ctx.extra.get("user_profile", {})
    if profile.get("printer"):
        parts.append(f"User's printer: {profile['printer']}")

    if ctx.candidates:
        names = [
            (c.get("name") if isinstance(c, dict) else c.name)
            for c in ctx.candidates[:3]
        ]
        parts.append(f"Last search results: {', '.join(names)}")

    if ctx.selected:
        name = ctx.selected.get("name", "unknown")
        parts.append(f"Currently selected model: {name}")

    lib_ctx = ctx.extra.get("_library_context", "")
    if lib_ctx:
        parts.append(lib_ctx)

    if not parts:
        return ""

    return "[Workspace state: " + " | ".join(parts) + "]"
