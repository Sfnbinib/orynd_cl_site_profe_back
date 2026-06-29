"""
POST /chat — main workspace endpoint with NDJSON streaming.

Pipeline:
  MemoryAgent(load) → WorkspaceAgent (agentic loop) → MemoryAgent(save)

WorkspaceAgent uses Claude tool use to call any agent on demand.
Falls back to algorithm routing if no ANTHROPIC_API_KEY.

Stream events (one JSON per line):
  {"type": "text",         "content": "..."}
  {"type": "agent_call",   "agent": "search_models", "input": {...}}
  {"type": "agent_result", "agent": "search_models", "result": {...}}
  {"type": "candidates",   "candidates": [...]}
  {"type": "model_ready",  "stl_url": "/cad/model/.../part.stl", ...}
  {"type": "done",         "session_id": "..."}
  {"type": "error",        "message": "..."}
"""
from __future__ import annotations
import json
import logging
import os
import re
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from orynd_core.auth import UserContext, optional_user

from orynd_core.agents.base import AgentContext
from orynd_core.agents.cad import CADAgent
from orynd_core.agents.memory import MemoryAgent
from orynd_core.agents.workspace import WorkspaceAgent
from orynd_core.agents.orchestrator import Pipeline
from orynd_core.models.schemas import Candidate
from orynd_core.services import session_store

log = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "anonymous"
    user_id: str = "anonymous"
    platform: str = "desktop"
    # Сквозной контекст: состояние workspace, которое агент видит автоматически
    # (текущая модель/размеры, выбранный примитив, идущие задачи, surface-источник).
    # Любая поверхность шлёт своё состояние → оркестратор в контексте происходящего.
    context: dict | None = None


def _is_complex_input(message: str) -> bool:
    """Founder principle: a clear command routes fine on the cheap local model, but a
    long, messy, multi-clause request ('big tirade', poorly structured) needs a
    stronger brain to UNDERSTAND before routing. Heuristic: many words OR many
    sentence/clause separators."""
    m = (message or "").strip()
    words = len(m.split())
    breaks = m.count(".") + m.count("?") + m.count("!") + m.count(",") + m.count(";")
    return words > 25 or breaks >= 4


def _make_provider(message: str = ""):
    """Pick the ORCHESTRATION provider (intent + tool routing/execution).

    Founder principle (MODEL_LAYER_MAP.md): orchestration is LIGHT — a clear command
    routes fine on local Ollama, so we DON'T burn a paid model on routing. EXCEPTION
    (A2): a long/messy/complex request needs a strong model to understand it — if a
    Claude key is present we use it for THAT case. Claude is otherwise reserved for
    the ANSWER step (formation). None → keyword algorithm routing.
    """
    key = os.getenv("ANTHROPIC_API_KEY", "")

    # A2: complex/messy input → strong model for understanding (only if key present).
    if key and _is_complex_input(message):
        from orynd_core.services.llm.claude import ClaudeProvider
        log.info("[chat] complex input → Claude orchestration (understanding)")
        return ClaudeProvider(api_key=key)

    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("ORYND_AGENT_OLLAMA_MODEL", "llama3.2:3b")
    try:
        import httpx
        r = httpx.get(f"{base_url}/api/tags", timeout=1.5)
        if r.status_code == 200:
            from orynd_core.services.llm.local import LocalProvider
            log.info("[chat] orchestration → local Ollama (%s)", model)
            return LocalProvider(base_url=base_url, model=model)
    except Exception as e:
        log.info("[chat] Ollama not reachable (%s)", e)

    # Ollama down — fall back to Claude for orchestration if a key exists.
    if key:
        from orynd_core.services.llm.claude import ClaudeProvider
        log.info("[chat] Ollama down → Claude orchestration fallback")
        return ClaudeProvider(api_key=key)
    return None


def _ev(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"


# ── Deterministic primitive fast-path ────────────────────────────────────────
# Simple build commands ("создай цилиндр r=20 h=30", "куб 40мм", "box 10x20x30")
# must build CORRECT geometry every time. The local 3b model is too weak — it
# draws a rectangle for a cylinder and leaks tool-call JSON into the text. So we
# parse these deterministically and build via the same path /cad/execute uses:
# instant, correct, no LLM. Anything ambiguous still falls through to the agent.
_SEARCH_INTENT = re.compile(r"\b(найд|ищ|поиск|поищ|search|find|download|скач)", re.I)

# filler words to drop from a search query, and common maker/CAD terms RU→EN so
# Russian queries actually hit the (English) model platforms.
_SEARCH_FILLER = {
    "по","модель","модели","моделям","моделей","модельку","мне","для","пожалуйста","please",
    "нужен","нужна","нужно","надо","типа","это","вот","давай","давайте","сделай","сделаем",
    "поиск","поищи","найди","найдите","ищу","ищи","search","find","download","скачай","скачать",
    "a","an","the","me","need","for","some","i",
}
_RU_EN_TERMS = {
    "споллер":"spoiler","спойлер":"spoiler","болт":"bolt","гайка":"nut","винт":"screw",
    "подшипник":"bearing","шестерня":"gear","шестерёнка":"gear","шестеренка":"gear",
    "кронштейн":"bracket","корпус":"housing","крышка":"lid","держатель":"holder",
    "крепление":"mount","зажим":"clip","ручка":"handle","колесо":"wheel","вал":"shaft",
    "втулка":"bushing","шкив":"pulley","муфта":"coupler","пружина":"spring","адаптер":"adapter",
    "разъём":"connector","разъем":"connector","фланец":"flange","редуктор":"gearbox",
    "диск":"disc","тормозной":"brake","рама":"frame","петля":"hinge","стойка":"standoff",
    "проставка":"spacer","заглушка":"cap","кожух":"cover","корпуса":"housing","дрон":"drone",
}

def _normalize_search_query(text: str) -> str:
    """Extract the actual object from a messy phrase + map common RU terms to EN."""
    words = re.findall(r"[0-9a-zA-Zа-яёА-ЯЁ]+", (text or "").lower())
    out = [_RU_EN_TERMS.get(w, w) for w in words if w not in _SEARCH_FILLER]
    return " ".join(out).strip()

# Deterministic research intent (Block A2 + D1): a clear "исследуй/research X" runs deep
# research and shows a real result, instead of the local model hallucinating an answer.
_RESEARCH_INTENT = re.compile(r"\b(исследуй|исследова\w*|ресёрч|ресерч|разузнай|изучи|deep\s*research|research|узна\w*\s+(по)?больше)\b", re.I)
_RESEARCH_STRIP = re.compile(r"\b(исследуй|исследова\w*|запусти|сделай|ресёрч|ресерч|разузнай|изучи|deep\s*research|research|про|по\s+теме|узна\w*\s+(по)?больше|давай|мне|нужен|нужно)\b", re.I)

def _normalize_research_topic(text: str) -> str:
    t = _RESEARCH_STRIP.sub(" ", text or "")
    return re.sub(r"\s+", " ", t).strip(" .,:—-") or (text or "").strip()
_PRIMITIVE_TYPES = {"cylinder", "box", "cube"}


def _primitive_summary(ops: list[dict], props: dict) -> str:
    parts: list[str] = []
    for op in ops:
        t = op.get("type")
        p = op.get("parameters", {}) or {}
        if t == "cylinder":
            parts.append(f"cylinder Ø{float(p.get('radius', 0)) * 2:.0f} mm × H{float(p.get('height', 0)):.0f} mm")
        elif t in ("box", "cube"):
            parts.append(f"box {p.get('sx', '?')}×{p.get('sy', '?')}×{p.get('sz', '?')} mm")
    out = "Built " + (", ".join(parts) if parts else "model") + "."
    vol = props.get("volume_mm3")
    if vol:
        out += f" Volume {float(vol):.0f} mm³. STL/STEP/OBJ ready."
    return out


# "build another one" / "ещё один" — repeat the last primitive as a new object.
_REPEAT_INTENT = re.compile(r"\b(ещ[её]\s+(один|раз)|так(ой|ую)\s+же|another|one\s+more|same\s+again)\b", re.I)
# explicit ADD intent — append the NEW primitive beside existing ones (multi-object).
# Without one of these, a plain "создай/build X" makes a FRESH scene (replace), not a pile.
_ADD_INTENT = re.compile(
    r"\b(добав\w*|добавь(?:-ка)?|присоедин\w*|и\s+ещ[её]|also\s+add|"
    r"add\s+(a\s+|an\s+)?(cylinder|box|cube|cyl))\b",
    re.I,
)
_HALF_SIZE_INTENT = re.compile(
    r"(?:в\s*(?:два|2)\s*раза\s*меньше|половин(?:а|ный|ного)?\s*размер|"
    r"поменьше\s*в\s*(?:два|2)\s*раза|half\s*(?:size)?|twice\s*smaller)",
    re.I,
)

# Block B1 — parts we cannot build procedurally yet (need computed tooth/thread/blade profiles).
# A clear BUILD command for one of these must NOT fall through to the 3b model, which fakes a
# box+hole with invented dimensions. Answer honestly and offer real alternatives instead.
_BUILD_INTENT = re.compile(r"\b(созда\w*|сдела\w*|постро\w*|нарису\w*|сгенерир\w*|build|make|create|generate|draw|model)\b", re.I)
_COMPLEX_PART = re.compile(
    r"(шестер[её]?н\w*|зубчат\w*|gear|звёздочк\w*|звездочк\w*|sprocket|спираль\w*|spiral|"
    r"резьб\w*|thread|пружин\w*|spring|крыльчатк\w*|impeller|турбин\w*|turbine|propeller|"
    r"кулач\w*|червяч\w*|worm|шлиц\w*|spline|тормозн\w*\s+диск\w*|brake\s+disc\w*)",
    re.I,
)


# Gear specifically — we DO build it procedurally (gear.py). Other complex parts stay honest.
_GEAR_PART = re.compile(r"(шестер[её]?н|зубчат\w*колес|\bgear\b)", re.I)
_BRAKE_DISC_PART = re.compile(r"(тормозн\w*\s+диск\w*|brake\s+disc\w*|brake\s+rotor\w*)", re.I)
# "добавь / ещё одну / append" also counts as a build for procedural parts (was: only создай/build).
_ADD_VERB = re.compile(r"\b(добав\w*|добавь(?:-ка)?|присоедин\w*|append|ещ[её]\s+(одну|один|одну\w*))\b", re.I)
_SECOND_OBJECT_INTENT = re.compile(r"\b(втор\w*|second|another)\b", re.I)
_GEAR_MESH_INTENT = re.compile(
    r"(соедин\w*|сцеп\w*|зацеп\w*|connect|mesh|mate).*(зуб\w*|шестер[её]?н\w*|gear)"
    r"|(?:по\s+зуб\w*)",
    re.I,
)
_ALIGN_AXIS_INTENT = re.compile(r"(соос\w*|по\s+оси|align\s+axis|coaxial|посад\w*.*(?:вал|ось))", re.I)


def _scale_macro_ops(ops: list[dict], factor: float) -> list[dict]:
    """Scale primitive macro parameters for relative commands like "half size"."""
    scaled: list[dict] = []
    for op in ops:
        item = dict(op)
        params = dict(item.get("parameters") or {})
        for key in ("sx", "sy", "sz", "radius", "height", "length", "outer_radius"):
            if isinstance(params.get(key), (int, float)):
                params[key] = float(params[key]) * factor
        item["parameters"] = params
        scaled.append(item)
    return scaled


def _workspace_mode(text: str) -> str:
    return "append" if (_ADD_VERB.search(text) or _REPEAT_INTENT.search(text) or _SECOND_OBJECT_INTENT.search(text)) else "replace"


async def _build_workspace_events(
    *,
    workspace_id: str,
    session_id: str,
    action_input: dict,
    action_result: dict,
    summary: str,
) -> list[dict] | None:
    from orynd_core.services.cad import workspace_document

    operations, doc = workspace_document.compile_to_coreops(workspace_id)
    if not operations:
        return None
    try:
        from orynd_core.services.cad import model_session

        for model_session_id in {session_id, workspace_id, "design" if workspace_id == "ws-design" else session_id}:
            model_session.clear(model_session_id)
            model_session.set_ops(model_session_id, operations)
    except Exception:
        log.exception("[chat] workspace build model_session sync failed")
    ctx = AgentContext(session_id=session_id, extra={"coreops": {"operations": operations, "units": "mm"}})
    await Pipeline([CADAgent()]).run(ctx)
    cad = ctx.extra.get("cad", {})
    if not cad or not cad.get("stl_path"):
        return None
    props = cad.get("properties", {})
    objects = doc.get("objects", [])
    constraints = doc.get("constraints", [])
    return [
        {"type": "agent_call", "agent": "build_3d_model", "input": action_input},
        {"type": "agent_result", "agent": "build_3d_model", "result": action_result | {"object_count": len(objects)}},
        {"type": "workspace.document.updated", "workspace_id": workspace_id, "document": doc},
        {"type": "text", "content": summary},
        {"type": "model_ready",
         "stl_url": f"/cad/model/{session_id}/part.stl",
         "step_url": f"/cad/model/{session_id}/part.step",
         "obj_url": f"/cad/model/{session_id}/part.obj",
         "properties": props,
         "primitives": [],
         "objects": objects,
         "constraints": constraints,
         "workspace_id": workspace_id},
    ]


def _gear_params_for_request(text: str, workspace_id: str) -> dict:
    from orynd_core.services.cad import workspace_document
    from orynd_core.services.macro.gear import parse_gear_params

    last_gear = workspace_document.last_object(workspace_id, "spur_gear")
    if last_gear and (_HALF_SIZE_INTENT.search(text) or (_SECOND_OBJECT_INTENT.search(text) and not re.search(r"\d+\s*зуб", text, re.I))):
        params = dict(last_gear.get("params") or {})
        params["teeth"] = max(6, int(round(float(params.get("teeth", 18)) * 0.5)))
        params["module"] = float(params.get("module", 2.0))
        params["thickness"] = float(params.get("thickness", 8.0))
        params["bore"] = max(4.0, float(params.get("bore", params["module"] * params["teeth"] / 6.0)) * 0.5)
        return params
    return parse_gear_params(text)


async def _semantic_repeat_fastpath_events(req: ChatRequest, session_id: str, workspace_id: str) -> list[dict] | None:
    text = req.message or ""
    if _SEARCH_INTENT.search(text):
        return None
    if not _REPEAT_INTENT.search(text):
        return None
    # Explicit primitive repeats are handled by _primitive_fastpath_events because
    # it can keep the legacy primitives payload. This path is for "такую же" style
    # references to the last semantic workspace object, especially gears/discs.
    if re.search(r"\b(куб|коробк|box|cube|цилиндр|cylinder)\b", text, re.I):
        return None

    try:
        from orynd_core.services.cad import workspace_document
        last = workspace_document.last_object(workspace_id)
        if not last or last.get("kind") in {"box", "cylinder"}:
            return None
        factor = 0.5 if _HALF_SIZE_INTENT.search(text) else 1.0
        obj = workspace_document.clone_last_object(workspace_id, scale=factor)
        if not obj:
            return None
        summary = (
            f"Добавил {obj['id']} как копию предыдущего объекта"
            f"{' в 2 раза меньше' if factor == 0.5 else ''}. "
            f"Kind: {obj['kind']}."
        )
        events = await _build_workspace_events(
            workspace_id=workspace_id,
            session_id=session_id,
            action_input={"clone_last_object": {"scale": factor}},
            action_result={"built": True, "object_id": obj["id"], "kind": obj["kind"], "scale": factor},
            summary=summary,
        )
        if events is None:
            return [{"type": "text", "content": "Object was cloned in the workspace, but CAD rebuild failed."}]
        events.insert(2, {"type": "workspace.object.created", "workspace_id": workspace_id, "object": obj})
        return events
    except Exception:
        log.exception("[chat] semantic repeat fast-path failed")
        return None


async def _assembly_fastpath_events(req: ChatRequest, session_id: str, workspace_id: str) -> list[dict] | None:
    text = req.message or ""
    if _ALIGN_AXIS_INTENT.search(text):
        try:
            from orynd_core.services.cad import workspace_document
            constraint, error = workspace_document.add_align_axis_constraint(workspace_id)
            if error:
                return [{"type": "text", "content": error}]
            if not constraint:
                return None
            summary = (
                f"Выровнял оси {constraint['a']} и {constraint['b']} "
                f"по connector {constraint['connector_a']}."
            )
            events = await _build_workspace_events(
                workspace_id=workspace_id,
                session_id=session_id,
                action_input={"align_axis": constraint},
                action_result={"built": True, "constraint": constraint},
                summary=summary,
            )
            if events is None:
                return [{"type": "text", "content": "Axis constraint was created, but CAD rebuild failed."}]
            events.insert(2, {"type": "workspace.constraint.created", "workspace_id": workspace_id, "constraint": constraint})
            return events
        except Exception:
            log.exception("[chat] align-axis fast-path failed")
            return None

    if not _GEAR_MESH_INTENT.search(text):
        return None
    try:
        from orynd_core.services.cad import workspace_document
        constraint, error = workspace_document.add_gear_mesh_constraint(workspace_id)
        if error:
            return [{"type": "text", "content": error}]
        if not constraint:
            return None
        summary = (
            f"Сцепил шестерни {constraint['a']} и {constraint['b']} по зубьям. "
            f"Межосевое расстояние {constraint['center_distance']:.1f} мм, ratio {constraint['ratio']}."
        )
        events = await _build_workspace_events(
            workspace_id=workspace_id,
            session_id=session_id,
            action_input={"gear_mesh": constraint},
            action_result={"built": True, "constraint": constraint},
            summary=summary,
        )
        if events is None:
            return [{"type": "text", "content": "Gear mesh constraint was created, but CAD rebuild failed."}]
        events.insert(2, {"type": "workspace.constraint.created", "workspace_id": workspace_id, "constraint": constraint})
        return events
    except Exception:
        log.exception("[chat] assembly fast-path failed")
        return None


async def _complex_part_fastpath_events(req: ChatRequest, session_id: str, workspace_id: str | None = None) -> list[dict] | None:
    """Complex procedural part on a BUILD request.

    GEAR → build a REAL parametric gear (CoreOps polygon → Extrude → bore).
    Other parts (thread/spring/turbine/…) → honest refusal, never invented geometry.
    Returns events, or None to fall through.
    """
    text = req.message or ""
    if _SEARCH_INTENT.search(text):
        return None  # "найди шестерню" → let search handle it
    if not ((_BUILD_INTENT.search(text) or _ADD_VERB.search(text)) and _COMPLEX_PART.search(text)):
        return None

    # ── GEAR → real procedural build ──────────────────────────────────────────
    if _GEAR_PART.search(text):
        try:
            from orynd_core.services.cad import workspace_document
            from orynd_core.services.macro.gear import gear_coreops
            doc_id = workspace_id or session_id
            params = _gear_params_for_request(text, doc_id)
            _ops, info = gear_coreops(**params)
            mode = _workspace_mode(text)
            obj = workspace_document.add_object(doc_id, "spur_gear", params, mode=mode)
            events = await _build_workspace_events(
                workspace_id=doc_id,
                session_id=session_id,
                action_input={"gear": info, "mode": mode},
                action_result={"built": True, "gear": info, "object_id": obj["id"]},
                summary="",
            )
            if events is not None:
                doc = next((e.get("document") for e in events if e.get("type") == "workspace.document.updated"), {}) or {}
                objects = doc.get("objects", [])
                model_event = next((e for e in events if e.get("type") == "model_ready"), {})
                props = model_event.get("properties", {}) or {}
                vol = round(props.get("volume_mm3", 0) / 1000.0, 1)
                summary = (
                    f"Built gear {obj['id']}: {info['teeth']} teeth, module {info['module']}, "
                    f"Ø{info['outer_diameter']} mm, thickness {info['thickness']} mm, "
                    f"bore Ø{info['bore']} mm. Volume {vol} cm³. "
                    f"Objects in workspace: {len(objects)}. (Simplified tooth profile, not involute.)"
                )
                for event in events:
                    if event.get("type") == "text":
                        event["content"] = summary
                events.insert(2, {"type": "workspace.object.created", "workspace_id": doc_id, "object": obj})
                return events
            log.warning("[chat] gear build produced no STL — falling back to honest msg")
        except Exception:
            log.exception("[chat] gear fast-path build failed")
        # fall through to honest message below if the build failed

    # ── BRAKE DISC → real procedural build ───────────────────────────────────
    if _BRAKE_DISC_PART.search(text):
        try:
            from orynd_core.services.cad import workspace_document
            from orynd_core.services.macro.disc import disc_coreops, parse_disc_params
            doc_id = workspace_id or session_id
            params = parse_disc_params(text)
            _ops, info = disc_coreops(**params)
            mode = _workspace_mode(text)
            obj = workspace_document.add_object(doc_id, "brake_disc", params, mode=mode)
            events = await _build_workspace_events(
                workspace_id=doc_id,
                session_id=session_id,
                action_input={"brake_disc": info, "mode": mode},
                action_result={"built": True, "brake_disc": info, "object_id": obj["id"]},
                summary="",
            )
            if events is not None:
                doc = next((e.get("document") for e in events if e.get("type") == "workspace.document.updated"), {}) or {}
                summary = (
                    f"Built brake disc {obj['id']}: Ø{info['diameter']} mm, thickness {info['thickness']} mm, "
                    f"bore Ø{info['bore']} mm, {info['bolts']} bolt holes on Ø{info['bolt_circle']} mm. "
                    f"Objects in workspace: {len(doc.get('objects', []))}."
                )
                for event in events:
                    if event.get("type") == "text":
                        event["content"] = summary
                events.insert(2, {"type": "workspace.object.created", "workspace_id": doc_id, "object": obj})
                return events
            log.warning("[chat] brake disc build produced no STL — falling back to honest msg")
        except Exception:
            log.exception("[chat] brake disc fast-path build failed")
        # fall through to honest message below if the build failed

    # ── Other complex parts → honest refusal ──────────────────────────────────
    part = _COMPLEX_PART.search(text).group(0).strip()
    msg = (
        f"“{part}” is a procedural part — its profile (thread/blades) must be computed, "
        f"so I won't fake the exact geometry. To avoid inventing dimensions:\n"
        f"• find a ready model — type “find {part}”;\n"
        f"• build a blank for it — give a primitive, e.g. “cylinder r=30 h=8”;\n"
        f"• procedural “{part}” generation — in progress (Block H)."
    )
    return [{"type": "text", "content": msg}]


# Block A — deterministic SELECT routing. "выбери первый / возьми №2 / последний" picks a
# candidate from the last search instead of relying on the 3b model to call select_model.
_SELECT_INTENT = re.compile(r"\b(выбер\w*|выбра\w*|возьм\w*|select|choose|останов\w*\s+на)\b", re.I)
_ORDINALS = [
    (re.compile(r"\b(перв\w*|first|1-?[йяое]?)\b", re.I), 0),
    (re.compile(r"\b(втор\w*|second|2-?[йяое]?)\b", re.I), 1),
    (re.compile(r"\b(трет\w*|third|3-?[йяое]?)\b", re.I), 2),
    (re.compile(r"\b(четв[её]рт\w*|fourth)\b", re.I), 3),
    (re.compile(r"\b(пят\w*|fifth)\b", re.I), 4),
]
_LAST = re.compile(r"\b(послед\w*|last)\b", re.I)
_NUMBERED = re.compile(r"(?:№|#|номер\s*|вариант\s*|model\s*)\s*(\d+)", re.I)


def _parse_select_index(text: str) -> int | None:
    if _LAST.search(text):
        return -1
    for rx, idx in _ORDINALS:
        if rx.search(text):
            return idx
    m = _NUMBERED.search(text)
    if m:
        return max(0, int(m.group(1)) - 1)
    # bare number only counts if a select verb is also present ("выбери 2")
    if _SELECT_INTENT.search(text):
        m2 = re.search(r"\b(\d{1,2})\b", text)
        if m2:
            return max(0, int(m2.group(1)) - 1)
    return None


async def _select_fastpath_events(req: ChatRequest, session_id: str) -> list[dict] | None:
    """Deterministically pick a candidate from the last search. Returns events or None."""
    text = req.message or ""
    if _SEARCH_INTENT.search(text) or _BUILD_INTENT.search(text):
        return None  # "найди/создай первый X" is search/build, not a pick
    verb = bool(_SELECT_INTENT.search(text))
    ordlike = any(rx.search(text) for rx, _ in _ORDINALS) or bool(_LAST.search(text))
    numbered = bool(_NUMBERED.search(text))
    short = len(text.split()) <= 3
    # Pick only on: explicit verb, a bare short ordinal ("второй"), or explicit «№2».
    if not (verb or numbered or (ordlike and short)):
        return None
    cands = session_store.get_candidates(session_id)
    if not cands:
        return [{"type": "text", "content": "Nothing to select yet — search first: “find <part>”."}]
    idx = _parse_select_index(text)
    if idx is None:
        return None  # ambiguous → let the agent handle it
    if idx == -1:
        idx = len(cands) - 1
    if idx < 0 or idx >= len(cands):
        return [{"type": "text", "content": f"There are {len(cands)} results — pick a number from 1 to {len(cands)}."}]
    c = cands[idx]
    session_store.set_selected(session_id, c)
    name = c.get("name", "model")
    src = c.get("source_url") or c.get("stl_url") or ""
    summary = f"Selected #{idx + 1}: {name}." + (f"\nSource: {src}" if src else "") + \
        "\nI can suggest fabrication (“how to print?”) or continue."
    return [
        {"type": "agent_call", "agent": "select_model", "input": {"index": idx}},
        {"type": "agent_result", "agent": "select_model", "result": {"selected": name, "index": idx}},
        {"type": "text", "content": summary},
    ]


# Block A — deterministic FABRICATION routing. "как напечатать / материал / параметры печати"
# must reliably produce a manufacturing recommendation (FabricationAgent algorithm fallback,
# no LLM needed) instead of relying on the 3b model to decide to call the tool (the "50/50" bug).
_FAB_INTENT = re.compile(
    r"(как\s+(?:напечат|печат|изготов|сдела\w*\s+деталь|произв)|чем\s+печат|"
    r"матери[ао]л\w*|как[ой|им]\s+пластик|параметр\w*\s+печат|настройк\w*\s+печат|"
    r"fabricat\w*|infill|заполнени\w*|\bcnc\b|чпу|фрезеров\w*|как\s+это\s+(?:напечат|изготов))",
    re.I,
)
# Stop-words removed when deriving the part name (whole-word, so we don't mangle "какого"→"ого").
_FAB_STOP = {
    "как", "чем", "какой", "какая", "какие", "каким", "какого", "это", "напечатать", "напечатай",
    "печать", "печати", "печатать", "изготовить", "изготовление", "изготовления", "произвести",
    "параметры", "параметр", "настройки", "настройка", "материал", "материала", "материалы",
    "пластик", "пластика", "сделать", "сделай", "деталь", "для", "мне", "пожалуйста", "из",
    "fabricate", "cnc", "чпу", "фрезеровка", "фрезеровать", "лазер", "recommend", "и", "а", "под",
}


def _fab_subject(text: str) -> str:
    words = re.findall(r"[\w-]+", text or "", re.UNICODE)
    keep = [w for w in words if w.lower() not in _FAB_STOP and not w.isdigit()]
    return " ".join(keep).strip()


async def _fabrication_fastpath_events(req: ChatRequest, session_id: str) -> list[dict] | None:
    """Deterministic fabrication recommendation. Returns events or None.

    Uses the FabricationAgent ALGORITHM fallback (provider=None) → always answers,
    keyword-driven method+material. Subject = selected candidate, else parsed from text.
    """
    text = (req.message or "").strip()
    if not _FAB_INTENT.search(text) or _SEARCH_INTENT.search(text):
        return None

    # Prefer a part named in the message; else the selected candidate; else last search top.
    subject = _fab_subject(text)
    sel = session_store.get_selected(session_id)
    cands = session_store.get_candidates(session_id)
    if subject and len(subject) > 1:
        name = description = subject
    elif sel:
        name, description = sel.get("name", "part") or "part", sel.get("description", "") or ""
    elif cands:
        name, description = cands[0].get("name", "part") or "part", cands[0].get("description", "") or ""
    else:
        name = description = "part"

    try:
        from orynd_core.agents.fabrication import FabricationAgent
        ctx = AgentContext(session_id=session_id, raw_text=name)
        ctx.selected = {"name": name, "description": description}
        await Pipeline([FabricationAgent(provider=None)]).run(ctx)
        pack = ctx.extra.get("fabrication", {}) or {}
    except Exception:
        log.exception("[chat] fabrication fast-path failed")
        return None
    if not pack:
        return None

    method = str(pack.get("recommended_method", "fdm")).upper()
    material = pack.get("material", "PLA")
    alts = ", ".join(str(a).upper() for a in (pack.get("alternative_methods") or [])[:2])
    reason = pack.get("material_reason", "") or pack.get("notes", "")
    summary = (
        f"Fabrication for “{name}”: method **{method}**, material **{material}**"
        f"{f' (alternatives: {alts})' if alts else ''}.\n"
        f"{reason}".strip()
    )
    return [
        {"type": "agent_call", "agent": "get_fabrication", "input": {"part": name}},
        {"type": "agent_result", "agent": "get_fabrication", "result": pack},
        {"type": "text", "content": summary},
    ]


async def _primitive_fastpath_events(req: ChatRequest, session_id: str, document_id: str | None = None) -> list[dict] | None:
    """Return NDJSON events for a deterministic primitive build, or None if the
    message isn't a clear primitive command (→ fall through to the LLM agent).

    Builds via the session model document (model_session) so each primitive ADDS
    a new object beside the existing ones instead of replacing the scene.
    """
    from orynd_core.services.cad import model_session

    text = (req.message or "").strip()
    if not text or _SEARCH_INTENT.search(text):
        return None
    doc_id = document_id or session_id

    # "построй ещё один" → repeat the last primitive in this session's document
    new_ops: list[dict]
    if _REPEAT_INTENT.search(text):
        existing = model_session.get_ops(doc_id)
        last_prim = next((o for o in reversed(existing) if (o.get("type") in _PRIMITIVE_TYPES)), None)
        if not last_prim:
            return None  # nothing to repeat → let the LLM ask for specifics
        clean = {k: v for k, v in (last_prim.get("parameters") or {}).items() if not k.startswith("_")}
        new_ops = [{"type": last_prim["type"], "parameters": clean}]
        if _HALF_SIZE_INTENT.search(text):
            new_ops = _scale_macro_ops(new_ops, 0.5)
    else:
        try:
            from orynd_core.services.macro.parser import parse_text_to_coreops
            parsed = parse_text_to_coreops(text)
        except Exception:
            return None
        ops = parsed.operations
        if not ops or parsed.confidence < 0.99:
            return None
        if any((op.get("type") not in _PRIMITIVE_TYPES) for op in ops):
            return None
        new_ops = ops

    # Intent: a plain "создай/build X" = FRESH scene (replace). Only "ещё один"
    # (repeat) or "добавь X" (add) accumulate beside existing objects. This fixes
    # the "I said create a cylinder and got 2" bug — builds no longer pile onto
    # stale session state left over from earlier (incl. across a UI reload).
    accumulate = bool(_REPEAT_INTENT.search(text) or _ADD_INTENT.search(text))
    try:
        from orynd_core.routers.cad import _adapt_operations
        if not accumulate:
            model_session.clear(doc_id)        # fresh single object, drop stale state
        for op in new_ops:
            model_session.append_ops(doc_id, [model_session.place_next(doc_id, op)])
        full = model_session.get_ops(doc_id)
        try:
            from orynd_core.services.cad import workspace_document
            workspace_doc = workspace_document.replace_primitives(doc_id, full)
        except Exception:
            log.exception("[chat] primitive workspace document sync failed")
            workspace_doc = {"workspace_id": doc_id, "objects": [], "constraints": []}
        operations, _notes, primitives = _adapt_operations(full)
        if not operations:
            return None
        ctx = AgentContext(
            session_id=session_id,
            extra={"coreops": {"operations": operations, "units": "mm"}},
        )
        await Pipeline([CADAgent()]).run(ctx)
    except Exception:
        log.exception("[chat] primitive fast-path build failed")
        return None

    cad = ctx.extra.get("cad", {})
    if not cad or not cad.get("stl_path"):
        return None

    props = cad.get("properties", {})
    n = len(primitives)
    summary = _primitive_summary(new_ops, props)
    if n > 1:
        summary += f" Objects in scene: {n}."
    return [
        {"type": "agent_call", "agent": "build_3d_model", "input": {"operations": new_ops}},
        {"type": "agent_result", "agent": "build_3d_model",
         "result": {"built": True, "object_count": n}},
        {"type": "workspace.document.updated", "workspace_id": doc_id, "document": workspace_doc},
        {"type": "text", "content": summary},
        {"type": "model_ready",
         "stl_url": f"/cad/model/{session_id}/part.stl",
         "step_url": f"/cad/model/{session_id}/part.step",
         "obj_url": f"/cad/model/{session_id}/part.obj",
         "properties": props,
         "primitives": primitives,
         "objects": workspace_doc.get("objects", []),
         "constraints": workspace_doc.get("constraints", []),
         "workspace_id": doc_id},
    ]


async def _search_fastpath_events(req: ChatRequest, session_id: str) -> list[dict] | None:
    """Deterministic search (Block A1): a clear 'найди/поиск/search X' ALWAYS runs the
    retrieval agent and returns candidates — never relies on the weak local model
    deciding to call the tool. Fixes 'найди F1 spoiler → ничего не произошло'."""
    text = (req.message or "").strip()
    if not text or not _SEARCH_INTENT.search(text):
        return None
    # don't hijack build / research / mesh phrasings
    if re.search(r"\b(создай|сделай\s+(?!поиск)|построй|build|нарисуй|extrude|исследуй|deep\s*research|research|decompose|разлож)", text, re.I):
        return None
    query = _normalize_search_query(text) or text
    try:
        from orynd_core.agents.retrieval import RetrievalAgent
        from orynd_core.agents.orchestrator import Pipeline as _P
        ctx = AgentContext(session_id=session_id, raw_text=query)
        ctx.intent = {"keywords": query.split(), "object_name": query, "_source": "search_fastpath"}
        ctx.extra["search_query"] = query
        await _P([RetrievalAgent(limit=6)]).run(ctx)
        cands = [c if isinstance(c, dict) else c.model_dump() for c in (ctx.candidates or [])]
    except Exception:
        log.exception("[chat] search fast-path failed")
        return None
    n = len(cands)
    if n:
        names = ", ".join((c.get("name") or "")[:36] for c in cands[:3] if c.get("name"))
        summary = f"Found {n} models for “{query}”. Top: {names}. Pick a card and I'll drop it into the scene."
    else:
        summary = f"No models found for “{query}” on the connected platforms. Try a different name or fewer words?"
    return [
        {"type": "agent_call", "agent": "search_models", "input": {"query": query}},
        {"type": "agent_result", "agent": "search_models", "result": {"found": n}},
        {"type": "candidates", "candidates": cands},
        {"type": "text", "content": summary},
    ]


async def _stream(req: ChatRequest) -> AsyncIterator[str]:
    session_id = req.session_id or str(uuid.uuid4())
    # Block A: mode from the UI dropdown. "deep" forces deep research (skip build/search/etc);
    # other modes use the normal deterministic cascade. (plan/ask behaviour is a later refinement.)
    mode = ((req.context or {}).get("mode") or "auto").lower()
    provider = _make_provider(req.message)

    ctx = AgentContext(
        session_id=session_id,
        user_id=req.user_id,
        raw_text=req.message,
        platform=req.platform,
    )
    # Сквозной контекст workspace → оркестратор видит происходящее автоматически
    ctx.extra["workspace_context"] = req.context or {}

    # workspace_id: explicit from context, or fallback to session_id
    workspace_id = (req.context or {}).get("workspace_id") or session_id

    # Load memory first
    await Pipeline([MemoryAgent(mode="load")]).run(ctx)

    from orynd_core.services.event_bus import bus

    async def _bridge_model_ready(event: dict) -> None:
        """Mirror a built model to shared state + SSE bus (so the 3D viewport loads it)."""
        try:
            from orynd_core.services import workspace_state
            await workspace_state.update(workspace_id, {
                "selected_model": {
                    "stl_url": event.get("stl_url"),
                    "step_url": event.get("step_url"),
                    "obj_url": event.get("obj_url"),
                    "properties": event.get("properties", {}),
                    "objects": event.get("objects", []),
                    "constraints": event.get("constraints", []),
                    "session_id": session_id,
                },
                "document": {
                    "workspace_id": event.get("workspace_id", workspace_id),
                    "objects": event.get("objects", []),
                    "constraints": event.get("constraints", []),
                },
                "last_tool": "build_3d_model",
            })
        except Exception:
            pass
        await bus.publish("model.ready", {
            "stl_url": event.get("stl_url"),
            "step_url": event.get("step_url"),
            "obj_url": event.get("obj_url"),
            "properties": event.get("properties", {}),
            "primitives": event.get("primitives", []),
            "objects": event.get("objects", []),
            "constraints": event.get("constraints", []),
            "workspace_id": event.get("workspace_id", workspace_id),
            "session_id": session_id,
            "source": "build",
        })

    assembly_events = None if mode == "deep" else await _assembly_fastpath_events(req, session_id, workspace_id)
    if assembly_events is not None:
        for event in assembly_events:
            yield _ev(event)
            if event.get("type") == "model_ready" and event.get("stl_url"):
                await _bridge_model_ready(event)
        ctx.extra["workspace_response"] = next(
            (e["content"] for e in assembly_events if e.get("type") == "text"), ""
        )
        ctx.extra["query"] = req.message
        await Pipeline([MemoryAgent(mode="save")]).run(ctx)
        yield _ev({"type": "done", "session_id": session_id})
        return

    semantic_repeat_events = None if mode == "deep" else await _semantic_repeat_fastpath_events(req, session_id, workspace_id)
    if semantic_repeat_events is not None:
        for event in semantic_repeat_events:
            yield _ev(event)
            if event.get("type") == "model_ready" and event.get("stl_url"):
                await _bridge_model_ready(event)
        ctx.extra["workspace_response"] = next(
            (e["content"] for e in semantic_repeat_events if e.get("type") == "text"), ""
        )
        ctx.extra["query"] = req.message
        await Pipeline([MemoryAgent(mode="save")]).run(ctx)
        yield _ev({"type": "done", "session_id": session_id})
        return

    # Deterministic fast-path: clear primitive commands build instantly + correct,
    # skipping the unreliable LLM loop entirely.
    fast_events = None if mode == "deep" else await _primitive_fastpath_events(req, session_id, workspace_id)
    if fast_events is not None:
        for event in fast_events:
            yield _ev(event)
            if event.get("type") == "model_ready" and event.get("stl_url"):
                await _bridge_model_ready(event)
        ctx.extra["workspace_response"] = next(
            (e["content"] for e in fast_events if e.get("type") == "text"), ""
        )
        ctx.extra["query"] = req.message
        await Pipeline([MemoryAgent(mode="save")]).run(ctx)
        yield _ev({"type": "done", "session_id": session_id})
        return

    # Deterministic search (Block A1): guarantee "найди/поиск X" → real candidates,
    # not a silent no-op from the local model failing to call the tool.
    search_events = None if mode == "deep" else await _search_fastpath_events(req, session_id)
    if search_events is not None:
        for event in search_events:
            yield _ev(event)
            if event.get("type") == "candidates":
                session_store.set_candidates(session_id, event.get("candidates", []))
        ctx.extra["workspace_response"] = next(
            (e["content"] for e in search_events if e.get("type") == "text"), ""
        )
        ctx.extra["query"] = req.message
        await Pipeline([MemoryAgent(mode="save")]).run(ctx)
        yield _ev({"type": "done", "session_id": session_id})
        return

    # Block B1: honest answer for a complex procedural part (gear/thread/spring/…) instead of
    # a 3b-hallucinated box+hole with invented dimensions.
    complex_events = None if mode == "deep" else await _complex_part_fastpath_events(req, session_id, workspace_id)
    if complex_events is not None:
        for event in complex_events:
            yield _ev(event)
            if event.get("type") == "model_ready" and event.get("stl_url"):
                await _bridge_model_ready(event)
        ctx.extra["workspace_response"] = next(
            (e["content"] for e in complex_events if e.get("type") == "text"), ""
        )
        ctx.extra["query"] = req.message
        await Pipeline([MemoryAgent(mode="save")]).run(ctx)
        yield _ev({"type": "done", "session_id": session_id})
        return

    # Block A: deterministic select — "выбери первый / возьми №2" picks from the last search.
    select_events = None if mode == "deep" else await _select_fastpath_events(req, session_id)
    if select_events is not None:
        for event in select_events:
            yield _ev(event)
        ctx.extra["workspace_response"] = next(
            (e["content"] for e in select_events if e.get("type") == "text"), ""
        )
        ctx.extra["query"] = req.message
        await Pipeline([MemoryAgent(mode="save")]).run(ctx)
        yield _ev({"type": "done", "session_id": session_id})
        return

    # Block A: deterministic fabrication — "как напечатать / материал" → guaranteed
    # recommendation (algorithm fallback), not a 3b coin-flip on calling the tool.
    fab_events = None if mode == "deep" else await _fabrication_fastpath_events(req, session_id)
    if fab_events is not None:
        for event in fab_events:
            yield _ev(event)
        ctx.extra["workspace_response"] = next(
            (e["content"] for e in fab_events if e.get("type") == "text"), ""
        )
        ctx.extra["query"] = req.message
        await Pipeline([MemoryAgent(mode="save")]).run(ctx)
        yield _ev({"type": "done", "session_id": session_id})
        return

    # Deterministic research (Block A2 + D1/D2): run deep research + stream progress +
    # show a REAL result (sources, recommendations, Library article), not a 3b hallucination.
    if mode == "deep" or _RESEARCH_INTENT.search(req.message or ""):
        topic = _normalize_research_topic(req.message) or (req.message or "").strip()
        yield _ev({"type": "agent_call", "agent": "deep_research", "input": {"topic": topic}})
        yield _ev({"type": "stage", "name": "Searching sources", "status": "run"})
        research: dict = {}
        try:
            from orynd_core.agents.research import DeepResearchAgent
            rctx = AgentContext(session_id=session_id, raw_text=topic)
            await DeepResearchAgent(depth=1).run(rctx)
            research = rctx.extra.get("research", {}) or {}
        except Exception:
            log.exception("[chat] research fast-path failed")
        sources = research.get("sources", []) or []
        recs = research.get("recommendations") or research.get("summary") or ""
        article_id = None
        n = len(sources)
        try:
            from orynd_core.routers.research import _push_article_to_library, _research_has_sources
            if _research_has_sources(research):
                article_id = await _push_article_to_library(topic, research)
        except Exception:
            pass
        yield _ev({"type": "stage", "name": "Synthesis", "status": "done"})
        if n:
            yield _ev({"type": "research_ready", "topic": topic, "sources": sources[:8],
                       "sources_total": research.get("sources_total", n),
                       "recommendations": recs, "gaps": research.get("gaps", [])[:5], "article_id": article_id})
            summary = (f"Deep research on “{topic}” done: {n} sources, findings synthesized"
                       f"{' → saved to Library' if article_id else ''}. Open the report?")
        else:
            yield _ev({"type": "research_empty", "topic": topic, "sources": [],
                       "sources_total": 0, "recommendations": recs, "article_id": None})
            summary = (f"Couldn't gather sources for “{topic}” (Ollama/network may be down). "
                       f"Try narrowing the topic.")
        yield _ev({"type": "text", "content": summary})
        ctx.extra["workspace_response"] = summary
        ctx.extra["query"] = req.message
        await Pipeline([MemoryAgent(mode="save")]).run(ctx)
        yield _ev({"type": "done", "session_id": session_id})
        return

    # Run workspace agent — stream all events
    agent = WorkspaceAgent(provider=provider)
    candidates_sent = False

    try:
        async for event in agent.stream(ctx):
            yield _ev(event)

            # Track if we already sent candidates
            if event.get("type") == "candidates":
                candidates_sent = True

                # Cache for /select
                raw = event.get("candidates", [])
                session_store.set_candidates(session_id, raw)

                # Wire #5: update shared workspace state so all surfaces see candidates
                try:
                    from orynd_core.services import workspace_state
                    await workspace_state.update(workspace_id, {
                        "candidates": raw,
                        "last_tool": "search_models",
                    })
                except Exception:
                    pass

            # Bridge a freshly-built model to the SSE event bus so design-system
            # frontends (which listen on /events/stream for "model.ready") load
            # it into the 3D viewport — not just chat clients reading the stream.
            if event.get("type") == "model_ready" and event.get("stl_url"):
                await _bridge_model_ready(event)

    except Exception as e:
        log.exception("[chat] stream error")
        yield _ev({"type": "error", "message": str(e)})

    # Persist turn
    ctx.extra["workspace_response"] = ctx.extra.get("workspace_response", "")
    ctx.extra["query"] = req.message
    await Pipeline([MemoryAgent(mode="save")]).run(ctx)

    # Emit credits summary for this turn (#credits → chat wire)
    try:
        from orynd_core.services.credits import session_tracker
        credits = session_tracker.get_session(session_id)
        if credits["tool_calls"] > 0:
            yield _ev({"type": "credits_update", **credits})
    except Exception:
        pass

    yield _ev({"type": "done", "session_id": session_id})


@router.get("/llm/status")
async def llm_status() -> dict:
    """Report the REAL active models so the UI can stop lying ('Sonnet 4.5' while
    3b runs). Frontend model-picker reads this to show the truth + which options
    need a key."""
    key = bool(os.getenv("ANTHROPIC_API_KEY", ""))
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    ollama_up, models = False, []
    try:
        import httpx
        r = httpx.get(f"{base_url}/api/tags", timeout=1.0)
        if r.status_code == 200:
            ollama_up = True
            models = [m.get("name") for m in r.json().get("models", [])]
    except Exception:
        pass

    local_orch = os.getenv("ORYND_AGENT_OLLAMA_MODEL", "llama3.2:3b")
    if ollama_up:
        orchestrator = local_orch
    elif key:
        orchestrator = "claude (fallback)"
    else:
        orchestrator = "none"
    formation = "claude-sonnet-4-5" if key else os.getenv("ORYND_FORMATION_MODEL", "llama3.2:3b")

    return {
        "orchestrator": orchestrator,         # routes + executes tools
        "formation": formation,               # writes the final answer
        "claude_key": key,                    # is a paid key active?
        "ollama_up": ollama_up,
        "ollama_models": models,
        "note": "orchestration local by design; Claude (if key) is used for the answer",
    }


class LLMKeyRequest(BaseModel):
    key: str = ""


@router.post("/llm/key")
async def set_llm_key(req: LLMKeyRequest) -> dict:
    """Let the user plug in THEIR OWN Anthropic key at runtime (founder #1).

    All providers read os.getenv('ANTHROPIC_API_KEY') live, so setting it here
    instantly routes the ANSWER step (and complex-input understanding) through
    Claude — no restart. Session-scoped: not written to disk, never echoed back,
    never logged. Send an empty key to clear it (back to local-only)."""
    key = (req.key or "").strip()
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    return {"ok": True, "claude_key": bool(os.getenv("ANTHROPIC_API_KEY", ""))}


@router.post("/chat")
async def chat(
    req: ChatRequest,
    user: UserContext | None = Depends(optional_user),
) -> StreamingResponse:
    # Wire #9-A: if JWT present, override anonymous user_id with real one
    if user:
        req = req.model_copy(update={"user_id": user.id})
    return StreamingResponse(
        _stream(req),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},
    )
