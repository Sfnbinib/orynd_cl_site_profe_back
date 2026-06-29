"""
DeepResearchAgent — multi-source parallel research with dynamic orchestration.

Architecture:
  Phase 1 (PARALLEL, algorithm):
    ├── 3D model sources (Printables, Thingiverse, MakerWorld)
    ├── Web articles (DuckDuckGo)
    └── GitHub repos + code

  Phase 2 (LLM, MEDIUM complexity — Haiku):
    CollectorAgent → analyzes raw results, scores relevance, removes noise

  Phase 3 (LLM, COMPLEX — Sonnet if available, else Haiku):
    SynthesisAgent → builds knowledge map, finds gaps, generates plan

  Phase 4 (DYNAMIC ORCHESTRATION):
    If synthesis finds unexpected gaps → spawns new sub-research tasks
    Loops until confident OR max_depth reached

  Phase 5 (LLM, MEDIUM):
    RecommendationAgent → actionable output:
      - ready-to-use open source solutions
      - what needs to be built from scratch
      - suggested next steps

Input:  ctx.raw_text (research topic)
        ctx.extra.get("research_depth", 1)  — 1=quick, 2=standard, 3=deep
Output: ctx.extra["research"] = {
    "topic": str,
    "sources": [...],       # all found sources
    "knowledge_map": {...}, # structured understanding
    "open_source": [...],   # ready solutions
    "gaps": [...],          # what doesn't exist yet
    "recommendations": str, # final actionable text
    "iterations": int,
}
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from orynd_core.agents.base import AgentContext, AgentResult, BaseAgent
from orynd_core.services.model_router import TaskComplexity, get_provider
from orynd_core.services.llm.base import LLMMessage

log = logging.getLogger(__name__)

# ── System prompts ────────────────────────────────────────────────────────────

_COLLECTOR_SYSTEM = """\
You are a research collector for an engineering workspace.
Given raw search results across multiple sources, analyze and filter them.
Return ONLY valid JSON.

Schema:
{
  "relevant": [
    {
      "title": "...",
      "url": "...",
      "type": "3d_model|github_repo|article|reference",
      "relevance": 0.0-1.0,
      "key_info": "one sentence why this is useful"
    }
  ],
  "total_found": 0,
  "quality": "low|medium|high"
}

Rules:
- Keep only items with relevance > 0.4
- Prioritize: open-source with files > educational articles > references
- Ignore marketing/spam
"""

_SYNTHESIS_SYSTEM = """\
You are a research synthesizer for an engineering workspace.
Given filtered research results, build a structured knowledge map.
Return ONLY valid JSON.

Schema:
{
  "summary": "2-3 sentence overview of what exists",
  "open_source_solutions": [
    {"name": "...", "url": "...", "fit": "perfect|partial|inspiration", "note": "..."}
  ],
  "knowledge_gaps": ["gap1", "gap2"],
  "build_from_scratch": ["component1", "component2"],
  "suggested_queries": ["follow_up_query1", "follow_up_query2"],
  "confidence": 0.0-1.0,
  "needs_more_research": true|false
}

Be honest about gaps. If something doesn't exist as open-source, say so.
Match user's language (Russian if topic in Russian).
"""

_RECOMMENDATION_SYSTEM = """\
You are an engineering advisor in ORYND workspace.
Given research synthesis, produce clear actionable recommendations.
Be direct, technical, specific. No fluff.
Match user's language. Max 300 words.

Structure your response:
1. What's available ready-to-use (open source)
2. What needs modification
3. What needs to be built from scratch
4. Recommended next step
"""

_DYNAMIC_PLAN_SYSTEM = """\
You are an autonomous research orchestrator.
You just got partial research results with gaps.
Decide what additional research is needed.
Return ONLY valid JSON.

Schema:
{
  "should_continue": true|false,
  "reason": "why continue or stop",
  "new_queries": [
    {"query": "...", "source": "web|github|3d_models", "priority": 1-3}
  ]
}

Rules:
- Only continue if gaps are critical to the user's goal
- Max 3 new queries
- If confidence > 0.7 or iterations > 2: should_continue = false
"""


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class ResearchSource:
    title: str
    url: str
    source_type: str   # 3d_model | github_repo | article | reference
    snippet: str = ""
    relevance: float = 0.5


@dataclass
class ResearchState:
    topic: str
    all_sources: list[dict] = field(default_factory=list)
    relevant_sources: list[dict] = field(default_factory=list)
    knowledge_map: dict = field(default_factory=dict)
    iterations: int = 0
    confidence: float = 0.0


# ── Parallel collectors ────────────────────────────────────────────────────────

async def _collect_3d_models(topic: str) -> list[dict]:
    from orynd_core.services.search import printables, thingiverse, makerworld
    results = await asyncio.gather(
        printables.search(topic, limit=4),
        thingiverse.search(topic, limit=4),
        makerworld.search(topic, limit=3),
        return_exceptions=True,
    )
    out = []
    for batch in results:
        if isinstance(batch, Exception):
            continue
        for c in batch:
            d = c.model_dump() if hasattr(c, "model_dump") else c
            out.append({
                "title": d.get("name", ""),
                "url": d.get("source_url", ""),
                "snippet": d.get("description", "")[:150],
                "source": d.get("source", "3d_model"),
                "type": "3d_model",
                "stl_url": d.get("stl_url", ""),
                "printability": d.get("printability", 5),
            })
    return out


async def _collect_web(topic: str) -> list[dict]:
    from orynd_core.services.search.web import search_multi
    queries = [
        f"{topic} 3D model open source",
        f"{topic} engineering design",
        f"{topic} DIY build guide",
    ]
    results = await search_multi(queries, limit_each=4)
    for r in results:
        r["type"] = "article"
    return results


async def _collect_github(topic: str) -> list[dict]:
    from orynd_core.services.search.github_search import search_repos, search_code
    repos, code = await asyncio.gather(
        search_repos(f"{topic} 3d print OR cad OR engineering", limit=5),
        search_code(f"{topic} stl OR scad OR step", limit=3),
        return_exceptions=True,
    )
    out = []
    if not isinstance(repos, Exception):
        out.extend(repos)
    if not isinstance(code, Exception):
        out.extend(code)
    return out


# ── LLM steps ─────────────────────────────────────────────────────────────────

async def _filter_results(raw: list[dict], topic: str) -> list[dict]:
    """Use SIMPLE model to filter and score raw results."""
    provider = get_provider(TaskComplexity.SIMPLE)
    if not provider:
        # No model — return all with default relevance
        return [{**r, "relevance": 0.6, "key_info": r.get("snippet", "")[:80]} for r in raw[:15]]

    payload = json.dumps(raw[:20], ensure_ascii=False)
    prompt = f"Topic: {topic}\n\nResults:\n{payload}"
    try:
        raw_resp = await provider.complete_json(
            [LLMMessage(role="user", content=prompt)],
            system=_COLLECTOR_SYSTEM,
            max_tokens=1500,
        )
        cleaned = re.sub(r"```(?:json)?\s*", "", raw_resp).strip().rstrip("`")
        data = json.loads(cleaned)
        return data.get("relevant", raw[:10])
    except Exception as e:
        log.warning("[research] filter failed: %s", e)
        return raw[:10]


async def _synthesize(relevant: list[dict], topic: str, prev_map: dict) -> dict:
    """Use COMPLEX model to synthesize knowledge map."""
    provider = get_provider(TaskComplexity.COMPLEX)
    if not provider:
        provider = get_provider(TaskComplexity.MEDIUM)
    if not provider:
        return {
            "summary": f"Found {len(relevant)} sources for '{topic}'.",
            "open_source_solutions": [],
            "knowledge_gaps": [],
            "build_from_scratch": [],
            "suggested_queries": [],
            "confidence": 0.4,
            "needs_more_research": False,
        }

    context = ""
    if prev_map:
        context = f"\nPrevious research map:\n{json.dumps(prev_map, ensure_ascii=False)}\n"

    payload = json.dumps(relevant[:12], ensure_ascii=False)
    prompt = f"Topic: {topic}{context}\n\nFiltered sources:\n{payload}"
    try:
        raw_resp = await provider.complete_json(
            [LLMMessage(role="user", content=prompt)],
            system=_SYNTHESIS_SYSTEM,
            max_tokens=1200,
        )
        cleaned = re.sub(r"```(?:json)?\s*", "", raw_resp).strip().rstrip("`")
        return json.loads(cleaned)
    except Exception as e:
        log.warning("[research] synthesis failed: %s", e)
        return {"confidence": 0.3, "needs_more_research": False, "summary": "Synthesis failed."}


async def _decide_next(state: ResearchState, synthesis: dict) -> list[dict]:
    """
    Dynamic orchestration: decide if more research needed and what queries to run.
    Uses SIMPLE model (fast, cheap).
    """
    if state.iterations >= 3:
        return []

    provider = get_provider(TaskComplexity.SIMPLE)
    if not provider:
        return []  # No model — stop here

    prompt = (
        f"Research topic: {state.topic}\n"
        f"Iterations done: {state.iterations}\n"
        f"Synthesis confidence: {synthesis.get('confidence', 0)}\n"
        f"Gaps found: {synthesis.get('knowledge_gaps', [])}\n"
        f"Needs more: {synthesis.get('needs_more_research', False)}\n"
        f"Suggested queries: {synthesis.get('suggested_queries', [])}"
    )
    try:
        raw_resp = await provider.complete_json(
            [LLMMessage(role="user", content=prompt)],
            system=_DYNAMIC_PLAN_SYSTEM,
            max_tokens=400,
        )
        cleaned = re.sub(r"```(?:json)?\s*", "", raw_resp).strip().rstrip("`")
        data = json.loads(cleaned)
        if not data.get("should_continue", False):
            return []
        return data.get("new_queries", [])
    except Exception as e:
        log.warning("[research] dynamic plan failed: %s", e)
        return []


async def _build_recommendations(state: ResearchState, synthesis: dict) -> str:
    """Final step: MEDIUM model builds actionable text recommendation."""
    provider = get_provider(TaskComplexity.MEDIUM)
    if not provider:
        # Fallback: build from synthesis dict
        oss = synthesis.get("open_source_solutions", [])
        gaps = synthesis.get("knowledge_gaps", [])
        lines = [synthesis.get("summary", "")]
        if oss:
            lines.append(f"\nReady solutions: " + ", ".join(s.get("name","") for s in oss[:3]))
        if gaps:
            lines.append(f"\nGaps to fill: " + ", ".join(gaps[:3]))
        return "\n".join(lines)

    prompt = (
        f"Topic: {state.topic}\n"
        f"Synthesis: {json.dumps(synthesis, ensure_ascii=False)}\n"
        f"Total sources reviewed: {len(state.all_sources)}\n"
        f"Research iterations: {state.iterations}"
    )
    try:
        resp = await provider.complete(
            [LLMMessage(role="user", content=prompt)],
            system=_RECOMMENDATION_SYSTEM,
            max_tokens=600,
        )
        return resp.content
    except Exception as e:
        log.warning("[research] recommendations failed: %s", e)
        return synthesis.get("summary", "Research complete.")


# ── Main agent ────────────────────────────────────────────────────────────────

# SAFETY: only ONE deep-research run at a time, process-wide. Each run loads the
# local LLM (2-3 GB) and fans out parallel HTTP — letting two stack can exhaust
# RAM and freeze the machine. A second concurrent request waits for the first.
_RESEARCH_LOCK = asyncio.Semaphore(1)


def _pack_sources(state: "ResearchState") -> list[dict]:
    """Flatten collected sources into a clean, de-duplicated list for the UI.
    Prefers LLM-filtered relevant_sources, falls back to raw all_sources so the
    real HTTP-collected sources are exposed even if the LLM filter failed."""
    src = state.relevant_sources or state.all_sources
    out: list[dict] = []
    seen: set[str] = set()
    for s in src:
        url = s.get("url") or s.get("source_url") or ""
        title = s.get("title") or s.get("name") or url
        if not title or title in seen:
            continue
        seen.add(title)
        out.append({
            "title": title,
            "url": url,
            "type": s.get("type", "reference"),
            "snippet": (s.get("snippet") or s.get("key_info") or "")[:160],
        })
    return out[:12]


def _pack_research_result(state: "ResearchState", synthesis: dict, recommendations) -> dict:
    """Build the research result dict. ALWAYS exposes a `sources` list so /chat
    and the UI can show real sources — even before/without LLM synthesis."""
    synthesis = synthesis or {}
    sources = _pack_sources(state)
    if not recommendations:
        if sources:
            recommendations = synthesis.get("summary") or (
                f"Found {len(sources)} sources on “{state.topic}”. "
                "Open the sources below or narrow the topic for synthesis."
            )
        else:
            recommendations = synthesis.get("summary") or "No sources found — narrow the topic."
    return {
        "topic": state.topic,
        "sources": sources,
        "sources_total": len(state.all_sources),
        "sources_relevant": len(state.relevant_sources),
        "open_source": synthesis.get("open_source_solutions", []),
        "gaps": synthesis.get("knowledge_gaps", []),
        "build_from_scratch": synthesis.get("build_from_scratch", []),
        "knowledge_map": synthesis,
        "recommendations": recommendations,
        "iterations": state.iterations,
        "confidence": synthesis.get("confidence", state.confidence),
    }


class DeepResearchAgent(BaseAgent):
    """
    Multi-source parallel research with dynamic orchestration.
    Uses model_router to pick right model per step.
    """

    name = "research_agent"

    def __init__(self, depth: int = 1) -> None:
        # depth: 1=quick (1 iteration), 2=standard (2), 3=deep (3)
        super().__init__(provider=None)  # model router picks per step
        self.depth = max(1, min(depth, 3))

    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        topic = ctx.raw_text or ctx.extra.get("research_topic", "")
        if not topic:
            return AgentResult.failure(self.name, "No research topic provided")

        # SAFETY: serialise runs (never load the model twice) AND enforce a hard
        # wall-clock budget here — covers EVERY caller (/chat, /research/light, …),
        # not just the workspace tool path. On timeout the caller gets a graceful
        # "timed out" research dict instead of a hung event loop / RAM blowup.
        async with _RESEARCH_LOCK:
            try:
                return await asyncio.wait_for(self._run_locked(ctx, topic), timeout=110)
            except asyncio.TimeoutError:
                log.warning("[research] 110s budget exceeded for: %s", topic)
                if not ctx.extra.get("research"):
                    ctx.extra["research"] = {
                        "topic": topic, "sources_total": 0, "sources_relevant": 0,
                        "open_source": [], "gaps": [], "build_from_scratch": [],
                        "knowledge_map": {"summary": "Research timed out (110s budget)."},
                        "recommendations": "Research exceeded the time budget. "
                        "Narrow the topic or enable a faster model.",
                        "iterations": 0, "confidence": 0.0,
                    }
                return AgentResult.success(self.name, {"timeout": True, "sources": 0})

    async def _run_locked(self, ctx: AgentContext, topic: str) -> AgentResult:
        state = ResearchState(topic=topic)
        synthesis: dict = {}

        max_iterations = self.depth

        while state.iterations < max_iterations:
            state.iterations += 1
            log.info("[research] iteration %d for: %s", state.iterations, topic)

            # ── Phase 1: parallel collection ──────────────────────────────────
            models_task = asyncio.create_task(_collect_3d_models(topic))
            web_task    = asyncio.create_task(_collect_web(topic))
            github_task = asyncio.create_task(_collect_github(topic))

            models_r, web_r, github_r = await asyncio.gather(
                models_task, web_task, github_task, return_exceptions=True
            )

            raw: list[dict] = []
            if not isinstance(models_r, Exception): raw.extend(models_r)
            if not isinstance(web_r,    Exception): raw.extend(web_r)
            if not isinstance(github_r, Exception): raw.extend(github_r)

            state.all_sources.extend(raw)
            log.info("[research] collected %d raw sources", len(raw))

            # Persist collected sources IMMEDIATELY — a later LLM-phase timeout (110s
            # on slow local model) must NOT discard the real HTTP-collected sources.
            # This is the root fix for "0 sources": the data exists, it was being thrown away.
            ctx.extra["research"] = _pack_research_result(state, {}, None)

            # ── Phase 2: filter (SIMPLE model) ────────────────────────────────
            relevant = await _filter_results(raw, topic)
            state.relevant_sources.extend(relevant)

            # ── Phase 3: synthesis (COMPLEX model) ────────────────────────────
            synthesis = await _synthesize(relevant, topic, state.knowledge_map)
            state.knowledge_map = synthesis
            state.confidence = synthesis.get("confidence", 0.5)

            log.info("[research] synthesis confidence=%.2f needs_more=%s",
                     state.confidence, synthesis.get("needs_more_research"))

            # ── Phase 4: dynamic orchestration ────────────────────────────────
            if state.iterations < max_iterations:
                new_queries = await _decide_next(state, synthesis)
                if not new_queries:
                    log.info("[research] orchestrator decided to stop at iter %d", state.iterations)
                    break

                # Run new queries and add to context for next iteration
                extra_sources: list[dict] = []
                for q_item in new_queries[:3]:
                    query = q_item.get("query", "")
                    source = q_item.get("source", "web")
                    if source == "web":
                        from orynd_core.services.search.web import search
                        r = await search(query, limit=5)
                        extra_sources.extend(r)
                    elif source == "github":
                        from orynd_core.services.search.github_search import search_repos
                        r = await search_repos(query, limit=4)
                        extra_sources.extend(r)
                    elif source == "3d_models":
                        from orynd_core.services.search import printables
                        r = await printables.search(query, limit=4)
                        extra_sources.extend(
                            c.model_dump() if hasattr(c, "model_dump") else c for c in r
                        )

                # Re-run with new topic augmented
                if extra_sources:
                    topic = f"{topic} {new_queries[0].get('query', '')}"
                    state.all_sources.extend(extra_sources)
                else:
                    break
            else:
                break

        # ── Phase 5: recommendations (MEDIUM model) ───────────────────────────
        recommendations = await _build_recommendations(state, synthesis)

        result = _pack_research_result(state, synthesis, recommendations)
        ctx.extra["research"] = result
        # Also populate candidates from 3D model results
        from orynd_core.services.search import printables as p_mod
        ctx.candidates = [
            s for s in state.relevant_sources
            if s.get("type") == "3d_model" and s.get("stl_url")
        ][:5]

        return AgentResult.success(self.name, {
            "sources": result["sources_total"],
            "iterations": state.iterations,
            "confidence": state.confidence,
        })
