"""
ChatAgent — the main brain of ORYND Desktop.

Receives a user message + session history.
Decides what to do:
  - "search"     → run RetrievalAgent, return candidates
  - "answer"     → answer directly (engineering knowledge)
  - "clarify"    → ask a clarifying question
  - "fabricate"  → fabrication advice for selected model
  - "ideas"      → suggest what to print / explore

Works with LLM (Claude) if key available.
Falls back to keyword routing algorithm if no key.

Input:  ctx.raw_text, ctx.extra["history"] (loaded by MemoryAgent)
Output: ctx.extra["chat_action"]   — what to do next
        ctx.extra["chat_response"] — text to show user (if action=answer/clarify/ideas)
        ctx.intent                 — filled for search routing
"""

from __future__ import annotations
import json
import logging
import re

from orynd_core.agents.base import AgentContext, AgentResult, BaseAgent
from orynd_core.services.llm.base import LLMMessage, LLMProvider

log = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are ORYND — an AI engineering workspace for 3D printing and fabrication.
You help engineers and makers find, create, and manufacture 3D models.

Your job is to understand what the user wants and decide the best action.

Return ONLY valid JSON, no markdown, no explanation.

JSON schema:
{
  "action": "search" | "answer" | "clarify" | "fabricate" | "ideas",
  "response": "your message to the user (conversational, concise)",
  "search_query": "search terms if action=search, else null",
  "confidence": 0.0-1.0
}

Action rules:
- "search": user wants to find a 3D model. Extract clean search terms for search_query.
- "answer": user asks a question about 3D printing, materials, settings, engineering.
  Answer directly and helpfully. response = your answer (2-4 sentences max).
- "clarify": user request is too vague, you need more info.
  response = a single specific question.
- "fabricate": user asks about print settings, material, orientation for the model they selected.
  response = practical fabrication advice.
- "ideas": user wants inspiration, doesn't know what to print.
  response = 3 concrete ideas with brief explanations.

Tone: direct, technical, no fluff. Like a senior engineer colleague, not a customer service bot.
Language: match user's language (respond in Russian if user writes in Russian).
"""

# ── Algorithm routing (no LLM) ────────────────────────────────────────────────

_SEARCH_SIGNALS = re.compile(
    r"\b(find|search|look|get|want|need|print|holder|mount|bracket|case|stand|"
    r"найди|ищи|хочу|нужен|нужна|найти|распечатать|скачай)\b",
    re.IGNORECASE,
)
_BUILD_SIGNALS = re.compile(
    r"\b(create|build|make|design|generate|draw|model|создай|сделай|построй|"
    r"нарисуй|сгенерируй|смоделируй)\b",
    re.IGNORECASE,
)
_QUESTION_SIGNALS = re.compile(
    r"\b(how|what|why|which|when|where|should|can|does|is|are|"
    r"как|что|почему|какой|какая|зачем|можно|нужно)\b",
    re.IGNORECASE,
)
_FABRICATION_SIGNALS = re.compile(
    r"\b(material|infill|layer|support|temperature|speed|settings|profile|"
    r"материал|заполнение|слой|поддержки|температура|настройки|профиль)\b",
    re.IGNORECASE,
)


def _algorithm_route(text: str) -> dict:
    if _FABRICATION_SIGNALS.search(text):
        return {
            "action": "fabricate",
            "response": "Based on typical settings: use PETG for mechanical parts (infill 20-40%, 3 walls, 0.2mm layers). PLA for visual/decorative. Add supports if overhangs >45°.",
            "search_query": None,
            "confidence": 0.5,
            "_source": "algorithm",
        }
    if _QUESTION_SIGNALS.search(text) and not _SEARCH_SIGNALS.search(text):
        return {
            "action": "answer",
            "response": "I can help with that. For detailed advice, connect your API key in settings for full AI responses.",
            "search_query": None,
            "confidence": 0.4,
            "_source": "algorithm",
        }
    if _BUILD_SIGNALS.search(text) and not _SEARCH_SIGNALS.search(text):
        return {
            "action": "clarify",
            "response": "I see this as a build request, but I need concrete geometry or a supported template. Give dimensions, or use a known template like gear/brake disc/box/cylinder.",
            "search_query": None,
            "confidence": 0.55,
            "_source": "algorithm",
        }
    # Default: treat as search
    words = re.findall(r"\b\w{3,}\b", text.lower())
    stopwords = {"the", "and", "for", "with", "this", "that", "want", "need",
                 "make", "print", "find", "get", "can", "you", "мне", "это"}
    keywords = [w for w in words if w not in stopwords][:5]
    return {
        "action": "search",
        "response": None,
        "search_query": " ".join(keywords) or text,
        "confidence": 0.6,
        "_source": "algorithm",
    }


# ── Agent ─────────────────────────────────────────────────────────────────────

class ChatAgent(BaseAgent):
    """
    Main routing brain. Decides action, fills ctx.extra["chat_action"].
    Does NOT run search — that's RetrievalAgent's job.
    This agent only decides and optionally generates a text response.
    """

    name = "chat_agent"

    def __init__(self, provider: LLMProvider | None = None) -> None:
        super().__init__(provider=provider)

    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        text = ctx.raw_text or ""
        if not text:
            ctx.extra["chat_action"] = "idle"
            return AgentResult.success(self.name, {"action": "idle"})

        history = ctx.extra.get("history", [])

        if self.provider:
            routing = await self._llm_route(text, history)
        else:
            routing = _algorithm_route(text)

        action = routing.get("action", "search")
        ctx.extra["chat_action"] = action
        ctx.extra["chat_response"] = routing.get("response")
        ctx.extra["chat_routing"] = routing

        # If search — populate intent for RetrievalAgent
        if action == "search":
            query = routing.get("search_query") or text
            words = re.findall(r"\b\w{3,}\b", query.lower())
            ctx.intent = {
                "object_name": " ".join(words[:2]) if words else query,
                "keywords": words[:5],
                "confidence": routing.get("confidence", 0.6),
                "_source": "chat_agent",
            }
            # Also update raw_text to the clean query
            ctx.extra["search_query"] = query

        log.info("[chat] action=%s confidence=%.2f source=%s",
                 action, routing.get("confidence", 0), routing.get("_source", "?"))

        return AgentResult.success(self.name, {
            "action": action,
            "has_response": bool(routing.get("response")),
        })

    async def _llm_route(self, text: str, history: list[dict]) -> dict:
        # Build conversation context (last 5 turns)
        messages: list[LLMMessage] = []
        for turn in history[-5:]:
            if turn.get("query"):
                messages.append(LLMMessage(role="user", content=turn["query"]))
            if turn.get("chat_response"):
                messages.append(LLMMessage(role="assistant", content=turn["chat_response"]))
        messages.append(LLMMessage(role="user", content=text))

        try:
            raw = await self.provider.complete_json(
                messages, system=_SYSTEM, max_tokens=512
            )
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
            result = json.loads(cleaned)
            result["_source"] = "llm"
            return result
        except Exception as e:
            log.warning("[chat] LLM routing failed (%s), algorithm fallback", e)
            fallback = _algorithm_route(text)
            fallback["_llm_error"] = str(e)
            return fallback
