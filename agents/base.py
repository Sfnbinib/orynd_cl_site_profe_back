"""
BaseAgent — foundation for all ORYND agents.

Every agent:
  - receives AgentContext (shared state across pipeline)
  - produces AgentResult (passed to next agent)
  - has access to LLMProvider (swappable)
  - can register tools (algorithms, API calls, etc.)

Pipeline flow:
  AgentContext → Agent1 → Agent2 → Agent3 → final AgentResult
"""
from __future__ import annotations
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from orynd_core.services.llm.base import LLMProvider

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Shared context — flows through the pipeline
# ─────────────────────────────────────────────

@dataclass
class AgentContext:
    """
    Mutable shared state passed between agents in a pipeline.
    Each agent reads what it needs and writes its output here.
    """
    session_id: str
    user_id: str | None = None

    # Input
    raw_text: str | None = None
    image_b64: str | None = None
    image_caption: str | None = None

    # Parsed intent (filled by IntentAgent)
    intent: dict = field(default_factory=dict)

    # Search candidates (filled by RetrievalAgent)
    candidates: list[dict] = field(default_factory=list)

    # Selected model (filled by SelectorAgent)
    selected: dict | None = None
    stl_url: str | None = None
    gcode_url: str | None = None

    # Pipeline metadata
    project_id: str | None = None
    platform: str = "desktop"  # desktop | telegram | web | api

    # Arbitrary extra data for custom agents
    extra: dict = field(default_factory=dict)

    def has_image(self) -> bool:
        return bool(self.image_b64)

    def has_text(self) -> bool:
        return bool(self.raw_text)


# ─────────────────────────────────────────────
# Agent result
# ─────────────────────────────────────────────

@dataclass
class AgentResult:
    ok: bool
    agent_name: str
    data: dict = field(default_factory=dict)
    error: str | None = None
    duration_ms: int = 0

    @classmethod
    def success(cls, agent_name: str, data: dict, duration_ms: int = 0) -> "AgentResult":
        return cls(ok=True, agent_name=agent_name, data=data, duration_ms=duration_ms)

    @classmethod
    def failure(cls, agent_name: str, error: str, duration_ms: int = 0) -> "AgentResult":
        return cls(ok=False, agent_name=agent_name, error=error, duration_ms=duration_ms)


# ─────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────

Tool = Callable[[AgentContext, dict], Awaitable[dict]]


class ToolRegistry:
    """
    Agents can call registered tools by name.
    Tools are async functions: (context, params) → result dict.
    """
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, fn: Tool) -> None:
        self._tools[name] = fn

    async def call(self, name: str, ctx: AgentContext, params: dict = {}) -> dict:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return await self._tools[name](ctx, params)

    def list(self) -> list[str]:
        return list(self._tools.keys())


# ─────────────────────────────────────────────
# Base agent
# ─────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Base class for all ORYND agents.

    Subclass and implement run_logic():
        async def run_logic(self, ctx: AgentContext) -> AgentResult:
            ...

    Swap the LLM provider at init — agents don't care which model runs.
    """

    name: str = "base_agent"

    def __init__(
        self,
        provider: LLMProvider | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.provider = provider          # None = algorithm-only agent (no LLM)
        self.tools = tools or ToolRegistry()

    @abstractmethod
    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        """Agent-specific logic. Must return AgentResult."""
        ...

    async def run(self, ctx: AgentContext) -> AgentResult:
        """Wrapper: timing, logging, error capture."""
        t0 = time.monotonic()
        log.info("[%s] start session=%s", self.name, ctx.session_id)
        try:
            result = await self.run_logic(ctx)
            result.duration_ms = int((time.monotonic() - t0) * 1000)
            log.info("[%s] done ok=%s ms=%d", self.name, result.ok, result.duration_ms)
            return result
        except Exception as exc:
            ms = int((time.monotonic() - t0) * 1000)
            log.exception("[%s] error: %s", self.name, exc)
            return AgentResult.failure(self.name, str(exc), ms)
