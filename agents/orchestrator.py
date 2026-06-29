"""
Pipeline — chains agents in sequence.
Each agent gets the same AgentContext (shared mutable state).
If any agent fails and stop_on_error=True, pipeline halts.

Usage:
    pipeline = Pipeline([IntentAgent(provider), RetrievalAgent(), SelectorAgent()])
    results = await pipeline.run(ctx)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

from .base import BaseAgent, AgentContext, AgentResult

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    ok: bool
    results: list[AgentResult] = field(default_factory=list)
    failed_at: str | None = None
    total_ms: int = 0

    @property
    def last(self) -> AgentResult | None:
        return self.results[-1] if self.results else None


class Pipeline:
    """
    Sequential agent pipeline.
    Agents share one AgentContext — each fills in what the next needs.
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        stop_on_error: bool = True,
    ) -> None:
        self.agents = agents
        self.stop_on_error = stop_on_error

    def add(self, agent: BaseAgent) -> "Pipeline":
        """Fluent API: pipeline.add(agent).add(another)"""
        self.agents.append(agent)
        return self

    async def run(self, ctx: AgentContext) -> PipelineResult:
        results: list[AgentResult] = []
        total_ms = 0

        log.info("[Pipeline] start | agents=%s session=%s",
                 [a.name for a in self.agents], ctx.session_id)

        for agent in self.agents:
            result = await agent.run(ctx)
            results.append(result)
            total_ms += result.duration_ms

            if not result.ok and self.stop_on_error:
                log.warning("[Pipeline] stopped at %s: %s", agent.name, result.error)
                return PipelineResult(
                    ok=False,
                    results=results,
                    failed_at=agent.name,
                    total_ms=total_ms,
                )

        log.info("[Pipeline] done | total_ms=%d", total_ms)
        return PipelineResult(ok=True, results=results, total_ms=total_ms)

    async def run_from(self, ctx: AgentContext, agent_name: str) -> PipelineResult:
        """Start pipeline from a specific agent (skip earlier ones)."""
        start = next((i for i, a in enumerate(self.agents) if a.name == agent_name), 0)
        trimmed = Pipeline(self.agents[start:], self.stop_on_error)
        return await trimmed.run(ctx)
