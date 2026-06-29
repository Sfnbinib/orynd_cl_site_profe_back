"""
SelectorAgent — picks a candidate by index and verifies the STL URL.

Input:  ctx.candidates, ctx.extra["select_index"] (default 0)
Output: ctx.selected (dict), ctx.stl_url (str)
"""

from __future__ import annotations
import httpx
import logging

from orynd_core.agents.base import AgentContext, AgentResult, BaseAgent

log = logging.getLogger(__name__)

_TIMEOUT = 8


async def _verify_stl(url: str) -> bool:
    if not url:
        return False
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = await client.head(url, headers={"User-Agent": "ORYND/0.1"})
            return resp.status_code == 200
    except Exception:
        return False


class SelectorAgent(BaseAgent):
    """
    Selects a candidate from ctx.candidates by index.
    Verifies the STL URL with a HEAD request.
    Falls back to source_url if STL not reachable.
    No LLM required.
    """

    name = "selector_agent"

    def __init__(self) -> None:
        super().__init__(provider=None)

    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        if not ctx.candidates:
            return AgentResult.failure(self.name, "No candidates to select from")

        index = int(ctx.extra.get("select_index", 0))
        if index >= len(ctx.candidates):
            index = 0

        candidate = ctx.candidates[index]
        stl_url = candidate.get("stl_url", "")
        source_url = candidate.get("source_url", "")

        verified = await _verify_stl(stl_url)
        final_url = stl_url if verified else source_url

        ctx.selected = candidate
        ctx.stl_url = final_url

        log.info(
            "[selector] picked index=%d name=%s verified=%s",
            index, candidate.get("name", "?"), verified,
        )

        return AgentResult.success(
            self.name,
            {
                "index": index,
                "name": candidate.get("name"),
                "stl_url": final_url,
                "verified": verified,
            },
        )
