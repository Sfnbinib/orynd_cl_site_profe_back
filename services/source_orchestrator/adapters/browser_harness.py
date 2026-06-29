"""
BrowserHarnessAdapter — JS-heavy sites via Playwright (browser-use).

Use case: sites that render content via JavaScript, hide content behind anti-bot,
or require complex navigation that simple HTTP can't do.

This is what founder called "harness" — agent that visits the site like a human.

Two backends:
  - browser-use (LLM-driven agentic browsing) — for complex sites
  - playwright direct (simple JS rendering) — for moderate JS sites

Falls back gracefully if Playwright / browser-use not installed.
"""
from __future__ import annotations
import logging
import time
from typing import Optional
from urllib.parse import urlparse, urljoin

from .base import AdapterBase, SearchHit, AdapterError, AdapterTimeout, AdapterBlocked

log = logging.getLogger(__name__)


class BrowserHarnessAdapter(AdapterBase):
    name = "browser_harness"
    timeout_s = 60  # browser ops are slower

    def __init__(self, use_browser_use: bool = False):
        """
        Args:
            use_browser_use: if True, use LLM-driven browser-use for navigation.
                             Else, simple Playwright page.goto + content extraction.
        """
        self.use_browser_use = use_browser_use

    async def search(self, query: str, source_url: str, limit: int = 10) -> list[SearchHit]:
        # Lazy import — browser deps are heavy
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise AdapterError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        if self.use_browser_use:
            return await self._search_with_browser_use(query, source_url, limit)
        return await self._search_with_playwright(query, source_url, limit)

    async def _search_with_playwright(self, query: str, source_url: str, limit: int) -> list[SearchHit]:
        """Simple JS render + parse."""
        from playwright.async_api import async_playwright
        from bs4 import BeautifulSoup

        start = time.monotonic()

        # Heuristic search URL — same as generic
        search_urls = [
            f"{source_url.rstrip('/')}/search?q={query.replace(' ', '+')}",
            f"{source_url.rstrip('/')}/?s={query.replace(' ', '+')}",
        ]

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (orynd-research) Chromium/120 Safari/537.36",
                )
                page = await context.new_page()

                for search_url in search_urls:
                    try:
                        await page.goto(search_url, wait_until="domcontentloaded", timeout=self.timeout_s * 1000)
                        # Wait for JS to render
                        await page.wait_for_load_state("networkidle", timeout=10_000)
                        html = await page.content()
                        soup = BeautifulSoup(html, "html.parser")
                        hits = self._extract_hits(soup, page.url, query, source_url, limit)

                        latency = int((time.monotonic() - start) * 1000)
                        for h in hits:
                            h.latency_ms = latency
                            h.adapter_used = self.name

                        if hits:
                            return hits[:limit]

                    except Exception as e:
                        log.debug("[browser_harness] %s failed: %s", search_url, e)
                        continue

                return []
            finally:
                await browser.close()

    async def _search_with_browser_use(self, query: str, source_url: str, limit: int) -> list[SearchHit]:
        """LLM-driven agentic browsing. Used for complex sites."""
        try:
            from browser_use import Agent, BrowserConfig
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise AdapterError(
                "browser-use not installed. Run: pip install browser-use langchain-anthropic"
            )

        # Use a low-cost LLM for the agent
        llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

        task = (
            f"Visit {source_url}, find search functionality, search for '{query}', "
            f"return top {limit} results as JSON list with fields: title, url, snippet, thumbnail."
        )

        agent = Agent(task=task, llm=llm)

        start = time.monotonic()
        try:
            result = await agent.run()
            # Parse agent's structured output
            return self._parse_agent_result(result, query, source_url, start, limit)
        except Exception as e:
            log.warning("[browser_use] agentic search failed: %s", e)
            raise AdapterError(str(e))

    def _extract_hits(self, soup, page_url, query: str, source_url: str, limit: int) -> list[SearchHit]:
        """Same extraction as GenericHTMLAdapter — reuse logic."""
        from .generic_html import GenericHTMLAdapter
        helper = GenericHTMLAdapter()
        return helper._parse_html(str(soup), page_url, query, source_url, limit)

    def _parse_agent_result(self, agent_result, query: str, source_url: str, start: float, limit: int) -> list[SearchHit]:
        """Parse browser-use agent output into SearchHits."""
        # browser-use returns various structures; we attempt to extract
        hits: list[SearchHit] = []
        domain = urlparse(source_url).netloc

        try:
            # Many possible result formats; try common ones
            if hasattr(agent_result, "final_result"):
                payload = agent_result.final_result
            else:
                payload = str(agent_result)

            import json
            if isinstance(payload, str):
                # Try parse JSON
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = []

            if isinstance(payload, list):
                latency = int((time.monotonic() - start) * 1000)
                for item in payload[:limit]:
                    if not isinstance(item, dict):
                        continue
                    hits.append(SearchHit(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("snippet", ""),
                        source_id=domain,
                        source_name=domain,
                        thumbnail_url=item.get("thumbnail"),
                        score=0.7,
                        adapter_used=f"{self.name}:browser_use",
                        latency_ms=latency,
                    ))
        except Exception as e:
            log.warning("Failed to parse browser-use result: %s", e)

        return hits
