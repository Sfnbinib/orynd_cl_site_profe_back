"""
DuckDuckGoFallbackAdapter — last-resort web search restricted to a specific domain.

When other adapters fail or no info found, use `site:example.com query`
via DuckDuckGo Instant Answer / HTML scraping.
"""
from __future__ import annotations
import logging
import time
from urllib.parse import urlencode, urlparse, urljoin

from .base import AdapterBase, SearchHit, AdapterError, AdapterBlocked

log = logging.getLogger(__name__)


class DuckDuckGoFallbackAdapter(AdapterBase):
    name = "duckduckgo_fallback"
    timeout_s = 10

    async def search(self, query: str, source_url: str, limit: int = 10) -> list[SearchHit]:
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise AdapterError(f"missing dep: {e}")

        domain = urlparse(source_url).netloc
        site_query = f"site:{domain} {query}"

        # Use DuckDuckGo HTML endpoint
        ddg_url = f"https://html.duckduckgo.com/html/?{urlencode({'q': site_query})}"
        start = time.monotonic()

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (orynd-research)"},
            ) as client:
                response = await client.get(ddg_url)
            if response.status_code == 429:
                raise AdapterBlocked("DDG rate limited")
            if response.status_code >= 400:
                raise AdapterError(f"DDG HTTP {response.status_code}")

            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.select(".result__title a, .result a.result__a")
            hits: list[SearchHit] = []
            latency = int((time.monotonic() - start) * 1000)

            for r in results[:limit]:
                href = r.get("href", "")
                title = r.get_text(strip=True)
                # DDG wraps hrefs in /l/?uddg= — extract
                if "uddg=" in href:
                    from urllib.parse import parse_qs, urlparse as up
                    qs = parse_qs(up(href).query)
                    if "uddg" in qs:
                        href = qs["uddg"][0]

                hits.append(SearchHit(
                    title=title,
                    url=href,
                    source_id=domain,
                    source_name=domain,
                    score=0.5,
                    adapter_used=self.name,
                    latency_ms=latency,
                ))
            return hits

        except httpx.TimeoutException as e:
            raise AdapterError(f"DDG timeout: {e}")
