"""
Web search via DuckDuckGo (no API key required).
Returns list of {title, url, snippet} dicts.
Used by DeepResearchAgent for articles, docs, references.
"""
from __future__ import annotations
import asyncio
import logging
import re

import httpx

log = logging.getLogger(__name__)

_TIMEOUT = 10
_DDG_URL = "https://html.duckduckgo.com/html/"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def _parse_ddg(html: str, limit: int) -> list[dict]:
    results = []
    # Extract result blocks
    blocks = re.findall(
        r'class="result__title".*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(.*?)</div>',
        html, re.DOTALL
    )
    for url, title_html, snippet_html in blocks[:limit]:
        title = re.sub(r'<[^>]+>', '', title_html).strip()
        snippet = re.sub(r'<[^>]+>', '', snippet_html).strip()
        if title and url.startswith('http'):
            results.append({"title": title, "url": url, "snippet": snippet[:200]})
    return results


async def search(query: str, limit: int = 8) -> list[dict]:
    """Search DuckDuckGo. Returns [] on failure."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            r = await client.post(
                _DDG_URL,
                data={"q": query, "b": "", "kl": "us-en"},
                headers=_HEADERS,
            )
            if r.status_code != 200:
                return []
            return _parse_ddg(r.text, limit)
    except Exception as e:
        log.warning("[web_search] failed: %s", e)
        return []


async def search_multi(queries: list[str], limit_each: int = 5) -> list[dict]:
    """Run multiple queries in parallel, merge results, deduplicate."""
    tasks = [search(q, limit_each) for q in queries]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_urls: set[str] = set()
    merged = []
    for batch in all_results:
        if isinstance(batch, Exception):
            continue
        for item in batch:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                merged.append(item)
    return merged
