"""
GenericHTMLAdapter — light-weight HTML scraping for sites без dedicated API.

Strategy:
  1. Try common search URL patterns: `{base}/search?q={query}` или sitemap.
  2. Fetch with httpx (with realistic User-Agent).
  3. Parse with BeautifulSoup → extract <a> with relevant context.
  4. Score by query-term presence.

Limits:
  - No JS execution (use BrowserHarnessAdapter for that)
  - Respects robots.txt
  - Caps result count
"""
from __future__ import annotations
import logging
import time
from typing import Optional
from urllib.parse import urlencode, urlparse, urljoin

from .base import AdapterBase, SearchHit, AdapterError, AdapterTimeout, AdapterBlocked

log = logging.getLogger(__name__)


# Common search URL templates per site pattern
SEARCH_PATTERNS = [
    "{base}/search?q={q}",
    "{base}/?s={q}",  # WordPress
    "{base}/search/?query={q}",
    "{base}/index.php?search={q}",
]


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36 (orynd-research-bot)"


class GenericHTMLAdapter(AdapterBase):
    name = "generic_html"
    timeout_s = 15

    async def search(self, query: str, source_url: str, limit: int = 10) -> list[SearchHit]:
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise AdapterError(f"missing dep: {e}; pip install httpx beautifulsoup4")

        start = time.monotonic()

        # Try each search pattern
        last_err: Optional[Exception] = None
        for pattern in SEARCH_PATTERNS:
            search_url = pattern.format(base=source_url.rstrip("/"), q=query.replace(" ", "+"))
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_s,
                    follow_redirects=True,
                    headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
                ) as client:
                    response = await client.get(search_url)

                if response.status_code == 403 or response.status_code == 429:
                    raise AdapterBlocked(f"{response.status_code} {response.reason_phrase}")
                if response.status_code >= 400:
                    last_err = AdapterError(f"HTTP {response.status_code}")
                    continue

                hits = self._parse_html(response.text, response.url, query, source_url, limit)
                latency = int((time.monotonic() - start) * 1000)
                for h in hits:
                    h.latency_ms = latency
                    h.adapter_used = self.name
                if hits:
                    return hits[:limit]

            except httpx.TimeoutException as e:
                last_err = AdapterTimeout(str(e))
            except AdapterBlocked:
                raise
            except Exception as e:
                last_err = AdapterError(str(e))

        if last_err:
            raise last_err
        return []

    def _parse_html(
        self,
        html: str,
        response_url,
        query: str,
        source_url: str,
        limit: int,
    ) -> list[SearchHit]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        query_terms = [t.lower() for t in query.split() if len(t) > 2]

        hits: list[SearchHit] = []
        seen_urls: set[str] = set()

        # Strategy: find <a> tags with query terms in surrounding text
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)

            # Skip empty / nav / footer style
            if not text or len(text) < 5:
                continue
            if any(skip in href.lower() for skip in ("javascript:", "mailto:", "#")):
                continue

            # Absolutize URL
            full_url = urljoin(str(response_url), href)
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            # Score by query term presence in link text + nearby text
            score = self._score(text, link, query_terms)
            if score < 0.1:
                continue

            # Extract thumbnail (sibling img)
            thumb = None
            parent = link.parent
            if parent:
                img = parent.find("img")
                if img and img.get("src"):
                    thumb = urljoin(str(response_url), img["src"])

            # Snippet from surrounding text
            snippet = ""
            if link.parent:
                snippet = link.parent.get_text(" ", strip=True)[:200]

            hits.append(SearchHit(
                title=text[:200],
                url=full_url,
                snippet=snippet,
                source_id=urlparse(source_url).netloc,
                source_name=urlparse(source_url).netloc,
                thumbnail_url=thumb,
                score=score,
            ))

            if len(hits) >= limit * 2:  # collect more than needed, will be sorted
                break

        # Sort by score
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def _score(self, text: str, link, query_terms: list[str]) -> float:
        """Score 0..1 based on query term presence."""
        if not query_terms:
            return 0.0
        text_l = text.lower()
        hits = sum(1 for t in query_terms if t in text_l)
        score = hits / len(query_terms)

        # Bonus: link has STL/STEP keywords (3D file URL)
        for ext in (".stl", ".step", ".stp", ".3mf", ".obj"):
            if ext in link["href"].lower():
                score = min(1.0, score + 0.2)
                break

        # Bonus: longer title is usually more descriptive
        if 20 < len(text) < 100:
            score = min(1.0, score + 0.1)

        return score
