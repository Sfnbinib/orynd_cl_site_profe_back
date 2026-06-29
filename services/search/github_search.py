"""
GitHub search — repos, code, topics.
No auth required for public search (60 req/hr unauthenticated).
GITHUB_TOKEN in env → 5000 req/hr.
"""
from __future__ import annotations
import logging
import os

import httpx

log = logging.getLogger(__name__)

_BASE = "https://api.github.com"
_TIMEOUT = 10


def _headers() -> dict:
    token = os.getenv("GITHUB_TOKEN", "")
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ORYND/0.1",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


async def search_repos(query: str, limit: int = 6) -> list[dict]:
    """Search GitHub repositories by topic/description."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(
                f"{_BASE}/search/repositories",
                headers=_headers(),
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": limit,
                },
            )
            if r.status_code != 200:
                log.warning("[github] repos search %d", r.status_code)
                return []

            items = r.json().get("items", [])
            return [
                {
                    "title": i.get("full_name", ""),
                    "url": i.get("html_url", ""),
                    "description": (i.get("description") or "")[:150],
                    "stars": i.get("stargazers_count", 0),
                    "language": i.get("language", ""),
                    "topics": i.get("topics", [])[:5],
                    "type": "github_repo",
                }
                for i in items
            ]
    except Exception as e:
        log.warning("[github] repos failed: %s", e)
        return []


async def search_code(query: str, limit: int = 5) -> list[dict]:
    """Search GitHub code files — useful for finding STL generators, CAD scripts."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(
                f"{_BASE}/search/code",
                headers=_headers(),
                params={"q": query, "per_page": limit},
            )
            if r.status_code != 200:
                return []

            items = r.json().get("items", [])
            return [
                {
                    "title": i.get("name", ""),
                    "url": i.get("html_url", ""),
                    "repo": i.get("repository", {}).get("full_name", ""),
                    "path": i.get("path", ""),
                    "type": "github_code",
                }
                for i in items
            ]
    except Exception as e:
        log.warning("[github] code search failed: %s", e)
        return []
