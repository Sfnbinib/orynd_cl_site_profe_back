"""
Thingiverse search — real OAuth2 REST API.
Reads THINGIVERSE_TOKEN from env. Returns [] gracefully if token missing.
"""

from __future__ import annotations
import os
import httpx

from orynd_core.models.schemas import Candidate

_BASE = "https://api.thingiverse.com"
_TIMEOUT = 12


def _score(index: int, like_count: int) -> float:
    base = max(0.0, 0.9 - index * 0.08)
    boost = min(0.1, like_count / 10_000)
    return round(base + boost, 3)


def _printability(like_count: int) -> int:
    if like_count >= 5000:
        return 9
    if like_count >= 1000:
        return 8
    if like_count >= 200:
        return 7
    if like_count >= 50:
        return 6
    return 5


async def search(query: str, limit: int = 5) -> list[Candidate]:
    token = os.getenv("THINGIVERSE_TOKEN", "")
    if not token:
        return []

    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            r = await client.get(
                f"{_BASE}/search/{query}",
                headers=headers,
                params={"per_page": limit, "sort": "popular", "type": "things"},
            )
            r.raise_for_status()
            hits = r.json().get("hits", [])
        except Exception:
            return []

        candidates: list[Candidate] = []

        for i, h in enumerate(hits[:limit]):
            thing_id = h.get("id")
            if not thing_id:
                continue

            stl_url = ""
            try:
                fr = await client.get(
                    f"{_BASE}/things/{thing_id}/files",
                    headers=headers,
                )
                files = fr.json() if fr.status_code == 200 else []
                stl_file = next(
                    (f for f in files if isinstance(f, dict) and f.get("name", "").lower().endswith(".stl")),
                    None,
                )
                if stl_file:
                    stl_url = stl_file.get("download_url") or stl_file.get("public_url", "")
            except Exception:
                pass

            if not stl_url:
                continue

            likes = h.get("like_count", 0)
            candidates.append(
                Candidate(
                    id=f"tv_{thing_id}",
                    name=h.get("name", "Untitled"),
                    description=h.get("description", "")[:200],
                    preview_url=h.get("thumbnail", ""),
                    stl_url=stl_url,
                    source="thingiverse",
                    source_url=f"https://www.thingiverse.com/thing:{thing_id}",
                    score=_score(i, likes),
                    printability=_printability(likes),
                )
            )

    return candidates
