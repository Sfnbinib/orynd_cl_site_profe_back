"""
Printables.com — public GraphQL search (no API key required).
Returns up to N Candidate objects ranked by best_match.

Adapted from deploy_teleg/api/services/printables.py.
Original not modified.
"""

from __future__ import annotations
import posixpath
import httpx
from orynd_core.models.schemas import Candidate

GRAPHQL_URL = "https://api.printables.com/graphql/"
IMAGE_BASE  = "https://media.printables.com/"
MODEL_BASE  = "https://www.printables.com/model/"

SEARCH_QUERY = """
query SearchPrints($query: String!, $limit: Int) {
  searchPrints2(query: $query, limit: $limit, ordering: best_match) {
    items {
      id
      name
      summary
      likesCount
      image { filePath }
      user { publicUsername }
      stls { id name fileSize filePreviewPath }
    }
  }
}
"""


def _printability(likes: int) -> int:
    """Map likes count to a 1-10 printability score (proxy metric)."""
    if likes >= 500: return 10
    if likes >= 200: return 9
    if likes >= 100: return 8
    if likes >= 50:  return 7
    if likes >= 20:  return 6
    if likes >= 10:  return 5
    return 4


def _score(index: int, likes: int) -> float:
    """Relevance score: best_match order wins, likes break ties."""
    base  = 1.0 - index * 0.08        # 1.00 / 0.92 / 0.84 / 0.76 / 0.68
    boost = min(likes / 1000, 0.05)   # up to +0.05 for popular models
    return round(min(base + boost, 1.0), 2)


async def search(query: str, limit: int = 5) -> list[Candidate]:
    """
    Search Printables.com via their public GraphQL API.
    Returns empty list on any error — caller handles fallback.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(
                GRAPHQL_URL,
                json={
                    "query": SEARCH_QUERY,
                    "variables": {"query": query, "limit": limit},
                },
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

    if "errors" in data:
        return []

    items = data.get("data", {}).get("searchPrints2", {}).get("items", [])

    candidates: list[Candidate] = []
    for i, item in enumerate(items[:limit]):
        model_id   = item.get("id", "")
        name       = item.get("name", "Unnamed model")
        summary    = item.get("summary") or "3D printable model from Printables.com"
        likes      = item.get("likesCount", 0)
        image_path = (item.get("image") or {}).get("filePath", "")

        # Thumbnail via Printables CDN
        if image_path:
            img_dir  = posixpath.dirname(image_path)
            img_file = posixpath.basename(image_path)
            preview_url = f"{IMAGE_BASE}{img_dir}/thumbs/inside/1280x960/jpg/{img_file}"
        else:
            preview_url = "https://via.placeholder.com/300x200/0a0a0a/888888?text=ORYND"

        # STL URL from filePreviewPath pattern
        stls    = item.get("stls") or []
        stl_url = f"{MODEL_BASE}{model_id}"  # fallback: model page
        if stls:
            first_stl  = stls[0]
            stl_preview = first_stl.get("filePreviewPath") or ""
            if stl_preview and "_preview.png" in stl_preview:
                stl_url = f"{IMAGE_BASE}{stl_preview.replace('_preview.png', '.stl')}"

        candidates.append(Candidate(
            id=f"p-{model_id}",
            name=name,
            description=summary[:200],
            preview_url=preview_url,
            stl_url=stl_url,
            source="printables",
            source_url=f"{MODEL_BASE}{model_id}",
            score=_score(i, likes),
            printability=_printability(likes),
        ))

    return candidates
