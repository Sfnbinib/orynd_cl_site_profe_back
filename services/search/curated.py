"""
Curated Index search — stub (Phase 5, requires Supabase).
Tier 0: fastest source (<500ms) once populated by Data Flywheel.
"""

from __future__ import annotations
from orynd_core.models.schemas import Candidate


async def search(query: str, limit: int = 5) -> list[Candidate]:  # noqa: ARG001
    # stub: Phase 5 — Supabase RPC search_curated_models
    return []
