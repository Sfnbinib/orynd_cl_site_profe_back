"""
ORYND Core — unified Pydantic schemas.
Single source of truth for all data models.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pydantic import BaseModel


# ── Search ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    session_id: str = "anonymous"
    user_id: str = "anonymous"
    image_b64: str | None = None     # base64 image for vision search
    caption: str | None = None       # optional caption with image
    platform: str = "desktop"        # desktop | telegram | web | api


class Candidate(BaseModel):
    id: str
    name: str
    description: str
    preview_url: str
    stl_url: str
    source: str                   # "printables" | "thingiverse" | ...
    source_url: str
    score: float = 0.5            # relevance score 0..1
    printability: int = 5         # 1..10, based on likes/community validation


class Intent(BaseModel):
    raw: str                      # original user input
    keywords: str                 # cleaned English search terms
    action: str = "search"        # "search" | "clarify" | "modify_params" | "change_mind"
    printer: str | None = None    # "prusa_mk4" | "bambu_x1c" | ...
    infill_pct: int | None = None # 10 | 25 | 50 | 75 | 100


class SearchResponse(BaseModel):
    session_id: str
    candidates: list[Candidate]
    intent: Intent
    action: str = "show"          # "show" | "clarify" | "ideas"
    clarify_question: str | None = None
    ideas: list[str] = []
    extra: dict = {}              # agent metadata (intent_parsed, llm_active, etc.)


# ── Select ────────────────────────────────────────────────────────────────────

class SelectRequest(BaseModel):
    session_id: str
    index: int


class SelectResponse(BaseModel):
    url: str
    name: str
    source: str
    source_url: str
    verified: bool = False


# ── Mesh Pipeline (AI Model 4 — Pipeline B) ─────────────────────────────────

class MeshAnalysisRequest(BaseModel):
    """Request to analyze a mesh file via Pipeline B."""
    session_id: str = "anonymous"
    mesh_path: str | None = None          # local path to STL/OBJ/PLY
    mesh_url: str | None = None           # URL to download mesh from
    mesh_format: str = "stl"              # file format hint
    scale: float = 1.0                    # coordinate scale (1.0 = mm, 25.4 = inch→mm)
    decompose_angle: float = 15.0         # region growing threshold degrees
    decompose_min_faces: int = 5          # min faces per region
    enrich_with_llm: bool = False         # use LLM for natural language description


class MeshFeatureOut(BaseModel):
    """A single manufacturing feature from mesh analysis."""
    feature_id: str
    feature_type: str                     # flat_face, hole, pocket, boss, fillet, etc.
    surface_type: str                     # planar, cylindrical, spherical, freeform
    region_ids: list[int]
    position: list[float]
    direction: list[float]
    dimensions_mm: dict = {}
    area_mm2: float = 0.0
    confidence: float = 0.0
    is_through: bool = False
    is_blind: bool = False
    depth_mm: float = 0.0


class MeshAnalysisResponse(BaseModel):
    """Response from mesh analysis pipeline."""
    session_id: str
    pipeline: str = "mesh"                # "mesh" (Pipeline B) vs "drawing" (Pipeline A)
    model_version: str = "4.0"

    # Mesh info
    mesh_info: dict = {}                  # vertices, triangles, bbox, watertight, volume

    # Decomposition
    regions_count: int = 0
    regions: list[dict] = []

    # Features
    features_count: int = 0
    features: list[MeshFeatureOut] = []
    feature_summary: dict = {}            # {"hole": 3, "flat_face": 8, ...}

    # CoreOps output
    coreops_json: dict = {}               # full CoreOps-compatible schema

    # Optional LLM enrichment
    llm_description: str | None = None

    # Stats
    decomposition_stats: dict = {}


# ── Agent Context (sidecar, not Pydantic — travels through agent chain) ───────

@dataclass
class AgentContext:
    user_id: str = "anonymous"
    session_id: str = "anonymous"
    printer: str = "prusa_mk4"
    infill: int = 20
    history: list[str] = field(default_factory=list)      # last 5 queries
    candidates: list[dict] = field(default_factory=list)  # current candidates
    intent: dict = field(default_factory=dict)             # last Intent
