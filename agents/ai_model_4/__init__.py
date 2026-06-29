"""
AI Model 4 — Dual-Pass Mesh Decomposition.

Per founder voice 2026-06-04:
  Pass 1 — rough decomposition (uses existing MeshAnalysisAgent / Pipeline B)
  Engineering Filter — buildable vs noise tagging
  Pass 2 — primitive rebuild (cylinder/box/sphere/torus/sweep/revolve fitting)
  Output — CoreOps JSON describing engineering-clean primitives

Public API:
    from orynd_core.agents.ai_model_4 import DualPassOrchestrator, run_dual_pass
"""
from .orchestrator import DualPassOrchestrator, run_dual_pass, DualPassResult
from .engineering_filter import EngineeringFilter, FilteredPart
from .pass2_rebuild import Pass2Rebuilder, PrimitiveFit, FitResult
from .cad_translator import translate_to_cad_coreops
from .cad_bridge import run_dual_pass_to_cad

__all__ = [
    "DualPassOrchestrator",
    "run_dual_pass",
    "DualPassResult",
    "EngineeringFilter",
    "FilteredPart",
    "Pass2Rebuilder",
    "PrimitiveFit",
    "FitResult",
    "translate_to_cad_coreops",
    "run_dual_pass_to_cad",
]
