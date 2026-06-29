"""
SlicerAgent — STL → G-code via PrusaSlicer CLI.

Phase 2: labeled stub — returns gcode_url = None.
Phase 5: real PrusaSlicer CLI integration.

Input:  ctx.stl_url, ctx.extra["printer"] (default "prusa_mk4"),
        ctx.extra["infill"] (default 20), ctx.extra["layer_height"] (default 0.2)
Output: ctx.gcode_url (str | None)
"""

from __future__ import annotations
import logging
import shutil
import subprocess
import tempfile
import os
from pathlib import Path

from orynd_core.agents.base import AgentContext, AgentResult, BaseAgent

log = logging.getLogger(__name__)

_PROFILES: dict[str, str] = {
    # Prusa
    "prusa_mk4":    "PrusaResearch/0.20mm QUALITY @MK4.ini",
    "prusa_mk3s":   "PrusaResearch/0.20mm QUALITY @MK3S.ini",
    "prusa_mini":   "PrusaResearch/0.20mm QUALITY @MINI.ini",
    "prusa_xl":     "PrusaResearch/0.20mm QUALITY @XL.ini",
    # Bambu Lab
    "bambu_x1c":    "Bambu Lab X1 Carbon 0.4 nozzle.ini",
    "bambu_p1s":    "Bambu Lab P1S 0.4 nozzle.ini",
    "bambu_a1":     "Bambu Lab A1 0.4 nozzle.ini",
    # Creality
    "ender_3":      "Creality Ender-3/0.20mm QUALITY @Ender-3.ini",
    "ender_3_v3":   "Creality Ender-3 V3/0.20mm QUALITY.ini",
    "cr10":         "Creality CR-10/0.20mm QUALITY.ini",
    # Voron
    "voron_2.4":    "Voron/0.20mm QUALITY @V2.4.ini",
    "voron_trident": "Voron/0.20mm QUALITY @Trident.ini",
    # Generic
    "generic":      "0.20mm QUALITY.ini",
}


def _prusaslicer_bin() -> str | None:
    """Find PrusaSlicer CLI on PATH or common Mac location."""
    if shutil.which("prusa-slicer"):
        return "prusa-slicer"
    mac_path = "/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer"
    if Path(mac_path).exists():
        return mac_path
    return None


class SlicerAgent(BaseAgent):
    """
    Converts STL to G-code using PrusaSlicer CLI.
    If PrusaSlicer is not installed, logs a clear warning and returns stub result.
    No LLM required.
    """

    name = "slicer_agent"

    def __init__(self) -> None:
        super().__init__(provider=None)

    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        if not ctx.stl_url:
            return AgentResult.failure(self.name, "No STL URL in context")

        # stub: Phase 5 — real PrusaSlicer CLI integration
        bin_path = _prusaslicer_bin()
        if not bin_path:
            log.warning("[slicer] PrusaSlicer not installed — returning stub (Phase 5)")
            ctx.gcode_url = None
            ctx.extra["slicer_stub"] = True
            return AgentResult.success(
                self.name,
                {
                    "stub": True,
                    "reason": "PrusaSlicer CLI not found. Install PrusaSlicer to enable slicing.",
                    "stl_url": ctx.stl_url,
                },
            )

        printer = ctx.extra.get("printer", "prusa_mk4")
        infill = int(ctx.extra.get("infill", 20))
        layer_h = float(ctx.extra.get("layer_height", 0.2))
        profile = _PROFILES.get(printer, _PROFILES["prusa_mk4"])

        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                stl_bytes = (await client.get(ctx.stl_url)).content
            except Exception as e:
                return AgentResult.failure(self.name, f"Failed to download STL: {e}")

        with tempfile.TemporaryDirectory() as tmp:
            stl_path = Path(tmp) / "model.stl"
            gcode_path = Path(tmp) / "model.gcode"
            stl_path.write_bytes(stl_bytes)

            cmd = [
                bin_path,
                "--export-gcode",
                "--load", profile,
                "--fill-density", f"{infill}%",
                "--layer-height", str(layer_h),
                "--output", str(gcode_path),
                str(stl_path),
            ]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if proc.returncode != 0:
                    return AgentResult.failure(
                        self.name, f"PrusaSlicer error: {proc.stderr[:300]}"
                    )
            except subprocess.TimeoutExpired:
                return AgentResult.failure(self.name, "PrusaSlicer timed out (>120s)")

            # Phase 5: upload gcode_path to Supabase Storage, return public URL
            # For now, return local path as placeholder
            gcode_size = gcode_path.stat().st_size if gcode_path.exists() else 0
            ctx.gcode_url = None  # Phase 5: set to Supabase URL
            ctx.extra["gcode_size_bytes"] = gcode_size
            log.info("[slicer] done, gcode=%d bytes", gcode_size)

            return AgentResult.success(
                self.name,
                {
                    "stub": False,
                    "printer": printer,
                    "infill": infill,
                    "layer_height": layer_h,
                    "gcode_size_bytes": gcode_size,
                    "note": "G-code generated locally. Supabase upload: Phase 5.",
                },
            )
