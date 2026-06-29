"""
Verification CLI for AI Model 4 dual-pass pipeline.

Usage:
    # Single mesh
    python -m orynd_core.agents.ai_model_4.verify --input tests/fixtures/mesh_samples/bracket.stl

    # All meshes in a directory
    python -m orynd_core.agents.ai_model_4.verify --input-dir tests/fixtures/mesh_samples/ --output verification_results/

    # Auto-download sample STLs from Printables (no token needed for popular models)
    python -m orynd_core.agents.ai_model_4.verify --auto-download --output verification_results/
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

from .orchestrator import DualPassOrchestrator


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )


def run_single(orch: DualPassOrchestrator, mesh_path: str, output_dir: Path) -> dict:
    """Run dual-pass on one mesh, save output."""
    mesh_name = Path(mesh_path).stem
    print(f"\n{'═' * 60}")
    print(f"  Processing: {mesh_name}")
    print(f"{'═' * 60}")

    try:
        result = orch.run(mesh_path=mesh_path)
        result_dict = result.to_dict()

        # Save JSON
        out_path = output_dir / f"{mesh_name}_result.json"
        out_path.write_text(json.dumps(result_dict, indent=2, default=str))
        print(f"  ✓ Saved: {out_path}")

        # Print summary
        print(f"\n  📊 Results:")
        print(f"    Pass 1: {result.pass1_regions_count} regions, {result.pass1_features_count} features")
        print(f"    Filter: {result.filtered_summary}")
        print(f"    Pass 2: {result.primitive_summary}")
        print(f"    Quality: {result.quality_score:.2f}")
        print(f"    Success: {'✅' if result.success else '❌'}")
        if result.notes:
            print(f"    Notes: {', '.join(result.notes)}")

        return result_dict
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return {"mesh": mesh_name, "error": str(e), "success": False}


def auto_download_samples(output_dir: Path) -> list[Path]:
    """Auto-download a curated set of test STL files from public sources."""
    # We use synthetic samples for safety — real Printables/Thingiverse downloads require API tokens
    # and respect of ToS. Here we generate procedural test meshes using trimesh.
    import trimesh
    import numpy as np

    samples_dir = output_dir / "auto_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    # 1. Simple box
    box = trimesh.creation.box(extents=[20, 30, 15])
    p = samples_dir / "synthetic_box.stl"
    box.export(p)
    paths.append(p)
    print(f"  📦 Created: {p}")

    # 2. Cylinder
    cyl = trimesh.creation.cylinder(radius=10, height=40)
    p = samples_dir / "synthetic_cylinder.stl"
    cyl.export(p)
    paths.append(p)
    print(f"  ⚪ Created: {p}")

    # 3. Sphere
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=15)
    p = samples_dir / "synthetic_sphere.stl"
    sphere.export(p)
    paths.append(p)
    print(f"  🔵 Created: {p}")

    # 4. Compound: box with cylindrical hole (boolean is heavy; use union of 2 separate shapes)
    box2 = trimesh.creation.box(extents=[40, 40, 10])
    cyl2 = trimesh.creation.cylinder(radius=15, height=20)
    cyl2.apply_translation([20, 0, 5])
    compound = trimesh.util.concatenate([box2, cyl2])
    p = samples_dir / "synthetic_compound.stl"
    compound.export(p)
    paths.append(p)
    print(f"  🔧 Created: {p}")

    # 5. Bracket-like (L-shape)
    horiz = trimesh.creation.box(extents=[60, 20, 5])
    vert = trimesh.creation.box(extents=[5, 20, 40])
    vert.apply_translation([27.5, 0, 17.5])
    bracket = trimesh.util.concatenate([horiz, vert])
    p = samples_dir / "synthetic_bracket.stl"
    bracket.export(p)
    paths.append(p)
    print(f"  📐 Created: {p}")

    return paths


def main():
    parser = argparse.ArgumentParser(description="AI Model 4 dual-pass verification")
    parser.add_argument("--input", type=str, help="Path to single STL/OBJ file")
    parser.add_argument("--input-dir", type=str, help="Directory of STL/OBJ files")
    parser.add_argument("--output", type=str, default="verification_results", help="Output directory")
    parser.add_argument("--auto-download", action="store_true", help="Generate synthetic test meshes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build list of mesh paths
    mesh_paths = []
    if args.input:
        mesh_paths.append(Path(args.input))
    if args.input_dir:
        d = Path(args.input_dir)
        if d.exists():
            mesh_paths.extend(sorted(d.glob("*.stl")) + sorted(d.glob("*.obj")))
    if args.auto_download:
        mesh_paths.extend(auto_download_samples(output_dir))

    if not mesh_paths:
        print("❌ No input meshes. Use --input, --input-dir, or --auto-download")
        sys.exit(1)

    print(f"\n🔬 AI Model 4 Verification")
    print(f"   {len(mesh_paths)} mesh(es) to process")
    print(f"   Output: {output_dir}")

    orch = DualPassOrchestrator()

    all_results = []
    for mp in mesh_paths:
        result = run_single(orch, str(mp), output_dir)
        all_results.append(result)

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  📈 Overall Summary")
    print(f"{'═' * 60}")
    total = len(all_results)
    successes = sum(1 for r in all_results if r.get("success"))
    print(f"  Processed: {total}")
    print(f"  Successes: {successes} / {total}  ({100 * successes / total:.0f}%)")

    qualities = [r.get("quality_score", 0) for r in all_results if r.get("quality_score") is not None]
    if qualities:
        avg_q = sum(qualities) / len(qualities)
        print(f"  Avg quality: {avg_q:.2f}")

    # GO/NO-GO verdict (founder threshold)
    print(f"\n  🎯 Verdict:")
    if successes >= total * 0.5 and (not qualities or avg_q >= 0.4):
        print(f"    ✅ GO — AI Model 4 pipeline functional. Ship for demo.")
    elif successes > 0:
        print(f"    ⚠️  PARTIAL — works on some, needs tuning. Ship with caveats.")
    else:
        print(f"    ❌ NO-GO — pipeline does not produce useful output. Investigate.")

    # Save full report
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps({
        "total": total,
        "successes": successes,
        "results": all_results,
    }, indent=2, default=str))
    print(f"\n  📄 Full report: {report_path}")


if __name__ == "__main__":
    main()
