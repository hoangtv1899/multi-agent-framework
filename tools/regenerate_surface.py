#!/usr/bin/env python3
"""
Regenerate a single ELM surface file with the fixed generator.

Use this after fixing a parser bug in elm_surface_generator.py — old
cached files do not reflect new parser behavior, so the cache must be
busted explicitly.

Usage:
    python3 regenerate_surface.py <lat> <lon> [substrate]

Defaults:
    substrate = 'extrapolate'
    Reads MCP data from the most recent reception brief at
    /tmp/elm_reception_test_brief.json (or whatever
    --brief-path points at). Regenerate that first with:

        python3 tests/test_reception_elm.py "Run ELM at <site>..."

After running, point your case's user_nl_elm at the returned path:

        echo "fsurdat = '<returned_path>'" > user_nl_elm
        ./preview_namelists
        grep fsurdat CaseDocs/lnd_in
        time ./case.submit --no-batch
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Make src/ importable from anywhere
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from core.elm_surface_generator import ELMSurfaceGenerator   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("lat", type=float, help="Latitude (deg N)")
    p.add_argument("lon", type=float, help="Longitude (deg E)")
    p.add_argument("substrate", nargs="?", default="extrapolate",
                   help="Substrate fill (default: extrapolate)")
    p.add_argument("--brief-path",
                   default="/tmp/elm_reception_test_brief.json",
                   help="Path to reception brief JSON")
    args = p.parse_args()

    brief_path = Path(args.brief_path)
    if not brief_path.exists():
        print(f"❌ Brief not found: {brief_path}")
        print("   Run: python3 tests/test_reception_elm.py "
              "\"Run ELM at <site>...\"")
        sys.exit(1)

    brief = json.loads(brief_path.read_text())
    soil_profile = brief.get("soil_profile", {})

    if not soil_profile.get("layers"):
        print(f"❌ No soil_profile.layers in brief at {brief_path}")
        sys.exit(2)

    n_layers   = soil_profile.get("num_layers", "?")
    coverage_m = soil_profile.get("depth_coverage_m", "?")
    print(f"Brief: {brief_path}")
    print(f"  soil_profile: {n_layers} layers, "
          f"depth_coverage={coverage_m} m")
    print()

    # Find and delete any existing cached file for this combination
    gen = ELMSurfaceGenerator()
    cached_path = (
        gen.output_dir /
        f"Surfacedata_{args.lat:.4f}_{args.lon:.4f}_"
        f"native_{args.substrate}.nc"
    )
    if cached_path.exists():
        size_mb = cached_path.stat().st_size / 1e6
        print(f"Deleting cached file: {cached_path}")
        print(f"  (was {size_mb:.2f} MB)")
        cached_path.unlink()
    else:
        print(f"No cached file to delete at: {cached_path}")
    print()

    # Regenerate with current MCP data
    print(f"Regenerating ({args.lat}, {args.lon}, "
          f"substrate={args.substrate})...")
    print()
    path = gen.generate_from_mcp(
        lat       = args.lat,
        lon       = args.lon,
        mcp_data  = soil_profile,
        substrate = args.substrate,
        force     = True,
    )
    print()
    print(f"✓ Regenerated surface file:")
    print(f"  {path}")
    print()

    # Verify content
    print("Verifying new file contents...")
    import netCDF4 as nc
    with nc.Dataset(path) as ds:
        sand = ds.variables["PCT_SAND"][:].flatten().tolist()
        clay = ds.variables["PCT_CLAY"][:].flatten().tolist()
        org  = ds.variables["ORGANIC"][:].flatten().tolist()
        lat_v = float(ds.variables["LATIXY"][:].flatten()[0])
        lon_v = float(ds.variables["LONGXY"][:].flatten()[0])

    print(f"  LATIXY = {lat_v:.4f}, LONGXY = {lon_v:.4f}")
    print(f"  PCT_SAND per layer: "
          f"[{', '.join(f'{s:5.2f}' for s in sand)}]")
    print(f"  PCT_CLAY per layer: "
          f"[{', '.join(f'{c:5.2f}' for c in clay)}]")
    print(f"  ORGANIC  per layer: "
          f"[{', '.join(f'{o:5.2f}' for o in org)}]")
    print()
    print(f"Next: point your case at this file, e.g.:")
    print(f"  cd <case_dir>")
    print(f"  echo \"fsurdat = '{path}'\" > user_nl_elm")
    print(f"  ./preview_namelists")
    print(f"  grep fsurdat CaseDocs/lnd_in")
    print(f"  time ./case.submit --no-batch")


if __name__ == "__main__":
    main()