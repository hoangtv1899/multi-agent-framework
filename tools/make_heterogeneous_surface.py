#!/usr/bin/env python3
"""
make_heterogeneous_surface.py — isolate the "uniform vs heterogeneous soil"
                                 test variable

Takes an ELM surface dataset with a uniform-flat soil profile and rewrites
PCT_SAND / PCT_CLAY / ORGANIC to have realistic depth variation, while
leaving coordinates, PFT distribution, land mask, and everything else
unchanged. The output file passes the same init consistency checks as the
original (same lat/lon, same domain), so we can swap it in directly via
user_nl_elm.

Usage:
    python3 make_heterogeneous_surface.py <input.nc> <output.nc>

Output profile (10 nlevsoi layers, top → bottom):
    PCT_SAND  : 75 → 55   (slight decrease with depth)
    PCT_CLAY  :  8 → 22   (typical clay accumulation B-horizon)
    ORGANIC   : 25 → 0    (exponential decay, ~0 below ~1m)

These values are chosen to be:
  - Realistic for a Mediterranean-climate alluvial site
  - Different enough from the original uniform profile (68/12.5/3) to
    create a meaningful solver-behavior contrast
  - Conservative: PCT_SAND + PCT_CLAY stays well under 100 at every layer
"""
import sys
import shutil
from pathlib import Path

import numpy as np
import netCDF4 as nc


def heterogeneous_profile(n_layers: int = 10):
    """Return realistic depth-varying sand/clay/organic profiles."""
    sand    = np.linspace(75.0, 55.0, n_layers)
    clay    = np.linspace( 8.0, 22.0, n_layers)
    organic = 25.0 * np.exp(-np.arange(n_layers) / 2.5)
    organic[organic < 0.1] = 0.0
    return sand, clay, organic


def patch_var(ds, var_name, profile):
    """Broadcast a depth profile across the (nlevsoi, lsmlat, lsmlon) array."""
    if var_name not in ds.variables:
        print(f"  ⚠️  {var_name} not present; skipped")
        return

    var = ds.variables[var_name]
    shp = var.shape
    n_lev = profile.shape[0]

    if shp[0] != n_lev:
        print(f"  ⚠️  {var_name} has shape {shp}, expected first dim={n_lev}; skipped")
        return

    # Broadcast: each layer gets the profile value at that level
    for i in range(n_lev):
        var[i, ...] = profile[i]

    formatted = ", ".join(f"{v:5.1f}" for v in profile)
    print(f"  ✓ {var_name:12s} [{formatted}]")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    if not src.exists():
        print(f"❌ Source file not found: {src}")
        sys.exit(2)

    print(f"Source:      {src}")
    print(f"Destination: {dst}")

    shutil.copy(src, dst)
    print(f"✓ Copied to {dst}")

    sand, clay, organic = heterogeneous_profile()
    print()
    print("Applying heterogeneous profile:")

    with nc.Dataset(dst, 'r+') as ds:
        patch_var(ds, 'PCT_SAND',  sand)
        patch_var(ds, 'PCT_CLAY',  clay)
        patch_var(ds, 'ORGANIC',   organic)

    # Verify
    print()
    print("Verification (re-read after write):")
    with nc.Dataset(dst, 'r') as ds:
        for v in ('PCT_SAND', 'PCT_CLAY', 'ORGANIC'):
            if v not in ds.variables:
                continue
            arr = np.array(ds.variables[v][:]).squeeze()
            if arr.ndim == 0:
                print(f"  {v:12s} = {float(arr):.2f}")
            else:
                top = float(arr.ravel()[0])
                bot = float(arr.ravel()[-1])
                print(f"  {v:12s} top={top:6.2f}  bottom={bot:6.2f}")

    sand_plus_clay_max = float(np.nanmax(sand + clay))
    print()
    print(f"  Sanity: max(PCT_SAND + PCT_CLAY) = {sand_plus_clay_max:.1f}  "
          f"({'OK' if sand_plus_clay_max <= 100 else 'EXCEEDS 100 — invalid'})")

    print(f"\n✓ Heterogeneous surface file ready: {dst}")


if __name__ == "__main__":
    main()