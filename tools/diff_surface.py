#!/usr/bin/env python3
"""
diff_surface_files.py — compare two ELM surface datasets

Usage:
    python3 diff_surface_files.py <generated.nc> <reference.nc>

Reports:
  - Schema differences (variables present in one but not the other)
  - Dimension shape mismatches
  - Per-variable min/max/mean for key soil + PFT parameters
  - Sanity flags: out-of-range values, NaN, sharp depth discontinuities
  - Specific check on the texture profile (sand/clay/organic by layer)

The motivating question: do the wrapper-generated surface files have
soil parameter profiles that could be making ELM's implicit soil-water
solver work much harder than necessary?
"""
import sys
import numpy as np
import xarray as xr


# Variables we expect on an ELM surface dataset
SOIL_VARS    = ['PCT_SAND', 'PCT_CLAY', 'ORGANIC']
PFT_VARS     = ['PCT_NAT_PFT', 'PCT_NATVEG', 'PCT_CROP']
LANDCOV_VARS = ['PCT_LAKE', 'PCT_GLACIER', 'PCT_URBAN', 'PCT_WETLAND']
COORD_VARS   = ['LATIXY', 'LONGXY', 'LANDFRAC_PFT', 'PFTDATA_MASK']


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def summarize_var(name, data):
    """Print a 1-line summary of a variable."""
    arr = np.asarray(data).ravel()
    if arr.size == 0:
        print(f"  {name:20s}  (empty)")
        return
    nan_count = np.isnan(arr).sum() if arr.dtype.kind == 'f' else 0
    print(f"  {name:20s}  shape={data.shape!s:20s}  "
          f"min={float(np.nanmin(arr)):.3f}  "
          f"max={float(np.nanmax(arr)):.3f}  "
          f"mean={float(np.nanmean(arr)):.3f}  "
          f"NaN={nan_count}")


def compare_var(name, a, b):
    """Compare a single variable between two datasets."""
    print(f"\n— {name} —")
    if name not in a.data_vars:
        print(f"  ✗ not in generated")
        if name in b.data_vars:
            print(f"  ✓ in reference: shape={b[name].shape}")
        return
    if name not in b.data_vars:
        print(f"  ✓ in generated: shape={a[name].shape}")
        print(f"  ✗ not in reference")
        return

    av = a[name].values
    bv = b[name].values
    summarize_var(f"generated", av)
    summarize_var(f"reference", bv)

    if av.shape != bv.shape:
        print(f"  ⚠️  SHAPE MISMATCH: {av.shape} vs {bv.shape}")
        return

    if av.dtype.kind in 'fi':
        diff = av - bv
        print(f"  diff               "
              f"min={float(np.nanmin(diff)):+.3f}  "
              f"max={float(np.nanmax(diff)):+.3f}  "
              f"mean={float(np.nanmean(diff)):+.3f}  "
              f"|max|={float(np.nanmax(np.abs(diff))):.3f}")


def check_soil_profile_sanity(name, ds):
    """Profile-by-layer sanity check for PCT_SAND, PCT_CLAY, ORGANIC."""
    section(f"SOIL PROFILE SANITY — {name}")
    for var in ['PCT_SAND', 'PCT_CLAY', 'ORGANIC']:
        if var not in ds.data_vars:
            print(f"  {var}: not present")
            continue
        v = ds[var].values
        # Pick the soil-level axis (usually 'nlevsoi' or 'nlevgrnd')
        # and squeeze single-point grids
        v_sq = np.squeeze(v)
        if v_sq.ndim == 0:
            print(f"  {var}: scalar = {float(v_sq):.2f}")
            continue
        # Show profile
        print(f"\n  {var} profile (per soil level):")
        if v_sq.ndim == 1:
            for i, val in enumerate(v_sq):
                bar = '█' * int(min(100, max(0, val)) / 2)
                print(f"    layer {i+1:2d}: {float(val):6.2f}  {bar}")
            # Discontinuity check
            adj_diffs = np.abs(np.diff(v_sq))
            big_jumps = np.where(adj_diffs > 40)[0]
            if big_jumps.size > 0:
                print(f"  ⚠️  Sharp discontinuity at layer boundary(ies) "
                      f"{[int(i)+1 for i in big_jumps]} (jump > 40%)")
        else:
            # multi-dim: just summarize
            print(f"    shape={v_sq.shape}, "
                  f"flat min/max/mean = "
                  f"{float(np.nanmin(v_sq)):.2f} / "
                  f"{float(np.nanmax(v_sq)):.2f} / "
                  f"{float(np.nanmean(v_sq)):.2f}")

        # Range check
        if var in ('PCT_SAND', 'PCT_CLAY'):
            bad = np.sum((v_sq < 0) | (v_sq > 100))
            if bad > 0:
                print(f"  ⚠️  {bad} cell(s) outside [0, 100]")
        if var == 'ORGANIC':
            bad = np.sum((v_sq < 0) | (v_sq > 130))
            if bad > 0:
                print(f"  ⚠️  {bad} cell(s) outside [0, 130]")

    # Sand+Clay+Silt (implied) should not exceed 100
    if 'PCT_SAND' in ds.data_vars and 'PCT_CLAY' in ds.data_vars:
        s = np.squeeze(ds['PCT_SAND'].values)
        c = np.squeeze(ds['PCT_CLAY'].values)
        if s.shape == c.shape:
            sc = s + c
            over = np.sum(sc > 100.01)
            if over > 0:
                print(f"\n  ⚠️  PCT_SAND + PCT_CLAY > 100 in {over} cell(s) — "
                      f"max sum = {float(np.nanmax(sc)):.2f}")
            else:
                print(f"\n  ✓ PCT_SAND + PCT_CLAY ≤ 100 everywhere "
                      f"(max sum = {float(np.nanmax(sc)):.2f})")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    gen_path = sys.argv[1]
    ref_path = sys.argv[2]

    print(f"GENERATED : {gen_path}")
    print(f"REFERENCE : {ref_path}")

    a = xr.open_dataset(gen_path)
    b = xr.open_dataset(ref_path)

    # ---------- SCHEMA ----------
    section("SCHEMA COMPARISON")
    a_vars = set(a.data_vars.keys())
    b_vars = set(b.data_vars.keys())
    a_dims = dict(a.dims)
    b_dims = dict(b.dims)

    print(f"  generated: {len(a_vars)} variables, dims={a_dims}")
    print(f"  reference: {len(b_vars)} variables, dims={b_dims}")
    print()

    only_a = sorted(a_vars - b_vars)
    only_b = sorted(b_vars - a_vars)
    if only_a:
        print(f"  Only in GENERATED  ({len(only_a)}): {only_a}")
    if only_b:
        print(f"  Only in REFERENCE  ({len(only_b)}): {only_b}")
    if not only_a and not only_b:
        print(f"  ✓ Identical variable list")

    # Dim mismatches
    print()
    common_dims = set(a_dims) & set(b_dims)
    for d in sorted(common_dims):
        if a_dims[d] != b_dims[d]:
            print(f"  ⚠️  Dim {d}: generated={a_dims[d]}, reference={b_dims[d]}")

    # ---------- VARIABLE COMPARISON ----------
    section("KEY VARIABLE COMPARISON")
    for v in COORD_VARS + SOIL_VARS + LANDCOV_VARS + PFT_VARS:
        compare_var(v, a, b)

    # ---------- SOIL PROFILE DEEP DIVE ----------
    check_soil_profile_sanity("GENERATED", a)
    check_soil_profile_sanity("REFERENCE", b)

    print()
    print("Done.")


if __name__ == "__main__":
    main()