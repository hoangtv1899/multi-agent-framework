#!/usr/bin/env python3
"""
WTD-informed WARM START — the scalable alternative to per-watershed spin-up.

Takes each column's completed-run restart file (the "carrier" state) and edits
its groundwater state to a target water-table depth from a CONUS-wide
equilibrium prior, producing a finidat file the next run can start from:

    ZWT  <- target WTD (clamped to ELM's aquifer range)
    WA   <- ELM's exact aquifer relation  WA = Sy*1000*(zi_bot + 25 - ZWT),
            verified against cold-start files (ZWT 8.802 m <-> WA 4000 mm)
    (soil moisture is left alone — it re-equilibrates in weeks; the aquifer is
     the year-scale state this fixes)

WTD sources (--source):
    fan      Fan et al. 2013 equilibrium WTD — read per column from the run's
             columns.json (fan_wtd_m), no network needed.  DEFAULT.
    parflow  ParFlow-CONUS2 (Maxwell group) 1-km steady-state WTD via the
             hf_hydrodata API — better CONUS prior, but needs a registered
             HydroFrame account:  pip install hf_hydrodata, then
             hf_hydrodata.register_api_pin(<email>, <pin>).

    python3 tools/make_warmstart.py --run-dir <dir> [--cases-file cases_all.json]
                                    [--source fan] [--out-dir <dir>/warmstart]

Writes Warmstart_<col>.nc per column + warmstart.json (col -> finidat path,
target vs applied WTD). Wire into a new run via columns_to_plan --finidat-map.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")

SY = 0.2                    # ELM unconfined-aquifer specific yield
ZI_BOT = 3.8019             # bottom interface of the 10-layer soil column (m)
ZWT_MIN, ZWT_MAX = 0.05, ZI_BOT + 25.0     # ELM aquifer conceptual range


def wa_for_zwt(zwt):
    """ELM's aquifer storage (mm) for a water table at zwt (m)."""
    return max(0.0, min(5000.0, SY * 1000.0 * (ZI_BOT + 25.0 - zwt)))


def fan_targets(run_dir):
    cols = json.loads((run_dir / "columns.json").read_text())
    cols = cols.get("columns", cols) if isinstance(cols, dict) else cols
    return {c["id"]: c.get("fan_wtd_m") for c in cols}


def parflow_targets(run_dir, plan_file):
    """ParFlow-CONUS2-based WTD (Ma et al. 2025, grid conus2_wtd.30) at each
    column via the HydroFrame hf_hydrodata API. Dataset/option names verified
    against the live catalog; gridded fetches need a registered account:
        1. sign up:  https://hydrogen.princeton.edu/signup
        2. get pin:  https://hydrogen.princeton.edu/pin
        3. python3 -c "import hf_hydrodata; hf_hydrodata.register_api_pin('<email>','<pin>')"
    """
    try:
        import hf_hydrodata as hf
    except ImportError:
        sys.exit("pip install --user hf_hydrodata, then register (see docstring)")
    import numpy as np
    plan = json.loads((run_dir / plan_file).read_text())
    out = {}
    d = 0.005                                   # ~500 m box around the column
    for cc in plan["CONDITIONS_COUPLERS"]:
        opts = {"dataset": "ma_2025", "variable": "water_table_depth",
                "period": "static",
                "latlng_bounds": [cc["lat"] - d, cc["lon"] - d,
                                  cc["lat"] + d, cc["lon"] + d]}
        try:
            v = np.asarray(hf.get_gridded_data(opts), dtype=float)
            if np.isfinite(v).any():
                out[cc["EXPERIMENT"]] = float(np.nanmean(v))
        except Exception as e:
            print(f"  ! {cc['EXPERIMENT']}: hf_hydrodata failed ({str(e)[:80]}) — skipped")
    return out


def edit_restart(src, dst, zwt_target):
    """Copy the restart and set ZWT/WA on every sub-grid column."""
    import netCDF4
    shutil.copy2(src, dst)
    zwt = min(max(zwt_target, ZWT_MIN), ZWT_MAX)
    with netCDF4.Dataset(dst, "r+") as d:
        d.variables["ZWT"][:] = zwt
        d.variables["WA"][:] = wa_for_zwt(zwt)
        d.setncattr("warmstart", f"ZWT set to {zwt:.3f} m (target {zwt_target:.3f}) "
                                 f"from equilibrium prior; WA from ELM aquifer relation")
    return zwt


def main():
    ap = argparse.ArgumentParser(description="Build WTD-informed finidat files")
    ap.add_argument("--run-dir", required=True,
                    help="a COMPLETED run dir (its cases' restart files are the carriers)")
    ap.add_argument("--cases-file", default="cases.json")
    ap.add_argument("--plan-file", default="run_plan.json")
    ap.add_argument("--source", choices=("fan", "parflow"), default="fan")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "warmstart"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = (fan_targets(run_dir) if args.source == "fan"
               else parflow_targets(run_dir, args.plan_file))
    cases = json.loads((run_dir / args.cases_file).read_text())

    manifest = {}
    print(f"warm-start from {args.source} WTD  (clamp {ZWT_MIN}-{ZWT_MAX:.1f} m; "
          f"deeper targets pin the aquifer empty)")
    for c in cases:
        name = c.split(".")[-1]
        t = targets.get(name)
        if t is None:
            print(f"  ! {name}: no {args.source} WTD — skipped")
            continue
        restarts = sorted(Path(c, "run").glob("*.elm.r.*.nc"))
        if not restarts:
            print(f"  ! {name}: no restart file in {c}/run — skipped")
            continue
        dst = out_dir / f"Warmstart_{name}.nc"
        applied = edit_restart(restarts[-1], dst, t)
        clamp = "" if abs(applied - t) < 1e-6 else f"  (clamped from {t:.1f})"
        print(f"  ✓ {name}: ZWT {applied:.2f} m, WA {wa_for_zwt(applied):.0f} mm{clamp}")
        # absolute path — ELM resolves finidat from the case run dir
        manifest[name] = {"finidat": str(dst.resolve()), "target_wtd_m": round(t, 3),
                          "applied_zwt_m": round(applied, 3), "source": args.source}

    (out_dir / "warmstart.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)}/{len(cases)} warm-start files -> {out_dir}/")
    print(f"use: columns_to_plan.py ... --finidat-map {out_dir}/warmstart.json")


if __name__ == "__main__":
    main()
