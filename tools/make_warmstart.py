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
             columns.json (fan_wtd_m), no network needed.  DEFAULT.  Edits only
             the aquifer (ZWT/WA) on the carrier.
    parflow  ParFlow-CONUS2 (Maxwell group) 1-km steady-state WTD via the
             hf_hydrodata API — better CONUS prior, but needs a registered
             HydroFrame account:  pip install hf_hydrodata, then
             hf_hydrodata.register_api_pin(<email>, <pin>).
    conus    Full slow-memory transplant from a spun-up CONUS 1-km ELM restart
             (--conus-restart). For each column, find the nearest gridcell and
             copy its natural-veg soil column's deep state — soil TEMPERATURE
             profile, soil moisture (liq/ice), and aquifer (ZWT/WA/ZWT_PERCH/
             FROST_TABLE/SOILP) — into the carrier's soil columns. This warm-
             starts the multi-year memory a 1-yr carrier and a WTD-only edit
             both miss (esp. deep soil temperature). Snow/canopy/PFT state are
             left as the carrier's (already a real-forcing January state).

    python3 tools/make_warmstart.py --run-dir <dir> [--cases-file cases_all.json]
                                    [--source fan] [--out-dir <dir>/warmstart]
    python3 tools/make_warmstart.py --run-dir <dir> --source conus \
                                    --conus-restart /path/conus_lat7.....elm.r.*.nc

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


# slow-memory, soil-column-level state to transplant from the CONUS restart.
# All are dimensioned [column] or [column, lev*]; snow/canopy/PFT are left as
# the carrier's (already a real-forcing January state for the target year).
CONUS_SLOW_STATE = ["T_SOISNO", "H2OSOI_LIQ", "H2OSOI_ICE", "SOILP",
                    "ZWT", "WA", "ZWT_PERCH", "FROST_TABLE"]
SOIL_LANDUNITS = {1, 2}          # natural-veg + crop columns (weight-bearing)


class ConusSource:
    """Nearest-gridcell state lookup into a big CONUS 1-km ELM restart.

    Opens the restart ONCE and holds the location + sub-grid index vectors in
    memory (a few hundred MB); each per-column query is a nearest-gridcell
    argmin plus single-column slices, so the 71 GiB file is never read whole.
    """
    def __init__(self, path):
        import netCDF4, numpy as np
        self.np = np
        self.d = netCDF4.Dataset(path)
        self.glat = self.d.variables["grid1d_lat"][:]
        glon = self.d.variables["grid1d_lon"][:]
        self.glon = np.where(glon > 180, glon - 360, glon)
        self.cgi = self.d.variables["cols1d_gridcell_index"][:]   # 1-based
        self.clun = self.d.variables["cols1d_ityplun"][:]
        self.lat_range = (float(self.glat.min()), float(self.glat.max()))

    def natveg_column(self, lat, lon):
        """Return (conus_soil_col_index, dist_km) for the nearest gridcell's
        natural-veg (ityplun==1) column, or (None, dist) if it has none."""
        np = self.np
        g = int(np.argmin((self.glat - lat) ** 2 + (self.glon - lon) ** 2))
        dist_km = 111.0 * np.hypot(self.glat[g] - lat,
                                   (self.glon[g] - lon) * np.cos(np.radians(lat)))
        cols = np.where(self.cgi == g + 1)[0]          # cgi is 1-based
        nat = cols[self.clun[cols] == 1]
        return (int(nat[0]) if len(nat) else None), float(dist_km)

    def read_state(self, col):
        """The slow-state arrays at one CONUS column, as {var: ndarray}."""
        return {v: self.np.array(self.d.variables[v][col]) for v in CONUS_SLOW_STATE
                if v in self.d.variables}


def conus_transplant(src, dst, conus, lat, lon):
    """Copy the CONUS nearest-cell natural-veg deep state into the carrier's
    soil columns (nat-veg + crop). Returns a provenance dict."""
    import netCDF4
    ncol, dist = conus.natveg_column(lat, lon)
    if ncol is None:
        return {"ok": False, "reason": f"no natural-veg column at nearest gridcell "
                                       f"({dist:.1f} km away)"}
    state = conus.read_state(ncol)
    shutil.copy2(src, dst)
    with netCDF4.Dataset(dst, "r+") as d:
        lun = d.variables["cols1d_ityplun"][:]
        soil_cols = [i for i, l in enumerate(lun) if int(l) in SOIL_LANDUNITS]
        for v, arr in state.items():
            for i in soil_cols:
                d.variables[v][i] = arr                 # [i] or [i,:] both OK
        zwt = float(state.get("ZWT", [float("nan")])[()]) if "ZWT" in state else float("nan")
        d.setncattr("warmstart", f"CONUS 1-km transplant: slow-memory soil state "
                    f"({', '.join(state)}) from natural-veg column {ncol} "
                    f"({dist:.1f} km from target); snow/canopy/PFT kept from carrier")
    tsoi = state.get("T_SOISNO")
    return {"ok": True, "conus_col": ncol, "dist_km": round(dist, 2),
            "applied_zwt_m": round(float(state["ZWT"]), 3) if "ZWT" in state else None,
            "deep_soiltemp_C": round(float(tsoi[-1]) - 273.15, 2) if tsoi is not None else None,
            "n_soil_columns": len(soil_cols)}


def conus_main(run_dir, out_dir, cases, args):
    """Build finidat files by transplanting slow state from a CONUS restart."""
    if not args.conus_restart:
        sys.exit("--source conus requires --conus-restart <file>")
    plan = json.loads((run_dir / args.plan_file).read_text())
    latlon = {cc["EXPERIMENT"]: (cc["lat"], cc["lon"])
              for cc in plan["CONDITIONS_COUPLERS"]}
    print(f"CONUS transplant from {Path(args.conus_restart).name}")
    conus = ConusSource(args.conus_restart)
    print(f"  source lat range {conus.lat_range[0]:.2f}..{conus.lat_range[1]:.2f} degN; "
          f"transplanting: {', '.join(CONUS_SLOW_STATE)}")

    manifest = {}
    for c in cases:
        name = c.split(".")[-1]
        ll = latlon.get(name)
        if ll is None:
            print(f"  ! {name}: no lat/lon in {args.plan_file} — skipped"); continue
        lat, lon = ll
        if not (conus.lat_range[0] <= lat <= conus.lat_range[1]):
            print(f"  ! {name}: lat {lat:.3f} outside this band file "
                  f"{conus.lat_range} — wrong band, skipped"); continue
        restarts = sorted(Path(c, "run").glob("*.elm.r.*.nc"))
        if not restarts:
            print(f"  ! {name}: no carrier restart in {c}/run — skipped"); continue
        dst = out_dir / f"Warmstart_{name}.nc"
        info = conus_transplant(restarts[-1], dst, conus, lat, lon)
        if not info["ok"]:
            print(f"  ! {name}: {info['reason']} — skipped"); continue
        print(f"  ✓ {name}: ZWT {info['applied_zwt_m']} m, deep Tsoil "
              f"{info['deep_soiltemp_C']} C  (CONUS cell {info['dist_km']} km, "
              f"{info['n_soil_columns']} soil cols)")
        manifest[name] = {"finidat": str(dst.resolve()), "source": "conus",
                          "conus_restart": str(Path(args.conus_restart).name), **info}
    if not manifest:
        sys.exit("no CONUS warm-start files produced — NOT overwriting warmstart.json")
    (out_dir / "warmstart.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)}/{len(cases)} CONUS warm-start files -> {out_dir}/")
    print(f"use: columns_to_plan.py ... --finidat-map {out_dir}/warmstart.json")


def main():
    ap = argparse.ArgumentParser(description="Build WTD-informed finidat files")
    ap.add_argument("--run-dir", required=True,
                    help="a COMPLETED run dir (its cases' restart files are the carriers)")
    ap.add_argument("--cases-file", default="cases.json")
    ap.add_argument("--plan-file", default="run_plan.json")
    ap.add_argument("--source", choices=("fan", "parflow", "conus"), default="fan")
    ap.add_argument("--conus-restart", default=None,
                    help="path to a CONUS 1-km ELM restart (required for --source conus)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "warmstart"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads((run_dir / args.cases_file).read_text())

    if args.source == "conus":
        conus_main(run_dir, out_dir, cases, args)
        return

    targets = (fan_targets(run_dir) if args.source == "fan"
               else parflow_targets(run_dir, args.plan_file))

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

    if not manifest:
        sys.exit(f"no warm-start files produced (source={args.source}) — "
                 "NOT overwriting any existing warmstart.json")
    (out_dir / "warmstart.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)}/{len(cases)} warm-start files -> {out_dir}/")
    print(f"use: columns_to_plan.py ... --finidat-map {out_dir}/warmstart.json")


if __name__ == "__main__":
    main()
