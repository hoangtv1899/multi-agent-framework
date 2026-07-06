#!/usr/bin/env python3
"""
Standalone PFLOTRAN ensembles — the groundwater-community path.

Builds one 1-D variably-saturated (RICHARDS) column per sampled site, straight
from the expander's columns.json — NO ELM anywhere:

    materials  <- the column's SSURGO horizons (van Genuchten params are
                  already in columns.json), deepest horizon extended as
                  substrate down to the domain bottom
    initial    <- hydrostatic profile pinned at the Fan 2013 water table
                  (same CONUS prior the ELM warm start uses)
    top BC     <- steady recharge flux (scenario, --recharge-mm-yr)
    bottom BC  <- hydrostatic at the Fan water table (--bottom fan, default)
                  or no-flow (--bottom none)

Reuses the legacy deck writer (core.pflotran_input_agent). Runs are SERIAL and
take seconds — `--run` executes them all right here (no batch queue needed).

    module load pytorch/2.8.0
    python3 tools/build_pflotran_cases.py --columns <run>/columns.json \
        --out-dir workflow_outputs/<study> [--recharge-mm-yr 100] [--run]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from core.pflotran_input_agent import PFLOTRANInputAgent

PFLOTRAN_EXE = "/global/homes/h/hvtran/petsc/pflotran/src/pflotran/bin/pflotran"
VISC, RHO_G = 1.002e-3, 998.0 * 9.81          # k[m2] = Ksat[m/s] * mu / rho*g
ATM = 101325.0


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def dominant_horizons(profile):
    """SSURGO profiles may stack horizons from several map-unit components —
    keep the first component's horizons, ordered by depth."""
    layers = (profile or {}).get("layers") or []
    if not layers:
        return []
    comp = layers[0].get("component")
    hz = [l for l in layers if l.get("component") == comp]
    hz.sort(key=lambda l: _f(l.get("depth_top_cm"), 0))
    return hz


def layer_props(hz):
    """PFLOTRAN material + curve from one SSURGO horizon (VG params ride along
    in columns.json). Residual SATURATION = theta_r / theta_s."""
    vg = hz.get("van_genuchten") or {}
    ts = _f(vg.get("theta_s"), 0.45)
    tr = _f(vg.get("theta_r"), 0.05)
    ksat = _f(vg.get("ksat_ms")) or (_f(hz.get("ksat_ums"), 3.0) * 1e-6)
    alpha = _f(vg.get("alpha_per_m"), 1e-4)     # value pattern is 1/Pa
    if alpha > 0.01:                            # ...unless it's really 1/m
        alpha /= RHO_G
    return {
        "porosity": round(ts, 4),
        "permeability": ksat * VISC / RHO_G,
        "alpha": alpha,
        "m": _f(vg.get("m"), 0.4),
        "residual_saturation": round(min(tr / ts, 0.4), 4),
    }


def build_column(col, out_dir, args):
    """One 1-D column deck; returns (case_dir, meta)."""
    hz = dominant_horizons(col.get("soil_profile"))
    fan = _f(col.get("fan_wtd_m"), 10.0)
    depth = min(args.depth_cap, max(fan + 5.0, 12.0))     # WT inside if it fits

    # cells: SSURGO horizons on top (one cell each), substrate coarsening below
    soil = []                                             # top -> down
    for l in hz:
        t = (_f(l.get("depth_bot_cm"), 0) - _f(l.get("depth_top_cm"), 0)) / 100.0
        if t > 0.005:
            soil.append((t, layer_props(l)))
    soil_depth = sum(t for t, _ in soil)
    sub_props = soil[-1][1] if soil else layer_props({})
    sub = []
    dz, z = 0.3, soil_depth
    while z < depth:
        dz = min(dz * 1.35, 2.5, depth - z)
        sub.append((dz, sub_props))
        z += dz

    cells = soil + sub                                    # top -> down
    thicknesses = [t for t, _ in cells][::-1]             # PFLOTRAN: bottom -> top
    H = sum(thicknesses)

    case = PFLOTRANInputAgent(nx=1, ny=1, dx=1.0, dy=1.0,
                              layer_thicknesses=thicknesses,
                              case_name=col["id"])
    names = []
    for i, (t, p) in enumerate(cells[::-1]):              # bottom -> top
        n = f"L{i+1:02d}"
        names.append(n)
        case.add_material_property({"name": n, "id": i + 1,
                                    "porosity": p["porosity"],
                                    "permeability": p["permeability"]})
        case.add_characteristic_curve({"name": n, "alpha": p["alpha"], "m": p["m"],
                                       "liquid_residual_saturation": p["residual_saturation"]})
    case.set_layer_order(names)
    case.add_strata_from_layer_order()

    # hydrostatic everywhere, water table (P=atm) at the Fan depth
    datum = (0.0, 0.0, H - fan)                           # may sit below domain
    case.flow_conditions[0].update({"datum": datum, "liquid_pressure": ATM})
    case.add_flow_condition({
        "name": "recharge", "type": "LIQUID_FLUX NEUMANN",
        "flux_data": {"time_units": "y", "data_units": "m/y",
                      "values": [(0.0, args.recharge_mm_yr / 1000.0)]}})
    case.add_boundary_condition("top_recharge", "recharge", "top")
    if args.bottom == "fan":
        case.add_flow_condition({"name": "water_table",
                                 "type": "LIQUID_PRESSURE HYDROSTATIC",
                                 "datum": datum, "liquid_pressure": ATM})
        case.add_boundary_condition("bottom_wt", "water_table", "bottom")

    case.time_config.update({"final_time": args.years, "initial_timestep": 1.0,
                             "maximum_timestep": 0.05})
    case.output_config["times"] = sorted({1.0, args.years / 4, args.years / 2, args.years})
    case_dir = case.prepare_case(output_dir=str(out_dir))
    return case_dir, {"id": col["id"], "elevation_m": col.get("elevation_m"),
                      "fan_wtd_m": fan, "depth_m": round(H, 2),
                      "n_cells": len(cells), "n_soil_horizons": len(soil),
                      "wt_in_domain": fan < H}


def run_case(case_dir, name):
    t0 = time.time()
    r = subprocess.run([PFLOTRAN_EXE, "-pflotranin", f"{name}.in"],
                       cwd=case_dir, capture_output=True, text=True, timeout=600)
    ok = "Simulation completed" in (r.stdout + r.stderr) or r.returncode == 0
    return ok, time.time() - t0, (r.stdout + r.stderr)


def main():
    ap = argparse.ArgumentParser(description="Standalone 1-D PFLOTRAN ensemble from columns.json")
    ap.add_argument("--columns", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--recharge-mm-yr", type=float, default=100.0)
    ap.add_argument("--depth-cap", type=float, default=50.0)
    ap.add_argument("--years", type=float, default=20.0)
    ap.add_argument("--bottom", choices=("fan", "none"), default="fan")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--run", action="store_true", help="run all cases serially right here")
    args = ap.parse_args()

    data = json.loads(Path(args.columns).read_text())
    cols = data.get("columns", data) if isinstance(data, dict) else data
    if args.limit:
        cols = cols[:args.limit]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for col in cols:
        case_dir, meta = build_column(col, out, args)
        meta["case_dir"] = case_dir
        note = "" if meta["wt_in_domain"] else "  (WT below domain — capped, unsaturated column)"
        print(f"  ✓ {meta['id']}: {meta['n_cells']} cells, {meta['depth_m']} m, "
              f"Fan WTD {meta['fan_wtd_m']} m{note}")
        manifest.append(meta)

    if args.run:
        print(f"\nrunning {len(manifest)} serial PFLOTRAN columns ...")
        for m in manifest:
            ok, dt, log = run_case(m["case_dir"], m["id"])
            m["run_ok"], m["run_seconds"] = ok, round(dt, 1)
            print(f"  {'✓' if ok else '✗'} {m['id']}: {dt:.1f}s")
            if not ok:
                tail = "\n".join(log.strip().splitlines()[-6:])
                print("    " + tail.replace("\n", "\n    "))

    (out / "pflotran_cases.json").write_text(json.dumps(
        {"scenario": {"recharge_mm_yr": args.recharge_mm_yr, "years": args.years,
                      "bottom_bc": args.bottom, "depth_cap_m": args.depth_cap},
         "cases": manifest}, indent=2))
    n_ok = sum(1 for m in manifest if m.get("run_ok"))
    print(f"\n{len(manifest)} decks -> {out}/pflotran_cases.json"
          + (f"  ({n_ok}/{len(manifest)} ran OK)" if args.run else ""))


if __name__ == "__main__":
    main()
