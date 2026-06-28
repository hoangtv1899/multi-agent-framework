#!/usr/bin/env python3
"""
Controlled SOIL SWEEP — emit an executable ELM plan that varies ONLY soil
texture (a clay gradient, sand → clay) at a SINGLE location, so forcing, PFT and
everything else are identical across runs. This is the conceptual-archetype
counterpart to the spatial expander: it isolates the soil control on the
runoff/recharge partitioning with no forcing/elevation confound.

Each coupler = one synthetic UNIFORM soil profile (constant organic + bulk
density; only clay/sand vary) at the same (lat,lon), native soil_config +
extrapolate substrate — the same surface-generation path the real run used.

    module load pytorch/2.8.0
    python3 tools/make_soil_sweep.py --out-dir workflow_outputs/soil_sweep
    # then:
    #   python3 tools/build_cases.py --plan <out>/soilsweep_plan.json --ref <ref_case> --out-dir <out>
    #   salloc ... bash tools/run_cases.sh "$(cat <out>/exe_path.txt)" <case dirs>
    #   python3 tools/analyze_run.py --run-dir <out> --cases-file cases.json --plan-file soilsweep_plan.json --plot
"""
import argparse
import json
from pathlib import Path

# clay gradient (sand co-varies, silt = remainder). organic + bulk density FIXED,
# so clay/sand are the only thing that changes across the sweep.
TEXTURES = [(5, 82), (10, 65), (18, 42), (27, 33), (35, 30), (45, 25), (55, 18)]


def uniform_profile(clay, sand, organic=2.0, depth_cm=200):
    silt = round(max(0.0, 100.0 - clay - sand), 1)
    return {
        "source": "synthetic uniform (soil sweep)",
        "num_layers": 1,
        "layers": [{
            "texture_class": f"clay{clay:.0f}",
            "depth_top_cm": "0", "depth_bot_cm": str(depth_cm),
            "sand_pct": float(sand), "silt_pct": silt, "clay_pct": float(clay),
            "organic_matter_pct": str(organic), "bulk_density_gcc": "1.4",
        }],
    }


def main():
    ap = argparse.ArgumentParser(description="Emit a controlled soil-sweep ELM plan")
    # default site = a Naches high-precip (1373 mm/yr) cell, so there is ample
    # water to partition; override with --lat/--lon for a different forcing cell.
    ap.add_argument("--lat", type=float, default=47.1106)
    ap.add_argument("--lon", type=float, default=-121.3851)
    ap.add_argument("--yr-start", type=int, default=1995)
    ap.add_argument("--yr-end", type=int, default=1995)
    ap.add_argument("--out-dir", default="workflow_outputs/soil_sweep")
    args = ap.parse_args()

    stop_n = args.yr_end - args.yr_start + 1
    couplers = [{
        "EXPERIMENT": f"clay{clay:02.0f}",
        "FORCING_PERIOD": "baseline",
        "DATM_CLMNCEP_YR_START": str(args.yr_start),
        "DATM_CLMNCEP_YR_END": str(args.yr_end),
        "STOP_N": str(stop_n),
        "RUN_STARTDATE": f"{args.yr_start}-01-01",
        "SOIL_CONFIG": "native", "SUBSTRATE": "extrapolate",
        "DESCRIPTION": f"uniform clay={clay}% sand={sand}% @ fixed site",
        "lat": args.lat, "lon": args.lon,
        "soil_profile": uniform_profile(clay, sand),
    } for clay, sand in TEXTURES]

    plan = {"CONDITIONS_COUPLERS": couplers,
            "ELM_CONFIG": {"base_stop_option": "nyears", "base_rest_n": "1",
                           "base_rest_option": "nyears"}}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "soilsweep_plan.json").write_text(json.dumps(plan, indent=2))
    print(f"{len(couplers)} soil treatments @ ({args.lat}, {args.lon}) "
          f"-> {out / 'soilsweep_plan.json'}")
    print("clay gradient (%):", [c for c, _ in TEXTURES])


if __name__ == "__main__":
    main()
