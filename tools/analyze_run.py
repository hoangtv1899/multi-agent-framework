#!/usr/bin/env python3
"""
Analyze a completed ELM run — read the per-column history files, extract the
hydrology, and summarize how recharge / runoff / soil moisture / water-table
depth vary across the ensemble (spatially, by elevation).

Wraps core.ELMResultsAnalyzer over a pipeline run dir that holds:
    phase3_cases.json   the case directories that were run
    phase3_plan.json    per-column metadata (forcing, lat/lon, years)
    columns.json        per-column elevation (optional, for the gradient)

Writes hydro_summary.json (+ an elevation-gradient figure with --plot) into
<run-dir>/04_analysis/. NOTHING is executed — read-only over existing output.

Run from the project root with the analysis env:
    module load pytorch/2.8.0
    python3 tools/analyze_run.py --run-dir workflow_outputs/pipeline_XXXX --plot
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")          # quiet xarray/netCDF futurewarnings
sys.path.insert(0, "src")
from core.elm_results_analyzer import ELMResultsAnalyzer


def soil_features(sp):
    """Per-column soil predictors from the SSURGO profile that drive drainage:
    max clay % (the impeding layer) and min Ksat (the drainage bottleneck)."""
    layers = (sp or {}).get("layers") or []
    if not layers:
        return None

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    clays = [c for c in (num(l.get("clay_pct")) for l in layers) if c is not None]
    ksats = [k for k in (num(l.get("ksat_ums")) for l in layers) if k is not None]
    return {
        "texture_top":  layers[0].get("texture_class"),
        "clay_top_pct": num(layers[0].get("clay_pct")),
        "clay_max_pct": max(clays) if clays else None,
        "ksat_min_ums": min(ksats) if ksats else None,
        "n_layers":     len(layers),
    }


def build_experiments(run_dir: Path, cases_file="phase3_cases.json",
                      plan_file="phase3_plan.json"):
    cases = json.load(open(run_dir / cases_file))
    plan = {c["EXPERIMENT"]: c for c in
            json.load(open(run_dir / plan_file))["CONDITIONS_COUPLERS"]}
    elev = {}
    cj = run_dir / "columns.json"
    if cj.exists():
        cols = json.load(open(cj))
        cols = cols["columns"] if isinstance(cols, dict) else cols
        elev = {c["id"]: c.get("elevation_m") for c in cols}

    exps = []
    for cd in cases:
        name = cd.split(".")[-1]                       # ...col_01 -> col_01
        cc = plan.get(name, {})
        exps.append({
            "case_name": name, "case_dir": cd,
            "scenario_name": f"{name} ({elev[name]:.0f} m)" if elev.get(name) else name,
            "forcing_period": cc.get("FORCING_PERIOD", "baseline"),
            "forcing_start": int(cc.get("DATM_CLMNCEP_YR_START", 0) or 0),
            "forcing_end":   int(cc.get("DATM_CLMNCEP_YR_END", 0) or 0),
            "lat": cc.get("lat"), "lon": cc.get("lon"),
            "elevation_m": elev.get(name),
            "soil": soil_features(cc.get("soil_profile")),
        })
    return exps


def _f(x, d=1):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else "-"


def print_summary(results, spatial):
    print("\n" + "=" * 84)
    print("PER-COLUMN HYDROLOGY  (annual means)")
    print("=" * 84)
    print(f"{'column':<9}{'elev_m':>8}{'precip':>9}{'recharge':>11}{'runoff':>9}"
          f"{'rech.frac':>11}{'WTD_m':>8}")
    print(f"{'':9}{'':>8}{'mm/yr':>9}{'mm/yr':>11}{'mm/yr':>9}{'':>11}{'':>8}")
    print("-" * 84)
    ok = sorted((r for r in results.values() if r["status"] == "ok"),
                key=lambda r: (r.get("elevation_m") or 0))
    for r in ok:
        m = r["metrics"]
        print(f"{r['case_name']:<9}{_f(r.get('elevation_m'), 0):>8}{_f(m.get('precip_mm_yr'), 0):>9}"
              f"{_f(m.get('annual_recharge_mm_yr')):>11}{_f(m.get('annual_runoff_mm_yr')):>9}"
              f"{_f(m.get('recharge_fraction'), 3):>11}{_f(m.get('water_table_depth_m'), 2):>8}")
    print("-" * 84)
    if spatial:
        fo, ve, dc = spatial["forcing"], spatial["vs_elevation"], spatial["driver_correlation"]
        print(f"forcing: {fo['n_forcing_bins']} distinct precip value(s) "
              f"{fo['precip_mm_yr_distinct']} mm/yr  (elevation-resolved: {fo['elevation_resolved']})")
        print(f"recharge vs elevation: slope {_f(ve['slope_per_1000m']['recharge_mm_yr'])}/1000m, "
              f"fit r2={ve['fit_r2']['recharge_mm_yr']}, corr r={dc['recharge_vs_elevation_r']}"
              f"  |  vs precip: corr r={dc['recharge_vs_precip_r']}")
        print("interpretation:")
        for n in spatial["interpretation"]:
            print(f"  • {n}")
    else:
        print("(single location or no elevation — no spatial summary)")
    print("=" * 84)


def plot_gradient(spatial, out_path):
    """Honest scatter: points coloured by forcing bin, with recharge plotted
    against BOTH elevation (confounded proxy) and precip (the actual driver) —
    no connecting lines that would imply a smooth gradient."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    rows = spatial["by_elevation"]
    nan = float("nan")
    e = np.array([r["elevation_m"] for r in rows], float)
    p = np.array([r["precip_mm_yr"] if r["precip_mm_yr"] is not None else nan for r in rows], float)
    rech = np.array([r["recharge_mm_yr"] for r in rows], float)
    frac = np.array([r["recharge_fraction"] for r in rows], float)

    bins = spatial["forcing"]["precip_mm_yr_distinct"]
    palette = ["#2c7fb8", "#d95f0e", "#31a354", "#756bb1", "#e7298a"]
    cmap = {b: palette[i % len(palette)] for i, b in enumerate(bins)}
    cols = [cmap.get(round(pi), "#999999") if not np.isnan(pi) else "#999999" for pi in p]

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.9))
    ax[0].scatter(e, rech, c=cols, s=60, edgecolor="#222", zorder=3)
    r2 = spatial["vs_elevation"]["fit_r2"]["recharge_mm_yr"]
    ax[0].set_title("Recharge vs elevation", fontweight="bold")
    ax[0].set_xlabel("elevation (m)"); ax[0].set_ylabel("recharge (mm/yr)")
    ax[0].text(0.04, 0.92, f"linear fit r²={r2}", transform=ax[0].transAxes,
               fontsize=9, color="#b91c1c")
    ax[1].scatter(p, rech, c=cols, s=60, edgecolor="#222", zorder=3)
    ax[1].set_title("Recharge vs precip  (actual driver)", fontweight="bold")
    ax[1].set_xlabel("precip / forcing (mm/yr)"); ax[1].set_ylabel("recharge (mm/yr)")
    ax[2].scatter(e, frac, c=cols, s=60, edgecolor="#222", zorder=3)
    ax[2].set_title("Recharge fraction vs elevation", fontweight="bold")
    ax[2].set_xlabel("elevation (m)"); ax[2].set_ylabel("recharge fraction"); ax[2].set_ylim(0, 1)

    handles = [Line2D([0], [0], marker="o", ls="", mfc=cmap[b], mec="#222",
                      label=f"{b} mm/yr") for b in bins]
    ax[1].legend(handles=handles, title="forcing bin", frameon=False, fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.25)
    fig.suptitle("Naches ensemble — recharge is forcing/soil-controlled, "
                 "not a smooth elevation gradient", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n   ✓ figure: {out_path}")


def print_soil(soil):
    if not soil:
        print("\n(no soil attribution — need >=3 columns sharing a forcing bin with soil data)")
        return
    print("\n" + "=" * 84)
    print(f"SOIL CONTROL  (forcing held at {soil['forcing_held_mm_yr']} mm/yr across "
          f"{soil['n_columns']} columns — spread is soil-driven)")
    print("=" * 84)
    print(f"{'column':<9}{'top texture':>14}{'clay_max%':>11}{'ksat_min':>11}"
          f"{'recharge':>11}{'runoff':>9}")
    print(f"{'':9}{'':>14}{'':>11}{'µm/s':>11}{'mm/yr':>11}{'mm/yr':>9}")
    print("-" * 84)
    for r in soil["by_recharge"]:
        print(f"{r['case_name']:<9}{str(r['texture_top']):>14}{_f(r['clay_max_pct'], 1):>11}"
              f"{_f(r['ksat_min_ums'], 1):>11}{_f(r['recharge_mm_yr']):>11}{_f(r['runoff_mm_yr']):>9}")
    print("-" * 84)
    sc = soil["soil_correlation"]
    print(f"recharge vs clay_max: r={sc['recharge_vs_clay_max']}   "
          f"vs ksat_min: r={sc['recharge_vs_ksat_min']}   "
          f"runoff vs clay_max: r={sc['runoff_vs_clay_max']}")
    print(f"strongest soil predictor of recharge: {soil['strongest_predictor']}")
    print("=" * 84)


def plot_soil(soil, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = soil["by_recharge"]
    clay = [r["clay_max_pct"] for r in rows]
    ksat = [r["ksat_min_ums"] for r in rows]
    rech = [r["recharge_mm_yr"] for r in rows]
    sc = soil["soil_correlation"]

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.8))
    ax[0].scatter(clay, rech, c="#8856a7", s=65, edgecolor="#222", zorder=3)
    ax[0].set_xlabel("max clay %  (impeding layer)"); ax[0].set_ylabel("recharge (mm/yr)")
    ax[0].set_title(f"Recharge vs clay  (r={sc['recharge_vs_clay_max']})", fontweight="bold")
    ax[1].scatter(ksat, rech, c="#2c7fb8", s=65, edgecolor="#222", zorder=3)
    ax[1].set_xlabel("min Ksat µm/s  (drainage bottleneck)"); ax[1].set_ylabel("recharge (mm/yr)")
    ax[1].set_title(f"Recharge vs Ksat  (r={sc['recharge_vs_ksat_min']})", fontweight="bold")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.25)
    fig.suptitle(f"Soil control on recharge — forcing held at "
                 f"{soil['forcing_held_mm_yr']} mm/yr", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ soil figure: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Analyze a completed ELM run (read-only)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cases-file", default="phase3_cases.json",
                    help="JSON list of case dirs, relative to run-dir")
    ap.add_argument("--plan-file", default="phase3_plan.json",
                    help="executable plan.json, relative to run-dir")
    ap.add_argument("--plot", action="store_true", help="also save the elevation-gradient figure")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = run_dir / "04_analysis"
    exps = build_experiments(run_dir, args.cases_file, args.plan_file)
    print(f"analyzing {len(exps)} column(s) from {run_dir}")

    az = ELMResultsAnalyzer(exps, str(analysis_dir))
    az.extract_all()
    spatial = az._compute_spatial_summary()
    soil = az._compute_soil_attribution()
    print_summary(az.results, spatial)
    print_soil(soil)

    if args.plot and spatial:
        plot_gradient(spatial, analysis_dir / "elevation_gradient.png")
    if args.plot and soil:
        plot_soil(soil, analysis_dir / "soil_control.png")
    print(f"\nanalysis written to {analysis_dir}/")


if __name__ == "__main__":
    main()
