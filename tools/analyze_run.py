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


def plot_gradient(spatial, out_path, basin="Spatial ensemble"):
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
    fig.suptitle(f"{basin} — recharge is forcing/soil-controlled, "
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
    frac = [r["recharge_fraction"] for r in rows]
    sc = soil["soil_correlation"]
    has_ksat = any(k is not None for k in ksat)

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.8))
    ax[0].scatter(clay, rech, c="#8856a7", s=65, edgecolor="#222", zorder=3)
    ax[0].set_xlabel("max clay %  (impeding layer)"); ax[0].set_ylabel("recharge (mm/yr)")
    ax[0].set_title(f"Recharge vs clay  (r={sc['recharge_vs_clay_max']})", fontweight="bold")
    if has_ksat:                                   # spatial run: Ksat available
        ax[1].scatter(ksat, rech, c="#2c7fb8", s=65, edgecolor="#222", zorder=3)
        ax[1].set_xlabel("min Ksat µm/s  (drainage bottleneck)"); ax[1].set_ylabel("recharge (mm/yr)")
        ax[1].set_title(f"Recharge vs Ksat  (r={sc['recharge_vs_ksat_min']})", fontweight="bold")
    else:                                          # synthetic sweep: show the partitioning flip
        ax[1].scatter(clay, frac, c="#31a354", s=65, edgecolor="#222", zorder=3)
        ax[1].set_xlabel("max clay %"); ax[1].set_ylabel("recharge fraction"); ax[1].set_ylim(0, 1)
        ax[1].set_title("Recharge fraction vs clay", fontweight="bold")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.25)
    fig.suptitle(f"Soil control on recharge — forcing held at "
                 f"{soil['forcing_held_mm_yr']} mm/yr", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ soil figure: {out_path}")


def plot_budget(results, out_path):
    """Per-column water-budget closure: stacked export terms (runoff, drainage,
    ET, Δstorage) vs the precipitation input, columns ordered by elevation.
    Recharge sits INSIDE Δstorage here (the aquifer is part of TWS) — a large
    Δstorage bar is the visible no-spin-up signature."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ok = sorted((r for r in results.values()
                 if r["status"] == "ok" and r["metrics"].get("water_budget")),
                key=lambda r: (r.get("elevation_m") or 0))
    if not ok:
        return False
    names = [f"{r['case_name']}\n{r.get('elevation_m') or 0:.0f} m" for r in ok]
    wb = [r["metrics"]["water_budget"] for r in ok]
    P = [r["metrics"].get("precip_total_mm_yr") or 0 for r in ok]
    parts = [("runoff", "runoff_mm_yr", "#d95f0e"),
             ("drainage", "drainage_mm_yr", "#fdae6b"),
             ("ET", "et_mm_yr", "#31a354"),
             ("Δstorage", "storage_change_mm", "#9ecae1")]

    x = np.arange(len(ok))
    fig, ax = plt.subplots(figsize=(12.8, 4.6))
    bottom = np.zeros(len(ok))
    for label, key, color in parts:
        v = np.array([max(b.get(key) or 0, 0) for b in wb], float)
        ax.bar(x, v, bottom=bottom, color=color, edgecolor="#222",
               width=.72, label=label)
        bottom += v
    ax.scatter(x, P, marker="_", s=420, color="k", lw=2.2, zorder=4,
               label="precipitation (P)")
    rech = [b.get("recharge_mm_yr") for b in wb]
    for xi, (rv, b) in enumerate(zip(rech, bottom)):
        if rv is not None:
            ax.text(xi, b + 28, f"R {rv:.0f}", ha="center", fontsize=7.5, color="#08519c")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("mm / yr")
    ax.set_title("Water budget per column — where the precipitation goes  "
                 "(R = recharge, contained in Δstorage; big Δstorage = no spin-up)",
                 fontweight="bold", fontsize=12)
    ax.legend(frameon=False, ncol=5, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ budget figure: {out_path}")
    return True


def plot_relations(results, out_path):
    """The full relationship grid: every response (runoff, infiltration, ET,
    recharge) scattered against every driver (elevation, precip, clay, Ksat),
    Pearson r annotated — the visual companion to the driver matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ok = [r for r in results.values() if r["status"] == "ok"]
    if len(ok) < 3:
        return False
    drivers = [("elevation (m)", lambda r: r.get("elevation_m")),
               ("precip P (mm/yr)", lambda r: r["metrics"].get("precip_total_mm_yr")
                                              or r["metrics"].get("precip_mm_yr")),
               ("max clay (%)", lambda r: (r.get("soil") or {}).get("clay_max_pct")),
               ("min Ksat (µm/s)", lambda r: (r.get("soil") or {}).get("ksat_min_ums"))]
    responses = [("runoff (mm/yr)", lambda m: m.get("annual_runoff_mm_yr")),
                 ("infiltration (mm/yr)", lambda m: (m.get("water_budget") or {}).get("infiltration_mm_yr")),
                 ("ET (mm/yr)", lambda m: (m.get("water_budget") or {}).get("et_mm_yr")),
                 ("recharge (mm/yr)", lambda m: m.get("annual_recharge_mm_yr"))]
    # keep only responses the run actually has (old runs lack budget terms)
    responses = [(lab, g) for lab, g in responses
                 if any(g(r["metrics"]) is not None for r in ok)]

    nr, nc = len(responses), len(drivers)
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.5 * nr),
                             sharey="row", squeeze=False)
    for i, (rlab, rget) in enumerate(responses):
        for j, (dlab, dget) in enumerate(drivers):
            a = axes[i][j]
            x = np.array([dget(r) if dget(r) is not None else np.nan for r in ok], float)
            y = np.array([rget(r["metrics"]) if rget(r["metrics"]) is not None
                          else np.nan for r in ok], float)
            m = ~np.isnan(x) & ~np.isnan(y)
            a.scatter(x[m], y[m], s=26, color="#2c7fb8", edgecolor="#222",
                      linewidth=.4, alpha=.85, zorder=3)
            if m.sum() >= 3 and np.ptp(x[m]) > 1e-9 and np.ptp(y[m]) > 1e-9:
                rr = float(np.corrcoef(x[m], y[m])[0, 1])
                a.text(.04, .88, f"r={rr:+.2f}", transform=a.transAxes,
                       fontsize=9, fontweight="bold",
                       color="#b91c1c" if abs(rr) >= .5 else "#64748b")
            if i == nr - 1:
                a.set_xlabel(dlab, fontsize=9)
            if j == 0:
                a.set_ylabel(rlab, fontsize=9)
            a.tick_params(labelsize=7.5)
            a.grid(alpha=.25)
            a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Driver → response relations (each point = one column)",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ relations figure: {out_path}")
    return True


def plot_wtd(results, run_dir, out_path):
    """Per-column water table: model ZWT initial vs final vs the Fan (2013)
    equilibrium WTD at the same point. Exposes the cold-start problem — every
    column begins at ELM's default (~8.8 m) regardless of the real water table,
    and barely moves in a 1-yr run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fan = {}
    cj = run_dir / "columns.json"
    if cj.exists():
        cols = json.loads(cj.read_text())
        cols = cols.get("columns", cols) if isinstance(cols, dict) else cols
        fan = {c["id"]: c.get("fan_wtd_m") for c in cols}

    ok = sorted((r for r in results.values() if r["status"] == "ok"),
                key=lambda r: (r.get("elevation_m") or 0))
    zwt = [(r["variables"].get("ZWT") or {}) for r in ok]
    if not any(z.get("first_m") is not None for z in zwt):
        return False
    x = np.arange(len(ok))
    names = [f"{r['case_name']}\n{r.get('elevation_m') or 0:.0f} m" for r in ok]

    fig, ax = plt.subplots(figsize=(12.6, 4.2))
    f = np.array([fan.get(r["case_name"]) or np.nan for r in ok], float)
    ax.scatter(x, np.clip(f, .05, None), marker="o", s=48, color="#8856a7",
               edgecolor="#222", label="Fan 2013 equilibrium WTD", zorder=3)
    firsts = [z.get("first_m") for z in zwt if z.get("first_m") is not None]
    uniform = firsts and (max(firsts) - min(firsts) < 0.05)
    init_lab = "model ZWT — initial (cold start)" if uniform else \
               "model ZWT — initial (Fan warm start)"
    ax.scatter(x, [z.get("first_m") for z in zwt], marker="s", s=40,
               color="#d95f0e", label=init_lab, zorder=4)
    ax.scatter(x, [z.get("last_m") for z in zwt], marker="x", s=48,
               color="#2c7fb8", label="model ZWT — end of run", zorder=5)
    ax.set_yscale("log"); ax.invert_yaxis()
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("water-table depth (m, log)")
    ax.set_title("Water table per column — every column cold-starts at the SAME default "
                 "and barely moves in 1 yr; the real (Fan) WTD varies by orders of magnitude"
                 if uniform else
                 "Water table per column — initialized from the Fan prior (warm start), "
                 "then relaxing toward ELM's own equilibrium",
                 fontweight="bold", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ WTD figure: {out_path}")
    return True


def print_matrix(dm):
    if not dm:
        return
    drivers = ["elevation_m", "precip_mm_yr", "clay_max_pct", "ksat_min_ums"]
    print("\n" + "=" * 76)
    print(f"DRIVER × RESPONSE  (Pearson r, all {dm['n_columns']} columns)")
    print("=" * 76)
    print(f"{'response':<20}" + "".join(f"{d.split('_')[0]:>13}" for d in drivers))
    print("-" * 76)
    for resp, row in dm["pearson_r"].items():
        cells = "".join(f"{row.get(d):>13.2f}" if row.get(d) is not None else f"{'—':>13}"
                        for d in drivers)
        print(f"{resp:<20}{cells}")
    print("-" * 76)
    print(f"note: {dm['note']}")
    print("=" * 76)


def main():
    ap = argparse.ArgumentParser(description="Analyze a completed ELM run (read-only)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cases-file", default="phase3_cases.json",
                    help="JSON list of case dirs, relative to run-dir")
    ap.add_argument("--plan-file", default="phase3_plan.json",
                    help="executable plan.json, relative to run-dir")
    ap.add_argument("--plot", action="store_true", help="also save the elevation-gradient figure")
    ap.add_argument("--last-year", action="store_true",
                    help="analyze only the final full simulated year (spin-up runs)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = run_dir / "04_analysis"
    exps = build_experiments(run_dir, args.cases_file, args.plan_file)
    print(f"analyzing {len(exps)} column(s) from {run_dir}")

    az = ELMResultsAnalyzer(exps, str(analysis_dir), last_year_only=args.last_year)
    az.extract_all()
    spatial = az._compute_spatial_summary()
    soil = az._compute_soil_attribution()
    print_summary(az.results, spatial)
    print_soil(soil)
    print_matrix(az._compute_driver_matrix())

    if args.plot and spatial:
        basin = "Spatial ensemble"
        bf = run_dir / "reception_brief.json"
        if bf.exists():
            basin = (json.loads(bf.read_text()).get("domain") or {}).get("name") or basin
        plot_gradient(spatial, analysis_dir / "elevation_gradient.png", basin=basin)
    if args.plot and soil:
        plot_soil(soil, analysis_dir / "soil_control.png")
    if args.plot:
        plot_budget(az.results, analysis_dir / "water_budget.png")
        plot_relations(az.results, analysis_dir / "driver_response.png")
        plot_wtd(az.results, run_dir, analysis_dir / "wtd_columns.png")
    print(f"\nanalysis written to {analysis_dir}/")


if __name__ == "__main__":
    main()
