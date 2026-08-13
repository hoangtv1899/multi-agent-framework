#!/usr/bin/env python3
"""
Analyze a completed ELM run — read the per-column history files, extract the
hydrology, and summarize how recharge / runoff / soil moisture / water-table
depth vary across the ensemble (spatially, by elevation).

Wraps core.ELMResultsAnalyzer over a pipeline run dir that holds:
    cases.json      the case directories that were run
    run_plan.json   per-column metadata (forcing, lat/lon, years)
    columns.json        per-column elevation (optional, for the gradient)

Prints the ensemble summary (+ figures with --plot) into
<run-dir>/04_analysis/. NOTHING is executed — read-only over existing output.

Run from the project root with the analysis env:
    source /qfs/people/tran289/IDEAS/env_compy.sh
    python3 tools/analyze_run.py --run-dir workflow_outputs/pipeline_XXXX --plot
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")          # quiet xarray/netCDF futurewarnings
_HERE = Path(__file__).resolve().parent                    # scripts/
_FRAMEWORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HERE.parent / "src"))              # ELM modules
sys.path.insert(0, str(_FRAMEWORK / "src"))                # framework
from elm_results_analyzer import ELMResultsAnalyzer
from agents.analysis import step2_derive as _drv


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


def _resolve(run_dir, name, legacy):
    """Accept the manager's filenames, falling back to the older phase3_* ones.

    ELMExpManager writes cases.json / run_plan.json; this CLI defaulted to
    phase3_* and so failed with a bare FileNotFoundError on every pipeline run.
    """
    p = run_dir / name
    return p if p.exists() else run_dir / legacy


def build_experiments(run_dir: Path, cases_file="cases.json",
                      plan_file="run_plan.json"):
    cases = json.load(open(_resolve(run_dir, cases_file, "phase3_cases.json")))
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
        wb = m.get("water_budget") or {}
        print(f"{r['case_name']:<9}{_f(r.get('elevation_m'), 0):>8}{_f(m.get('precip_mm_yr'), 0):>9}"
              f"{_f(wb.get('recharge_mm_yr')):>11}{_f(wb.get('runoff_mm_yr')):>9}"
              f"{_f(wb.get('recharge_frac_of_P'), 3):>11}{_f(m.get('water_table_depth_m'), 2):>8}")
    print("-" * 84)
    # THE SPATIAL BLOCK IS step2_derive.spatial_summary's NOW. The version this
    # printed came from a private method on ELMResultsAnalyzer that also fitted
    # recharge against elevation and wrote its own `interpretation` sentences —
    # an extractor drawing conclusions. The fit is not lost: the elevation and
    # precip correlations print immediately below, out of the driver matrix,
    # for every response rather than recharge alone.
    if spatial:
        fo = spatial["forcing"]
        print(f"forcing: {fo['n_forcing_bins']} distinct precip value(s) "
              f"{fo['precip_mm_yr_distinct']} mm/yr  (elevation-resolved: {fo['elevation_resolved']})")
        er = spatial.get("elevation_range_m")
        if er:
            print(f"elevation: {er[0]}-{er[1]} m over {spatial['n_columns']} "
                  f"column(s) in {spatial['n_bands']} band(s)")
        for band, b in (spatial.get("by_band") or {}).items():
            mean = b.get("mean") or {}
            print(f"  band {band:<10} n={b['n_columns']:<3} "
                  f"elev {b.get('elevation_range_m')}  "
                  + "  ".join(f"{k.replace('_mm_yr',''):}={_f(v)}"
                              for k, v in mean.items()))
    else:
        print("(single location or no elevation — no spatial summary)")
    print("=" * 84)


def print_forcing_groups(results):
    """Columns that share a forcing cell — the soil control, read off the rows.

    NOT a precomputed attribution (deleted 2026-08-13). Columns inside one
    NLDAS-2 cell got the same rain, so a difference between them is soil or
    terrain. Printed as the pairs and their numbers; no correlation is offered,
    because two columns is not a sample and the old function computed one
    anyway.
    """
    import collections
    ok = [r for r in results.values() if r.get("status") == "ok"]
    groups = collections.defaultdict(list)
    for r in ok:
        cell = r.get("forcing_cell")
        if cell is not None:
            groups[tuple(cell)].append(r)
    shared = {c: g for c, g in groups.items()
              if len(g) > 1 and len({(x.get("lat"), x.get("lon")) for x in g}) > 1}
    if not shared:
        print("\n(no soil control — no two columns at different places share a "
              "forcing cell)")
        return
    print(f"\nSOIL CONTROL — same forcing, different soil "
          f"({len(shared)} group(s))")
    for cell, g in sorted(shared.items()):
        print(f"  forcing cell {cell}:")
        for r in g:
            ss = r.get("soil_summary") or {}
            wb = (r.get("metrics") or {}).get("water_budget") or {}
            clay = ss.get("clay_pct")
            print(f"    {r['case_name']:<9} {str(ss.get('texture_top')):<12} "
                  f"clay {clay[0] if clay else '?'}-{clay[1] if clay else '?'}%   "
                  f"recharge {wb.get('recharge_mm_yr')}  "
                  f"runoff {wb.get('runoff_mm_yr')}  "
                  f"drainage {wb.get('drainage_mm_yr')} mm/yr")


def plot_soil(soil, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = soil["by_recharge"]
    clay = [r["clay_max_pct"] for r in rows]
    ksat = [r["ksat_min_ums"] for r in rows]
    rech = [r["recharge_mm_yr"] for r in rows]
    frac = [r.get("recharge_frac_of_P") for r in rows]
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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"   ✓ soil figure: {out_path}")


def plot_partitioning(results, out_path):
    """Where precipitation goes, as FRACTIONS of P — the partitioning question.

    Replaces the old absolute-mm stack, which had two defects:

      * it clamped Δstorage with max(v, 0), so a column RELEASING storage showed
        no storage term at all and its export stack simply exceeded P with
        nothing to explain it (col_12: 3750 mm of exports against 1870 mm of
        precipitation).
      * absolute mm/yr across columns whose P spans 472-1575 mm/yr cannot be
        compared by eye, which is what "how does P partition" asks for.

    Here every column is normalised by its own P, a 100% reference line marks
    the input, and storage is SIGNED: a stack short of 100% put water into
    storage, a stack past it took water out. Recharge is annotated rather than
    stacked — it is an internal flux (soil -> aquifer), not a sink parallel to
    drainage (aquifer -> stream), and stacking it double-counts.
    """
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
    P = np.array([(r["metrics"].get("precip_mm_yr") or np.nan) for r in ok], float)

    def frac(key):
        return np.array([(b.get(key) or 0.0) for b in wb], float) / np.where(P > 0, P, np.nan)

    parts = [("runoff", "runoff_mm_yr", "#d95f0e"),
             ("ET", "et_mm_yr", "#31a354"),
             ("drainage", "drainage_mm_yr", "#fdae6b")]
    x = np.arange(len(ok))
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.0),
                           gridspec_kw={"width_ratios": [1.35, 1]})

    # ── panel 1: fractions of P ──
    bottom = np.zeros(len(ok))
    for label, key, color in parts:
        v = frac(key)
        ax[0].bar(x, v, bottom=bottom, color=color, edgecolor="#222",
                  width=.72, label=label)
        bottom += np.nan_to_num(v)
    ax[0].axhline(1.0, color="k", ls="--", lw=1.4, zorder=5)
    ax[0].text(len(ok) - .35, 1.0, " 100% of P", fontsize=8.5, color="k",
               va="center", ha="left")
    ax[0].set_xlim(-.8, len(ok) + .6)

    # storage, signed and visible in both directions
    gain = np.clip(1.0 - bottom, 0, None)
    rel  = np.clip(bottom - 1.0, 0, None)
    ax[0].bar(x, gain, bottom=bottom, color="#9ecae1", edgecolor="#222",
              width=.72, label="→ into storage")
    ax[0].bar(x, -rel, bottom=1.0 + rel, color="none", edgecolor="#c00",
              width=.72, hatch="///", lw=1.2, label="← released from storage")

    rf = frac("recharge_mm_yr")
    for xi, (v, top) in enumerate(zip(rf, np.maximum(bottom, 1.0))):
        if np.isfinite(v):
            ax[0].text(xi, top + .04, f"R {v:.2f}", ha="center", fontsize=7.5,
                       color="#08519c")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=7.5)
    ax[0].set_ylabel("fraction of precipitation")
    ax[0].legend(frameon=False, ncol=5, fontsize=8.5, loc="upper left")
    ax[0].set_title("Partitioning — every column normalised by its OWN P.\n"
                    "R = recharge fraction (internal flux: soil → aquifer)",
                    fontweight="bold", fontsize=11)

    # ── panel 2: magnitudes, with storage signed ──
    w = .38
    ax[1].bar(x - w / 2, P, w, color="#4d4d4d", edgecolor="#222", label="P")
    exp = np.array([sum((b.get(k) or 0.0) for _, k, _ in parts) for b in wb], float)
    ax[1].bar(x + w / 2, exp, w, color="#fdae6b", edgecolor="#222",
              label="exports (runoff+ET+drainage)")
    ds = np.array([(b.get("storage_change_mm") or 0.0) for b in wb], float)
    ax[1].bar(x, ds, .72, color="#9ecae1", edgecolor="#222", alpha=.85,
              label="Δstorage (signed)")
    ax[1].axhline(0, color="#222", lw=.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([r["case_name"] for r in ok], fontsize=7, rotation=60,
                          ha="right")
    ax[1].set_ylabel("mm / yr")
    ax[1].legend(frameon=False, fontsize=8.5)
    ax[1].set_title("Magnitudes — exports above P means storage was drained\n"
                    "(the un-equilibrated signature; spin-up closes it)",
                    fontweight="bold", fontsize=11)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"   ✓ budget figure: {out_path}")
    return True


def plot_controls(results, out_path):
    """What controls the partitioning — FRACTIONS of P against each driver.

    Replaces the old 4x4 absolute-flux matrix and the separate elevation-gradient
    figure. Three deliberate choices:

      * Responses are FRACTIONS. "How does P partition" is a ratio question, and
        a fraction is comparable across columns whose P differs three-fold.
      * Elevation panels are COLOURED BY PRECIPITATION, so the orographic
        confound is visible rather than asserted. High columns are wet columns;
        an elevation correlation and a precipitation correlation are not
        independent claims, and no partial-correlation statistic on n=13 would
        be more honest than simply showing it.
      * A within-forcing-bin panel does the one piece of real attribution
        available: where two columns share a precipitation cell (12 km NLDAS
        quantises the forcing heavily) any difference between them CANNOT be
        forcing, so it is soil.

    Correlations are screening only at this sample size and the figure says so.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ok = [r for r in results.values() if r["status"] == "ok"]
    if len(ok) < 3:
        return False

    def P_of(r):
        m = r["metrics"]
        return m.get("precip_mm_yr")

    def fr(r, key):
        wb = r["metrics"].get("water_budget") or {}
        v, P = wb.get(key), P_of(r)
        return (v / P) if (v is not None and P) else np.nan

    responses = [("runoff / P", "runoff_mm_yr"),
                 ("ET / P", "et_mm_yr"),
                 ("recharge / P", "recharge_mm_yr")]
    drivers = [("elevation (m)", lambda r: r.get("elevation_m"), True),
               ("precip P (mm/yr)", P_of, False),
               ("max clay (%)", lambda r: (r.get("soil") or {}).get("clay_max_pct"), False)]

    nr, nc = len(responses), len(drivers)
    fig = plt.figure(figsize=(3.4 * nc + 4.6, 2.7 * nr))
    # dedicated narrow slot for the colourbar: letting matplotlib steal space
    # from the axes list put it on top of the within-bin panel's y-axis.
    gs = fig.add_gridspec(nr, nc + 2, width_ratios=[1] * nc + [.10, 1.25],
                          wspace=.34)
    pv = np.array([P_of(r) or np.nan for r in ok], float)
    norm = plt.Normalize(np.nanmin(pv), np.nanmax(pv))

    sc = None
    for i, (rlab, rkey) in enumerate(responses):
        y = np.array([fr(r, rkey) for r in ok], float)
        for j, (dlab, dget, colour_by_p) in enumerate(drivers):
            a = fig.add_subplot(gs[i, j])
            x = np.array([dget(r) if dget(r) is not None else np.nan for r in ok], float)
            m = ~np.isnan(x) & ~np.isnan(y)
            if colour_by_p:
                sc = a.scatter(x[m], y[m], c=pv[m], cmap="viridis", norm=norm,
                               s=42, edgecolor="#222", linewidth=.4, zorder=3)
            else:
                a.scatter(x[m], y[m], s=42, color="#2c7fb8", edgecolor="#222",
                          linewidth=.4, zorder=3)
            if m.sum() >= 3 and np.ptp(x[m]) > 1e-9 and np.ptp(y[m]) > 1e-9:
                rr = float(np.corrcoef(x[m], y[m])[0, 1])
                a.text(.03, .955, f"r={rr:+.2f}", transform=a.transAxes,
                       fontsize=8.5, fontweight="bold", color="#64748b",
                       va="top", bbox=dict(boxstyle="round,pad=.18", fc="white",
                                           ec="none", alpha=.75))
            if i == nr - 1:
                a.set_xlabel(dlab, fontsize=9)
            if j == 0:
                a.set_ylabel(rlab, fontsize=9)
            a.tick_params(labelsize=7.5)
            a.grid(alpha=.25)
            a.spines[["top", "right"]].set_visible(False)

    # ── within-forcing-bin attribution ──
    ab = fig.add_subplot(gs[:, nc + 1])
    bins = {}
    for r in ok:
        P = P_of(r)
        if P:
            bins.setdefault(round(P, 1), []).append(r)
    shared = {k: v for k, v in bins.items() if len(v) >= 2}
    if shared:
        yy, lbl = [], []
        for k in sorted(shared):
            grp = shared[k]
            for r in grp:
                yy.append((k, fr(r, "recharge_mm_yr"),
                           (r.get("soil") or {}).get("clay_max_pct"),
                           r["case_name"]))
            lbl.append(k)
        ks = sorted({q[0] for q in yy})
        for gi, k in enumerate(ks):
            grp = [q for q in yy if q[0] == k]
            xs = [gi] * len(grp)
            cl = [q[2] if q[2] is not None else np.nan for q in grp]
            ab.scatter(xs, [q[1] for q in grp], c=cl, cmap="copper_r",
                       s=70, edgecolor="#222", zorder=3)
            for q, xq in zip(grp, xs):
                ab.annotate(q[3].replace("col_", ""), (xq, q[1]),
                            textcoords="offset points", xytext=(7, -2),
                            fontsize=7, color="#444")
            if len(grp) >= 2:
                ab.plot([gi, gi], [min(q[1] for q in grp), max(q[1] for q in grp)],
                        color="#999", lw=1, zorder=1)
        ab.set_xticks(range(len(ks)))
        ab.set_xticklabels([f"{k:.0f}" for k in ks], fontsize=8, rotation=45)
        ab.set_xlabel("precipitation bin (mm/yr)", fontsize=9)
        ab.set_ylabel("recharge / P", fontsize=9)
        ab.set_title("Same forcing, different soil\n"
                     "spread within a bin is NOT forcing\n(colour = max clay %)",
                     fontweight="bold", fontsize=9.5)
    else:
        ab.axis("off")
        ab.text(.5, .5, "no two columns share\na precipitation bin",
                ha="center", va="center", color="#888", fontsize=9)
    # ticks on the RIGHT: this panel sits immediately beside the colourbar and
    # a left-hand axis draws straight over it.
    ab.yaxis.tick_right()
    ab.yaxis.set_label_position("right")
    ab.grid(alpha=.25)
    ab.spines[["top", "left"]].set_visible(False)

    if sc is not None:
        cax = fig.add_subplot(gs[:, nc])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label("precipitation (mm/yr)", fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)
    fig.suptitle(f"Controls on partitioning — each point is one column (n={len(ok)}). "
                 "Elevation panels are coloured by precipitation: high columns are "
                 "wet columns, so the two drivers are confounded.\n"
                 "Correlations are SCREENING ONLY at this sample size.",
                 fontweight="bold", fontsize=10.5, y=1.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"   ✓ controls figure: {out_path}")
    return True


def plot_spatial(results, run_dir, out_path, annotate=False):
    """The same partitioning quantities, MAPPED over the watershed.

    Every other analysis figure is aspatial — scatter against elevation, bars
    per column, series through time. None of them can answer "is the
    low-recharge cluster spatially coherent, or scattered?", and the difference
    matters: coherent structure points at orography or geology, scatter points
    at soil heterogeneity. sampling_design.png shows where we SAMPLED; this
    shows what we FOUND.

    Deliberately NOT interpolated. Thirteen points over ~2900 km2 is a sample,
    not a field; kriging or IDW would manufacture spatial structure out of 13
    numbers and render it authoritatively. Markers only.
    """
    import json as _json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ok = [r for r in results.values() if r["status"] == "ok"
          and r.get("lat") is not None and r.get("lon") is not None]
    if len(ok) < 2:
        return False
    cj = Path(run_dir) / "columns.json"
    meta = _json.loads(cj.read_text()) if cj.exists() else {}
    boundary = meta.get("boundary") or []
    grid = meta.get("grid") or []

    def P_of(r):
        m = r["metrics"]
        return m.get("precip_mm_yr")

    def frac(r, key):
        wb = r["metrics"].get("water_budget") or {}
        v, P = wb.get(key), P_of(r)
        return (v / P) if (v is not None and P) else np.nan

    panels = [
        ("recharge / P", lambda r: frac(r, "recharge_mm_yr"), "viridis"),
        ("runoff / P",   lambda r: frac(r, "runoff_mm_yr"),   "Oranges"),
        ("ET / P",       lambda r: frac(r, "et_mm_yr"),       "Greens"),
        ("peak SWE (mm)", lambda r: r["metrics"].get("peak_swe_modelled_mm"), "Blues"),
    ]
    lat = np.array([r["lat"] for r in ok], float)
    lon = np.array([r["lon"] for r in ok], float)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
    for ax, (label, get, cmap) in zip(axes.ravel(), panels):
        # terrain context from the DEM points the expander already sampled
        if grid:
            gx = [g["lon"] for g in grid]
            gy = [g["lat"] for g in grid]
            ge = [g.get("elevation_m") for g in grid]
            ax.scatter(gx, gy, c=ge, cmap="Greys", s=14, alpha=.45,
                       linewidth=0, zorder=1)
        for ring in boundary:
            ax.plot([q[0] for q in ring], [q[1] for q in ring],
                    color="#123", lw=1.3, zorder=2)

        v = np.array([get(r) if get(r) is not None else np.nan for r in ok], float)
        m = ~np.isnan(v)
        sc = ax.scatter(lon[m], lat[m], c=v[m], cmap=cmap, s=150,
                        edgecolor="#111", linewidth=.7, zorder=4)
        if m.sum():
            cb = fig.colorbar(sc, ax=ax, fraction=.046, pad=.02)
            cb.ax.tick_params(labelsize=7.5)
        for xi, yi, r in zip(lon, lat, ok):
            ax.annotate(r["case_name"].replace("col_", ""), (xi, yi),
                        textcoords="offset points", xytext=(8, 4), fontsize=6.5,
                        color="#333", zorder=5)
        ax.set_title(label, fontweight="bold", fontsize=10.5)
        ax.set_xlabel("longitude", fontsize=8.5)
        ax.set_ylabel("latitude", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.set_aspect(1.0 / max(np.cos(np.radians(float(np.nanmean(lat)))), 1e-6))
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)

    title = f"Spatial distribution — {len(ok)} sampled columns"
    if annotate:
        title += ("\nMARKERS ONLY: 13 samples over ~2900 km², not an interpolated "
                  "field. Grey = sampled terrain; navy = watershed boundary.")
    fig.suptitle(title, fontweight="bold", fontsize=12, y=.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"   ✓ spatial figure: {out_path}")
    return True


def plot_wtd(results, run_dir, out_path):
    """Per-column water table: model ZWT initial vs final vs the water-table
    prior at the same point, whichever prior the run recorded
    (`wtd_prior_source`). Exposes the cold-start problem — every column begins
    at ELM's default (~8.8 m) regardless of the prior, and barely moves in a
    1-yr run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fan, src = {}, "prior"
    cj = run_dir / "columns.json"
    if cj.exists():
        cols = json.loads(cj.read_text())
        cols = cols.get("columns", cols) if isinstance(cols, dict) else cols
        fan = {c["id"]: c.get("wtd_prior_m") for c in cols}
        src = next((c.get("wtd_prior_source") for c in cols
                    if c.get("wtd_prior_source")), "prior")

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
               edgecolor="#222", label=f"water-table prior ({src})", zorder=3)
    firsts = [z.get("first_m") for z in zwt if z.get("first_m") is not None]
    uniform = firsts and (max(firsts) - min(firsts) < 0.05)
    init_lab = "model ZWT — initial (cold start)" if uniform else \
               "model ZWT — initial (warm start)"
    ax.scatter(x, [z.get("first_m") for z in zwt], marker="s", s=40,
               color="#d95f0e", label=init_lab, zorder=4)
    ax.scatter(x, [z.get("last_m") for z in zwt], marker="x", s=48,
               color="#2c7fb8", label="model ZWT — end of run", zorder=5)
    ax.set_yscale("log"); ax.invert_yaxis()
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("water-table depth (m, log)")
    ax.set_title("Water table per column — every column cold-starts at the SAME default "
                 "and barely moves in 1 yr; the prior WTD varies by orders of magnitude"
                 if uniform else
                 "Water table per column — initialized from the warm start, "
                 "then relaxing toward ELM's own equilibrium",
                 fontweight="bold", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
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
    ap.add_argument("--cases-file", default="cases.json",
                    help="JSON list of case dirs, relative to run-dir")
    ap.add_argument("--plan-file", default="run_plan.json",
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
    # THE ENSEMBLE VIEWS ARE THE ANALYZER'S, not the extractor's. This script
    # used to reach through the class into az._compute_spatial_summary() and
    # az._compute_driver_matrix() — private methods that were a second copy of
    # step2_derive's functions. Deleted 2026-08-13; analyze_agentic.py and
    # interpret_run.py already read them from here.
    rows = list(az.results.values())
    spatial = _drv.spatial_summary(rows)
    print_summary(az.results, spatial)
    print_forcing_groups(az.results)
    print_matrix(_drv.driver_matrix(rows))

    if args.plot:
        # partitioning + controls replace the old water_budget, driver_response
        # and elevation_gradient figures (see their docstrings for why).
        plot_partitioning(az.results, analysis_dir / "partitioning.png")
        plot_controls(az.results, analysis_dir / "controls.png")
        plot_spatial(az.results, run_dir, analysis_dir / "spatial.png")
        plot_wtd(az.results, run_dir, analysis_dir / "wtd_columns.png")
        if soil:
            plot_soil(soil, analysis_dir / "soil_control.png")
    print(f"\nanalysis written to {analysis_dir}/")


if __name__ == "__main__":
    main()
