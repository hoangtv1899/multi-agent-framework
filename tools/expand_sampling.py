#!/usr/bin/env python3
"""
Tier-2 expander: planner sampling STRATEGY -> concrete column points.

Deterministic geospatial expansion (no LLM, no invented coordinates). Samples
the real DEM via the terrain MCP across the domain bbox, stratifies by elevation
band, allocates the planner's N columns proportionally to occupied area, picks
spatially-spread points per band, and enriches each with Fan equilibrium WTD +
point soil texture. NOTHING is executed.

Operates on a pipeline run dir (reads reception_brief.json for the bbox and
plan.json for N / band count), or standalone via --bbox/--n/--bands.

Run from the project root with the MCP runtime env:
    source /qfs/people/tran289/IDEAS/env_compy.sh
    python3 tools/expand_sampling.py --run-dir workflow_outputs/pipeline_XXXX
    python3 tools/expand_sampling.py --bbox -121.52,46.46,-120.51,47.14 --n 12 --bands 4
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
from core.mcp_manager import MCPManager


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC HELPERS (pure, unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def _make_bands(elevs, n_bands):
    """Equal-interval elevation bands as (lo, hi) pairs."""
    lo, hi = min(elevs), max(elevs)
    if hi <= lo or n_bands <= 1:
        return [(lo, hi)]
    step = (hi - lo) / n_bands
    return [(lo + i * step, hi if i == n_bands - 1 else lo + (i + 1) * step)
            for i in range(n_bands)]


def _assign_band(e, bands):
    """Index of the band containing elevation e (last band inclusive on hi)."""
    for i, (lo, hi) in enumerate(bands):
        last = (i == len(bands) - 1)
        if (lo <= e <= hi) if last else (lo <= e < hi):
            return i
    return len(bands) - 1


def _allocate(counts, n_total):
    """Distribute n_total columns across bands ~proportional to point counts,
    with at least 1 per occupied band (unless n_total is smaller than the number
    of occupied bands, in which case the largest bands win)."""
    nonempty = [i for i, c in enumerate(counts) if c > 0]
    alloc = [0] * len(counts)
    if not nonempty:
        return alloc
    if n_total <= len(nonempty):
        for i in sorted(nonempty, key=lambda i: counts[i], reverse=True)[:n_total]:
            alloc[i] = 1
        return alloc

    total = sum(counts)
    raw = [(n_total * counts[i] / total) if counts[i] > 0 else 0
           for i in range(len(counts))]
    for i in nonempty:
        alloc[i] = max(1, round(raw[i]))
    # reconcile to exactly n_total
    while sum(alloc) > n_total:
        cand = [i for i in nonempty if alloc[i] > 1]
        if not cand:
            break
        alloc[max(cand, key=lambda i: alloc[i])] -= 1
    while sum(alloc) < n_total:
        alloc[max(nonempty, key=lambda i: raw[i] - alloc[i])] += 1
    return alloc


def _farthest_point_select(pts, k):
    """Greedy farthest-point sampling for spatial spread within a band.

    Array-wise: one running array of "distance to the nearest chosen point",
    updated with a single minimum against the newest pick. The scalar form
    recomputed every candidate against every chosen point on each step
    (O(k^2 n) dict lookups); this is O(k n) in numpy and gives identical picks.
    """
    import numpy as np
    if k >= len(pts):
        return list(pts)
    if k <= 0:
        return []
    lat = np.fromiter((p["lat"] for p in pts), float, len(pts))
    lon = np.fromiter((p["lon"] for p in pts), float, len(pts))

    first = int(np.argmin((lat - lat.mean()) ** 2 + (lon - lon.mean()) ** 2))
    picks = [first]
    d2 = (lat - lat[first]) ** 2 + (lon - lon[first]) ** 2
    while len(picks) < k:
        nxt = int(np.argmax(d2))
        picks.append(nxt)
        np.minimum(d2, (lat - lat[nxt]) ** 2 + (lon - lon[nxt]) ** 2, out=d2)
    return [pts[i] for i in picks]


# ─────────────────────────────────────────────────────────────────────────────
# EXPANSION (uses the MCP servers)
# ─────────────────────────────────────────────────────────────────────────────

def _clip_to_polygon(pts, rings):
    """Keep only points inside the watershed polygon (largest ring). Falls back
    to the original list if shapely/polygon is unavailable or clips everything."""
    try:
        from shapely.geometry import Polygon, Point
    except Exception:
        return pts
    polys = []
    for r in rings or []:
        if len(r) >= 4:
            try:
                polys.append(Polygon(r))
            except Exception:
                pass
    if not polys:
        return pts
    poly = max(polys, key=lambda p: p.area)
    # One prepared-geometry pass over all points instead of a Python-level
    # contains() per point: shapely builds the edge index once and vectorises
    # the test. Identical predicate, same points kept.
    try:
        import numpy as np
        from shapely import points as _shp_points, contains as _shp_contains
        arr = _shp_points(np.fromiter((q["lon"] for q in pts), float, len(pts)),
                          np.fromiter((q["lat"] for q in pts), float, len(pts)))
        keep = np.asarray(_shp_contains(poly, arr), dtype=bool)
        inside = [q for q, k in zip(pts, keep) if k]
    except Exception:
        # shapely < 2 has no vectorised API; the scalar predicate is the same.
        inside = [q for q in pts if poly.contains(Point(q["lon"], q["lat"]))]
    return inside or pts          # never drop everything on a bad clip


def expand(clients, bbox, n_total, n_bands, grid_n=120, do_soil=True, boundary=None):
    terr = clients["terrain"]
    fan = clients.get("fan_wtd")
    geo = clients.get("geology")

    grid = terr.call_tool_json("sample_elevation_grid", {**bbox, "n": grid_n}) or {}
    pts = [p for p in grid.get("points", []) if p.get("elevation_m") is not None]
    if boundary:                  # clip the rectangular bbox sample to the real basin
        n0 = len(pts)
        pts = _clip_to_polygon(pts, boundary)
        print(f"  clipped DEM grid -> {len(pts)}/{n0} points inside the watershed")
    if not pts:
        return {"error": "no elevation points returned for bbox", "bbox": bbox}

    bands = _make_bands([p["elevation_m"] for p in pts], n_bands)
    by_band = {i: [] for i in range(len(bands))}
    for p in pts:
        by_band[_assign_band(p["elevation_m"], bands)].append(p)
    counts = [len(by_band[i]) for i in range(len(bands))]
    alloc = _allocate(counts, n_total)

    columns, cid = [], 1
    for i in range(len(bands)):
        for p in _farthest_point_select(by_band[i], alloc[i]):
            col = {"id": f"col_{cid:02d}",
                   "lat": round(p["lat"], 5), "lon": round(p["lon"], 5),
                   "elevation_m": p["elevation_m"], "band": i + 1,
                   "band_range_m": [round(bands[i][0]), round(bands[i][1])]}
            columns.append(col)
            cid += 1

    # Enrichment is batched, not per column. Every MCP call is a fresh session
    # (HPC-safe by design), so asking per column paid a process spawn plus a
    # dataset open per column -- for Fan the open IS the cost. Two calls now
    # serve the whole design.
    lats = [c["lat"] for c in columns]
    lons = [c["lon"] for c in columns]
    if fan and columns:
        fr = fan.call_tool_json("get_fan_wtd_points",
                                {"lats": lats, "lons": lons}) or {}
        pts_out = fr.get("points") or []
        for col, entry in zip(columns, pts_out):
            col["fan_wtd_m"] = (entry or {}).get("depth_to_water_m")
        for col in columns[len(pts_out):]:
            col["fan_wtd_m"] = None
    if geo and do_soil and columns:
        sr = geo.call_tool_json("get_soil_profiles",
                                {"lats": lats, "lons": lons}) or {}
        profs = sr.get("profiles") or []
        for col, sp in zip(columns, profs):
            layers = (sp or {}).get("layers") or []
            col["soil_top_texture"] = layers[0].get("texture_class") if layers else None
            col["soil_layers"] = (sp or {}).get("num_layers")
            col["soil_profile"] = sp if layers else None
        for col in columns[len(profs):]:
            col["soil_top_texture"] = col["soil_layers"] = col["soil_profile"] = None

    return {"bbox": bbox, "n_requested": n_total, "n_columns": len(columns),
            "bands": [{"band": i + 1, "elev_lo_m": round(bands[i][0]),
                       "elev_hi_m": round(bands[i][1]), "grid_points": counts[i],
                       "allocated": alloc[i]} for i in range(len(bands))],
            "columns": columns,
            # full DEM sample kept for the hypsometry / map plot (not saved to columns.json)
            "grid": [{"lat": round(p["lat"], 5), "lon": round(p["lon"], 5),
                      "elevation_m": p["elevation_m"]} for p in pts]}


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING (--plot) — illustration of the sampling design
# ─────────────────────────────────────────────────────────────────────────────

# NOTE (Compy): this directory exists but is currently empty — the ctsmforc
# monthly Precip files are not yet staged (/compyfs/tran289/raw_nldas holds
# only raw hourly NLDAS_FORA files, a different format).
NLDAS_PRECIP = ("/compyfs/inputdata/atm/datm7/"
                "atm_forcing.datm7.NLDAS2.0.125d.v1/Precip")
# Second NLDAS-2 layout on Compy: the single-stream CLM files that
# DATM_MODE=CLMMOSARTTEST actually reads (complete 1979-2023, whereas the
# ...NLDAS2.0.125d.v1/Precip directory above is EMPTY here). Same 12 km grid
# and the same LATIXY/LONGXY + PRECTmms structure, so the reader below works
# on either -- only the filename differs. Preferring this one means the
# preview panel shows the forcing the run is actually driven by.
NLDAS_CLM_DIR = "/compyfs/inputdata/atm/datm7/NLDAS"
# Months read in parallel; see nldas_annual_precip. Six keeps enough seeks in
# flight to hide the latency without monopolising a login node.
NLDAS_WORKERS = int(os.environ.get("IDEAS_NLDAS_WORKERS", "6"))


def _nldas_month_file(year, mm):
    """Monthly NLDAS precip file, whichever of the two layouts is staged."""
    import os
    cands = [
        f"{NLDAS_PRECIP}/ctsmforc.NLDAS2.0.125d.v1.Prec.{year}-{mm:02d}.nc",
        f"{NLDAS_CLM_DIR}/clmforc.nldas.{year}-{mm:02d}.nc",
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"no NLDAS precip file for {year}-{mm:02d}; looked in "
        f"{NLDAS_PRECIP} and {NLDAS_CLM_DIR}")


def _soil_cov(c):
    """(clay_max %, ksat_min um/s) from the column's dominant-component horizons."""
    layers = (c.get("soil_profile") or {}).get("layers") or []
    comp = layers[0].get("component") if layers else None
    hz = [l for l in layers if l.get("component") == comp]
    def num(x):
        try: return float(x)
        except (TypeError, ValueError): return None
    clays = [v for v in (num(l.get("clay_pct")) for l in hz) if v is not None]
    ks = [v for v in (num(l.get("ksat_ums")) for l in hz) if v is not None]
    return (max(clays) if clays else None, min(ks) if ks else None)


def _nldas_month_slab(args):
    """(slab, mm_per_step) for one month's lat/lon box, or (None, 0) if absent.

    Module-level and self-contained so it can cross a process boundary; returns
    only the small box, never the 2 GB file.
    """
    year, mm, i0, i1, j0, j1 = args
    import numpy as np
    import xarray as xr
    try:
        ds = xr.open_dataset(_nldas_month_file(year, mm))
    except FileNotFoundError:
        return None, 0.0
    try:
        var = next(v for v in ds.data_vars if "PREC" in v.upper())
        nt = ds.sizes["time"]
        slab = np.asarray(ds[var][:, i0:i1 + 1, j0:j1 + 1].values)
        per_step = 86400.0 / (24 if nt > 400 else 8)    # hourly vs 3-hourly
    finally:
        ds.close()
    return slab, per_step


def nldas_annual_precip(cols, year):
    """Per-column annual NLDAS precipitation (mm/yr) from the nearest 12 km cell.

    Reads ONE small lat/lon slab per month, then takes every column out of it in
    memory.

    The obvious form -- `ds[var][:, i, j]` once per column -- looks like a cheap
    point read and is not. Precipitation is stored (time, lat, lon) contiguously
    in a ~2 GB monthly file, so pulling one (i, j) across all timesteps strides
    the whole array, and doing it per column repeats that traversal once per
    column per month: 14 columns x 12 months was 168 passes over 24 GB, roughly
    an hour of Lustre time to fill one panel of the design figure.

    The columns of a HUC8 span a handful of 12 km cells, so their bounding box
    is tiny; reading it whole costs one pass and a few MB.
    """
    import numpy as np
    import xarray as xr
    d0 = xr.open_dataset(_nldas_month_file(year, 1))
    lats = d0["LATIXY"].values[:, 0]
    lons = d0["LONGXY"].values[0, :]
    d0.close()
    idx = {c["id"]: (int(np.abs(lats - c["lat"]).argmin()),
                     int(np.abs(lons - (c["lon"] % 360.0)).argmin())) for c in cols}
    if not idx:
        return {}
    i0 = min(i for i, _ in idx.values()); i1 = max(i for i, _ in idx.values())
    j0 = min(j for _, j in idx.values()); j1 = max(j for _, j in idx.values())

    # The twelve months are read CONCURRENTLY. Precipitation is
    # (time, lat, lon) and contiguous, so a lat/lon box across all times is one
    # strided read PER TIMESTEP -- 744 per month, 8928 for the year. That is
    # seek latency, not bandwidth or CPU (the slab is a few MB), and it was
    # 4 min 14 s of a run's step 0. Independent files, so overlapping them is
    # the same trick that fixed the warm start.
    tot = {cid: 0.0 for cid in idx}
    jobs = [(year, mm, i0, i1, j0, j1) for mm in range(1, 13)]
    parts = None
    if len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        try:
            with ProcessPoolExecutor(max_workers=NLDAS_WORKERS) as ex:
                parts = list(ex.map(_nldas_month_slab, jobs))
        except Exception:
            parts = None                      # fall through to serial
    if parts is None:
        parts = [_nldas_month_slab(j) for j in jobs]

    for slab, per_step in parts:
        if slab is None:
            continue
        for cid, (i, j) in idx.items():
            tot[cid] += float(slab[:, i - i0, j - j0].sum()) * per_step
    return tot


def plot_columns(res, out_path, forcing_year=None):
    """Render a 2x2 illustration of the sampling design from an expand() result.

    Purpose-built for the spatial sampling layer (NOT the ELM-case domain plots
    per-column setup figures, which draw one configured column each).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cols = res.get("columns", [])
    bands = res.get("bands", [])
    bbox = res.get("bbox", {})
    grid = res.get("grid", [])
    nb = max(len(bands), 1)
    bcols = plt.cm.viridis(np.linspace(0.12, 0.9, nb))

    def bcolor(b):
        return bcols[min(max(int(b) - 1, 0), nb - 1)]

    fig, ax = plt.subplots(2, 3, figsize=(17.5, 9))
    fig.suptitle(f"Sampling design: {res.get('n_columns')} columns",
                 fontsize=15, fontweight="bold")

    # P1 — domain map: terrain background + watershed outline + sample points
    a = ax[0, 0]
    if grid and len(grid) >= 4:
        glon = [g["lon"] for g in grid]
        glat = [g["lat"] for g in grid]
        gelev = [g["elevation_m"] for g in grid]
        try:
            tcf = a.tricontourf(glon, glat, gelev, levels=12, cmap="terrain", alpha=0.85)
            fig.colorbar(tcf, ax=a, shrink=0.75, label="elevation (m)")
        except Exception:
            a.scatter(glon, glat, c=gelev, cmap="terrain", s=12)
    elif grid:
        a.scatter([g["lon"] for g in grid], [g["lat"] for g in grid], s=9, c="0.82")
    for ring in res.get("boundary", []):          # actual watershed polygon (WBD)
        a.plot([p[0] for p in ring], [p[1] for p in ring],
               color="navy", lw=1.8, zorder=4)
    for c in cols:
        a.scatter(c["lon"], c["lat"], s=85, color=bcolor(c["band"]),
                  edgecolor="white", linewidth=1.0, zorder=5)
    if bbox:
        mlat = (bbox["min_lat"] + bbox["max_lat"]) / 2
        a.set_aspect(1.0 / max(np.cos(np.radians(mlat)), 1e-3))
    a.set_title("Sample points over domain (terrain + watershed)")
    a.set_xlabel("lon"); a.set_ylabel("lat")

    # scale bar + north arrow (publication map furniture)
    if bbox:
        import math
        mlat = (bbox["min_lat"] + bbox["max_lat"]) / 2
        km_per_deg = 111.32 * math.cos(math.radians(mlat))
        bar_km = 20
        bar_deg = bar_km / km_per_deg
        x0 = bbox["min_lon"] + 0.05 * (bbox["max_lon"] - bbox["min_lon"])
        y0 = bbox["min_lat"] + 0.045 * (bbox["max_lat"] - bbox["min_lat"])
        a.plot([x0, x0 + bar_deg], [y0, y0], color="k", lw=2.5,
               solid_capstyle="butt", zorder=6)
        a.text(x0 + bar_deg / 2, y0 + 0.012, f"{bar_km} km", ha="center",
               fontsize=7.5, zorder=6)
        xn = bbox["min_lon"] + 0.07 * (bbox["max_lon"] - bbox["min_lon"])
        yn = bbox["max_lat"] - 0.12 * (bbox["max_lat"] - bbox["min_lat"])
        a.annotate("N", xy=(xn, yn + 0.05), xytext=(xn, yn), zorder=6,
                   ha="center", fontsize=9, fontweight="bold",
                   arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))

    # locator inset (cartopy, optional — skipped gracefully if unavailable)
    if bbox:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            ins = a.inset_axes([0.72, 0.02, 0.27, 0.27],
                               projection=ccrs.PlateCarree())
            ins.set_extent([-125, -110, 41, 50], crs=ccrs.PlateCarree())
            ins.add_feature(cfeature.STATES.with_scale("50m"),
                            edgecolor="0.5", linewidth=.4, facecolor="0.95")
            ins.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=.5)
            ins.plot([bbox["min_lon"], bbox["max_lon"], bbox["max_lon"],
                      bbox["min_lon"], bbox["min_lon"]],
                     [bbox["min_lat"], bbox["min_lat"], bbox["max_lat"],
                      bbox["max_lat"], bbox["min_lat"]],
                     color="#b91c1c", lw=1.2, transform=ccrs.PlateCarree())
        except Exception as e:
            print(f"  (locator inset skipped: {str(e)[:60]})")
    a.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=bcolor(b["band"]),
                                 markeredgecolor="white",
                                 label=f"band {b['band']}: {b['elev_lo_m']}-{b['elev_hi_m']} m")
                      for b in bands], fontsize=8, loc="best", framealpha=0.9)

    # P2 — elevation hypsometry + bands
    a = ax[0, 1]
    if grid:
        a.hist([g["elevation_m"] for g in grid], bins=25, color="0.78",
               edgecolor="white")
    for b in bands:
        a.axvline(b["elev_lo_m"], color="0.4", ls="--", lw=0.8)
    if bands:
        a.axvline(bands[-1]["elev_hi_m"], color="0.4", ls="--", lw=0.8)
    ymax = a.get_ylim()[1]
    for c in cols:
        a.plot([c["elevation_m"], c["elevation_m"]], [0, ymax * 0.12],
               color=bcolor(c["band"]), lw=1.5)
    a.set_title("Elevation distribution + bands (ticks = columns)")
    a.set_xlabel("elevation (m)"); a.set_ylabel("DEM grid count")

    # P3 — elevation vs Fan WTD, marker = soil texture
    a = ax[1, 0]
    textures = sorted({c.get("soil_top_texture") for c in cols
                       if c.get("soil_top_texture")})
    marks = ["o", "^", "s", "D", "v", "P", "X", "*"]
    tmark = {t: marks[i % len(marks)] for i, t in enumerate(textures)}
    plotted = False
    for c in cols:
        y = c.get("fan_wtd_m")
        if y is None:
            continue
        a.scatter(c["elevation_m"], y, color=bcolor(c["band"]),
                  marker=tmark.get(c.get("soil_top_texture"), "x"),
                  s=85, edgecolor="k", linewidth=0.4)
        plotted = True
    a.set_title("Fan water-table depth vs elevation")
    a.set_xlabel("elevation (m)"); a.set_ylabel("Fan WTD (m below surface)")
    if textures:
        a.legend(handles=[plt.Line2D([], [], marker=tmark[t], ls="", color="0.4",
                                     label=t) for t in textures],
                 fontsize=8, title="top soil", loc="best")
    if not plotted:
        a.text(0.5, 0.5, "no Fan WTD values", transform=a.transAxes, ha="center")

    # P4 — columns per band
    a = ax[1, 1]
    labels = [f"{b['elev_lo_m']}-{b['elev_hi_m']}" for b in bands]
    a.bar(range(nb), [b["allocated"] for b in bands],
          color=[bcolor(b["band"]) for b in bands], edgecolor="k")
    for i, b in enumerate(bands):
        a.text(i, b["allocated"] + 0.04, f"{b['allocated']}\n/{b['grid_points']} pts",
               ha="center", va="bottom", fontsize=8)
    a.set_xticks(range(nb)); a.set_xticklabels(labels, rotation=20, fontsize=8)
    a.set_ylabel("columns allocated"); a.set_title("Columns per elevation band")

    # P5 — soil configuration coverage: clay vs Ksat actually sampled
    a = ax[0, 2]
    plotted = False
    for c in cols:
        clay, ks = _soil_cov(c)
        if clay is None or ks is None:
            continue
        a.scatter(max(ks, 0.05), clay, color=bcolor(c["band"]),
                  marker=tmark.get(c.get("soil_top_texture"), "x"),
                  s=85, edgecolor="k", linewidth=0.4)
        plotted = True
    a.set_xscale("log")
    a.set_xlabel("min Ksat (µm/s, log)"); a.set_ylabel("max clay (%)")
    a.set_title("Soil configurations sampled (SSURGO)")
    if not plotted:
        a.text(0.5, 0.5, "no soil profiles", transform=a.transAxes, ha="center")

    # P6 — forcing coverage: NLDAS annual precip vs elevation (12 km cells)
    a = ax[1, 2]
    if forcing_year:
        try:
            pr = nldas_annual_precip(cols, forcing_year)
            for c in cols:
                a.scatter(c["elevation_m"], pr[c["id"]], color=bcolor(c["band"]),
                          s=85, edgecolor="k", linewidth=0.4)
            a.set_title(f"Forcing sampled: NLDAS precip {forcing_year} (12 km)")
            a.set_xlabel("elevation (m)"); a.set_ylabel("annual precip (mm/yr)")
        except Exception as e:
            a.text(0.5, 0.5, f"NLDAS preview unavailable\n{str(e)[:60]}",
                   transform=a.transAxes, ha="center", fontsize=9)
            a.set_title("Forcing sampled (NLDAS)")
    else:
        a.text(0.5, 0.5, "pass --forcing-year to preview\nthe NLDAS precip gradient",
               transform=a.transAxes, ha="center", fontsize=9, color="0.4")
        a.set_title("Forcing sampled (NLDAS)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# RUN-DIR INPUTS + CLI
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_from_brief(brief):
    b = (brief.get("domain") or {}).get("bbox") or {}
    keys = ("min_lon", "min_lat", "max_lon", "max_lat")
    return {k: b[k] for k in keys} if all(k in b for k in keys) else None


def _n_from_plan(plan):
    return ((plan.get("sampling_strategy") or {}).get("n_exploratory")
            or (plan.get("experiment_summary") or {}).get("exploratory"))


def _print_table(res):
    print("\nBANDS:")
    for b in res["bands"]:
        print(f"  band {b['band']}: {b['elev_lo_m']:>5}-{b['elev_hi_m']:>5} m  "
              f"| {b['grid_points']:>3} grid pts -> {b['allocated']} columns")
    print(f"\n{res['n_columns']} CONCRETE COLUMNS:")
    print(f"  {'id':<8}{'lat':>9}{'lon':>11}{'elev_m':>8}{'band':>5}"
          f"{'fan_m':>8}  soil")
    print("  " + "-" * 64)
    for c in res["columns"]:
        print(f"  {c['id']:<8}{c['lat']:>9}{c['lon']:>11}{c['elevation_m']:>8}"
              f"{c['band']:>5}{(c.get('fan_wtd_m') if c.get('fan_wtd_m') is not None else '-'):>8}"
              f"  {c.get('soil_top_texture') or '-'}")


def main():
    ap = argparse.ArgumentParser(description="Tier-2 expander: strategy -> concrete columns")
    ap.add_argument("--run-dir", help="pipeline output dir (reads brief + plan)")
    ap.add_argument("--bbox", help="min_lon,min_lat,max_lon,max_lat (overrides brief)")
    ap.add_argument("--n", type=int, help="number of columns (overrides plan)")
    ap.add_argument("--bands", type=int, default=0, help="number of elevation bands")
    ap.add_argument("--grid-n", type=int, default=120, help="DEM sample density")
    ap.add_argument("--no-soil", action="store_true", help="skip soil enrichment (faster)")
    ap.add_argument("--plot", action="store_true",
                    help="render sampling_design.png (domain map, hypsometry, WTD vs elev, allocation)")
    ap.add_argument("--forcing-year", type=int, default=None,
                    help="add an NLDAS precip-vs-elevation forcing-coverage panel for this year")
    args = ap.parse_args()

    out_dir = Path(args.run_dir) if args.run_dir else Path(".")
    out = out_dir / "columns.json"

    print("=" * 72)
    print("TIER-2 EXPANDER  —  strategy -> concrete columns (deterministic; no execution)")
    print("=" * 72)

    if args.run_dir and out.exists():
        # Pure read: re-plot/inspect an already-materialized run-dir — no fetch.
        res = json.loads(out.read_text())
        print(f"reading existing {out} ({res.get('n_columns')} columns — no MCP fetch)")
    else:
        # Materialize once: resolve domain + N, then sample DEM/soil/Fan.
        bbox = n_total = huc = None
        n_bands = args.bands or 4
        if args.run_dir:
            rd = Path(args.run_dir)
            brief = json.loads((rd / "reception_brief.json").read_text())
            plan = json.loads((rd / "plan.json").read_text()) if (rd / "plan.json").exists() else {}
            if (plan.get("model_choice") or {}).get("design_archetype") == "conceptual":
                sys.exit("Plan is 'conceptual' archetype — no spatial expansion needed.")
            bbox = _bbox_from_brief(brief)
            n_total = _n_from_plan(plan)
            huc = (brief.get("domain") or {}).get("huc")
            if not args.bands:
                eb = (brief.get("heterogeneity") or {}).get("elevation_bands") or []
                n_bands = len(eb) if eb else 4
        if args.bbox:
            v = [float(x) for x in args.bbox.split(",")]
            bbox = {"min_lon": v[0], "min_lat": v[1], "max_lon": v[2], "max_lat": v[3]}
        if args.n:
            n_total = args.n
        if not bbox:
            sys.exit("No bbox (give --run-dir with a site brief, or --bbox).")
        if not n_total:
            sys.exit("No column count (give --run-dir with a plan, or --n).")

        print(f"bbox: {bbox} | N={n_total} | bands={n_bands}")
        clients = MCPManager("mcp_config.json").get_all_clients()
        boundary = None
        if huc:   # watershed polygon — clips sampling to the basin + outlines the map
            b = clients["terrain"].call_tool_json(
                "get_watershed_boundary", {"huc": huc, "huc_level": len(huc)}) or {}
            boundary = b.get("rings")
        res = expand(clients, bbox, n_total, n_bands, grid_n=args.grid_n,
                     do_soil=not args.no_soil, boundary=boundary)
        if "error" in res:
            sys.exit(res["error"])
        if boundary:
            res["boundary"] = boundary
        out.write_text(json.dumps(res, indent=2))   # self-contained (grid + boundary)
        print(f"\nSaved {res['n_columns']} columns -> {out}")

    _print_table(res)
    if args.plot:
        png = plot_columns(res, str(out_dir / "sampling_design.png"),
                           forcing_year=args.forcing_year)
        print(f"Saved plot -> {png}")
    print()


if __name__ == "__main__":
    main()
