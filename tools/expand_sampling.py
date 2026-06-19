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
    module load pytorch/2.8.0
    python3 tools/expand_sampling.py --run-dir workflow_outputs/pipeline_XXXX
    python3 tools/expand_sampling.py --bbox -121.52,46.46,-120.51,47.14 --n 12 --bands 4
"""
import argparse
import json
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
    """Greedy farthest-point sampling for spatial spread within a band."""
    if k >= len(pts):
        return list(pts)
    if k <= 0:
        return []
    clat = sum(p["lat"] for p in pts) / len(pts)
    clon = sum(p["lon"] for p in pts) / len(pts)
    chosen = [min(pts, key=lambda p: (p["lat"] - clat) ** 2 + (p["lon"] - clon) ** 2)]
    while len(chosen) < k:
        nxt = max(pts, key=lambda p: min((p["lat"] - c["lat"]) ** 2
                                         + (p["lon"] - c["lon"]) ** 2 for c in chosen))
        chosen.append(nxt)
    return chosen


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
    inside = [p for p in pts if poly.contains(Point(p["lon"], p["lat"]))]
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
            if fan:
                fr = fan.call_tool_json("get_fan_wtd",
                                        {"lat": p["lat"], "lon": p["lon"]}) or {}
                col["fan_wtd_m"] = fr.get("depth_to_water_m")
            if geo and do_soil:
                sp = geo.call_tool_json("get_soil_profile",
                                        {"lat": p["lat"], "lon": p["lon"]}) or {}
                layers = sp.get("layers") or []
                col["soil_top_texture"] = layers[0].get("texture_class") if layers else None
                col["soil_layers"] = sp.get("num_layers")
                col["soil_profile"] = sp if layers else None   # full profile (None if no SSURGO)
            columns.append(col)
            cid += 1

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

def plot_columns(res, out_path):
    """Render a 2x2 illustration of the sampling design from an expand() result.

    Purpose-built for the spatial sampling layer (NOT the ELM-case domain plots
    in elm_setup_plotting.py, which draw a single configured column).
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

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Sampling design — {res.get('n_columns')} columns",
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

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
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
        png = plot_columns(res, str(out_dir / "sampling_design.png"))
        print(f"Saved plot -> {png}")
    print()


if __name__ == "__main__":
    main()
