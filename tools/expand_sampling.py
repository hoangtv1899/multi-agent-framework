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

def expand(clients, bbox, n_total, n_bands, grid_n=180, do_soil=True):
    terr = clients["terrain"]
    fan = clients.get("fan_wtd")
    geo = clients.get("geology")

    grid = terr.call_tool_json("sample_elevation_grid", {**bbox, "n": grid_n}) or {}
    pts = [p for p in grid.get("points", []) if p.get("elevation_m") is not None]
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
            columns.append(col)
            cid += 1

    return {"bbox": bbox, "n_requested": n_total, "n_columns": len(columns),
            "bands": [{"band": i + 1, "elev_lo_m": round(bands[i][0]),
                       "elev_hi_m": round(bands[i][1]), "grid_points": counts[i],
                       "allocated": alloc[i]} for i in range(len(bands))],
            "columns": columns}


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
    ap.add_argument("--grid-n", type=int, default=180, help="DEM sample density")
    ap.add_argument("--no-soil", action="store_true", help="skip soil enrichment (faster)")
    args = ap.parse_args()

    bbox = n_total = None
    n_bands = args.bands or 4
    out_dir = Path(".")

    if args.run_dir:
        rd = Path(args.run_dir)
        out_dir = rd
        brief = json.loads((rd / "reception_brief.json").read_text())
        plan = json.loads((rd / "plan.json").read_text()) if (rd / "plan.json").exists() else {}
        if (plan.get("model_choice") or {}).get("design_archetype") == "conceptual":
            sys.exit("Plan is 'conceptual' archetype — no spatial expansion needed "
                     "(the controlled treatments ARE the columns).")
        bbox = _bbox_from_brief(brief)
        n_total = _n_from_plan(plan)
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

    print("=" * 72)
    print("TIER-2 EXPANDER  —  strategy -> concrete columns (deterministic; no execution)")
    print("=" * 72)
    print(f"bbox: {bbox} | N={n_total} | bands={n_bands}")

    clients = MCPManager("mcp_config.json").get_all_clients()
    res = expand(clients, bbox, n_total, n_bands,
                 grid_n=args.grid_n, do_soil=not args.no_soil)
    if "error" in res:
        sys.exit(res["error"])

    _print_table(res)
    out = out_dir / "columns.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nSaved {res['n_columns']} columns -> {out}\n")


if __name__ == "__main__":
    main()
