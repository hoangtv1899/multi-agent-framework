#!/usr/bin/env python3
"""
Reception's deterministic gather phase
src/agents/reception_gather.py

Reception is the ONLY component that reaches outside the framework. Everything
downstream — the planner, the sampler, the validator, the analyzer — reads what
this module wrote. That rule exists because the alternative kept producing two
answers to one question: reception reported 25 stream gauges where validation
found 93, because the LLM was summarising a tool result truncated to 12,000
characters while the validator queried directly.

So nothing here is decided by the model. Once the LLM has fixed the DOMAIN and
the PERIOD, these fetches always happen, always the same way, and their results
are recorded verbatim with provenance. The LLM may summarise them afterwards; it
may not author them.

Two consequences worth stating:

  * The grid is fetched at the SAMPLER's resolution, not a coarser "briefing"
    one, so Tier 2 never has to fetch anything. It reads this grid, picks
    columns from it, and looks up each column's water table — no network, fully
    reproducible from reception.json alone.

  * A failed fetch is recorded as `ok: false` WITH its error. It is never an
    empty result, because "we could not look" and "there is nothing there" are
    different findings and were being conflated.

Soil is deliberately absent. It comes from the CONUS 1 km surface dataset at
the donor gridcell, which the warm start already subsets per column, so there
is nothing to fetch here.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

GRID_N = 120          # sampler resolution; expand_sampling's own default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _call(clients, server: str, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """One MCP call, with provenance and an explicit ok/error.

    Never raises: a stage that cannot fetch still produces a valid record
    saying so, which is what lets downstream tell failure from absence.
    """
    rec = {"tool": f"{server}.{tool}", "args": args, "fetched_at": _now()}
    client = (clients or {}).get(server)
    if client is None:
        return {**rec, "ok": False, "error": f"no {server} client configured",
                "result": None}
    try:
        out = client.call_tool_json(tool, args)
    except Exception as e:                       # noqa: BLE001 - recorded, not raised
        return {**rec, "ok": False, "error": f"{type(e).__name__}: {e}"[:200],
                "result": None}
    if not isinstance(out, dict):
        return {**rec, "ok": False, "error": "tool returned no JSON object",
                "result": None}
    if out.get("error"):
        return {**rec, "ok": False, "error": str(out["error"])[:200], "result": out}
    return {**rec, "ok": True, "error": None, "result": out}


def _rings(boundary) -> List[List]:
    """WBD boundary -> exterior rings, whatever shape the terrain server used."""
    if not boundary:
        return []
    if isinstance(boundary, dict):
        geom = boundary.get("geometry") or boundary
        t = geom.get("type")
        if t == "Polygon":
            return [geom["coordinates"][0]]
        if t == "MultiPolygon":
            return [p[0] for p in geom["coordinates"]]
        return []
    if isinstance(boundary, list) and boundary and isinstance(boundary[0], list):
        return boundary
    return []


def _inside(lat, lon, rings) -> bool:
    """Ray casting. Rings are (lon, lat), as GeoJSON stores them."""
    if lat is None or lon is None or not rings:
        return True                              # no boundary -> keep everything
    hit = False
    for ring in rings:
        n = len(ring)
        for i in range(n):
            x1, y1 = ring[i][0], ring[i][1]
            x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
            if (y1 > lat) != (y2 > lat):
                xin = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
                if lon < xin:
                    hit = not hit
    return hit


def gather_grid(clients, bbox: Dict[str, float], huc: str = "", boundary=None,
                n: int = GRID_N, provenance: Optional[List] = None) -> Dict[str, Any]:
    """The DEM grid at sampler resolution, clipped to the basin, with water table.

    Clipping happens before the water-table lookup so the expensive call only
    sees points that can actually host a column: for the Naches that is 58
    points, not the 121 the bbox produces.
    """
    prov = provenance if provenance is not None else []

    # The WBD polygon. Fetched HERE because reception is the only component
    # that reaches outside — the Experiment Manager used to make this call
    # itself, at which point the grid it clipped and the grid reception
    # described were two different things.
    if boundary is None and huc:
        b = _call(clients, "terrain", "get_watershed_boundary",
                  {"huc": str(huc), "huc_level": 8})
        prov.append({k: b[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
        boundary = (b.get("result") or {}).get("rings")

    g = _call(clients, "terrain", "sample_elevation_grid", {**bbox, "n": int(n)})
    prov.append({k: g[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
    pts = [p for p in ((g.get("result") or {}).get("points") or [])
           if p.get("elevation_m") is not None]

    rings = _rings(boundary)
    clipped = [p for p in pts if _inside(p.get("lat"), p.get("lon"), rings)]
    if not clipped:                              # a bad polygon must not empty the grid
        clipped, rings = pts, []

    grid = [{"lat": round(p["lat"], 5), "lon": round(p["lon"], 5),
             "elevation_m": p["elevation_m"]} for p in clipped]

    if grid:
        f = _call(clients, "fan_wtd", "get_fan_wtd_points",
                  {"lats": [p["lat"] for p in grid],
                   "lons": [p["lon"] for p in grid]})
        prov.append({k: f[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
        for p, entry in zip(grid, ((f.get("result") or {}).get("points") or [])):
            p["fan_wtd_m"] = (entry or {}).get("depth_to_water_m")

    elevs = [p["elevation_m"] for p in grid]
    return {
        "n_requested": int(n),
        "n_returned": len(pts),
        "n_in_basin": len(grid),
        "clipped_to_watershed": bool(rings),
        "boundary": rings or None,        # kept so a re-plot needs no refetch
        "elevation_min_m": min(elevs) if elevs else None,
        "elevation_max_m": max(elevs) if elevs else None,
        "relief_m": (round(max(elevs) - min(elevs), 1) if elevs else None),
        "points": grid,
    }


def gather_observations(clients, bbox_str: str, yr_start: int, yr_end: int,
                        provenance: Optional[List] = None) -> Dict[str, Any]:
    """Streamflow, water table and snow for the RESOLVED period.

    Called once, after the period is fixed, and never again — the validator
    reads this rather than re-querying, so the two cannot disagree.

    `with_values` is on: the series are what the analyzer compares against, and
    fetching them here is what makes the run reproducible from one file.
    """
    prov = provenance if provenance is not None else []
    start, end = f"{yr_start}-01-01", f"{yr_end}-12-31"
    # snow is a WATER year: the melt that feeds this calendar year began in the
    # previous October, so scoring it on calendar bounds would cut the peak.
    swe_start, swe_end = f"{yr_start - 1}-10-01", f"{yr_end}-09-30"

    out: Dict[str, Any] = {"fetched_for": {"yr_start": yr_start, "yr_end": yr_end,
                                           "bbox": bbox_str}}

    q = _call(clients, "usgs_water", "get_streamflow",
              {"bbox": bbox_str, "start_date": start, "end_date": end,
               "with_values": True, "min_days": 300})
    prov.append({k: q[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
    qr = q.get("result") or {}
    out["streamflow"] = {
        "ok": q["ok"], "error": q["error"],
        "n_in_bbox": qr.get("n_sites"),
        "n_with_records": qr.get("n_available"),
        "stations": qr.get("available") or [],
    }

    w = _call(clients, "usgs_water", "get_water_table",
              {"bbox": bbox_str, "start_date": start, "end_date": end,
               "with_values": True})
    prov.append({k: w[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
    wr = w.get("result") or {}
    out["water_table"] = {
        "ok": w["ok"], "error": w["error"],
        "n_with_records": wr.get("n_wells_with_records"),
        "wells": wr.get("wells") or [],
    }

    s = _call(clients, "snotel", "get_swe",
              {"bbox": bbox_str, "start_date": swe_start, "end_date": swe_end,
               "with_values": False})
    prov.append({k: s[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
    sr = s.get("result") or {}
    out["swe"] = {
        "ok": s["ok"], "error": s["error"],
        "period": f"{swe_start}/{swe_end}",
        "n_stations": sr.get("n_stations"),
        "n_reporting": sr.get("n_reporting"),
        "stations": sr.get("stations") or [],
    }
    return out


def summarise(observations: Dict[str, Any]) -> Dict[str, Any]:
    """The compact view the PLANNER gets — never the raw series.

    Feeding a planner 170 kB of well measurements repeats the mistake that
    started all this: an LLM reasoning over a payload it cannot hold. Station
    coordinates ARE included, because pinning validation columns to observation
    locations needs them.
    """
    q = observations.get("streamflow") or {}
    w = observations.get("water_table") or {}
    s = observations.get("swe") or {}
    return {
        "streamflow": {
            "ok": q.get("ok"), "n_in_bbox": q.get("n_in_bbox"),
            "n_with_records": q.get("n_with_records"),
            "stations": [{k: st.get(k) for k in
                          ("id", "name", "lat", "lon", "drainage_area_km2", "n_days")}
                         for st in (q.get("stations") or [])],
        },
        "water_table": {
            "ok": w.get("ok"), "n_with_records": w.get("n_with_records"),
            "wells": [{k: st.get(k) for k in ("id", "lat", "lon", "n_obs", "wtd_m")}
                      for st in (w.get("wells") or [])[:25]],
        },
        "swe": {
            "ok": s.get("ok"), "n_stations": s.get("n_stations"),
            "n_reporting": s.get("n_reporting"),
            "stations": [{k: st.get(k) for k in
                          ("triplet", "name", "lat", "lon", "elevation_m",
                           "peak_swe_mm", "peak_date")}
                         for st in (s.get("stations") or [])],
        },
    }
