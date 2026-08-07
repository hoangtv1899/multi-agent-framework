#!/usr/bin/env python3
"""
External data gathering
src/core/data_gather.py

Infrastructure, not an agent: it makes no decisions and calls no model. It sits
in core/ because that is what it is — reception CALLS it and receives a report,
rather than containing one.

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
MIN_IN_BASIN = 55     # below this the sampler has too little to choose from
MAX_GRID_N = 480      # ceiling on the retry, so a pathological shape cannot
                      # turn one terrain call into a very large one


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

    # A grid is requested over the BOUNDING BOX but used inside the BASIN, and
    # the two differ by the shape of the watershed: compact basins keep ~50% of
    # the points, an elongated coastal strip keeps 18%. Central Coastal
    # California came back with 21 usable points for 4984 km2 — 4.2 per
    # 1000 km2 against 20-37 elsewhere — which leaves farthest-point selection
    # almost nothing to choose from and barely samples the elevation bands.
    # So the request is scaled by the fill ratio and retried once.
    if pts and len(clipped) < MIN_IN_BASIN and int(n) < MAX_GRID_N:
        fill = max(len(clipped) / len(pts), 0.05)
        bigger = min(int(int(n) / fill), MAX_GRID_N)
        if bigger > int(n):
            g2 = _call(clients, "terrain", "sample_elevation_grid",
                       {**bbox, "n": bigger})
            prov.append({k: g2[k] for k in
                         ("tool", "args", "fetched_at", "ok", "error")})
            pts2 = [p for p in ((g2.get("result") or {}).get("points") or [])
                    if p.get("elevation_m") is not None]
            clip2 = [p for p in pts2 if _inside(p.get("lat"), p.get("lon"), rings)]
            if len(clip2) > len(clipped):
                pts, clipped, n = pts2, clip2, bigger

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


# A station-year of daily SWE is ~7.7 KB columnar. Five stations is fine; a
# basin with forty SNOTEL sites would put 300 KB of series into every
# reception.json. Cap the series, keep every station's SUMMARY, and record
# what was dropped — a silent first-N would quietly bias the sample toward
# whatever order the server returned.
MAX_SWE_SERIES = 12


def _cap_series(stations, limit, key="daily"):
    """Keep the series on the `limit` deepest-snowpack stations, summaries on all."""
    ranked = sorted(stations, key=lambda s: -(s.get("peak_swe_mm") or 0))
    dropped = 0
    for st in ranked[limit:]:
        if st.pop(key, None) is not None:
            dropped += 1
    if dropped:
        for st in ranked[:limit]:
            st.setdefault("_note", f"{dropped} further station(s) kept their "
                                   f"summary but not their daily series")
    return ranked


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

    # with_values: the DAILY series, not just the peak. Peak alone cannot
    # answer the question SWE is actually good for — accumulation and melt
    # TIMING, which is far less sensitive to the elevation offset between a
    # SNOTEL site and a model column than magnitude is. Streamflow and water
    # table already carry their series; this was the one observable that
    # threw its away, and n_obs=365 in the old output proves the server had
    # it all along.
    s = _call(clients, "snotel", "get_swe",
              {"bbox": bbox_str, "start_date": swe_start, "end_date": swe_end,
               "with_values": True})
    prov.append({k: s[k] for k in ("tool", "args", "fetched_at", "ok", "error")})
    sr = s.get("result") or {}
    out["swe"] = {
        "ok": s["ok"], "error": s["error"],
        "period": f"{swe_start}/{swe_end}",
        "n_stations": sr.get("n_stations"),
        "n_reporting": sr.get("n_reporting"),
        "stations": _cap_series(sr.get("stations") or [], MAX_SWE_SERIES),
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
    # THE ERROR TRAVELS WITH ok=False. Carrying the flag alone was not enough:
    # on 2026-08-07 the Chattahoochee fetch was rate-limited (HTTP 429) and the
    # planner, seeing {"ok": false, "stations": []}, wrote "observations_summary
    # lists no streamflow gauges with records in the domain" -- reporting an
    # absence for a basin thick with USGS gauges. "We could not look" and
    # "there is nothing there" are different findings, and a bare false is too
    # easy to skim past. The reason is now in front of whoever reads it.
    return {
        "streamflow": {
            "ok": q.get("ok"), "error": q.get("error"),
            "n_in_bbox": q.get("n_in_bbox"),
            "n_with_records": q.get("n_with_records"),
            "stations": [{k: st.get(k) for k in
                          ("id", "name", "lat", "lon", "drainage_area_km2", "n_days")}
                         for st in (q.get("stations") or [])],
        },
        "water_table": {
            "ok": w.get("ok"), "error": w.get("error"), "n_with_records": w.get("n_with_records"),
            "wells": [{k: st.get(k) for k in ("id", "lat", "lon", "n_obs", "wtd_m")}
                      for st in (w.get("wells") or [])[:25]],
        },
        "swe": {
            "ok": s.get("ok"), "error": s.get("error"), "n_stations": s.get("n_stations"),
            "n_reporting": s.get("n_reporting"),
            "stations": [{k: st.get(k) for k in
                          ("triplet", "name", "lat", "lon", "elevation_m",
                           "peak_swe_mm", "peak_date")}
                         for st in (s.get("stations") or [])],
        },
    }
