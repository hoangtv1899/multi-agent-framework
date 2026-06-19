#!/usr/bin/env python3
"""
Terrain MCP Server — elevation + watershed-domain context for spatial ELM.

Unblocks the watershed-scale sampling path (see the planner capability probe):
turns a named/HUC watershed into a real bounding box, and samples USGS 3DEP
elevation across a domain so the downstream expander can stratify columns by
elevation.

Data sources (free, no key):
    USGS 3DEP elevation  — https://epqs.nationalmap.gov/v1/json   (point query)
    USGS WBD watershed   — https://hydro.nationalmap.gov/arcgis/rest/services/wbd

Tools:
    get_elevation(lat, lon)
        -> point elevation in metres (3DEP)

    sample_elevation_grid(min_lon, min_lat, max_lon, max_lat, n=50)
        -> ~n points on a roughly-square grid with elevations + summary stats

    elevation_summary(min_lon, min_lat, max_lon, max_lat, n=80, n_bands=4)
        -> min/max/mean, percentiles, and equal-interval elevation bands
           (feeds elevation stratification and the domain brief directly)

    resolve_watershed(huc="", name="", huc_level=8)
        -> {huc, name, bbox, area_km2} for a HUC code or name (WBD)
           NOTE: watershed resolution lives here for v1 alongside terrain; it
           may move to a dedicated hydrography server later.

Design: each tool is a thin wrapper over a plain helper (`_get_elevation`,
`_sample_grid`, `_resolve_watershed`) so the logic is unit-testable without
the MCP stdio transport.
"""
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from mcp.server.fastmcp import FastMCP

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
EPQS_URL = "https://epqs.nationalmap.gov/v1/json"
WBD_URL  = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer"

# WBD MapServer layer index per HUC level (verified: HUC8 = layer 4).
WBD_LAYER = {2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6}

_NODATA = -1.0e5          # EPQS returns a large negative sentinel off-coverage
_MAX_GRID = 400           # safety cap on points per grid request
_HTTP_TIMEOUT = 30

mcp = FastMCP("terrain")


# ─────────────────────────────────────────────────────────────────────────────
# ELEVATION HELPERS (3DEP / EPQS)
# ─────────────────────────────────────────────────────────────────────────────

def _get_elevation(lat: float, lon: float) -> dict:
    """Point elevation in metres from USGS 3DEP via EPQS. Returns a dict with
    elevation_m=None if the point is off-coverage or the query fails."""
    try:
        r = requests.get(
            EPQS_URL,
            params={"x": lon, "y": lat, "units": "Meters",
                    "wkid": 4326, "includeDate": "false"},
            timeout=_HTTP_TIMEOUT, verify=False,
        )
        r.raise_for_status()
        d = r.json()
        val = float(d.get("value"))
        if val <= _NODATA:
            return {"lat": lat, "lon": lon, "elevation_m": None,
                    "note": "off 3DEP coverage / no-data"}
        return {"lat": lat, "lon": lon, "elevation_m": round(val, 2),
                "resolution_m": d.get("resolution")}
    except Exception as e:
        return {"lat": lat, "lon": lon, "elevation_m": None, "error": str(e)}


def _grid_points(min_lon, min_lat, max_lon, max_lat, n):
    """Place ~n points at cell centres on a roughly-square geographic grid."""
    mean_lat = (min_lat + max_lat) / 2.0
    w = max(1e-9, (max_lon - min_lon)) * math.cos(math.radians(mean_lat))
    h = max(1e-9, (max_lat - min_lat))
    nx = max(1, round(math.sqrt(max(1, n) * w / h)))
    ny = max(1, math.ceil(max(1, n) / nx))
    pts = []
    for j in range(ny):
        for i in range(nx):
            lon = min_lon + (i + 0.5) * (max_lon - min_lon) / nx
            lat = min_lat + (j + 0.5) * (max_lat - min_lat) / ny
            pts.append((lat, lon))
    return pts


def _sample_grid(min_lon, min_lat, max_lon, max_lat, n):
    """Sample elevation at ~n grid points concurrently. Returns
    (points, summary) where summary covers the valid (non-None) elevations."""
    n = min(int(n), _MAX_GRID)
    pts = _grid_points(min_lon, min_lat, max_lon, max_lat, n)

    results = [None] * len(pts)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_get_elevation, lat, lon): k
                for k, (lat, lon) in enumerate(pts)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()

    elevs = [p["elevation_m"] for p in results if p.get("elevation_m") is not None]
    summary = _stats(elevs)
    summary["n_requested"] = n
    summary["n_valid"] = len(elevs)
    return results, summary


def _stats(values):
    """Min/max/mean/percentiles for a list of numbers (empty-safe)."""
    if not values:
        return {"min_m": None, "max_m": None, "mean_m": None, "pct": {}}
    s = sorted(values)

    def pct(p):
        if len(s) == 1:
            return s[0]
        k = (len(s) - 1) * p / 100.0
        lo = math.floor(k)
        hi = math.ceil(k)
        return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)

    return {
        "min_m": round(s[0], 1),
        "max_m": round(s[-1], 1),
        "mean_m": round(sum(s) / len(s), 1),
        "relief_m": round(s[-1] - s[0], 1),
        "pct": {"p10": pct(10), "p25": pct(25), "p50": pct(50),
                "p75": pct(75), "p90": pct(90)},
    }


def _equal_interval_bands(min_m, max_m, n_bands, elevs):
    """Equal-interval elevation bands with a count of sampled points each."""
    if min_m is None or max_m is None or max_m <= min_m:
        return []
    step = (max_m - min_m) / n_bands
    bands = []
    for b in range(n_bands):
        lo = min_m + b * step
        hi = max_m if b == n_bands - 1 else min_m + (b + 1) * step
        count = sum(1 for e in elevs if (lo <= e <= hi if b == n_bands - 1
                                         else lo <= e < hi))
        bands.append({"band": b + 1,
                      "elev_lo_m": round(lo, 1),
                      "elev_hi_m": round(hi, 1),
                      "n_points": count})
    return bands


# ─────────────────────────────────────────────────────────────────────────────
# WATERSHED HELPER (WBD)
# ─────────────────────────────────────────────────────────────────────────────

def _wbd_get(layer, params):
    url = f"{WBD_URL}/{layer}/query"
    r = requests.get(url, params={**params, "f": "json"},
                     timeout=_HTTP_TIMEOUT, verify=False)
    r.raise_for_status()
    return r.json()


def _resolve_watershed(huc="", name="", huc_level=8):
    """Resolve a HUC code or name to {huc, name, bbox, area_km2} via WBD."""
    huc_level = int(huc_level)
    if huc_level not in WBD_LAYER:
        return {"error": f"huc_level must be one of {sorted(WBD_LAYER)}"}
    layer = WBD_LAYER[huc_level]
    field = f"huc{huc_level}"

    if huc:
        where = f"{field}='{huc}'"
    elif name:
        safe = name.replace("'", "''")
        where = f"name LIKE '%{safe}%'"
    else:
        return {"error": "provide either huc or name"}

    try:
        d = _wbd_get(layer, {"where": where,
                             "outFields": f"{field},name,areasqkm",
                             "returnGeometry": "false"})
    except Exception as e:
        return {"error": f"WBD query failed: {e}"}

    feats = d.get("features", [])
    if not feats:
        return {"error": f"no HUC{huc_level} match for {where}",
                "matches": []}

    matches = [{"huc": f["attributes"].get(field),
                "name": f["attributes"].get("name"),
                "area_km2": f["attributes"].get("areasqkm")}
               for f in feats]

    # Ambiguous name → return the list so the caller can pick a HUC.
    if len(matches) > 1:
        return {"ambiguous": True, "n_matches": len(matches),
                "matches": matches,
                "note": "multiple matches — re-call with a specific huc"}

    m = matches[0]
    bbox = None
    try:
        ext = _wbd_get(layer, {"where": f"{field}='{m['huc']}'",
                               "returnExtentOnly": "true",
                               "outSR": "4326"}).get("extent", {})
        if ext:
            bbox = {"min_lon": round(ext["xmin"], 5),
                    "min_lat": round(ext["ymin"], 5),
                    "max_lon": round(ext["xmax"], 5),
                    "max_lat": round(ext["ymax"], 5)}
    except Exception as e:
        bbox = {"error": f"extent query failed: {e}"}

    return {"huc": m["huc"], "name": m["name"], "huc_level": huc_level,
            "bbox": bbox, "area_km2": m["area_km2"],
            "source": "USGS Watershed Boundary Dataset (WBD)"}


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS (thin FastMCP wrappers — return JSON strings)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_elevation(lat: float, lon: float) -> str:
    """Point elevation in metres (USGS 3DEP) for a lat/lon."""
    return json.dumps({**_get_elevation(lat, lon),
                       "source": "USGS 3DEP (EPQS)"})


@mcp.tool()
def sample_elevation_grid(min_lon: float, min_lat: float,
                          max_lon: float, max_lat: float,
                          n: int = 50) -> str:
    """Sample ~n elevations on a grid across a bbox. Returns points + stats."""
    points, summary = _sample_grid(min_lon, min_lat, max_lon, max_lat, n)
    return json.dumps({"bbox": {"min_lon": min_lon, "min_lat": min_lat,
                                "max_lon": max_lon, "max_lat": max_lat},
                       "summary": summary, "points": points,
                       "source": "USGS 3DEP (EPQS)"})


@mcp.tool()
def elevation_summary(min_lon: float, min_lat: float,
                      max_lon: float, max_lat: float,
                      n: int = 80, n_bands: int = 4) -> str:
    """Elevation stats + equal-interval bands across a bbox (for stratification)."""
    points, summary = _sample_grid(min_lon, min_lat, max_lon, max_lat, n)
    elevs = [p["elevation_m"] for p in points if p.get("elevation_m") is not None]
    bands = _equal_interval_bands(summary["min_m"], summary["max_m"],
                                  int(n_bands), elevs)
    return json.dumps({"bbox": {"min_lon": min_lon, "min_lat": min_lat,
                                "max_lon": max_lon, "max_lat": max_lat},
                       "summary": summary, "n_bands": int(n_bands),
                       "elevation_bands": bands,
                       "source": "USGS 3DEP (EPQS)"})


@mcp.tool()
def resolve_watershed(huc: str = "", name: str = "", huc_level: int = 8) -> str:
    """Resolve a watershed by HUC code or name to a bbox + area (USGS WBD)."""
    return json.dumps(_resolve_watershed(huc=huc, name=name,
                                         huc_level=huc_level))


if __name__ == "__main__":
    mcp.run(transport="stdio")
