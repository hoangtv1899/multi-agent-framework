#!/usr/bin/env python3
"""
Fan WTD MCP Server — equilibrium water-table depth from Fan, Li & Miguez-Macho
(2013), "Global Patterns of Groundwater Table Depth", Science 339:940-943.

STATIC-FILE server: reads a Fan 2013 NetCDF tile from disk and serves point /
bbox queries. Fan 2013 is a *modeled, observationally-constrained equilibrium*
water-table depth — use it as a spatial PRIOR / benchmark surface, complementary
to (not a substitute for) the raw USGS well measurements served by usgs_water.

Dataset: provisioned NAMERICA tiles live in data/fan_wtd/. The server
auto-discovers an annual-mean tile there, or honours the FAN_WTD_NC env var.
Loaded lazily on first query, so the server starts cleanly even with no data.

Robustness handled here:
  * a `time` dimension (size-1 annual -> squeezed; multi-month -> averaged)
  * a land `mask` variable (out-of-domain cells -> no-data)
  * sign convention auto-detected; tools report a positive `depth_to_water_m`
    (metres below land surface) regardless of the file's internal sign.

Tools:
    data_status()                  -> dataset present? grid info, convention, how to get it
    get_fan_wtd(lat, lon)          -> depth-to-water (m below surface) at nearest cell
    sample_fan_wtd(bbox, n)        -> grid of depth-to-water across a bbox + summary
"""
import json
import math
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "fan_wtd"

_VAR_CANDIDATES = ["WTD", "wtd", "water_table_depth", "watertabledepth", "Band1"]
_LAT_CANDIDATES = ["lat", "latitude", "y", "Y"]
_LON_CANDIDATES = ["lon", "longitude", "x", "X"]
_MASK_CANDIDATES = ["mask", "MASK", "land_mask", "landmask"]

_SOURCE = "Fan, Li & Miguez-Macho (2013) equilibrium water-table depth"
_MAX_GRID = 400

mcp = FastMCP("fan_wtd")

_DS = {"loaded": False, "da": None, "lat": None, "lon": None,
       "sign": 1.0, "path": None, "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING (lazy, cached)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_path():
    """FAN_WTD_NC if set, else auto-discover a tile in data/fan_wtd (prefer annual mean)."""
    env = os.environ.get("FAN_WTD_NC")
    if env:
        return os.path.expanduser(env)
    if DATA_DIR.is_dir():
        ncs = sorted(DATA_DIR.glob("*.nc"))
        for pref in ("annualmean", "annual", "mean"):
            for f in ncs:
                if pref in f.name.lower():
                    return str(f)
        if ncs:
            return str(ncs[0])
    return str(DATA_DIR / "fan2013_wtd.nc")


def _load():
    if _DS["loaded"]:
        return _DS
    _DS["loaded"] = True
    path = _resolve_path()
    _DS["path"] = path
    if not Path(path).exists():
        _DS["error"] = (
            f"Fan WTD dataset not found at {path}. Put a Fan 2013 NetCDF tile in "
            f"{DATA_DIR} or set FAN_WTD_NC. See README.md."
        )
        return _DS
    try:
        import numpy as np
        import xarray as xr
        ds = xr.open_dataset(path)
        var = next((v for v in _VAR_CANDIDATES if v in ds.variables), None)
        lat = next((c for c in _LAT_CANDIDATES if c in ds.coords or c in ds.variables), None)
        lon = next((c for c in _LON_CANDIDATES if c in ds.coords or c in ds.variables), None)
        if not (var and lat and lon):
            _DS["error"] = (f"could not identify WTD var / lat / lon in {path} "
                            f"(vars={list(ds.variables)})")
            return _DS

        da = ds[var]
        if "time" in da.dims:                       # annual: squeeze; monthly: mean
            da = da.isel(time=0) if da.sizes["time"] == 1 else da.mean("time")

        mask = next((m for m in _MASK_CANDIDATES if m in ds.variables), None)
        if mask is not None:                        # blank out out-of-domain cells
            md = ds[mask]
            if "time" in md.dims:
                md = md.isel(time=0)
            da = da.where(md == 1)

        # Detect sign: if valid land values are mostly negative, the file stores
        # WTD as negative-below-surface -> depth = -value. Sample a coarse stride.
        strided = da.isel({lat: slice(None, None, 200), lon: slice(None, None, 200)})
        med = float(strided.median(skipna=True))
        _DS["sign"] = -1.0 if med < 0 else 1.0

        _DS.update(da=da, lat=lat, lon=lon, error=None)
    except Exception as e:
        _DS["error"] = f"failed to open {path}: {e}"
    return _DS


def _grid_points(min_lon, min_lat, max_lon, max_lat, n):
    mean_lat = (min_lat + max_lat) / 2.0
    w = max(1e-9, (max_lon - min_lon)) * math.cos(math.radians(mean_lat))
    h = max(1e-9, (max_lat - min_lat))
    nx = max(1, round(math.sqrt(max(1, n) * w / h)))
    ny = max(1, math.ceil(max(1, n) / nx))
    return [((min_lat + (j + 0.5) * (max_lat - min_lat) / ny),
             (min_lon + (i + 0.5) * (max_lon - min_lon) / nx))
            for j in range(ny) for i in range(nx)]


# ─────────────────────────────────────────────────────────────────────────────
# QUERY HELPERS (testable without the MCP transport)
# ─────────────────────────────────────────────────────────────────────────────

def _depth(val):
    """Convert a raw WTD value to positive metres-below-surface."""
    return round(_DS["sign"] * float(val), 3)


def _point(lat, lon):
    d = _load()
    if d["error"]:
        return {"error": d["error"]}
    import numpy as np
    sel = d["da"].sel({d["lat"]: lat, d["lon"]: lon}, method="nearest")
    val = float(np.asarray(sel.values).squeeze())
    if math.isnan(val):
        return {"lat": lat, "lon": lon, "wtd_m": None, "depth_to_water_m": None,
                "note": "outside Fan land domain (masked / no-data)"}
    return {"lat": lat, "lon": lon,
            "wtd_m": round(val, 3),
            "depth_to_water_m": _depth(val),
            "grid_lat": round(float(sel[d["lat"]].values), 5),
            "grid_lon": round(float(sel[d["lon"]].values), 5)}


def _sample(min_lon, min_lat, max_lon, max_lat, n):
    d = _load()
    if d["error"]:
        return {"error": d["error"]}
    import numpy as np
    import xarray as xr
    pts = _grid_points(min_lon, min_lat, max_lon, max_lat, min(int(n), _MAX_GRID))
    tlat = xr.DataArray([p[0] for p in pts], dims="pt")
    tlon = xr.DataArray([p[1] for p in pts], dims="pt")
    vals = np.atleast_1d(np.asarray(
        d["da"].sel({d["lat"]: tlat, d["lon"]: tlon}, method="nearest").values).squeeze())

    points, valid = [], []
    for (la, lo), v in zip(pts, vals):
        depth = None if (v is None or np.isnan(v)) else _depth(v)
        if depth is not None:
            valid.append(depth)
        points.append({"lat": round(la, 5), "lon": round(lo, 5),
                       "depth_to_water_m": depth})
    summary = {"n_points": len(points), "n_valid": len(valid)}
    if valid:
        valid.sort()
        summary.update(min_depth_m=round(valid[0], 3), max_depth_m=round(valid[-1], 3),
                       mean_depth_m=round(sum(valid) / len(valid), 3))
    return {"bbox": {"min_lon": min_lon, "min_lat": min_lat,
                     "max_lon": max_lon, "max_lat": max_lat},
            "summary": summary, "points": points}


def _status():
    d = _load()
    out = {"path": d["path"], "present": Path(d["path"]).exists() if d["path"] else False,
           "source": _SOURCE}
    if d["error"]:
        out.update(status="unavailable", detail=d["error"])
        return out
    da = d["da"]
    out.update(status="ready",
               depth_convention="positive depth_to_water_m = metres below land surface"
                                + (" (file stores negative-below; auto-corrected)"
                                   if d["sign"] < 0 else ""),
               lat_range=[round(float(da[d["lat"]].min()), 3), round(float(da[d["lat"]].max()), 3)],
               lon_range=[round(float(da[d["lon"]].min()), 3), round(float(da[d["lon"]].max()), 3)],
               shape=list(da.shape))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def data_status() -> str:
    """Report whether the Fan 2013 dataset is present, its grid/convention, and how to obtain it."""
    return json.dumps(_status())


@mcp.tool()
def get_fan_wtd(lat: float, lon: float) -> str:
    """Equilibrium depth-to-water (m below surface) at the nearest Fan 2013 grid cell."""
    return json.dumps({**_point(lat, lon), "source": _SOURCE})


@mcp.tool()
def sample_fan_wtd(min_lon: float, min_lat: float,
                   max_lon: float, max_lat: float, n: int = 80) -> str:
    """Sample ~n Fan 2013 depth-to-water values across a bbox; returns points + summary."""
    return json.dumps({**_sample(min_lon, min_lat, max_lon, max_lat, n),
                       "source": _SOURCE})


if __name__ == "__main__":
    mcp.run(transport="stdio")
