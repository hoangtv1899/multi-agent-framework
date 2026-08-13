#!/usr/bin/env python3
"""
The modelled water table, read locally
src/core/static_wtd.py

Reception fetches the ParFlow CONUS2 steady-state water table ONCE per basin and
writes it as a GeoTIFF (hydrodata's `download_conus2_wtd`). This reads that file.

NO NETWORK, NO PIN, NO MCP SESSION. That is the whole point. Sampling the field
per point used to mean one request to Princeton per column, and the sampler, the
validator and the analyzer each did it again — which is traffic a university's
server should not be carrying, and a hard dependency on credentials for anything
that wanted a number. The file is a few kilobytes and answers anywhere inside
the basin, forever.

WHY A RASTER AND NOT A LIST OF POINTS. `sample_columns` SNAPS each column onto
the CONUS grid, which moves it. The locations reception knows about are
therefore not the locations anyone later asks about, and a list of values cannot
answer a question about a point that was not in the list. A raster can.

Everything is metres below the land surface, positive downward — the same sign
convention as the framework's `water_table_depth_m`.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# THE FILENAME IS A CONTRACT with mcp/hydrodata-mcp/main.py, which writes it.
# The framework and the MCP are separate processes and cannot import each other,
# so this name is agreed by convention. Change it in both or in neither.
RASTER_NAME = "wtd_conus2.tif"

# WGS84 lat/lon, which is what every caller here has. The raster is in ParFlow's
# Lambert Conformal Conic, so each point is reprojected before it is looked up.
_LATLON_CRS = "EPSG:4326"


def raster_path(where) -> Optional[Path]:
    """The GeoTIFF for a run, or None if it was never written.

    `where` may be the run directory OR a path to a file inside it — normally
    reception.json. THAT MATTERS: compare_to_obs can be pointed at ANOTHER
    run's reception.json to score these columns against that domain's
    observations, in which case the water table has to come from the same place
    the observations did, not from the run being analysed.
    """
    p = Path(where)
    d = p.parent if p.is_file() else p
    f = d / RASTER_NAME
    return f if f.is_file() else None


def sample(where, lats: Sequence[float],
           lons: Sequence[float]) -> List[Optional[float]]:
    """Depth to water at each point, in input order. None where unavailable.

    None means one of three things, and `describe` tells them apart: the raster
    was never written, the point is outside it, or the cell holds no data.
    A missing value is NEVER silently replaced by the nearest edge cell — a
    clamped read returns a real-looking number from the wrong place, which is
    the failure mode this whole file exists to avoid.
    """
    return [r["wtd_m"] for r in describe(where, lats, lons)["points"]]


def describe(where, lats: Sequence[float],
             lons: Sequence[float]) -> Dict[str, Any]:
    """`sample` with its reasons: per-point rows plus what the raster is.

    Use this when a None has to be explained — in a provenance record, or when
    deciding whether a column can be initialised at all.
    """
    lats = list(lats)
    lons = list(lons)
    if len(lats) != len(lons):
        return {"ok": False,
                "error": f"lats has {len(lats)} entries, lons has {len(lons)}",
                "points": []}

    path = raster_path(where)
    if path is None:
        return {"ok": False,
                "error": (f"no {RASTER_NAME} beside {where}. Reception writes "
                          f"it once per basin via hydrodata's "
                          f"download_conus2_wtd; re-run reception for this "
                          f"domain rather than fetching it here."),
                "points": [{"lat": la, "lon": lo, "wtd_m": None,
                            "note": "no raster"}
                           for la, lo in zip(lats, lons)]}

    try:
        import rasterio
        from rasterio.warp import transform as warp_transform
    except ImportError as e:
        return {"ok": False, "error": f"needs rasterio: {e}", "points": []}

    with rasterio.open(path) as src:
        # READ ONCE, INDEX MANY. The whole basin is a few thousand cells; a
        # windowed read per point would be slower and buys nothing.
        band = src.read(1)
        inv = ~src.transform
        height, width = band.shape
        tags = src.tags()
        crs = src.crs
        res = abs(src.transform.a)
        xs, ys = warp_transform(_LATLON_CRS, crs, lons, lats)

    pts, n_ok, n_outside, n_nodata = [], 0, 0, 0
    for la, lo, x, y in zip(lats, lons, xs, ys):
        col, rowf = inv * (x, y)
        # FLOOR, NOT ROUND, matching hydrodata's _xy: a cell covers [i, i+1),
        # so the cell containing a point is the floor of its fractional index.
        # Rounding reads the neighbour whenever the fraction passes 0.5 — half
        # of all points — and at 1 km that is a kilometre away. At Naches
        # col_05 that difference was 0.05 m against 177.58 m.
        c, r = int(col // 1), int(rowf // 1)
        rec: Dict[str, Any] = {"lat": la, "lon": lo, "col": c, "row": r}
        if not (0 <= r < height and 0 <= c < width):
            rec.update({"wtd_m": None, "note": "outside the raster"})
            n_outside += 1
        else:
            v = float(band[r, c])
            if v != v:                              # NaN is how no-data is stored
                rec.update({"wtd_m": None, "note": "no data at this cell"})
                n_nodata += 1
            else:
                rec["wtd_m"] = round(v, 3)
                n_ok += 1
        pts.append(rec)

    return {"ok": True, "path": str(path), "points": pts,
            "n_points": len(pts), "n_with_value": n_ok,
            "n_outside": n_outside, "n_nodata": n_nodata,
            "shape": [height, width], "resolution_m": round(res, 4),
            "crs": str(crs),
            "dataset": tags.get("dataset"), "variable": tags.get("variable"),
            "kind": tags.get("kind"), "source": tags.get("source"),
            "units": "m below land surface, positive down"}
