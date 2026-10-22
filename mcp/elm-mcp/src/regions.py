#!/usr/bin/env python3
"""Cells of a gridded ELM run grouped into named regions, and numbers over them.

gridded.py reads the history; this module decides WHICH cells a number covers
and reduces them, so a whole-domain run reaches the analysis as a few regional
series instead of 13,824 cell series (35 million values per variable for seven
daily years, which no downstream step should hold).

ONE MASK RECORD, AND EVERY NUMBER CARRIES ITS MASK. The hand analysis of the
Naches run built two watershed masks that differ by 13 cells, plus a third
with burned cells removed, inside one report, and the same band read -3.2 %
in one section and -4.0 % in another because the cell sets were different.
Here a region is a named mask with its rule and its sources; MaskSet saves
them all to one netCDF file; and every reduced series names its region.

WHAT EACH BUILDER READS:
  vegetated, elevation_bands   the SURFACE data (PCT_NATVEG, TOPO). The
                               history's own `topo` is zero everywhere on the
                               Naches run, so bands read from it would put
                               every cell below 1000 m without complaint.
  polygon_mask                 a GeoJSON polygon; a cell belongs when its
                               centre lies inside (even-odd rule, holes and
                               multi-polygons honoured, 180 degrees crossed;
                               no shapely needed).
  station_cells                the cell that contains each station, and the
                               cell within a radius closest in elevation.
  mask_from_field              any gridded file on the same cells (MODIS
                               burned area, data-coverage counts, a fire
                               footprint), after checking its coordinates and
                               its dimension order.

Statistics: area-weighted, NaN-aware means per region and record
(reduce_regions); totals or means per year or water year, each record weighted
by the days it covers, with a completeness check (RegionSeries.by_year); a
normal with anomalies and ranks (anomalies); and the date by which half of a
period's total has passed (centre_of_timing).
"""
from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from gridded import (SECONDS_PER_DAY, GriddedReadError, Grid, Stream,
                     days_in_month, expected_keys, label_text, per_day_units,
                     read_clean, to_pm180)

EARTH_RADIUS_KM = 6371.0088        # mean Earth radius (IUGG), as the hand analysis used


class RegionError(Exception):
    """A region cannot be built or reduced, and the message says why."""


# ─────────────────────────────────────────────────────────────────────────────
# SURFACE DATA ON THE RUN'S CELLS
# ─────────────────────────────────────────────────────────────────────────────
def read_surface(path, grid: Grid, names: Sequence[str] = ("TOPO", "PCT_NATVEG"),
                 tol_deg: float = 1e-4) -> Dict[str, np.ndarray]:
    """Surface-data fields on the history grid, after checking that the two
    files describe the same cells (LONGXY/LATIXY against the history)."""
    import netCDF4
    out: Dict[str, np.ndarray] = {}
    with netCDF4.Dataset(path) as s:
        if "LATIXY" not in s.variables or "LONGXY" not in s.variables:
            raise RegionError(f"{Path(path).name}: no LATIXY/LONGXY to check the cells")
        grid.check_coordinates(read_clean(s["LATIXY"]), read_clean(s["LONGXY"]),
                               what=f"surface data {Path(path).name}", tol_deg=tol_deg)
        for n in names:
            if n not in s.variables:
                raise RegionError(f"{Path(path).name} has no {n}")
            out[n] = read_clean(s[n])
    return out


def vegetated(pct_natveg, min_pct: float = 100.0) -> np.ndarray:
    """Cells whose natural-vegetation land unit covers at least min_pct %.

    100 (the default) is what the Naches carbon numbers used: 13,394 of 13,824
    cells, the rest carrying some urban or lake area.
    """
    p = np.asarray(pct_natveg, dtype=np.float64)
    return np.isfinite(p) & (p >= min_pct - 1e-9)


def band_label(lo: Optional[float], hi: Optional[float]) -> str:
    if lo is None:
        return f"below {hi:g} m"
    if hi is None:
        return f"{lo:g} m and above"
    return f"{lo:g}-{hi:g} m"


def elevation_bands(topo, edges: Sequence[float] = (1000.0, 1500.0, 2000.0)
                    ) -> "OrderedDict[str, np.ndarray]":
    """{label: mask} for elevation bands; lower edge inclusive, upper exclusive.

    Cells without an elevation fall in no band. An elevation field that is
    zero everywhere is refused: that is the history file's placeholder `topo`,
    not a terrain, and would put every cell in the lowest band.
    """
    z = np.asarray(topo, dtype=np.float64)
    ok = np.isfinite(z)
    if not ok.any():
        raise RegionError("elevation is missing everywhere")
    if np.all(z[ok] == 0.0):
        raise RegionError(
            "elevation is zero in every cell: that is the history file's "
            "placeholder topo, not terrain; read TOPO from the surface data")
    e = [float(x) for x in edges]
    if not e or any(b <= a for a, b in zip(e, e[1:])):
        raise RegionError(f"band edges must increase: {edges}")
    bounds = [(None, e[0])] + list(zip(e, e[1:])) + [(e[-1], None)]
    out: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for lo, hi in bounds:
        m = ok.copy()
        if lo is not None:
            m &= z >= lo
        if hi is not None:
            m &= z < hi
        out[band_label(lo, hi)] = m
    return out


# ─────────────────────────────────────────────────────────────────────────────
# POLYGONS
# ─────────────────────────────────────────────────────────────────────────────
def polygon_rings(geojson) -> List[List[np.ndarray]]:
    """[[exterior, hole, ...], ...] from GeoJSON, rings as (n, 2) lon/lat.

    Accepts a path, a FeatureCollection (all its features together), a
    Feature, a Polygon or a MultiPolygon. EACH RING IS MADE CONTINUOUS IN
    LONGITUDE: it starts in [-180, 180) and no edge jumps more than 180
    degrees, so a ring that crosses 180 E (176 to 184, or written as 176 to
    -176) keeps its true shape instead of being turned inside out.
    """
    if isinstance(geojson, (str, os.PathLike)):
        geojson = json.loads(Path(geojson).read_text())
    t = geojson.get("type")
    if t == "FeatureCollection":
        geoms = [f.get("geometry") or {} for f in geojson.get("features") or []]
    elif t == "Feature":
        geoms = [geojson.get("geometry") or {}]
    else:
        geoms = [geojson]
    polys: List[List[np.ndarray]] = []
    for g in geoms:
        gt = g.get("type")
        if gt == "Polygon":
            parts = [g["coordinates"]]
        elif gt == "MultiPolygon":
            parts = g["coordinates"]
        else:
            raise RegionError(f"geometry type {gt!r} is not a polygon")
        for p in parts:
            rings = []
            for ring in p:
                a = np.asarray(ring, dtype=np.float64)[:, :2].copy()
                if len(a) < 3:
                    raise RegionError("a polygon ring has fewer than 3 points")
                if np.any(np.abs(a[:, 1]) > 90.0) or np.any(np.abs(a[:, 0]) > 540.0):
                    raise RegionError(
                        "polygon coordinates are not longitude/latitude in degrees "
                        "(a projected file?)")
                lon = np.unwrap(to_pm180(a[:, 0]), period=360.0)
                a[:, 0] = lon - 360.0 * np.floor((lon[0] + 180.0) / 360.0)
                rings.append(a)
            if rings:
                polys.append(rings)
    if not polys:
        raise RegionError("no polygon in the GeoJSON")
    return polys


def _even_odd(x, y, rings) -> np.ndarray:
    odd = np.zeros(x.size, dtype=bool)
    for r in rings:
        x1, y1 = r[:, 0], r[:, 1]
        x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
        for k in range(len(r)):
            ya, yb = y1[k], y2[k]
            if ya == yb:
                continue
            cross = (ya > y) != (yb > y)
            if not cross.any():
                continue
            xin = x1[k] + (y[cross] - ya) * (x2[k] - x1[k]) / (yb - ya)
            hit = np.zeros(x.size, dtype=bool)
            hit[np.flatnonzero(cross)[x[cross] < xin]] = True
            odd ^= hit
    return odd


def points_in_polygons(lon, lat, polys: List[List[np.ndarray]]) -> np.ndarray:
    """Even-odd test of points against polygons with holes; True inside.

    Each polygon counts its ring crossings together (a point in a hole crosses
    twice and is outside); polygons are then combined by union. A ring that
    runs past 180 E is also tested against the points shifted by 360 degrees.
    A point on an edge is a measure-zero case and may land either way.
    """
    x = np.asarray(lon, dtype=np.float64)
    y = np.asarray(lat, dtype=np.float64)
    shape = np.broadcast(x, y).shape
    x = to_pm180(np.broadcast_to(x, shape).ravel())
    y = np.broadcast_to(y, shape).ravel()
    inside = np.zeros(x.size, dtype=bool)
    for rings in polys:
        lo = min(float(r[:, 0].min()) for r in rings)
        hi = max(float(r[:, 0].max()) for r in rings)
        shifts = [0.0] + ([360.0] if hi > 180.0 else []) + ([-360.0] if lo < -180.0 else [])
        for s in shifts:
            inside |= _even_odd(x + s, y, rings)
    return inside.reshape(shape)


def ring_area_km2(ring: np.ndarray, radius_km: float = EARTH_RADIUS_KM) -> float:
    """Area of one ring on a sphere (Chamberlain and Duquette 2007). The ring
    must be continuous in longitude, as polygon_rings makes it."""
    lon = np.radians(ring[:, 0])
    lat = np.radians(ring[:, 1])
    if lon[0] == lon[-1] and lat[0] == lat[-1]:
        lon, lat = lon[:-1], lat[:-1]
    s = np.sum((np.roll(lon, -1) - np.roll(lon, 1)) * np.sin(lat))
    return float(abs(s) * radius_km ** 2 / 2.0)


def polygon_area_km2(polys: List[List[np.ndarray]]) -> float:
    """Exterior minus holes, summed over polygons, on a sphere: an audit
    number. Against the WGS84 ellipsoid it reads 0.27 % low on the Naches
    basin (47 N), about 0.45 % high at the equator and 0.6-0.7 % low at
    60-70 N; it is not a replacement for a geodesic library."""
    return float(sum(ring_area_km2(p[0]) - sum(ring_area_km2(h) for h in p[1:])
                     for p in polys))


def _grid_lon_range(grid: Grid) -> Optional[Tuple[float, float]]:
    """The grid's west and east limits in one continuous frame, or None when
    they cannot be told (a one-row grid, or an unstructured grid spanning more
    than half the globe)."""
    try:
        e = cell_edges(grid)
    except RegionError:
        return None
    w, ea = float(e["west"].min()), float(e["east"].max())
    return (w, ea) if ea - w < 360.0 else None


def polygon_mask(grid: Grid, geojson) -> Tuple[np.ndarray, Dict]:
    """(mask, audit): cells whose centre lies inside the polygon(s).

    The audit sets the mask area beside the polygon's own, so a mask that
    lost part of the basin off the edge of the grid shows up as a gap.
    polygon_inside_grid is None when the grid's extent cannot be told.
    """
    polys = polygon_rings(geojson)
    m = points_in_polygons(grid.lon, grid.lat, polys)
    w = grid.weight_km2
    pa = polygon_area_km2(polys)
    xs = np.concatenate([r[:, 0] for p in polys for r in p])
    ys = np.concatenate([r[:, 1] for p in polys for r in p])
    inside = None
    rng = _grid_lon_range(grid)
    if rng is not None:
        lat_lo, lat_hi = _grid_lat_range(grid)
        x0, x1 = float(xs.min()), float(xs.max())
        shift = 360.0 * np.round(((rng[0] + rng[1]) / 2.0 - (x0 + x1) / 2.0) / 360.0)
        inside = bool(x0 + shift >= rng[0] - 1e-9 and x1 + shift <= rng[1] + 1e-9
                      and ys.min() >= lat_lo - 1e-9 and ys.max() <= lat_hi + 1e-9)
    audit = {
        "rule": "cell centre inside the polygon (even-odd, holes honoured)",
        "n_polygons": len(polys),
        "n_vertices": int(sum(len(r) for p in polys for r in p)),
        "polygon_bounds_lon_lat": [float(xs.min()), float(ys.min()),
                                   float(xs.max()), float(ys.max())],
        "polygon_area_km2_spherical": pa,
        "n_cells": int(m.sum()),
        "mask_area_km2": float(w[m].sum()),
        "mask_minus_polygon_pct": (100.0 * (float(w[m].sum()) - pa) / pa) if pa else None,
        # A polygon reaching past the grid's outer edges loses that part of
        # the basin, and the mask area alone would not say so.
        "polygon_inside_grid": inside,
    }
    return m, audit


def _grid_lat_range(grid: Grid) -> Tuple[float, float]:
    try:
        e = cell_edges(grid)
        return float(e["south"].min()), float(e["north"].max())
    except RegionError:
        return float(np.nanmin(grid.lat)), float(np.nanmax(grid.lat))


# ─────────────────────────────────────────────────────────────────────────────
# STATIONS
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2, radius_km: float = EARTH_RADIUS_KM):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def cell_edges(grid: Grid, corners: Optional[Tuple] = None) -> Dict[str, np.ndarray]:
    """West, east, south and north edge of every cell.

    From a domain file's corners (xv, yv, shape grid + (nv,)) when given, else
    half-way between the centres of a regular grid. Longitudes are continuous
    within a cell (and along a regular grid's rows), so a cell ending at 180 E
    is as wide as its neighbours and a grid running across 180 E keeps its
    order; `west` may lie below -180 or `east` above 180 there. Cells must be
    the latitude-longitude rectangles their corners span; anything else is
    refused rather than approximated.
    """
    if corners is not None:
        xv = np.asarray(corners[0], dtype=np.float64)
        yv = np.asarray(corners[1], dtype=np.float64)
        if xv.shape[:-1] != grid.shape or yv.shape[:-1] != grid.shape:
            raise RegionError(f"corners have shape {xv.shape}, the grid {grid.shape}")
        x0 = to_pm180(xv[..., :1])
        xv = x0 + to_pm180(xv - xv[..., :1])          # continuous within each cell
        for a in (xv, yv):
            lo, hi = a.min(axis=-1, keepdims=True), a.max(axis=-1, keepdims=True)
            # ABSOLUTE tolerance only (1e-6 deg, about 0.1 m): numpy's default
            # relative one would forgive a corner 0.002 deg out at 238 E.
            near = np.isclose(a, lo, rtol=0.0, atol=1e-6) | \
                np.isclose(a, hi, rtol=0.0, atol=1e-6)
            if not np.all(near):
                raise RegionError("cells are not latitude-longitude rectangles")
        e = {"west": xv.min(axis=-1), "east": xv.max(axis=-1),
             "south": yv.min(axis=-1), "north": yv.max(axis=-1)}
        if np.any(e["east"] - e["west"] >= 180.0):
            raise RegionError("a cell spans 180 degrees of longitude or more")
        return e
    if not grid.regular:
        raise RegionError("an unstructured grid needs the domain file's corners")
    if min(grid.shape) < 2:
        raise RegionError(
            "a grid one row or one column wide has no spacing to infer cell edges "
            "from; pass the domain file's corners")

    def _edges(c):
        mid = (c[1:] + c[:-1]) / 2.0
        return (np.concatenate([[c[0] - (mid[0] - c[0])], mid]),
                np.concatenate([mid, [c[-1] + (c[-1] - mid[-1])]]))
    lo_x, hi_x = _edges(np.unwrap(np.asarray(grid.lon_axis, float), period=360.0))
    lo_y, hi_y = _edges(np.asarray(grid.lat_axis, float))
    west, south = np.meshgrid(np.minimum(lo_x, hi_x), np.minimum(lo_y, hi_y))
    east, north = np.meshgrid(np.maximum(lo_x, hi_x), np.maximum(lo_y, hi_y))
    return {"west": west, "east": east, "south": south, "north": north}


def _cell_index(grid: Grid, flat: int) -> Dict:
    if grid.regular:
        j, i = np.unravel_index(flat, grid.shape)
        return {"row": int(j), "col": int(i), "flat": int(flat)}
    return {"flat": int(flat)}


def station_cells(grid: Grid, stations: Iterable[Dict], topo=None,
                  radius_km: float = 3.0, corners: Optional[Tuple] = None,
                  within: Optional[np.ndarray] = None) -> List[Dict]:
    """For each station: the cell that contains it, and the cell within
    radius_km whose elevation is closest to the station's.

    A station belongs to a cell when west <= lon < east and south <= lat <
    north (longitudes compared modulo 360), so a station on a shared edge goes
    to exactly one cell. The elevation match ranks candidates by |cell
    elevation - station elevation|, then by distance; `within` (a mask) can
    restrict the candidates. Stations need 'id', 'lat', 'lon' and, for the
    match, 'elevation_m'. Every record carries the same keys; those that do
    not apply are None.
    """
    e = cell_edges(grid, corners)
    west, east = e["west"].ravel(), e["east"].ravel()
    south, north = e["south"].ravel(), e["north"].ravel()
    lat_c, lon_c = grid.lat.ravel(), grid.lon.ravel()
    z = None if topo is None else np.asarray(topo, dtype=np.float64).ravel()
    ok_cand = np.ones(lat_c.size, dtype=bool)
    if within is not None:
        ok_cand &= np.asarray(within, dtype=bool).ravel()
    out = []
    for st in stations:
        lat, lon = float(st["lat"]), float(to_pm180(st["lon"]))
        elev = st.get("elevation_m")
        rec: Dict = {"id": st.get("id"), "name": st.get("name"), "lat": lat,
                     "lon": lon, "elevation_m": elev}
        in_lon = np.mod(lon - west, 360.0) < (east - west)
        hit = np.flatnonzero(in_lon & (south <= lat) & (lat < north))
        if len(hit) > 1:
            raise RegionError(f"station {st.get('id')} falls in {len(hit)} cells; "
                              f"the cell edges overlap")
        d = haversine_km(lat, lon, lat_c, lon_c)
        rec["inside_grid"] = bool(len(hit) == 1)
        rec["nearest_centre_km"] = float(np.nanmin(d))
        rec["containing"] = None
        if len(hit) == 1:
            k = int(hit[0])
            rec["containing"] = dict(_cell_index(grid, k), distance_km=float(d[k]),
                                     elevation_m=None if z is None else float(z[k]))
        rec["n_within_radius"] = None
        rec["elevation_matched"] = None
        if z is not None and elev is not None and np.isfinite(float(elev)):
            cand = np.flatnonzero((d <= radius_km) & np.isfinite(z) & ok_cand)
            rec["n_within_radius"] = int(len(cand))
            if len(cand):
                dz = np.abs(z[cand] - float(elev))
                best = int(cand[np.lexsort((d[cand], dz))[0]])
                rec["elevation_matched"] = dict(
                    _cell_index(grid, best), distance_km=float(d[best]),
                    elevation_m=float(z[best]),
                    elevation_difference_m=float(z[best] - float(elev)))
        rec["same_cell"] = (rec["containing"]["flat"] == rec["elevation_matched"]["flat"]
                            if rec["containing"] is not None
                            and rec["elevation_matched"] is not None else None)
        out.append(rec)
    return out


def cell_mask(grid: Grid, flat_index: int) -> np.ndarray:
    m = np.zeros(grid.n_cells, dtype=bool)
    m[int(flat_index)] = True
    return m.reshape(grid.shape)


# ─────────────────────────────────────────────────────────────────────────────
# MASKS FROM OTHER GRIDDED FILES
# ─────────────────────────────────────────────────────────────────────────────
_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less,
        "<=": np.less_equal, "==": np.equal, "!=": np.not_equal}


def mask_from_field(path, var: str, grid: Grid, op: str = ">", value: float = 0.0,
                    over: str = "any", nan_as: bool = False,
                    tol_deg: float = 1e-4) -> np.ndarray:
    """Cells where `var op value`, combined over its leading axes by any/all.

    The file must carry cell centres (lat/lon, LATIXY/LONGXY or yc/xc) that
    match the history grid, and `var` must end on the same dimensions as those
    centres, in the same order: a field stored (lon, lat) on a square grid
    would otherwise pass every size check and come back transposed. A file on
    other cells is refused, not regridded. NaN, fill and packed fill count as
    `nan_as` (False: a missing value never selects a cell, and with over='all'
    it deselects it).
    """
    import netCDF4
    if op not in _OPS:
        raise RegionError(f"op must be one of {sorted(_OPS)}")
    if over not in ("any", "all"):
        raise RegionError("over must be 'any' or 'all'")
    with netCDF4.Dataset(path) as f:
        for la, lo in (("lat", "lon"), ("LATIXY", "LONGXY"), ("yc", "xc")):
            if la in f.variables and lo in f.variables:
                grid.check_coordinates(read_clean(f[la]), read_clean(f[lo]),
                                       what=f"{Path(path).name}", tol_deg=tol_deg)
                if f[la].ndim == 1 and f[lo].ndim == 1 and \
                        f[la].dimensions != f[lo].dimensions:
                    cdims = (f[la].dimensions[0], f[lo].dimensions[0])
                else:
                    cdims = tuple(f[la].dimensions)
                break
        else:
            raise RegionError(f"{Path(path).name}: no cell centres to check against")
        if var not in f.variables:
            raise RegionError(f"{Path(path).name} has no {var}")
        vdims = tuple(f[var].dimensions)
        if vdims[len(vdims) - len(cdims):] != cdims:
            raise RegionError(
                f"{var} is stored on {vdims}, but the cell centres run {cdims}; "
                f"its last dimensions must be those, in that order")
        a = read_clean(f[var])
    if a.shape[a.ndim - len(grid.shape):] != grid.shape:
        raise RegionError(f"{var} has shape {a.shape}; its last axes must be {grid.shape}")
    with np.errstate(invalid="ignore"):
        sel = _OPS[op](a, value)
    sel = np.where(np.isfinite(a), sel, nan_as)
    lead = tuple(range(a.ndim - len(grid.shape)))
    if lead:
        sel = sel.any(axis=lead) if over == "any" else sel.all(axis=lead)
    return np.asarray(sel, dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# THE MASK RECORD
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Mask:
    name: str
    array: np.ndarray
    rule: str
    sources: Tuple[str, ...]
    n_cells: int
    area_km2: float


def _varname(name: str, taken: set) -> str:
    base = "mask_" + (re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "region")
    out, k = base, 2
    while out in taken:
        out, k = f"{base}_{k}", k + 1
    return out


def _as_mask(name: str, array) -> np.ndarray:
    """A boolean copy, refusing anything that is not plainly inside/outside.

    np.asarray(x, bool) would call NaN and a netCDF fill value inside the
    region, so a float 1/NaN mask or a masked integer mask would claim every
    cell; both are refused with what was found.
    """
    if np.ma.isMaskedArray(array):
        if np.ma.getmaskarray(array).any():
            raise RegionError(f"{name}: the mask has masked (fill) cells; decide "
                              f"whether they are inside before adding it")
        array = np.ma.getdata(array)
    a = np.asarray(array)
    if a.dtype != bool:
        bad = ~np.isin(a, (0, 1))
        if bad.any():
            sample = sorted({str(v) for v in np.asarray(a[bad]).ravel()[:5]})
            raise RegionError(f"{name}: a mask holds 0/1 or True/False only; found "
                              f"{', '.join(sample)}")
    return a.astype(bool)


class MaskSet:
    """Named regions on one grid, each with its rule and sources.

    Region names are the labels every reduced number carries, so they are
    meant to be read: 'watershed, vegetated', '1000-1500 m'. Stored masks are
    read-only: a region edited in place would no longer match its own cell
    count, area or saved record.
    """

    def __init__(self, grid: Grid):
        self.grid = grid
        self._m: "OrderedDict[str, Mask]" = OrderedDict()

    def add(self, name: str, array, rule: str, sources: Sequence[str] = ()) -> Mask:
        a = _as_mask(name, array)
        if a.shape != self.grid.shape:
            raise RegionError(f"{name}: mask shape {a.shape}, grid {self.grid.shape}")
        if name in self._m:
            raise RegionError(f"region {name!r} is already defined")
        a = a.copy()
        a.flags.writeable = False
        m = Mask(name=name, array=a, rule=rule,
                 sources=tuple(str(s) for s in sources), n_cells=int(a.sum()),
                 area_km2=float(self.grid.weight_km2[a].sum()))
        self._m[name] = m
        return m

    def combine(self, name: str, include: Sequence[str],
                exclude: Sequence[str] = ()) -> Mask:
        """A new region: every `include` region at once, none of `exclude`."""
        if not include:
            raise RegionError("combine needs at least one region to include")
        a = np.ones(self.grid.shape, dtype=bool)
        for n in include:
            a &= self[n]
        for n in exclude:
            a &= ~self[n]
        rule = " & ".join(f"[{n}]" for n in include)
        if exclude:
            rule += "".join(f" & not [{n}]" for n in exclude)
        srcs: List[str] = []
        for n in list(include) + list(exclude):
            srcs += [s for s in self._m[n].sources if s not in srcs]
        return self.add(name, a, rule, srcs)

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._m:
            raise RegionError(f"no region {name!r}; defined: {list(self._m)}")
        return self._m[name].array

    def __contains__(self, name: str) -> bool:
        return name in self._m

    def mask(self, name: str) -> Mask:
        self[name]
        return self._m[name]

    def names(self) -> List[str]:
        return list(self._m)

    def summary(self) -> "OrderedDict[str, Dict]":
        return OrderedDict((n, {"n_cells": m.n_cells, "area_km2": m.area_km2,
                                "rule": m.rule, "sources": list(m.sources)})
                           for n, m in self._m.items())

    def to_netcdf(self, path) -> Path:
        """Every region in one file, written atomically; read back with
        MaskSet.from_netcdf on the same grid."""
        import netCDF4
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        g = self.grid
        with netCDF4.Dataset(tmp, "w") as d:
            d.title = "Region masks on the cells of a gridded ELM run"
            d.created = datetime.now().isoformat(timespec="seconds")
            d.grid_dims = " ".join(g.dims)
            d.weight = "area * landfrac from the history file, km2"
            for name, n in zip(g.dims, g.shape):
                d.createDimension(name, n)
            if g.regular:
                d.createVariable("lat", "f8", ("lat",))[:] = g.lat_axis
                d.createVariable("lon", "f8", ("lon",))[:] = g.lon_axis
            else:
                d.createVariable("lat", "f8", g.dims)[:] = g.lat
                d.createVariable("lon", "f8", g.dims)[:] = g.lon
            d["lon"].note = "degrees east in [-180, 180)"
            wv = d.createVariable("weight_km2", "f8", g.dims)
            wv[:] = g.weight_km2
            taken: set = set()
            for name, m in self._m.items():
                vn = _varname(name, taken)
                taken.add(vn)
                v = d.createVariable(vn, "i1", g.dims)
                v[:] = m.array.astype(np.int8)
                v.region = name
                v.rule = m.rule
                v.sources = json.dumps(list(m.sources))
                v.n_cells = np.int64(m.n_cells)
                v.area_km2 = m.area_km2
        os.replace(tmp, out)
        return out

    @classmethod
    def from_netcdf(cls, path, grid: Grid) -> "MaskSet":
        import netCDF4
        ms = cls(grid)
        with netCDF4.Dataset(path) as d:
            d.set_auto_mask(False)
            lat = np.asarray(d["lat"][:], dtype=np.float64)
            lon = np.asarray(d["lon"][:], dtype=np.float64)
            grid.check_coordinates(lat, lon, what=Path(path).name, tol_deg=1e-9)
            for vn, v in d.variables.items():
                if not vn.startswith("mask_"):
                    continue
                ms.add(v.region, np.asarray(v[:]) == 1, v.rule, json.loads(v.sources))
        return ms


# ─────────────────────────────────────────────────────────────────────────────
# REDUCING A STREAM TO REGIONAL SERIES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Field:
    """A quantity to reduce: one history variable, or the per-cell sum of
    several (evapotranspiration = QVEGT + QVEGE + QSOIL). A cell where any
    part is fill is fill for the sum, never a partial total.

    to_per_day turns a per-second rate into a per-day one (x 86400, units
    renamed); a pure unit change. layer picks one level of a layered variable.
    """
    name: str
    parts: Tuple[str, ...] = ()
    to_per_day: bool = True
    layer: Optional[int] = None

    @property
    def sources(self) -> Tuple[str, ...]:
        return self.parts or (self.name,)


@dataclass
class RegionSeries:
    """Area-weighted regional means of each field, one value per record.

    values[(region, field)] and valid_fraction[(region, field)] are arrays
    over `keys`. valid_fraction is the share of the region's weight that had
    a value at that record; below 1 the mean covers fewer cells than named.
    per[field] is 'day' for a rate converted to per day, 'second' for a rate
    left per second, None for anything else.
    """
    stream: str
    cadence: str
    calendar: str
    steps_per_day: Optional[int]
    keys: List[Tuple]
    regions: List[str]
    fields: List[str]
    units: Dict[str, str]
    values: Dict[Tuple[str, str], np.ndarray]
    valid_fraction: Dict[Tuple[str, str], np.ndarray]
    region_table: Dict[str, Dict]
    field_parts: Dict[str, List[str]] = field(default_factory=dict)
    per: Dict[str, Optional[str]] = field(default_factory=dict)

    @property
    def labels(self) -> List[str]:
        return [label_text(k, self.cadence) for k in self.keys]

    def series(self, region: str, fld: str) -> np.ndarray:
        if (region, fld) not in self.values:
            raise RegionError(f"no series for region {region!r}, field {fld!r}")
        return self.values[(region, fld)]

    def to_frame(self):
        """Tidy table: label | region | variable | value | units | valid_area_fraction."""
        import pandas as pd
        rows = []
        labels = self.labels
        for (r, f), v in self.values.items():
            vf = self.valid_fraction[(r, f)]
            rows.append(pd.DataFrame({"label": labels, "region": r, "variable": f,
                                      "value": v, "units": self.units[f],
                                      "valid_area_fraction": vf}))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def record_days(self, keys: Sequence[Tuple]) -> np.ndarray:
        """How many days each record covers: 1 daily, 1/steps per day
        sub-daily, the month's length monthly."""
        if self.cadence == "daily":
            return np.ones(len(keys))
        if self.cadence == "subdaily":
            return np.full(len(keys), 1.0 / float(self.steps_per_day))
        return np.array([float(days_in_month(k[0], k[1], self.calendar)) for k in keys])

    def _period_keys(self, year: int, months: Optional[Sequence[int]],
                     water_year: bool) -> List[Tuple]:
        if water_year:
            ks = [k for k in expected_keys(year - 1, self.cadence, self.calendar,
                                           self.steps_per_day) if k[1] >= 10]
            ks += [k for k in expected_keys(year, self.cadence, self.calendar,
                                            self.steps_per_day) if k[1] <= 9]
        else:
            ks = expected_keys(year, self.cadence, self.calendar, self.steps_per_day)
        if months:
            ms = set(int(m) for m in months)
            ks = [k for k in ks if k[1] in ms]
        return ks

    def by_year(self, region: str, fld: str, years: Sequence[int], how: str = "sum",
                months: Optional[Sequence[int]] = None, water_year: bool = False
                ) -> Tuple[Dict[int, Optional[float]], Dict[int, str]]:
        """({year: value}, {year: why it is None}) over each year's records.

        how = 'sum', 'mean', 'max' or 'min'. EACH RECORD COUNTS FOR THE DAYS
        IT COVERS: 'sum' is the time integral of a per-day rate (sum of value
        x days), 'mean' the time-weighted mean, so a monthly or 3-hourly
        series gives the same annual total as a daily one. For daily records
        they are the plain sum and mean. 'sum' is refused for a rate left per
        second, which would come out 86,400 times too small. months restricts
        the period (e.g. [4, 5, 6, 7]); water_year makes year Y run 1 October
        Y-1 to 30 September Y. A year missing any record of the period, or
        with any record NaN, gets None and a reason: a total over part of a
        period is not that period's total.
        """
        if how not in ("sum", "mean", "max", "min"):
            raise RegionError("how must be sum, mean, max or min")
        if how == "sum" and self.per.get(fld) == "second":
            raise RegionError(
                f"{fld} is a rate per second; a total over days needs it per day "
                f"(reduce it with Field(..., to_per_day=True))")
        v = self.series(region, fld)
        pos = {k: i for i, k in enumerate(self.keys)}
        out: Dict[int, Optional[float]] = {}
        why: Dict[int, str] = {}
        for y in years:
            ks = self._period_keys(int(y), months, water_year)
            idx = [pos.get(k) for k in ks]
            miss = [k for k, i in zip(ks, idx) if i is None]
            if miss:
                out[y], why[y] = None, (f"{len(miss)} of {len(ks)} records not read "
                                        f"(first {label_text(miss[0], self.cadence)})")
                continue
            x = v[np.array(idx)]
            if not np.all(np.isfinite(x)):
                nb = int((~np.isfinite(x)).sum())
                out[y], why[y] = None, f"{nb} of {len(ks)} records have no value"
                continue
            if how in ("sum", "mean"):
                d = self.record_days(ks)
                out[y] = float(np.sum(x * d)) if how == "sum" else \
                    float(np.sum(x * d) / np.sum(d))
            else:
                out[y] = float(np.max(x) if how == "max" else np.min(x))
        return out, why


def reduce_regions(stream: Stream, fields: Sequence, masks: MaskSet,
                   years: Sequence[int], regions: Optional[Sequence[str]] = None,
                   require_complete: bool = True,
                   max_records: Optional[int] = None) -> RegionSeries:
    """Area-weighted, NaN-aware means of each field over each region, record
    by record, for the given calendar years (in calendar order).

    Only the rectangle around the regions is read, one part and one piece of
    the year at a time, and a part is kept only while another field still
    needs it. The mean at a record is sum(w x) / sum(w) over the region's
    cells that have a value; valid_fraction says how much of the region that
    was. A repeated region, field or year is refused: each would silently
    misalign the series against their labels.
    """
    flds = [f if isinstance(f, Field) else Field(str(f)) for f in fields]
    names = [f.name for f in flds]
    if len(set(names)) != len(names):
        raise RegionError(f"field names repeat: {names}")
    regs = masks.names() if regions is None else list(regions)
    if not regs:
        raise RegionError("no regions to reduce over")
    if len(set(regs)) != len(regs):
        raise RegionError(f"region names repeat: {regs}")
    ys = [int(y) for y in years]
    if len(set(ys)) != len(ys):
        raise RegionError(f"years repeat: {ys}")
    ys = sorted(ys)
    if max_records is not None and int(max_records) < 1:
        raise RegionError(f"max_records must be at least 1, not {max_records}")
    g = stream.grid
    if not masks.grid.same_cells(g):
        raise RegionError("the masks were built on a different grid from this stream")
    union = np.zeros(g.shape, dtype=bool)
    for r in regs:
        union |= masks[r]
    if not union.any():
        raise RegionError(f"the regions {regs} hold no cells")
    empty = [r for r in regs if not masks[r].any()]
    if empty:
        raise RegionError(f"regions with no cells: {empty}")

    window = None
    if g.regular:
        rows = np.flatnonzero(union.any(axis=1))
        cols = np.flatnonzero(union.any(axis=0))
        window = (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)
        r0, r1, c0, c1 = window
        sub_union = union[r0:r1, c0:c1].ravel()
        sub_w = g.weight_km2[r0:r1, c0:c1].ravel()[sub_union]
        sub_masks = {r: masks[r][r0:r1, c0:c1].ravel()[sub_union] for r in regs}
    else:
        sub_union = union.ravel()
        sub_w = g.weight_km2.ravel()[sub_union]
        sub_masks = {r: masks[r].ravel()[sub_union] for r in regs}
    W = np.stack([np.where(sub_masks[r], sub_w, 0.0) for r in regs], axis=1)
    wtot = W.sum(axis=0)
    if np.any(wtot <= 0):
        raise RegionError("a region has cells but zero total weight (landfrac 0?)")

    # Units, checked before anything is read.
    units: Dict[str, str] = {}
    scale: Dict[str, float] = {}
    per: Dict[str, Optional[str]] = {}
    for f in flds:
        us = {p: stream.units(p) for p in f.sources}
        ps = {p: stream.per_second(p) for p in f.sources}
        if len(set(us.values())) > 1 or len(set(ps.values())) > 1:
            raise RegionError(f"{f.name}: parts have different units {us}; "
                              f"they cannot be summed")
        p0 = f.sources[0]
        for p in f.sources:
            lev = stream.levels(p)
            if lev and f.layer is None:
                raise RegionError(f"{p} is layered {lev}; give Field(layer=...)")
            if not lev and f.layer is not None:
                raise RegionError(f"{p} has no levels; layer={f.layer} does not apply")
        if ps[p0] and f.to_per_day:
            scale[f.name], units[f.name], per[f.name] = \
                SECONDS_PER_DAY, per_day_units(p0, us[p0]), "day"
        else:
            scale[f.name], units[f.name] = 1.0, us[p0]
            per[f.name] = "second" if ps[p0] else None

    uses: Dict[Tuple[str, Optional[int]], int] = {}
    for f in flds:
        for p in f.sources:
            uses[(p, f.layer)] = uses.get((p, f.layer), 0) + 1

    keys_all: List[Tuple] = []
    vals = {(r, f.name): [] for r in regs for f in flds}
    vfrac = {(r, f.name): [] for r in regs for f in flds}
    if max_records is None:
        max_records = 31 * (stream.steps_per_day or 1) if stream.cadence == "subdaily" \
            else 10 ** 9
    for y in ys:
        have, missing = stream.year_records(y)
        if not have:
            raise RegionError(f"stream {stream.name}: no records in {y}")
        if missing and require_complete:
            raise GriddedReadError(stream._missing_message(y, missing))
        for a in range(0, len(have), int(max_records)):
            part = have[a:a + int(max_records)]
            left = dict(uses)
            cache: Dict = {}
            for f in flds:
                total = None
                for p in f.sources:
                    ck = (p, f.layer)
                    x = cache.get(ck)
                    if x is None:
                        x = stream.read_records(p, part, window=window, layer=f.layer)
                        x = x.reshape(len(part), -1)[:, sub_union]
                    left[ck] -= 1
                    if left[ck] > 0:
                        cache[ck] = x
                    else:
                        cache.pop(ck, None)
                    # A copy: the cached part must survive this field's scaling.
                    total = x.copy() if total is None else total + x
                total *= scale[f.name]
                ok = np.isfinite(total)
                num = np.where(ok, total, 0.0) @ W
                den = ok.astype(np.float64) @ W
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
                for j, r in enumerate(regs):
                    vals[(r, f.name)].append(mean[:, j])
                    vfrac[(r, f.name)].append(den[:, j] / wtot[j])
            keys_all += [rec.key for rec in part]

    values = {k: np.concatenate(v) for k, v in vals.items()}
    vf = {k: np.concatenate(v) for k, v in vfrac.items()}
    for k, v in values.items():
        if len(v) != len(keys_all):
            raise RegionError(f"internal: {k} has {len(v)} values for "
                              f"{len(keys_all)} records")
    table = OrderedDict()
    for r in regs:
        m = masks.mask(r)
        table[r] = {"n_cells": m.n_cells, "area_km2": m.area_km2, "rule": m.rule,
                    "sources": list(m.sources)}
    return RegionSeries(
        stream=stream.name, cadence=stream.cadence, calendar=stream.calendar,
        steps_per_day=stream.steps_per_day, keys=keys_all, regions=regs,
        fields=names, units=units, values=values, valid_fraction=vf,
        region_table=table, field_parts={f.name: list(f.sources) for f in flds},
        per=per)


# ─────────────────────────────────────────────────────────────────────────────
# NORMALS, ANOMALIES AND TIMING
# ─────────────────────────────────────────────────────────────────────────────
def anomalies(by_year: Dict[int, Optional[float]], normal_years: Sequence[int],
              target_years: Sequence[int], percent: bool = True) -> Dict:
    """The normal (mean of normal_years) and each target year against it.

    anomaly = value - normal; anomaly_pct = 100 x anomaly / normal (None when
    percent is off, e.g. for temperature in K, or when the normal is zero or
    negligible beside the values). rank_lowest_first is 1 plus the number of
    years with a lower value, so tied years share a rank (tied_with names
    them). A normal year without a value is refused: a normal of four years
    is not a normal of five.
    """
    miss = [y for y in normal_years if by_year.get(y) is None]
    if miss:
        raise RegionError(f"normal years without a value: {miss}")
    nv = np.array([by_year[y] for y in normal_years], dtype=np.float64)
    normal = float(nv.mean())
    have = {y: v for y, v in by_year.items() if v is not None}
    scale = max([abs(v) for v in have.values()] + [abs(normal)])
    out = {"normal": normal, "normal_years": list(normal_years),
           "normal_sd": float(nv.std(ddof=1)) if nv.size > 1 else None,
           "normal_min": float(nv.min()), "normal_max": float(nv.max()),
           "by_year": dict(by_year), "years": {}}
    for y in target_years:
        v = by_year.get(y)
        if v is None:
            out["years"][y] = None
            continue
        a = v - normal
        pct_ok = percent and normal != 0.0 and abs(normal) > 1e-12 * scale
        out["years"][y] = {
            "value": v, "anomaly": a,
            "anomaly_pct": (100.0 * a / normal) if pct_ok else None,
            "rank_lowest_first": 1 + sum(1 for w in have.values() if w < v),
            "tied_with": sorted(z for z, w in have.items() if w == v and z != y),
            "of": len(have),
            "position": ("below the normal-years minimum" if v < nv.min() else
                         "above the normal-years maximum" if v > nv.max() else
                         "within the normal-years range")}
    return out


def day_of_year(key: Tuple, calendar: str) -> int:
    y, m, d = key[0], key[1], key[2]
    return sum(days_in_month(y, k, calendar) for k in range(1, m)) + d


def centre_of_timing(rs: RegionSeries, region: str, fld: str, year: int,
                     months: Sequence[int]) -> Optional[Dict]:
    """The first record by which half of the period's total has passed.

    The Naches report's half-flow date: daily runoff, January-September,
    first day the running total reaches 50 %. Each record counts for the days
    it covers, as in by_year. None when the period is incomplete, has a NaN,
    or totals zero or less.
    """
    pos = {k: i for i, k in enumerate(rs.keys)}
    ks = rs._period_keys(int(year), months, False)
    idx = [pos.get(k) for k in ks]
    if any(i is None for i in idx):
        return None
    x = rs.series(region, fld)[np.array(idx)] * rs.record_days(ks)
    if not np.all(np.isfinite(x)) or x.sum() <= 0:
        return None
    c = np.cumsum(x) / x.sum()
    k = ks[int(np.argmax(c >= 0.5))]
    return {"label": label_text(k, rs.cadence), "day_of_year": day_of_year(k, rs.calendar)}


def to_volume_m3s(mean_rate, area_km2: float, units: str) -> np.ndarray:
    """A region-mean water flux as a volume flux in m3/s.

    units must say what the rate is ('mm/s' as stored, or 'mm/d' as
    reduce_regions reports a converted rate), because the two differ by 86,400
    and the numbers alone cannot tell them apart.
    """
    u = (units or "").replace(" ", "").lower()
    per_s = {"mm/s": 1.0, "mms-1": 1.0, "mm/d": 1.0 / SECONDS_PER_DAY,
             "mmd-1": 1.0 / SECONDS_PER_DAY}
    if u not in per_s:
        raise RegionError(f"to_volume_m3s needs a rate in mm/s or mm/d, not {units!r}")
    return np.asarray(mean_rate, dtype=np.float64) * per_s[u] * float(area_km2) * 1.0e3
