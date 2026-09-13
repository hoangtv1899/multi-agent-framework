#!/usr/bin/env python3
"""The drainage datum per column: how far below its surface the stream sits.

    python tools/drainage_datum.py <run_dir> --source hand --out <json>

Section 4 of docs/coupling/lateral_sink_design.md. The lateral sink drains a
column toward a datum below its surface. Two candidates are computed and
recorded for every column, and --source picks the one that becomes
sink_datum_m:

  hand    height above the nearest drainage (HAND) from D8 routing on the
          1-km CONUS TOPO the warm start was cut from, floored at half the
          column's own sub-grid relief (STD_ELEV) because a 1-km cell mean
          cannot see the channel a valley cell drains to;
  conus2  the CONUS2 steady-state water-table depth already sampled per
          column (columns.json water_table_m, else the run's wtd_conus2.tif).

The stream threshold A is fixed ONCE per basin by the gauge check: the run's
in-basin gauges must land on a stream cell whose TOPO matches their
altitude_m within the cell-mean error. A and the score are recorded.

Read-only on the run: the report goes wherever --out says, never into the
run directory. Nothing here touches the network.
"""
import argparse
import glob
import heapq
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core import static_wtd                                  # noqa: E402

# The 1-km CONUS surface dataset (design section 4): 2-degree latitude
# bands, lat{N} spanning 23 + 2N to 25 + 2N degrees north, 240 rows of
# 1/120 degree by 6960 columns; LONGXY is degrees east on 0-360.
CONUS_ROOT = Path("/compyfs/bish218/conus1k/netcdf")
BAND_PATTERN = "surfdata_conus_1k_small_lat{n}_*.nc"
BAND_LAT0_DEG = 23.0
BAND_WIDTH_DEG = 2.0
CELL_DEG = 1.0 / 120.0

# Grid spacing for routing: dy = 0.93 km, dx = 0.93 km * cos(lat); a cell's
# plan area is dx * dy, so accumulation comes out in km^2.
CELL_KM = 0.93

# The read window is the basin bbox plus this margin on every side, so a
# path that leaves the bbox still finds its stream before it meets the edge.
MARGIN_DEG = 0.1

# Priority-flood epsilon: a filled cell sits at least this far above the
# cell it was flooded from, so D8 always has a downhill direction across a
# filled pit. Routing only; HAND is read from the raw TOPO.
FILL_EPS_M = 1e-3

# The gauge check: candidate stream thresholds, and how far a 1-km cell mean
# may sit from a gauge's altitude_m (the Naches-scale cell-mean error).
STREAM_THRESHOLDS_KM2 = (5.0, 10.0, 15.0, 25.0, 40.0)
GAUGE_TOPO_TOL_M = 60.0
PICKS = ("smallest", "largest")

# sink_datum_m = max(HAND, STD_ELEV_FLOOR * STD_ELEV) for source hand.
STD_ELEV_FLOOR = 0.5
SOURCES = ("hand", "conus2")

NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1))


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


# ── the 1-km TOPO over the basin ──────────────────────────────────────────

def band_index(lat):
    """Which 2-degree band a latitude falls in (lat11 spans 45 to 47 N)."""
    return int(math.floor((float(lat) - BAND_LAT0_DEG) / BAND_WIDTH_DEG))


def _swap_band(path, n):
    """The same file name with the band number swapped for n."""
    return path.with_name(re.sub(r"lat\d+_", f"lat{n}_", path.name))


def band_files(run_dir, bands):
    """The surfdata file per band, {n: Path}, from the warm start's record.

    warmstart.json names the conus_surfdata each column was cut from, so a
    band a column sits in is read from exactly that file. A band only the
    margin touches, or the far half of a basin that straddles a band edge,
    takes a recorded file name with the band number swapped, and the
    design's pattern on /compyfs (newest name) only when that does not
    exist. Every choice is recorded in the report under surfdata_files.
    """
    ws = Path(run_dir) / "warmstart" / "warmstart.json"
    if not ws.is_file():
        raise ValueError(f"{ws} is missing: without the warm start the run "
                         f"does not say which CONUS surfdata it was cut from")
    recorded = {}
    for rec in json.loads(ws.read_text()).values():
        if rec.get("conus_surfdata"):
            p = Path(rec["conus_surfdata"])
            recorded.setdefault(rec.get("conus_band") or
                                f"lat{band_index(rec['donor_lat'])}", p)
    out = {}
    for n in bands:
        if f"lat{n}" in recorded:
            out[n] = recorded[f"lat{n}"]
            continue
        swapped = [_swap_band(p, n) for p in recorded.values()]
        found = [p for p in swapped if p.is_file()]
        if not found:
            found = [Path(p) for p in
                     sorted(glob.glob(str(CONUS_ROOT / BAND_PATTERN.format(n=n))))]
        if not found:
            raise ValueError(f"no surfdata for band lat{n}: neither "
                             f"{[str(p) for p in swapped]} nor "
                             f"{CONUS_ROOT / BAND_PATTERN.format(n=n)} exists")
        out[n] = found[-1]
    return out


def _band_window(path, lo_lat, hi_lat, lo_lon, hi_lon):
    """One band's TOPO and STD_ELEV inside the window, or None if disjoint."""
    import netCDF4 as nc
    with nc.Dataset(path) as d:
        lat = np.array(d.variables["LATIXY"][:, 0], dtype=float)
        lon = np.array(d.variables["LONGXY"][0, :], dtype=float)
        rows = np.nonzero((lat >= lo_lat) & (lat <= hi_lat))[0]
        cols = np.nonzero((lon >= lo_lon) & (lon <= hi_lon))[0]
        if rows.size == 0 or cols.size == 0:
            return None
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        # The axes are separable (LATIXY constant along a row, LONGXY along
        # a column) and everything below relies on it, so check the window.
        if not (np.allclose(d.variables["LATIXY"][r0:r1, c0],
                            d.variables["LATIXY"][r0:r1, c1 - 1]) and
                np.allclose(d.variables["LONGXY"][r0, c0:c1],
                            d.variables["LONGXY"][r1 - 1, c0:c1])):
            raise ValueError(f"{path}: LATIXY/LONGXY are not a regular "
                             f"lat-lon grid over the window")
        part = {v: np.array(d.variables[v][r0:r1, c0:c1], dtype=float)
                for v in ("TOPO", "STD_ELEV")}
    part["lat"] = lat[r0:r1]
    part["lon"] = lon[c0:c1]
    part["file"] = str(path)
    return part


def read_topo(files_by_band, bbox, margin_deg=MARGIN_DEG):
    """TOPO, STD_ELEV and the axes over the bbox plus margin, bands stitched.

    Returns {topo, std_elev, lat, lon, files}: 2-D arrays with rows running
    south to north, 1-D axes, lon in degrees east on 0-360 as the file has
    it. Two bands are stitched only when they abut cell to cell and share
    the longitude axis; anything else is refused rather than padded.
    """
    lo_lat, hi_lat = bbox["min_lat"] - margin_deg, bbox["max_lat"] + margin_deg
    lo_lon = (bbox["min_lon"] - margin_deg) % 360.0
    hi_lon = (bbox["max_lon"] + margin_deg) % 360.0
    if not lo_lon < hi_lon:
        raise ValueError(f"the window {lo_lon}..{hi_lon} E wraps the meridian")
    parts = [p for p in (_band_window(f, lo_lat, hi_lat, lo_lon, hi_lon)
                         for _n, f in sorted(files_by_band.items()))
             if p is not None]
    if not parts:
        raise ValueError(f"no TOPO cell inside {bbox} plus {margin_deg} deg "
                         f"in {[str(f) for f in files_by_band.values()]}")
    parts.sort(key=lambda p: p["lat"][0])
    for a, b in zip(parts[:-1], parts[1:]):
        gap = b["lat"][0] - a["lat"][-1]
        if not np.allclose(a["lon"], b["lon"]) or abs(gap - CELL_DEG) > 1e-6:
            raise ValueError(f"{a['file']} and {b['file']} do not abut: "
                             f"latitude gap {gap:.6f} deg, "
                             f"same longitudes {np.allclose(a['lon'], b['lon'])}")
    return {"topo": np.concatenate([p["TOPO"] for p in parts]),
            "std_elev": np.concatenate([p["STD_ELEV"] for p in parts]),
            "lat": np.concatenate([p["lat"] for p in parts]),
            "lon": parts[0]["lon"],
            "files": [p["file"] for p in parts]}


def cell_of(lat_axis, lon_axis, lat, lon):
    """Row and column of the cell holding a point; refuses one outside."""
    lon = float(lon) % 360.0
    i = int(np.argmin(np.abs(lat_axis - lat)))
    j = int(np.argmin(np.abs(lon_axis - lon)))
    if (abs(lat_axis[i] - lat) > 0.5 * CELL_DEG + 1e-6 or
            abs(lon_axis[j] - lon) > 0.5 * CELL_DEG + 1e-6):
        raise ValueError(f"({lat}, {lon}) lies outside the TOPO window "
                         f"{lat_axis[0]:.4f}..{lat_axis[-1]:.4f} N, "
                         f"{lon_axis[0]:.4f}..{lon_axis[-1]:.4f} E")
    return i, j


# ── routing ───────────────────────────────────────────────────────────────

def _is_seed(finite, i, j):
    """On the grid edge, or beside a hole: where the flood starts."""
    ny, nx = finite.shape
    if i in (0, ny - 1) or j in (0, nx - 1):
        return True
    return any(not finite[i + di, j + dj] for di, dj in NEIGHBOURS)


def fill_pits(z, eps=FILL_EPS_M):
    """Priority-flood pit fill (Barnes et al. 2014) with an epsilon slope.

    Floods inward from the seeds, lowest first; a cell reached from a
    neighbour is raised to at least eps above it. Afterwards every finite
    cell has a strictly lower neighbour except the outlets among the seeds,
    so D8 routing never stops inside the grid. Returns the filled surface;
    the input is not touched.
    """
    z = np.asarray(z, dtype=float)
    ny, nx = z.shape
    filled = z.copy()
    finite = np.isfinite(z)
    closed = ~finite
    heap = []
    for i in range(ny):
        for j in range(nx):
            if finite[i, j] and _is_seed(finite, i, j):
                closed[i, j] = True
                heapq.heappush(heap, (z[i, j], i, j))
    while heap:
        zc, i, j = heapq.heappop(heap)
        for di, dj in NEIGHBOURS:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and not closed[a, b]:
                closed[a, b] = True
                filled[a, b] = max(z[a, b], zc + eps)
                heapq.heappush(heap, (filled[a, b], a, b))
    return filled


def d8_receivers(filled, lat):
    """Steepest-descent receiver per cell, flattened.

    Returns (rec, step_km, area_km2): rec is the receiver's flat index or -1
    (an outlet, or a hole), step_km the distance to it, area_km2 the cell's
    own plan area. dx follows cos(lat) row by row; dy is CELL_KM everywhere.
    """
    z = np.asarray(filled, dtype=float)
    ny, nx = z.shape
    dx = CELL_KM * np.cos(np.radians(np.asarray(lat, dtype=float)))[:, None]
    dy = np.full((ny, 1), CELL_KM)
    diag = np.hypot(dx, dy)
    rec = np.full((ny, nx), -1, dtype=np.int64)
    step = np.zeros((ny, nx))
    best = np.zeros((ny, nx))
    for di, dj in NEIGHBOURS:
        dist = dy if dj == 0 else (dx if di == 0 else diag)
        zn = np.full_like(z, np.nan)
        src = (slice(max(0, -di), ny - max(0, di)),
               slice(max(0, -dj), nx - max(0, dj)))
        dst = (slice(max(0, di), ny - max(0, -di)),
               slice(max(0, dj), nx - max(0, -dj)))
        zn[src] = z[dst]
        grad = (z - zn) / dist
        ii, jj = np.nonzero(np.isfinite(grad) & (grad > best))
        rec[ii, jj] = (ii + di) * nx + (jj + dj)
        best[ii, jj] = grad[ii, jj]
        step[ii, jj] = np.broadcast_to(dist, z.shape)[ii, jj]
    area = np.broadcast_to(dx * dy, z.shape).ravel().copy()
    return rec.ravel(), step.ravel(), area


def accumulate(filled, rec, area_km2):
    """Upslope area per cell in km^2, each cell counting its own area.

    Cells are visited from the highest down, so every donor has passed its
    total on before its receiver's turn comes.
    """
    zf = np.asarray(filled, dtype=float).ravel()
    finite = np.isfinite(zf)
    acc = np.where(finite, area_km2, 0.0).astype(float)
    for k in np.argsort(-np.where(finite, zf, -np.inf), kind="stable"):
        if rec[k] >= 0:
            acc[rec[k]] += acc[k]
    return acc


def walk_to_stream(k, rec, step_km, stream):
    """Follow receivers from cell k to the first stream cell.

    Returns (cell, path_km, end): end is "stream", or "edge" when the path
    ran off the grid or into a hole before reaching one.
    """
    km = 0.0
    n = 0
    while not stream[k]:
        if rec[k] < 0:
            return k, km, "edge"
        km += step_km[k]
        k = rec[k]
        n += 1
        if n > rec.size:
            raise ValueError("the receiver graph loops, which a filled "
                             "surface cannot do")
    return k, km, "stream"


def route(grid):
    """Fill, D8 and accumulation: everything per cell the columns walk on."""
    filled = fill_pits(grid["topo"])
    rec, step_km, area = d8_receivers(filled, grid["lat"])
    return {"filled": filled, "rec": rec, "step_km": step_km,
            "acc": accumulate(filled, rec, area)}


def hand_of(grid, routing, threshold_km2, lat, lon):
    """HAND, path length and stream area for one point, from the raw TOPO."""
    i, j = cell_of(grid["lat"], grid["lon"], lat, lon)
    k = i * grid["lon"].size + j
    end, km, how = walk_to_stream(k, routing["rec"], routing["step_km"],
                                  routing["acc"] >= threshold_km2)
    topo = grid["topo"].ravel()
    return {"cell_topo_m": float(topo[k]),
            "hand_m": float(topo[k] - topo[end]),
            "hand_path_km": float(km),
            "hand_stream_area_km2": float(routing["acc"][end]),
            "hand_stream_topo_m": float(topo[end]),
            "hand_path_end": how}


# ── the gauge check ───────────────────────────────────────────────────────

def gauge_check(grid, acc, gauges, thresholds=STREAM_THRESHOLDS_KM2,
                tol_m=GAUGE_TOPO_TOL_M, pick="smallest"):
    """Fix the stream threshold A by the run's in-basin gauges.

    For each candidate A a gauge scores when its cell is a stream cell
    (accumulation >= A) and that cell's TOPO is within tol_m of the gauge's
    altitude_m. Stream sets nest as A grows, so the score can only fall
    with A; the task's rule takes the SMALLEST A among those sharing the
    best score, and pick="largest" takes the sparsest network that still
    scores as well. Every per-threshold score is returned for inspection.
    """
    if pick not in PICKS:
        raise ValueError(f"pick={pick!r} is not one of {PICKS}")
    if not gauges:
        raise ValueError("no in-basin gauge to check the stream threshold "
                         "against; A cannot be fixed for this basin")
    topo = grid["topo"].ravel()
    nx = grid["lon"].size
    rows = []
    for g in gauges:
        i, j = cell_of(grid["lat"], grid["lon"], g["lat"], g["lon"])
        k = i * nx + j
        alt = g.get("altitude_m")
        rows.append({"id": g["id"], "name": g.get("name"), "altitude_m": alt,
                     "cell_topo_m": _r(topo[k]), "cell_area_km2": float(acc[k]),
                     "drainage_area_km2": g.get("drainage_area_km2"),
                     "topo_matches": bool(alt is not None and
                                          abs(topo[k] - alt) <= tol_m)})
    scores = {A: sum(1 for r in rows
                     if r["topo_matches"] and r["cell_area_km2"] >= A)
              for A in thresholds}
    best = max(scores.values())
    tied = [A for A in thresholds if scores[A] == best]
    chosen = min(tied) if pick == "smallest" else max(tied)
    for r in rows:
        r["on_stream_at_chosen"] = r["cell_area_km2"] >= chosen
        r["cell_area_km2"] = _r(r["cell_area_km2"])
    return {"threshold_km2": chosen, "score": best, "n_gauges": len(rows),
            "pick": pick, "topo_tolerance_m": tol_m,
            "scores": {str(A): s for A, s in scores.items()}, "gauges": rows}


# ── per column ────────────────────────────────────────────────────────────

def column_std_elev(run_dir, cid):
    """STD_ELEV from the column's own warm-start surfdata, in metres."""
    import netCDF4 as nc
    f = Path(run_dir) / "warmstart" / f"surfdata_{cid}.nc"
    if not f.is_file():
        raise ValueError(f"{f} is missing: the column's own STD_ELEV comes "
                         f"from its warm-start surfdata")
    with nc.Dataset(f) as d:
        v = np.array(d.variables["STD_ELEV"][:], dtype=float).ravel()
    if v.size != 1:
        raise ValueError(f"{f}: STD_ELEV holds {v.size} cells, a column's "
                         f"surfdata holds one")
    return float(v[0])


def conus2_wtd(run_dir, columns):
    """CONUS2 water-table depth per column id: columns.json water_table_m
    where present, else the run's raster as static_wtd samples it."""
    need = [c for c in columns if c.get("water_table_m") is None]
    sampled = {}
    if need:
        got = static_wtd.sample(run_dir, [c["lat"] for c in need],
                                [c["lon"] for c in need])
        sampled = {c["id"]: v for c, v in zip(need, got)}
    return {c["id"]: (c["water_table_m"] if c.get("water_table_m") is not None
                      else sampled.get(c["id"])) for c in columns}


def sink_datum(cid, source, hand_m, std_elev_m, conus2_wtd_m):
    """The chosen datum, metres below the surface."""
    if source == "hand":
        return max(float(hand_m), STD_ELEV_FLOOR * float(std_elev_m))
    if source == "conus2":
        if conus2_wtd_m is None:
            raise ValueError(f"{cid}: no CONUS2 water table (neither "
                             f"columns.json water_table_m nor a raster "
                             f"value), so --source conus2 has nothing to set")
        return float(conus2_wtd_m)
    raise ValueError(f"source={source!r} is not one of {SOURCES}")


def compute(run_dir, source="hand", margin_deg=MARGIN_DEG, pick="smallest"):
    """The report: basin routing facts, the gauge check, one row per column."""
    if source not in SOURCES:
        raise ValueError(f"source={source!r} is not one of {SOURCES}")
    run_dir = Path(run_dir).resolve()
    reception = json.loads((run_dir / "reception.json").read_text())
    columns = json.loads((run_dir / "columns.json").read_text())["columns"]
    domain = reception["brief"]["domain"]
    bbox = domain["bbox"]
    gauges = [s for s in reception["observations"]["streamflow"]["stations"]
              if s.get("in_basin")]

    bands = range(band_index(bbox["min_lat"] - margin_deg),
                  band_index(bbox["max_lat"] + margin_deg) + 1)
    grid = read_topo(band_files(run_dir, bands), bbox, margin_deg)
    routing = route(grid)
    check = gauge_check(grid, routing["acc"], gauges, pick=pick)
    threshold = check["threshold_km2"]
    wtd = conus2_wtd(run_dir, columns)

    rows = []
    for c in columns:
        h = hand_of(grid, routing, threshold, c["lat"], c["lon"])
        std = column_std_elev(run_dir, c["id"])
        rows.append({
            "id": c["id"], "lat": c["lat"], "lon": c["lon"],
            "elevation_m": c.get("elevation_m"),
            "cell_topo_m": _r(h["cell_topo_m"]),
            "hand_m": _r(h["hand_m"]),
            "hand_path_km": _r(h["hand_path_km"]),
            "hand_stream_area_km2": _r(h["hand_stream_area_km2"]),
            "hand_stream_topo_m": _r(h["hand_stream_topo_m"]),
            "hand_path_end": h["hand_path_end"],
            "hand_threshold_km2": threshold,
            "std_elev_m": _r(std),
            "conus2_wtd_m": _r(wtd[c["id"]]),
            "sink_datum_m": _r(sink_datum(c["id"], source, h["hand_m"], std,
                                          wtd[c["id"]])),
            "sink_datum_source": source,
        })

    return {
        "run_dir": str(run_dir), "basin": domain.get("name"), "bbox": bbox,
        "margin_deg": margin_deg, "surfdata_files": grid["files"],
        "grid": {"n_rows": int(grid["lat"].size), "n_cols": int(grid["lon"].size),
                 "lat_min": _r(grid["lat"][0], 5), "lat_max": _r(grid["lat"][-1], 5),
                 "lon_min_east": _r(grid["lon"][0], 5),
                 "lon_max_east": _r(grid["lon"][-1], 5),
                 "n_filled_cells": int(np.sum(routing["filled"] > grid["topo"]))},
        "cell_km": CELL_KM, "fill_eps_m": FILL_EPS_M,
        "stream_thresholds_km2": list(STREAM_THRESHOLDS_KM2),
        "gauge_check": check, "hand_threshold_km2": threshold,
        "std_elev_floor": STD_ELEV_FLOOR,
        "sink_datum_source": source,
        "sink_datum_rule": ("max(hand_m, 0.5 * std_elev_m)" if source == "hand"
                            else "conus2_wtd_m"),
        "n_columns": len(rows),
        "n_columns_without_stream": sum(r["hand_path_end"] != "stream"
                                        for r in rows),
        "columns": rows,
    }


def outside_run(out, run_dir):
    """Refuse an --out inside the run: the run is read-only to this tool."""
    out, run_dir = Path(out).resolve(), Path(run_dir).resolve()
    if out == run_dir or run_dir in out.parents:
        raise ValueError(f"--out {out} lies inside the run {run_dir}, which "
                         f"this tool never writes into")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--source", choices=SOURCES, default="hand")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin-deg", type=float, default=MARGIN_DEG)
    ap.add_argument("--pick", choices=PICKS, default="smallest",
                    help="which A among those sharing the best gauge score")
    a = ap.parse_args()
    try:
        out = outside_run(a.out, a.run_dir)
        report = compute(a.run_dir, a.source, a.margin_deg, a.pick)
    except ValueError as e:
        sys.exit(str(e))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    g = report["gauge_check"]
    print(f"{report['basin']}: A = {g['threshold_km2']} km^2 (gauge score "
          f"{g['score']}/{g['n_gauges']}, scores {g['scores']}); "
          f"{report['n_columns']} columns, source {report['sink_datum_source']}")
    for r in report["columns"]:
        print(f"  {r['id']} hand {r['hand_m']:8.1f} m over {r['hand_path_km']:4.1f} km "
              f"({r['hand_path_end']}, {r['hand_stream_area_km2']:.0f} km^2) "
              f"std_elev {r['std_elev_m']:5.1f} conus2 {r['conus2_wtd_m']} "
              f"-> sink_datum_m {r['sink_datum_m']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
