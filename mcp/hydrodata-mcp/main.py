#!/usr/bin/env python3
"""
HydroFrame MCP Server — water-table data from HydroFrame's `hf_hydrodata`
catalogue (Princeton).

NAMED FOR THE ARCHIVE, NOT ONE MODEL IN IT (2026-08-12). It was `parflow_clm`
until it started serving datasets that are not ParFlow; a server named after one
of the products it carries will mislabel the next one too.

WHAT IS ON OFFER — two things, and they are different in kind:

    conus2_domain / ss_water_table_depth    MODELLED. 1 km, static.
        The spun-up equilibrium water table of the ParFlow CONUS2 domain — a
        state ParFlow itself considers self-consistent. Every land cell has a
        value, so a sampling design is never constrained by coverage.

    fan_2013 / water_table_depth            MEASURED. Point sites.
        The wells behind Fan, Li & Miguez-Macho (2013): one long-term mean
        depth per site, spanning 1927-2009. HydroFrame's own catalogue types
        this `point_observations`. NOT the modelled equilibrium surface from
        that paper — that was a NetCDF tile, deleted 2026-08-12.

A modelled field and a set of well records are never merged here, and nothing
downstream may score one against the other.

ONE REQUEST PER BASIN, AND THEN NEVER AGAIN. `download_conus2_wtd` fetches a
bounding box once and writes a GeoTIFF; everything afterwards reads that file
locally. Per-point tools used to live here — get_parflow_wtd, _points, and
sample_parflow_wtd — and all three were the same bounded fetch wearing different
output shapes. They were deleted 2026-08-12 along with ma_2025 and the CONUS1
daily field, because this is academic infrastructure and the traffic it carries
should be one request per basin, not one per column.

CREDENTIALS. The catalogue is open — datasets, variables, grid geometry and
lat/lon conversion all work with no account. THE DATA DOES NOT: it needs a free
email + PIN from Princeton, registered once per machine. So this server, like
`ameriflux`, answers discovery questions always and data questions only when the
host is set up, and `data_status()` says which it is.

    data_status()                      registered? reachable? how to fix
    download_conus2_wtd(bbox, out_dir) the modelled field -> one GeoTIFF
    get_fan2013_wells(bbox)            the well observations

There is no describe_capabilities() — see the note at the foot of this file.
What it would have said is here, and reading this costs no request.

Sign convention: depth is POSITIVE DOWNWARD from the land surface, matching the
framework's `water_table_depth_m`.
"""
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ── the modelled field ──────────────────────────────────────────────────────
_SS_DATASET = "conus2_domain"
_SS_VARIABLE = "ss_water_table_depth"
_SS_GRID = "conus2"

# ── the well observations ───────────────────────────────────────────────────
# Fan, Li & Miguez-Macho (2013) as HydroFrame serves it: point sites, one
# long-term mean depth each, spanning 1927-2009. Restored 2026-08-12 at the
# user's request — a mean over decades is a different quantity from a single
# USGS visit, and worth having beside it.
#
# READ `n_records` BEFORE TRUSTING THE WORD "long-term". Measured 2026-08-12 on
# the Naches HUC8 bounding box: 3,919 sites, and 3,620 of them (92%) have
# exactly ONE measurement behind their "mean"; only 26 have 12 or more.
# Brandywine is the same story — 3,780 of 4,201. The long records are real and
# are the reason to fetch this; the rest are single visits wearing an
# eighty-year date range, and the count is what tells them apart.
_FAN_DATASET = "fan_2013"
_FAN_VARIABLE = "water_table_depth"

# NAMES THE ARCHIVE, NOT ONE MODEL IN IT (fixed 2026-08-12). This string is
# stamped on every result, and it used to read "ParFlow-CLM CONUS domains" —
# which labelled Fan's WELL MEASUREMENTS as ParFlow model output. Each tool
# reports its own `dataset` and `kind`; this says only where it came from.
_SOURCE = "HydroFrame hf_hydrodata (Princeton)"
_SIGNUP_URL = "https://hydrogen.princeton.edu/signup"
_PIN_URL = "https://hydrogen.princeton.edu/pin"

# THE FILENAME IS A CONTRACT. src/core/static_wtd.py opens this exact name and
# the two cannot import each other — the framework and the MCP are separate
# processes. Change it in both or in neither.
RASTER_NAME = "wtd_conus2.tif"

# A guard against "fetch the domain", not a budget. CONUS2 is 4442 x 3256 =
# 14.5 million cells; a HUC8 is a few thousand. One million allows a bbox far
# larger than any basin this framework runs and still refuses the whole domain.
_MAX_CELLS = 1_000_000

# Points per edge when turning a lat/lon bbox into grid indices. Four would
# capture the rotation on its own; more also covers the curvature of an edge in
# a conic projection, and it is local arithmetic either way.
_EDGE_SAMPLES = 9

mcp = FastMCP("hydrodata")


# ─────────────────────────────────────────────────────────────────────────────
# CREDENTIALS AND REACHABILITY
# ─────────────────────────────────────────────────────────────────────────────
def _pin():
    """(email, pin) if this machine has registered, else None.

    hf_hydrodata raises rather than returns when nothing is registered, which
    is right for a script and wrong for a status tool — so it is caught here
    and turned into an answer.
    """
    try:
        import hf_hydrodata as hf
        return hf.get_registered_api_pin()
    except Exception:                                           # noqa: BLE001
        return None


def _hf():
    """The client, or None if it is not installed."""
    try:
        import hf_hydrodata as hf
        return hf
    except ImportError:
        return None


def _not_ready():
    """The one refusal every data tool gives, so they cannot drift apart.

    ok=false WITH a reason, never an empty result: "we could not look" and
    "there is nothing there" are different findings, and conflating them is
    what turned a rate-limited USGS fetch into a planner reporting a basin
    with no stream gauges.
    """
    if _hf() is None:
        return {"ok": False, "error": "hf_hydrodata is not installed on this "
                                      "host — `pip install hf_hydrodata`",
                "source": _SOURCE}
    if _pin() is None:
        return {"ok": False,
                "error": ("no HydroFrame PIN registered on this machine. The "
                          "CATALOGUE is open but the gridded data is not. "
                          f"Sign up at {_SIGNUP_URL}, create a PIN at "
                          f"{_PIN_URL}, then run "
                          "hf_hydrodata.register_api_pin('<email>', '<pin>') "
                          "once — it is stored in ~/.hydrodata and persists."),
                "signup": _SIGNUP_URL, "pin": _PIN_URL,
                "next": "data_status", "source": _SOURCE}
    return None


# ─────────────────────────────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────────────────────────────
def _xy(grid, lat, lon):
    """(x, y) grid indices for a lat/lon, or None if outside the domain.

    ORDER MATTERS AND IT IS (x, y). hf_hydrodata.from_latlon returns x first,
    verified by round-tripping through to_latlon: from_latlon("conus2", 38.9,
    -107.0) -> [1368.997, 1602.683], and to_latlon("conus2", 1368.997,
    1602.683) -> [38.900, -107.000]. Reading it as (y, x) — the natural guess
    for a raster — resolves to a real cell roughly 300 km away and returns a
    perfectly plausible number, which is the worst kind of wrong.

    FLOOR, NOT ROUND (fixed 2026-08-12). `from_latlon` returns a FRACTIONAL
    index, and `grid_bounds` is a half-open slice — [200,200,300,250] yields
    exactly 100 x 50 cells — so cell i covers the index range [i, i+1) and the
    cell containing a point is the floor of its index, never the nearest one.
    Rounding read the NEIGHBOURING cell whenever the fraction reached 0.5,
    which is about half of all points: on the 1 km ParFlow grid that is a whole
    kilometre, and at Naches col_05 it returned 0.05 m where the containing
    cell holds 177.58 m. The raster written below anchors on the same floor
    rule, so a raster read and a grid index agree.
    """
    hf = _hf()
    try:
        x, y = hf.from_latlon(grid, float(lat), float(lon))
        return int(x // 1), int(y // 1)
    except Exception:                                           # noqa: BLE001
        return None


def _fetch(dataset, variable, grid, bounds, **extra):
    """One gridded read. bounds is [x_min, y_min, x_max, y_max]."""
    hf = _hf()
    opts = {"dataset": dataset, "variable": variable, "grid": grid,
            "grid_bounds": list(bounds), **extra}
    return hf.get_gridded_data(opts)


def _clean(v):
    """A single value as a float, or None when the grid says no-data."""
    import math
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or f <= -9990:
        return None
    return round(f, 3)


def _covers(path, lo_lon, lo_lat, hi_lon, hi_lat):
    """Does an existing raster answer questions about this bbox? Local only.

    THE FILE'S NAME IS NOT ITS CONTENTS (added 2026-08-12). Reuse used to test
    only that wtd_conus2.tif existed. Two run_pipeline runs into one output
    directory — Naches, then Brandywine — therefore handed Brandywine the
    Washington State raster and reported ok=true with a mean depth of 104.8 m
    over a piedmont basin whose wells sit at 8 m. Every later read came back
    "outside the raster", which is a loud failure, but reception.json had
    already recorded a summary belonging to a different watershed.

    Asked of the FILE ITSELF — its own transform and projection — so this needs
    no catalogue, no PIN and no network. The whole perimeter is tested, for the
    same reason the window is built from the whole perimeter: a lat/lon
    rectangle is a rotated quadrilateral on this grid.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform

    n = _EDGE_SAMPLES
    ring = []
    for k in range(n):
        t = k / (n - 1)
        la = lo_lat + (hi_lat - lo_lat) * t
        lo = lo_lon + (hi_lon - lo_lon) * t
        ring += [(lo_lat, lo), (hi_lat, lo), (la, lo_lon), (la, hi_lon)]
    with rasterio.open(path) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs,
                                [p[1] for p in ring], [p[0] for p in ring])
        inv = ~src.transform
        for x, y in zip(xs, ys):
            c, r = inv * (x, y)
            if not (0 <= int(r // 1) < src.height and 0 <= int(c // 1) < src.width):
                return False
    return True


def _write_tif(path, block, crs, res, x_west, y_north, tags):
    """A north-up, georeferenced GeoTIFF from a HydroFrame block.

    SEPARATE FROM THE FETCH ON PURPOSE. The georeferencing is the part that can
    be silently wrong — a mirrored field still produces plausible depths — so it
    is a plain function that can be tested with a synthetic array and no
    network. `x_west` / `y_north` are the projected coordinates of the block's
    top-LEFT corner.

    HydroFrame counts y NORTHWARD from a lower-left origin; a GeoTIFF is written
    top row first. The flip here, and anchoring the transform at the top-left
    corner rather than the bottom-left, is what makes the file agree with every
    reader that assumes north-up.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    # No-data becomes NaN before it is written, so a reader never has to know
    # HydroFrame's -9999 convention to avoid averaging it into a mean.
    a = np.asarray(block, dtype="float32")
    a[a <= -9990] = np.nan

    with rasterio.open(path, "w", driver="GTiff",
                       height=a.shape[0], width=a.shape[1], count=1,
                       dtype="float32", crs=crs,
                       transform=from_origin(x_west, y_north, res, res),
                       nodata=float("nan"), compress="lzw", predictor=2,
                       tiled=False) as dst:
        dst.write(np.flipud(a), 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})

    good = a[np.isfinite(a)]
    return ({"min": round(float(good.min()), 3),
             "max": round(float(good.max()), 3),
             "mean": round(float(good.mean()), 3),
             "n_with_value": int(good.size),
             "n_cells": int(a.size)} if good.size else
            {"min": None, "max": None, "mean": None,
             "n_with_value": 0, "n_cells": int(a.size)})


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def data_status() -> str:
    """Can this host read HydroFrame data right now, and if not, what is missing?

    Separate from describe_capabilities because the answer CHANGES per machine:
    capabilities are what the server can do in principle, this is whether it can
    do it here. A capability list that says "returns water table depth" on a
    host with no PIN is a promise the run will break.
    """
    hf = _hf()
    out = {"source": _SOURCE,
           "client_installed": hf is not None,
           "pin_registered": _pin() is not None,
           "catalogue_open": None,
           "modelled_field": f"{_SS_DATASET}/{_SS_VARIABLE} ({_SS_GRID}, static)",
           "well_observations": f"{_FAN_DATASET}/{_FAN_VARIABLE} (points)"}
    if hf is not None:
        try:                        # catalogue needs no account — prove it
            out["catalogue_open"] = bool(hf.get_datasets())
        except Exception as e:                                  # noqa: BLE001
            out["catalogue_open"] = False
            out["catalogue_error"] = f"{type(e).__name__}: {e}"[:200]
    ready = bool(hf is not None and out["pin_registered"] and out["catalogue_open"])
    out["data_ready"] = ready
    out["how_to_enable"] = None if ready else [
        "1. pip install hf_hydrodata" if hf is None else None,
        f"2. Sign up (free): {_SIGNUP_URL}",
        f"3. Create a PIN: {_PIN_URL}",
        "4. Register it ONCE on this machine, in python:\n"
        "     import hf_hydrodata as hf\n"
        "     hf.register_api_pin('<your email>', '<your pin>')\n"
        "   It is written to ~/.hydrodata and persists across sessions.",
        "5. Re-run data_status() — data_ready should be true.",
    ]
    if out["how_to_enable"]:
        out["how_to_enable"] = [s for s in out["how_to_enable"] if s]
    return json.dumps(out, indent=2)


@mcp.tool()
def download_conus2_wtd(bbox: str, out_dir: str, pad_cells: int = 2,
                        overwrite: bool = False) -> str:
    """Fetch the ParFlow CONUS2 steady-state water table over a bbox, as a GeoTIFF.

    ONE REQUEST, THEN A LOCAL FILE. This is the only call that reaches Princeton
    for the modelled field, and it is made once per basin by reception. Every
    later lookup — the sampler, the analyzer — reads the file through
    src/core/static_wtd.py, with no network, no PIN, and no MCP session. That is
    what lets the framework's rule hold that reception is the only component
    reaching outside.

    A GEOTIFF, NOT AN ARRAY DUMP, because the projection travels inside the
    file. HydroFrame publishes the grid's CRS, origin and cell size, so the
    result is georeferenced properly and any reader — rasterio, QGIS, GDAL —
    can sample it at a latitude and longitude with no knowledge of HydroFrame's
    index space. No-data is written as NaN rather than -9999.

    STEADY STATE, NOT A DATE. This is the spun-up equilibrium field. There is no
    period argument because there is no period: for what the water table did on
    particular days you would need a transient dataset, which this server no
    longer carries.

    pad_cells: extra cells around the bbox. The default of 2 exists because
        sample_columns SNAPS columns onto the CONUS grid, which moves them — a
        column near the basin edge can land just outside the box reception
        asked for. At 1 km, 2 cells costs a few kilobytes and makes that
        impossible.

    Returns the path, shape, projection and a summary. An existing file is
    reused unless `overwrite`, since the field is static.
    """
    bad = _not_ready()
    if bad:
        return json.dumps(bad)
    try:
        lo_lon, lo_lat, hi_lon, hi_lat = [float(v) for v in bbox.split(",")]
    except (ValueError, AttributeError):
        return json.dumps({"ok": False, "error": "bbox must be "
                                                 "'min_lon,min_lat,max_lon,max_lat'"})
    try:
        import numpy as np
        import rasterio                                          # noqa: F401
        from hf_hydrodata.data_model_access import load_data_model
    except ImportError as e:
        return json.dumps({"ok": False,
                           "error": f"needs rasterio and hf_hydrodata: {e}"})

    # ── THE EXISTING-FILE CHECK COMES FIRST, BEFORE ANY LOOKUP ──────────────
    # It used to sit after the grid metadata was read, which meant a run that
    # needed no data at all still called the catalogue to find out. Nothing
    # below this branch touches the network.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / RASTER_NAME
    # AN EXISTING FILE IS ONLY REUSABLE IF IT COVERS THIS BBOX. It is checked
    # against the file's own geometry, so this costs nothing; a file for some
    # other basin is refetched over rather than silently served.
    stale = False
    if path.is_file() and not overwrite:
        try:
            stale = not _covers(path, lo_lon, lo_lat, hi_lon, hi_lat)
        except Exception:                                       # noqa: BLE001
            stale = True            # unreadable is as good as wrong
    if path.is_file() and not overwrite and not stale:
        # DESCRIBED FROM THE FILE, NOT FROM NOTHING. Returning a bare path here
        # left reception.json with `summary: null` and `shape: null` on any
        # re-run into the same directory — the same record saying two different
        # things depending on whether the file happened to exist. Reading it
        # back is local and costs nothing.
        try:
            with rasterio.open(path) as src:
                a = src.read(1)
                good = a[np.isfinite(a)]
                t = src.tags()
                gb = [int(v) for v in (t.get("grid_bounds") or "").split(",")
                      if v.strip().lstrip("-").isdigit()]
                reread = {
                    "shape": [src.height, src.width],
                    "resolution_m": abs(src.transform.a),
                    "crs": str(src.crs),
                    # from the file's own tags, so a reused raster describes
                    # itself as fully as a freshly written one
                    "grid_bounds": gb or None,
                    "pad_cells": (int(t["pad_cells"])
                                  if str(t.get("pad_cells", "")).isdigit()
                                  else None),
                    "bbox_written_for": t.get("bbox"),
                    "summary": ({"min": round(float(good.min()), 3),
                                 "max": round(float(good.max()), 3),
                                 "mean": round(float(good.mean()), 3),
                                 "n_with_value": int(good.size),
                                 "n_cells": int(a.size)} if good.size else None)}
        except Exception as e:                                  # noqa: BLE001
            reread = {"summary": None,
                      "read_back_error": f"{type(e).__name__}: {e}"[:200]}
        return json.dumps({
            "ok": True, "path": str(path), "reused": True,
            "mb": round(path.stat().st_size / 1e6, 3),
            "covers_bbox": True,
            "note": "the field is static, so an existing file covering this "
                    "bbox is not refetched; pass overwrite=true to force it",
            "dataset": _SS_DATASET, "variable": _SS_VARIABLE,
            "grid": _SS_GRID, "source": _SOURCE, **reread})

    row = load_data_model().get_table("grid").get_row(_SS_GRID)
    crs = row.get_value("crs")
    res = float(row.get_value("resolution_meters"))
    ox, oy = [float(v) for v in row.get_value("origin")]
    # shape is [z, y, x] — the domain's own extent, used to clamp the pad so a
    # basin on the edge of CONUS cannot ask for negative indices.
    _, ny_dom, nx_dom = [int(v) for v in row.get_value("shape")]

    # ── THE WHOLE PERIMETER, NOT TWO CORNERS ────────────────────────────────
    # A LAT/LON RECTANGLE IS A ROTATED QUADRILATERAL ON THIS GRID. CONUS2 is
    # Lambert Conformal Conic centred on 97 W, so away from that meridian the
    # projection turns: at Naches (121 W) the convergence is about 17 degrees.
    # Taking the window from the south-west and north-east corners alone —
    # which this did until 2026-08-12 — leaves the other two corners OUTSIDE
    # the box. Measured on the real grid: Naches lost 22 rows to the south and
    # 21 to the north of a 50-row window, Brandywine 17 columns on each side,
    # a five-degree western box 116 and 109 rows. Columns in those corners
    # would have read back as "outside the raster" and returned no water table
    # at all — a plausible-looking absence with a projection behind it.
    #
    # Walking the perimeter catches the rotation exactly (the corners do that)
    # and the slight curvature of the edges as well. from_latlon is local
    # arithmetic, so this costs nothing and reaches nobody's server.
    n = _EDGE_SAMPLES
    ring = []
    for k in range(n):
        t = k / (n - 1)
        la = lo_lat + (hi_lat - lo_lat) * t
        lo = lo_lon + (hi_lon - lo_lon) * t
        ring += [(lo_lat, lo), (hi_lat, lo),      # south and north edges
                 (la, lo_lon), (la, hi_lon)]      # west and east edges
    idx = [p for p in (_xy(_SS_GRID, la, lo) for la, lo in ring) if p]
    if not idx:
        return json.dumps({"ok": False, "bbox": bbox,
                           "error": "bbox is outside the CONUS2 domain",
                           "source": _SOURCE})
    xs = [p[0] for p in idx]
    ys = [p[1] for p in idx]
    pad = max(0, int(pad_cells))
    # +1 on the upper edge because grid_bounds is a HALF-OPEN slice: the cell
    # containing the last point is included only if the box reaches past it.
    xa = max(0, min(xs) - pad)
    xb = min(nx_dom, max(xs) + 1 + pad)
    ya = max(0, min(ys) - pad)
    yb = min(ny_dom, max(ys) + 1 + pad)
    n_edge_outside = len(ring) - len(idx)
    n_cells = (xb - xa) * (yb - ya)
    if n_cells > _MAX_CELLS:
        return json.dumps({
            "ok": False, "bbox": bbox,
            "error": f"that bbox is {xb - xa}x{yb - ya} = {n_cells} cells, over "
                     f"the {_MAX_CELLS}-cell cap. Ask for a smaller area.",
            "source": _SOURCE})

    try:
        # RESHAPED, NEVER SQUEEZED. np.squeeze on a 1-cell-wide block drops the
        # axis and the write then fails or, worse, transposes; the fetch returns
        # exactly (yb-ya) x (xb-xa) cells, so say so.
        block = np.asarray(
            _fetch(_SS_DATASET, _SS_VARIABLE, _SS_GRID, [xa, ya, xb, yb],
                   period="static")).reshape(yb - ya, xb - xa)
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False, "bbox": bbox,
                           "error": f"{type(e).__name__}: {e}"[:300],
                           "source": _SOURCE})

    try:
        summary = _write_tif(
            path, block, crs, res,
            x_west=ox + xa * res, y_north=oy + yb * res,
            tags={"dataset": _SS_DATASET, "variable": _SS_VARIABLE,
                  "grid": _SS_GRID, "units": "m",
                  "positive": "down from land surface",
                  "kind": "steady state (spun-up equilibrium)",
                  # WHICH BBOX THIS IS, written into the file. The geometry
                  # already answers "does it cover X"; this answers "what was
                  # it asked for", which is what a human reads when a run
                  # directory has been reused for a second basin.
                  "bbox": bbox, "grid_bounds": f"{xa},{ya},{xb},{yb}",
                  "pad_cells": pad,
                  "source": _SOURCE})
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False, "bbox": bbox,
                           "error": f"writing {path}: {type(e).__name__}: {e}"[:300]})

    return json.dumps({
        "ok": True, "path": str(path), "reused": False,
        "shape": [yb - ya, xb - xa], "pad_cells": pad,
        "grid_bounds": [xa, ya, xb, yb],
        # >0 means part of the bbox perimeter falls outside CONUS2, so the
        # raster covers less than was asked for — a coastal or border basin.
        "n_perimeter_points_outside_domain": n_edge_outside,
        "resolution_m": res, "crs": crs,
        "mb": round(path.stat().st_size / 1e6, 3),
        "bbox": bbox, "summary": summary,
        # says a file for a DIFFERENT bbox was found here and written over —
        # normally a run directory reused for a second basin
        "replaced_raster_for_another_bbox": bool(stale),
        "dataset": _SS_DATASET, "variable": _SS_VARIABLE, "grid": _SS_GRID,
        "kind": "modelled steady state, clipped to the bbox and georeferenced",
        "read_with": "src/core/static_wtd.py sample(dir, lats, lons)",
        "source": _SOURCE})


@mcp.tool()
def get_fan2013_wells(bbox: str, min_records: int = 1) -> str:
    """The WELLS behind Fan et al. (2013): one long-term mean depth per site.

    OBSERVATIONS, not a field. HydroFrame serves fan_2013 as the point
    compilation the paper was fitted to — its catalogue types it
    `point_observations` — so this returns measurement sites and cannot be
    sampled at an arbitrary column. For a value anywhere, use the modelled
    field: download_conus2_wtd.

    WHAT MAKES IT WORTH HAVING: each value is a mean over 1927-2009, which is a
    different quantity from a USGS field visit on one day. What the run's own
    period cannot show — where this water table typically sits — this can.

    WHAT TO WATCH: `n_records` says how many measurements went into each mean,
    and it is 1 for 92% of Naches sites. A mean of one measurement is a single
    visit with a long date range attached, not a climatology. `min_records`
    filters on it; the counts come back either way so nothing is hidden.

    bbox is 'min_lon,min_lat,max_lon,max_lat'.
    """
    bad = _not_ready()
    if bad:
        return json.dumps(bad)
    try:
        lo_lon, lo_lat, hi_lon, hi_lat = [float(v) for v in bbox.split(",")]
    except (ValueError, AttributeError):
        return json.dumps({"ok": False,
                           "error": "bbox must be "
                                    "'min_lon,min_lat,max_lon,max_lat'"})
    hf = _hf()
    q = {"dataset": _FAN_DATASET, "variable": _FAN_VARIABLE,
         "temporal_resolution": "long_term", "aggregation": "mean",
         "latitude_range": (lo_lat, hi_lat),
         "longitude_range": (lo_lon, hi_lon)}
    try:
        md = hf.get_point_metadata(q)
        df = hf.get_point_data(q)
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False, "bbox": bbox,
                           "error": f"{type(e).__name__}: {e}"[:300],
                           "source": _SOURCE})
    merged = md.merge(df, on="site_id", suffixes=("", "_v"))
    wells, n_dropped = [], 0
    for _, r in merged.iterrows():
        v = _clean(r.get("wtd"))
        if v is None:
            continue
        n_rec = _clean(r.get("record_count"))
        n_rec = int(n_rec) if n_rec is not None else None
        if n_rec is not None and n_rec < int(min_records):
            n_dropped += 1
            continue
        wells.append({"id": str(r.get("site_id")),
                      "name": (str(r.get("site_name"))
                               if r.get("site_name") is not None else None),
                      "lat": round(float(r["latitude"]), 5),
                      "lon": round(float(r["longitude"]), 5),
                      "wtd_m": round(v, 3),
                      "n_records": n_rec,
                      "period": [str(r.get("first_date_data_available"))[:10],
                                 str(r.get("last_date_data_available"))[:10]]})
    wells.sort(key=lambda w: (-(w["n_records"] or 0), w["id"]))
    depths = [w["wtd_m"] for w in wells]
    counts = [w["n_records"] for w in wells if w["n_records"] is not None]
    out = {
        "ok": True, "bbox": bbox, "n_wells": len(wells), "wells": wells,
        "min_records": int(min_records),
        "observation_kind": "long-term mean, one value per site",
        "period": "1927-2009",
        "summary": ({"min": min(depths), "max": max(depths),
                     "median": round(sorted(depths)[len(depths) // 2], 3)}
                    if depths else None),
        "dataset": _FAN_DATASET, "source": _SOURCE}
    if counts:
        out["records_per_site"] = {
            "n_single_measurement": sum(1 for c in counts if c == 1),
            "n_12_or_more": sum(1 for c in counts if c >= 12),
            "max": max(counts),
            "note": ("a site with n_records = 1 has a 'long-term mean' built "
                     "from one visit — the date range is long, the record is "
                     "not")}
    if n_dropped:
        out["n_below_min_records"] = n_dropped
    return json.dumps(out)


# NO describe_capabilities() HERE, DELIBERATELY (2026-08-12). Every other server
# in this framework has one; this one had one too, and it called data_status(),
# which calls hf.get_datasets() to prove the catalogue is reachable. So merely
# ASKING WHAT THE SERVER DOES sent a request to Princeton. On a server whose
# whole design is one request per basin, a free-standing description that costs
# a round trip is the wrong trade. What it said is in the module docstring above,
# where reading it costs nothing, and data_status() is still there for the
# question that genuinely needs the network: whether this host can fetch at all.


if __name__ == "__main__":
    mcp.run(transport="stdio")
