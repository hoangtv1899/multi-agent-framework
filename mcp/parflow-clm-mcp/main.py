#!/usr/bin/env python3
"""
ParFlow-CLM MCP Server — water-table depth from the CONUS ParFlow-CLM domains,
served through HydroFrame's `hf_hydrodata` client (Princeton).

WHY THIS EXISTS, ALONGSIDE fan_wtd. Both answer "how deep is the water table
here", and they are not the same claim:

    fan_wtd      Fan, Li & Miguez-Macho (2013): an equilibrium water table,
                 observationally constrained, provisioned here as static
                 NetCDF TILES. Coverage is whatever tiles were copied in.
    parflow_clm  a SIMULATED water table from an integrated hydrologic model
                 (ParFlow-CLM) on a regular 1 km CONUS grid. Every land cell
                 in the domain has a value, so a sampling design can put a
                 column anywhere without asking whether the field covers it.

That last difference is the point. A tiled prior constrains WHERE you may
sample; a full-domain field does not. It also means the two disagree in an
informative way — Fan is an equilibrium surface fitted to observations,
ParFlow-CLM is a physics simulation with its own biases, and a column where
they disagree strongly is a column worth looking at twice.

WHAT IS ON OFFER, and the distinction that matters most:

    conus2_domain / ss_water_table_depth    STEADY STATE, 1 km, static.
        The spun-up equilibrium water table of the CONUS2 domain. This is the
        direct analogue of Fan 2013 and the right field for INITIALISING a
        column — it is a state the model considers self-consistent.

    conus1_baseline_mod / water_table_depth  DAILY, 1 km, transient.
        A time series from the CONUS1 baseline simulation. This is the right
        field for asking what the water table DID in a particular period, and
        the wrong one for a warm start unless you want that specific day.

Reaching for the daily field when you meant the steady state gives you one
day's weather written into an initial condition. The tools keep them apart by
name rather than by a `period` argument, so the choice is visible at the call.

CREDENTIALS. The catalogue is open — datasets, variables, grid geometry and
lat/lon conversion all work with no account. THE GRIDDED DATA DOES NOT: it
needs a free email + PIN from Princeton, registered once per machine. So this
server, like `ameriflux`, answers discovery questions always and data questions
only when the host is set up, and `data_status()` says which it is.

    data_status()                          registered? reachable? how to fix
    get_parflow_wtd(lat, lon)              steady-state depth at one point
    get_parflow_wtd_points(lats, lons)     the same for many points, one call
    sample_parflow_wtd(bbox, n)            a grid across a bbox + summary
    get_parflow_wtd_daily(lat, lon, ...)   the TRANSIENT field, named apart
    describe_parflow_clm_capabilities()    what this is and is not

Sign convention: every tool reports `depth_to_water_m`, POSITIVE DOWNWARD from
the land surface, matching fan_wtd and the framework's `water_table_depth_m`.
"""
import json
import os

from mcp.server.fastmcp import FastMCP

# ── datasets, named rather than parameterised (see the header) ──────────────
_SS_DATASET = "conus2_domain"
_SS_VARIABLE = "ss_water_table_depth"
_SS_GRID = "conus2"

_DAILY_DATASET = "conus1_baseline_mod"
_DAILY_VARIABLE = "water_table_depth"
_DAILY_GRID = "conus1"

_SOURCE = ("ParFlow-CLM CONUS domains via HydroFrame hf_hydrodata "
           "(Princeton / HydroFrame)")
_SIGNUP_URL = "https://hydrogen.princeton.edu/signup"
_PIN_URL = "https://hydrogen.princeton.edu/pin"

# Cap a bbox sample so a whole-CONUS request cannot be issued by accident: the
# conus2 grid is 4442 x 3256, and "sample the domain" is 14 million cells.
_MAX_CELLS = 40_000

mcp = FastMCP("parflow_clm")


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
    """
    hf = _hf()
    try:
        x, y = hf.from_latlon(grid, float(lat), float(lon))
        return int(round(x)), int(round(y))
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


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def data_status() -> str:
    """Can this host read ParFlow-CLM data right now, and if not, what is missing?

    Separate from describe_parflow_clm_capabilities because the answer CHANGES
    per machine: capabilities are what the server can do in principle, this is
    whether it can do it here. A capability list that says "returns water table
    depth" on a host with no PIN is a promise the run will break.
    """
    hf = _hf()
    out = {"source": _SOURCE,
           "client_installed": hf is not None,
           "pin_registered": _pin() is not None,
           "catalogue_open": None,
           "steady_state": f"{_SS_DATASET}/{_SS_VARIABLE} ({_SS_GRID}, static)",
           "daily": f"{_DAILY_DATASET}/{_DAILY_VARIABLE} ({_DAILY_GRID}, daily)"}
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
def get_parflow_wtd(lat: float, lon: float) -> str:
    """STEADY-STATE depth to water (m below surface) at the nearest 1 km cell.

    The CONUS2 spun-up equilibrium water table — the field to use for
    initialising a column, and the direct analogue of get_fan_wtd. For what the
    water table did on a particular date, use get_parflow_wtd_daily, which is a
    different question and a different dataset.
    """
    bad = _not_ready()
    if bad:
        return json.dumps(bad)
    xy = _xy(_SS_GRID, lat, lon)
    if xy is None:
        return json.dumps({"ok": False, "lat": lat, "lon": lon,
                           "error": "outside the CONUS2 domain",
                           "source": _SOURCE})
    x, y = xy
    try:
        import numpy as np
        arr = _fetch(_SS_DATASET, _SS_VARIABLE, _SS_GRID, [x, y, x + 1, y + 1],
                     period="static")
        val = _clean(np.ravel(np.asarray(arr))[0])
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False, "lat": lat, "lon": lon,
                           "error": f"{type(e).__name__}: {e}"[:300],
                           "source": _SOURCE})
    return json.dumps({
        "ok": True, "lat": lat, "lon": lon,
        "depth_to_water_m": val,
        "note": None if val is not None else "no-data at this cell",
        "grid_x": x, "grid_y": y, "grid": _SS_GRID,
        "dataset": _SS_DATASET, "variable": _SS_VARIABLE,
        "kind": "steady state (spun-up equilibrium)",
        "source": _SOURCE})


@mcp.tool()
def get_parflow_wtd_points(lats: list[float], lons: list[float]) -> str:
    """Steady-state depth to water at MANY points, in one call.

    Same answer as calling get_parflow_wtd per point, but one MCP session and
    one bounding read instead of N of each — the round trip, not the lookup, is
    the cost. Use this when enriching a set of sampling columns.

    lats/lons are parallel lists; results come back in input order. A point
    outside the domain gets depth_to_water_m=None with a note rather than
    dropping out of the list, so the caller's columns stay aligned.
    """
    bad = _not_ready()
    if bad:
        return json.dumps(bad)
    if len(lats) != len(lons):
        return json.dumps({"ok": False,
                           "error": f"lats has {len(lats)} entries, lons has "
                                    f"{len(lons)} — they must be parallel"})
    if not lats:
        return json.dumps({"ok": True, "n_points": 0, "points": [],
                           "source": _SOURCE})

    xys = [_xy(_SS_GRID, la, lo) for la, lo in zip(lats, lons)]
    inside = [p for p in xys if p]
    if not inside:
        return json.dumps({"ok": False,
                           "error": "every point is outside the CONUS2 domain",
                           "source": _SOURCE})
    # ONE read over the enclosing box, then index into it. N scattered reads of
    # a single cell each is N HTTP round trips to Princeton; a sampling design
    # of 19 columns is 19 of them, and they are the whole runtime.
    xs = [p[0] for p in inside]
    ys = [p[1] for p in inside]
    x0, x1, y0, y1 = min(xs), max(xs) + 1, min(ys), max(ys) + 1
    if (x1 - x0) * (y1 - y0) > _MAX_CELLS:
        return json.dumps({
            "ok": False,
            "error": f"those points span {(x1-x0)}x{(y1-y0)} cells, over the "
                     f"{_MAX_CELLS}-cell cap. Split them into nearer groups.",
            "source": _SOURCE})
    try:
        import numpy as np
        block = np.squeeze(np.asarray(
            _fetch(_SS_DATASET, _SS_VARIABLE, _SS_GRID, [x0, y0, x1, y1],
                   period="static")))
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:300],
                           "source": _SOURCE})

    pts = []
    for (la, lo), xy in zip(zip(lats, lons), xys):
        if xy is None:
            pts.append({"lat": la, "lon": lo, "depth_to_water_m": None,
                        "note": "outside the CONUS2 domain"})
            continue
        x, y = xy
        try:
            val = _clean(block[y - y0, x - x0])
        except Exception:                                       # noqa: BLE001
            val = None
        pts.append({"lat": la, "lon": lo, "depth_to_water_m": val,
                    "grid_x": x, "grid_y": y,
                    **({"note": "no-data at this cell"} if val is None else {})})
    got = [p["depth_to_water_m"] for p in pts if p["depth_to_water_m"] is not None]
    return json.dumps({
        "ok": True, "n_points": len(pts), "n_with_data": len(got),
        "points": pts,
        "dataset": _SS_DATASET, "variable": _SS_VARIABLE,
        "kind": "steady state (spun-up equilibrium)",
        "source": _SOURCE})


@mcp.tool()
def sample_parflow_wtd(min_lon: float, min_lat: float,
                       max_lon: float, max_lat: float, n: int = 80) -> str:
    """Sample steady-state depth to water across a bbox; points plus a summary.

    n is a TARGET, not a promise: the request is snapped to whole 1 km cells,
    so a small bbox returns fewer points than asked and that is the resolution
    speaking, not a failure.
    """
    bad = _not_ready()
    if bad:
        return json.dumps(bad)
    c0 = _xy(_SS_GRID, min_lat, min_lon)
    c1 = _xy(_SS_GRID, max_lat, max_lon)
    if c0 is None or c1 is None:
        return json.dumps({"ok": False,
                           "error": "bbox corner(s) outside the CONUS2 domain",
                           "source": _SOURCE})
    x0, x1 = sorted((c0[0], c1[0]))
    y0, y1 = sorted((c0[1], c1[1]))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    if (x1 - x0) * (y1 - y0) > _MAX_CELLS:
        return json.dumps({
            "ok": False,
            "error": f"that bbox is {(x1-x0)}x{(y1-y0)} = "
                     f"{(x1-x0)*(y1-y0)} cells, over the {_MAX_CELLS} cap. "
                     f"Ask for a smaller area.",
            "source": _SOURCE})
    try:
        import numpy as np
        block = np.squeeze(np.asarray(
            _fetch(_SS_DATASET, _SS_VARIABLE, _SS_GRID, [x0, y0, x1, y1],
                   period="static")))
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:300],
                           "source": _SOURCE})

    import numpy as np
    ny, nx = block.shape[-2], block.shape[-1]
    step = max(1, int(((nx * ny) / max(1, n)) ** 0.5))
    pts = []
    for j in range(0, ny, step):
        for i in range(0, nx, step):
            v = _clean(block[j, i])
            if v is None:
                continue
            hf = _hf()
            try:
                la, lo = hf.to_latlon(_SS_GRID, float(x0 + i), float(y0 + j))
            except Exception:                                   # noqa: BLE001
                continue
            pts.append({"lat": round(la, 5), "lon": round(lo, 5),
                        "depth_to_water_m": v})
    vals = [p["depth_to_water_m"] for p in pts]
    summ = ({"min": min(vals), "max": max(vals),
             "mean": round(sum(vals) / len(vals), 3), "n": len(vals)}
            if vals else None)
    return json.dumps({
        "ok": True, "n_points": len(pts), "points": pts, "summary": summ,
        "cells_in_bbox": int(nx * ny), "step_cells": step,
        "dataset": _SS_DATASET, "variable": _SS_VARIABLE,
        "kind": "steady state (spun-up equilibrium)",
        "source": _SOURCE})


@mcp.tool()
def get_parflow_wtd_daily(lat: float, lon: float,
                          start_date: str, end_date: str = "") -> str:
    """TRANSIENT depth to water at one point, daily, from the CONUS1 baseline.

    A DIFFERENT QUESTION from get_parflow_wtd, and a different dataset and grid
    (conus1, not conus2). Use this to ask what the water table did over a
    period. Do NOT use one day of it as an initial condition unless you
    specifically want that day's state: a warm start wants the equilibrium
    field, which is get_parflow_wtd.

    Dates are ISO, 'YYYY-MM-DD'. end_date defaults to start_date (one day).
    """
    bad = _not_ready()
    if bad:
        return json.dumps(bad)
    xy = _xy(_DAILY_GRID, lat, lon)
    if xy is None:
        return json.dumps({"ok": False, "lat": lat, "lon": lon,
                           "error": "outside the CONUS1 domain",
                           "source": _SOURCE})
    x, y = xy
    end = str(end_date or start_date)
    try:
        import numpy as np
        arr = np.squeeze(np.asarray(
            _fetch(_DAILY_DATASET, _DAILY_VARIABLE, _DAILY_GRID,
                   [x, y, x + 1, y + 1], period="daily",
                   start_time=str(start_date), end_time=end)))
        series = [_clean(v) for v in np.ravel(arr)]
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False, "lat": lat, "lon": lon,
                           "error": f"{type(e).__name__}: {e}"[:300],
                           "source": _SOURCE})
    got = [v for v in series if v is not None]
    return json.dumps({
        "ok": True, "lat": lat, "lon": lon,
        "start_date": str(start_date), "end_date": end,
        "n_days": len(series), "depth_to_water_m": series,
        "summary": ({"min": min(got), "max": max(got),
                     "mean": round(sum(got) / len(got), 3)} if got else None),
        "grid_x": x, "grid_y": y, "grid": _DAILY_GRID,
        "dataset": _DAILY_DATASET, "variable": _DAILY_VARIABLE,
        "kind": "transient (daily) — NOT an initial condition",
        "source": _SOURCE})


@mcp.tool()
def describe_parflow_clm_capabilities() -> str:
    """What this server is, what it is not, and how it differs from fan_wtd."""
    return json.dumps({
        "source": _SOURCE,
        "variable": "water table depth (m below land surface, positive down)",
        "resolution": "1 km",
        "can": [
            "steady-state water table depth at a point, at many points in one "
            "call, or sampled across a bbox (CONUS2, spun-up equilibrium)",
            "daily transient water table depth at a point (CONUS1 baseline)",
            "answer anywhere in the CONUS domain — it is a full field, not tiles",
        ],
        "cannot": [
            "return data without a HydroFrame PIN registered on this machine. "
            "The catalogue is open; the gridded data is not. data_status() "
            "says exactly what is missing.",
            "run ParFlow. This reads PUBLISHED ParFlow-CLM output. Running the "
            "model is the pflotran/reaction server's neighbourhood, not this.",
            "give a measured water table. Every value here is SIMULATED. For "
            "measurements use usgs_water; for an observationally constrained "
            "equilibrium surface use fan_wtd.",
        ],
        "vs_fan_wtd": (
            "Both answer 'how deep is the water table'. fan_wtd is Fan et al. "
            "2013 — an equilibrium surface constrained by observations, "
            "provisioned here as static tiles, so coverage is whatever was "
            "copied in. This is a physics simulation on a regular 1 km CONUS "
            "grid, so every land cell has a value and a sampling design is not "
            "constrained by coverage. They are independent estimates: where "
            "they disagree strongly, that column is worth a second look."),
        "steady_state_vs_daily": (
            "ss_water_table_depth (CONUS2, static) is the equilibrium state and "
            "the right field to INITIALISE from. water_table_depth "
            "(conus1_baseline_mod, daily) is transient and answers what the "
            "water table did on given dates. Using one day of the daily field "
            "as an initial condition writes that day's weather into the initial "
            "state — which is why they are separate tools rather than one tool "
            "with a period argument."),
        "signup": _SIGNUP_URL, "pin": _PIN_URL,
        "data_ready": json.loads(data_status())["data_ready"],
    }, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
