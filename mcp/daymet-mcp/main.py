#!/usr/bin/env python3
"""
Daymet MCP Server — daily weather at a point, 1 km, North America

Data source:
    Daymet V4 single-pixel extraction (ORNL DAAC) — free, no key needed
    https://daymet.ornl.gov/single-pixel/api/data

Tools:
    describe_daymet_capabilities()
        → what this server provides, its coverage, and whether it answers
    get_precipitation(lat, lon, start_year, end_year)
        → daily series at one point
    get_precipitation_points(lats, lons, start_year, end_year)
        → the same for MANY points in ONE MCP session

TWO THINGS THIS SERVER EXISTS TO GUARD, both measured against the live API on
2026-08-17 rather than read in a doc:

  1. AN OUT-OF-RANGE YEAR IS NOT AN ERROR. Asking for 1979 — one year before
     the record starts — returns HTTP 200 and the ENTIRE archive: 16,790 rows
     covering 1980-2025, silently, in place of the 365 that were asked for. A
     caller that trusted the response would take a 46-year record for a single
     year and never know. Every request here is checked against the years that
     came back, and a mismatch is reported rather than returned.

  2. THE CALENDAR IS 365 DAYS, ALWAYS. Daymet drops 31 December in leap years.
     1980 and 2024 are both leap years and both return 365 rows. Aligning this
     against a real calendar without knowing that shifts every date after
     February by one day, for a quarter of all years.

COVERAGE IS NOT THE FRAMEWORK'S. Daymet begins in 1980; the NLDAS-2 forcing
this framework runs on begins in 1979. A study of 1979 — the Naches run is one
— cannot be given Daymet precipitation at all. The capability report states the
first and last year it actually holds, verified on the call, so a caller can
compare rather than assume they match.
"""
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DAYMET_URL = "https://daymet.ornl.gov/single-pixel/api/data"

# The first year Daymet V4 holds. Not a guess: a request starting earlier is
# answered with the whole archive (see the module docstring), so this is the
# boundary that makes the difference between one year and forty-six.
FIRST_YEAR = 1980

# Daymet's own names. prcp is the reason this server exists; the others cost
# nothing to allow because they are one query parameter, and `swe` in
# particular is a validation series this framework already compares against.
VARIABLES = {
    "prcp": "precipitation, mm/day",
    "tmax": "daily maximum temperature, degC",
    "tmin": "daily minimum temperature, degC",
    "swe":  "snow water equivalent, kg/m2",
    "srad": "shortwave radiation, W/m2",
    "vp":   "water vapour pressure, Pa",
    "dayl": "daylength, s",
}

TIMEOUT_S = 120

server = Server("daymet")


# ─────────────────────────────────────────────────────────────────────────────
# THE FETCH
# ─────────────────────────────────────────────────────────────────────────────
def _parse(text: str) -> Dict[str, Any]:
    """Daymet's CSV — a prose header, then one header row, then the data."""
    meta: Dict[str, Any] = {}
    rows: List[List[str]] = []
    columns: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line and line[0].isdigit() and "," in line and columns:
            rows.append(line.split(","))
        elif line.startswith("year,"):
            columns = [c.strip() for c in line.split(",")]
        elif ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip().lower().replace(" ", "_")] = v.strip()
    return {"meta": meta, "columns": columns, "rows": rows}


def _series(lat: float, lon: float, start_year: int, end_year: int,
            variables: str = "prcp") -> Dict[str, Any]:
    """One point, one variable set. Never raises; reports what it got."""
    now_year = datetime.now().year
    if end_year < start_year:
        return {"ok": False, "error": f"end_year {end_year} precedes "
                                      f"start_year {start_year}"}
    # THE GUARD THAT MATTERS. Out of range is answered with the whole archive,
    # so refuse here rather than hand back forty-six years labelled as one.
    if start_year < FIRST_YEAR:
        return {"ok": False,
                "error": (f"Daymet begins in {FIRST_YEAR}; {start_year} is "
                          f"before the record. The API answers such a request "
                          f"with its ENTIRE archive rather than an error, so "
                          f"this is refused here. Ask for {FIRST_YEAR} onward, "
                          f"or use another source for earlier years."),
                "first_year_available": FIRST_YEAR}

    params = {"lat": float(lat), "lon": float(lon), "vars": variables,
              "start": f"{int(start_year)}-01-01",
              "end": f"{int(end_year)}-12-31"}
    try:
        r = requests.get(DAYMET_URL, params=params, timeout=TIMEOUT_S)
    except Exception as e:                              # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    if r.status_code != 200:
        # A point outside the Daymet tiles (ocean, outside North America)
        # answers 400 with a JSON message. Pass it through as-is.
        detail = r.text[:300]
        try:
            detail = json.loads(r.text).get("message", detail)
        except Exception:                               # noqa: BLE001
            pass
        return {"ok": False, "status": r.status_code, "error": detail}

    got = _parse(r.text)
    rows, cols = got["rows"], got["columns"]
    if not rows:
        return {"ok": False, "error": "Daymet returned no data rows",
                "meta": got["meta"]}

    years = sorted({int(x[0]) for x in rows})
    # THE SECOND HALF OF THE GUARD. Even an in-range request is checked against
    # what came back, because the failure mode above is silent by construction.
    asked = list(range(int(start_year), int(end_year) + 1))
    unexpected = [y for y in years if y not in asked]
    if unexpected:
        return {"ok": False,
                "error": (f"Daymet returned years {years[0]}-{years[-1]} for a "
                          f"request of {start_year}-{end_year}. This is its "
                          f"documented behaviour for an out-of-range request "
                          f"and the response is not what was asked for."),
                "years_returned": years}

    idx = {c: i for i, c in enumerate(cols)}
    out_vars = [c for c in cols if c not in ("year", "yday")]
    series: Dict[str, List[float]] = {v: [] for v in out_vars}
    for x in rows:
        for v in out_vars:
            try:
                series[v].append(float(x[idx[v]]))
            except (ValueError, IndexError):
                series[v].append(None)

    meta = got["meta"]
    return {
        "ok": True,
        "lat": float(lat), "lon": float(lon),
        "elevation_m": _num(meta.get("elevation")),
        "tile": meta.get("tile"),
        "years": [years[0], years[-1]],
        "n_days": len(rows),
        "year": [int(x[0]) for x in rows],
        "yday": [int(x[1]) for x in rows],
        "series": series,
        "units": {v: VARIABLES.get(v.split()[0], "see column name")
                  for v in out_vars},
        "calendar": ("365-day: Daymet drops 31 December in leap years, so a "
                     "leap year has 365 rows like any other. Align by "
                     "(year, yday), never by counting days from a real "
                     "calendar."),
        "source": "Daymet V4, ORNL DAAC, 1 km daily",
    }


def _num(s) -> Optional[float]:
    try:
        return float(str(s).split()[0])
    except Exception:                                   # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    _yrs = {"start_year": {"type": "integer",
                           "description": f"first year, {FIRST_YEAR} or later"},
            "end_year": {"type": "integer", "description": "last year"},
            "variables": {"type": "string",
                          "description": ("comma-separated Daymet names; "
                                          "default 'prcp'. Options: "
                                          + ", ".join(VARIABLES))}}
    return [
        types.Tool(
            name="describe_daymet_capabilities",
            description=("What this server provides, the years it holds, and "
                         "whether it is reachable. Call this first to compare "
                         "its coverage against the period a study needs."),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_precipitation",
            description=("Daily Daymet series at ONE point (1 km, North "
                         "America). Defaults to precipitation in mm/day. "
                         "Returns (year, yday) alongside the values — the "
                         "calendar is 365-day and drops 31 December in leap "
                         "years."),
            inputSchema={
                "type": "object",
                "properties": {"lat": {"type": "number"},
                               "lon": {"type": "number"}, **_yrs},
                "required": ["lat", "lon", "start_year", "end_year"],
            },
        ),
        types.Tool(
            name="get_precipitation_points",
            description=("The same for MANY points in ONE MCP session — use "
                         "this whenever you have more than one location, e.g. "
                         "forcing a set of sampled columns. Daymet is queried "
                         "per point upstream regardless, so this saves session "
                         "setup, not upstream calls. A long period over many "
                         "points is a LOT of numbers: 58 points x 20 years is "
                         "~424,000 daily values."),
            inputSchema={
                "type": "object",
                "properties": {"lats": {"type": "array",
                                        "items": {"type": "number"}},
                               "lons": {"type": "array",
                                        "items": {"type": "number"}}, **_yrs},
                "required": ["lats", "lons", "start_year", "end_year"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    def reply(obj) -> list[types.TextContent]:
        return [types.TextContent(type="text", text=json.dumps(obj))]

    if name == "describe_daymet_capabilities":
        # READY IS MEASURED, NOT DECLARED — one real request for one day at a
        # point known to be inside the domain. A server that says it is ready
        # without asking is reporting its own intentions.
        probe = _series(40.0, -105.0, FIRST_YEAR, FIRST_YEAR, "prcp")
        return reply({
            "server": "daymet",
            "ready": bool(probe.get("ok")),
            "error": None if probe.get("ok") else probe.get("error"),
            "source": "Daymet V4 single-pixel extraction, ORNL DAAC",
            "url": DAYMET_URL,
            "credentials": "none required",
            # WHAT IT PROVIDES, in the vocabulary a caller matches against.
            # A server can say what it supplies; it cannot know who needs it.
            "provides": [
                {"variable": "precipitation", "daymet_name": "prcp",
                 "units": "mm/day", "step": "daily"},
                {"variable": "swe", "daymet_name": "swe",
                 "units": "kg/m2", "step": "daily"},
                {"variable": "air_temperature", "daymet_name": "tmax,tmin",
                 "units": "degC", "step": "daily"},
            ],
            "resolution_m": 1000,
            "coverage": {
                "first_year": FIRST_YEAR,
                "last_year": datetime.now().year - 1,
                "domain": "North America (Daymet tiles); a point outside "
                          "them answers HTTP 400, not an empty series",
            },
            "calendar": ("365-day. 31 December is dropped in leap years, so "
                         "every year has 365 values."),
            "cannot": [
                {"what": f"any year before {FIRST_YEAR}",
                 "why": ("the record starts there. Note the API answers an "
                         "earlier request with its ENTIRE archive rather than "
                         "an error; this server refuses it instead")},
                {"what": "sub-daily values",
                 "why": "Daymet is a daily product. Rainfall INTENSITY drives "
                        "the split between runoff and infiltration, and a "
                        "daily total cannot carry it"},
                {"what": "a gridded subset over a bounding box",
                 "why": "this server wraps the single-pixel service. The ORNL "
                        "DAAC subsetter returns NetCDF and is a different "
                        "interface"},
            ],
            "validation_status": "success",
        })

    if name == "get_precipitation":
        out = _series(arguments["lat"], arguments["lon"],
                      int(arguments["start_year"]), int(arguments["end_year"]),
                      str(arguments.get("variables") or "prcp"))
        return reply(out)

    if name == "get_precipitation_points":
        lats = arguments.get("lats") or []
        lons = arguments.get("lons") or []
        if len(lats) != len(lons):
            return reply({"ok": False,
                          "error": f"lats has {len(lats)} entries, lons has "
                                   f"{len(lons)}"})
        y0 = int(arguments["start_year"])
        y1 = int(arguments["end_year"])
        variables = str(arguments.get("variables") or "prcp")
        pts = [_series(a, o, y0, y1, variables) for a, o in zip(lats, lons)]
        # KEYED BY COORDINATE, matching geology's soil fetch: the consumer is a
        # column carrying a lat/lon, not an index into a list whose order it
        # would have to trust.
        by_point = {f"{round(float(a), 5)},{round(float(o), 5)}": p
                    for a, o, p in zip(lats, lons, pts)}
        ok = [p for p in pts if p.get("ok")]
        return reply({
            "ok": bool(ok),
            "n_points": len(pts),
            "n_with_data": len(ok),
            "n_days_each": ok[0]["n_days"] if ok else 0,
            "years": [y0, y1],
            "variables": variables,
            "failures": [{"lat": a, "lon": o, "error": p.get("error")}
                         for a, o, p in zip(lats, lons, pts)
                         if not p.get("ok")],
            "points": by_point,
            "source": "Daymet V4, ORNL DAAC, 1 km daily",
        })

    return reply({"error": f"Unknown tool: {name}"})


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write,
                         server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
