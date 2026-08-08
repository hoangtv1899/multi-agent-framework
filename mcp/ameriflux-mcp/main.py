#!/usr/bin/env python3
"""
AmeriFlux MCP Server — eddy-covariance flux towers, for ET validation.

WHY THIS EXISTS. The framework runs 1-D ELM columns: vertical water and energy,
no lateral routing. Of the three observables reception fetched before this
server, only SWE and water table are quantities a column can produce AT A POINT.
Streamflow is not — a gauge measures discharge integrated and routed over an
upstream area, so a column at the gauge's coordinates produces a point runoff
flux, not the thing the instrument measured. Pinning a column there spends a
column on a comparison that cannot be co-located (measured 2026-08-07: 14 of 40
pinned columns across 13 basins went to gauges, and they are what dragged the
ensemble into valley-bottom elevation bands).

Evapotranspiration is the observable best matched to this model. It is vertical,
it is local to the tower footprint (hundreds of metres, the same order as a
column), and ELM computes it directly. A column pinned at a flux tower compares
like with like.

Source: AmeriFlux web services (https://amfcdn.lbl.gov/api/v1), Lawrence Berkeley
National Laboratory. Site metadata is open and needs no key.

    get_et(bbox[, start_date, end_date, with_values])
        -> no dates: towers that exist, with coordinates, elevation and cover
           dates:    which of them were OPERATING then, and whether their data
                     is released and under what licence
           with_values: NOT AVAILABLE without credentials — see below

WHAT THIS SERVER CANNOT DO, AND WHY IT SAYS SO LOUDLY. AmeriFlux flux DATA
(BASE/FLUXNET products) is not served over an open endpoint: downloading it
requires a registered account and acceptance of the data-use policy, and the
files arrive by an emailed link rather than a REST response. So `with_values`
returns ok=false WITH the reason and the URL to request access. It does not
return an empty series, because "we could not look" and "there is nothing there"
are different findings — the same rule data_gather.py states, and the same
confusion that on 2026-08-07 turned a rate-limited USGS fetch into a planner
claiming a basin had no stream gauges.

So this server is a DISCOVERY source today: it answers "where are the towers,
were they running in my period, and is their data obtainable" — which is exactly
what column pinning needs, since pinning needs coordinates. The comparison step
needs the series, and that needs credentials.
"""
import json
from datetime import datetime

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from mcp.server.fastmcp import FastMCP

AMF = "https://amfcdn.lbl.gov/api/v1"
_SOURCE = "AmeriFlux (LBNL) web services"
_TIMEOUT = 90
_REQUEST_URL = "https://ameriflux.lbl.gov/data/download-data/"

mcp = FastMCP("ameriflux")


def _get(endpoint):
    r = requests.get(f"{AMF}/{endpoint}", timeout=_TIMEOUT, verify=False)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# PURE PARSERS (network-free, unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_bbox(bbox):
    """'min_lon,min_lat,max_lon,max_lat' -> four floats."""
    v = [float(x) for x in str(bbox).split(",")]
    if len(v) != 4:
        raise ValueError("bbox must be 'min_lon,min_lat,max_lon,max_lat'")
    return v


def _year(date_str, default=None):
    """Leading year of an ISO date, or default. Tolerates a bare year."""
    s = str(date_str or "").strip()
    if not s:
        return default
    try:
        return int(s[:4])
    except ValueError:
        return default


def _filter_sites(rows, bbox):
    """AmeriFlux sites inside the bbox, flattened to one record each.

    LOCATION_* arrive as strings and TOWER_END is ABSENT for an active tower
    (296 of 837 sites carry one), so a missing end is 'still running', not
    'unknown'. Both are normalised here so no consumer has to know.
    """
    mn_lon, mn_lat, mx_lon, mx_lat = bbox
    out = []
    for s in rows or []:
        loc = s.get("GRP_LOCATION") or {}
        try:
            lat = float(loc.get("LOCATION_LAT"))
            lon = float(loc.get("LOCATION_LONG"))
        except (TypeError, ValueError):
            continue
        if not (mn_lat <= lat <= mx_lat and mn_lon <= lon <= mx_lon):
            continue
        try:
            elev = float(loc.get("LOCATION_ELEV"))
        except (TypeError, ValueError):
            elev = None
        clim = s.get("GRP_CLIM_AVG") or {}
        out.append({
            "id": s.get("SITE_ID"),
            "name": s.get("SITE_NAME"),
            "lat": lat, "lon": lon, "elevation_m": elev,
            "igbp": s.get("IGBP"),                  # land cover, e.g. ENF, GRA
            "koeppen": clim.get("CLIMATE_KOEPPEN"),
            "mat_c": clim.get("MAT"), "map_mm": clim.get("MAP"),
            "tower_began": _year(s.get("TOWER_BEGAN")),
            "tower_end": _year(s.get("TOWER_END")),   # None => still active
            "url": s.get("URL_AMERIFLUX"),
        })
    return sorted(out, key=lambda r: r["id"] or "")


def _operating(site, yr_start, yr_end):
    """Did the tower overlap [yr_start, yr_end]?

    Conservative on both ends: an unknown start cannot be ruled out, and a
    missing end means the tower is still running. Returning None for 'cannot
    tell' keeps that distinct from False.
    """
    if yr_start is None or yr_end is None:
        return None
    began, end = site.get("tower_began"), site.get("tower_end")
    if began is None:
        return None
    return began <= yr_end and (end is None or end >= yr_start)


def _availability_index(av):
    """{site_id: {product: licence}} from the site_availability payload.

    Shape is {product: {licence: [[site_id, name], ...]}}, so it is inverted
    once here rather than scanned per site.
    """
    idx = {}
    for product, by_lic in (av or {}).items():
        for lic, rows in (by_lic or {}).items():
            for row in rows or []:
                sid = row[0] if isinstance(row, (list, tuple)) and row else None
                if sid:
                    idx.setdefault(sid, {})[product] = lic
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_et(bbox: str, start_date: str = "", end_date: str = "",
           with_values: bool = False) -> str:
    """AmeriFlux eddy-covariance towers in a bbox — the ET validation source.

    Same three shapes as get_swe and the usgs_water tools:

      no dates                 -> the towers that exist, with coordinates,
                                  elevation, IGBP land cover and their years.
      dates                    -> which of them were OPERATING in that window,
                                  and whether their data is released (product +
                                  licence).
      dates + with_values=True -> REFUSED, with the reason. AmeriFlux flux data
                                  needs a registered account and acceptance of
                                  the data-use policy; there is no open endpoint
                                  that returns the series. The refusal carries
                                  ok=false so a caller records a failed fetch
                                  rather than an empty basin.

    ET is the observable a 1-D column is best matched to: vertical, local to the
    tower footprint, and computed directly by ELM. A column pinned at a tower
    compares like with like, which a column pinned at a stream gauge cannot.
    """
    try:
        bb = _parse_bbox(bbox)
    except (TypeError, ValueError) as e:
        return json.dumps({"ok": False, "error": f"bad bbox: {e}",
                           "source": _SOURCE})

    try:
        rows = _get("site_display/AmeriFlux")
    except Exception as e:                       # noqa: BLE001 - reported, not raised
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:200],
                           "source": _SOURCE, "n_in_bbox": None, "towers": []})

    sites = _filter_sites(rows, bb)
    out = {"ok": True, "error": None, "source": _SOURCE,
           "n_in_bbox": len(sites), "towers": sites}

    yr_start, yr_end = _year(start_date), _year(end_date)
    if yr_start is not None or yr_end is not None:
        yr_start = yr_start if yr_start is not None else yr_end
        yr_end = yr_end if yr_end is not None else yr_start

        try:
            idx = _availability_index(_get("site_availability/AmeriFlux"))
        except Exception as e:                   # noqa: BLE001
            idx = {}
            out["availability_error"] = f"{type(e).__name__}: {e}"[:200]

        for s in sites:
            s["operating"] = _operating(s, yr_start, yr_end)
            s["data_products"] = idx.get(s["id"]) or {}
        running = [s for s in sites if s.get("operating")]
        out["period"] = f"{yr_start}-{yr_end}"
        out["n_operating"] = len(running)
        out["n_with_released_data"] = sum(1 for s in running if s["data_products"])

    if with_values:
        out["ok"] = False
        out["error"] = (
            "AmeriFlux flux data (BASE/FLUXNET) is not available over an open "
            "endpoint: it requires a registered AmeriFlux account and acceptance "
            "of the data-use policy, and is delivered as files rather than a REST "
            "response. The tower list above is real and complete; the SERIES was "
            "not fetched. Treat this as a failed fetch, not as an absence of "
            f"observations. Request access at {_REQUEST_URL}")
        out["values_available"] = False
        out["request_access"] = _REQUEST_URL
    return json.dumps(out)


@mcp.tool()
def describe_ameriflux_capabilities() -> str:
    """What this server can and cannot answer. Read before planning with it."""
    return json.dumps({
        "source": _SOURCE,
        "variable": "evapotranspiration (latent heat flux, LE -> ET)",
        "can": [
            "locate eddy-covariance towers in a bbox, with coordinates and elevation",
            "report IGBP land cover and Koeppen climate per tower",
            "say whether a tower was operating in a given period",
            "say whether its data is released, and under which licence",
        ],
        "cannot": [
            "return ET time series — AmeriFlux data download needs a registered "
            "account and data-use-policy acceptance; there is no open series "
            "endpoint. get_et(with_values=True) returns ok=false saying so.",
        ],
        "why_it_matters": (
            "The framework runs 1-D columns with no lateral routing. ET is "
            "vertical and local to the tower footprint, so a column pinned at a "
            "tower measures the same quantity the instrument does. A stream "
            "gauge integrates and routes an upstream area, so a column pinned "
            "there cannot be a co-located comparison however it is labelled."),
        "n_sites_global": 837,
        "n_sites_usa": 677,
        "request_access": _REQUEST_URL,
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
