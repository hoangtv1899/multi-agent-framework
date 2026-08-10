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
import os
from datetime import datetime

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from mcp.server.fastmcp import FastMCP

AMF = "https://amfcdn.lbl.gov/api/v1"
_SOURCE = "AmeriFlux (LBNL) web services"
_TIMEOUT = 90
# BOTH CONFIRMED LIVE from Compy, 2026-08-10 (HTTP 200). Checked rather than
# recalled: the obvious-looking ameriflux-data.lbl.gov/Pages/RequestAccount.aspx
# answers 522 (origin not responding), and a dead link as step one of "how to
# unblock this" is worse than saying nothing. Both hosts 403 a bare
# python-requests user agent and 200 a browser one, so a 403 here means "we were
# taken for a bot", not "the page is gone".
_REQUEST_URL = "https://ameriflux.lbl.gov/data/download-data/"
_REGISTER_URL = "https://ameriflux.lbl.gov/data/download-data/"
_POLICY_URL = "https://ameriflux.lbl.gov/data/data-policy/"

# THE DOWNLOAD ENDPOINT EXISTS AND IS POST-ONLY. Probed 2026-08-10 from Compy:
# GET returns 405, and OPTIONS answers `Allow: OPTIONS, POST`. So the series is
# reachable in principle — what is missing is an identity to reach it with, not
# an endpoint. That is why this server refuses rather than reports an absence.
_DOWNLOAD = f"{AMF}/data_download"

# Who is asking. AmeriFlux ties every download to a registered account and to an
# accepted data-use policy; there is no anonymous mode and no API key. Set both
# in env_compy.sh, beside the USGS key:
#     export AMERIFLUX_USER_ID=<your ameriflux username>
#     export AMERIFLUX_EMAIL=<the address the account was registered with>
_USER_ID = "AMERIFLUX_USER_ID"
_EMAIL = "AMERIFLUX_EMAIL"

mcp = FastMCP("ameriflux")


def _creds():
    """(user_id, email) from the environment, or (None, None).

    Never guessed and never defaulted. A download is attributed to a person who
    accepted the data-use policy, so inventing a plausible username would be
    submitting a request in someone else's name.
    """
    uid = (os.environ.get(_USER_ID) or "").strip()
    mail = (os.environ.get(_EMAIL) or "").strip()
    return (uid or None, mail or None)


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
        uid, mail = _creds()
        out["values_available"] = False
        out["ok"] = False
        if not (uid and mail):
            out["error"] = (
                "AmeriFlux flux data (BASE/FLUXNET) needs a REGISTERED ACCOUNT: "
                "every download is attributed to a person who accepted the "
                f"data-use policy. Set ${_USER_ID} and ${_EMAIL} and this tool "
                "will fetch. The tower list above is real and complete; the "
                "SERIES was not fetched. Treat this as a failed fetch, not as an "
                "absence of observations. "
                f"Register: {_REGISTER_URL}  ·  Policy: {_REQUEST_URL}")
            out["missing_credentials"] = [v for v, s in
                                          ((_USER_ID, uid), (_EMAIL, mail)) if not s]
            out["register"] = _REGISTER_URL
        else:
            out["error"] = (
                "credentials are set, but fetching series through get_et is not "
                "wired yet — call request_flux_data(site_ids=[...]) instead, "
                "which submits the download request and returns its URLs.")
            out["next"] = "request_flux_data"
    return json.dumps(out)


@mcp.tool()
def data_status() -> str:
    """Can this server fetch flux SERIES right now, and if not, what is missing?

    Separate from describe_ameriflux_capabilities because the answer CHANGES:
    capabilities are what the server can do in principle, this is whether the
    machine it is running on is currently able to do it. Same split as the
    fan_wtd server, and for the same reason — a capability list that says "can
    fetch ET" on a host with no credentials is a promise the run will break.
    """
    uid, mail = _creds()
    reachable, detail = None, None
    try:                        # POST-only: a GET returning 405 IS the proof
        r = requests.get(_DOWNLOAD, timeout=30, verify=False)
        reachable = r.status_code in (200, 405)
        detail = f"HTTP {r.status_code} (405 = POST-only, which is expected)"
    except Exception as e:                                      # noqa: BLE001
        reachable, detail = False, f"{type(e).__name__}: {e}"[:200]

    ready = bool(uid and mail and reachable)
    return json.dumps({
        "source": _SOURCE,
        "site_discovery": "always available — no account needed",
        "series_download_ready": ready,
        "endpoint": _DOWNLOAD,
        "endpoint_reachable": reachable,
        "endpoint_detail": detail,
        "credentials": {_USER_ID: bool(uid), _EMAIL: bool(mail)},
        "how_to_enable": None if ready else [
            f"1. Go to {_REGISTER_URL} and create an account (name, email, "
            "institution). It is free and immediate — there is no approval "
            "queue for the CC-BY-4.0 sites, which is most of them.",
            f"2. Accept the AmeriFlux Data Use Policy: {_POLICY_URL}. Most "
            "sites are CC-BY-4.0; some are LEGACY, which additionally asks you "
            "to notify the site PI before publishing.",
            f"3. Put both in env_compy.sh:  export {_USER_ID}=<username>  and  "
            f"export {_EMAIL}=<registered address>",
            "4. Re-run data_status() — series_download_ready should be true.",
        ],
        "policy_url": _POLICY_URL,
        "unverified": (
            "The download REQUEST BODY below has never been exercised against a "
            "real account, because this machine has none. The endpoint and its "
            "method are confirmed (GET->405, OPTIONS->'OPTIONS, POST', probed "
            "2026-08-10); the field names come from AmeriFlux's documented "
            "download API and are the part to check first if the first real "
            "call fails."),
        "request_shape": {
            "user_id": "<account username>", "user_email": "<address>",
            "data_product": "BASE-BADM", "data_policy": "CCBY4.0",
            "agree_policy": True, "intended_use": "model",
            "description": "<why you want it>", "site_ids": ["US-xxx"],
        },
    }, indent=2)


@mcp.tool()
def request_flux_data(site_ids: str, intended_use: str = "model",
                      description: str = "",
                      data_product: str = "BASE-BADM",
                      data_policy: str = "CCBY4.0") -> str:
    """Submit the AmeriFlux download request for these sites. NEEDS CREDENTIALS.

    site_ids: comma-separated, e.g. "US-Me2,US-Wrc".

    THIS IS AN OUTWARD-FACING ACTION and it is deliberately its own tool rather
    than a flag on get_et. It submits a request in the account holder's name,
    it is logged against them, and AmeriFlux emails the PIs of LEGACY-policy
    sites. A tool that did this as a side effect of "fetch the observations"
    would be doing something the caller did not ask for.

    Returns the response as-is. Downloads are delivered as file URLs (and by
    email), so the caller fetches them; this server does not warehouse data.

    NOT YET EXERCISED against a real account — see data_status()['unverified'].
    The failure to expect is a 400 naming a field, which is reported verbatim
    rather than swallowed, precisely so the first real call is diagnostic.
    """
    uid, mail = _creds()
    if not (uid and mail):
        return json.dumps({
            "ok": False,
            "error": f"no AmeriFlux credentials — set ${_USER_ID} and ${_EMAIL}",
            "missing": [v for v, s in ((_USER_ID, uid), (_EMAIL, mail)) if not s],
            "register": _REGISTER_URL,
            "next": "data_status",
        }, indent=2)

    sites = [s.strip() for s in str(site_ids).split(",") if s.strip()]
    if not sites:
        return json.dumps({"ok": False, "error": "no site_ids given"})

    body = {
        "user_id": uid, "user_email": mail,
        "data_product": data_product, "data_policy": data_policy,
        "agree_policy": True,
        "intended_use": intended_use,
        "description": description or
                       "1-D ELM column evaluation: ET at co-located towers.",
        "site_ids": sites,
        "is_test": False,
    }
    try:
        r = requests.post(_DOWNLOAD, json=body, timeout=_TIMEOUT, verify=False)
    except Exception as e:                                      # noqa: BLE001
        return json.dumps({"ok": False, "n_sites": len(sites),
                           "error": f"{type(e).__name__}: {e}"[:300]}, indent=2)

    try:
        payload = r.json()
    except ValueError:
        payload = {"raw": r.text[:2000]}
    return json.dumps({
        "ok": r.ok, "http_status": r.status_code,
        "n_sites_requested": len(sites), "site_ids": sites,
        "data_policy": data_policy,
        "response": payload,
        # A 400 here is the useful case: it names the field that is wrong, which
        # is exactly what an unverified request body needs told about it.
        "note": None if r.ok else
                "request rejected — the response above names what it objected "
                "to. Check the field names against data_status()['request_shape'].",
    }, indent=2)


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
            "return ET time series WITHOUT CREDENTIALS — every AmeriFlux "
            "download is attributed to a registered account that accepted the "
            "data-use policy. get_et(with_values=True) returns ok=false saying "
            "so, and data_status() says exactly what is missing.",
            "warehouse data — downloads arrive as file URLs and by email; the "
            "caller fetches them.",
        ],
        "with_credentials": (
            f"Set ${_USER_ID} and ${_EMAIL}, then request_flux_data(site_ids) "
            "submits the download. That is a separate tool on purpose: it is an "
            "outward-facing action logged against the account holder, not a "
            "side effect of fetching observations."),
        "credentials_present": all(_creds()),
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
