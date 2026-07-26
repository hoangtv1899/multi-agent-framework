#!/usr/bin/env python3
"""
usgs_water MCP — data layer (network + pure parsers).

Split out of main.py so the parsers are importable and unit-testable without
the stdio transport (tests/test_mcp_tools.py::TestGroundwaterParsers loads this
module directly). main.py is a thin FastMCP wrapper over the functions here.

Upstream USGS web services (free, no key):
    OGC API   https://api.waterdata.usgs.gov/ogcapi/v0
        collections/monitoring-locations/items   -> site discovery
        collections/field-measurements/items     -> groundwater level (param 72019,
                                                     depth to water, ft below land surface)
    NWIS IV   https://waterservices.usgs.gov/nwis/iv/   -> legacy instantaneous-values
                                                     passthrough (fetch_usgs_data)

Notes / lessons carried from the NERSC original:
  * Groundwater levels have NO dedicated OGC collection — they live in
    `field-measurements` filtered by `parameter_code=72019`.
  * `value` in field-measurements is a STRING in FEET; convert to metres here.
  * The instantaneous-values host is `waterservices.usgs.gov`. The legacy
    `waterdata.usgs.gov/nwis/iv/` path now 404s (verified from Compy 2026-07-23);
    the OGC endpoints stay on `api.waterdata.usgs.gov`.

Pure parsers `_parse_sites` / `_parse_wtd` are network-free by design.
HTTP client is httpx (synchronous), one short timeout, no retry loops.
"""
import httpx

OGC_BASE = "https://api.waterdata.usgs.gov/ogcapi/v0"
NWIS_IV = "https://waterservices.usgs.gov/nwis/iv/"

FT_TO_M = 0.3048
WTD_PARAMETER_CODE = "72019"          # depth to water, ft below land surface
_TIMEOUT = 30                         # seconds; matches mcp_config timeout


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK (thin, no retries — one request, explicit timeout)
# ─────────────────────────────────────────────────────────────────────────────

def _ogc_items(collection, params):
    """GET one OGC API `items` page and return the parsed FeatureCollection."""
    url = f"{OGC_BASE}/collections/{collection}/items"
    q = {"f": "json", **params}
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True, verify=True) as cx:
        r = cx.get(url, params=q)
        r.raise_for_status()
        return r.json()


def fetch_monitoring_locations(bbox, agency_code="", site_type_code="", limit=50):
    """OGC monitoring-locations FeatureCollection for a bbox (near-raw passthrough).

    bbox is the OGC string 'min_lon,min_lat,max_lon,max_lat'.
    """
    params = {"bbox": bbox, "limit": int(limit)}
    if site_type_code:
        params["site_type_code"] = site_type_code
    if agency_code:
        params["agency_code"] = agency_code
    return _ogc_items("monitoring-locations", params)


def fetch_field_measurements(monitoring_location_id, limit=100,
                             parameter_code=WTD_PARAMETER_CODE):
    """OGC field-measurements FeatureCollection (depth-to-water, param 72019)."""
    params = {"monitoring_location_id": monitoring_location_id,
              "parameter_code": parameter_code, "limit": int(limit)}
    return _ogc_items("field-measurements", params)


def fetch_nwis_iv(sites, parameter_codes="00060,00065,00010", period="P1D"):
    """Legacy NWIS instantaneous-values (WaterML-JSON) passthrough."""
    params = {"format": "json", "sites": sites,
              "parameterCd": parameter_codes, "period": period}
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True, verify=True) as cx:
        r = cx.get(NWIS_IV, params=params)
        r.raise_for_status()
        return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# PURE PARSERS (network-free, unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _parse_sites(data):
    """OGC monitoring-locations FeatureCollection -> list of flat site dicts.

    Each site: id (== OGC feature id, e.g. 'USGS-1', usable directly as
    monitoring_location_id), lat, lon, name, site_type_code, aquifer_code,
    altitude. Empty / missing features -> [].
    """
    sites = []
    for f in (data or {}).get("features", []) or []:
        props = f.get("properties", {}) or {}
        coords = ((f.get("geometry") or {}).get("coordinates")) or []
        lon = round(coords[0], 5) if len(coords) > 0 and coords[0] is not None else None
        lat = round(coords[1], 5) if len(coords) > 1 and coords[1] is not None else None
        sid = f.get("id") or props.get("id") or props.get("monitoring_location_id")
        sites.append({
            "id": sid,
            "monitoring_location_id": sid,
            "lat": lat,
            "lon": lon,
            "name": props.get("monitoring_location_name"),
            "site_type_code": props.get("site_type_code"),
            "aquifer_code": props.get("aquifer_code"),
            "altitude": props.get("altitude"),
        })
    return sites


def _parse_wtd(data):
    """OGC field-measurements FeatureCollection -> (observations, summary).

    Input feature properties carry `time`, `value` (STRING, feet below land
    surface) and `approval_status`. Non-numeric values are skipped; obs are
    sorted ascending by time; feet are converted to metres. Depths are metres
    below land surface (larger = deeper water table). Empty -> ([], zeroed
    summary with None stats).
    """
    obs = []
    for f in (data or {}).get("features", []) or []:
        props = f.get("properties", {}) or {}
        try:
            ft = float(props.get("value"))
        except (TypeError, ValueError):
            continue
        obs.append({
            "time": props.get("time"),
            "depth_to_water_ft": round(ft, 3),
            "depth_to_water_m": round(ft * FT_TO_M, 3),
            "approval_status": props.get("approval_status"),
            "qualifier": props.get("qualifier"),
            "unit": props.get("unit_of_measure"),
        })
    obs.sort(key=lambda o: o["time"] or "")

    depths = [o["depth_to_water_m"] for o in obs]
    if depths:
        summary = {
            "n_obs": len(depths),
            "min_depth_m": round(min(depths), 3),
            "max_depth_m": round(max(depths), 3),
            "mean_depth_m": round(sum(depths) / len(depths), 3),
            "median_depth_m": round(_median(depths), 3),
            "latest_depth_m": obs[-1]["depth_to_water_m"],
            "first_time": obs[0]["time"],
            "latest_time": obs[-1]["time"],
        }
    else:
        summary = {"n_obs": 0, "min_depth_m": None, "max_depth_m": None,
                   "mean_depth_m": None, "median_depth_m": None,
                   "latest_depth_m": None, "first_time": None, "latest_time": None}
    return obs, summary
