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
MI2_TO_KM2 = 2.58999                  # USGS reports drainage area in sq miles
WTD_PARAMETER_CODE = "72019"          # depth to water, ft below land surface
Q_PARAMETER_CODE = "00060"            # discharge, cubic feet per second
DAILY_PAGE = 5000                     # rows per page on the daily collection
MAX_PAGES  = 10                       # ~50k records; beyond this, say truncated
# The OGC API answers a COLD (bbox, period) query in ~30-60 s and the same
# query in 0.2 s once its cache is warm. 30 s therefore failed exactly where it
# hurt most — the first time anyone asked about a new basin or a new year — and
# the timeout surfaced as "no data", which reads like a finding. Must stay below
# the usgs_water timeout in mcp_config.json (120 s) so an HTTP error is reported
# as an error rather than being cut off by the client first.
_TIMEOUT = 90


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


def fetch_daily_coverage(bbox, start_date, end_date, limit=DAILY_PAGE,
                         parameter_code=Q_PARAMETER_CODE, max_pages=MAX_PAGES):
    """Which sites in this bbox actually have records in this window.

    This is the question a station list cannot answer. A bbox can hold 93
    stream gauges while ONE of them has data for the year you mean to simulate
    — and which one varies by year. Discovering that after a run has finished
    is too late to act on it.

    Two details are load-bearing, both learned the hard way on a 3-year window:

    `properties=` — we count records, we never read their values, so asking for
    the two fields we use instead of all twenty cuts the payload ~100x. Without
    it a 3-year query took 60 s and came back **HTTP 200 with zero features** —
    the API gives up quietly on a wide response, and a silent "no gauge has
    data" is a far worse answer than an error, because it looks like a finding.

    Pagination — the collection returns a `next` link rather than a total, so a
    single page silently truncates a multi-year window (2135 records for 1987-89
    against a 1000-row page) and every count downstream comes out short. We
    follow the links and report `truncated` if we ever hit the cap, so a partial
    count is never mistaken for a complete one.
    """
    url = f"{OGC_BASE}/collections/daily/items"
    q = {"f": "json", "bbox": bbox, "parameter_code": parameter_code,
         "datetime": f"{start_date}/{end_date}", "limit": int(limit),
         "properties": "monitoring_location_id,value"}
    feats, pages, truncated = [], 0, False
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True, verify=True) as cx:
        r = cx.get(url, params=q)
        r.raise_for_status()
        page = r.json()
        while True:
            feats.extend(page.get("features") or [])
            pages += 1
            nxt = next((l.get("href") for l in (page.get("links") or [])
                        if l.get("rel") == "next"), None)
            if not nxt:
                break
            if pages >= max_pages:
                truncated = True
                break
            r = cx.get(nxt)
            r.raise_for_status()
            page = r.json()
    return {"type": "FeatureCollection", "features": feats,
            "truncated": truncated, "n_pages": pages}


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


def _parse_coverage(daily_data, sites_data, min_days=300):
    """(daily FeatureCollection, monitoring-locations FeatureCollection)
    -> {available[], n_available, n_sites, n_without_records}.

    Network-free, like the other parsers. `available` carries what a caller
    needs to decide whether a station is USEFUL, not merely present:
    coordinates (to place a column or draw a map) and drainage area in km2
    (to judge whether the station represents the modelled domain at all).
    """
    counts = {}
    for f in (daily_data or {}).get("features", []) or []:
        props = f.get("properties", {}) or {}
        sid = props.get("monitoring_location_id")
        if sid and props.get("value") is not None:
            counts[sid] = counts.get(sid, 0) + 1

    meta = {}
    for f in (sites_data or {}).get("features", []) or []:
        props = f.get("properties", {}) or {}
        num = props.get("monitoring_location_number")
        sid = f.get("id") or (f"USGS-{num}" if num else None)
        if not sid:
            continue
        coords = ((f.get("geometry") or {}).get("coordinates")) or []
        da = props.get("drainage_area")
        try:
            da_km2 = round(float(da) * MI2_TO_KM2, 1) if da is not None else None
        except (TypeError, ValueError):
            da_km2 = None
        meta[sid] = {
            "name": props.get("monitoring_location_name"),
            "lat": round(coords[1], 5) if len(coords) > 1 and coords[1] is not None else None,
            "lon": round(coords[0], 5) if len(coords) > 0 and coords[0] is not None else None,
            "drainage_area_km2": da_km2,
        }

    available = []
    for sid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if n < int(min_days):
            continue
        available.append({"id": sid, "n_days": n, **meta.get(sid, {})})

    n_sites = len((sites_data or {}).get("features", []) or [])
    out = {"n_sites": n_sites, "n_available": len(available),
           "n_without_records": max(n_sites - len(available), 0),
           "min_days": int(min_days), "available": available}
    if (daily_data or {}).get("truncated"):
        # An undercount looks exactly like a real finding, so it must announce
        # itself rather than be inferred from a suspiciously round number.
        out["truncated"] = True
        out["warning"] = ("record counts are INCOMPLETE — the daily query hit "
                          "its page cap; treat n_days as a lower bound")
    return out


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
