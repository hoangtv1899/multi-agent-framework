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
import os
import random
import time
import sys
from typing import Any, Dict, List, Optional

import httpx

OGC_BASE = "https://api.waterdata.usgs.gov/ogcapi/v0"

# The OGC API allows 1000 requests per window unauthenticated and answers 429
# OVER_RATE_LIMIT after that, with a `retry-after` telling you exactly how long.
# A 15-basin evaluation exhausted it, and every fetch in the tail of that run
# came back ok=false -- correctly reported, but no data.
#
# Set USGS_API_KEY (free, https://api.waterdata.usgs.gov/signup/) for a higher
# limit. Without one the retry below still recovers, just slowly.
USGS_API_KEY = os.environ.get("USGS_API_KEY", "").strip()
_RATE_RETRIES = 3
_RATE_BASE_WAIT = 5        # seconds; doubles per attempt when the server sends
                           # no retry-after, with jitter so parallel callers do
                           # not retry in lockstep
_MIN_REQUEST_ROOM = 20     # leave this much of the budget for the retry itself
# ONE budget for all waiting inside a single call, kept below the usgs_water
# timeout in mcp_config.json so a retry that would succeed is not killed by the
# client first. That mismatch is what made the old retry unreachable.
#
# 200 left a third of the client's 300 s patience unused, and _get gives up the
# moment a wait does not fit — so a server asking for 200 s was refused with 100
# spare. Raised to fit: 240 + _MIN_REQUEST_ROOM + the 30 s a cold OGC query takes
# still lands inside 300.
_RATE_BUDGET = 240
NWIS_IV = "https://waterservices.usgs.gov/nwis/iv/"

FT_TO_M = 0.3048
MI2_TO_KM2 = 2.58999                  # USGS reports drainage area in sq miles
Q_PARAMETER_CODE = "00060"            # discharge, cubic feet per second
DAILY_PAGE = 5000                     # rows per page on the daily collection
MAX_PAGES  = 10                       # ~50k records; beyond this, say truncated
MAX_YEARS  = 12                       # chunks per dated query; see fetch_daily
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

def _headers():
    return {"X-Api-Key": USGS_API_KEY} if USGS_API_KEY else {}


def _retry_after(r, attempt):
    """Seconds to wait before retrying a 429, from the server or from backoff.

    The server's own `retry-after` wins when present — it knows its window. With
    no header, back off exponentially from _RATE_BASE_WAIT with jitter, so N
    concurrent callers that all trip the limit do not then retry in lockstep.
    """
    hdr = r.headers.get("retry-after")
    if hdr:
        try:
            return max(0.0, float(hdr))
        except (TypeError, ValueError):
            pass
    return _RATE_BASE_WAIT * (2 ** attempt) * (1.0 + random.random() * 0.25)


def _get(cx, url, params=None, deadline=None):
    """One GET that respects the server's own rate-limit instruction.

    429 carries `retry-after` in seconds. Honouring it is the difference between
    recovering and reporting a basin as unobserved: on the 2026-08-07 chain run
    6 of 13 basins came back 429, and the planner had to report them as fetch
    failures.

    TWO BUGS LIVED HERE, and they cancelled the retry entirely.

    1. `wait = min(retry_after, _RATE_MAX_WAIT)` then `if wait >= _RATE_MAX_WAIT:
       return r`. The clamp set wait to exactly the ceiling, which then tripped
       the give-up test — so any retry-after at or above the ceiling produced an
       INSTANT failure with no wait at all. The clamp and the check fought each
       other, and the longer the server asked us to wait, the faster we gave up.

    2. The retry budget did not fit inside the client's patience. Three attempts
       sleeping up to the ceiling each is minutes, while mcp_config gave
       usgs_water 120 s — so even a retry that would have succeeded was killed by
       the MCP client first, and the caller saw a timeout rather than data.

    Now there is ONE budget, `deadline`, and the loop only sleeps when the wait
    plus one more request still fits inside it. Running out of budget returns the
    429 so the caller reports a failed fetch — which is the honest answer, and
    distinguishable from an empty basin.

    THE RETRY STILL DID NOT HOLD, and on 2026-08-08 chattahoochee_2000 and
    centralcoast_1998 came back 429 on their FIRST field-measurements request
    with 30 s between cases. The message said only "429", which is not enough to
    tell an exhausted retry from a retry that never slept — so the attempt log
    now travels with the error. Guessing at which of the two it was is how the
    previous two fixes were written.
    """
    if deadline is None:
        deadline = time.monotonic() + _RATE_BUDGET
    log = []
    for attempt in range(_RATE_RETRIES):
        r = cx.get(url, params=params, headers=_headers())
        if r.status_code != 429:
            if log:
                print(f"recovered from 429 after {log}", file=sys.stderr)
            return r
        hdr = r.headers.get("retry-after")
        if attempt == _RATE_RETRIES - 1:
            log.append(f"attempt {attempt + 1}: 429, retry-after={hdr}, "
                       f"no attempts left")
            break
        wait = _retry_after(r, attempt)
        left = deadline - time.monotonic()
        # Only sleep if the wait AND a further request still fit the budget.
        if wait + _MIN_REQUEST_ROOM > left:
            log.append(f"attempt {attempt + 1}: 429, retry-after={hdr}, "
                       f"wanted {wait:.0f}s but only {left:.0f}s of budget left")
            break
        log.append(f"attempt {attempt + 1}: 429, retry-after={hdr}, "
                   f"slept {wait:.0f}s")
        time.sleep(wait)
    # The caller raises on this response; attach what we tried so the failure is
    # diagnosable from the record the planner is shown.
    r.rate_limit_log = log
    print(f"429 giving up: {'; '.join(log)}", file=sys.stderr)
    return r


def _ogc_items(collection, params, max_pages=MAX_PAGES):
    """GET an OGC API `items` query TO THE END and return the merged features.

    `limit` is a PAGE SIZE, never a total. This function used to read the first
    page and stop, which made every caller's `limit` a silent ceiling: the
    server returns exactly that many rows, reports `numberReturned` equal to it,
    and offers no `numberMatched` to compare against — so a truncated answer is
    indistinguishable from a complete one.

    That cost 236 of 251 Naches wells their coordinates on 2026-08-11. The site
    catalogue was asked for 500 of the 5,312 wells in the bbox; the 251 we
    needed were mostly not in the page we got, so they came back unplaceable and
    the basin looked like it had no usable wells. The same ceiling sits under
    every other caller here — stream-gauge drainage areas at 200, record spans
    at 500 — and for a gauge a missed lookup is worse than a missed coordinate,
    because `attach_daily_series` skips any gauge without a drainage area and
    the whole hydrograph disappears.

    Following the `next` link is what the collection is designed for. Cost is
    small: the full 5,312-well Naches catalogue pages through in 5.4 s.

    Truncation is now only possible at max_pages, and it SAYS SO.
    """
    url = f"{OGC_BASE}/collections/{collection}/items"
    q = {"f": "json", **params}
    feats, pages, truncated, head = [], 0, False, None
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True, verify=True) as cx:
        r = _get(cx, url, q)
        r.raise_for_status()
        page = r.json()
        while True:
            if head is None:
                head = {k: v for k, v in page.items() if k != "features"}
            feats.extend(page.get("features") or [])
            pages += 1
            nxt = next((l.get("href") for l in (page.get("links") or [])
                        if l.get("rel") == "next"), None)
            if not nxt:
                break
            if pages >= max_pages:
                truncated = True
                break
            r = _get(cx, nxt)
            r.raise_for_status()
            page = r.json()
    return {**(head or {}), "type": "FeatureCollection", "features": feats,
            "truncated": truncated, "n_pages": pages}


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


def _year_chunks(start_date, end_date):
    """Split a date window into one (start, end) pair per calendar year.

    The OGC `daily` collection cancels any query that exceeds ~60 s of server
    time — "Long running query has been cancelled", returned as a 400. A
    four-year window for one bbox sits just under that; five years is over it.
    Since the failure is TIME, not result size, paging does not help: the first
    page never arrives. Splitting by year keeps every request well inside the
    budget and makes long windows work at all.
    """
    y0, y1 = int(str(start_date)[:4]), int(str(end_date)[:4])
    if y1 < y0:
        return []
    out = []
    for y in range(y0, y1 + 1):
        lo = start_date if y == y0 else f"{y}-01-01"
        hi = end_date if y == y1 else f"{y}-12-31"
        out.append((lo, hi))
    return out


def fetch_daily(bbox, start_date, end_date, limit=DAILY_PAGE,
                parameter_code=Q_PARAMETER_CODE, max_pages=MAX_PAGES,
                with_time=False):
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
    props = ("monitoring_location_id,time,value" if with_time
             else "monitoring_location_id,value")
    chunks = _year_chunks(start_date, end_date)
    if not chunks:
        raise ValueError(f"end_date {end_date} precedes start_date {start_date}")
    if len(chunks) > MAX_YEARS:
        raise ValueError(
            f"{len(chunks)}-year window is too long to confirm record by record. "
            f"Call without dates for the period of record instead, or ask about "
            f"at most {MAX_YEARS} years.")

    feats, pages, truncated = [], 0, False
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True, verify=True) as cx:
        for lo, hi in chunks:
            q = {"f": "json", "bbox": bbox, "parameter_code": parameter_code,
                 "datetime": f"{lo}/{hi}", "limit": int(limit),
                 "properties": props}
            r = _get(cx, url, q)
            if r.status_code == 400 and "Long running" in r.text:
                # Server-side cancellation is transient (it depends on cache
                # warmth), so one retry usually succeeds where the first failed.
                r = _get(cx, url, q)
            r.raise_for_status()
            page = r.json()
            while True:
                feats.extend(page.get("features") or [])
                pages += 1
                nxt = next((l.get("href") for l in (page.get("links") or [])
                            if l.get("rel") == "next"), None)
                if not nxt:
                    break
                if pages >= max_pages * len(chunks):
                    truncated = True
                    break
                r = _get(cx, nxt)
                r.raise_for_status()
                page = r.json()
            if truncated:
                break
    return {"type": "FeatureCollection", "features": feats,
            "truncated": truncated, "n_pages": pages, "n_chunks": len(chunks)}


def fetch_record_spans(bbox, parameter_code=Q_PARAMETER_CODE, limit=500):
    """Period of record per station, WITHOUT pulling any records.

    Answers "which years could this basin ever be validated in" in one 0.3 s
    query, where probing candidate years costs ~45 s each on a cold cache.

    Read it as an OUTER ENVELOPE, never as truth: the Naches outlet gauge
    reports a span of 1899-1990 and yet has nothing at all in 1985. A span rules
    a year OUT reliably; only a counts query rules one IN.
    """
    return _ogc_items("time-series-metadata",
                      {"bbox": bbox, "parameter_code": parameter_code,
                       "limit": int(limit),
                       "properties": "monitoring_location_id,begin,end"})


def coverage_by_station(bbox, start_date, end_date, sites_data=None,
                        parameter_code=Q_PARAMETER_CODE, max_probe=25):
    """Which stations have records in this window — WITHOUT a bbox+datetime query.

    That query shape is the problem. Scoped to a bbox and a date range the
    `daily` collection is wildly variable: the same 2020 request took 9.5 s one
    hour and 75.8 s the next, and the collection hard-cancels at ~60 s with a
    400. Retrying does not help, because the cost is the shape, not luck. When
    it failed, the validator recorded "no in-domain gauge had 2020 daily
    records" for a basin whose gauge has reported every year since 1979 — a
    silent wrong answer of exactly the kind this module exists to avoid.

    So the window is never asked of the whole bbox. Instead:
      1. period-of-record spans for the bbox (one query, ~0.3 s) rule out every
         station that cannot possibly have data — usually most of them;
      2. each survivor is asked directly (station-scoped, ~0.4 s), which is the
         cheap shape.
    Spans are an outer envelope, so step 1 only ever discards stations that are
    certainly empty, never one that might report.
    """
    spans = _parse_spans(fetch_record_spans(bbox, parameter_code=parameter_code),
                         sites_data or {"features": []})
    y0, y1 = int(str(start_date)[:4]), int(str(end_date)[:4])
    cands = [s for s in spans["stations"]
             if s.get("first_year") is not None
             and s["first_year"] <= y1 and s["last_year"] >= y0]

    feats, probed = [], 0
    for s in cands[:max_probe]:
        try:
            raw = fetch_station_series(s["id"], start_date, end_date,
                                       parameter_code=parameter_code, limit=5000)
        except Exception:
            continue
        probed += 1
        feats.extend(raw.get("features") or [])
    return {"type": "FeatureCollection", "features": feats,
            "truncated": len(cands) > max_probe, "n_pages": probed,
            "n_candidates": len(cands)}


def fetch_station_series(site_id, start_date, end_date,
                         parameter_code=Q_PARAMETER_CODE, limit=500):
    """One station's daily values WITH timestamps.

    Per station, not per bbox. Asking the bbox query to carry `time` pushes it
    past the collection's ~60 s server budget even for a single year -- 60.6 s
    and a 400 against 9.5 s without -- because the timestamp defeats whatever
    aggregation makes the wide query cheap. Scoped to one station the same year
    returns in 0.4 s. Coverage stays a bbox query; only the VALUES are fetched
    per station, and there are rarely more than a handful of qualifying ones.
    """
    return _ogc_items("daily", {
        "monitoring_location_id": site_id,
        "parameter_code": parameter_code,
        "datetime": f"{start_date}/{end_date}",
        "limit": int(limit),
        "properties": "monitoring_location_id,time,value"})


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


def _site_meta(sites_data):
    """{station id: name/lat/lon/drainage_area_km2} from monitoring-locations.

    Coordinates and catchment area travel with every station because a caller
    needs both to judge whether a station is USEFUL, not merely present: where
    to place it on a map, and whether its catchment resembles the domain being
    modelled at all.
    """
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
            # LAND-SURFACE ALTITUDE, in metres. Needed twice: to turn a
            # groundwater ELEVATION series into a depth below ground, and to
            # give a well an elevation at all so it can be matched against a
            # column's. `altitude_accuracy` travels with it because it is
            # usually 10 ft, "interpolated from topographic map" — fine for a
            # 150 m band, not fine to treat as survey-grade.
            **_altitude(props),
        }
    return meta


def _altitude(props) -> Dict[str, Any]:
    """{altitude_m, altitude_accuracy_m} from a monitoring-location, in metres."""
    out: Dict[str, Any] = {}
    for src, dst in (("altitude", "altitude_m"),
                     ("altitude_accuracy", "altitude_accuracy_m")):
        try:
            v = float(props.get(src))
        except (TypeError, ValueError):
            continue
        out[dst] = round(v * FT_TO_M, 2)
    return out


def _parse_spans(md_data, sites_data):
    """time-series-metadata -> one first/last year per station.

    A station can carry several series (different statistics or sublocations);
    they are merged into the outer envelope, since any of them existing means
    the year is not ruled out.
    """
    meta = _site_meta(sites_data)
    spans = {}
    for f in (md_data or {}).get("features", []) or []:
        props = f.get("properties", {}) or {}
        sid = props.get("monitoring_location_id")
        b, e = str(props.get("begin") or "")[:4], str(props.get("end") or "")[:4]
        if not (sid and b.isdigit() and e.isdigit()):
            continue
        lo, hi = int(b), int(e)
        cur = spans.get(sid)
        spans[sid] = (min(cur[0], lo), max(cur[1], hi)) if cur else (lo, hi)

    stations = [{"id": sid, "first_year": lo, "last_year": hi, **meta.get(sid, {})}
                for sid, (lo, hi) in spans.items()]
    # Widest catchment first: the station that could validate the whole domain
    # is the one a caller is looking for, and it is rarely the first returned.
    stations.sort(key=lambda s: (-(s.get("drainage_area_km2") or 0), s["id"]))
    return {"n_stations": len(stations), "stations": stations,
            "caveat": ("spans are an OUTER ENVELOPE — a year inside a span may "
                       "still hold no records; confirm with a dated query")}


# ── the water table AS A TIME SERIES ─────────────────────────────────────────
# A RECORDER WELL, NOT A WELL SOMEBODY VISITED (2026-08-11, user's decision).
# Field measurements answer "how deep was the water on the day someone came by";
# for a run they are one date per well, and a single static level is a quantity
# Fan already supplies everywhere at once. What USGS has that Fan does not is
# VARIATION THROUGH TIME, and that lives in the daily/continuous collections.
#
# Three parameters carry it, and they are not the same quantity:
#   72019  depth to water, ft BELOW LAND SURFACE  -> comparable as it stands
#   62610  groundwater elevation, ft above NGVD29 -> needs the site altitude
#   62611  groundwater elevation, ft above NAVD88 -> needs the site altitude
# Restricting to 72019 would be cleaner and would have cost 20 of 24 Brandywine
# sites, so the elevation series are converted and TAGGED rather than dropped:
# the subtraction inherits the altitude's error (usually 10 ft, interpolated
# from a topographic map), which offsets the LEVEL and leaves the VARIATION —
# the thing this is fetched for — untouched.
WTD_SERIES_PARAMETERS = {"72019": "depth",
                         "62610": "elevation",
                         "62611": "elevation"}

# Sites publish Min, Mean and Max as separate series, so a bbox reporting "239
# series" holds ~24 wells. Mean is present wherever Daily is, and Mean is what
# a daily model mean should be put beside; a Min series against ELM's daily
# mean would be biased by the diurnal range every day of the record.
DAILY_MEAN_STATISTIC = "00003"


def fetch_wtd_series(bbox, start_date=None, end_date=None, limit=DAILY_PAGE):
    """Daily-mean water-table series for every recorder well in the bbox.

    ONE QUERY PER PARAMETER, not one per well: the daily collection accepts the
    bbox, and every row carries its own geometry, so 23 Brandywine wells and
    their 6,835 daily values arrive in three calls and 13 s.
    """
    per, where, units, kinds, truncated = {}, {}, {}, {}, False
    for pc, kind in WTD_SERIES_PARAMETERS.items():
        params = {"bbox": bbox, "parameter_code": pc,
                  "statistic_id": DAILY_MEAN_STATISTIC, "limit": int(limit)}
        if start_date and end_date:
            params["datetime"] = f"{start_date}/{end_date}"
        data = _ogc_items("daily", params)
        truncated = truncated or bool(data.get("truncated"))
        for f in data.get("features") or []:
            props = f.get("properties") or {}
            sid, tm = props.get("monitoring_location_id"), props.get("time")
            try:
                val = float(props.get("value"))
            except (TypeError, ValueError):
                continue
            if not (sid and tm):
                continue
            # A well reporting BOTH a depth and an elevation series would be
            # counted twice and its two records interleaved. Depth wins, since
            # it needs no altitude and so carries no borrowed error.
            if sid in kinds and kinds[sid] != kind:
                if kinds[sid] == "depth":
                    continue
                per.pop(sid, None)
            kinds[sid] = kind
            units[sid] = props.get("unit_of_measure")
            c = (f.get("geometry") or {}).get("coordinates") or []
            if sid not in where and len(c) > 1 and c[0] is not None \
                    and c[1] is not None:
                where[sid] = {"lat": round(c[1], 5), "lon": round(c[0], 5)}
            per.setdefault(sid, {})[str(tm)[:10]] = val
    return {"per": per, "where": where, "kinds": kinds, "units": units,
            "truncated": truncated}


def _parse_wtd_series(raw, sites_data):
    """Recorder wells -> the `daily` shape every other observable already uses.

    Depths are metres below land surface either way. An elevation series is
    subtracted from the site's land-surface altitude and the well says so, so a
    reader can see which wells carry the topographic-map error in their level.
    """
    meta = _site_meta(sites_data)
    wells, no_altitude = [], []
    for sid, byday in (raw.get("per") or {}).items():
        kind = (raw.get("kinds") or {}).get(sid, "depth")
        site = meta.get(sid) or {}
        alt = site.get("altitude_m")
        if kind == "elevation" and alt is None:
            # Reported, never guessed: without the land surface there is no
            # depth to be had, and inventing one would put a whole well's
            # record at an arbitrary offset.
            no_altitude.append(sid)
            continue
        dates = sorted(byday)
        vals = []
        for d in dates:
            v_m = byday[d] * FT_TO_M
            vals.append(round(alt - v_m if kind == "elevation" else v_m, 3))
        xy = (raw.get("where") or {}).get(sid) or {
            k: v for k, v in site.items() if k in ("lat", "lon") and v is not None}
        w = {"id": sid, "n_obs": len(dates), "n_days": len(dates),
             "observation_kind": "series",
             "wtd_m": round(_median(vals), 2),
             "min_depth_m": round(min(vals), 2),
             "max_depth_m": round(max(vals), 2),
             "daily": {"dates": dates, "values": vals},
             "parameter_kind": kind,
             **({"name": site["name"]} if site.get("name") else {}),
             **({"elevation_m": alt} if alt is not None else {}), **xy}
        if kind == "elevation":
            w["depth_note"] = (
                "depth = site altitude - groundwater elevation; the LEVEL "
                "carries the altitude's error"
                + (f" (+/- {site['altitude_accuracy_m']} m, USGS)"
                   if site.get("altitude_accuracy_m") is not None else "")
                + ", the VARIATION does not")
        wells.append(w)
    wells.sort(key=lambda w: -w["n_days"])

    out = {"n_wells_with_records": len(wells), "n_series_wells": len(wells),
           "n_located": sum(1 for w in wells if w.get("lat") is not None),
           "observation_kind": "series",
           "statistic": "daily mean (USGS statistic 00003)",
           "wells": wells}
    warnings = []
    if no_altitude:
        out["n_without_altitude"] = len(no_altitude)
        warnings.append(f"{len(no_altitude)} well(s) report groundwater "
                        f"ELEVATION but their site has no altitude, so no "
                        f"depth below ground could be formed")
    if raw.get("truncated"):
        out["truncated"] = True
        warnings.append("series list is INCOMPLETE — the daily query hit its "
                        "page cap")
    if not wells:
        # THE FINDING, not an error. A basin with no recorder well cannot have
        # its water table evaluated through time, and that is worth saying
        # plainly: it decides whether wtd is an observable for this study at
        # all. Measured 2026-08-11: of the 10 chain-eval basins, 9 have some
        # groundwater series and 6 have one covering their simulation year.
        out["note"] = ("no well in this bbox records a daily water-table "
                       "series for this period. Field measurements may exist "
                       "— one visit per well — but a single static level is "
                       "what Fan already supplies everywhere, so there is "
                       "nothing here to compare a run against through time.")
    if warnings:
        out["warning"] = "; ".join(warnings)
    return out


CFS_TO_M3S = 0.0283168


def attach_daily_series(coverage, start_date, end_date,
                        parameter_code=Q_PARAMETER_CODE):
    """Add each available gauge's daily series as SPECIFIC DISCHARGE (mm/day).

    Flow divided by catchment area. A raw cfs hydrograph cannot be compared with
    a modelled column at all — the column has no catchment and no routing — but
    depth per unit area is the same quantity in both, so this is the conversion
    that makes the comparison meaningful rather than merely plottable.

    A gauge with no drainage area gets no series rather than a wrong one — and
    SAYS SO, because a gauge that never appears is indistinguishable from a
    basin that has none. Drainage area comes only from the site catalogue, so
    this is the failure mode a truncated catalogue produces.
    """
    for g in coverage.get("available", []) or []:
        da_km2 = g.get("drainage_area_km2")
        if not da_km2:
            g["no_series_reason"] = ("no drainage area, so flow cannot be "
                                     "converted to specific discharge")
            continue
        try:
            raw = fetch_station_series(g["id"], start_date, end_date,
                                       parameter_code=parameter_code)
        except Exception as e:
            g["series_error"] = str(e)[:120]
            continue
        series = {}
        for f in (raw or {}).get("features", []) or []:
            props = f.get("properties", {}) or {}
            tm, val = props.get("time"), props.get("value")
            if not tm or val is None:
                continue
            try:
                series[str(tm)[:10]] = float(val)
            except (TypeError, ValueError):
                continue
        if not series:
            continue
        to_mm_day = CFS_TO_M3S * 86400 / (da_km2 * 1e6) * 1000
        g["mm_day"] = {d: round(v * to_mm_day, 4) for d, v in sorted(series.items())}
        g["mean_cfs"] = round(sum(series.values()) / len(series), 1)
    return coverage


def _parse_coverage(daily_data, sites_data, min_days=300):
    """(daily FeatureCollection, monitoring-locations FeatureCollection)
    -> {available[], n_available, n_sites, n_without_records}.

    Network-free, like the other parsers. `available` carries what a caller
    needs to decide whether a station is USEFUL, not merely present:
    coordinates (to place a column or draw a map) and drainage area in km2
    (to judge whether the station represents the modelled domain at all).

    MIN_DAYS TAGS, IT DOES NOT DROP (2026-08-11). Every gauge with at least one
    day in the window is returned, carrying `n_days` and `meets_min_days`, and
    the caller decides. A fixed day count cannot see how much of the basin a
    gauge represents, and on Naches 1979 it made exactly the wrong call: it kept
    USGS-12488500 (204 km2, 7% of the basin, 365 days) and discarded
    USGS-12494000 (2,437 km2, 85% of the basin) for holding 272 days rather than
    300 — a complete, gap-free January-to-September record, which is the whole
    snowmelt season a snow-driven basin is judged on. That gauge was simply
    retired on 29 September 1979. Reception reported "1 gauge with records" and
    said nothing about the other.

    This also restores the rule the rest of the pipeline already follows:
    _tag_in_basin TAGS stations and never drops them, which is why a gauge just
    outside the divide still reaches the caller. The day count was the last
    place upstream that silently decided.

    Widest catchment first, for the same reason _parse_spans sorts that way: the
    gauge that could speak for the whole domain is the one a caller is looking
    for, and it is rarely the one with the most days.
    """
    counts, where = {}, {}
    for f in (daily_data or {}).get("features", []) or []:
        props = f.get("properties", {}) or {}
        sid = props.get("monitoring_location_id")
        if not (sid and props.get("value") is not None):
            continue
        counts[sid] = counts.get(sid, 0) + 1
        # COORDINATES FROM THE RECORD ITSELF. The daily collection carries the
        # point each value was measured at — its bbox filter could not work
        # otherwise — so a gauge stays placeable even when the site catalogue
        # misses it. Drainage area is the one field with no other source, and
        # is why the catalogue is still read at all.
        c = (f.get("geometry") or {}).get("coordinates") or []
        if sid not in where and len(c) > 1 and c[0] is not None \
                and c[1] is not None:
            where[sid] = {"lat": round(c[1], 5), "lon": round(c[0], 5)}

    meta = _site_meta(sites_data)

    available, n_meeting = [], 0
    for sid, n in counts.items():
        m = dict(meta.get(sid) or {})
        for k, v in (where.get(sid) or {}).items():
            if m.get(k) is None:
                m[k] = v
        meets = n >= int(min_days)
        n_meeting += 1 if meets else 0
        available.append({"id": sid, "n_days": n, "meets_min_days": meets, **m})
    available.sort(key=lambda s: (-(s.get("drainage_area_km2") or 0),
                                  -s["n_days"], s["id"]))

    n_sites = len((sites_data or {}).get("features", []) or [])
    out = {"n_sites": n_sites, "n_available": len(available),
           "n_meeting_min_days": n_meeting,
           "n_without_records": max(n_sites - len(available), 0),
           "min_days": int(min_days), "available": available,
           "min_days_note": ("min_days TAGS, it does not drop — every gauge "
                             "with records is here; read meets_min_days and "
                             "drainage_area_km2 together, never n_days alone")}
    warnings = []
    n_unlocated = sum(1 for s in available if s.get("lat") is None)
    if n_unlocated:
        out["n_unlocated"] = n_unlocated
        warnings.append(f"{n_unlocated} of {len(available)} gauges have no "
                        f"coordinates and cannot be placed or tagged")
    n_no_area = sum(1 for s in available if not s.get("drainage_area_km2"))
    if n_no_area:
        # Named because attach_daily_series gives these no series at all: a
        # gauge can only become specific discharge if its catchment area is
        # known, so this is the difference between a gauge that disagrees and
        # a gauge that never appears.
        out["n_without_drainage_area"] = n_no_area
        warnings.append(f"{n_no_area} of {len(available)} gauges have no "
                        f"drainage area, so they get no series")
    if (daily_data or {}).get("truncated"):
        # An undercount looks exactly like a real finding, so it must announce
        # itself rather than be inferred from a suspiciously round number.
        out["truncated"] = True
        warnings.append("record counts are INCOMPLETE — the daily query hit "
                        "its page cap; treat n_days as a lower bound")
    if warnings:
        # Joined, not assigned: these used to overwrite one another.
        out["warning"] = "; ".join(warnings)
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
