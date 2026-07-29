#!/usr/bin/env python3
"""
SNOTEL MCP Server — snow water equivalent (SWE) from the NRCS SNOTEL network.

The validation source for snow-driven runoff/infiltration partitioning: SNOTEL
stations report daily SWE in mountain basins. Source: NRCS AWDB REST API
(https://wcc.sc.egov.usda.gov/awdbRestApi), free, no key.

Tools:
    get_swe(bbox[, start_date, end_date, with_values])
        -> no dates: stations + elevation; dates: which REPORTED, with peak SWE
           and its date; with_values: the daily series, columnar
           ({units, dates, values}) — the same shape the model uses.

Pure parsers (_filter_stations / _parse_swe / _parse_swe_bulk) are network-free
and unit-tested.
"""
import json

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from mcp.server.fastmcp import FastMCP

AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
IN_TO_MM = 25.4
_SOURCE = "NRCS AWDB SNOTEL"
_TIMEOUT = 60

mcp = FastMCP("snotel")


def _get(endpoint, params):
    r = requests.get(f"{AWDB}/{endpoint}", params=params,
                     timeout=_TIMEOUT, verify=False)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# PURE PARSERS (network-free, unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_stations(rows, bbox):
    """Keep active SNOTEL (network SNTL) stations inside the bbox."""
    mn_lon, mn_lat, mx_lon, mx_lat = bbox
    out = []
    for s in rows:
        if s.get("networkCode") != "SNTL":
            continue
        lat, lon = s.get("latitude"), s.get("longitude")
        if lat is None or lon is None:
            continue
        if mn_lat <= lat <= mx_lat and mn_lon <= lon <= mx_lon:
            out.append({"triplet": s.get("stationTriplet"), "name": s.get("name"),
                        "lat": lat, "lon": lon,
                        "elevation_ft": s.get("elevation"), "huc": s.get("huc")})
    return out


def _parse_swe_bulk(data, with_values=False):
    """AWDB /data response for MANY stations -> {triplet: summary}.

    The single-station parser below flattens every station into one list, which
    silently merges six mountains into one bogus hydrograph as soon as a bulk
    query is used. Kept separate rather than generalised so the old shape stays
    obvious.
    """
    out = {}
    for st in data or []:
        trip = st.get("stationTriplet")
        obs = []
        for de in st.get("data", []):
            for v in de.get("values", []):
                val = v.get("value")
                if val is None:
                    continue
                obs.append({"date": v.get("date"),
                            "swe_mm": round(val * IN_TO_MM, 1)})
        if not trip:
            continue
        obs.sort(key=lambda o: o["date"] or "")
        swe = [o["swe_mm"] for o in obs]
        peak = max(swe) if swe else None
        summary = {"n_obs": len(obs),
                   "first": obs[0]["date"] if obs else None,
                   "last": obs[-1]["date"] if obs else None,
                   "peak_swe_mm": round(peak, 1) if peak is not None else None,
                   "peak_date": (obs[swe.index(peak)]["date"]
                                 if peak is not None else None)}
        if with_values:
            # COLUMNAR, and deliberately the same shape the model's daily
            # series uses ({units, values, dates}): row-wise
            # [{"date":…,"swe_mm":…}] is 2x the characters for the same
            # numbers (15.0 KB vs 7.7 KB per station-year) and has to be
            # unpacked and re-packed to build a frame from it.
            summary["daily"] = {
                "units":  "mm",
                "dates":  [o["date"] for o in obs],
                "values": [o["swe_mm"] for o in obs],
            }
        out[trip] = summary
    return out


def _parse_swe(data):
    """AWDB /data response -> (sorted daily SWE obs, summary)."""
    obs = []
    for st in data or []:
        for de in st.get("data", []):
            for v in de.get("values", []):
                val = v.get("value")
                if val is None:
                    continue
                obs.append({"date": v.get("date"), "swe_in": val,
                            "swe_mm": round(val * IN_TO_MM, 1)})
    obs.sort(key=lambda o: o["date"] or "")
    swe = [o["swe_mm"] for o in obs]
    peak_i = swe.index(max(swe)) if swe else None
    summary = {"n_obs": len(obs),
               "first": obs[0]["date"] if obs else None,
               "last": obs[-1]["date"] if obs else None,
               "peak_swe_mm": round(max(swe), 1) if swe else None,
               "peak_date": obs[peak_i]["date"] if peak_i is not None else None}
    return obs, summary


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_swe(bbox: str, start_date: str = "", end_date: str = "",
            with_values: bool = False) -> str:
    """SNOTEL snow-water-equivalent stations in a bbox — and their records.

    Same three shapes as the usgs_water tools:

      no dates                 -> the stations that exist, with elevation.
      dates                    -> which of them REPORTED in that window, with
                                  peak SWE and its date (the number a snow
                                  comparison actually turns on).
      dates + with_values=True -> plus the daily series per station.

    All stations are fetched in ONE bulk AWDB call rather than one call each:
    six stations cost 0.5 s together against ~7 round trips before.

    bbox='min_lon,min_lat,max_lon,max_lat'; dates 'YYYY-MM-DD'. SWE in mm.
    """
    try:
        mn_lon, mn_lat, mx_lon, mx_lat = [float(x) for x in bbox.split(",")]
    except Exception:
        return json.dumps({"error": "bbox must be 'min_lon,min_lat,max_lon,max_lat'"})
    try:
        rows = _get("stations", {"networkCds": "SNTL"})
    except Exception as e:
        return json.dumps({"error": f"AWDB stations query failed: {e}"})
    sts = _filter_stations(rows, (mn_lon, mn_lat, mx_lon, mx_lat))
    for s in sts:
        ft = s.pop("elevation_ft", None)
        s["elevation_m"] = round(ft * 0.3048, 1) if ft is not None else None

    if not (start_date and end_date) or not sts:
        return json.dumps({"bbox": bbox, "mode": "stations",
                           "n_stations": len(sts), "stations": sts,
                           "source": _SOURCE})
    try:
        data = _get("data", {"stationTriplets": ",".join(s["triplet"] for s in sts),
                             "elements": "WTEQ", "duration": "DAILY",
                             "beginDate": start_date, "endDate": end_date})
    except Exception as e:
        return json.dumps({"error": f"AWDB data query failed: {e}"})
    per = _parse_swe_bulk(data, with_values=with_values)

    reporting = []
    for s in sts:
        summ = per.get(s["triplet"])
        if summ and summ["n_obs"]:
            reporting.append({**s, **summ})
    reporting.sort(key=lambda s: -(s.get("peak_swe_mm") or 0))
    return json.dumps({"bbox": bbox, "period": f"{start_date}/{end_date}",
                       "mode": "with_values" if with_values else "coverage",
                       "n_stations": len(sts),
                       "n_reporting": len(reporting),
                       "stations": reporting, "source": _SOURCE})


if __name__ == "__main__":
    mcp.run(transport="stdio")
