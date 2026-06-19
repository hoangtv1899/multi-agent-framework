#!/usr/bin/env python3
"""
SNOTEL MCP Server — snow water equivalent (SWE) from the NRCS SNOTEL network.

The validation source for snow-driven runoff/infiltration partitioning: SNOTEL
stations report daily SWE in mountain basins. Source: NRCS AWDB REST API
(https://wcc.sc.egov.usda.gov/awdbRestApi), free, no key.

Tools:
    get_snotel_stations(bbox)
        -> SNOTEL stations inside a bbox (triplet, name, lat/lon, elevation)
    get_snotel_swe(station_triplet, start_date, end_date)
        -> daily SWE series (in + mm) and peak-SWE summary

Pure parsers (_filter_stations / _parse_swe) are network-free and unit-tested.
"""
import json

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from mcp.server.fastmcp import FastMCP

AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
IN_TO_MM = 25.4
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
def get_snotel_stations(min_lon: float, min_lat: float,
                        max_lon: float, max_lat: float) -> str:
    """SNOTEL stations (snow water equivalent) inside a bbox — snow validation points."""
    try:
        rows = _get("stations", {"networkCds": "SNTL"})
    except Exception as e:
        return json.dumps({"error": f"AWDB stations query failed: {e}"})
    sts = _filter_stations(rows, (min_lon, min_lat, max_lon, max_lat))
    return json.dumps({"bbox": {"min_lon": min_lon, "min_lat": min_lat,
                                "max_lon": max_lon, "max_lat": max_lat},
                       "n_stations": len(sts), "stations": sts,
                       "source": "NRCS AWDB SNOTEL"})


@mcp.tool()
def get_snotel_swe(station_triplet: str, start_date: str, end_date: str) -> str:
    """Daily SWE (in + mm below) and peak-SWE summary for a SNOTEL station
    (e.g. '375:WA:SNTL'). Dates 'YYYY-MM-DD'."""
    try:
        data = _get("data", {"stationTriplets": station_triplet, "elements": "WTEQ",
                             "duration": "DAILY", "beginDate": start_date,
                             "endDate": end_date})
    except Exception as e:
        return json.dumps({"error": f"AWDB data query failed: {e}"})
    obs, summary = _parse_swe(data)
    return json.dumps({"station_triplet": station_triplet,
                       "element": "WTEQ (snow water equivalent)",
                       "summary": summary, "observations": obs,
                       "source": "NRCS AWDB SNOTEL"})


if __name__ == "__main__":
    mcp.run(transport="stdio")
