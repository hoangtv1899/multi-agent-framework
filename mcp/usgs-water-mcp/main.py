#!/usr/bin/env python3
"""
USGS Water MCP Server — site discovery + observed groundwater levels (and a
legacy instantaneous-values passthrough) for the spatial-ELM data pipeline.

Site DISCOVERY layer of the workflow: it turns a bbox into USGS monitoring
locations (stream gauges and wells) and serves observed depth-to-water at wells
from the USGS OGC API. It complements fan_wtd (a modeled equilibrium WTD prior)
with the raw field measurements. Streamflow timeseries and OGC-daily yields are
fetched directly by the gatherer/validators, NOT here — this server only does
site discovery, field-measurement WTD, and the legacy fetch_usgs_data passthrough.

Data sources (free, no key):
    OGC API   https://api.waterdata.usgs.gov/ogcapi/v0
              (monitoring-locations, field-measurements; GW level = param 72019)
    NWIS IV   https://waterservices.usgs.gov/nwis/iv/   (legacy fetch_usgs_data)

Tools:
    get_monitoring_locations(bbox, agency_code="", site_type_code="", limit=50)
        -> near-raw OGC monitoring-locations FeatureCollection
           (features[].properties.monitoring_location_number / _name /
           drainage_area, plus numberReturned)
    get_groundwater_sites(bbox, limit=25)
        -> {n_sites, sites[]} — wells (site_type_code=GW) as flat dicts
           (id, lat, lon, name, aquifer_code, altitude)
    get_water_table_depth(monitoring_location_id, limit=100)
        -> {observations[], summary{n_obs,min/mean/median/max/latest_depth_m}}
           observed depth-to-water (ft->m) from field-measurements param 72019
    fetch_usgs_data(sites, parameter_codes="00060,00065,00010", period="P1D")
        -> raw NWIS instantaneous-values WaterML-JSON (legacy passthrough)

bbox is always the OGC string 'min_lon,min_lat,max_lon,max_lat'. Real logic and
the pure parsers live in groundwater_api.py (network-free, unit-tested). Tools
never raise: upstream failures come back as {"error": "..."} JSON.

Compy rebuild of the NERSC-only original (that repo was unavailable during the
port); reconstructed from the framework's call sites and verified against the
live USGS OGC/NWIS endpoints on 2026-07-23.
"""
import json

import groundwater_api as gw

from mcp.server.fastmcp import FastMCP

_SOURCE = "USGS Water Data (api.waterdata.usgs.gov OGC API + NWIS)"

mcp = FastMCP("usgs_water")


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_monitoring_locations(bbox: str, agency_code: str = "",
                             site_type_code: str = "", limit: int = 50) -> str:
    """USGS monitoring locations in a bbox (near-raw OGC FeatureCollection).
    bbox='min_lon,min_lat,max_lon,max_lat'; site_type_code e.g. 'ST' (stream
    gauge) or 'GW' (well); agency_code e.g. 'USGS'. Read
    features[].properties.monitoring_location_number / _name / drainage_area."""
    try:
        data = gw.fetch_monitoring_locations(bbox, agency_code, site_type_code, limit)
    except Exception as e:
        return json.dumps({"error": f"USGS monitoring-locations query failed: {e}"})
    return json.dumps({**data, "source": _SOURCE})


@mcp.tool()
def get_groundwater_sites(bbox: str, limit: int = 25) -> str:
    """USGS groundwater wells (site_type_code=GW) in a bbox.
    Returns {n_sites, sites[]} where each site has id (usable as
    monitoring_location_id), lat, lon, name, aquifer_code, altitude.
    bbox='min_lon,min_lat,max_lon,max_lat'."""
    try:
        data = gw.fetch_monitoring_locations(bbox, site_type_code="GW", limit=limit)
    except Exception as e:
        return json.dumps({"error": f"USGS groundwater sites query failed: {e}"})
    sites = gw._parse_sites(data)
    return json.dumps({"bbox": bbox, "n_sites": len(sites), "sites": sites,
                       "source": _SOURCE})


@mcp.tool()
def get_water_table_depth(monitoring_location_id: str, limit: int = 100) -> str:
    """Observed depth-to-water (metres below land surface) at a USGS well from
    field measurements (parameter 72019). monitoring_location_id e.g.
    'USGS-465728120401801'. Returns {observations[], summary{n_obs, min/mean/
    median/max/latest_depth_m}} — use it to confirm a site has records."""
    try:
        data = gw.fetch_field_measurements(monitoring_location_id, limit=limit)
    except Exception as e:
        return json.dumps({"error": f"USGS field-measurements query failed: {e}"})
    obs, summary = gw._parse_wtd(data)
    return json.dumps({"monitoring_location_id": monitoring_location_id,
                       "parameter_code": gw.WTD_PARAMETER_CODE,
                       "summary": summary, "observations": obs,
                       "source": _SOURCE})


@mcp.tool()
def fetch_usgs_data(sites: str, parameter_codes: str = "00060,00065,00010",
                    period: str = "P1D") -> str:
    """Legacy passthrough: raw NWIS instantaneous-values WaterML-JSON.
    sites='01646500' (comma-sep ok); parameter_codes e.g. '00060,00065,00010'
    (discharge, gage height, water temp); period ISO-8601 e.g. 'P1D'.
    Returns the NWIS JSON verbatim (value.timeSeries[...])."""
    try:
        data = gw.fetch_nwis_iv(sites, parameter_codes, period)
    except Exception as e:
        return json.dumps({"error": f"USGS NWIS IV query failed: {e}"})
    return json.dumps(data)


if __name__ == "__main__":
    mcp.run(transport="stdio")
