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
    get_streamflow(bbox[, start_date, end_date, with_values])
        -> no dates: period of record per gauge; dates: gauges that HAVE daily
           discharge in that window; with_values: their series in mm/day.
           Every gauge carries lat/lon + drainage_area_km2.
    get_water_table(bbox[, start_date, end_date, with_values])
        -> wells with depth-to-water records (parameter 72019), metres below
           land surface, same three shapes.
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
def get_streamflow(bbox: str, start_date: str = "", end_date: str = "",
                   with_values: bool = False, min_days: int = 300,
                   limit: int = 1000) -> str:
    """Stream gauges in a bbox — what exists, what has records, and the records.

    ONE tool, three uses, chosen by the arguments:

      no dates                 -> period of record per gauge (first_year,
                                  last_year). One cheap query. Use this to see
                                  which years the basin could EVER be validated
                                  in, instead of guessing a year and probing.
      dates                    -> gauges with >= min_days of daily discharge in
                                  that window, the truth a span cannot give.
      dates + with_values=True -> the same, plus each gauge's daily series as
                                  specific discharge in mm/day (flow divided by
                                  catchment area, so it compares directly with a
                                  modelled column). For validators, NOT for
                                  conversational use — it is large.

    Every gauge carries lat/lon and drainage_area_km2, so a caller can judge
    whether its catchment resembles the domain being modelled: a bbox may hold
    93 gauges of which one reports, draining 7% of the basin.

    MIN_DAYS TAGS, IT DOES NOT DROP. Every gauge with records in the window is
    returned, carrying `n_days` and `meets_min_days`; read those beside
    drainage_area_km2 rather than filtering on days alone. Naches 1979 is the
    case that settled it — the 300-day threshold kept a 204 km2 tributary and
    discarded the 2,437 km2 main stem for holding 272 days.

    `limit` is a PAGE SIZE, not a ceiling: the catalogue is read to the end.

    bbox='min_lon,min_lat,max_lon,max_lat'; dates 'YYYY-MM-DD'.
    """
    try:
        sites = gw.fetch_monitoring_locations(bbox, site_type_code="ST",
                                              limit=int(limit))
        if not (start_date and end_date):
            out = gw._parse_spans(gw.fetch_record_spans(bbox), sites)
            out["mode"] = "record_spans"
        else:
            # Coverage is ALWAYS the cheap bbox query without timestamps;
            # values are then pulled per qualifying station. Asking the bbox
            # query for `time` exceeds the collection's server budget and 400s.
            daily = gw.coverage_by_station(bbox, start_date, end_date,
                                           sites_data=sites)
            out = gw._parse_coverage(daily, sites, min_days=min_days)
            out["mode"] = "with_values" if with_values else "coverage"
            out["period"] = f"{start_date}/{end_date}"
            if with_values:
                gw.attach_daily_series(out, start_date, end_date)
        out["source"] = _SOURCE
        return json.dumps(out)
    except Exception as e:
        return json.dumps({"error": str(e)[:200]})


@mcp.tool()
def get_water_table(bbox: str, start_date: str = "", end_date: str = "",
                    with_values: bool = False, min_obs: int = 1,
                    limit: int = 2000) -> str:
    """Wells in a bbox that RECORD the water table daily through this period.

    RECORDER WELLS ONLY (2026-08-11). A depth-to-water TIME SERIES is what this
    returns: daily means, metres below land surface, one entry per well under
    `daily` — the same shape the SWE and streamflow stations use.

    IT DOES NOT RETURN FIELD MEASUREMENTS, and that is the point. A field
    measurement is one visit; at Naches in 1979 all 56 in-basin wells had
    exactly one static reading between them, and a single static level is the
    quantity Fan already supplies at every point in the domain. What a well can
    say that Fan cannot is how the water table MOVED, and only a recorder well
    says it.

    A BASIN WITH NO RECORDER WELL RETURNS NO WELLS AND SAYS SO, in `note`. That
    is a finding about the basin, not a failure: of the 10 chain-eval basins, 9
    have some groundwater series and 6 have one covering their simulation year.
    Naches has none in any year.

    Three USGS parameters carry the series and they are not the same quantity:
    72019 is depth below land surface and is used as it stands; 62610/62611 are
    groundwater ELEVATION and are subtracted from the site's land-surface
    altitude, which puts that altitude's error (usually +/- 10 ft, interpolated
    from a topographic map) into the well's LEVEL but not into its VARIATION.
    Wells converted that way carry `depth_note`.

    `min_obs` filters on the number of DAYS in the returned series. `limit` is a
    page size, never a ceiling. `with_values` is accepted and ignored — a series
    without its values is not an observation of anything.
    """
    try:
        raw = gw.fetch_wtd_series(bbox, start_date or None, end_date or None)
        sites = gw.fetch_monitoring_locations(bbox, site_type_code="GW",
                                              limit=int(limit))
        out = gw._parse_wtd_series(raw, sites)
        if int(min_obs) > 1:
            kept = [w for w in out["wells"] if w["n_days"] >= int(min_obs)]
            if len(kept) != len(out["wells"]):
                out["n_below_min_obs"] = len(out["wells"]) - len(kept)
            out["wells"] = kept
            out["n_series_wells"] = out["n_wells_with_records"] = len(kept)
        out["period"] = (f"{start_date}/{end_date}"
                         if start_date and end_date else "all records")
        out["source"] = _SOURCE
        return json.dumps(out)
    except Exception as e:
        return json.dumps({"error": str(e)[:200]})


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
