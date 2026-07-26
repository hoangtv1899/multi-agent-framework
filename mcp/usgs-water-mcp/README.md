# USGS Water MCP

Serves **USGS site discovery** and **observed groundwater levels** for the
spatial-ELM data pipeline. It turns a bounding box into USGS monitoring
locations (stream gauges + wells) and returns raw depth-to-water field
measurements at wells. It complements the `fan_wtd` server: `fan_wtd` is a
*modeled equilibrium* water-table prior, this server is the *raw USGS well
measurements* (`get_water_table_depth`), not a substitute for them.

Scope: **site discovery + field-measurement WTD** only. Streamflow timeseries
and OGC-daily specific-yield comparisons are fetched directly by the gatherer
and validators (not through this server); the only streamflow tool here is the
legacy `fetch_usgs_data` passthrough.

## Tools
- `get_monitoring_locations(bbox, agency_code="", site_type_code="", limit=50)`
  — near-raw OGC monitoring-locations FeatureCollection (`features[].properties.
  monitoring_location_number` / `_name` / `drainage_area`, plus `numberReturned`).
  `site_type_code="ST"` for stream gauges, `"GW"` for wells.
- `get_groundwater_sites(bbox, limit=25)` — wells only (`site_type_code=GW`) as
  flat dicts: `{n_sites, sites:[{id, lat, lon, name, aquifer_code, altitude}]}`.
  Each `id` (e.g. `USGS-382553106392801`) is usable directly as a
  `monitoring_location_id`.
- `get_water_table_depth(monitoring_location_id, limit=100)` — observed
  depth-to-water (feet -> metres below land surface) from field measurements
  (parameter **72019**): `{observations[], summary{n_obs, min/mean/median/max/
  latest_depth_m}}`.
- `fetch_usgs_data(sites, parameter_codes="00060,00065,00010", period="P1D")`
  — legacy passthrough returning raw NWIS instantaneous-values WaterML-JSON.

`bbox` is always the OGC string `"min_lon,min_lat,max_lon,max_lat"`.

## Endpoints (free, no key)
- OGC API — `https://api.waterdata.usgs.gov/ogcapi/v0`
  - `collections/monitoring-locations/items` — site discovery.
  - `collections/field-measurements/items?parameter_code=72019` — groundwater
    levels. **There is no `groundwater-levels` collection**; GW levels live in
    `field-measurements` filtered by parameter 72019 (depth to water, ft below
    land surface; `value` is a **string in feet**).
- NWIS IV — `https://waterservices.usgs.gov/nwis/iv/?format=json` — legacy
  `fetch_usgs_data`. The old `waterdata.usgs.gov/nwis/iv/` path now returns 404
  (verified from Compy 2026-07-23), so this uses `waterservices.usgs.gov`.

## Layout
- `main.py` — FastMCP stdio server (`FastMCP("usgs_water")`), thin tool wrappers.
- `groundwater_api.py` — network layer (httpx) plus the pure, network-free
  parsers `_parse_sites` / `_parse_wtd` (feet -> metres). Unit-tested by
  `tests/test_mcp_tools.py::TestGroundwaterParsers`.

## Note
This is a **Compy (PNNL) rebuild** of the original `usgs-water-mcp`, which lived
only on NERSC (a nested repo that was unavailable during the port, so
`mcp/usgs-water-mcp/` arrived empty on Compy). The interface was reconstructed
from the framework's call sites (`mcp_gatherer`, `tools/validate_run.py`,
`tools/mcp_conus_sweep.py`, `tools/scout_watersheds.py`, `reception_llm`) and
verified against the live USGS OGC/NWIS endpoints. Register it in
`mcp_config.json` under key `usgs_water`.

## Run
```bash
python3 main.py            # stdio MCP server
```
Requires `mcp` and `httpx` (see `requirements.txt`).
