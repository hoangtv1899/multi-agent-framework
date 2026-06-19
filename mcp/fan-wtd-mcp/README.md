# Fan WTD MCP

Serves **equilibrium water-table depth** from Fan, Li & Miguez-Macho (2013),
*"Global Patterns of Groundwater Table Depth"*, Science 339:940–943.

This is a **static-file** server: it reads a Fan 2013 NetCDF tile from disk and
answers point / bbox queries. The dataset is a *modeled, observationally-
constrained equilibrium* WTD — treat it as a spatial **prior / benchmark
surface**, complementary to the raw USGS well measurements from the
`usgs_water` MCP (`get_water_table_depth`), not a substitute for them.

## Tools
- `data_status()` — is the dataset present? grid info + this guidance.
- `get_fan_wtd(lat, lon)` — equilibrium WTD (m below surface) at the nearest cell.
- `sample_fan_wtd(min_lon, min_lat, max_lon, max_lat, n)` — WTD grid over a bbox + stats.

## Provisioning the dataset

The server starts fine **without** the data — every tool then returns a clear
"not found" status. To enable real queries, obtain a Fan 2013 NetCDF tile and
point the server at it.

1. **Get the file.** The Fan 2013 equilibrium WTD is distributed as regional
   NetCDF tiles (North America, etc.). Canonical source is the Miguez-Macho
   group's THREDDS server (`GLOBALWTDFTP` collection); the dataset is also
   mirrored by several groundwater-data archives. Download the North America
   (or global) equilibrium WTD NetCDF.

2. **Place / point at it.** Either drop it at the default path
   `multi-agent/data/fan_wtd/fan2013_wtd.nc`, or set an env var:

   ```bash
   export FAN_WTD_NC=/path/to/your/fan2013_NorthAmerica_wtd.nc
   ```

   `MCPClient` forwards the current environment to the server subprocess, so
   exporting `FAN_WTD_NC` in your shell is enough.

3. **Verify.**
   ```bash
   python download_fan_wtd.py --check          # confirms the file opens + grid info
   ```

The server auto-detects the WTD variable (`WTD` / `wtd` / `water_table_depth` …)
and the lat/lon coordinate names, so most Fan distributions work unmodified.

## Conventions (handled automatically)
- The server reports **`depth_to_water_m` = positive metres below land surface**.
  Tiles differ in internal sign (the provisioned `NAMERICA_WTD_*` tiles store
  WTD as *negative* below surface); the server auto-detects this and always
  returns a positive depth. The raw file value is also returned as `wtd_m`.
- A `time` dimension is reduced automatically (annual tile squeezed; monthly
  tile averaged). The annual-mean tile is preferred by auto-discovery.
- Out-of-domain cells (ocean / outside the tile) are blanked via the `mask`
  variable and returned as `null`.
- Provisioned `NAMERICA` tiles use −180..180 longitude (Naches lon ≈ −120.7).
