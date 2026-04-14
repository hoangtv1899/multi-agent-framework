#!/usr/bin/env python3
"""
Weather MCP Server - Provides weather and climate data for PFLOTRAN workflows

Data sources:
    NWS         (no key): Current 7-day forecast         (US only)
    Open-Meteo  (no key): Historical precip + temperature (global, back to 1940)

Tools:
    get_forecast(lat, lon)
    get_historical_climate(lat, lon, start_year, end_year)
    get_climate_summary(lat, lon, start_year, end_year)
"""
import asyncio
import json
import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
NWS_BASE         = "https://api.weather.gov"
OPEN_METEO_BASE  = "https://archive-api.open-meteo.com/v1/archive"
NWS_HEADERS      = {"User-Agent": "PFLOTRAN-MCP/1.0 hoang.tran@pnnl.gov"}

server = Server("weather")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, headers: dict = None, params: dict = None) -> dict:
    """Simple GET with error handling."""
    try:
        r = requests.get(url, headers=headers or {},
                         params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _fetch_open_meteo(lat: float, lon: float,
                      start_year: int, end_year: int) -> dict:
    """
    Fetch daily precip + temp from Open-Meteo historical archive.
    Returns raw API response or error dict.
    """
    params = {
        "latitude":    lat,
        "longitude":   lon,
        "start_date":  f"{start_year}-01-01",
        "end_date":    f"{end_year}-12-31",
        "daily":       "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "timezone":    "auto",
        "wind_speed_unit": "ms"
    }
    return _get(OPEN_METEO_BASE, params=params)


def _monthly_averages(daily: dict) -> dict:
    """
    Aggregate Open-Meteo daily data into monthly averages.
    - Monthly precip: average of that month's total across all years
    - Annual precip:  sum of all 12 monthly averages
    - Temperature:    average across all days
    """
    from collections import defaultdict

    times  = daily.get("time", [])
    precip = daily.get("precipitation_sum", [])
    tmax   = daily.get("temperature_2m_max", [])
    tmin   = daily.get("temperature_2m_min", [])

    # Accumulate daily precip per (year, month) bucket
    # temp: all daily values per month across all years
    precip_buckets = defaultdict(float)   # (year, month) → monthly total
    monthly_tmax   = defaultdict(list)    # month → [daily values]
    monthly_tmin   = defaultdict(list)

    for i, date in enumerate(times):
        year  = date[:4]
        month = date[5:7]

        if i < len(precip) and precip[i] is not None:
            precip_buckets[(year, month)] += precip[i]
        if i < len(tmax) and tmax[i] is not None:
            monthly_tmax[month].append(tmax[i])
        if i < len(tmin) and tmin[i] is not None:
            monthly_tmin[month].append(tmin[i])

    # Average each calendar month across years
    # e.g. Jan avg = mean of [Jan2015, Jan2016, Jan2017, Jan2018, Jan2019, Jan2020]
    from collections import defaultdict as dd
    monthly_precip_means = defaultdict(list)
    for (year, month), total in precip_buckets.items():
        monthly_precip_means[month].append(total)

    result       = {}
    total_precip = 0.0
    all_tmax     = []
    all_tmin     = []

    for month in sorted(monthly_precip_means.keys()):
        # Mean of that calendar month across all years
        avg_precip = (sum(monthly_precip_means[month]) /
                      len(monthly_precip_means[month]))

        avg_tmax = (sum(monthly_tmax[month]) / len(monthly_tmax[month])
                    if monthly_tmax[month] else None)
        avg_tmin = (sum(monthly_tmin[month]) / len(monthly_tmin[month])
                    if monthly_tmin[month] else None)

        result[month] = {
            "precip_mm": round(avg_precip, 1),
            "tmax_c":    round(avg_tmax, 1) if avg_tmax is not None else None,
            "tmin_c":    round(avg_tmin, 1) if avg_tmin is not None else None,
        }

        # Annual = sum of all 12 monthly averages
        total_precip += avg_precip
        if avg_tmax is not None: all_tmax.append(avg_tmax)
        if avg_tmin is not None: all_tmin.append(avg_tmin)

    mean_tmax = round(sum(all_tmax) / len(all_tmax), 1) if all_tmax else None
    mean_tmin = round(sum(all_tmin) / len(all_tmin), 1) if all_tmin else None

    recharge_flux = round(
        total_precip / 1000.0 * 0.3 / (365.25 * 86400), 12
    )

    annual = {
        "precip_mm_per_year": round(total_precip, 1),
        "mean_tmax_c":        mean_tmax,
        "mean_tmin_c":        mean_tmin,
        "recharge_flux_ms":   recharge_flux
    }

    return {"monthly": result, "annual": annual}

# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_forecast",
            description=(
                "Get current 7-day weather forecast for a US location "
                "using National Weather Service API"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude"},
                    "lon": {"type": "number", "description": "Longitude"}
                },
                "required": ["lat", "lon"]
            }
        ),
        types.Tool(
            name="get_historical_climate",
            description=(
                "Get historical daily precipitation and temperature "
                "from Open-Meteo archive for any location and year range. "
                "Returns monthly averages. Data available back to 1940."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "lat":        {"type": "number"},
                    "lon":        {"type": "number"},
                    "start_year": {"type": "integer",
                                   "description": "Start year e.g. 2010"},
                    "end_year":   {"type": "integer",
                                   "description": "End year e.g. 2020"}
                },
                "required": ["lat", "lon", "start_year", "end_year"]
            }
        ),
        types.Tool(
            name="get_climate_summary",
            description=(
                "Get a single climate summary for PFLOTRAN experiment design. "
                "Returns mean annual precip, temperature range, and "
                "suggested recharge flux boundary condition in m/s."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "lat":        {"type": "number"},
                    "lon":        {"type": "number"},
                    "start_year": {"type": "integer"},
                    "end_year":   {"type": "integer"}
                },
                "required": ["lat", "lon", "start_year", "end_year"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str,
                    arguments: dict) -> list[types.TextContent]:

    # ── Tool 1: get_forecast ─────────────────────────────────────────────────
    if name == "get_forecast":
        lat = arguments["lat"]
        lon = arguments["lon"]

        # Step 1: get NWS grid point
        point = _get(f"{NWS_BASE}/points/{lat},{lon}", NWS_HEADERS)
        if "error" in point:
            return [types.TextContent(type="text",
                    text=json.dumps({"error": point["error"]}))]

        forecast_url = point.get("properties", {}).get("forecast", "")
        if not forecast_url:
            return [types.TextContent(type="text",
                    text=json.dumps({"error": "No forecast URL from NWS"}))]

        # Step 2: get forecast periods
        forecast = _get(forecast_url, NWS_HEADERS)
        if "error" in forecast:
            return [types.TextContent(type="text",
                    text=json.dumps({"error": forecast["error"]}))]

        periods = forecast.get("properties", {}).get("periods", [])
        result  = {
            "location": {"lat": lat, "lon": lon},
            "source":   "National Weather Service",
            "forecast": [
                {
                    "name":            p["name"],
                    "temperature_f":   p["temperature"],
                    "temperature_c":   round((p["temperature"] - 32) * 5/9, 1),
                    "precip_chance":   (p.get("probabilityOfPrecipitation",
                                             {}) or {}).get("value", 0),
                    "short_forecast":  p["shortForecast"],
                    "wind":            p["windSpeed"]
                }
                for p in periods[:7]
            ]
        }
        return [types.TextContent(type="text", text=json.dumps(result))]

    # ── Tool 2: get_historical_climate ───────────────────────────────────────
    elif name == "get_historical_climate":
        lat        = arguments["lat"]
        lon        = arguments["lon"]
        start_year = arguments["start_year"]
        end_year   = arguments["end_year"]

        raw = _fetch_open_meteo(lat, lon, start_year, end_year)
        if "error" in raw:
            return [types.TextContent(type="text",
                    text=json.dumps({"error": raw["error"]}))]

        daily    = raw.get("daily", {})
        averages = _monthly_averages(daily)

        result = {
            "location":     {"lat": lat, "lon": lon},
            "source":       "Open-Meteo Historical Archive",
            "period":       f"{start_year}-{end_year}",
            "record_count": len(daily.get("time", [])),
            "timezone":     raw.get("timezone", "auto"),
            **averages
        }
        return [types.TextContent(type="text", text=json.dumps(result))]

    # ── Tool 3: get_climate_summary ──────────────────────────────────────────
    elif name == "get_climate_summary":
        lat        = arguments["lat"]
        lon        = arguments["lon"]
        start_year = arguments["start_year"]
        end_year   = arguments["end_year"]

        raw = _fetch_open_meteo(lat, lon, start_year, end_year)
        if "error" in raw:
            return [types.TextContent(type="text",
                    text=json.dumps({"error": raw["error"]}))]

        daily    = raw.get("daily", {})
        averages = _monthly_averages(daily)
        annual   = averages["annual"]

        result = {
            "location":                  {"lat": lat, "lon": lon},
            "source":                    "Open-Meteo Historical Archive",
            "period":                    f"{start_year}-{end_year}",
            "precip_mm_per_year":        annual["precip_mm_per_year"],
            "mean_tmax_c":               annual["mean_tmax_c"],
            "mean_tmin_c":               annual["mean_tmin_c"],
            "suggested_recharge_flux_ms": annual["recharge_flux_ms"],
            "note": (
                "Recharge flux = 30% of annual precip / seconds per year. "
                "Adjust fraction based on site ET and runoff characteristics."
            )
        }
        return [types.TextContent(type="text", text=json.dumps(result))]

    else:
        return [types.TextContent(type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}))]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write,
                         server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())