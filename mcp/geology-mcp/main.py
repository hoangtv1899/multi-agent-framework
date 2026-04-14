#!/usr/bin/env python3
"""
Geology MCP Server - Soil properties from SSURGO for PFLOTRAN workflows

Data source:
    SSURGO SDA (USDA) - free, no key needed
    https://sdmdataaccess.sc.egov.usda.gov

Tools:
    get_soil_profile(lat, lon)
        → soil horizons with texture, depth, hydraulic properties
        → PFLOTRAN-ready Van Genuchten parameters per layer

    get_pflotran_materials(lat, lon)
        → directly injectable MATERIAL_PROPERTIES for PFLOTRAN input
"""
import asyncio
import json
import math
import requests
import urllib3                                          # ← add
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # ← add
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SSURGO_URL = ("https://sdmdataaccess.sc.egov.usda.gov"
              "/Tabular/SDMTabularService/post.rest")

server = Server("geology")

# ─────────────────────────────────────────────────────────────────────────────
# SSURGO QUERY
# ─────────────────────────────────────────────────────────────────────────────

def _query_ssurgo(lat: float, lon: float) -> list:
    """
    Query SSURGO for soil horizons at a point.
    Returns list of horizon dicts with raw soil properties.
    """
    query = f"""
SELECT
    co.compname,
    co.comppct_r,
    hz.hzdept_r,
    hz.hzdepb_r,
    hz.texturerv,
    hz.sandtotal_r,
    hz.silttotal_r,
    hz.claytotal_r,
    hz.wsatiated_r,
    hz.wthirdbar_r,
    hz.wfifteenbar_r,
    hz.ksat_r,
    hz.dbthirdbar_r,
    hz.om_r
FROM
    component co
    INNER JOIN chorizon hz ON hz.cokey = co.cokey
    INNER JOIN mapunit mu ON mu.mukey = co.mukey
    INNER JOIN SDA_Get_Mukey_from_intersection_with_WktWgs84(
        'point({lon} {lat})'
    ) AS t ON mu.mukey = t.mukey
WHERE
    co.majcompflag = 'Yes'
ORDER BY co.comppct_r DESC, hz.hzdept_r ASC
"""
    try:
        r = requests.post(
            SSURGO_URL,
            data={'query': query, 'format': 'json+columnname'},
            timeout=30,
            verify=False
        )
        r.raise_for_status()
        data = r.json()
        rows = data.get('Table', [])
        if len(rows) < 2:
            return []

        # First row is column names
        cols = rows[0]
        return [dict(zip(cols, row)) for row in rows[1:]]

    except Exception as e:
        return [{"error": str(e)}]


# ─────────────────────────────────────────────────────────────────────────────
# VAN GENUCHTEN PARAMETER DERIVATION
# ─────────────────────────────────────────────────────────────────────────────

def _derive_van_genuchten(horizon: dict) -> dict:
    """
    Derive Van Genuchten parameters from SSURGO soil properties.

    Uses Rawls & Brakensiek (1985) pedotransfer functions - standard
    approach for deriving VG params from texture and bulk density.

    Parameters derived:
        theta_r  → residual water content
        theta_s  → saturated water content (porosity)
        alpha    → Van Genuchten alpha (1/m)
        n        → Van Genuchten n
        m        → Van Genuchten m = 1 - 1/n
        Ksat     → saturated hydraulic conductivity (m/s)

    Returns PFLOTRAN-formatted Fortran notation values.
    """
    # Extract raw values (handle None)
    sand  = _safe_float(horizon.get('sandtotal_r'), 40.0)   # %
    silt  = _safe_float(horizon.get('silttotal_r'), 40.0)   # %
    clay  = _safe_float(horizon.get('claytotal_r'), 20.0)   # %
    db    = _safe_float(horizon.get('dbthirdbar_r'), 1.4)   # g/cm³
    ksat  = _safe_float(horizon.get('ksat_r'), 1.0)         # µm/s
    wsat  = _safe_float(horizon.get('wsatiated_r'), 0.45)   # cm³/cm³
    w3bar = _safe_float(horizon.get('wthirdbar_r'), 0.25)   # cm³/cm³
    w15bar= _safe_float(horizon.get('wfifteenbar_r'), 0.10) # cm³/cm³

    # ── Porosity (theta_s) ────────────────────────────────────────────────
    # Use saturated water content directly if available
    # Otherwise derive from bulk density: n = 1 - db/2.65
    if wsat and wsat > 0:
        theta_s = wsat
    else:
        theta_s = 1.0 - (db / 2.65)
    theta_s = max(0.25, min(0.75, theta_s))

    # ── Residual water content (theta_r) ──────────────────────────────────
    # Rawls & Brakensiek (1985)
    theta_r = max(0.0, 0.0018 + 0.0009 * sand + 0.005 * clay
                  + 0.029 * theta_s - 0.0002 * sand * clay
                  - 0.0003 * sand * theta_s + 0.0033 * clay * theta_s)
    theta_r = min(theta_r, theta_s * 0.5)

    # ── Van Genuchten n ───────────────────────────────────────────────────
    # Schaap et al. (2001) - Rosetta model simplified
    # n increases with sand content, decreases with clay
    n = 1.0 + 0.3 * (sand / 100.0) - 0.1 * (clay / 100.0)
    n = max(1.1, min(3.5, n))

    # ── Van Genuchten m ───────────────────────────────────────────────────
    m = 1.0 - (1.0 / n)

    # ── Van Genuchten alpha (1/m) ─────────────────────────────────────────
    # Rawls & Brakensiek (1985) - depends on texture
    # Fine soils: low alpha, coarse soils: high alpha
    alpha = 0.001 * math.exp(
        -2.796 + 0.0249 * sand - 0.0618 * clay
        + 0.0348 * (silt - clay) / 100.0
    )
    alpha = max(0.0001, min(0.1, alpha))   # reasonable bounds (1/m)

    # ── Saturated hydraulic conductivity (m/s) ────────────────────────────
    # SSURGO ksat is in µm/s → convert to m/s
    if ksat and ksat > 0:
        ksat_ms = ksat * 1e-6
    else:
        # Rawls & Brakensiek estimate from texture
        ksat_ms = 10 ** (-0.0693 - 0.0340 * clay
                         + 0.0012 * sand) * 1e-6
    ksat_ms = max(1e-12, min(1e-3, ksat_ms))

    # ── Permeability (m²) from Ksat ───────────────────────────────────────
    # k = Ksat * mu / (rho * g)
    # mu=1e-3 Pa·s, rho=1000 kg/m³, g=9.81 m/s²
    permeability = ksat_ms * 1e-3 / (1000.0 * 9.81)

    # ── Format as Fortran scientific notation ─────────────────────────────
    def _fortran(val: float, decimals: int = 4) -> str:
        """Convert float to Fortran double notation e.g. 1.5d-12"""
        if val == 0:
            return "0.0d0"
        exp  = int(math.floor(math.log10(abs(val))))
        mant = val / (10 ** exp)
        return f"{mant:.{decimals}f}d{exp:+d}".replace("e", "d")

    return {
        # Van Genuchten parameters
        "theta_s":     round(theta_s, 4),
        "theta_r":     round(theta_r, 4),
        "alpha_per_m": round(alpha, 6),
        "n":           round(n, 4),
        "m":           round(m, 4),
        "ksat_ms":     ksat_ms,
        # Permeability
        "permeability_m2": permeability,
        # PFLOTRAN Fortran-formatted strings
        "pflotran": {
            "POROSITY":  f"{theta_s:.4f}d0",
            "PERM_ISO":  _fortran(permeability),
            "ALPHA":     _fortran(alpha),
            "M":         f"{m:.4f}d0",
            "LIQUID_RESIDUAL_SATURATION": f"{theta_r:.4f}d0"
        },
        # Derivation notes
        "source": "Rawls & Brakensiek (1985) pedotransfer functions",
        "note": (
            f"Derived from: sand={sand:.1f}% silt={silt:.1f}% "
            f"clay={clay:.1f}% db={db:.2f}g/cm³"
        )
    }


def _safe_float(val, default: float) -> float:
    """Safely convert value to float, return default if None/invalid."""
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _texture_class(sand: float, silt: float, clay: float) -> str:
    """USDA texture class from sand/silt/clay percentages."""
    if clay >= 40:
        return "clay"
    elif clay >= 27 and sand <= 20:
        return "silty clay"
    elif clay >= 27:
        return "clay loam"
    elif clay >= 20 and sand <= 45:
        return "loam"
    elif silt >= 50 and clay < 27:
        return "silt loam"
    elif sand >= 70 and clay < 15:
        return "sandy loam"
    elif sand >= 85:
        return "sand"
    else:
        return "loam"


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_soil_profile",
            description=(
                "Get soil horizon profile from SSURGO for a location. "
                "Returns layer depths, texture, and hydraulic properties "
                "with derived Van Genuchten parameters for PFLOTRAN."
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
            name="get_pflotran_materials",
            description=(
                "Get PFLOTRAN-ready MATERIAL_PROPERTIES for a location. "
                "Returns directly injectable Van Genuchten parameters "
                "in Fortran notation for each soil layer."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"}
                },
                "required": ["lat", "lon"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str,
                    arguments: dict) -> list[types.TextContent]:

    lat = arguments["lat"]
    lon = arguments["lon"]

    # Query SSURGO
    horizons = _query_ssurgo(lat, lon)

    if not horizons:
        return [types.TextContent(type="text", text=json.dumps(
            {"error": "No SSURGO data found for this location"}
        ))]

    if "error" in horizons[0]:
        return [types.TextContent(type="text", text=json.dumps(
            {"error": horizons[0]["error"]}
        ))]

    # ── Tool 1: get_soil_profile ─────────────────────────────────────────────
    if name == "get_soil_profile":
        layers = []
        for hz in horizons:
            sand  = _safe_float(hz.get('sandtotal_r'), 40.0)
            silt  = _safe_float(hz.get('silttotal_r'), 40.0)
            clay  = _safe_float(hz.get('claytotal_r'), 20.0)
            vg    = _derive_van_genuchten(hz)

            layers.append({
                "component":    hz.get('compname', 'Unknown'),
                "horizon":      hz.get('texturerv', 'Unknown'),
                "depth_top_cm": hz.get('hzdept_r'),
                "depth_bot_cm": hz.get('hzdepb_r'),
                "texture_class": _texture_class(sand, silt, clay),
                "sand_pct":     sand,
                "silt_pct":     silt,
                "clay_pct":     clay,
                "bulk_density_gcc":  hz.get('dbthirdbar_r'),
                "ksat_ums":          hz.get('ksat_r'),
                "organic_matter_pct": hz.get('om_r'),
                "van_genuchten": {
                    "theta_s":     vg["theta_s"],
                    "theta_r":     vg["theta_r"],
                    "alpha_per_m": vg["alpha_per_m"],
                    "n":           vg["n"],
                    "m":           vg["m"],
                    "ksat_ms":     vg["ksat_ms"],
                },
                "note": vg["note"]
            })

        result = {
            "location":   {"lat": lat, "lon": lon},
            "source":     "USDA SSURGO Soil Data Access",
            "num_layers": len(layers),
            "layers":     layers
        }
        return [types.TextContent(type="text", text=json.dumps(result))]

    # ── Tool 2: get_pflotran_materials ───────────────────────────────────────
    elif name == "get_pflotran_materials":
        materials = []

        for i, hz in enumerate(horizons, 1):
            vg   = _derive_van_genuchten(hz)
            sand = _safe_float(hz.get('sandtotal_r'), 40.0)
            silt = _safe_float(hz.get('silttotal_r'), 40.0)
            clay = _safe_float(hz.get('claytotal_r'), 20.0)

            top_cm = _safe_float(hz.get('hzdept_r'), 0)
            bot_cm = _safe_float(hz.get('hzdepb_r'), 30)

            materials.append({
                # Layer identity
                "layer_id":       i,
                "layer_name":     f"layer_{i}",
                "component_name": hz.get('compname', 'Unknown'),
                "horizon_label":  hz.get('texturerv', f'hz{i}'),
                "texture_class":  _texture_class(sand, silt, clay),
                "depth_top_cm":   top_cm,
                "depth_bot_cm":   bot_cm,
                "thickness_cm":   bot_cm - top_cm,

                # PFLOTRAN MATERIAL_PROPERTIES (Fortran notation)
                "PFLOTRAN_MATERIAL_PROPERTIES": {
                    "ID":       i,
                    "POROSITY": vg["pflotran"]["POROSITY"],
                    "PERMEABILITY": {
                        "PERM_ISO": vg["pflotran"]["PERM_ISO"]
                    },
                    "CHARACTERISTIC_CURVE": {
                        "SATURATION_FUNCTION":        "VAN_GENUCHTEN",
                        "ALPHA":                       vg["pflotran"]["ALPHA"],
                        "M":                           vg["pflotran"]["M"],
                        "LIQUID_RESIDUAL_SATURATION":
                            vg["pflotran"]["LIQUID_RESIDUAL_SATURATION"]
                    },
                    "PERMEABILITY_FUNCTION": {
                        "TYPE":   "MUALEM_VG_LIQ",
                        "M":      vg["pflotran"]["M"],
                        "LIQUID_RESIDUAL_SATURATION":
                            vg["pflotran"]["LIQUID_RESIDUAL_SATURATION"]
                    }
                },

                # Human-readable equivalents
                "human_readable": {
                    "porosity":        vg["theta_s"],
                    "permeability_m2": vg["permeability_m2"],
                    "ksat_ms":         vg["ksat_ms"],
                    "alpha_per_m":     vg["alpha_per_m"],
                    "n":               vg["n"],
                    "m":               vg["m"],
                    "residual_sat":    vg["theta_r"]
                },

                # Derivation transparency
                "derivation": {
                    "method": "Rawls & Brakensiek (1985)",
                    "inputs": vg["note"]
                }
            })

        result = {
            "location": {"lat": lat, "lon": lon},
            "source":   "USDA SSURGO + Rawls & Brakensiek (1985)",
            "num_layers": len(materials),
            "materials":  materials,
            "usage_note": (
                "PFLOTRAN_MATERIAL_PROPERTIES can be directly injected "
                "into MATERIAL_PROPERTIES block. Layer depths in cm - "
                "convert to metres for PFLOTRAN coordinates."
            )
        }
        return [types.TextContent(type="text", text=json.dumps(result))]

    else:
        return [types.TextContent(type="text", text=json.dumps(
            {"error": f"Unknown tool: {name}"}
        ))]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write,
                         server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
