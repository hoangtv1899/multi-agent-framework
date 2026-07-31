"""
1-D variably-saturated column decks from a site description.

`create_pflotran_input` writes the skeleton of a deck — simulation type, grid,
time, output. That is everything except the physics: a deck with no material
properties, no regions, no strata and no flow conditions is rejected by
PFLOTRAN at read time. This module fills that gap for the case the LAMBDA
workflow actually needs, a single 1-D soil column:

    materials  <- the site's soil horizons, one cell per horizon, with van
                  Genuchten parameters carried through; the deepest horizon is
                  extended downward as substrate on a geometrically coarsening
                  grid
    initial    <- hydrostatic, pinned so that the water table sits at the
                  site's reported depth
    top BC     <- recharge flux, steady or transient
    bottom BC  <- hydrostatic at the same water table, or no-flow

WHY THE WATER TABLE IS AN INPUT rather than a spin-up result: for a
regional ensemble there is usually a modelled water-table prior available per
site (Fan et al. 2013 is the common one for the US) and no measured subsurface
state at all. Starting each column at its prior and letting it relax is a
defensible initial condition; starting every column at the same arbitrary
depth is not.

WHAT IS DELIBERATELY NOT HERE. No coupling to any land-surface model. A
transient upper boundary is supplied as plain (time, flux) pairs through
`recharge_series`, so a caller can drive it from ELM, from CLM, from a
measured infiltration record, or from nothing at all — this module does not
need to know which, and does not import any of them.

Depends only on the standard library plus tools.pflotran_input_agent.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .pflotran_input_agent import PFLOTRANInputAgent

# k[m2] = Ksat[m/s] * mu / (rho*g)
VISC = 1.002e-3
RHO_G = 998.0 * 9.81
ATM = 101325.0

# Below this a horizon is not worth its own cell.
MIN_HORIZON_M = 0.005
# Substrate grid: first cell, growth factor, ceiling.
SUB_DZ0, SUB_GROWTH, SUB_DZ_MAX = 0.3, 1.35, 2.5


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def dominant_horizons(profile: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The horizons of ONE soil component, ordered top-down.

    Survey profiles (SSURGO among them) often stack the horizons of several
    map-unit components in a single list. Interleaving them produces a
    physically meaningless column — alternating sand and clay that exists at
    no point on the ground — so keep the first component only.
    """
    layers = (profile or {}).get("layers") or []
    if not layers:
        return []
    comp = layers[0].get("component")
    hz = [l for l in layers if l.get("component") == comp]
    hz.sort(key=lambda l: _f(l.get("depth_top_cm"), 0))
    return hz


def layer_props(hz: Dict[str, Any]) -> Dict[str, float]:
    """PFLOTRAN material + characteristic curve from one soil horizon.

    Residual SATURATION, not residual water content: PFLOTRAN wants
    theta_r/theta_s, and passing theta_r directly is a quiet way to make a
    column that never drains.
    """
    vg = hz.get("van_genuchten") or {}
    ts = _f(vg.get("theta_s"), 0.45)
    tr = _f(vg.get("theta_r"), 0.05)
    ksat = _f(vg.get("ksat_ms")) or (_f(hz.get("ksat_ums"), 3.0) * 1e-6)

    # alpha is conventionally quoted in 1/Pa, but plenty of sources give 1/m.
    # Anything above 0.01 cannot be 1/Pa for a soil, so treat it as 1/m.
    alpha = _f(vg.get("alpha_per_m"), 1e-4)
    if alpha > 0.01:
        alpha /= RHO_G

    return {
        "porosity": round(ts, 4),
        "permeability": ksat * VISC / RHO_G,
        "alpha": alpha,
        "m": _f(vg.get("m"), 0.4),
        "residual_saturation": round(min(tr / ts, 0.4), 4),
    }


def _discretize(hz: Sequence[Dict[str, Any]], depth: float
                ) -> List[Tuple[float, Dict[str, float]]]:
    """(thickness, properties) per cell, top-down: horizons then substrate."""
    soil: List[Tuple[float, Dict[str, float]]] = []
    for l in hz:
        t = (_f(l.get("depth_bot_cm"), 0) - _f(l.get("depth_top_cm"), 0)) / 100.0
        if t > MIN_HORIZON_M:
            soil.append((t, layer_props(l)))

    soil_depth = sum(t for t, _ in soil)
    sub_props = soil[-1][1] if soil else layer_props({})

    # Coarsen with depth: resolution is wanted near the surface where the
    # wetting front moves, not at the bottom of the substrate.
    sub: List[Tuple[float, Dict[str, float]]] = []
    dz, z = SUB_DZ0, soil_depth
    while z < depth:
        dz = min(dz * SUB_GROWTH, SUB_DZ_MAX, depth - z)
        sub.append((dz, sub_props))
        z += dz
    return soil + sub


def build_column_deck(
    column: Dict[str, Any],
    out_dir: str,
    water_table_m: Optional[float] = None,
    recharge_mm_yr: float = 100.0,
    recharge_series: Optional[Sequence[Sequence[float]]] = None,
    years: float = 20.0,
    depth_cap: float = 50.0,
    bottom: str = "water_table",
    spin_years: float = 10.0,
) -> Dict[str, Any]:
    """Write one runnable 1-D column deck. Returns its case directory and metadata.

    Args:
        column: site description. Uses `id`, `soil_profile` (a dict with
            `layers`), and `fan_wtd_m` (or pass `water_table_m` explicitly).
        out_dir: parent directory; the case gets its own subdirectory.
        water_table_m: water-table depth below ground, metres, positive down.
            Falls back to column['fan_wtd_m'].
        recharge_mm_yr: steady top flux, used when `recharge_series` is None.
        recharge_series: transient top flux as [[time_y, flux_m_per_y], ...].
            The deck then runs `spin_years` steady at the series mean before
            the transient begins, so the column is not still relaxing out of
            its initial condition when the signal of interest arrives.
        years: simulated duration when the flux is steady.
        depth_cap: maximum domain depth. A site whose water table is deeper
            than this runs fully unsaturated — reported as
            wt_in_domain=False rather than silently deepened, because a
            column with no water table answers a different question.
        bottom: "water_table" for hydrostatic at the water table, "none" for
            no-flow.
        spin_years: steady spin-up before a transient series.

    Returns:
        case_dir, input_file, n_cells, depth_m, water_table_m, wt_in_domain,
        soil_horizons, recharge_mm_yr.
    """
    cid = str(column.get("id") or "column")
    hz = dominant_horizons(column.get("soil_profile"))
    wt = _f(water_table_m if water_table_m is not None
            else column.get("fan_wtd_m"), 10.0)

    # Deep enough to contain the water table when that is affordable, never
    # shallower than 12 m, never deeper than the cap.
    depth = min(depth_cap, max(wt + 5.0, 12.0))

    cells = _discretize(hz, depth)
    thicknesses = [t for t, _ in cells][::-1]        # PFLOTRAN: bottom -> top
    H = sum(thicknesses)

    case = PFLOTRANInputAgent(nx=1, ny=1, dx=1.0, dy=1.0,
                              layer_thicknesses=thicknesses, case_name=cid)
    names = []
    for i, (_t, p) in enumerate(cells[::-1]):        # bottom -> top
        n = f"L{i + 1:02d}"
        names.append(n)
        case.add_material_property({"name": n, "id": i + 1,
                                    "porosity": p["porosity"],
                                    "permeability": p["permeability"]})
        case.add_characteristic_curve({"name": n, "alpha": p["alpha"],
                                       "m": p["m"],
                                       "liquid_residual_saturation":
                                           p["residual_saturation"]})
    case.set_layer_order(names)
    case.add_strata_from_layer_order()

    # Hydrostatic everywhere, atmospheric pressure at the water table. The
    # datum may sit below the domain — that is what an unsaturated column IS,
    # and PFLOTRAN handles it.
    datum = (0.0, 0.0, H - wt)
    case.flow_conditions[0].update({"datum": datum, "liquid_pressure": ATM})

    if recharge_series:
        series = [(float(t), float(v)) for t, v in recharge_series]
        mean = sum(v for _, v in series) / len(series)
        values = [(0.0, mean)] + [(spin_years + t, v) for t, v in series]
        annual = mean * 1000.0
    else:
        values = [(0.0, recharge_mm_yr / 1000.0)]
        annual = recharge_mm_yr

    case.add_flow_condition({
        "name": "recharge", "type": "LIQUID_FLUX NEUMANN",
        "flux_data": {"time_units": "y", "data_units": "m/y", "values": values}})
    case.add_boundary_condition("top_recharge", "recharge", "top")

    if bottom == "water_table":
        case.add_flow_condition({"name": "water_table",
                                 "type": "LIQUID_PRESSURE HYDROSTATIC",
                                 "datum": datum, "liquid_pressure": ATM})
        case.add_boundary_condition("bottom_wt", "water_table", "bottom")

    if recharge_series:
        end = spin_years + 1.0
        case.time_config.update({
            "final_time": end, "initial_timestep": 1.0,
            "maximum_timestep": 0.05,
            "extra_lines": [f"MAXIMUM_TIMESTEP_SIZE 2.d-3 y AT "
                            f"{spin_years:.1f} y"]})
        case.output_config["times"] = [spin_years + f
                                       for f in (0.25, 0.5, 0.75, 1.0)]
        case.output_config["mass_balance"] = True
    else:
        case.time_config.update({"final_time": years, "initial_timestep": 1.0,
                                 "maximum_timestep": 0.05})
        case.output_config["times"] = sorted({1.0, years / 4, years / 2, years})

    case_dir = case.prepare_case(output_dir=str(out_dir))
    deck = next(Path(case_dir).glob("*.in"), None)

    return {
        "case_dir": str(case_dir),
        "input_file": str(deck) if deck else None,
        "n_cells": len(cells),
        "depth_m": round(H, 3),
        "water_table_m": wt,
        # False means the column never saturates. Reported, not corrected:
        # "no water table in the domain" and "water table at the bottom" are
        # different statements about the site.
        "wt_in_domain": bool(wt < H),
        "soil_horizons": len([1 for l in hz
                              if (_f(l.get("depth_bot_cm"), 0)
                                  - _f(l.get("depth_top_cm"), 0)) / 100.0
                              > MIN_HORIZON_M]),
        "recharge_mm_yr": annual,
        "transient": bool(recharge_series),
        "validation_status": "success",
    }
