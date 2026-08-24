#!/usr/bin/env python3
"""
Write an initial water table into a single-column ELM finidat.
mcp/elm-mcp/src/set_water_table.py

    in   a finidat subset (make_finidat_subset's output) and a depth in metres
    out  the file edited in place: ZWT and WA mutually consistent, the edit
         reported — old and new values, the regime, any clamp

THE ONE CAPABILITY TWO-WAY COUPLING NEEDED. ELM's conceptual declaration has
listed "a graded initial water table — needs perturbed restart files, a netCDF
edit on WA/ZWT" under cannot_vary since 2026-08; this is that edit. A coupled
follow-up hands each column PFLOTRAN's solved water table as its next initial
state, so the two models start an iteration agreeing about where the water
stands.

THE ZWT<->WA RELATION IS ELM'S OWN, RECOVERED FROM ELM'S OWN RESTART rather
than recalled from a manual (measured 2026-08-19 on the CONUS lat-band 1
restart, 221,851 aquifer-regime columns):

    zwt = 28.802 - wa / 200            for 82.4% of them, EXACTLY
          (28.802 m = the 15-level grid's soil bottom zi(10) = 3.802 m + a
           25 m conceptual aquifer; 200 mm/m = specific yield 0.2)

    the rest deviate by up to ~4 m — columns whose water table sits inside or
    near the soil column, where ELM diagnoses zwt from the moisture profile
    instead of from WA.

TWO REGIMES, ONE OF THEM EXACT:

    aquifer   target BELOW the soil column (> 3.802 m): ZWT and WA are the
              whole state, both are written, the relation above holds to the
              millimetre. This is the regime the coupling lives in — a basin's
              deep columns, the ones whose QDRAI is ~0.
    in-soil   target INSIDE the soil column (<= 3.802 m): ZWT is written and
              WA parks at its 5000 mm ceiling, but ELM re-diagnoses an in-soil
              water table from the MOISTURE profile, which this edit does not
              touch (saturating layers honestly needs each layer's porosity,
              which lives in ELM's pedotransfer, not in this file). The edit
              is therefore APPROXIMATE here and the result says so.

A target deeper than 28.802 m is CLAMPED to it and the clamp reported: ELM's
aquifer simply ends there, and pretending otherwise would write a WA below
zero. Nothing else in the file is touched.
"""
from pathlib import Path
from typing import Any, Dict

# ELM's own numbers, recovered from its own restart (module docstring).
SOIL_BOTTOM_M = 3.802
AQUIFER_MAX_M = 28.802
SY_MM_PER_M = 200.0
WA_MAX_MM = 5000.0

# ELM's soil discretisation: 15 ground layers on the standard exponential
# grid, the top 10 hydrologically active. Computed from ELM's own formula
# (zsoi_j = 0.025*(exp(0.5*(j-0.5))-1)) and cross-checked below against the
# 3.802 m soil bottom this module already measured off the CONUS restart.
ELM_NLEVGRND = 15
ELM_NLEVSOI = 10
WATMIN_KG_M2 = 0.01          # ELM's own floor on layer liquid water
ORGANIC_MAX = 130.0          # kg/m3, ELM's organic-matter density ceiling
OM_WATSAT = 0.9              # porosity of pure organic matter, ELM's value


def _layer_grid():
    """Node depths and thicknesses of ELM's 15 ground layers, in metres."""
    import math
    z = [0.025 * (math.exp(0.5 * (j - 0.5)) - 1.0)
         for j in range(1, ELM_NLEVGRND + 1)]
    dz = ([0.5 * (z[0] + z[1])]
          + [0.5 * (z[j + 1] - z[j - 1]) for j in range(1, ELM_NLEVGRND - 1)]
          + [z[ELM_NLEVGRND - 1] - z[ELM_NLEVGRND - 2]])
    # the interface below layer 10 IS the module's measured soil bottom —
    # if the formula and the restart ever disagree, stop rather than stamp.
    zi10 = 0.5 * (z[ELM_NLEVSOI - 1] + z[ELM_NLEVSOI])
    if abs(zi10 - SOIL_BOTTOM_M) > 0.01:
        raise RuntimeError(f"layer grid zi(10)={zi10:.4f} m does not match "
                           f"the measured soil bottom {SOIL_BOTTOM_M} m")
    return z, dz


def _watsat_from_surfdata(surfdata: str):
    """Layer porosity via ELM's OWN pedotransfer, from the column's surface
    data (PCT_SAND, ORGANIC): watsat = (1-f_om)*(0.489 - 0.00126*sand)
    + f_om*0.9, f_om = ORGANIC/130. The same arithmetic ELM runs at init —
    which is the point: the moisture written here is the moisture ELM would
    diagnose for these pores, not the driving model's idea of them."""
    import numpy as np
    import netCDF4
    with netCDF4.Dataset(surfdata) as d:
        for name in ("PCT_SAND", "ORGANIC"):
            if name not in d.variables:
                raise KeyError(f"{Path(surfdata).name} has no {name} — "
                               f"not an ELM surface dataset?")
        sand = np.asarray(d.variables["PCT_SAND"][:]).reshape(-1)[:ELM_NLEVSOI]
        org = np.asarray(d.variables["ORGANIC"][:]).reshape(-1)[:ELM_NLEVSOI]
    f_om = np.clip(org / ORGANIC_MAX, 0.0, 1.0)
    watsat_min = 0.489 - 0.00126 * sand
    return (1.0 - f_om) * watsat_min + f_om * OM_WATSAT


def apply_profile(finidat: str, depth_m, saturation, surfdata: str,
                  quiet: bool = False) -> Dict[str, Any]:
    """Write the driving model's saturation profile into the soil layers.

    THE COMPANION TO apply(): that stamp sets the aquifer state (ZWT/WA);
    this one sets the layer moisture the in-soil diagnosis reads, so a
    coupled column starts from ONE coherent state instead of a solved water
    table over hydrostatic guesswork. Saturation is interpolated onto ELM's
    own layer nodes and converted with ELM's own porosity (pedotransfer from
    the column's surface data) — the driving model supplies WHERE the water
    is, ELM's soil decides how much fits. Ice is left in place and liquid
    reduced where ice already occupies pore space; only the 10 active layers
    (to 3.802 m) are touched, only in the unmasked entries.
    """
    import numpy as np
    import netCDF4

    f = Path(finidat)
    if not f.is_file():
        raise FileNotFoundError(f"no finidat at {finidat}")
    dep = np.asarray(depth_m, dtype=float)
    sat = np.asarray(saturation, dtype=float)
    if dep.ndim != 1 or dep.shape != sat.shape or dep.size < 2:
        raise ValueError("depth_m and saturation must be 1-D, equal length, "
                         f"length >= 2; got {dep.shape} and {sat.shape}")
    order = np.argsort(dep)
    dep, sat = dep[order], np.clip(sat[order], 0.0, 1.0)

    z, dz = _layer_grid()
    z10, dz10 = np.asarray(z[:ELM_NLEVSOI]), np.asarray(dz[:ELM_NLEVSOI])
    sat_j = np.interp(z10, dep, sat)          # clamped at the profile's ends
    watsat = _watsat_from_surfdata(surfdata)
    total_j = sat_j * watsat * dz10 * 1000.0  # kg/m2 per layer

    with netCDF4.Dataset(f, "r+") as d:
        for name in ("H2OSOI_LIQ", "H2OSOI_ICE", "ZWT"):
            if name not in d.variables:
                raise KeyError(f"{f.name} has no {name} — not an ELM restart?")
        if d.dimensions["levgrnd"].size != ELM_NLEVGRND:
            raise ValueError(f"{f.name}: levgrnd = "
                             f"{d.dimensions['levgrnd'].size}, expected "
                             f"{ELM_NLEVGRND} — a different ELM grid")
        off = d.variables["H2OSOI_LIQ"].shape[1] - ELM_NLEVGRND  # snow layers
        live = ~np.ma.getmaskarray(d.variables["ZWT"][:])
        rows = np.where(live)[0]
        if rows.size == 0:
            raise ValueError(f"{f.name}: every ZWT entry is masked — nothing "
                             f"to initialise")
        liq = d.variables["H2OSOI_LIQ"][:]
        ice = d.variables["H2OSOI_ICE"][:]
        old_liq = [round(float(v), 3)
                   for v in np.ravel(liq[rows[0], off:off + ELM_NLEVSOI])]
        n_ice = 0
        for r in rows:
            ice_r = np.asarray(ice[r, off:off + ELM_NLEVSOI], dtype=float)
            new_r = np.maximum(total_j - ice_r, WATMIN_KG_M2)
            liq[r, off:off + ELM_NLEVSOI] = new_r
            if r == rows[0]:
                n_ice = int((ice_r > 0).sum())
                new_liq = [round(float(v), 3) for v in new_r]
        d.variables["H2OSOI_LIQ"][:] = liq

    note = (f"top {round(0.5 * (z[ELM_NLEVSOI - 1] + z[ELM_NLEVSOI]), 3)} m "
            f"rewritten from the driving model's saturation profile using "
            f"ELM's own porosity (pedotransfer from the column's surface "
            f"data)"
            + (f"; ice left in place in {n_ice} layer(s), liquid reduced "
               f"where it occupies pore space" if n_ice else "")
            + "; deeper layers and the aquifer are the ZWT/WA stamp's")
    if not quiet:
        print(f"   {f.name}: soil moisture <- profile "
              f"(sat {sat_j[0]:.3f}..{sat_j[-1]:.3f} over 10 layers)")
    return {
        "finidat": str(f), "n_entries": int(rows.size),
        "layers_written": ELM_NLEVSOI,
        "saturation_at_layers": [round(float(v), 4) for v in sat_j],
        "watsat": [round(float(v), 4) for v in watsat],
        "old_liq_kg_m2": old_liq, "new_liq_kg_m2": new_liq,
        "n_ice_layers": n_ice, "note": note,
    }


def apply(finidat: str, water_table_m: float,
          quiet: bool = False) -> Dict[str, Any]:
    """Edit ZWT/WA in place. Returns what changed, per unmasked entry."""
    import numpy as np
    import netCDF4

    f = Path(finidat)
    if not f.is_file():
        raise FileNotFoundError(f"no finidat at {finidat}")
    target = float(water_table_m)
    if not target > 0:
        raise ValueError(f"water_table_m must be metres below ground, > 0; "
                         f"got {target!r}")

    clamped = target > AQUIFER_MAX_M
    zwt_new = min(target, AQUIFER_MAX_M)
    regime = "aquifer" if zwt_new > SOIL_BOTTOM_M else "in-soil"
    wa_new = (WA_MAX_MM - (zwt_new - SOIL_BOTTOM_M) * SY_MM_PER_M
              if regime == "aquifer" else WA_MAX_MM)

    with netCDF4.Dataset(f, "r+") as d:
        for name in ("ZWT", "WA"):
            if name not in d.variables:
                raise KeyError(f"{f.name} has no {name} — not an ELM restart?")
        zwt = d.variables["ZWT"][:]
        wa = d.variables["WA"][:]
        # ONLY THE LIVE ENTRIES. A subset finidat carries every landunit's
        # column slot; the inactive ones are masked fill and must stay so.
        live = ~np.ma.getmaskarray(zwt)
        n = int(live.sum())
        if n == 0:
            raise ValueError(f"{f.name}: every ZWT entry is masked — nothing "
                             f"to initialise")
        old_zwt = [round(float(v), 4) for v in np.ma.compressed(zwt)]
        old_wa = [round(float(v), 4) for v in np.ma.compressed(wa)]
        z2, w2 = np.ma.copy(zwt), np.ma.copy(wa)
        z2[live] = zwt_new
        w2[live] = wa_new
        d.variables["ZWT"][:] = z2
        d.variables["WA"][:] = w2

    note = (f"{regime} regime: ZWT={zwt_new:g} m, WA={wa_new:g} mm "
            f"(zwt = {AQUIFER_MAX_M} - wa/{SY_MM_PER_M:g}, ELM's own relation)"
            + (f"; CLAMPED from {target:g} m — ELM's aquifer ends at "
               f"{AQUIFER_MAX_M} m" if clamped else "")
            + ("; APPROXIMATE — an in-soil water table is re-diagnosed from "
               "the moisture profile, which this edit leaves untouched"
               if regime == "in-soil" else ""))
    if not quiet:
        print(f"   {f.name}: ZWT {old_zwt[0]:g} -> {zwt_new:g} m ({note})")
    return {
        "finidat": str(f), "n_entries": n,
        "requested_m": target, "written_m": zwt_new,
        "old_zwt_m": old_zwt[0], "old_wa_mm": old_wa[0],
        "new_wa_mm": round(wa_new, 4),
        "regime": regime, "clamped": clamped, "exact": regime == "aquifer",
        "note": note,
    }
