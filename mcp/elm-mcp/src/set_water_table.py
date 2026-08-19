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
