#!/usr/bin/env python3
"""
The CONUS2 subsurface, read locally
src/core/conus2_subsurface.py

Reception fetches ParFlow CONUS2's subsurface parameter fields ONCE per basin
and writes them beside the water table. This reads that file.

NO NETWORK, NO PIN, NO MCP SESSION — the sibling of core/static_wtd.py, for the
same reason and with the same discipline. Five 3-D fields over a basin-sized box
are 3.8 MB fetched and 0.07 MB on disk once compressed; sampling them per column
would be five requests per column to a university's server, and the answer never
changes because every field is `static`.

WHAT IT REPLACES. Until 2026-08-17 a PFLOTRAN column was SSURGO horizons to
about 1.5 m and then invented material to 12-50 m — `_discretize` copied the
deepest surveyed horizon downward, so 88-97% of a column was one extrapolated
layer. CONUS2 parameterises the whole 392 m: SSURGO-derived soil in its top four
layers and GLHYMPS hydrogeologic units below (Yang et al. 2021, ESSD 13:3263).

VOCABULARY IS CONUS2'S, NOT PFLOTRAN'S. This module hands back exactly what the
dataset holds — hydraulic conductivity in m/h, van Genuchten alpha in 1/m, and
`n` rather than `m`. Converting those into what a deck wants is knowledge of the
model, and it lives behind the PFLOTRAN MCP. Reading a raster at a point is not.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# THE FILENAMES ARE A CONTRACT with core/data_gather.gather_subsurface, which
# writes them. Change them in both or in neither.
ARRAYS_NAME = "conus2_subsurface.npz"
META_NAME = "conus2_subsurface.json"

# CONUS2 is 10 layers to 392 m, and the ORDER IS BOTTOM-TO-TOP: index 0 is the
# deepest 200 m layer, index 9 the 0.1 m layer at the surface. That is the
# dataset's own convention and it happens to match PFLOTRAN's, so nothing is
# reversed anywhere — but reading it the other way round would put 200 m of
# bedrock at the land surface and produce a column that looks entirely
# plausible, which is why it is stated here rather than assumed.
THICKNESS_M_BOTTOM_TO_TOP = (200.0, 100.0, 50.0, 25.0, 10.0,
                             5.0, 1.0, 0.6, 0.3, 0.1)
N_LAYERS = len(THICKNESS_M_BOTTOM_TO_TOP)

FIELDS = ("porosity", "permeability_z", "vg_alpha", "vg_n", "sres")

# Where each layer sits below the land surface, top-down. Pure arithmetic on the
# thicknesses above; computed once so no caller re-derives it and gets it wrong.
def _depths() -> List[Dict[str, float]]:
    out, z = [], 0.0
    for i in range(N_LAYERS - 1, -1, -1):          # 9 (surface) .. 0 (deepest)
        t = THICKNESS_M_BOTTOM_TO_TOP[i]
        out.append({"index": i, "thickness_m": t,
                    "depth_top_m": round(z, 3), "depth_bot_m": round(z + t, 3)})
        z += t
    return out


LAYERS = _depths()                        # top-down, each carrying its index
TOTAL_DEPTH_M = LAYERS[-1]["depth_bot_m"]                 # 392.0

# The boundaries a domain may end on. A column stops on one of these rather than
# at an arbitrary depth, so its cells are whole CONUS2 layers and no layer is
# half-represented.
BOUNDARIES_M = tuple(l["depth_bot_m"] for l in LAYERS)    # 0.1 .. 392.0


def paths(where) -> Optional[Dict[str, Path]]:
    """The fetched files for a run, or None if they were never written.

    `where` may be the run directory OR a path to a file inside it — normally
    reception.json, matching static_wtd.raster_path so a caller holding one
    path can ask both modules.
    """
    p = Path(where)
    d = p.parent if p.is_file() else p
    a, m = d / ARRAYS_NAME, d / META_NAME
    return {"arrays": a, "meta": m} if a.is_file() and m.is_file() else None


def _cell(meta: Dict[str, Any], lat: float, lon: float):
    """(ix, iy) into the fetched box, or None outside it.

    FLOOR, NOT ROUND, and the box origin subtracted — the same rule
    hydrodata-mcp._xy uses. `from_latlon` returns a FRACTIONAL index and
    `grid_bounds` is a half-open slice, so the cell containing a point is the
    floor of its index. Rounding reads the neighbouring cell for about half of
    all points, which on a 1 km grid is a whole kilometre away.
    """
    try:
        import hf_hydrodata as hf
        x, y = hf.from_latlon("conus2", float(lat), float(lon))
    except Exception:                              # noqa: BLE001
        return None
    x0, y0, x1, y1 = meta["grid_bounds"]
    ix, iy = int(x // 1) - x0, int(y // 1) - y0
    if 0 <= ix < (x1 - x0) and 0 <= iy < (y1 - y0):
        return ix, iy
    return None


def sample(where, lats: Sequence[float],
           lons: Sequence[float]) -> List[Optional[Dict[str, Any]]]:
    """One subsurface profile per point, in input order. None where unavailable.

    None means the files were never written, or the point lies outside the box
    that was fetched. A missing value is NEVER replaced by the nearest edge
    cell: a clamped read returns a real-looking profile from the wrong place.

    Each profile is TOP-DOWN — layer 0 of the returned list is the 0.1 m layer
    at the surface — because every consumer reasons downward from the ground.
    The dataset's own bottom-to-top index is carried on each layer as `index`
    so the two can always be reconciled.
    """
    got = paths(where)
    if not got:
        return [None] * len(lats)
    import numpy as np
    meta = json.loads(got["meta"].read_text())
    arr = np.load(got["arrays"])
    missing = [f for f in FIELDS if f not in arr.files]
    if missing:
        return [None] * len(lats)

    out: List[Optional[Dict[str, Any]]] = []
    for la, lo in zip(lats, lons):
        c = _cell(meta, la, lo)
        if c is None:
            out.append(None)
            continue
        ix, iy = c
        layers = []
        ok = True
        for L in LAYERS:                            # top-down
            row = {"index": L["index"], "thickness_m": L["thickness_m"],
                   "depth_top_m": L["depth_top_m"],
                   "depth_bot_m": L["depth_bot_m"]}
            for f in FIELDS:
                v = float(arr[f][L["index"], iy, ix])
                if not np.isfinite(v):
                    ok = False
                row[f] = v
            layers.append(row)
        if not ok:
            out.append(None)                        # a no-data cell, said plainly
            continue
        out.append({
            "source": meta.get("source"),
            "grid": "conus2", "resolution_m": 1000,
            "cell": {"ix": ix, "iy": iy},
            "total_depth_m": TOTAL_DEPTH_M,
            "units": {"porosity": "-", "permeability_z": "m/h",
                      "vg_alpha": "1/m", "vg_n": "-", "sres": "-"},
            "soil_layers": "indices 9,8,7,6 (0-2 m) are SSURGO-derived soil; "
                           "5..0 (2-392 m) are GLHYMPS hydrogeologic units",
            "layers": layers,
        })
    return out


# The shallowest domain worth solving. A water table at the land surface asks
# for a 0.1 m column under the boundary rule below, which is not a simulation —
# it is one cell. 2.0 m is CONUS2's own soil/geology boundary, so the floor
# lands on a real interface rather than on a number someone liked.
MIN_DEPTH_M = 2.0


def domain_depth_for(water_table_m: Optional[float],
                     cap_m: float = TOTAL_DEPTH_M,
                     min_m: float = MIN_DEPTH_M) -> Dict[str, Any]:
    """The CONUS2 boundary a column should end on to contain its water table.

    THE DOMAIN ENDS ON A LAYER BOUNDARY, not at an arbitrary depth. It used to
    be `min(depth_cap, max(wt + 5, 12))` — three constants with nothing behind
    them, which also cut layers in half. Ending on a boundary means every cell
    is a whole CONUS2 layer subdivided, and the deepest one is not a fragment.

    Returns the depth, whether the water table fits, and why — a column whose
    water table lies below the cap is REPORTED, never silently deepened, since
    it is no longer the site it was placed at.
    """
    wt = None if water_table_m is None else float(water_table_m)
    usable = [b for b in BOUNDARIES_M if b <= cap_m] or [BOUNDARIES_M[0]]

    def out(depth, in_dom, why):
        # HOW MUCH UNSATURATED COLUMN THERE IS TO SOLVE, which is the quantity
        # a vadose-zone study actually spends its columns on. A column whose
        # water table sits at the surface is fully saturated: it builds, it
        # runs, and it answers nothing about percolation. Said here rather than
        # discovered in the output.
        uns = None if wt is None else round(min(wt, depth), 3)
        r = {"depth_m": depth, "water_table_m": wt, "wt_in_domain": in_dom,
             "unsaturated_m": uns, "why": why}
        if uns is not None and uns < 0.5:
            r["warning"] = (f"only {uns:.2f} m of unsaturated column — the "
                            f"water table is at or near the surface, so this "
                            f"column is essentially saturated and cannot show "
                            f"vertical transit")
        return r

    if wt is None:
        return out(usable[-1], False,
                   "no water table given; the domain runs to the cap")
    fits = [b for b in usable if b > wt and b >= min_m]
    if fits:
        why = f"shallowest CONUS2 boundary below a {wt:.2f} m water table"
        if fits[0] > wt and wt < min_m:
            why = (f"the {min_m:.1f} m floor — a {wt:.2f} m water table would "
                   f"otherwise give a domain too thin to solve")
        return out(fits[0], True, why)
    return out(usable[-1], False,
               f"the water table is {wt:.2f} m, below the deepest boundary "
               f"within the {cap_m:.0f} m cap — the column is CAPPED and "
               f"does not reach it")
