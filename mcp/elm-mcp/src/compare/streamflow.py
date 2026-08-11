#!/usr/bin/env python3
"""Streamflow: model QOVER + QDRAI against a gauge's specific discharge.

CONTEXT, NOT A SKILL CLAIM, and this is the observable the distinction was
invented for. A gauge measures discharge integrated and routed over an upstream
area; a 1-D column produces point runoff on a square metre with no routing.
They are not the same quantity, so no metric between them scores the model, and
the planner refuses to spend a column pinning one at a gauge.

The hydrograph SHAPE is still worth seeing — timing of rise and recession is a
real, comparable signal even when magnitude is not. So the comparison runs, the
numbers are reported, and `colocated: false` travels with them.

The scale contrast is made explicit rather than left implicit: a gauge's
drainage area is reported beside the column's 1 m², when the caller supplies it
in the station metadata as `drainage_area_km2`.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from . import _common as C

SPEC = C.Spec(
    name="streamflow", model_vars=["QOVER", "QDRAI"], units="mm/day",
    comparand="QOVER + QDRAI, point surface runoff plus subsurface drainage",
    obs_quantity="gauge discharge per unit contributing area",
    colocated=False, pair_on="distance")

COLUMN_AREA_M2 = 1.0


def compare(rows: List[Dict], series: Dict, meta: Dict, **kw) -> Dict[str, Any]:
    rec = C.standard_compare(SPEC, rows, series, meta)
    if rec.get("error"):
        return rec
    rec["scale"] = {
        "column_area_m2": COLUMN_AREA_M2,
        "note": ("a column is 1 m^2 and unrouted; each gauge's drainage area "
                 "is below where the caller supplied it. The ratio is the "
                 "reason these are not co-located.")}
    for e in rec["pairs"]:
        st = meta.get((e["station_id"], SPEC.name), {}) or {}
        area = st.get("drainage_area_km2")
        if area:
            e["drainage_area_km2"] = area
            # Orders of magnitude between the two footprints. NOTHING is scaled
            # by this — it is here so the mismatch is a number rather than an
            # adjective. A 500 km² basin against a 1 m² column is 8.7 orders.
            e["area_ratio_orders"] = round(
                math.log10(float(area) * 1e6 / COLUMN_AREA_M2), 1)
    return rec


def plot(rec: Dict, rows: List[Dict], series: Dict, out_path: str) -> Optional[str]:
    """Two panels: gauges and columns on one time axis, then the pairs.

    Deliberately NOT a 1:1 scatter alone. For a comparison that is about shape
    rather than magnitude, the time axis is the informative panel and the
    scatter is the caveat.
    """
    model = C.model_series(rows, SPEC.model_vars)
    got = list(C.assigned_pairs(rec, model, series, SPEC.name))
    if not got:
        return None
    import datetime as dt
    fig, (ax1, ax2) = C.new_figure()
    d = lambda s: dt.date.fromisoformat(s)                      # noqa: E731

    for i, m in enumerate(model.values()):
        ax1.plot([d(x) for x in m["dates"]], m["values"], lw=0.7,
                 color="#9AA7B0", alpha=0.75, zorder=1,
                 label="model columns" if i == 0 else None)
    for sid, case, dates, mm, oo, qq in got:
        ax1.plot([d(x) for x in dates], oo, lw=1.4, zorder=3, label=sid)
        ax2.scatter(oo, mm, s=16, alpha=0.7, zorder=3, label=f"{sid} · {case}")
    ax1.set_ylabel(f"specific discharge  [{SPEC.units}]")
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_title(f"{len(model)} columns, {rec['n_stations']} gauge(s) "
                  f"· not co-located")
    C.legend(ax1)

    C.one_to_one(ax2, [v for _, _, _, _, o, _ in got for v in o],
                 [v for _, _, _, m, _, _ in got for v in m])
    ax2.set_xlabel(f"gauge  [{SPEC.units}]")
    ax2.set_ylabel(f"column  [{SPEC.units}]")
    ax2.set_title("every pair — magnitude is not comparable")
    C.legend(ax2)
    return C.save(fig, out_path)
