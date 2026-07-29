#!/usr/bin/env python3
"""
Analyzer step 1c — water-table depth
src/agents/analysis/step1_validate_wtd.py

    in   ctx (model ZWT daily, Fan 2013 per column, USGS wells)
    out  {wells, fan, model, caveats}  and a 2- or 3-panel map

THREE SOURCES, AND ONLY TWO OF THEM ARE USUALLY AVAILABLE.

    USGS wells   measurements. Fetched by BBOX, so often none lie inside the
                 watershed — on the 2019 Upper Gunnison run, ALL TEN were
                 outside. Not 3 of 5 like SNOTEL, not 11 of 19 like the
                 gauges: every one.
    Fan 2013     modelled equilibrium depth, COLLOCATED WITH EVERY COLUMN by
                 construction. No station matching, no in-basin question. This
                 is what makes a WTD figure possible at all when the wells
                 fail, and it is a spatial prior rather than a measurement.
    ELM ZWT      the model's own diagnosed depth.

The panel count follows the data: two panels when no well lies in the basin,
three when one does. An empty third axis would read as "measured nothing"
rather than "nothing to measure".

WHY THIS IS CONTEXT AND NOT VALIDATION. ELM's hydrologically active soil
column is about 3.8 m deep; ZWT beyond that is diagnosed from an unconfined
aquifer store below the column, which is how it reports 70-75 m. Fan's median
for this basin is ~25 m, with values past 400 m. The two describe the same
quantity on scales that barely overlap, and a column whose water table sits
25 m down can never interact with a 3.8 m soil column.

That is not a footnote — it explains the rest of the run. Sixteen of nineteen
columns have a water table that moves less than 1 cm all year and produce
essentially no drainage. The two that DO produce credible snowmelt runoff,
col_19 and col_18, have Fan depths of 0.0 m and 4.4 m: the only ones shallow
enough to reach the soil column at all.

LOG COLOUR SCALE. Depths span 0.0 to 251 m across nineteen columns and the
Fan grid reaches 859 m in this window. Linear would put everything shallower
than ~50 m into one indistinguishable colour, and shallow is precisely where
the interesting behaviour is.
"""
from typing import Any, Dict, List, Optional

from agents.analysis.step1_geo import (        # noqa: E402
    split_by_basin, plot_panels, _num)


def _zwt_daily(row: Dict[str, Any]) -> Dict[str, Any]:
    return ((row.get("variables") or {}).get("ZWT") or {}).get("daily") or {}


def compare(ctx) -> Dict[str, Any]:
    """Wells (in-basin only), Fan per column, and modelled ZWT per column."""
    rings = ctx.data.get("boundary") or []
    raw = ((ctx.data.get("observations") or {}).get("water_table") or {}) \
        .get("wells") or []
    kept, outside = split_by_basin(raw, rings)

    caveats: List[Dict[str, Any]] = []
    if outside:
        caveats.append({
            "id": "wells_outside_basin",
            "severity": "blocking" if not kept else "context",
            "statement": (f"{len(outside)} of {len(raw)} USGS wells lie outside "
                          f"the watershed"
                          + (". NO well remains inside it, so there is no "
                             "measured water-table depth for this basin and "
                             "the comparison rests entirely on the Fan 2013 "
                             "modelled prior."
                             if not kept else
                             f": {', '.join(outside[:5])}")),
            "applies_to": "any measured water-table claim",
            "source": "step1_validate_wtd"})

    wells = [{"id": w.get("id"), "name": w.get("name"),
              "lat": _num(w.get("lat")), "lon": _num(w.get("lon")),
              "wtd_m": _num(w.get("wtd_m")),
              "min_depth_m": _num(w.get("min_depth_m")),
              "max_depth_m": _num(w.get("max_depth_m")),
              "n_obs": w.get("n_obs"),
              "series": w.get("series") or []}
             for w in kept]

    fan, model = [], []
    static = 0
    for r in ctx.columns:
        lat, lon = _num(r.get("lat")), _num(r.get("lon"))
        f = _num(r.get("fan_wtd_m"))
        if f is not None:
            fan.append({"id": r.get("case_name"), "lat": lat, "lon": lon,
                        "elevation_m": _num(r.get("elevation_m")),
                        "wtd_m": f})
        d = _zwt_daily(r)
        v = [x for x in (d.get("values") or []) if x is not None]
        if v:
            rng = max(v) - min(v)
            if rng < 0.01:
                static += 1
            model.append({"id": r.get("case_name"), "lat": lat, "lon": lon,
                          "elevation_m": _num(r.get("elevation_m")),
                          "wtd_m": round(sum(v) / len(v), 3),
                          "min_m": round(min(v), 3), "max_m": round(max(v), 3),
                          "annual_range_m": round(rng, 4),
                          "series": {"units": d.get("units"),
                                     "dates": d.get("dates"),
                                     "values": d.get("values")}})

    # The finding that explains the rest of the run, stated as a caveat because
    # it BOUNDS what the runoff results can mean rather than being one of them.
    if model:
        caveats.append({
            "id": "water_table_below_soil_column", "severity": "blocking",
            "statement": (f"ELM's hydrologically active soil column is ~3.8 m "
                          f"deep; ZWT beyond that is diagnosed from an aquifer "
                          f"store below it. {static} of {len(model)} columns "
                          f"have a water table that moves less than 1 cm all "
                          f"year, so their soil column and their water table "
                          f"never interact. No drainage claim about those "
                          f"columns is a statement about groundwater."),
            "applies_to": "recharge, subsurface drainage and any WTD claim",
            "source": "step1_validate_wtd"})

    return {"wells": wells, "fan": fan, "model": model,
            "wells_excluded_outside_basin": outside,
            "n_static_columns": static,
            "has_measured_wtd": bool(wells),
            "caveats": caveats}


def plot_maps(result: Dict[str, Any], ctx, out_path, **kw) -> str:
    """Two panels when no well lies in the basin, three when one does.

    Wells are sized by their observation count — a well read 11 times in a
    year and one read twice are not equivalent evidence, and drawn at one size
    they look it.
    """
    def pts(items):
        return [(i["lon"], i["lat"], i["wtd_m"], i["id"])
                for i in items
                if i.get("lat") is not None and i.get("lon") is not None
                and i.get("wtd_m") is not None]

    panels = []
    wells = result.get("wells") or []
    if wells:
        counts = [(w.get("n_obs") or 1) for w in wells]
        hi = max(counts) or 1
        panels.append(("USGS wells", pts(wells),
                       [140 + 700 * (c / hi) ** 0.5 for c in counts]))
    panels.append(("Fan 2013", pts(result.get("fan") or []), None))
    panels.append(("ELM", pts(result.get("model") or []), None))

    kw.setdefault("log", True)
    return plot_panels(panels, ctx.data.get("boundary") or [], out_path,
                       label="water-table depth (m below surface)", **kw)
