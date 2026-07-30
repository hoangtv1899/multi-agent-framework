#!/usr/bin/env python3
"""
Analyzer step 1c — water-table depth
src/agents/analysis/step1_validate_wtd.py

    in   ctx (model ZWT daily, Fan 2013 per column, USGS wells)
    out  {wells, fan, model, caveats}
         map_points      feeds the combined validation spatial map
         plot_timeseries modelled depth over the run, wells overlaid

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
    split_by_basin, _num)

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


def map_points(result: Dict[str, Any], ctx=None):
    """Map points for the three sources: (wells, fan, model, well_sizes).

    Wells are SIZED BY OBSERVATION COUNT. A well read 11 times in a year and
    one read twice are not equivalent evidence, and drawn at one size they
    look it. Returned here rather than computed at each call site so the
    standalone map and the combined grid size them identically.
    """
    def pts(items):
        return [(i["lon"], i["lat"], i["wtd_m"], i.get("id"))
                for i in items
                if i.get("lat") is not None and i.get("lon") is not None
                and i.get("wtd_m") is not None]

    wells = [w for w in (result.get("wells") or [])
             if w.get("lat") is not None and w.get("lon") is not None
             and w.get("wtd_m") is not None]
    counts = [(w.get("n_obs") or 1) for w in wells]
    hi = max(counts) if counts else 0
    sizes = ([140 + 700 * (c / hi) ** 0.5 for c in counts] if hi else None)
    return pts(wells), pts(result.get("fan") or []), \
        pts(result.get("model") or []), sizes


def _well_series(well: Dict[str, Any]):
    """A well's readings as [(date, depth_m)].

    Wells carry series as [{date, wtd_m}] — a list of records, unlike the
    model's columnar {dates, values} and unlike SNOTEL's. Normalised here so
    the shape lives in one place.
    """
    import datetime as _dt
    out = []
    for rec in (well.get("series") or []):
        if not isinstance(rec, dict):
            continue
        v = _num(rec.get("wtd_m"))
        if v is None:
            continue
        try:
            out.append((_dt.date.fromisoformat(str(rec.get("date"))[:10]), v))
        except Exception:
            pass
    return sorted(out)


SERIES_LOG_SPAN = 30.0        # see _series_yscale


def _series_yscale(result: Dict[str, Any]) -> str:
    """"log" or "linear" for the depth-series panel, from the data it draws.

    Wells at 5 m beside ELM at 70 m compress the measurements into a flat line
    against the axis on a linear scale, hiding the one series a reader most
    wants to see. But a log axis over a narrow range exaggerates noise — the
    same reason the streamflow hydrographs are linear — so the scale follows
    the span actually drawn rather than being fixed either way.
    """
    span = [v for m in (result.get("model") or [])
            for v in ((m.get("series") or {}).get("values") or [])
            if v is not None and v > 0]
    span += [v for w in (result.get("wells") or [])
             for _d, v in _well_series(w) if v > 0]
    if span and max(span) / min(span) > SERIES_LOG_SPAN:
        return "log"
    return "linear"


def plot_timeseries(result: Dict[str, Any], out_path) -> str:
    """Modelled water-table depth over the run, with measured wells on top.

    MEASURED WELLS ARE DRAWN WHEN THERE ARE ANY, as marked points rather than
    lines. Wells are read a handful of times a year, and a line joining two
    readings months apart asserts a trajectory that was never measured. They
    are the only actual measurement of this quantity, so a figure that omitted
    them when they existed would be showing the model alone and calling it
    validation. On the 2019 Upper Gunnison run all ten lie outside the
    watershed and none is drawn — the honest result for that basin rather than
    a missing feature.

    FAN 2013 IS ABSENT HERE BY CONSTRUCTION, not by omission: it is a static
    equilibrium field with no time dimension. It appears in the combined
    validation spatial map, which is where the Fan-vs-ELM comparison lives.

    LOG y ONLY when the fields drawn span more than SERIES_LOG_SPAN. Wells at
    5 m beside ELM at 70 m would compress the measurements into a flat line
    against the axis on a linear scale, hiding the one series a reader most
    wants to see; a log axis over a narrow range exaggerates noise instead,
    the same reason the streamflow hydrographs are linear. Depth increases
    downward either way.

    No soil-column reference line. It was drawn to say "below this the water
    table cannot reach the soil", which is true — SOILLIQ is flat at
    0.020 kg/m2 in every layer below 3.8 m — but it framed the deep values as a
    depth-scale mismatch when they are something else entirely: a diagnostic of
    NEGATIVE aquifer storage inherited from the CONUS restart. A line implying
    the numbers are water tables at an awkward depth reads as reassurance, and
    the figure looking plausible was the actual danger here.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import datetime as _dt

    wells = result.get("wells") or []

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    handles, labels = [], []

    drawn = 0
    for m in (result.get("model") or []):
        ser = m.get("series") or {}
        pts = []
        for d, v in zip(ser.get("dates") or [], ser.get("values") or []):
            if v is None:
                continue
            try:
                pts.append((_dt.date.fromisoformat(str(d)[:10]), v))
            except Exception:
                pass
        if pts:
            ln, = ax.plot([q[0] for q in pts], [q[1] for q in pts],
                          lw=1.6, alpha=0.85, color="#2c7fb8")
            drawn += 1
            if drawn == 1:
                handles.append(ln)

    n_wells = 0
    for w in wells:
        pts = _well_series(w)
        if not pts:
            continue
        h = ax.plot([q[0] for q in pts], [q[1] for q in pts],
                    linestyle="none", marker="o", markersize=9,
                    markerfacecolor="#31a354", markeredgecolor="#111",
                    markeredgewidth=1.4, zorder=5)[0]
        n_wells += 1
        if n_wells == 1:
            handles.append(h)

    if handles:
        labels = [f"ELM (n={drawn})"]
        if n_wells:
            labels.append(f"USGS wells (n={n_wells})")
        ax.legend(handles[:len(labels)], labels, fontsize=16,
                  loc="lower right", framealpha=0.92)

    ax.set_yscale(_series_yscale(result))   # see _series_yscale
    ax.invert_yaxis()          # depth increases downward
    ax.set_xlabel("date", fontsize=19)
    ax.set_ylabel("water-table depth (m)", fontsize=19)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.25)
    for lb in ax.get_xticklabels():
        lb.set_rotation(30); lb.set_ha("right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=135)
    plt.close(fig)
    return str(out_path)
