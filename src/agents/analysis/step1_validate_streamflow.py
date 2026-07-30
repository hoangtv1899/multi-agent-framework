#!/usr/bin/env python3
"""
Analyzer step 1b — streamflow against USGS gauges
src/agents/analysis/step1_validate_streamflow.py

    in   ctx (model QOVER + QDRAI daily + USGS gauge daily specific discharge)
    out  {gauges, columns, caveats}  and the two spatial maps

STARTING WITH THE MAPS, deliberately. For SWE the map was the figure that
found the real problem — three of five stations outside the basin — while the
scatter and the hydrograph both looked fine. Streamflow has the same trap and
worse, so the question "are these two networks sampling the same place" comes
before any skill statistic.

The comparable quantity is SPECIFIC DISCHARGE, mm/day, not m3/s. A gauge
integrates a catchment; a column is 1 m2 with no routing and no run-on. USGS
mm_day is already discharge divided by contributing area, and QOVER + QDRAI is
local generation per unit area, so the units match even though the meanings do
not. That mismatch is a caveat, not something to be arithmetic'd away — the
run's own structural limitation already says columns provide no native
integrated streamflow.

What this module deliberately does NOT do yet:

  * pick an outlet gauge. On the 2019 Upper Gunnison run the largest in-basin
    gauge drains 10,285 km2 against a 6,234 km2 basin — 165%, water that was
    never simulated. No gauge drains the modelled domain.
  * attribute columns to a gauge's catchment. That needs flowlines, which the
    NLDI provided before it was removed. Without them every gauge would be
    compared against the whole ensemble, which is wrong for a headwater gauge
    draining 3% of the basin.

Both are real decisions, and a map is the right place to see them before
making them.
"""
from typing import Any, Dict, List, Optional, Tuple

from agents.analysis.step1_geo import (        # noqa: E402
    in_polygon, split_by_basin, _num)

# QOVER is surface runoff; QDRAI is subsurface drainage. A stream gauge sees
# both. Comparing QOVER alone understates the column by however much leaves
# below the surface — on col_19 of the Gunnison run that is 205 mm/yr against
# 1901 mm/yr, a factor of nine in the direction that looks like a dry bias.
MODEL_RUNOFF_VARS = ("QOVER", "QDRAI")


def _daily(row: Dict[str, Any], var: str) -> Dict[str, Any]:
    return ((row.get("variables") or {}).get(var) or {}).get("daily") or {}


def model_runoff(row: Dict[str, Any]) -> Dict[str, Any]:
    """One column's total daily runoff, mm/day, as {dates, values}.

    Summed across QOVER and QDRAI on matching dates. A date present in one
    variable and not the other is dropped rather than treated as zero: a
    missing series is not a series of zeros, and silently filling it would
    invent low-flow days.
    """
    parts = [_daily(row, v) for v in MODEL_RUNOFF_VARS]
    parts = [p for p in parts if p.get("dates") and p.get("values")]
    if not parts:
        return {}
    common: Optional[set] = None
    maps = []
    for p in parts:
        m = {d: _num(v) for d, v in zip(p["dates"], p["values"])
             if _num(v) is not None}
        maps.append(m)
        common = set(m) if common is None else (common & set(m))
    dates = sorted(common or [])
    return {"units": "mm/day", "dates": dates,
            "values": [sum(m[d] for m in maps) for d in dates]}


def gauge_daily(gauge: Dict[str, Any]) -> Dict[str, Any]:
    """A gauge's daily specific discharge as {dates, values}, mm/day.

    Reception stores it as mm_day: {date: value} — a dict, unlike the model's
    columnar {dates, values} and unlike SNOTEL's. Normalised here rather than
    at the call sites, so the shape lives in one place.
    """
    md = gauge.get("mm_day")
    if not isinstance(md, dict) or not md:
        # {} , not an empty series. "No record" and "a record of nothing" are
        # different, and model_runoff already returns {} for the same case —
        # two shapes for one condition is how a caller ends up handling only
        # the one it happened to see.
        return {}
    dates = sorted(md)
    return {"units": "mm/day", "dates": dates,
            "values": [_num(md[d]) for d in dates]}


def _mean(series: Dict[str, Any],
          window: Optional[Tuple[str, str]] = None) -> Optional[float]:
    vals = [(d, v) for d, v in zip(series.get("dates") or [],
                                   series.get("values") or [])
            if v is not None]
    if window:
        vals = [(d, v) for d, v in vals if window[0] <= d <= window[1]]
    return round(sum(v for _, v in vals) / len(vals), 4) if vals else None


def compare(ctx) -> Dict[str, Any]:
    """Mean specific discharge for every in-basin gauge and every column.

    Out-of-basin gauges are excluded before anything is computed, and the
    exclusion is named. Reception fetches by bounding box; on the 2019
    Gunnison run that pulled in 12 gauges from neighbouring basins —
    Uncompahgre, Taylor, East, Tomichi — whose flow says nothing about the
    watershed that was simulated.
    """
    rings = ctx.data.get("boundary") or []
    raw = ((ctx.data.get("observations") or {}).get("streamflow") or {}) \
        .get("stations") or []
    kept, outside = split_by_basin(raw, rings)

    caveats: List[Dict[str, Any]] = []
    if outside:
        caveats.append({
            "id": "gauges_outside_basin", "severity": "context",
            "statement": (f"{len(outside)} of {len(raw)} gauges were excluded "
                          f"for lying outside the watershed. Reception fetches "
                          f"by bounding box, which reaches into neighbouring "
                          f"basins: {', '.join(outside[:6])}"
                          + ("…" if len(outside) > 6 else "")),
            "applies_to": "streamflow coverage",
            "source": "step1_validate_streamflow"})

    # window: the overlap between the model record and the gauge records
    m_spans = [s for s in (model_runoff(r) for r in ctx.columns) if s.get("dates")]
    g_spans = [s for s in (gauge_daily(g) for g in kept) if s.get("dates")]
    window = None
    if m_spans and g_spans:
        lo = max(min(s["dates"][0] for s in m_spans),
                 min(s["dates"][0] for s in g_spans))
        hi = min(max(s["dates"][-1] for s in m_spans),
                 max(s["dates"][-1] for s in g_spans))
        if lo < hi:
            window = (lo, hi)

    basin_km2 = _basin_area_km2(rings)

    gauges = []
    for g in kept:
        ser = gauge_daily(g)
        area = _num(g.get("drainage_area_km2"))
        gauges.append({
            "id": g.get("id"), "name": g.get("name"),
            "lat": _num(g.get("lat")), "lon": _num(g.get("lon")),
            "drainage_area_km2": area,
            # The honesty metric. Below 100% the gauge is a partial sample of
            # the same terrain; above it, the gauge integrates water that was
            # never simulated and the comparison is not like-for-like.
            "area_fraction_of_basin": (round(area / basin_km2, 3)
                                       if area and basin_km2 else None),
            "mean_mm_day": _mean(ser, window),
            "n_days": len(ser.get("dates") or []),
            "series": ser,
        })

    # A gauge draining more than the basin is flagged BLOCKING: its flow
    # includes inflow from outside the modelled domain, so a bias against it
    # is not attributable to the model.
    over = [g["id"] for g in gauges
            if (g.get("area_fraction_of_basin") or 0) > 1.0]
    if over:
        caveats.append({
            "id": "gauge_exceeds_modelled_domain", "severity": "blocking",
            "statement": (f"{', '.join(over)} drain more area than the "
                          f"{basin_km2:.0f} km2 basin that was simulated, so "
                          f"their flow includes water never modelled. A bias "
                          f"against them is not attributable to the model."),
            "applies_to": "any skill claim against these gauges",
            "source": "step1_validate_streamflow"})

    columns = []
    for r in ctx.columns:
        ser = model_runoff(r)
        columns.append({
            "id": r.get("case_name"),
            "lat": _num(r.get("lat")), "lon": _num(r.get("lon")),
            "elevation_m": _num(r.get("elevation_m")),
            "mean_mm_day": _mean(ser, window),
            "n_days": len(ser.get("dates") or []),
            "series": ser,
        })

    caveats.append({
        "id": "unrouted_columns_vs_integrated_gauge", "severity": "blocking",
        "statement": ("A gauge integrates a catchment; a column is 1 m2 with "
                      "no routing and no run-on. Both are mm/day of specific "
                      "discharge, so the units match, but the model value is "
                      "local generation and the gauge value is routed "
                      "discharge. Timing especially cannot be compared "
                      "without routing."),
        "applies_to": "any hydrograph or timing claim",
        "source": "step1_validate_streamflow"})

    return {"window": window, "basin_area_km2": round(basin_km2, 1) if basin_km2 else None,
            "gauges": gauges, "columns": columns,
            "gauges_excluded_outside_basin": outside,
            "model_runoff_from": list(MODEL_RUNOFF_VARS),
            "caveats": caveats}


def _basin_area_km2(rings) -> Optional[float]:
    """Spherical polygon area. Used only to express a gauge's drainage area as
    a fraction of what was simulated, which is the honesty metric here."""
    import math
    if not rings:
        return None
    R, total = 6371.0, 0.0
    for ring in rings:
        try:
            s = 0.0
            for i in range(len(ring)):
                x1, y1 = ring[i][0], ring[i][1]
                x2, y2 = ring[(i + 1) % len(ring)][0], ring[(i + 1) % len(ring)][1]
                s += math.radians(x2 - x1) * (
                    2 + math.sin(math.radians(y1)) + math.sin(math.radians(y2)))
            total += abs(s * R * R / 2.0)
        except Exception:
            continue
    return total or None


def map_points(result, ctx=None):
    """Mean specific discharge as map points: (gauges, columns, gauge_sizes).

    Gauges are SIZED BY DRAINAGE AREA. A gauge draining 40 km2 and one
    draining 5,000 km2 are not equivalent evidence about a 6,245 km2 basin,
    and drawn at one size they look it. Returned here rather than computed at
    each call site so the standalone map and the combined grid size them
    identically.
    """
    gauges = [g for g in (result.get("gauges") or [])
              if g.get("lat") is not None and g.get("lon") is not None
              and g.get("mean_mm_day") is not None]
    cols = [c for c in (result.get("columns") or [])
            if c.get("lat") is not None and c.get("lon") is not None
            and c.get("mean_mm_day") is not None]
    obs = [(g["lon"], g["lat"], g["mean_mm_day"], g.get("id")) for g in gauges]
    mod = [(c["lon"], c["lat"], c["mean_mm_day"], c.get("id")) for c in cols]
    areas = [g.get("drainage_area_km2") or 0 for g in gauges]
    hi = max(areas) if areas else 0
    sizes = ([120 + 900 * (a / hi) ** 0.5 for a in areas] if hi else None)
    return obs, mod, sizes


def plot_timeseries(result: Dict[str, Any], out_path,
                log: bool = False) -> str:
    """Two panels side by side: every in-basin gauge, every ELM column.

    NOT paired, and not overlaid. The maps already established these two
    fields are not comparable point-for-point — a gauge integrates a
    catchment, a column is 1 m2 unrouted — so drawing them against each other
    on one axis would assert a correspondence that does not exist. Side by
    side each field is shown as it is, and the reader compares the SHAPES.

    ONE SHARED y-AXIS. The whole content of the figure is that one field has a
    coherent seasonal pulse and the other does not; independent axes would
    scale both to fill their panel and destroy exactly that.

    LINEAR y, SCALED TO THE OBSERVED RANGE. A log axis made every column's
    10^-5 flicker as visually loud as the gauges' snowmelt pulse, which is the
    opposite of the weight those two deserve. But a linear axis spanning the
    full data range is no better: col_19's 870 mm/day first-day warm-start
    spike would compress all seven gauges into a flat line at the bottom.

    So the axis is set by the GAUGES, and model values above it are counted
    rather than drawn. The count is stated on the panel, because an axis that
    silently hides data is worse than one that admits it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import datetime as _dt

    def _to_dates(ds):
        out = []
        for x in ds:
            try:
                out.append(_dt.date.fromisoformat(str(x)[:10]))
            except Exception:
                out.append(None)
        return out

    gauges = [g for g in (result.get("gauges") or []) if (g.get("series") or {}).get("dates")]
    cols   = [c for c in (result.get("columns") or []) if (c.get("series") or {}).get("dates")]

    obsv = [v for g in gauges for v in (g["series"].get("values") or [])
            if v is not None]
    allv = [v for grp in (gauges, cols) for it in grp
            for v in (it["series"].get("values") or []) if v is not None]
    hi = (max(obsv) * 1.15) if obsv else (max(allv) if allv else 1.0)
    pos = [v for v in allv if v > 0]
    lo = min(pos) if pos else 1e-4

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 6.6), sharey=True)
    for ax, items, title in ((axes[0], gauges, "USGS"),
                             (axes[1], cols, "ELM")):
        for it in items:
            ser = it["series"]
            xs = _to_dates(ser.get("dates") or [])
            ys = ser.get("values") or []
            # Exact zeros are BROKEN OUT of the line rather than drawn at the
            # axis floor. On a symlog axis a day of zero flow between two
            # positive days draws two full-height vertical strokes, and 9 of
            # 19 columns are zero on most days — the panel became a hatch
            # pattern that hid the series it was made of. A gap reads as what
            # a zero means here: no flow that day.
            pts = [(x, (y if (y is not None and y > 0) else None))
                   for x, y in zip(xs, ys) if x is not None]
            if any(q[1] is not None for q in pts):
                ax.plot([q[0] for q in pts], [q[1] for q in pts],
                        lw=1.5, alpha=0.85)
        if log:
            ax.set_yscale("symlog", linthresh=max(lo, 1e-5))
            ax.set_ylim(0, max(allv) * 1.3 if allv else 1.0)
        else:
            ax.set_ylim(0, hi)
            n_over = sum(1 for it in items
                         for v in (it["series"].get("values") or [])
                         if v is not None and v > hi)
            if n_over:
                ax.text(0.985, 0.86, f"{n_over} value(s) above axis",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=13, color="#b30000")
        ax.set_xlabel("date", fontsize=18)
        ax.tick_params(labelsize=13)
        ax.grid(alpha=0.25)
        for lb in ax.get_xticklabels():
            lb.set_rotation(30); lb.set_ha("right")
        ax.text(0.5, 0.97, f"{title}   (n={len(items)})",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=24, fontweight="bold", color="#111", zorder=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#333",
                          alpha=0.92))
    axes[0].set_ylabel("runoff (mm/day)", fontsize=18)

    fig.tight_layout()
    fig.savefig(out_path, dpi=135)
    plt.close(fig)
    return str(out_path)
