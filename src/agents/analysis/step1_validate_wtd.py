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

# ELM's hydrologically active soil column. A water table below this cannot
# exchange water with the soil the model actually solves, which is why it is
# drawn on both panels rather than left to the caption.
SOIL_COLUMN_M = 3.8


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


def plot_distribution(result: Dict[str, Any], out_path) -> str:
    """Two panels: the depth DISTRIBUTIONS, then the modelled series.

    The maps show where each depth is; these show what the two fields are
    made of. Fan spans 0 to 251 m across nineteen columns while ELM occupies a
    much narrower band, and the histogram is where that compression is visible
    rather than inferred from colours.

    LOG x on the histogram, with bins spaced logarithmically. Depths run over
    three orders of magnitude and linear bins would put sixteen columns in the
    first bucket. Zeros are drawn as their own leftmost bar because a water
    table AT the surface is a distinct state, not a small number.

    The series panel has no observed counterpart to draw: on this basin every
    USGS well lies outside it, so it shows the model alone.

    No soil-column reference line. It was drawn to say "below this the water
    table cannot reach the soil", which is true — SOILLIQ is flat at
    0.020 kg/m2 in every layer below 3.8 m — but it framed the deep values as
    a depth-scale mismatch when they are something else entirely: a diagnostic
    of NEGATIVE aquifer storage. A line implying the numbers are water tables
    at an awkward depth reads as reassurance, and the figure looking plausible
    was the actual danger here.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import datetime as _dt
    import numpy as np

    fan = [_num(f.get("wtd_m")) for f in (result.get("fan") or [])]
    mod = [_num(m.get("wtd_m")) for m in (result.get("model") or [])]
    fan = [v for v in fan if v is not None]
    mod = [v for v in mod if v is not None]

    fig, axes = plt.subplots(1, 2, figsize=(17.0, 6.4))

    # ── (1) distributions ────────────────────────────────────────────────
    ax = axes[0]
    pos = [v for v in fan + mod if v > 0]
    if pos:
        lo, hi = min(pos), max(pos)
        bins = np.logspace(np.log10(lo * 0.8), np.log10(hi * 1.25), 14)
        for vals, colour, name in ((fan, "#e6550d", "Fan 2013"),
                                   (mod, "#2c7fb8", "ELM")):
            p = [v for v in vals if v > 0]
            if p:
                ax.hist(p, bins=bins, alpha=0.6, color=colour, label=name,
                        edgecolor="#222", linewidth=0.8)
        ax.set_xscale("log")
        # Zeros cannot sit on a log axis, and a water table at the surface is
        # a state of its own rather than a small depth. Counted separately.
        zf, zm = sum(1 for v in fan if v <= 0), sum(1 for v in mod if v <= 0)
        if zf or zm:
            ax.text(0.02, 0.97,
                    f"at surface (0 m):  Fan {zf}   ELM {zm}",
                    transform=ax.transAxes, va="top", fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#bbb", alpha=0.92))
    ax.set_xlabel("water-table depth (m below surface)", fontsize=18)
    ax.set_ylabel("columns", fontsize=18)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=15)
    ax.grid(alpha=0.25)

    # ── (2) modelled series ─────────────────────────────────────────────
    ax = axes[1]
    drawn = 0
    for m in (result.get("model") or []):
        ser = m.get("series") or {}
        ds, vs = ser.get("dates") or [], ser.get("values") or []
        pts = []
        for d, v in zip(ds, vs):
            if v is None:
                continue
            try:
                pts.append((_dt.date.fromisoformat(str(d)[:10]), v))
            except Exception:
                pass
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.6,
                    alpha=0.85)
            drawn += 1
    ax.invert_yaxis()          # depth increases downward
    ax.set_xlabel("date", fontsize=18)
    ax.set_ylabel("ELM water-table depth (m)", fontsize=18)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.25)
    for lb in ax.get_xticklabels():
        lb.set_rotation(30); lb.set_ha("right")
    ax.text(0.5, 0.04, f"ELM   (n={drawn})", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=22, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#333",
                      alpha=0.92))

    fig.tight_layout()
    fig.savefig(out_path, dpi=135)
    plt.close(fig)
    return str(out_path)
