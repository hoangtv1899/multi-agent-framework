#!/usr/bin/env python3
"""SWE: model H2OSNO against a snow pillow.

What this adds over the shared comparison, and why generic metrics are not
enough here: a thin pack lasting months and a deep one melting fast can produce
the same mean. Peak agreeing while duration disagrees is a different finding
from both disagreeing, and bias/RMSE cannot express either. So phenology —
peak, peak date, first snow, melt-out, days above threshold — is reported for
both sides, as numbers, for the caller to difference.

Pairing is on ELEVATION, not distance. See _common.pair_stations.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from . import _common as C

SPEC = C.Spec(
    name="swe", model_vars=["H2OSNO"], units="mm",
    comparand="H2OSNO, snow water equivalent on the column",
    obs_quantity="snow water equivalent at a snow pillow",
    colocated=True, pair_on="elevation")

THRESHOLD_MM = 25.0     # ~1 inch SWE: a pack, not a dusting


def phenology(dates: List[str], values: List[float],
              threshold: float = THRESHOLD_MM) -> Dict[str, Any]:
    """WHEN the snowpack happened, not only how much of it there was."""
    pairs = [(d, v) for d, v in zip(dates or [], values or [])
             if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not pairs:
        return {"available": False, "reason": "no SWE series"}
    ds = [d for d, _ in pairs]
    vs = [v for _, v in pairs]
    peak = max(vs)
    above = [i for i, v in enumerate(vs) if v > threshold]
    out = {"available": True,
           "peak_swe_mm": round(peak, 1),
           "mean_swe_mm": round(sum(vs) / len(vs), 1),
           "peak_date": ds[vs.index(peak)],
           "peak_dowy": day_of_water_year(ds[vs.index(peak)]),
           "threshold_mm": threshold,
           "days_above_threshold": len(above),
           "has_snowpack": bool(above)}
    if above:
        out["first_snow_date"] = ds[above[0]]
        # Melt-out is the LAST day above threshold, not the first day below: a
        # mid-winter thaw dipping under the line for a week would otherwise end
        # the season in January.
        out["melt_out_date"] = ds[above[-1]]
    return out


def day_of_water_year(d: Optional[str]) -> Optional[int]:
    """Oct 1 = 1. Snow years do not respect January.

    A calendar day-of-year puts a 15 December peak at 349 and a 5 January peak
    at 5 — adjacent events 344 apart, which makes the peak-date panel unreadable
    and any mean of it meaningless.
    """
    if not d:
        return None
    import datetime as dt
    try:
        day = dt.date.fromisoformat(d[:10])
    except ValueError:
        return None
    start = dt.date(day.year - (1 if day.month < 10 else 0), 10, 1)
    return (day - start).days + 1


def water_year_window(model: Dict, stations: Dict) -> Optional[List[str]]:
    """The window both sides actually cover.

    SNOTEL is reported by water year and the model runs a calendar year, so the
    two spans routinely differ at both ends. Metrics over the union would count
    months where one side has nothing.
    """
    m_dates = sorted({d for m in model.values() for d in m["dates"]})
    o_dates = sorted({d for s in stations.values() for d in s["dates"]})
    if not (m_dates and o_dates):
        return None
    lo, hi = max(m_dates[0], o_dates[0]), min(m_dates[-1], o_dates[-1])
    return [lo, hi] if lo <= hi else None


def compare(rows: List[Dict], series: Dict, meta: Dict, **kw) -> Dict[str, Any]:
    rec = C.standard_compare(SPEC, rows, series, meta)
    if rec.get("error"):
        return rec
    model = C.model_series(rows, SPEC.model_vars)
    stations = C.stations_for(series, SPEC.name)
    rec["shared_window"] = water_year_window(model, stations)
    rec["threshold_mm"] = THRESHOLD_MM
    for e in rec["pairs"]:
        obs = stations.get(e["station_id"]) or {}
        e["phenology_obs"] = phenology(obs.get("dates"), obs.get("values"))
        m = model.get(e.get("assigned_column"))
        if m:
            e["phenology_model"] = phenology(m["dates"], m["values"])
    return rec


def plot(rec: Dict, rows: List[Dict], series: Dict, out_path: str) -> Optional[str]:
    """Three panels: the series, every pair, and peak timing.

    The third panel is the SWE-specific one. Peak date on a day-of-water-year
    axis asks whether the model melts at the right time, which the other two
    cannot see at all.
    """
    model = C.model_series(rows, SPEC.model_vars)
    got = list(C.assigned_pairs(rec, model, series, SPEC.name))
    if not got:
        return None
    import datetime as dt
    fig, (ax1, ax2, ax3) = C.new_figure(ncols=3, width=16.0)
    d = lambda s: dt.date.fromisoformat(s)                      # noqa: E731

    for i, m in enumerate(model.values()):
        ax1.plot([d(x) for x in m["dates"]], m["values"], lw=0.7,
                 color="#9AA7B0", alpha=0.75, zorder=1,
                 label="model columns" if i == 0 else None)
    for sid, case, dates, mm, oo, qq in got:
        meas, fill = C.split_quality(qq)
        if meas:
            ax1.scatter([d(dates[i]) for i in meas], [oo[i] for i in meas],
                        s=14, zorder=3, label=f"{sid}")
        if fill:
            ax1.scatter([d(dates[i]) for i in fill], [oo[i] for i in fill],
                        s=14, zorder=3, facecolors="none",
                        edgecolors="#C1440E", linewidths=0.7,
                        label=f"{sid} (gap-filled)")
        ax2.scatter(oo, mm, s=16, alpha=0.7, zorder=3, label=f"{sid} · {case}")
    ax1.set_ylabel(f"SWE  [{SPEC.units}]")
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_title(f"{len(model)} columns, {rec['n_stations']} station(s)")
    C.legend(ax1)

    C.one_to_one(ax2, [v for _, _, _, _, o, _ in got for v in o],
                 [v for _, _, _, m, _, _ in got for v in m])
    ax2.set_xlabel(f"observed  [{SPEC.units}]")
    ax2.set_ylabel(f"model  [{SPEC.units}]")
    ax2.set_title("every pair, assigned column")
    C.legend(ax2)

    xs, ys, labels = [], [], []
    for e in rec.get("pairs") or []:
        po, pm = e.get("phenology_obs") or {}, e.get("phenology_model") or {}
        if po.get("peak_dowy") and pm.get("peak_dowy"):
            xs.append(po["peak_dowy"])
            ys.append(pm["peak_dowy"])
            labels.append(e["station_id"])
    if xs:
        for x, y, lab in zip(xs, ys, labels):
            ax3.scatter([x], [y], s=64, zorder=3, label=lab)
        C.one_to_one(ax3, xs, ys)
        C.legend(ax3)
    ax3.set_xlabel("observed peak (day of water year)")
    ax3.set_ylabel("model peak (day of water year)")
    ax3.set_title("peak timing")
    return C.save(fig, out_path)
