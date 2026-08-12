#!/usr/bin/env python3
"""Evapotranspiration: model QSOIL + QVEGE + QVEGT against a flux tower.

THE OBSERVABLE BEST MATCHED TO A 1-D COLUMN. ET is vertical, local to the tower
footprint (hundreds of metres, the same order as a column), and ELM computes it
directly. Where a stream gauge cannot be co-located with a column however it is
labelled, a flux tower can.

QSOIL + QVEGE + QVEGT, not QFLX_EVAP_TOT: this ELM build registers the
components — ground evaporation, canopy evaporation, transpiration — and does
not write the total. Summing the wrong subset silently under-reports ET.

WHAT THIS MODULE ADDS: metrics computed TWICE, once on the measured pairs alone
and once on everything. That is the US-NR1 lesson made operational rather than
written in a docstring. Measured 2026-08-10 at that tower: gap-filled annual ET
is 464 mm in 2016 against 89 mm from the measured half-hours alone, because 36%
of the year was observed. A single number cannot carry both, so both are here
and the caller can see how much of the agreement came from the gap-filling
model rather than from the instrument.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import _common as C

SPEC = C.Spec(
    name="et", model_vars=["QSOIL", "QVEGE", "QVEGT"], units="mm/day",
    comparand=("QSOIL + QVEGE + QVEGT — ground evaporation, canopy "
               "evaporation, transpiration. This ELM build registers the "
               "components, not a single QFLX_EVAP_TOT"),
    obs_quantity="eddy-covariance latent heat flux, expressed as water depth",
    colocated=True, pair_on="distance")


def compare(rows: List[Dict], series: Dict, meta: Dict, **kw) -> Dict[str, Any]:
    rec = C.standard_compare(SPEC, rows, series, meta,
                             domain=kw.get("domain"))
    if rec.get("error"):
        return rec
    model = C.model_series(rows, SPEC.model_vars)
    stations = C.stations_for(series, SPEC.name)

    for e in rec["pairs"]:
        obs = stations.get(e["station_id"]) or {}
        for col in e["columns"]:
            m = model.get(col["case_name"])
            if not (m and col.get("n_pairs")):
                continue
            _, mm, oo, qq = C.pair(m, obs)
            meas, _ = C.split_quality(qq)
            # The same comparison twice. If these two disagree, the difference
            # IS the finding — it says how much of the agreement is the tower's
            # and how much is its gap-filling model's.
            col["metrics_measured_only"] = (
                C.metrics([mm[i] for i in meas], [oo[i] for i in meas])
                if meas else {"n": 0,
                              "note": "no measured pairs — every point is filled"})
    return rec


def plot(rec: Dict, rows: List[Dict], series: Dict, out_path: str,
         **kw) -> Optional[str]:
    """Two panels: the flux series, and the pairs split by provenance.

    Gap-filled points are hollow in both. The whole reason this observable has
    its own plot is that the filled fraction is not a footnote here — at some
    towers it is most of the record.
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
        meas, fill = C.split_quality(qq)
        if meas:
            ax1.scatter([d(dates[i]) for i in meas], [oo[i] for i in meas],
                        s=14, zorder=3, label=f"{sid} (measured)")
            ax2.scatter([oo[i] for i in meas], [mm[i] for i in meas], s=16,
                        alpha=0.75, zorder=3, label=f"{sid} · {case}")
        if fill:
            ax1.scatter([d(dates[i]) for i in fill], [oo[i] for i in fill],
                        s=14, zorder=3, facecolors="none",
                        edgecolors="#C1440E", linewidths=0.7,
                        label=f"{sid} (gap-filled)")
            ax2.scatter([oo[i] for i in fill], [mm[i] for i in fill], s=16,
                        facecolors="none", edgecolors="#C1440E",
                        linewidths=0.7, zorder=3, label=f"{sid} (gap-filled)")
    ax1.set_ylabel(f"ET  [{SPEC.units}]")
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_title(f"{len(model)} columns, {rec['n_stations']} tower(s)")
    C.legend(ax1)

    C.one_to_one(ax2, [v for _, _, _, _, o, _ in got for v in o],
                 [v for _, _, _, m, _, _ in got for v in m])
    ax2.set_xlabel(f"tower  [{SPEC.units}]")
    ax2.set_ylabel(f"model  [{SPEC.units}]")
    ax2.set_title("every pair · hollow = gap-filled")
    C.legend(ax2)
    return C.save(fig, out_path)
