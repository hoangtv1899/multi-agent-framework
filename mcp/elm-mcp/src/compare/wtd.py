#!/usr/bin/env python3
"""Water table: model ZWT against wells, and against the priors.

THIS OBSERVABLE HAS MORE THAN TWO SOURCES, which is why it needs its own module
rather than a row in a table. A well is a measurement. Fan 2013 is an
observationally-constrained equilibrium surface. ParFlow-CLM is a simulated
steady state. They answer different questions and disagreeing with one is not
the same finding as disagreeing with another — so each is reported separately
and none is called "the truth".

The priors are static per column, not time series, so they are passed in as
`references={case_name: {"fan": 12.3, "parflow_clm": 10.1}}` rather than
squeezed into the observation table. The caller has them from the sampling
design; inventing a fake station per column to carry them would be a shape lie.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import _common as C

SPEC = C.Spec(
    name="wtd", model_vars=["ZWT"], units="m",
    comparand="ZWT, diagnosed water-table depth (positive down)",
    obs_quantity="depth to water in a well (positive down)",
    colocated=True, pair_on="distance")

# ELM's hydrologically active soil column. Below this, ZWT is diagnosed from an
# unconfined aquifer rather than simulated — a fact about where the model's
# answer came from, counted rather than averaged away.
ACTIVE_SOIL_DEPTH_M = 3.8


def compare(rows: List[Dict], series: Dict, meta: Dict,
            references: Optional[Dict[str, Dict[str, float]]] = None,
            **kw) -> Dict[str, Any]:
    rec = C.standard_compare(SPEC, rows, series, meta)
    if rec.get("error"):
        return rec
    model = C.model_series(rows, SPEC.model_vars)
    rec["active_soil_depth_m"] = ACTIVE_SOIL_DEPTH_M

    # How much of each comparison came from below the simulated soil column.
    stations = C.stations_for(series, SPEC.name)
    for e in rec["pairs"]:
        obs = stations.get(e["station_id"]) or {}
        for col in e["columns"]:
            m = model.get(col["case_name"])
            if not (m and col.get("n_pairs")):
                continue
            _, mm, _, _ = C.pair(m, obs)
            deep = sum(1 for v in mm if v is not None and v > ACTIVE_SOIL_DEPTH_M)
            col["frac_below_active_soil"] = round(deep / len(mm), 4)

    # The priors, per column, each differenced against the model's own mean.
    if references:
        refs = []
        for case, m in model.items():
            vals = [v for v in m["values"] if v is not None]
            if not vals:
                continue
            model_mean = sum(vals) / len(vals)
            row = {"case_name": case, "model_mean_m": round(model_mean, 3)}
            for label, val in (references.get(case) or {}).items():
                if val is None:
                    continue
                row[f"{label}_m"] = round(float(val), 3)
                row[f"model_minus_{label}_m"] = round(model_mean - float(val), 3)
            if len(row) > 2:
                refs.append(row)
        rec["references"] = {
            "per_column": refs,
            "note": ("static priors, differenced against each column's mean "
                     "ZWT. A well measures; Fan is an equilibrium surface fitted "
                     "to observations; ParFlow-CLM is a simulated steady state. "
                     "Disagreeing with one is not the same finding as "
                     "disagreeing with another.")}
    return rec


def _yscale(values: List[float]) -> str:
    """'log' when the depths span orders of magnitude, else 'linear'.

    A warm-started column can legitimately sit at 60 m while a valley well sits
    at 2 m. On a linear axis the shallow half of the basin is a flat line at the
    bottom of the panel and nothing about it can be read.
    """
    vs = [v for v in values if v is not None and v > 0]
    if len(vs) < 2:
        return "linear"
    return "log" if max(vs) / max(min(vs), 1e-6) > 50 else "linear"


def plot(rec: Dict, rows: List[Dict], series: Dict, out_path: str) -> Optional[str]:
    """Two panels: the depth series, and every pair against 1:1."""
    model = C.model_series(rows, SPEC.model_vars)
    got = list(C.assigned_pairs(rec, model, series, SPEC.name))
    if not got:
        return None
    import datetime as dt
    fig, (ax1, ax2) = C.new_figure()
    d = lambda s: dt.date.fromisoformat(s)                      # noqa: E731

    allv = []
    for i, m in enumerate(model.values()):
        ax1.plot([d(x) for x in m["dates"]], m["values"], lw=0.7,
                 color="#9AA7B0", alpha=0.75, zorder=1,
                 label="model columns" if i == 0 else None)
        allv += [v for v in m["values"] if v is not None]
    for sid, case, dates, mm, oo, qq in got:
        ax1.scatter([d(x) for x in dates], oo, s=14, zorder=3, label=sid)
        ax2.scatter(oo, mm, s=16, alpha=0.7, zorder=3, label=f"{sid} · {case}")
        allv += oo
    ax1.axhline(ACTIVE_SOIL_DEPTH_M, ls=":", lw=1.2, color="#A4522A", zorder=2,
                label=f"active soil {ACTIVE_SOIL_DEPTH_M} m")
    ax1.set_yscale(_yscale(allv))
    ax1.invert_yaxis()          # depth: down the page is deeper
    ax1.set_ylabel(f"depth to water  [{SPEC.units}]")
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_title(f"{len(model)} columns, {rec['n_stations']} well(s)")
    C.legend(ax1)

    C.one_to_one(ax2, [v for _, _, _, _, o, _ in got for v in o],
                 [v for _, _, _, m, _, _ in got for v in m])
    ax2.set_xlabel(f"observed  [{SPEC.units}]")
    ax2.set_ylabel(f"model  [{SPEC.units}]")
    ax2.set_title("every pair, assigned column")
    C.legend(ax2)
    return C.save(fig, out_path)
