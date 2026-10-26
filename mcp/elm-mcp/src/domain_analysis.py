#!/usr/bin/env python3
"""One whole-domain experiment's numbers, by region: every run, every year.

The analysis half of docs/ELM_DOMAIN_EXPERIMENTS.md. A plan names the regions
the answer is reported over, the history fields it turns on and, for a
question about an event, the normal years; this module makes those numbers
from each run's own history, with gridded.py and regions.py, so their
conventions (records labelled by the start of their interval, rates per
second put per day, area x landfrac weights) are the ones every number here
carries.

  masks(...)         the regions a plan names: "vegetated cells", "elevation
                     bands", and any region registered for the control (a
                     polygon, or a gridded field), each restricted to the
                     vegetated cells, the land the carbon cycle runs on
  run_values(...)    per region, field and year: a rate's annual total, a
                     state's annual mean, with units
  differences(...)   each run minus the control, region, field and year
  event(...)         each year of a run against the normal years
  observed_burned(...)  the control's registered burned-area observation,
                     per region and year, beside the model's own
  LIMITATIONS        what a regional number of a whole-domain run cannot
                     support, for the experiment's package (the framework
                     turns them into the caveats that bind the Analyzer)

Numbers only: what they mean is the Analyzer's and the report's business.
A region, a field or a year that cannot be made is listed in `problems`
with the reason, never dropped in silence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import gridded as G
import regions as R

VEGETATED = "vegetated cells"
BANDS = "elevation bands"
BAND_EDGES = (1000.0, 1500.0, 2000.0)
# A field read from the first stream that holds it, monthly first: a year's
# total or mean is the same from any cadence (regions.RegionSeries.by_year),
# and the monthly files are read fastest.
STREAM_ORDER = ("h0", "h1", "h2")
# The model's own burned fraction, put beside a burned-area observation. A
# rate per second despite its label (gridded.FIELD_NOTES), so a year's value is
# the fraction of the region burned that year, and that times the region's
# area is the area burned.
BURNED_FRACTION = "FAREA_BURNED"

# What a regional number of a whole-domain run cannot support, in the shape
# core.limitations.select_limitations returns, so step 0 of the Analyzer reads
# them as it reads a site run's. ELM knowledge, so it lives here.
LIMITATIONS = {
    "structural": [
        {"applies_to": "any comparison of runoff (QRUNOFF, QOVER, QDRAI) with a stream gauge",
         "caveat": "a regional number is the area-weighted mean, or total, of the grid cells "
                   "in the region, and ELM's grid cells exchange no water with each other: "
                   "regional runoff is what the cells generate, not streamflow routed to a "
                   "gauge.",
         "kind": "structural", "severity": "blocking"},
    ],
    "configuration": [
        {"applies_to": "differences between runs",
         "caveat": "every run starts from the control's starting state and reads the "
                   "control's recorded weather; a run with weather_scale receives that "
                   "weather with its own change applied, as that change's meaning says; the "
                   "case stage checked that each run's settings differ from the control's "
                   "only by its named changes.",
         "kind": "configuration", "severity": "context"},
    ],
}


class AnalysisError(Exception):
    """An analysis cannot be made, and the message says why."""


def _surface(case_dir) -> str:
    from run_gate import case_namelist
    nl, _ = case_namelist(case_dir)
    s = (nl.get("fsurdat") or [None])[0]
    if not s:
        raise AnalysisError(f"{case_dir}: its lnd_in names no fsurdat")
    return str(s)


def masks(grid: G.Grid, surface, names: Sequence[str],
          registered: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
    """{"masks": MaskSet, "problems": [...]} for the regions a plan names.

    Every region is restricted to the vegetated cells (PCT_NATVEG at 100), so
    one rule holds for every number; a registered region keeps its own name,
    with ", vegetated" added."""
    s = R.read_surface(surface, grid, ("PCT_NATVEG", "TOPO"))
    veg = R.vegetated(s["PCT_NATVEG"])
    ms = R.MaskSet(grid)
    problems: List[str] = []
    src = [Path(surface).name]
    ms.add(VEGETATED, veg, "PCT_NATVEG >= 100", src)
    for n in names:
        key = str(n).strip()
        if key.lower() in (VEGETATED, "vegetated"):
            continue
        if key.lower() in (BANDS, "elevation_bands", "bands"):
            for b, m in R.elevation_bands(s["TOPO"], BAND_EDGES).items():
                if (veg & m).any():
                    ms.add(b, veg & m, f"vegetated, TOPO {b}", src)
            continue
        r = (registered or {}).get(key)
        if not r:
            problems.append(f"region {key!r} is neither a standard one ({VEGETATED}, "
                            f"{BANDS}) nor registered for the control "
                            f"({sorted(registered or {})})")
            continue
        try:
            if r.get("polygon"):
                m, info = R.polygon_mask(grid, r["polygon"])
                rule, files = "cell centre inside the polygon", [Path(r["polygon"]).name]
            elif r.get("field"):
                m = R.mask_from_field(r["field"], r["var"], grid, r.get("op", ">"),
                                      float(r.get("value", 0.0)), over=r.get("over", "any"))
                rule = f"{r['var']} {r.get('op', '>')} {r.get('value', 0.0)} ({r.get('over', 'any')})"
                files = [Path(r["field"]).name]
            else:
                raise AnalysisError("names neither a polygon nor a field")
            m = np.asarray(m, dtype=bool) & veg
            if not m.any():
                problems.append(f"region {key!r} has no vegetated cell")
                continue
            ms.add(f"{key}, vegetated", m, f"{rule}, vegetated", files + src)
        except Exception as e:                                      # noqa: BLE001
            problems.append(f"region {key!r} cannot be made: {type(e).__name__}: {e}")
    return {"masks": ms, "problems": problems}


def _stream_for(streams: Dict[str, G.Stream], var: str) -> Optional[str]:
    return next((s for s in STREAM_ORDER if s in streams and var in streams[s].variables),
                None)


def run_values(run_dir, ms: R.MaskSet, variables: Sequence[str], years: Sequence[int]
               ) -> Dict[str, Any]:
    """Per region, field and year, from one run's history: a rate per second
    is totalled over the year (regions' sum of value x days), anything else is
    the year's time-weighted mean. A year that is not whole is None, with the
    reason."""
    have = G.discover(run_dir)
    streams = {s: G.open_stream(run_dir, s) for s in STREAM_ORDER if s in have}
    out: Dict[str, Any] = {"run_dir": str(run_dir), "fields": {}, "values": {},
                           "missing": {}, "problems": []}
    by_stream: Dict[str, List[str]] = {}
    for v in variables:
        st = _stream_for(streams, v)
        if st is None:
            out["problems"].append(f"{v} is in none of the run's streams "
                                   f"({sorted(streams)})")
            continue
        by_stream.setdefault(st, []).append(v)
    for st, vs in by_stream.items():
        s = streams[st]
        ys = [y for y in years if y in s.complete_years()]
        for y in sorted(set(years) - set(ys)):
            out["problems"].append(f"{st} has no complete {y}: its fields have no "
                                   f"value that year")
        if not ys:
            continue
        rs = R.reduce_regions(s, [R.Field(v) for v in vs], ms, ys)
        for v in vs:
            rate = s.per_second(v)
            how = "sum" if rate else "mean"
            units = rs.units[v]
            out["fields"][v] = {"stream": st, "how": how,
                                "units": (units.replace("/d", "/yr") if rate else units),
                                "note": G.FIELD_NOTES.get(v)}
            for reg in ms.names():
                vals, why = rs.by_year(reg, v, ys, how=how)
                out["values"].setdefault(reg, {})[v] = {
                    str(y): (None if vals.get(y) is None else float(vals[y])) for y in ys}
                if why:
                    out["missing"].setdefault(reg, {})[v] = {str(y): w for y, w in why.items()}
    return out


def differences(runs: Dict[str, Dict], control: str = "control") -> Dict[str, Any]:
    """Each run minus the control, for every region, field and year both have."""
    c = (runs.get(control) or {}).get("values") or {}
    out: Dict[str, Any] = {}
    for name, r in runs.items():
        if name == control:
            continue
        for reg, fields in (r.get("values") or {}).items():
            for v, years in fields.items():
                for y, x in years.items():
                    cy = ((c.get(reg) or {}).get(v) or {}).get(y)
                    if x is None or cy is None:
                        continue
                    out.setdefault(name, {}).setdefault(reg, {}).setdefault(v, {})[y] = \
                        x - cy
    return out


def event(run: Dict[str, Any], normal_years: Sequence[int], target_years: Sequence[int]
          ) -> Dict[str, Any]:
    """Each target year of one run against its normal years, every region and
    field (regions.anomalies: the normal, the anomaly, its percent, the rank)."""
    out: Dict[str, Any] = {}
    problems: List[str] = []
    for reg, fields in (run.get("values") or {}).items():
        for v, years in fields.items():
            by = {int(y): x for y, x in years.items()}
            try:
                out.setdefault(reg, {})[v] = R.anomalies(
                    by, list(normal_years), list(target_years),
                    percent=not str((run.get("fields") or {}).get(v, {}).get("units", ""))
                    .strip().upper().startswith("K"))
            except R.RegionError as e:
                problems.append(f"{reg}, {v}: {e}")
    return {"anomalies": out, "problems": problems}


def observed_burned(path, var: str, ms: R.MaskSet, years: Sequence[int]) -> Dict[str, Any]:
    """An observed burned fraction per cell and year on the model grid, as
    km2 burned per region and year (fraction x area x landfrac)."""
    import netCDF4
    from gridded import read_clean
    with netCDF4.Dataset(path) as f:
        ms.grid.check_coordinates(read_clean(f["lat"]), read_clean(f["lon"]),
                                  what=Path(path).name)
        ys = [int(y) for y in f["year"][:]]
        bf = read_clean(f[var])
    w = ms.grid.weight_km2
    out: Dict[str, Dict[str, float]] = {}
    for reg in ms.names():
        m = ms[reg]
        for k, y in enumerate(ys):
            if y in years:
                out.setdefault(reg, {})[str(y)] = float(np.nansum(np.nan_to_num(bf[k])[m] *
                                                                  w[m]))
    return out


def experiment(control: Dict[str, Any], runs: Dict[str, str], analysis: Dict[str, Any],
               years: Sequence[int]) -> Dict[str, Any]:
    """The whole table of one experiment.

    control   {"case_dir", "regions": {...registered...}, "observations": {...}}
    runs      {run name: run directory}, the control's under "control"
    analysis  the plan's: regions (names, or {"name"} objects), variables,
              normal_years
    """
    names = [r.get("name") if isinstance(r, dict) else r
             for r in (analysis.get("regions") or [])] or [VEGETATED, BANDS]
    variables = list(analysis.get("variables") or [])
    if not variables:
        raise AnalysisError("the plan's analysis names no variables")
    surface = _surface(control["case_dir"])
    first = next(iter(runs.values()))
    grid = G.open_stream(first, next(s for s in STREAM_ORDER if s in G.discover(first))).grid
    mk = masks(grid, surface, names, control.get("regions"))
    ms = mk["masks"]
    table: Dict[str, Any] = {
        "regions": {n: {"cells": ms.mask(n).n_cells, "area_km2": float(ms.mask(n).area_km2),
                        "rule": ms.mask(n).rule} for n in ms.names()},
        "years": [int(y) for y in years], "runs": {}, "problems": list(mk["problems"])}
    for name, rd in runs.items():
        try:
            table["runs"][name] = run_values(rd, ms, variables, [int(y) for y in years])
        except Exception as e:                                      # noqa: BLE001
            table["problems"].append(f"run {name}: {type(e).__name__}: {e}")
    table["differences"] = differences(table["runs"])
    normal = [int(y) for y in (analysis.get("normal_years") or [])]
    if normal and "control" in table["runs"]:
        target = [int(y) for y in years if int(y) not in normal]
        ev = event(table["runs"]["control"], normal, target)
        table["event"] = {"normal_years": normal, "target_years": target, **ev}
    for on, o in (control.get("observations") or {}).items():
        if o.get("var") == "burned_fraction_year":
            try:
                table.setdefault("observed", {})["burned_area_km2"] = {
                    "what": "burned area", "units": "km2", "source": o.get("what") or on,
                    "values": observed_burned(o["path"], o["var"], ms, [int(y) for y in years]),
                    "model_field": BURNED_FRACTION,
                    "model": model_burned(table, runs, ms, [int(y) for y in years],
                                          variables)}
            except Exception as e:                                  # noqa: BLE001
                table["problems"].append(f"observation {on}: {type(e).__name__}: {e}")
    return table


def model_burned(table: Dict[str, Any], runs: Dict[str, str], ms: R.MaskSet,
                 years: Sequence[int], variables: Sequence[str] = ()) -> Dict[str, Any]:
    """Each run's own burned area, km2 per region and year: the region's mean
    burned fraction for the year (BURNED_FRACTION, read for the analysis or
    here) times the region's area. A run without the field is a problem of the
    table, not a zero."""
    out: Dict[str, Any] = {}
    for name, rd in runs.items():
        if name not in table["runs"]:
            continue                       # the run itself could not be read
        vals = table["runs"][name].get("values") or {}
        if BURNED_FRACTION not in variables:
            try:
                rv = run_values(rd, ms, [BURNED_FRACTION], years)
            except Exception as e:                                  # noqa: BLE001
                table["problems"].append(f"run {name}, {BURNED_FRACTION}: "
                                         f"{type(e).__name__}: {e}")
                continue
            vals = rv["values"]
            table["problems"] += [f"run {name}: {q}" for q in rv["problems"]]
        for reg, info in table["regions"].items():
            by = (vals.get(reg) or {}).get(BURNED_FRACTION) or {}
            if by:
                out.setdefault(name, {})[reg] = {
                    y: (None if x is None else float(x) * info["area_km2"])
                    for y, x in by.items()}
    return out
