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
                     bands" (over the domain, or within a registered
                     region), and any region registered for the control (a
                     polygon, or a gridded field), each restricted to the
                     vegetated cells, the land the carbon cycle runs on
  run_values(...)    per region, field and year: a rate's annual total, a
                     state's annual mean, with units; and the same per month
                     (2015-06) and season (2015-JJA, DJF with the December
                     before it), with the files each number was read from
  differences(...)   each run minus the control, region, field and year
  event(...)         each year of a run against the normal years
  burned_area_field(...)  each run's own burned area, km2, as a field of its
                     values (the model's output only: no observation)
  LIMITATIONS        what a regional number of a whole-domain run cannot
                     support, for the experiment's package (the framework
                     turns them into the caveats that bind the Analyzer)

Numbers only: what they mean is the Analyzer's and the report's business.
A region, a field or a year that cannot be made is listed in `problems`
with the reason, never dropped in silence.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
                   "control's recorded weather, except as its weather change says: with "
                   "weather_scale, that weather with its change applied; with "
                   "weather_repeat_year, the mapped model years read another recorded "
                   "year's weather, set by each year's job and checked in that year. The "
                   "case stage checked that each run's settings at build differ from the "
                   "control's only by its named changes.",
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


def region_spec(n) -> Tuple[str, Optional[str]]:
    """A plan's region as (name, within): a name, or {"name", "within"}, where
    within names the registered region elevation bands are made inside."""
    if isinstance(n, dict):
        w = n.get("within")
        return str(n.get("name") or "").strip(), (str(w).strip() if w else None)
    return str(n).strip(), None


def plan_regions(analysis: Dict[str, Any]) -> List[Any]:
    """The plan's regions, or the default when it names none.

    ONE DEFAULT (2026-11-03): the table, the design page and the server's
    check each filled an empty list on their own, and the check filled it
    with nothing, so a run changed in one region passed a plan that named no
    regions and then got whole-domain regions that average changed and
    unchanged cells."""
    return list((analysis or {}).get("regions") or []) or [VEGETATED, BANDS]


def is_bands(name: str) -> bool:
    return str(name).strip().lower() in (BANDS, "elevation_bands", "bands")


def band_within(within: str, band: str) -> str:
    """The table's name of one elevation band inside a registered region."""
    return f"{within}, vegetated, {band}"


def _registered_mask(grid: G.Grid, r: Dict[str, Any]):
    """(mask, rule, files) of one registered region, before the vegetated rule."""
    if r.get("polygon"):
        m, _ = R.polygon_mask(grid, r["polygon"])
        return m, "cell centre inside the polygon", [Path(r["polygon"]).name]
    if r.get("field"):
        m = R.mask_from_field(r["field"], r["var"], grid, r.get("op", ">"),
                              float(r.get("value", 0.0)), over=r.get("over", "any"))
        rule = f"{r['var']} {r.get('op', '>')} {r.get('value', 0.0)} ({r.get('over', 'any')})"
        return m, rule, [Path(r["field"]).name]
    raise AnalysisError("names neither a polygon nor a field")


def masks(grid: G.Grid, surface, names: Sequence[Any],
          registered: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
    """{"masks": MaskSet, "problems": [...]} for the regions a plan names.

    Every region is restricted to the vegetated cells (PCT_NATVEG at 100), so
    one rule holds for every number; a registered region keeps its own name,
    with ", vegetated" added. Elevation bands cover the whole domain, or, with
    {"name": "elevation bands", "within": <a registered region>}, only that
    region's cells, named band_within(region, band)."""
    s = R.read_surface(surface, grid, ("PCT_NATVEG", "TOPO"))
    veg = R.vegetated(s["PCT_NATVEG"])
    ms = R.MaskSet(grid)
    problems: List[str] = []
    src = [Path(surface).name]
    ms.add(VEGETATED, veg, "PCT_NATVEG >= 100", src)
    seen = set()
    for n in names:
        key, within = region_spec(n)
        if (key.lower(), within) in seen:
            continue
        seen.add((key.lower(), within))
        if within and not is_bands(key):
            problems.append(f"region {key!r}: within applies to {BANDS} only")
            continue
        if key.lower() in (VEGETATED, "vegetated"):
            continue
        if is_bands(key) and not within:
            for b, m in R.elevation_bands(s["TOPO"], BAND_EDGES).items():
                if (veg & m).any():
                    ms.add(b, veg & m, f"vegetated, TOPO {b}", src)
            continue
        target = within or key
        r = (registered or {}).get(target)
        if not r:
            problems.append(f"region {target!r} is neither a standard one ({VEGETATED}, "
                            f"{BANDS}) nor registered for the control "
                            f"({sorted(registered or {})})"
                            + (f", so no {BANDS} can be made within it" if within else ""))
            continue
        try:
            m, rule, files = _registered_mask(grid, r)
            m = np.asarray(m, dtype=bool) & veg
            if not m.any():
                problems.append(f"region {target!r} has no vegetated cell")
                continue
            if within:
                for b, bm in R.elevation_bands(s["TOPO"], BAND_EDGES).items():
                    if (m & bm).any():
                        ms.add(band_within(within, b), m & bm,
                               f"{rule}, vegetated, TOPO {b}", files + src)
                continue
            ms.add(f"{key}, vegetated", m, f"{rule}, vegetated", files + src)
        except Exception as e:                                      # noqa: BLE001
            problems.append(f"region {target!r} cannot be made: {type(e).__name__}: {e}")
    return {"masks": ms, "problems": problems}


def layered_problem(var: str, levels: Sequence[str]) -> str:
    """Why a field with levels has no regional number."""
    return (f"{var} has a value at each of its levels ({', '.join(levels)}); a regional "
            f"number is made from one value per cell, so it cannot be made from this field. "
            f"For soil water, SOILWATER_10CM holds the top 10 cm as one value.")


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
        if streams[st].levels(v):
            out["problems"].append(layered_problem(v, streams[st].levels(v)))
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
        prev = _previous_years(s, ys)
        rs = R.reduce_regions(s, [R.Field(v) for v in vs], ms, ys + prev,
                              require_complete=not prev)
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
        months_and_seasons(out, rs, st, ms.names(), vs, ys, prev,
                           first_year=min(s.years()) if s.years() else None)
        out.setdefault("provenance", {"run_dir": str(run_dir), "streams": {}})[
            "streams"][st] = stream_record(s, vs, sorted(ys + prev))
    return out


# MONTHS AND SEASONS (2026-11-03). Every number above was a calendar year's,
# and nothing said so, so a question about a dry summer was answered from the
# year's total, which mixes the summer with the winter around it. Each run now
# carries, beside its yearly values, a value per month and per season, made by
# the same rule from the same records (regions.RegionSeries.by_year with its
# months): a rate's total over the month, a state's time-weighted mean.
# A season is made from its months, so a total is their sum and a mean is
# weighted by their days. DJF belongs to the year of its January and needs the
# December before it: a run whose output starts in January has none in its
# first year, and that season is left out with the reason, never filled from
# two months.
SEASONS = (("DJF", (12, 1, 2)), ("MAM", (3, 4, 5)), ("JJA", (6, 7, 8)),
           ("SON", (9, 10, 11)))


def season_parts(year: int, name: str, months: Sequence[int]) -> List[Tuple[int, int]]:
    """(year, month) of each month of a season of `year`: DJF's December is
    the year before's."""
    return [((int(year) - 1) if (name == "DJF" and m == 12) else int(year), m)
            for m in months]


SEASON_LEFT_OUT = "is left out, not made from two months"


def periods_plan() -> Dict[str, Any]:
    """The periods every field is reported over, as months_and_seasons makes
    them, for the design page: each month, and each season by its months."""
    import calendar
    seasons = []
    for name, months in SEASONS:
        seasons.append({"name": name, "months": [
            calendar.month_name[m] + (" of the year before" if py < 1 else "")
            for py, m in season_parts(1, name, months)]})
    return {"year": "each calendar year", "month": "each calendar month of each year",
            "seasons": seasons,
            "season_left_out": f"A season whose December the run's output does not hold "
                               f"{SEASON_LEFT_OUT}"}


def how_text(rate: bool, period: str) -> str:
    """How a field's value over a period is made, in words."""
    return f"{'total' if rate else 'mean'} over the {period}"


def month_key(year: int, month: int) -> str:
    """A month as the table names it: 2015-06."""
    return f"{int(year)}-{int(month):02d}"


def season_key(year: int, season: str) -> str:
    """A season as the table names it: 2015-JJA (DJF: the year of its January)."""
    return f"{int(year)}-{season}"


def period_units(units: str, rate: bool, period: str) -> str:
    """A field's units over a month or a season: a rate's total over the year
    ("gC/m^2/yr") is "gC/m^2/month" over a month; a state keeps its units."""
    u = str(units or "")
    return u.replace("/yr", f"/{period}") if rate else u


def _previous_years(stream: G.Stream, years: Sequence[int]) -> List[int]:
    """The year before each asked year that is not itself asked and that the
    stream holds records of: read only for its December, the first month of
    the next year's DJF."""
    have = set(stream.years())
    return sorted(({int(y) - 1 for y in years} - {int(y) for y in years}) & have)


# WHY A WINTER HAS NO DECEMBER (2026-11-03). The reason quoted the earliest
# year asked for as the year the output starts, which is false when the year
# before an asked year is missing from a stream that holds earlier ones. It now
# quotes the stream's own first year, or says the year itself is missing.
def _no_december_why(year: int, first_year: Optional[int]) -> str:
    if first_year is None:
        return "it holds no records"
    if int(first_year) > int(year) - 1:
        return f"its output starts in {int(first_year)}"
    return f"it holds no records of {int(year) - 1}, though its output starts in " \
           f"{int(first_year)}"


def months_and_seasons(out: Dict[str, Any], rs: R.RegionSeries, stream: str,
                       regions: Sequence[str], variables: Sequence[str],
                       years: Sequence[int], previous: Sequence[int] = (),
                       first_year: Optional[int] = None) -> None:
    """Fill out["monthly"] and out["seasonal"] ({region: {field: {key: value}}})
    for one stream's fields, with out["period_units"], and the reason for a
    value that is None (out["monthly_missing"], out["seasonal_missing"]) or a
    season left out (out["seasons_left_out"]). first_year is the first year
    the stream holds records of, quoted in that reason."""
    ys = sorted(int(y) for y in years)
    prev = sorted(int(y) for y in previous)
    for v in variables:
        how = "mean" if rs.per.get(v) is None else "sum"
        units = rs.units[v].replace("/d", "/yr") if how == "sum" else rs.units[v]
        out.setdefault("period_units", {})[v] = {
            "month": period_units(units, how == "sum", "month"),
            "season": period_units(units, how == "sum", "season")}
        for reg in regions:
            by: Dict[Tuple[int, int], Optional[float]] = {}
            whys: Dict[Tuple[int, int], str] = {}
            for m in range(1, 13):
                vals, why = rs.by_year(reg, v, ys, how=how, months=[m])
                for y in ys:
                    by[(y, m)] = vals.get(y)
                    if why.get(y):
                        whys[(y, m)] = why[y]
            if prev:
                vals, why = rs.by_year(reg, v, prev, how=how, months=[12])
                for y in prev:
                    by[(y, 12)] = vals.get(y)
                    if why.get(y):
                        whys[(y, 12)] = why[y]
            mon = out.setdefault("monthly", {}).setdefault(reg, {})[v] = {}
            for y in ys:
                for m in range(1, 13):
                    x = by[(y, m)]
                    mon[month_key(y, m)] = None if x is None else float(x)
                    if (y, m) in whys:
                        out.setdefault("monthly_missing", {}).setdefault(reg, {}) \
                            .setdefault(v, {})[month_key(y, m)] = whys[(y, m)]
            sea = out.setdefault("seasonal", {}).setdefault(reg, {})[v] = {}
            for y in ys:
                for name, months in SEASONS:
                    parts = season_parts(y, name, months)
                    key = season_key(y, name)
                    if parts[0] not in by:
                        out.setdefault("seasons_left_out", {})[key] = (
                            f"{name} {y} needs December {y - 1}, which the run's {stream} "
                            f"stream does not hold ({_no_december_why(y, first_year)}), "
                            f"so it {SEASON_LEFT_OUT}")
                        continue
                    xs = [by[p] for p in parts]
                    if any(x is None for x in xs):
                        sea[key] = None
                        gone = [month_key(*p) for p, x in zip(parts, xs) if x is None]
                        out.setdefault("seasonal_missing", {}).setdefault(reg, {}) \
                            .setdefault(v, {})[key] = (f"{', '.join(gone)} "
                                                       f"{'has' if len(gone) == 1 else 'have'}"
                                                       f" no value")
                        continue
                    if how == "sum":
                        sea[key] = float(sum(xs))
                    else:
                        d = [float(G.days_in_month(py, pm, rs.calendar)) for py, pm in parts]
                        sea[key] = float(sum(x * w for x, w in zip(xs, d)) / sum(d))


def period_differences(runs: Dict[str, Dict], key: str, control: str = "control"
                       ) -> Dict[str, Any]:
    """Each run minus the control over its months (key "monthly") or seasons
    ("seasonal"), for every region, field and period both have: differences()
    over another of the run's tables."""
    c = (runs.get(control) or {}).get(key) or {}
    out: Dict[str, Any] = {}
    for name, r in runs.items():
        if name == control:
            continue
        for reg, fields in (r.get(key) or {}).items():
            for v, periods in fields.items():
                for p, x in periods.items():
                    cp = ((c.get(reg) or {}).get(v) or {}).get(p)
                    if x is None or cp is None:
                        continue
                    out.setdefault(name, {}).setdefault(reg, {}).setdefault(v, {})[p] = \
                        x - cp
    return out


def period_normals(run: Dict[str, Any], normal_years: Sequence[int],
                   target_years: Sequence[int] = ()) -> Dict[str, Any]:
    """The normal years' mean of each calendar month ("01" to "12") and each
    season, every region and field of one run: what an event year's months
    are set against. A month with no value in any normal year has no normal,
    and a normal over fewer years than named says how many it holds.

    NO EVENT DECEMBER IN A NORMAL WINTER (2026-11-03). DJF of a year holds
    the December before it, so the DJF of the normal year after an event year
    (target_years) holds that event year's December. That winter is left out
    of the DJF normal; out["normal_left_out"] gives the reason per season and
    out["normal_years_used"] the years each such normal is the mean of."""
    ny = [int(y) for y in normal_years]
    ty = {int(y) for y in target_years}
    out: Dict[str, Any] = {"monthly_normal": {}, "seasonal_normal": {},
                           "normal_counts": {}}
    skip = [y for y in ny if y - 1 in ty]
    if skip:
        out["normal_left_out"] = {"DJF": (
            f"DJF {', '.join(str(y) for y in skip)} "
            f"{'is' if len(skip) == 1 else 'are'} left out of the normal winter: "
            f"{'it holds' if len(skip) == 1 else 'they hold'} the December of "
            f"{', '.join(str(y - 1) for y in skip)}, an event year")}
        out["normal_years_used"] = {"DJF": [y for y in ny if y not in skip]}
    for kind, table, labels in (
            ("monthly_normal", run.get("monthly") or {},
             [(f"{m:02d}", lambda y, m=m: month_key(y, m)) for m in range(1, 13)]),
            ("seasonal_normal", run.get("seasonal") or {},
             [(n, lambda y, n=n: season_key(y, n)) for n, _ in SEASONS])):
        for reg, fields in table.items():
            for v, periods in fields.items():
                for lab, key_of in labels:
                    xs = [periods.get(key_of(y)) for y in ny
                          if not (lab == "DJF" and y in skip)]
                    xs = [x for x in xs if x is not None]
                    out[kind].setdefault(reg, {}).setdefault(v, {})[lab] = \
                        (float(sum(xs) / len(xs)) if xs else None)
                    if len(xs) != len(ny):
                        out["normal_counts"].setdefault(reg, {}).setdefault(v, {})[lab] = \
                            len(xs)
    return out


# WHERE EVERY NUMBER COMES FROM (2026-11-03). The table said which stream a
# field was read from and nothing else, so a reader could not tell which files
# a number was made from, or that a file was rewritten after the numbers were
# made. Each run now records its directory and, per stream, the fields read
# from it, the files that hold the records read (name, size, modification
# time) and how many records each year has; each region its cells, area, rule
# and the files it is drawn from.
def _file_facts(p: Path) -> Dict[str, Any]:
    from datetime import datetime, timezone
    st = p.stat()
    return {"name": p.name, "bytes": int(st.st_size),
            "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc)
            .isoformat(timespec="seconds")}


def stream_record(stream: G.Stream, variables: Sequence[str], years: Sequence[int]
                  ) -> Dict[str, Any]:
    """What one stream's numbers were read from: its fields, cadence and
    calendar, the records of each year read, and the files that hold them."""
    per_year, files = {}, set()
    for y in sorted(int(y) for y in years):
        have = stream.year_records(y)[0]
        per_year[str(y)] = len(have)
        files |= {r.file for r in have}
    return {"fields": list(variables), "cadence": stream.cadence,
            "calendar": stream.calendar, "years_read": [int(y) for y in sorted(years)],
            "records_per_year": per_year,
            "files": [_file_facts(stream.files[i]) for i in sorted(files)],
            "records_left_out": len(stream.dropped)}


def region_record(ms: R.MaskSet, surface, registered: Optional[Dict[str, Dict]],
                  names: Sequence[Any]) -> Dict[str, Any]:
    """Each region's cells, area, rule and source files (names), with the full
    path of the surface file and of each registered region the plan named."""
    used = {}
    for n in names:
        key, within = region_spec(n)
        r = (registered or {}).get(within or key) or {}
        src = r.get("polygon") or r.get("field")
        if src:
            used[within or key] = str(src)
    return {"surface_file": str(surface), "registered_files": used,
            "regions": {n: {"cells": int(m["n_cells"]), "area_km2": float(m["area_km2"]),
                            "rule": m["rule"], "sources": list(m["sources"])}
                        for n, m in ms.summary().items()}}


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
                    percent=not is_kelvin((run.get("fields") or {}).get(v, {}).get("units")))
            except R.RegionError as e:
                problems.append(f"{reg}, {v}: {e}")
    return {"anomalies": out, "problems": problems}


def is_kelvin(units) -> bool:
    """A field in kelvin has no percent anomaly: +2 K is not 0.7 % of anything.

    KELVIN, NOT KILOGRAMS (2026-11-03). The rule was any unit starting with "K"
    in either case, so soil water in kg/m2 got no percent in an event study as
    if it were a temperature. A kelvin unit is a capital K not followed by a
    letter, the rule the numbers report uses (domain_report._no_percent)."""
    return re.match(r"K(?![A-Za-z])", str(units or "").strip()) is not None


def changed_share(grid: G.Grid, ms: R.MaskSet, registered: Optional[Dict[str, Dict]],
                  confined: Dict[str, str], problems: List[str]) -> Dict[str, Dict]:
    """{run: {region: share}} for each run whose change applies in one
    registered region: the share of each reported region's area inside it.
    Below 1, a region's difference from the control averages changed and
    unchanged cells; at 0 the change does not reach it."""
    out: Dict[str, Dict] = {}
    w = np.asarray(grid.weight_km2, dtype=np.float64)
    for run, where in sorted(confined.items()):
        r = (registered or {}).get(where)
        if not r:
            problems.append(f"run {run}: its change's region {where!r} is not registered, "
                            f"so the share of each region it changed is unknown")
            continue
        try:
            m = np.asarray(_registered_mask(grid, r)[0], dtype=bool)
        except Exception as e:                                      # noqa: BLE001
            problems.append(f"run {run}: its change's region {where!r} cannot be made: "
                            f"{type(e).__name__}: {e}")
            continue
        out[run] = {"region": where, "share": {}}
        for n in ms.names():
            a = ms.mask(n).array
            tot = float(w[a].sum())
            out[run]["share"][n] = round(float(w[a & m].sum()) / tot, 4) if tot else None
    return out


def experiment(control: Dict[str, Any], runs: Dict[str, str], analysis: Dict[str, Any],
               years: Sequence[int], confined: Optional[Dict[str, str]] = None
               ) -> Dict[str, Any]:
    """The whole table of one experiment.

    control   {"case_dir", "regions": {...registered...}}; its observations are
              not read: the table is the model's output only
    runs      {run name: run directory}, the control's under "control"
    analysis  the plan's: regions (names, or {"name", "within"} objects),
              variables, normal_years
    confined  {run name: the registered region its change applies in}, for a
              run whose change does not cover the whole domain
              (weather_scale's region): the table's changed_share
    """
    names = plan_regions(analysis)
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
    if BURNED_FRACTION in variables:
        burned_area_field(table)
        burned_area_periods(table)
    table["differences"] = differences(table["runs"])
    # months, seasons and where the numbers come from (2026-11-03)
    table["monthly_differences"] = period_differences(table["runs"], "monthly")
    table["seasonal_differences"] = period_differences(table["runs"], "seasonal")
    table["provenance"] = dict(region_record(ms, surface, control.get("regions"), names),
                               runs={n: r.pop("provenance", None) or {
                                   "run_dir": r.get("run_dir"), "streams": {}}
                                     for n, r in table["runs"].items()})
    if confined:
        table["changed_share"] = changed_share(grid, ms, control.get("regions"), confined,
                                               table["problems"])
    normal = [int(y) for y in (analysis.get("normal_years") or [])]
    if normal and "control" in table["runs"]:
        target = [int(y) for y in years if int(y) not in normal]
        ev = event(table["runs"]["control"], normal, target)
        table["event"] = {"normal_years": normal, "target_years": target, **ev}
        table["event"].update(period_normals(table["runs"]["control"], normal,
                                               target_years=target))
    # THE MODEL'S OUTPUT ONLY (2026-10-27). The table carried the control's
    # observed burned area beside the model's, and the Analyzer argued from it
    # ("MODIS saw 340 km2, so the carbon drop is not a fire response"), which
    # no comparison with an observation can show. How well the control matches
    # observations is told to the scientist before the design (the control's
    # card and the design page); the numbers an answer is read from are the
    # model's own.
    return table


def analysis_plan(control: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """The plan's analysis made concrete before any job, for the design page.

    Each region as masks() makes it on the control's own grid (its rule, the
    files it is drawn from, its cells and area), and each field as run_values
    will read it (the stream, whether a year, a month and a season is the
    field's total or its mean over it, the units of each), with the periods
    every field is reported over (periods_plan). Read from the control's
    existing history: a control
    that has none yet (one that runs too) gives no counts, and says so in
    `problems`, as a region or field that cannot be made does.

    control   {"case_dir", "run_dir", "regions": {...registered...}}
    analysis  the plan's: regions, variables, normal_years
    """
    # MONTHS AND SEASONS ON THE PAGE (2026-11-03). Each field was described by
    # its year's value only, while the table also reports it by month and by
    # season; each field now carries how and in what units those are made, and
    # the plan the periods, from the definitions months_and_seasons uses.
    out: Dict[str, Any] = {"regions": [], "fields": [], "problems": [],
                           "periods": periods_plan()}
    names = plan_regions(analysis)
    variables = [str(v) for v in (analysis.get("variables") or [])]
    run_dir = control.get("run_dir")
    have = G.discover(run_dir) if run_dir and Path(str(run_dir)).is_dir() else {}
    streams = {s: G.open_stream(run_dir, s) for s in STREAM_ORDER if s in have}
    if not streams:
        out["problems"].append("the control has no history yet, so the regions' cells and "
                               "the fields' streams are read when its output exists")
        return out
    try:
        surface = _surface(control["case_dir"])
    except Exception as e:                                          # noqa: BLE001
        out["problems"].append(f"the control's surface file: {type(e).__name__}: {e}")
        return out
    mk = masks(next(iter(streams.values())).grid, surface, names, control.get("regions"))
    for n, m in mk["masks"].summary().items():
        out["regions"].append({"name": n, "cells": m["n_cells"],
                               "area_km2": round(float(m["area_km2"]), 1),
                               "rule": m["rule"], "sources": m["sources"]})
    out["problems"] += list(mk["problems"])
    for v in variables:
        st = _stream_for(streams, v)
        if st is None:
            out["problems"].append(f"{v} is in none of the control's streams "
                                   f"({sorted(streams)})")
            continue
        s = streams[st]
        if s.levels(v):
            out["problems"].append(layered_problem(v, s.levels(v)))
            continue
        rate = s.per_second(v)
        # the history's units, with a rate's per second put per year, as
        # run_values reports a year's total of it
        units = str(s.units(v) or "")
        if rate:
            units = G.per_day_units(v, units).replace("/d", "/yr")
        out["fields"].append({"name": v, "stream": st, "how": how_text(rate, "year"),
                              "units": units, "note": G.FIELD_NOTES.get(v),
                              "by_month": how_text(rate, "month"),
                              "month_units": period_units(units, rate, "month"),
                              "by_season": how_text(rate, "season"),
                              "season_units": period_units(units, rate, "season")})
    if BURNED_FRACTION in variables:
        out["fields"].append({"name": BURNED_AREA, "stream": None, "how": how_text(True, "year"),
                              "units": "km2",
                              "note": f"the model's own burned area: the region's burned "
                                      f"fraction for the year ({BURNED_FRACTION}) times the "
                                      f"region's area",
                              "by_month": how_text(True, "month"), "month_units": "km2",
                              "by_season": how_text(True, "season"), "season_units": "km2"})
    return out


BURNED_AREA = "burned_area_km2"


def burned_area_field(table: Dict[str, Any]) -> None:
    """Each run's own burned area, km2 per region and year, as a field of its
    values beside the others: the region's burned fraction for the year
    (BURNED_FRACTION, a rate per second totalled over the year) times the
    region's area. A region-year without the fraction has no area."""
    for name, r in (table.get("runs") or {}).items():
        vals = r.get("values") or {}
        made = False
        for reg, info in table["regions"].items():
            by = (vals.get(reg) or {}).get(BURNED_FRACTION) or {}
            if by:
                vals.setdefault(reg, {})[BURNED_AREA] = {
                    y: (None if x is None else float(x) * info["area_km2"])
                    for y, x in by.items()}
                made = True
        if made:
            src = (r.get("fields") or {}).get(BURNED_FRACTION) or {}
            r.setdefault("fields", {})[BURNED_AREA] = {
                "stream": src.get("stream"), "how": "sum", "units": "km2",
                "note": f"the model's own burned area: the region's burned fraction for "
                        f"the year ({BURNED_FRACTION}) times the region's area"}


# BURNED AREA BY MONTH AND SEASON (2026-11-03). burned_area_field gave each
# run its burned area by year only, while the prompts say every number is
# reported by month and by season, so a summer's burned area was designed as
# answerable and then had no value to be read from. Each month's and season's
# burned area is made by the same rule: that period's burned fraction
# (BURNED_FRACTION totalled over it) times the region's area, with the
# fraction's reasons for a missing value copied across. Run before the period
# differences are taken, so those carry it too.
def burned_area_periods(table: Dict[str, Any]) -> None:
    for name, r in (table.get("runs") or {}).items():
        made = False
        for kind in ("monthly", "seasonal"):
            per = r.get(kind) or {}
            for reg, info in table["regions"].items():
                by = (per.get(reg) or {}).get(BURNED_FRACTION) or {}
                if not by:
                    continue
                per[reg][BURNED_AREA] = {
                    k: (None if x is None else float(x) * info["area_km2"])
                    for k, x in by.items()}
                made = True
                why = ((r.get(f"{kind}_missing") or {}).get(reg) or {}).get(BURNED_FRACTION)
                if why:
                    r[f"{kind}_missing"][reg][BURNED_AREA] = dict(why)
        if made:
            r.setdefault("period_units", {})[BURNED_AREA] = {"month": "km2", "season": "km2"}
