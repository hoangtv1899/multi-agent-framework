#!/usr/bin/env python3
"""
What a new run is given, beside the control run, for the design pages
mcp/elm-mcp/src/domain_given.py

A design page is written before any new run is made, so it cannot show what
a run produces. It can show what the run is given: the inputs its changes
alter, beside the control run's own. The person who reads the page asked for
figures in place of paragraphs (the review of 2026-11-12) and agreed to this
section: a small table of the changes, then a figure with one panel per
changed input, the control run's line beside the new run's with the changed
months shaded, and a map of where the change applies when it covers part of
the grid.

WHO DOES WHAT. This module is the model server's half. It works out the
values, from the change list (domain_changes), the run's own weather file
(domain_weather) and the control run's monthly history (gridded). The design
page's code draws them. The record is the contract between the two halves,
and its keys do not change:

    page_record(experiment, control, inputs_dir) ->
      {"control", "years", "problems",
       "runs": [{"name", "rows", "panels", "map", "problems"}]}

    a row     {"knob", "change", "when", "where"}, words from the change's
              parameters alone, so every run has its table even when no
              figure can be made
    a panel   {"title", "units", "step", "labels", "control", "run",
               "changed"}, two series as long as the labels, None where
              there is no value; step is "month" or "year"
    the map   {"title", "units", "lat", "lon", "values", "outline"} or None

WHAT EACH CHANGE IS GIVEN (2026-11-13):

  weather_scale        rain and snow, or the air temperature, month by month,
                       averaged over the cells the change reaches; the run's
                       values are the control run's under the run's own
                       weather file, read as the year check reads it
                       (domain_weather.check_year); a map when the change
                       reaches some of the land and not all of it
  weather_repeat_year  rain and snow and the air temperature over the whole
                       grid; a model year the map names takes its recorded
                       year's months
  fire_off             the area the control run's own fire burns each year,
                       from the control's facts, beside zero
  co2_ppmv, ndep_year, nutrient_supplement
                       rows only: the control run's monthly history holds no
                       series of these settings to draw

NOTHING HERE RAISES. A part that cannot be given is named under `problems`,
in words, and the rest is still returned: a failure to read a record must
never stop a design page.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

EXPECTATIONS = "domain_expectations.json"     # 01_inputs/runs/<run>/, as main.py names it
SECONDS_PER_DAY = 86400.0
# ELM's freezing point of water (SHR_CONST_TKFRZ), by which TBOT in K is
# written in degrees C.
FREEZING_K = 273.15
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")
ALL_MONTHS = list(range(1, 13))
EVERY_YEAR = "every year of the run"
EVERYWHERE = "everywhere"
WHOLE_GRID = "over the whole grid"
REACHED = "over the cells the change reaches"

# HOW MANY IN A ROW MAKE A SPAN (2026-11-13). The instructions for the page
# write months in a row as a span ("June to August") and years only from
# three in a row ("2033 and 2034", "2033 to 2035"). Two months in a row are
# therefore "June to July". A span never wraps the year end: months 12, 1
# and 2 of 2033 and 2034 are not the winter from December 2033 to February
# 2034, and "December to February" would say they were.
MONTH_SPAN_FROM = 2
YEAR_SPAN_FROM = 3

# THE TWO KNOBS OF weather_scale, in the order the page shows them (rain and
# snow first, as the instructions' table lists them; the map is of rain and
# snow when both are changed). The variables, the identities and how each
# acts on the weather are domain_weather.ANOMALY's, which wrote the file.
KNOBS = {
    "precip_factor": {"knob": "Rain and snow", "words": "Rain and snow",
                      "units": "mm/month", "map_units": "mm/month",
                      "map_title": "Change in rain and snow, mean of the changed months"},
    "temperature_K": {"knob": "Air temperature", "words": "Air temperature",
                      "units": "degrees C", "map_units": "K",
                      "map_title": "Change in air temperature, mean of the changed months"},
}
KNOB_ORDER = ("precip_factor", "temperature_K")

FIRE_TITLE = "Area the control run's own fire burns"
FIRE_UNITS = "km2"
SUPPLY = {"nitrogen": ("Nitrogen supply", "suplnitro"),
          "phosphorus": ("Phosphorus supply", "suplphos")}
SUPPLY_WORDS = {"ALL": "all that plants demand", "NONE": "none beyond the model's own"}


class GivenError(Exception):
    """A part of what a run is given cannot be worked out, and the message
    says why in plain words."""


# ─────────────────────────────────────────────────────────────────────────────
# WORDS
# ─────────────────────────────────────────────────────────────────────────────
def number_words(x) -> str:
    """A number as short as it is exact: 0.7, 2, 1.5, 550. A setting is
    printed as the person or the design wrote it, never rounded."""
    v = float(x)
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def _joined(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _runs(values: Sequence[int]) -> List[List[int]]:
    """Sorted distinct values cut into runs of consecutive ones."""
    out: List[List[int]] = []
    for v in sorted({int(x) for x in values}):
        if out and v == out[-1][-1] + 1:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def months_words(months: Sequence[int]) -> str:
    """"June to August", "March and October", "every month"."""
    ms = sorted({int(m) for m in months})
    if ms == ALL_MONTHS:
        return "every month"
    items: List[str] = []
    for r in _runs(ms):
        if len(r) >= MONTH_SPAN_FROM:
            items.append(f"{MONTHS[r[0] - 1]} to {MONTHS[r[-1] - 1]}")
        else:
            items += [MONTHS[m - 1] for m in r]
    return _joined(items)


def years_words(years: Sequence[int]) -> str:
    """"2033 and 2034", "2033 to 2035", "2033 to 2035 and 2038"."""
    items: List[str] = []
    for r in _runs(years):
        if len(r) >= YEAR_SPAN_FROM:
            items.append(f"{r[0]} to {r[-1]}")
        else:
            items += [str(y) for y in r]
    return _joined(items)


def _row(knob: str, change: str, when: str, where: str) -> Dict[str, str]:
    return {"knob": knob, "change": change, "when": when, "where": where}


def _setting(facts: Optional[Dict[str, Any]], key: str):
    return ((facts or {}).get("settings") or {}).get(key)


def _choice(v) -> Optional[str]:
    """A namelist choice as one word: 'ALL', "NONE", none -> ALL, NONE."""
    if v is None:
        return None
    s = str(v).strip().strip("'\"").strip().upper()
    return s or None


def rows(rc: Dict[str, Any], facts: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """The table's rows of one change, resolved (domain_changes.resolve), in
    the words the page prints. Built from the change's parameters and, for
    CO2 and the nutrient supplements, the control run's own settings."""
    cid, p = rc["id"], rc.get("params") or {}
    if cid == "weather_scale":
        import domain_weather as W
        on = W.active(p)
        named = p.get("years")
        when = (f"{months_words(p.get('months') or ALL_MONTHS)} of "
                f"{years_words(named) if named else EVERY_YEAR}")
        where = f"the region `{p['region']}`" if p.get("region") else EVERYWHERE
        out = []
        if "precip_factor" in on:
            out.append(_row(KNOBS["precip_factor"]["knob"],
                            f"multiplied by {number_words(p['precip_factor'])}", when, where))
        if "temperature_K" in on:
            t = float(p["temperature_K"])
            out.append(_row(KNOBS["temperature_K"]["knob"],
                            f"{number_words(abs(t))} K {'warmer' if t > 0 else 'cooler'}",
                            when, where))
        return out
    if cid == "weather_repeat_year":
        by_recorded: Dict[int, List[int]] = {}
        for model_year, recorded in (p.get("map") or {}).items():
            if int(model_year) != int(recorded):
                by_recorded.setdefault(int(recorded), []).append(int(model_year))
        return [_row("Weather", f"the recorded weather of {w}", years_words(ys), EVERYWHERE)
                for w, ys in sorted(by_recorded.items())]
    if cid == "fire_off":
        return [_row("Fire", "switched off", EVERY_YEAR, EVERYWHERE)]
    if cid == "co2_ppmv":
        change = f"{number_words(p['ppmv'])} ppmv"
        # THE CONTROL'S OWN LEVEL (2026-11-13): the namelist's value where the
        # case sets one, else the case's own, as domain_controls reads it; and
        # none where the control's CO2 is not held constant, since a level
        # that changes by year is no one number to set beside the run's.
        have = None
        kind = str(_setting(facts, "co2_type") or _setting(facts, "ELM_CO2_TYPE") or "")
        if not kind or kind.strip().strip("'\"").lower() == "constant":
            for key in ("co2_ppmv", "CCSM_CO2_PPMV"):
                try:
                    if _setting(facts, key) is not None:
                        have = number_words(_setting(facts, key))
                        break
                except (TypeError, ValueError):
                    continue
        if have is not None:
            change += f", where the control run has {have} ppmv"
        return [_row("CO2", change, EVERY_YEAR, EVERYWHERE)]
    if cid == "ndep_year":
        return [_row("Nitrogen deposition", f"held at its level of {int(p['year'])}",
                     EVERY_YEAR, EVERYWHERE)]
    if cid == "nutrient_supplement":
        # A nutrient whose setting is the control run's own changes nothing,
        # so it has no row; one whose control setting is not known has one.
        out = []
        for nutrient, (knob, key) in SUPPLY.items():
            want, have = _choice(p.get(nutrient)), _choice(_setting(facts, key))
            if want is None or (have is not None and have == want):
                continue
            out.append(_row(knob, SUPPLY_WORDS.get(want, want), EVERY_YEAR, EVERYWHERE))
        return out
    return []


# ─────────────────────────────────────────────────────────────────────────────
# THE CONTROL RUN'S MONTHLY HISTORY
# ─────────────────────────────────────────────────────────────────────────────
class _History:
    """The control run's monthly history (h0), opened once for a page and
    read a field and a year at a time, as domain_measured._point_series
    reads it."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self._stream = None
        self._error: Optional[str] = None
        self._months: Dict[tuple, np.ndarray] = {}

    def stream(self):
        if self._stream is not None:
            return self._stream
        if self._error is None:
            if not self.run_dir:
                self._error = "the control run names no folder of history files"
            else:
                try:
                    import gridded as G
                    s = G.open_stream(self.run_dir, "h0")
                    if s.cadence != "monthly":
                        self._error = (f"the control run's h0 history is {s.cadence}, "
                                       f"not monthly")
                    else:
                        self._stream = s
                except Exception as e:                              # noqa: BLE001
                    self._error = (f"the control run's monthly history cannot be read "
                                   f"({type(e).__name__}: {e})")
        if self._stream is None:
            raise GivenError(self._error)
        return self._stream

    @property
    def grid(self):
        return self.stream().grid

    def weights(self) -> np.ndarray:
        """The grid's own weights, area times land fraction, as km2."""
        return np.asarray(self.grid.weight_km2, dtype=np.float64)

    def units(self, var: str) -> str:
        s = self.stream()
        if var not in s.variables:
            raise GivenError(f"the control run's monthly history holds no {var}")
        return str(s.units(var) or "").strip()

    def months(self, var: str, year: int) -> np.ndarray:
        """(12, *grid) of one field in one year, NaN where a month or a cell
        has no value. A year the history does not hold at all is refused."""
        key = (var, int(year))
        if key in self._months:
            return self._months[key]
        s = self.stream()
        if var not in s.variables:
            raise GivenError(f"the control run's monthly history holds no {var}")
        if s.levels(var):
            raise GivenError(f"{var} has levels in the control run's monthly history")
        if int(year) not in s.years():
            raise GivenError(f"the control run's monthly history holds no {int(year)}")
        b = s.read(var, int(year), require_complete=False)
        out = np.full((12,) + tuple(s.grid.shape), np.nan)
        data = np.asarray(b.data, dtype=np.float64)
        for k, row in zip(b.keys, data):
            out[int(k[1]) - 1] = row
        self._months[key] = out
        return out


def _precip(h: _History, year: int) -> np.ndarray:
    """Rain and snow in mm/month, cell by cell: RAIN + SNOW, rates in mm/s,
    times the seconds of each month."""
    import gridded as G
    for v in ("RAIN", "SNOW"):
        u = h.units(v)
        if not (G.is_per_second(v, u) and u.lower().startswith("mm")):
            raise GivenError(f"the control run's {v} is in {u!r}, not a rate in mm per "
                             f"second")
    cal = h.stream().calendar
    secs = np.array([G.days_in_month(int(year), m, cal) * SECONDS_PER_DAY
                     for m in ALL_MONTHS])
    secs = secs.reshape((12,) + (1,) * len(h.grid.shape))
    return (h.months("RAIN", year) + h.months("SNOW", year)) * secs


def _temperature(h: _History, year: int) -> np.ndarray:
    """The air temperature the land model receives, TBOT, in degrees C."""
    u = h.units("TBOT")
    if u != "K":
        raise GivenError(f"the control run's TBOT is in {u!r}, not K")
    return h.months("TBOT", year) - FREEZING_K


QUANTITY = {"precip_factor": _precip, "temperature_K": _temperature}


def _mean(field: np.ndarray, cells: np.ndarray, w: np.ndarray) -> Optional[float]:
    """The mean of one month's field over the cells, weighted by the grid's
    weights; None when one of the cells has no value in the month."""
    x, ww = field[cells], w[cells]
    if not x.size or not np.isfinite(x).all() or not ww.sum() > 0:
        return None
    return float(np.sum(x * ww) / np.sum(ww))


def _labels(years: Sequence[int]) -> List[str]:
    return [f"{int(y):04d}-{m:02d}" for y in years for m in ALL_MONTHS]


# ─────────────────────────────────────────────────────────────────────────────
# weather_scale: THE RUN'S OWN WEATHER FILE
# ─────────────────────────────────────────────────────────────────────────────
def _weather_rules(inputs_dir, run_name: str) -> Dict[str, Dict[str, Any]]:
    """The year-check rules the server wrote for the run's weather file, by
    the file's variable (tas, pr)."""
    import domain_changes as CH
    if inputs_dir is None:
        raise GivenError("the run's inputs are not built yet, so the weather file it "
                         "reads cannot be read")
    if not CH.RUN_NAME.fullmatch(str(run_name or "")):
        raise GivenError(f"{run_name!r} is not a run's name")
    p = Path(str(inputs_dir)) / "runs" / run_name / EXPECTATIONS
    if not p.is_file():
        raise GivenError(f"the run's inputs hold no {EXPECTATIONS}, so the run is not "
                         f"built and the weather file it reads is not known")
    try:
        e = json.loads(p.read_text())
    except ValueError:
        raise GivenError(f"the run's {EXPECTATIONS} cannot be read") from None
    out = {}
    for o in (e.get("output") if isinstance(e, dict) else None) or []:
        if isinstance(o, dict) and o.get("rule") == "matches_anomaly" and o.get("anomaly_var"):
            out[str(o["anomaly_var"])] = o
    return out


def _weather_file(rule: Dict[str, Any], inputs_dir, run_name: str) -> Path:
    """The weather file a rule names; when the study's folder was moved
    since the run was built, the file of that name in the run's own weather
    folder. Its checksum must be the one the run was built with."""
    import domain_weather as W
    named = Path(str(rule.get("anomaly_file") or ""))
    here = Path(str(inputs_dir)) / "runs" / run_name / "weather" / named.name
    path = named if named.name and named.is_file() else here
    if not named.name or not path.is_file():
        raise GivenError("the weather file the run was built with is missing")
    want = rule.get("anomaly_sha256")
    if want and W.sha256(path) != want:
        raise GivenError("the weather file is not the one the run was built with: its "
                         "content has changed")
    return path


def _same_settings(path: Path, params: Dict[str, Any], years: Sequence[int]) -> None:
    """The weather file was written for the change and the years the design
    now holds. A design changed after its runs were built would otherwise
    show a table of one change and a figure of another. The file's records
    run from the year before the run's first to the year after its last
    (domain_weather.month_starts), so a file built for fewer years still
    holds identity records for a year added later, and only its time axis
    tells."""
    import netCDF4
    with netCDF4.Dataset(str(path)) as f:
        base = int(str(f["time"].units).split("since")[1].strip()[:4])
        n = len(f["time"])
        wrote = json.loads(f.getncattr("params")) if "params" in f.ncattrs() else None
    if base != min(years) - 1 or n != 12 * (max(years) - min(years) + 3):
        raise GivenError(f"the weather file was written for other years than the "
                         f"experiment's {years_words(years)}: the run must be built again")
    if wrote is None:
        return
    keys = ("temperature_K", "precip_factor", "months", "years", "region")
    now = json.loads(json.dumps({k: params.get(k) for k in keys}))
    differ = [k for k in keys if wrote.get(k) != now.get(k)]
    if differ:
        raise GivenError(f"the weather file was written for other settings than the "
                         f"run's change ({', '.join(differ)}): the run must be built "
                         f"again")


def _file_fields(path: Path, var: str, years: Sequence[int], grid) -> np.ndarray:
    """(months, *grid) of one field of the weather file over the run's
    years, matched to the control run's cells as the year check matches it
    (domain_weather.check_year)."""
    import domain_weather as W
    stack = []
    for y in years:
        try:
            by_month, xc, yc = W.anomaly_by_month(str(path), var, int(y))
        except W.WeatherError as e:
            raise GivenError(f"the weather file cannot be read ({e})") from None
        lon = np.where(xc > 180.0, xc - 360.0, xc)
        glon = np.where(grid.lon > 180.0, grid.lon - 360.0, grid.lon)
        if grid.lat.shape != yc.shape or not (
                np.allclose(grid.lat, yc, rtol=0.0, atol=W.CELL_MATCH_DEG)
                and np.allclose(glon, lon, rtol=0.0, atol=W.CELL_MATCH_DEG)):
            raise GivenError("the weather file's cells are not the control run's cells")
        missing = [m for m in ALL_MONTHS if m not in by_month]
        if missing:
            raise GivenError(f"the weather file holds no record for "
                             f"{months_words(missing)} of {int(y)}")
        stack += [np.asarray(by_month[m], dtype=np.float64) for m in ALL_MONTHS]
    return np.stack(stack)


def _map_values(diff: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """The mean over the changed months, cell by cell, NaN outside the cells."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # a cell with no value
        m = np.nanmean(diff, axis=0) if diff.shape[0] else np.full(cells.shape, np.nan)
    return np.where(cells, m, np.nan)


def weather_scale_panels(rc: Dict[str, Any], run_name: str, years: Sequence[int],
                         h: _History, inputs_dir, problems: List[str]):
    """(panels, map) of one weather_scale change. The file's cells and months
    are the run's; a knob whose values cannot be read is left out and named
    under `problems`, and the other knob still has its panel."""
    import domain_measured as DM
    import domain_weather as W
    if not years:
        raise GivenError("the experiment names no years")
    on = W.active(rc["params"])
    knobs = [k for k in KNOB_ORDER if k in on]
    rules = _weather_rules(inputs_dir, run_name)
    grid = h.grid
    w = h.weights()
    land = w > 0
    labels = _labels(years)
    files: Dict[str, Path] = {}
    for k in knobs:
        a = W.ANOMALY[k]
        rule = rules.get(a["var"])
        if rule is None:
            raise GivenError("the run's checks name no weather file")
        if rule.get("how") != a["how"]:
            raise GivenError(f"the run's checks read {a['var']} as {rule.get('how')!r}, "
                             f"not {a['how']!r}")
        files[k] = _weather_file(rule, inputs_dir, run_name)
    for path in set(files.values()):
        _same_settings(path, rc["params"], years)
    fields = {k: _file_fields(files[k], W.ANOMALY[k]["var"], years, grid) for k in knobs}
    panels, maps = [], []
    for k in knobs:
        a, words = W.ANOMALY[k], KNOBS[k]
        try:
            f = fields[k]
            departs = f != a["identity"]
            # THE CELLS THE CHANGE REACHES (2026-11-13): where the file departs
            # from no change in some month of the run's years, among the
            # cells the grid gives a weight (area times land fraction). A cell
            # that weighs nothing is not land, and a change there changes no
            # land; with it counted, a region drawn over every land cell
            # would not be "the whole grid".
            cells = departs.any(axis=0) & land
            if not cells.any():
                raise GivenError("the weather file changes no land cell of the control "
                                 "run in the run's years")
            whole = bool(np.all(cells[land]))
            q = np.concatenate([QUANTITY[k](h, y) for y in years])
            run = q * f if a["how"] == "factor" else q + f
            changed = [t for t in range(len(labels)) if (departs[t] & cells).any()]
            panels.append({
                "title": f"{words['words']} {WHOLE_GRID if whole else REACHED}",
                "units": words["units"], "step": "month", "labels": list(labels),
                "control": DM._clean([_mean(q[t], cells, w) for t in range(len(labels))]),
                "run": DM._clean([_mean(run[t], cells, w) for t in range(len(labels))]),
                "changed": [labels[t] for t in changed]})
            if not whole:
                maps.append((k, run[changed] - q[changed], cells))
        except GivenError as e:
            problems.append(f"{words['words']}: {e}")
    the_map = None
    if maps:
        # one map per run: rain and snow when that knob is changed, else the
        # air temperature
        k, diff, cells = maps[0]
        if not grid.regular or grid.lat_axis is None or grid.lon_axis is None:
            problems.append("the map is not given: the control run's grid is not a "
                            "regular latitude-longitude grid")
        else:
            the_map = {"title": KNOBS[k]["map_title"], "units": KNOBS[k]["map_units"],
                       "lat": [float(x) for x in grid.lat_axis],
                       "lon": [float(x) for x in grid.lon_axis],
                       "values": DM._grid_list(_map_values(diff, cells)),
                       "outline": [[int(bool(x)) for x in row] for row in cells]}
    return panels, the_map


# ─────────────────────────────────────────────────────────────────────────────
# weather_repeat_year AND fire_off
# ─────────────────────────────────────────────────────────────────────────────
def repeat_year_panels(rc: Dict[str, Any], years: Sequence[int], h: _History,
                       problems: List[str]) -> List[Dict[str, Any]]:
    """Rain and snow and the air temperature over the whole grid. A model
    year the map names takes its recorded year's months; the others keep
    their own."""
    import domain_measured as DM
    import domain_weather as W
    if not years:
        raise GivenError("the experiment names no years")
    ymap = W.year_map(rc["params"], years)
    named = {y: r for y, r in ymap.items() if r != y}
    w = h.weights()
    land = w > 0
    labels = _labels(years)
    changed = [f"{y:04d}-{m:02d}" for y in years for m in ALL_MONTHS if int(y) in named]
    out = []
    for k in KNOB_ORDER:
        words = KNOBS[k]
        try:
            control, run = [], []
            for y in years:
                own = QUANTITY[k](h, int(y))
                read = QUANTITY[k](h, named.get(int(y), int(y)))
                control += [_mean(own[m], land, w) for m in range(12)]
                run += [_mean(read[m], land, w) for m in range(12)]
            out.append({"title": f"{words['words']} {WHOLE_GRID}", "units": words["units"],
                        "step": "month", "labels": list(labels),
                        "control": DM._clean(control), "run": DM._clean(run),
                        "changed": list(changed)})
        except GivenError as e:
            problems.append(f"{words['words']}: {e}")
    return out


def fire_panel(years: Sequence[int], facts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The area the control run's own fire burns in each of the run's years,
    from its facts (domain_controls.compute_facts: burned_area_km2.model, the
    model's own yearly values), beside the run's, which is zero."""
    import domain_measured as DM
    if not years:
        raise GivenError("the experiment names no years")
    model = ((facts or {}).get("burned_area_km2") or {}).get("model") or {}
    if not isinstance(model, dict):
        model = {}
    values = DM._clean([_number(model.get(str(int(y)))) for y in years])
    if all(v is None for v in values):
        raise GivenError("the control run's facts hold no burned area for the run's years")
    labels = [str(int(y)) for y in years]
    return {"title": FIRE_TITLE, "units": FIRE_UNITS, "step": "year", "labels": labels,
            "control": values, "run": [0.0] * len(labels), "changed": list(labels)}


def _number(v) -> Optional[float]:
    try:
        return None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# THE RECORD
# ─────────────────────────────────────────────────────────────────────────────
def _years(v) -> Optional[List[int]]:
    if not isinstance(v, list) or not v or \
            not all(isinstance(y, int) and not isinstance(y, bool) for y in v):
        return None
    return [int(y) for y in v]


def _run_record(run, years: List[int], facts: Optional[Dict[str, Any]], h: _History,
                inputs_dir) -> Dict[str, Any]:
    """One run's rows, panels and map. Never raises."""
    import domain_changes as CH
    out: Dict[str, Any] = {"name": None, "changes": [], "rows": [], "panels": [],
                           "map": None, "problems": []}
    try:
        if not isinstance(run, dict):
            out["problems"].append(f"a run is an object with a name and changes, not "
                                   f"{run!r}")
            return out
        out["name"] = run.get("name")
        # THE CHANGES THE RECORD WAS MADE FROM (2026-11-13), as the design gave
        # them. A page drawn again from its saved records checks them against
        # its design, so a record an earlier design left, with the same run
        # name and years, is not drawn as this design's.
        out["changes"] = [c for c in (run.get("changes") or []) if isinstance(c, dict)]
        resolved = []
        for item in run.get("changes") or []:
            cid = item.get("id") if isinstance(item, dict) else None
            try:
                rc = CH.resolve(item)
            except Exception as e:                                  # noqa: BLE001
                out["problems"].append(f"the change {cid or item!r} cannot be read, so it "
                                       f"has no row ({e})")
                continue
            resolved.append(rc)
            try:
                out["rows"] += rows(rc, facts)
            except Exception as e:                                  # noqa: BLE001
                out["problems"].append(f"{rc['id']}: its row cannot be written "
                                       f"({type(e).__name__}: {e})")
        for rc in resolved:
            try:
                if rc["id"] == "weather_scale":
                    panels, the_map = weather_scale_panels(rc, out["name"], years, h,
                                                           inputs_dir, out["problems"])
                    out["panels"] += panels
                    if out["map"] is None:
                        out["map"] = the_map
                elif rc["id"] == "weather_repeat_year":
                    out["panels"] += repeat_year_panels(rc, years, h, out["problems"])
                elif rc["id"] == "fire_off":
                    out["panels"].append(fire_panel(years, facts))
            except GivenError as e:
                out["problems"].append(f"{rc['id']}: no figure, {e}")
            except Exception as e:                                  # noqa: BLE001
                out["problems"].append(f"{rc['id']}: no figure ({type(e).__name__}: {e})")
    except Exception as e:                                          # noqa: BLE001
        out["problems"].append(f"what the run is given cannot be worked out "
                               f"({type(e).__name__}: {e})")
    return out


def page_record(experiment, control, inputs_dir=None) -> Dict[str, Any]:
    """What each new run of an experiment is given, beside the control run,
    for the design page.

    experiment  {"control": {"name", "reuse"}, "runs": [{"name", "changes":
                [{"id", "params"}]}], "years": [...]}
    control     the control's registry entry with its facts:
                dict(domain_controls.get(name), facts=domain_controls.facts(name))
    inputs_dir  the study's 01_inputs folder, where the server built each run
                (runs/<name>/domain_expectations.json names its weather file);
                None, or a run not built there, gives that run no weather
                figure

    Never raises. Anything it cannot give is named under `problems`."""
    rec: Dict[str, Any] = {"control": None, "years": [], "problems": [], "runs": []}
    try:
        exp = experiment if isinstance(experiment, dict) else {}
        ctl = control if isinstance(control, dict) else {}
        named = exp.get("control") if isinstance(exp.get("control"), dict) else {}
        rec["control"] = named.get("name") or ctl.get("name")
        years = _years(exp.get("years"))
        if years is None:
            rec["problems"].append("the experiment names no years, so no figure can be made")
            years = []
        rec["years"] = years
        facts = ctl.get("facts") if isinstance(ctl.get("facts"), dict) else None
        if facts is None:
            rec["problems"].append("the control run's facts are not known, so its own "
                                   "settings and burned area are not given")
        runs = exp.get("runs")
        if not isinstance(runs, list):
            rec["problems"].append("the experiment lists no runs")
            runs = []
        h = _History(ctl.get("run_dir"))
        for run in runs:
            rec["runs"].append(_run_record(run, years, facts, h, inputs_dir))
    except Exception as e:                                          # noqa: BLE001
        rec["problems"].append(f"what the new runs are given cannot be worked out "
                               f"({type(e).__name__}: {e})")
    return rec
