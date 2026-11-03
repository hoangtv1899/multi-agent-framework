#!/usr/bin/env python3
"""Every number in 03_results/regional.json, made again by a second code path.

A whole-domain experiment's numbers are made once, by the model server
(domain_analysis.py over gridded.py and regions.py), and everything after
reads them: the numbers report, the Analyzer, the e-mail. A wrong number there
is wrong everywhere, and no audit of what cites it can see it (the even-count
median of 2026-09-13 read 10.7 m where the data said 12.1 m, and every
citation of it checked out). So the numbers are made a second time here, from
each run's own history, by code that SHARES NO REDUCTION CODE with those three
modules, and the table is reported only when the two agree.

WHAT IS INDEPENDENT. The netCDF reading (netCDF4 and numpy directly), the time
weighting (each record counts for the part of its own time_bounds that falls in
the year, so the start convention, the year boundary and the run's first
30-minute step need no rule of their own), the area weights (area x landfrac,
read from the history), and every mask: the polygon test is matplotlib's on
the cell centres, a gridded-field region applies its rule to the file here,
and vegetated cells and elevation bands are read from the surface file the
control's lnd_in names, with the band edges read back from the band names the
table carries. WHAT IS SHARED is the definition of each quantity (a rate per
second is totalled over the year, anything else is the year's time-weighted
mean, a region's value is the area-weighted mean over its cells with a value)
and where the inputs are: the manager's records name each run's output
directory (never the table under test), and the control registry names each
registered region's file.

MONTHS AND SEASONS (2026-11-03) are made by the same means as a year: a
month is the records' overlap with that calendar month, a season with its
three months (MAM, JJA and SON within the year; DJF from 1 December of the
year before, made only when the history covers a day or more of that
December), each made directly from the records, not from the table's months.
The monthly and seasonal differences and the event's normals of each month
and season follow by their definitions. The provenance block is held to the
files on disk (each named file exists with its stated size) and to the masks
(each region's cells and area).

  check(run_dir)  the comparison record, written to 04_analysis/numbers_check.json
  gate(run_dir)   the analyze stage's gate: the check, and when it fails, the
                  numbers report written WITHHELD in place of the numbers

THE TOLERANCE. History is float32 (about 7 significant digits); both paths read
the same float32 values and reduce them in float64, so two right answers agree
to about 1e-12. A number is reproduced when

    |table - recomputed| <= RTOL x |recomputed| + EPS x scale + ATOL

RTOL 1e-5 is about 100 float32 steps of the number itself, so a change in its
fifth significant digit fails. The scale is the size of the quantities a
derived number is made from (a difference is judged against the two values it
subtracts, an anomaly against the years it compares, a percent against the
value and its base), and EPS 1e-9 of it is the floor two float64 reductions of
the same data stay inside: a small difference of two large stocks is held to
its own fifth digit, not to the stocks'. ATOL 1e-9 lets two zeros (a no-fire
run's burned area) agree.

WHERE EACH RUN'S OUTPUT IS. The control's directory is read from the manager's
record of it (01_inputs/domain_control_facts.json, else domain_runs.json), each
other run's from the year checks its chain wrote (domain_case/<run>/checks/
y*.json), never from the table under test; a table that names another
directory, or two runs that resolve to one directory, is a mismatch.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CHECK_JSON = "numbers_check.json"
TABLE = "03_results/regional.json"
RTOL = 1e-5
ATOL = 1e-9
# TOLERANCE OF A DERIVED NUMBER (2026-11-03). A difference, an anomaly or a
# percent was judged against its operands at RTOL, so a difference of 4.39
# between two stocks near 30000 could sit in the table as 4.44 and pass. Two
# float64 reductions of the same float32 data agree to about 1e-12 of the
# operands (4.6e-15 on the demos); EPS of the operands is the floor, and the
# number is otherwise held to RTOL of itself.
EPS = 1e-9
# The table rounds a changed share to 4 decimals (domain_analysis.changed_share),
# and so does the check; a share on a rounding edge may land one step apart.
SHARE_ATOL = 1e-4 + 1e-9
SECONDS_PER_DAY = 86400.0
FILL_ABS = 1e30
# Cell centres of two files are the same cells within this, in degrees.
COORD_TOL_DEG = 1e-4
# Two record bounds meet when they are this close, in days (about 0.1 s).
TIME_TOL_DAYS = 1e-6
# A field stored per second whose units do not say so (ELM's FireMod.F90
# writes FAREA_BURNED per second under the units 'proportion').
PER_SECOND_ANYWAY = ("FAREA_BURNED",)
BURNED_AREA, BURNED_FRACTION = "burned_area_km2", "FAREA_BURNED"
VEGETATED = "vegetated cells"
VEG_SUFFIX = ", vegetated"
# The keys of a regional table this check knows how to recompute; any other
# top-level key is listed as not checked, never failed and never silent.
KNOWN = ("regions", "years", "runs", "problems", "differences", "event",
         "changed_share", "observed", "error", "monthly_differences",
         "seasonal_differences", "provenance")
MAX_ROWS_SHOWN = 40


class CheckError(Exception):
    """The numbers cannot be recomputed, and the message says why."""


# ─────────────────────────────────────────────────────────────────────────────
# WHERE THE INPUTS ARE
# ─────────────────────────────────────────────────────────────────────────────
def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}


def _registry(name: Optional[str]) -> Dict[str, Any]:
    """The control's entry in the server's control registry, or {}.

    Read through the server's own loader (domain_controls.load), which finds the
    git-ignored registry the way the server does (IDEAS_DOMAIN_CONTROLS, then
    mcp/elm-mcp/domain_controls.json). A loader, not reduction code."""
    if not name:
        return {}
    try:
        import domain_controls as DCT                               # noqa: N812
    except ImportError:
        src = Path(__file__).resolve().parents[3] / "mcp" / "elm-mcp" / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import domain_controls as DCT                               # noqa: N812
    return (DCT.load().get("controls") or {}).get(name) or {}


def inputs(run_dir, registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What the check reads, from the run's own records.

    The control: 01_inputs/domain_control_facts.json (the manager's record of the
    reused control), else 01_inputs/domain_runs.json (the server's build answer).
    Its regions: the registry entry of that name, as the manager hands them to
    domain_analysis. The surface file: fsurdat in the control's CaseDocs/lnd_in,
    as domain_analysis reads it, else in its run directory's lnd_in."""
    rd = Path(run_dir)
    facts = _read_json(rd / "01_inputs" / "domain_control_facts.json")
    built = _read_json(rd / "01_inputs" / "domain_runs.json")
    ctl = dict(facts.get("control") or built.get("control") or {})
    if not ctl.get("case_dir"):
        ent = next((r for r in built.get("runs") or [] if r.get("name") == "control"), {})
        ctl.setdefault("case_dir", ent.get("case_dir"))
    entry = registry if registry is not None else _registry(ctl.get("name"))
    case_dir = ctl.get("case_dir") or entry.get("case_dir")
    return {"control": ctl.get("name"), "case_dir": case_dir,
            "control_run_dir": ctl.get("run_dir"),
            "registry_run_dir": entry.get("run_dir"),
            "regions": dict(entry.get("regions") or {}),
            "observations": dict(entry.get("observations") or {}),
            "registry_entry": bool(entry)}


def _same_dir(a, b) -> bool:
    return os.path.realpath(str(a)) == os.path.realpath(str(b))


def run_dirs(run_dir, table: Dict[str, Any], inp: Dict[str, Any]
             ) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, str]]:
    """({run: output directory}, mismatch rows, {run: the record it came from}).

    EACH RUN'S OUTPUT FROM THE MANAGER'S RECORDS (2026-11-03). The table under
    test named the directories the check read, so a table made from the wrong
    run's history was recomputed from that same history and agreed. Now a run
    the chain executed is found where its own year checks put it
    (domain_case/<run>/checks/y*.json, each naming the run directory it
    checked), and a reused control where the manager recorded it
    (01_inputs/domain_control_facts.json, else domain_runs.json). A table that
    names another directory, a run with no such record, records that disagree,
    and two runs that resolve to one directory are each a mismatch."""
    rd = Path(run_dir)
    dirs: Dict[str, str] = {}
    src: Dict[str, str] = {}
    bad: List[Dict[str, Any]] = []
    for run, r in (table.get("runs") or {}).items():
        # a run's name is a folder name of the chain's records, never a path
        found = sorted((rd / "domain_case" / run / "checks").glob("y*.json")) \
            if run not in (".", "..") and "/" not in run else []
        named = sorted({str(_read_json(p).get("run_dir")) for p in found
                        if _read_json(p).get("run_dir")})
        if named:
            where = f"domain_case/{run}/checks/{found[0].name}"
        elif run == "control" and inp.get("control_run_dir"):
            named, where = [str(inp["control_run_dir"])], \
                "01_inputs/domain_control_facts.json or domain_runs.json"
            reg = inp.get("registry_run_dir")
            if reg and not _same_dir(reg, named[0]):
                bad.append({"section": "runs", "run": run, "what": "run_dir",
                            "table": r.get("run_dir"), "recomputed": named[0],
                            "why": f"the control registry names {reg}, the manager's "
                                   f"record {named[0]}"})
        else:
            bad.append({"section": "runs", "run": run, "what": "run_dir",
                        "table": r.get("run_dir"), "recomputed": None,
                        "why": "no record of the manager names this run's output "
                               "directory, so its numbers cannot be recomputed"})
            continue
        if len(named) > 1:
            bad.append({"section": "runs", "run": run, "what": "run_dir",
                        "table": r.get("run_dir"), "recomputed": named[0],
                        "why": "the run's year checks name different directories: "
                               + ", ".join(named)})
        dirs[run], src[run] = named[0], where
        if not r.get("run_dir") or not _same_dir(r["run_dir"], named[0]):
            bad.append({"section": "runs", "run": run, "what": "run_dir",
                        "table": r.get("run_dir"), "recomputed": named[0],
                        "why": f"the table names another output directory than "
                               f"{where}"})
    seen: Dict[str, str] = {}
    for run, d in dirs.items():
        key = os.path.realpath(d)
        if key in seen:
            bad.append({"section": "runs", "run": run, "what": "run_dir",
                        "table": (table["runs"][run] or {}).get("run_dir"), "recomputed": d,
                        "why": f"runs {seen[key]} and {run} resolve to one output "
                               f"directory"})
        else:
            seen[key] = run
    return dirs, bad, src


def _fsurdat(case_dir, run_dir=None) -> Path:
    for p in ([Path(case_dir) / "CaseDocs" / "lnd_in"] if case_dir else []) + \
             ([Path(run_dir) / "lnd_in"] if run_dir else []):
        if p.is_file():
            m = re.search(r"^\s*fsurdat\s*=\s*['\"]([^'\"]+)['\"]", p.read_text(),
                          re.MULTILINE | re.IGNORECASE)
            if m:
                return Path(m.group(1).strip())
            raise CheckError(f"{p} names no fsurdat")
    raise CheckError(f"no lnd_in in {case_dir}/CaseDocs or {run_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# READING THE HISTORY
# ─────────────────────────────────────────────────────────────────────────────
_HIST = re.compile(r"\.elm\.(h\d+)\.\d{4}-\d{2}(?:-\d{2}-\d{5})?\.nc$")


def _clean(a) -> np.ndarray:
    """float64, with masked, non-finite and |x| > 1e30 values as NaN."""
    x = np.ma.filled(np.ma.asarray(a).astype(np.float64), np.nan)
    x = np.asarray(x, dtype=np.float64)
    x[~np.isfinite(x) | (np.abs(x) > FILL_ABS)] = np.nan
    return x


def _lon180(lon) -> np.ndarray:
    x = np.asarray(lon, dtype=np.float64)
    return (x + 180.0) % 360.0 - 180.0


def _same_cells(lat, lon, glat, glon, what: str) -> None:
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    if lat.ndim == 1 and lon.ndim == 1 and (lat.size, lon.size) == glat.shape:
        lon, lat = np.meshgrid(lon, lat)
    if lat.shape != glat.shape or lon.shape != glat.shape:
        raise CheckError(f"{what}: cells of shape {lat.shape}, the history {glat.shape}")
    d = max(float(np.nanmax(np.abs(lat - glat))),
            float(np.nanmax(np.abs(_lon180(lon - glon)))))
    if not d <= COORD_TOL_DEG:
        raise CheckError(f"{what}: cell centres differ from the history's by {d:.3g} deg")


def per_second(var: str, units: str) -> bool:
    """A rate per second: its year is a total, never a mean.

    Units read as a quotient ('mm/s', 'gC/m^2/s', 'proportion/sec', 'per
    second'), a negative power ('kg m-2 s-1', 's^-1') or 'm2s' (umol/m2s), but
    not a second squared ('kg/m/s^2', 's-2'). FAREA_BURNED is per second
    whatever its units say."""
    if var in PER_SECOND_ANYWAY:
        return True
    u = " ".join(str(units or "").lower().split())
    quotient = re.search(r"(?:/|\bper\s)\s*(?:s|sec|second)(?![a-z0-9])", u)
    squared = re.search(r"(?:/|\bper\s)\s*(?:s|sec|second)\s*(?:\^|\*\*)\s*\+?[02-9]", u)
    power = re.search(r"(?<![a-z])s\s*(?:\^|\*\*)?\s*-\s*1(?!\d)", u)
    return bool((quotient and not squared) or power or re.search(r"m\^?2\s*s$", u))


class History:
    """One run's history files, read with netCDF4 and nothing else of ours."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise CheckError(f"{run_dir} is not a directory")
        self.files: Dict[str, List[Path]] = {}
        for p in sorted(self.run_dir.iterdir()):
            m = _HIST.search(p.name)
            if m and ".elm_" not in p.name:
                self.files.setdefault(m.group(1), []).append(p)
        if not self.files:
            raise CheckError(f"{run_dir} holds no ELM history")
        self._vars: Dict[str, set] = {}

    def streams(self) -> List[str]:
        return sorted(self.files, key=lambda s: int(s[1:]))

    def variables(self, stream: str) -> set:
        if stream not in self._vars:
            import netCDF4
            with netCDF4.Dataset(self.files[stream][0]) as d:
                self._vars[stream] = set(d.variables)
        return self._vars[stream]

    def stream_for(self, var: str) -> Optional[str]:
        return next((s for s in self.streams() if var in self.variables(s)), None)

    def grid(self, stream: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Cell centres and weights (area x landfrac, km2; 0 where either is
        missing), from the first file of a stream."""
        import netCDF4
        st = stream or self.streams()[0]
        with netCDF4.Dataset(self.files[st][0]) as d:
            for v in ("lat", "lon", "area", "landfrac"):
                if v not in d.variables:
                    raise CheckError(f"{self.files[st][0].name} has no {v}")
            lat, lon = _clean(d["lat"][:]), _clean(d["lon"][:])
            area, landfrac = _clean(d["area"][:]), _clean(d["landfrac"][:])
        if lat.ndim == 1 and lon.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        if area.shape != lat.shape:
            raise CheckError(f"area has shape {area.shape}, the cells {lat.shape}: "
                             f"only a lat x lon grid is checked here")
        w = area * landfrac
        return {"lat": lat, "lon": lon, "weight": np.where(np.isfinite(w), w, 0.0)}

    def units(self, stream: str, var: str) -> str:
        import netCDF4
        with netCDF4.Dataset(self.files[stream][0]) as d:
            return str(getattr(d[var], "units", "") or "")

    def year(self, stream: str, var: str, year: int, W: np.ndarray, cells: np.ndarray
             ) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[str]]:
        """(region means per record (n, regions), each record's days inside the
        year, why the year is not whole or None).

        W is (cells, regions): each region's weight on the cells `cells` (a flat
        index into the grid). A record counts for the overlap of its own
        time_bounds with [1 Jan year, 1 Jan year+1); the records found must tile
        that interval exactly, or the year is not whole."""
        chunks = self.records(stream, var, W, cells, (year, 1, 1), (year + 1, 1, 1))
        return span(chunks, (year, 1, 1), (year + 1, 1, 1), var, str(year))

    def records(self, stream: str, var: str, W: np.ndarray, cells: np.ndarray,
                start: Tuple[int, int, int], end: Tuple[int, int, int]
                ) -> List[Dict[str, Any]]:
        """Every record of var that overlaps [start, end), read once, as the
        region means at each record and its time_bounds, file by file:
        [{"units", "calendar", "bounds" (n, 2), "means" (n, regions)}]. Any
        period inside [start, end) is then made from these (span)."""
        import netCDF4
        out: List[Dict[str, Any]] = []
        prev_end = None
        for p in self.files[stream]:
            with netCDF4.Dataset(p) as d:
                if var not in d.variables:
                    continue
                t = d["time"]
                bname = getattr(t, "bounds", None) or "time_bounds"
                if bname not in d.variables:
                    raise CheckError(f"{p.name} has no time bounds")
                cal = str(getattr(t, "calendar", "standard"))
                units = str(t.units)
                tb = np.asarray(d[bname][:], dtype=np.float64)
                tb, prev_end = restart_clip(tb, prev_end, units, cal)
                a, b = _day(start, units, cal), _day(end, units, cal)
                lo, hi = np.maximum(tb[:, 0], a), np.minimum(tb[:, 1], b)
                idx = np.flatnonzero(hi - lo > TIME_TOL_DAYS)
                if not idx.size:
                    continue
                v = d[var]
                v.set_auto_maskandscale(True)
                if v.ndim != 3:
                    raise CheckError(f"{var} has {v.ndim} dimensions; a regional number "
                                     f"is made from a (time, lat, lon) field")
                x = _clean(v[idx.tolist()]).reshape(len(idx), -1)[:, cells]
                ok = np.isfinite(x)
                num = np.where(ok, x, 0.0) @ W
                den = ok.astype(np.float64) @ W
                with np.errstate(invalid="ignore", divide="ignore"):
                    means = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
                out.append({"units": units, "calendar": cal, "bounds": tb[idx],
                            "means": means})
        return out


# A RUN SEGMENT RESTARTED AT A MONTH'S START (2026-11-03). ELM folds a
# segment's first model step into its first monthly record, so a segment
# restarted on 1 January 2016 writes January 2016 from one step before
# midnight: it overlaps December 2015 by that step, and the tiling test found
# a gap or overlap in 2015 and in DJF 2016 that the history does not have.
# The longest step taken as such is 6 hours (ELM steps 30 or 60 minutes).
MAX_STEP_DAYS = 0.25


def _on_month_start(x: float, units: str, cal: str) -> bool:
    """True when x (in the file's own time units) is the start of a month."""
    import netCDF4
    dt = netCDF4.num2date(x, units, calendar=cal, only_use_cftime_datetimes=True)
    nxt = (dt.year + 1, 1) if dt.month == 12 else (dt.year, dt.month + 1)
    return any(abs(_day((y, m, 1), units, cal) - x) <= TIME_TOL_DAYS
               for y, m in ((dt.year, dt.month), nxt))


def restart_clip(tb: np.ndarray, prev_end: Optional[float], units: str, cal: str
                 ) -> Tuple[np.ndarray, Optional[float]]:
    """(time_bounds with each restart step clipped, the last record's end).

    A record that starts before the previous record ends, by no more than one
    model step (MAX_STEP_DAYS), and ends on the first of a month is a month
    with a restart step folded in: its start is moved to the previous end, so
    it counts for its own month only. Any other overlap is left for span to
    find. prev_end carries the last end over from the file before."""
    tb = np.array(tb, dtype=np.float64, copy=True)
    for k in range(tb.shape[0]):
        if prev_end is not None:
            ov = prev_end - tb[k, 0]
            if TIME_TOL_DAYS < ov <= MAX_STEP_DAYS + TIME_TOL_DAYS and \
                    _on_month_start(tb[k, 1], units, cal):
                tb[k, 0] = prev_end
        prev_end = float(tb[k, 1])
    return tb, prev_end


def _day(date: Tuple[int, int, int], units: str, cal: str) -> float:
    """A date as a number in a file's own time units and calendar."""
    import cftime
    import netCDF4
    return float(netCDF4.date2num(cftime.datetime(*date, calendar=cal), units, calendar=cal))


def span(chunks: List[Dict[str, Any]], start: Tuple[int, int, int],
         end: Tuple[int, int, int], var: str, label: str
         ) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[str]]:
    """(region means per record, each record's days inside [start, end), why
    the period is not whole or None), from History.records. A record counts
    for the overlap of its own time_bounds with the period; the records found
    must tile it exactly."""
    means: List[np.ndarray] = []
    spans: List[Tuple[float, float]] = []
    length = None
    for c in chunks:
        a = _day(start, c["units"], c["calendar"])
        b = _day(end, c["units"], c["calendar"])
        tb = c["bounds"]
        lo, hi = np.maximum(tb[:, 0], a), np.minimum(tb[:, 1], b)
        idx = np.flatnonzero(hi - lo > TIME_TOL_DAYS)
        if not idx.size:
            continue
        means.append(c["means"][idx])
        spans += [(float(lo[k] - a), float(hi[k] - a)) for k in idx]
        length = b - a
    if not spans:
        return None, np.zeros(0), f"no record of {var} in {label}"
    order = np.argsort([s[0] for s in spans])
    sp = [spans[k] for k in order]
    m = np.concatenate(means, axis=0)[order]
    edge = 0.0
    for a, b in sp:
        if abs(a - edge) > TIME_TOL_DAYS:
            return None, np.zeros(0), (f"{label}: the records leave a gap or overlap at "
                                       f"day {edge:.4f}")
        edge = b
    if abs(edge - length) > TIME_TOL_DAYS:
        return None, np.zeros(0), f"{label}: the records end at day {edge:.4f} of {length:g}"
    return m, np.array([b - a for a, b in sp]), None


def covered_days(chunks: List[Dict[str, Any]], start: Tuple[int, int, int],
                 end: Tuple[int, int, int]) -> float:
    """How many days of [start, end) the records cover."""
    tot = 0.0
    for c in chunks:
        a = _day(start, c["units"], c["calendar"])
        b = _day(end, c["units"], c["calendar"])
        tb = c["bounds"]
        tot += float(np.clip(np.minimum(tb[:, 1], b) - np.maximum(tb[:, 0], a), 0.0,
                             None).sum())
    return tot


# ─────────────────────────────────────────────────────────────────────────────
# THE MASKS
# ─────────────────────────────────────────────────────────────────────────────
def _rings(geo) -> List[List[np.ndarray]]:
    if geo.get("type") == "FeatureCollection":
        gs = [f.get("geometry") or {} for f in geo.get("features") or []]
    elif geo.get("type") == "Feature":
        gs = [geo.get("geometry") or {}]
    else:
        gs = [geo]
    out = []
    for g in gs:
        parts = [g["coordinates"]] if g.get("type") == "Polygon" else \
            g["coordinates"] if g.get("type") == "MultiPolygon" else None
        if parts is None:
            raise CheckError(f"geometry {g.get('type')!r} is not a polygon")
        for p in parts:
            rs = []
            for r in p:
                a = np.asarray(r, dtype=np.float64)[:, :2].copy()
                a[:, 0] = np.unwrap(_lon180(a[:, 0]), period=360.0)
                rs.append(a)
            out.append(rs)
    if not out:
        raise CheckError("no polygon in the GeoJSON")
    return out


def polygon_cells(path, grid) -> np.ndarray:
    """Cells whose centre is inside the polygon: matplotlib's test, exterior
    minus holes, union over polygons, with the centres shifted by 360 degrees
    for a ring that runs past 180."""
    from matplotlib.path import Path as MPath
    pts = np.column_stack([_lon180(grid["lon"]).ravel(), grid["lat"].ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for rings in _rings(json.loads(Path(path).read_text())):
        lo = min(float(r[:, 0].min()) for r in rings)
        hi = max(float(r[:, 0].max()) for r in rings)
        for s in [0.0] + ([360.0] if hi > 180.0 else []) + ([-360.0] if lo < -180.0 else []):
            q = pts + np.array([s, 0.0])
            m = MPath(rings[0]).contains_points(q)
            for h in rings[1:]:
                m &= ~MPath(h).contains_points(q)
            inside |= m
    return inside.reshape(grid["lat"].shape)


_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal,
        "==": np.equal, "!=": np.not_equal}


def field_cells(r: Dict[str, Any], grid) -> np.ndarray:
    """Cells where the registered field's rule holds (var op value, combined
    over the leading axes by any or all); a missing value never selects."""
    import netCDF4
    op, over = r.get("op", ">"), r.get("over", "any")
    if op not in _OPS or over not in ("any", "all"):
        raise CheckError(f"region rule {op!r} over {over!r} is not one this check knows")
    with netCDF4.Dataset(r["field"]) as f:
        pair = next(((a, b) for a, b in (("lat", "lon"), ("LATIXY", "LONGXY"), ("yc", "xc"))
                     if a in f.variables and b in f.variables), None)
        if pair is None:
            raise CheckError(f"{Path(r['field']).name}: no cell centres")
        la, lo = f[pair[0]], f[pair[1]]
        _same_cells(_clean(la[:]), _clean(lo[:]), grid["lat"], grid["lon"],
                    Path(r["field"]).name)
        dims = (la.dimensions[0], lo.dimensions[0]) if la.ndim == 1 else tuple(la.dimensions)
        v = f[r["var"]]
        if tuple(v.dimensions[-2:]) != tuple(dims):
            raise CheckError(f"{r['var']} runs {v.dimensions}, its cells {dims}")
        a = _clean(v[:])
    with np.errstate(invalid="ignore"):
        sel = np.where(np.isfinite(a), _OPS[op](a, float(r.get("value", 0.0))), False)
    lead = tuple(range(a.ndim - 2))
    if lead:
        sel = sel.any(axis=lead) if over == "any" else sel.all(axis=lead)
    return np.asarray(sel, dtype=bool)


def band_rule(label: str):
    """(low, high) of an elevation band, read back from its name: 'below 1000 m',
    '1000-1500 m', '2000 m and above'. None when the name is not a band."""
    s = label.strip()
    m = re.fullmatch(r"below (-?[\d.]+) m", s)
    if m:
        return None, float(m.group(1))
    m = re.fullmatch(r"(-?[\d.]+)-(-?[\d.]+) m", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.fullmatch(r"(-?[\d.]+) m and above", s)
    if m:
        return float(m.group(1)), None
    return None


def _band(topo: np.ndarray, lohi) -> np.ndarray:
    lo, hi = lohi
    m = np.isfinite(topo)
    if lo is not None:
        m &= topo >= lo
    if hi is not None:
        m &= topo < hi
    return m


def region_mask(name: str, grid, surface: Dict[str, np.ndarray],
                registered: Dict[str, Dict], cache: Dict[str, np.ndarray]) -> np.ndarray:
    """The cells of one region the table names, rebuilt from its name:
    'vegetated cells'; a band; '<registered>, vegetated'; or '<registered>,
    vegetated, <band>'. Every region is restricted to the vegetated cells."""
    veg = surface["veg"]
    if name == VEGETATED:
        return veg.copy()
    b = band_rule(name)
    if b is not None:
        return veg & _band(surface["topo"], b)
    head, sep, tail = name.partition(VEG_SUFFIX)
    if not sep:
        raise CheckError(f"region {name!r}: not a name this check can rebuild")
    if head not in cache:
        r = registered.get(head)
        if not r:
            raise CheckError(f"region {head!r} is not in the control's registry entry")
        if r.get("polygon"):
            cache[head] = polygon_cells(r["polygon"], grid)
        elif r.get("field"):
            cache[head] = field_cells(r, grid)
        else:
            raise CheckError(f"region {head!r} names neither a polygon nor a field")
    m = cache[head] & veg
    tail = tail.lstrip(", ").strip()
    if tail:
        b = band_rule(tail)
        if b is None:
            raise CheckError(f"region {name!r}: {tail!r} is not an elevation band")
        m &= _band(surface["topo"], b)
    return m


def read_surface(path, grid) -> Dict[str, np.ndarray]:
    import netCDF4
    with netCDF4.Dataset(path) as s:
        for v in ("LATIXY", "LONGXY", "PCT_NATVEG", "TOPO"):
            if v not in s.variables:
                raise CheckError(f"{Path(path).name} has no {v}")
        _same_cells(_clean(s["LATIXY"][:]), _clean(s["LONGXY"][:]), grid["lat"],
                    grid["lon"], f"surface {Path(path).name}")
        nat, topo = _clean(s["PCT_NATVEG"][:]), _clean(s["TOPO"][:])
    return {"veg": np.isfinite(nat) & (nat >= 100.0 - 1e-6), "topo": topo}


# ─────────────────────────────────────────────────────────────────────────────
# THE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
class _Ledger:
    def __init__(self):
        self.n, self.absent, self.bad, self.worst = 0, 0, [], None

    def add(self, where: Dict[str, Any], table, mine, scale=None, atol=ATOL,
            why: Optional[str] = None, exact: bool = False) -> None:
        if table is None and mine is None:
            self.absent += 1
            return
        row = dict(where, table=table, recomputed=mine)
        if table is None or mine is None or isinstance(table, bool) or \
                not isinstance(table, (int, float)):
            self.n += 1
            row["why"] = why or ("the table has a number the check could not make"
                                 if mine is None else
                                 "the table has no number where the check made one"
                                 if table is None else "the table's entry is not a number")
            self.bad.append(row)
            return
        self.n += 1
        t, m = float(table), float(mine)
        d = abs(t - m)
        # Relative to the number itself, with the operands' floor (EPS/RTOL of
        # the scale) so a difference near zero is not divided by zero: a
        # rel_diff above RTOL is outside the tolerance, whatever the number.
        s = abs(m) + (EPS / RTOL) * abs(float(scale) if scale is not None else 0.0)
        rel = d / s if s > 0 else (0.0 if d == 0 else math.inf)
        row.update(abs_diff=d, rel_diff=rel)
        if self.worst is None or rel > self.worst["rel_diff"]:
            self.worst = dict(row)
        allowed = 0.0 if exact else RTOL * s + atol
        if not (d <= allowed) or not math.isfinite(t):
            row["allowed"] = allowed
            self.bad.append(row)

    def mismatch(self, where: Dict[str, Any], table, mine, why: str) -> None:
        """A number compared that disagrees whatever its value: a period one
        side leaves out and the other makes."""
        self.n += 1
        self.bad.append(dict(where, table=table, recomputed=mine, why=why))

    def agree_absent(self) -> None:
        self.absent += 1

    def same(self, where: Dict[str, Any], table, mine) -> None:
        """A fact that is not a number (a path, a list of years), equal or not."""
        self.n += 1
        if table != mine:
            self.bad.append(dict(where, table=table, recomputed=mine,
                                 why="the table and the check differ"))


def _year_value(means, days, rate: bool) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if means is None:
        return None, None
    if not np.all(np.isfinite(means)):
        return None, "a record has no value in the region"
    if rate:
        return (means * days[:, None]).sum(axis=0) * SECONDS_PER_DAY, None
    return (means * days[:, None]).sum(axis=0) / days.sum(), None


# MONTHS AND SEASONS, MADE HERE TOO (2026-11-03). The table carries each
# run's value per month and per season beside its year's, and the report draws
# and writes them, so they are made again here by the same means as a year's:
# the records' overlap with the period, by their own time_bounds. A season is
# made directly over its three months (a total is the sum over them, a mean
# is weighted by time), not from the table's months. DJF of a year runs from
# 1 December of the year before to 1 March; it is made only when the history
# covers at least a day of that December, so the run's first model step, which
# ELM writes before 1 January, is not a December.
# (year offset, month) of each season's first day, and of the first day after it
SEASONS = {"DJF": ((-1, 12), (0, 3)), "MAM": ((0, 3), (0, 6)), "JJA": ((0, 6), (0, 9)),
           "SON": ((0, 9), (0, 12))}
HELD_DAYS = 1.0


def _month_bounds(y: int, m: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    return (y, m, 1), ((y, m + 1, 1) if m < 12 else (y + 1, 1, 1))


def _season_bounds(y: int, name: str):
    """[start, end) of a season of year y: its first month's first day to the
    first day of the month after its last."""
    (da, ma), (db, mb) = SEASONS[name]
    return (y + da, ma, 1), (y + db, mb, 1)


def recompute_runs(table, masks: Dict[str, np.ndarray], grid,
                   extra_fields: Sequence[str] = (),
                   dirs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """{run: {"values": {region: {field: {year: v}}}, "monthly": {region:
    {field: {"2015-06": v}}}, "seasonal": {... "2015-JJA": v}, "left_out":
    {field: [season keys the history cannot make]}, "how": {field: ...},
    "stream": {field: ...}, "why": {...}}} from each run's own history, in the
    directory `dirs` names for it (run_dirs); a run it does not name is not
    recomputed, and every number of it is then a mismatch."""
    regs = list(masks)
    out: Dict[str, Any] = {}
    years = sorted({int(y) for y in table.get("years") or []})
    for run, r in (table.get("runs") or {}).items():
        if dirs is not None and run not in dirs:
            continue
        h = History(dirs[run] if dirs is not None else r["run_dir"])
        g = h.grid()
        if g["weight"].shape != grid["weight"].shape:
            raise CheckError(f"run {run}: its grid {g['weight'].shape} is not the "
                             f"control's {grid['weight'].shape}")
        union = np.zeros(g["weight"].shape, dtype=bool)
        for m in masks.values():
            union |= m
        cells = np.flatnonzero(union.ravel())
        W = np.stack([np.where(masks[k].ravel()[cells], g["weight"].ravel()[cells], 0.0)
                      for k in regs], axis=1)
        fields = [f for f in (r.get("fields") or {}) if f != BURNED_AREA]
        fields += [f for f in extra_fields if f not in fields]
        ys = sorted(set(years) | {int(y) for reg in (r.get("values") or {}).values()
                                  for f in (reg or {}).values() for y in (f or {})})
        rec = {"values": {k: {} for k in regs}, "monthly": {k: {} for k in regs},
               "seasonal": {k: {} for k in regs}, "left_out": {}, "how": {},
               "stream": {}, "why": {}}
        for f in fields:
            st = h.stream_for(f)
            if st is None:
                rec["why"][f] = f"{f} is in none of the run's streams"
                continue
            rate = per_second(f, h.units(st, f))
            rec["how"][f], rec["stream"][f] = ("sum" if rate else "mean"), st
            if not ys:
                continue
            chunks = h.records(st, f, W, cells, (ys[0] - 1, 12, 1), (ys[-1] + 1, 1, 1))

            def put(kind, key, a, b, label):
                means, days, why = span(chunks, a, b, f, label)
                vals, why2 = _year_value(means, days, rate)
                for j, k in enumerate(regs):
                    rec[kind][k].setdefault(f, {})[key] = \
                        None if vals is None else float(vals[j])
                if why or why2:
                    rec["why"][f"{f} {key}"] = why or why2

            for y in ys:
                put("values", str(y), (y, 1, 1), (y + 1, 1, 1), str(y))
                for mo in range(1, 13):
                    a, b = _month_bounds(y, mo)
                    put("monthly", f"{y}-{mo:02d}", a, b, f"{y}-{mo:02d}")
                for name in SEASONS:
                    a, b = _season_bounds(y, name)
                    key = f"{y}-{name}"
                    if name == "DJF" and covered_days(chunks, *_month_bounds(y - 1, 12)) \
                            < HELD_DAYS - TIME_TOL_DAYS:
                        rec["left_out"].setdefault(f, []).append(key)
                        continue
                    put("seasonal", key, a, b, key)
        out[run] = rec
    return out


def _anomaly_block(by: Dict[int, Optional[float]], normal: Sequence[int],
                   target: Sequence[int], percent: bool) -> Optional[Dict[str, Any]]:
    """The event numbers from recomputed years, by their definitions: the
    normal is the mean of the normal years, the anomaly the value minus it, the
    percent 100 x anomaly / normal, the rank 1 + the number of years lower."""
    nv = [by.get(y) for y in normal]
    if any(v is None for v in nv):
        return None
    nv = np.array(nv, dtype=np.float64)
    mean = float(nv.sum() / nv.size)
    have = {y: v for y, v in by.items() if v is not None}
    out = {"normal": mean, "normal_min": float(nv.min()), "normal_max": float(nv.max()),
           "normal_sd": (float(np.sqrt(((nv - mean) ** 2).sum() / (nv.size - 1)))
                         if nv.size > 1 else None),
           "by_year": {str(y): v for y, v in by.items()}, "years": {}}
    for y in target:
        v = by.get(y)
        if v is None:
            out["years"][str(y)] = None
            continue
        out["years"][str(y)] = {
            "value": v, "anomaly": v - mean,
            "anomaly_pct": (100.0 * (v - mean) / mean) if percent and mean != 0 else None,
            "rank_lowest_first": 1 + sum(1 for w in have.values() if w < v),
            "of": len(have)}
    return out


def _yearkey(d) -> Dict[str, Any]:
    return {str(k): v for k, v in (d or {}).items()}


def compare(table: Dict[str, Any], mine: Dict[str, Any], masks, grid,
            registered: Dict[str, Dict], observations: Dict[str, Dict],
            dirs: Optional[Dict[str, str]] = None, surface=None) -> Dict[str, Any]:
    L = _Ledger()
    notes: List[str] = []
    w = grid["weight"]
    area = {k: float(w[m].sum()) for k, m in masks.items()}
    # regions: cells exactly, area within the tolerance
    for k, info in (table.get("regions") or {}).items():
        L.add({"section": "regions", "region": k, "what": "cells"}, info.get("cells"),
              int(masks[k].sum()) if k in masks else None, exact=True)
        L.add({"section": "regions", "region": k, "what": "area_km2"}, info.get("area_km2"),
              area.get(k))

    def val(run, reg, f, y):
        if f == BURNED_AREA:
            x = ((mine.get(run) or {}).get("values") or {}).get(reg, {}).get(
                BURNED_FRACTION, {}).get(y)
            return None if x is None or reg not in area else x * area[reg]
        return ((mine.get(run) or {}).get("values") or {}).get(reg, {}).get(f, {}).get(y)

    # each run's values, and how each field's year is made
    for run, r in (table.get("runs") or {}).items():
        for f, meta in (r.get("fields") or {}).items():
            if f == BURNED_AREA:
                continue
            got = (mine.get(run) or {}).get("how", {}).get(f)
            if got and meta.get("how") and got != meta.get("how"):
                L.bad.append({"section": "fields", "run": run, "field": f,
                              "table": meta.get("how"), "recomputed": got,
                              "why": "the table and the check make this field's year "
                                     "differently (a total against a mean)"})
            st = (mine.get(run) or {}).get("stream", {}).get(f)
            if st and meta.get("stream") and st != meta.get("stream"):
                notes.append(f"run {run}, {f}: the table read {meta['stream']}, the check "
                             f"{st}")
        for reg, fields in (r.get("values") or {}).items():
            for f, years in (fields or {}).items():
                for y, x in _yearkey(years).items():
                    m = val(run, reg, f, y)
                    why = (mine.get(run) or {}).get("why", {}).get(f"{f} {y}") or \
                        (mine.get(run) or {}).get("why", {}).get(f)
                    L.add({"section": "values", "run": run, "region": reg, "field": f,
                           "year": y}, x, m, why=why)
    # each run minus the control
    for run, regs in (table.get("differences") or {}).items():
        for reg, fields in (regs or {}).items():
            for f, years in (fields or {}).items():
                for y, x in _yearkey(years).items():
                    a, b = val(run, reg, f, y), val("control", reg, f, y)
                    L.add({"section": "differences", "run": run, "region": reg, "field": f,
                           "year": y}, x, None if a is None or b is None else a - b,
                          scale=max(abs(a or 0.0), abs(b or 0.0)))
    # the event: each target year of the control against its normal years
    ev = table.get("event") or {}
    if ev.get("anomalies"):
        normal = [int(y) for y in ev.get("normal_years") or []]
        target = [int(y) for y in ev.get("target_years") or []]
        for reg, fields in ev["anomalies"].items():
            for f, a in (fields or {}).items():
                by = {int(y): val("control", reg, f, str(y))
                      for y in (a.get("by_year") or {}) or (normal + target)}
                b = _anomaly_block(by, normal, target, percent=True) or {}
                at = {"section": "event", "region": reg, "field": f}
                big = max([abs(v) for v in by.values() if v is not None] + [0.0])
                for k in ("normal", "normal_min", "normal_max"):
                    L.add(dict(at, what=k), a.get(k), b.get(k), scale=big)
                if a.get("normal_sd") is not None or b.get("normal_sd") is not None:
                    L.add(dict(at, what="normal_sd"), a.get("normal_sd"), b.get("normal_sd"),
                          scale=big)
                for y, x in _yearkey(a.get("by_year")).items():
                    L.add(dict(at, what="by_year", year=y), x, (b.get("by_year") or {}).get(y))
                for y, yt in _yearkey(a.get("years")).items():
                    ym = (b.get("years") or {}).get(y)
                    if yt is None or ym is None:
                        L.add(dict(at, what="year", year=y), None if yt is None else
                              yt.get("value"), None if ym is None else ym["value"])
                        continue
                    nrm = abs(b.get("normal") or 0.0)
                    L.add(dict(at, what="value", year=y), yt.get("value"), ym["value"])
                    L.add(dict(at, what="anomaly", year=y), yt.get("anomaly"), ym["anomaly"],
                          scale=big)
                    # a percent the table leaves out (a temperature, a zero
                    # normal) is not a number to compare
                    if yt.get("anomaly_pct") is not None:
                        L.add(dict(at, what="anomaly_pct", year=y), yt["anomaly_pct"],
                              ym["anomaly_pct"], scale=(100.0 * big / nrm) if nrm else None)
                    for k in ("rank_lowest_first", "of"):
                        if k in yt:
                            L.add(dict(at, what=k, year=y), yt.get(k), ym.get(k), exact=True)
    # the share of each region a confined change reaches (rounded to 4 places)
    for run, cs in (table.get("changed_share") or {}).items():
        where = (cs or {}).get("region")
        try:
            r = registered.get(where) or {}
            c = polygon_cells(r["polygon"], grid) if r.get("polygon") else \
                field_cells(r, grid) if r.get("field") else None
        except Exception as e:                                      # noqa: BLE001
            c, notes_e = None, f"{type(e).__name__}: {e}"
            notes.append(f"changed_share of run {run}: {notes_e}")
        for reg, x in ((cs or {}).get("share") or {}).items():
            m = None
            if c is not None and reg in masks and area.get(reg):
                m = round(float(w[masks[reg] & c].sum()) / area[reg], 4)
            L.add({"section": "changed_share", "run": run, "region": reg}, x, m,
                  atol=SHARE_ATOL)
    # an older table's burned area beside an observation (before 2026-10-27)
    for name, o in (table.get("observed") or {}).items():
        for run, regs in (o.get("model") or {}).items():
            for reg, years in (regs or {}).items():
                for y, x in _yearkey(years).items():
                    L.add({"section": "observed", "what": "model", "run": run, "region": reg,
                           "year": y}, x, val(run, reg, BURNED_AREA, y))
        obs = _observed(o, observations, masks, grid, notes)
        for reg, years in (o.get("values") or {}).items():
            for y, x in _yearkey(years).items():
                L.add({"section": "observed", "what": "observation", "region": reg,
                       "year": y}, x, (obs.get(reg) or {}).get(y))
    # each month and season, and where the numbers come from (2026-11-03)
    compare_periods(table, mine, area, L, notes)
    compare_provenance(table, masks, area, dirs or {}, surface, registered, L)
    return {"ledger": L, "notes": notes}


# ─────────────────────────────────────────────────────────────────────────────
# MONTHS, SEASONS AND WHERE THE NUMBERS COME FROM
# ─────────────────────────────────────────────────────────────────────────────
# THE MONTHS AND SEASONS ARE CHECKED (2026-11-03). The table's monthly and
# seasonal values, their differences from the control and the event's normals
# were listed as not checked, so the numbers the report draws and writes to
# its CSV by month were never made a second time. Each is now compared, and a
# period is held to both ways: a value the table has that the check leaves
# out, and a period the table leaves out that the check makes, are mismatches.
MONTH_LABELS = tuple(f"{m:02d}" for m in range(1, 13))
PERIOD_KINDS = ("monthly", "seasonal")


def _period_keys(kind: str, years) -> List[str]:
    """The periods of the years a field has a value for: 12 months, or 4
    seasons, each year."""
    ys = sorted({int(y) for y in years})
    if kind == "monthly":
        return [f"{y}-{m}" for y in ys for m in MONTH_LABELS]
    return [f"{y}-{s}" for y in ys for s in SEASONS]


def _left_out(mine, run, f) -> set:
    src = BURNED_FRACTION if f == BURNED_AREA else f
    return set(((mine.get(run) or {}).get("left_out") or {}).get(src) or [])


def _pval(mine, area, run, kind, reg, f, key):
    """A run's recomputed value of one month or season; burned area is the
    period's burned fraction times the region's area."""
    t = ((mine.get(run) or {}).get(kind) or {}).get(reg) or {}
    if f == BURNED_AREA:
        x = (t.get(BURNED_FRACTION) or {}).get(key)
        return None if x is None or reg not in area else x * area[reg]
    return (t.get(f) or {}).get(key)


def _pwhy(mine, run, f, key):
    w = (mine.get(run) or {}).get("why") or {}
    src = BURNED_FRACTION if f == BURNED_AREA else f
    return w.get(f"{src} {key}") or w.get(src)


PERIOD_KEYS = ("monthly_differences", "seasonal_differences", "provenance")
# the event's keys the server writes whenever it has normal years
NORMAL_KEYS = ("monthly_normal", "seasonal_normal", "normal_counts")


def carries_periods(table: Dict[str, Any]) -> bool:
    """True for a table made since months and seasons were reported: it has
    any of their keys at the top, or months or seasons in a run."""
    runs = table.get("runs") or {}
    return any(k in table for k in PERIOD_KEYS) or \
        any(k in (r or {}) for r in runs.values() for k in PERIOD_KINDS)


def compare_periods(table: Dict[str, Any], mine: Dict[str, Any], area: Dict[str, float],
                    L: "_Ledger", notes: List[str]) -> None:
    """Each run's monthly and seasonal values, each run minus the control by
    month and season, and the event's monthly and seasonal normals.

    A table made before months and seasons were reported (before 2026-11-03)
    carries none of them, and says so in the notes; a table that carries any
    of them is held to all of them."""
    runs = table.get("runs") or {}
    if not carries_periods(table):
        notes.append("the table was made before months and seasons were reported: it "
                     "has none to check")
        return
    # which (region, field) each run has, and the years it has values for
    have: Dict[str, Dict[Tuple[str, str], set]] = {}
    for run, r in runs.items():
        hv = have.setdefault(run, {})
        for reg, fields in (r.get("values") or {}).items():
            for f, years in (fields or {}).items():
                hv.setdefault((reg, f), set()).update(str(y) for y in (years or {}))
        for kind in PERIOD_KINDS:
            for reg, fields in (r.get(kind) or {}).items():
                for f in (fields or {}):
                    hv.setdefault((reg, f), set())
    # each run's values
    for run, r in runs.items():
        why_left = r.get("seasons_left_out") or {}
        for (reg, f), years in sorted(have[run].items()):
            for kind in PERIOD_KINDS:
                tab = ((r.get(kind) or {}).get(reg) or {}).get(f) or {}
                left = _left_out(mine, run, f) if kind == "seasonal" else set()
                for key in sorted(set(_period_keys(kind, years)) | set(tab)):
                    at = {"section": kind, "run": run, "region": reg, "field": f,
                          "period": key}
                    if key in left:
                        if key in tab:
                            L.mismatch(at, tab[key], None,
                                       "the table has this season, which the check leaves "
                                       "out: the history holds no December before it")
                        else:
                            L.agree_absent()
                        continue
                    m = _pval(mine, area, run, kind, reg, f, key)
                    if key not in tab:
                        L.mismatch(at, None, m,
                                   f"the table leaves this period out "
                                   f"({why_left.get(key) or 'with no reason'}), and the check "
                                   f"makes it from the history")
                        continue
                    L.add(at, tab[key], m, why=_pwhy(mine, run, f, key))
    # each run minus the control
    ctl = have.get("control") or {}
    for kind in PERIOD_KINDS:
        sec = f"{kind}_differences"
        tab_all = table.get(sec) or {}
        for run in runs:
            if run == "control":
                continue
            pairs = {k: v for k, v in (have.get(run) or {}).items() if k in ctl}
            for reg, fields in (tab_all.get(run) or {}).items():
                for f in (fields or {}):
                    pairs.setdefault((reg, f), set())
            for (reg, f), years in sorted(pairs.items()):
                tab = ((tab_all.get(run) or {}).get(reg) or {}).get(f) or {}
                for key in sorted(set(_period_keys(kind, years)) | set(tab)):
                    a = _pval(mine, area, run, kind, reg, f, key)
                    b = _pval(mine, area, "control", kind, reg, f, key)
                    d = None if a is None or b is None else a - b
                    if d is None and key not in tab:
                        continue
                    L.add({"section": sec, "run": run, "region": reg, "field": f,
                           "period": key}, tab.get(key), d,
                          scale=max(abs(a or 0.0), abs(b or 0.0)),
                          why=None if key in tab else
                          "the table has no difference where both runs have a value")
    # the event's normals of each month and season
    ev = table.get("event") or {}
    # A NORMALS BLOCK DELETED IS NOT A PASS (2026-11-03): this table carries
    # months, and the server writes every one of these keys beside its normal
    # years, so a key missing is a block the table dropped, not one it never had.
    if not ev.get("normal_years"):
        return
    for k in NORMAL_KEYS:
        if k not in ev:
            L.mismatch({"section": "event", "what": k}, None, None,
                       f"the table has normal years but no {k} block")
    normal = [int(y) for y in ev.get("normal_years") or []]
    target = {int(y) for y in ev.get("target_years") or []}
    skip = [y for y in normal if y - 1 in target]
    used = [y for y in normal if y not in skip]
    if skip or ev.get("normal_years_used") or ev.get("normal_left_out"):
        L.same({"section": "event", "what": "normal_years_used", "period": "DJF"},
               (ev.get("normal_years_used") or {}).get("DJF"), used if skip else None)
        L.same({"section": "event", "what": "normal_left_out", "period": "DJF"},
               bool((ev.get("normal_left_out") or {}).get("DJF")), bool(skip))
    counts = ev.get("normal_counts") or {}
    for kind, labels in (("monthly_normal", MONTH_LABELS), ("seasonal_normal", tuple(SEASONS))):
        if kind not in ev:
            continue                    # one mismatch for the block, above
        per = "monthly" if kind == "monthly_normal" else "seasonal"
        tab_k = ev.get(kind) or {}
        pairs = set(ctl) | {(reg, f) for reg, fs in tab_k.items() for f in (fs or {})}
        for reg, f in sorted(pairs):
            tab = (tab_k.get(reg) or {}).get(f) or {}
            for lab in sorted(set(labels) | set(tab)):
                ys = used if lab == "DJF" else normal
                xs = [_pval(mine, area, "control", per, reg, f, f"{y}-{lab}") for y in ys]
                xs = [x for x in xs if x is not None]
                m = float(sum(xs) / len(xs)) if xs else None
                at = {"section": "event", "region": reg, "field": f, "what": kind,
                      "period": lab}
                L.add(at, tab.get(lab), m, scale=max([abs(x) for x in xs] + [0.0]))
                tc = ((counts.get(reg) or {}).get(f) or {}).get(lab)
                if tc is not None or len(xs) != len(normal):
                    L.add(dict(at, what="normal_count"),
                          tc if tc is not None else len(normal), len(xs), exact=True)


def compare_provenance(table: Dict[str, Any], masks: Dict[str, np.ndarray],
                       area: Dict[str, float], dirs: Dict[str, str], surf_path,
                       registered: Dict[str, Dict], L: "_Ledger") -> None:
    """The table's record of where its numbers come from: each file it names
    exists with the size it states, its surface file and region files are the
    ones the check read, its run directories the manager's, and each region's
    cells and area are the masks'."""
    pv = table.get("provenance")
    if pv is None:
        # A PROVENANCE BLOCK DELETED IS NOT A PASS (2026-11-03): only a table
        # made before months were reported was made without one.
        if carries_periods(table):
            L.mismatch({"section": "provenance", "what": "provenance"}, None, None,
                       "the table carries months and seasons but no provenance block")
        return
    at = {"section": "provenance"}
    sf = pv.get("surface_file")
    L.same(dict(at, what="surface_file"), _real(sf), _real(surf_path))
    for name, p in sorted((pv.get("registered_files") or {}).items()):
        r = registered.get(name) or {}
        L.same(dict(at, what="registered_file", region=name), _real(p),
               _real(r.get("polygon") or r.get("field")))
    for name, info in sorted((pv.get("regions") or {}).items()):
        w = dict(at, region=name)
        L.add(dict(w, what="cells"), (info or {}).get("cells"),
              int(masks[name].sum()) if name in masks else None, exact=True,
              why=None if name in masks else "the table's regions have no such region")
        L.add(dict(w, what="area_km2"), (info or {}).get("area_km2"), area.get(name))
    for name in sorted(set(table.get("regions") or {}) - set(pv.get("regions") or {})):
        L.mismatch(dict(at, region=name, what="cells"), None, int(masks[name].sum())
                   if name in masks else None, "the provenance block leaves this region out")
    runs = pv.get("runs") or {}
    for run in sorted(set(table.get("runs") or {}) | set(runs)):
        p = runs.get(run) or {}
        w = dict(at, run=run)
        if run not in runs:
            L.mismatch(dict(w, what="run_dir"), None, dirs.get(run),
                       "the provenance block leaves this run out")
            continue
        L.same(dict(w, what="run_dir"), _real(p.get("run_dir")), _real(dirs.get(run)))
        base = Path(str(p.get("run_dir") or ""))
        for st, s in sorted((p.get("streams") or {}).items()):
            for fi in (s or {}).get("files") or []:
                q = base / str((fi or {}).get("name"))
                L.add(dict(w, what="bytes", stream=st, file=(fi or {}).get("name")),
                      (fi or {}).get("bytes"), q.stat().st_size if q.is_file() else None,
                      exact=True, why=None if q.is_file() else "the file does not exist")


def _real(p) -> Optional[str]:
    return os.path.realpath(str(p)) if p else None


def _observed(o, observations, masks, grid, notes) -> Dict[str, Dict[str, float]]:
    """An older table's observed burned area: the registered observation's
    burned fraction per cell and year, missing as none burned, times each
    cell's weight, summed over the region."""
    import netCDF4
    ent = next((e for e in observations.values() if e.get("what") == o.get("source")),
               None) or next((e for e in observations.values()
                              if e.get("var") == "burned_fraction_year"), None)
    if not ent:
        notes.append("the table's observed burned area: no registered observation to "
                     "recompute it from")
        return {}
    with netCDF4.Dataset(ent["path"]) as f:
        _same_cells(_clean(f["lat"][:]), _clean(f["lon"][:]), grid["lat"], grid["lon"],
                    Path(ent["path"]).name)
        years = [int(y) for y in f["year"][:]]
        bf = np.nan_to_num(_clean(f[ent["var"]][:]))
    w = grid["weight"]
    return {reg: {str(y): float((bf[k] * w)[m].sum()) for k, y in enumerate(years)}
            for reg, m in masks.items()}


def check(run_dir, registry: Optional[Dict[str, Any]] = None,
          write: bool = True) -> Dict[str, Any]:
    """Recompute every number of the run's regional table and compare.

    Returns, and writes to 04_analysis/numbers_check.json: pass, how many
    numbers were compared, every mismatch, the largest relative difference,
    the method and the tolerances. A table that cannot be recomputed at all
    (an input missing) does not pass, and says why under `error`."""
    rd = Path(run_dir)
    rec: Dict[str, Any] = {
        "kind": "numbers_check", "checked_at": datetime.now().isoformat(timespec="seconds"),
        "table": TABLE, "pass": False, "n_compared": 0, "n_mismatched": 0,
        "n_absent_in_both": 0, "mismatches": [], "largest_relative_difference": None,
        "tolerance": {"relative": RTOL, "operands": EPS, "absolute": ATOL,
                      "changed_share_absolute": SHARE_ATOL,
                      "rule": "|table - recomputed| <= relative x |recomputed| + operands x "
                              "scale + absolute, the scale being the size of the quantities "
                              "a difference, anomaly or percent is made from (0 for any "
                              "other number); cell counts and ranks must be equal"},
        "method": METHOD, "sources": {}, "notes": [], "not_checked": []}
    try:
        table = json.loads((rd / TABLE).read_text())
        if table.get("error"):
            raise CheckError(f"the table holds no numbers: {table['error']}")
        inp = inputs(rd, registry)
        if "control" not in (table.get("runs") or {}):
            raise CheckError("the table has no control run")
        dirs, dir_bad, dir_src = run_dirs(rd, table, inp)
        ctl_run = dirs.get("control")
        if not ctl_run:
            raise CheckError("no record of the manager names the control's output "
                             "directory")
        h = History(ctl_run)
        grid = h.grid()
        surf_path = _fsurdat(inp["case_dir"], ctl_run)
        surface = read_surface(surf_path, grid)
        cache: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}
        for name in (table.get("regions") or {}):
            masks[name] = region_mask(name, grid, surface, inp["regions"], cache)
        extra = [BURNED_FRACTION] if table.get("observed") else []
        mine = recompute_runs(table, masks, grid, extra, dirs=dirs)
        res = compare(table, mine, masks, grid, inp["regions"], inp["observations"],
                      dirs=dirs, surface=surf_path)
        L = res["ledger"]
        L.bad[:0] = dir_bad
        rec.update({
            "n_compared": L.n, "n_absent_in_both": L.absent, "n_mismatched": len(L.bad),
            "mismatches": L.bad, "largest_relative_difference": L.worst,
            "pass": not L.bad and L.n > 0,
            "notes": res["notes"],
            "not_checked": [k for k in table if k not in KNOWN],
            "sources": {"control": inp["control"], "control_case_dir": inp["case_dir"],
                        "surface": str(surf_path),
                        "regions": {k: (v.get("polygon") or v.get("field"))
                                    for k, v in inp["regions"].items()},
                        "runs": dict(dirs), "runs_from": dict(dir_src),
                        "registry_entry_found": inp["registry_entry"]}})
        if L.n == 0:
            rec["error"] = "the table holds no number to compare"
    except Exception as e:                                          # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    if write:
        out = rd / "04_analysis"
        out.mkdir(parents=True, exist_ok=True)
        (out / CHECK_JSON).write_text(json.dumps(rec, indent=1, default=str))
    return rec


METHOD = (
    "Each number in 03_results/regional.json made again from each run's own history "
    "(its output directory read from the manager's records, not from the table) "
    "by code that shares no reduction code with the server's domain_analysis.py, "
    "gridded.py or regions.py: history read with netCDF4; fill (masked, non-finite "
    "or above 1e30 in size) as no value; each record weighted by the days of its own "
    "time_bounds inside the calendar year, the records required to cover the year "
    "without gap or overlap; a region's value at a record is the mean over its cells "
    "with a value, weighted by area x landfrac from the history; a field whose units "
    "are per second (and FAREA_BURNED) is totalled over the year, any other field is "
    "the year's time-weighted mean; cells from the surface file the control's lnd_in "
    "names (vegetated: PCT_NATVEG at 100; elevation bands: TOPO, with each band's "
    "edges read from its name), a polygon region by matplotlib's point-in-polygon test "
    "on cell centres, a gridded-field region by its rule on the file; burned area is "
    "the year's burned fraction times the region's area; differences, the event's "
    "normal, anomalies, percents and ranks by their definitions from the recomputed "
    "values. Each month and season the same way over its own days (DJF from 1 December "
    "of the year before, made only when the history covers a day or more of that "
    "December), with the run-minus-control differences and the event's normals of "
    "each month and season (the mean over the normal years, DJF without a normal "
    "year whose December is an event year's) by their definitions; a period one side "
    "leaves out and the other makes is a mismatch. The provenance block: each file it "
    "names exists with the size it states, its surface, region files and run "
    "directories are the ones the check read, and each region's cells and area are the "
    "masks'; the files' modification times and record counts are not compared.")


# ─────────────────────────────────────────────────────────────────────────────
# THE GATE
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(x) -> str:
    if x is None:
        return "none"
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return f"{float(x):.6g}"
    return str(x)


def _where(row: Dict[str, Any]) -> str:
    parts = [row.get("section")]
    for k in ("run", "region", "field", "what", "year", "period", "stream", "file"):
        if row.get(k) is not None:
            parts.append(str(row[k]))
    return ", ".join(str(p) for p in parts if p)


def reason(rec: Dict[str, Any]) -> str:
    """One line: why the numbers are withheld."""
    if rec.get("error") and not rec.get("n_compared"):
        return f"the regional numbers could not be recomputed ({rec['error']})"
    rows = rec.get("mismatches") or []
    runs = sorted({str(r.get("run")) for r in rows if r.get("section") == "runs"})
    n_num = (rec.get("n_mismatched") or 0) - sum(1 for r in rows if r.get("section") == "runs")
    parts = []
    if n_num or not runs:
        parts.append(f"{n_num} of {rec.get('n_compared')} regional numbers did not "
                     f"reproduce when recomputed from the model output")
    if runs:
        parts.append(f"the output director{'ies' if len(runs) > 1 else 'y'} the table "
                     f"names for {', '.join(runs)} do{'' if len(runs) > 1 else 'es'} not "
                     f"match the manager's records")
    return "; ".join(parts)


def withheld_markdown(rec: Dict[str, Any], set_aside_names: Sequence[str] = ()) -> str:
    """The numbers report in place of the numbers: how many did not reproduce,
    and the mismatch table. A renderer: every value is the record's."""
    lines = [f"# WITHHELD: {reason(rec)}", "",
             "The numbers in 03_results/regional.json were made again from each run's own "
             "history by a second, independent code path (04_analysis/numbers_check.json "
             "says how). "
             + ("They could not be made again, so they are not shown."
                if rec.get("error") and not rec.get("n_compared") else
                f"{rec.get('n_mismatched')} of {rec.get('n_compared')} differ by more than "
                f"the tolerance (relative {RTOL:g} of the number, with a floor of {EPS:g} "
                f"of the values it is made from and {ATOL:g} absolute)."),
             "This report is withheld and the Analyzer was not run, until the difference "
             "is explained.", ""]
    rows = rec.get("mismatches") or []
    if rows:
        lines += ["| number | in the table | recomputed | difference | relative "
                  "difference | why |", "|---|---|---|---|---|---|"]
        for r in rows[:MAX_ROWS_SHOWN]:
            lines.append(f"| {_where(r)} | {_fmt(r.get('table'))} | "
                         f"{_fmt(r.get('recomputed'))} | {_fmt(r.get('abs_diff'))} | "
                         f"{_fmt(r.get('rel_diff'))} | {r.get('why') or ''} |")
        if len(rows) > MAX_ROWS_SHOWN:
            lines.append(f"\n{len(rows) - MAX_ROWS_SHOWN} more in 04_analysis/{CHECK_JSON}.")
        lines.append("")
    if rec.get("error"):
        lines += [f"The check stopped: {rec['error']}", ""]
    if set_aside_names:
        lines += [f"An earlier analysis of this run presented these numbers; its "
                  f"{len(set_aside_names)} file(s) were moved to "
                  f"04_analysis/{Path(set_aside_names[0]).parent}/.", ""]
    return "\n".join(lines)


def gate(run_dir, registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The analyze stage's gate, run after regional.json is made and before the
    numbers report and the Analyzer.

    {"pass", "reason", "record", "report", "n_compared", "n_mismatched"}. On a
    failure, 04_analysis/DOMAIN_REPORT.md is written WITHHELD (its first line
    says how many numbers did not reproduce, then the mismatch table) and the
    caller runs neither the numbers report nor the Analyzer.

    None when the table holds no numbers (absent, or the server's error in
    their place): that failure is the numbers report's to state, and there is
    nothing here to recompute."""
    rd = Path(run_dir)
    t = _read_json(rd / TABLE)
    if not isinstance(t, dict) or not t or t.get("error"):
        return None
    try:
        rec = check(rd, registry=registry)
    except Exception as e:                                          # noqa: BLE001
        rec = {"pass": False, "n_compared": 0, "n_mismatched": 0,
               "error": f"{type(e).__name__}: {e}"}
    w = rec.get("largest_relative_difference") or {}
    out = {"pass": bool(rec.get("pass")), "record": str(rd / "04_analysis" / CHECK_JSON),
           "n_compared": rec.get("n_compared"), "n_mismatched": rec.get("n_mismatched"),
           "largest_relative_difference": w.get("rel_diff"),
           "reason": None, "report": None}
    if not out["pass"]:
        p = rd / "04_analysis" / "DOMAIN_REPORT.md"
        out["reason"] = reason(rec)
        try:
            out["set_aside"] = set_aside(rd / "04_analysis")
        except OSError as e:
            out["set_aside"] = []
            out["reason"] += f" (an earlier analysis could not be set aside: {e})"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(withheld_markdown(rec, out["set_aside"]))
            out["report"] = str(p)
        except OSError as e:
            out["reason"] += f" (the withheld report could not be written: {e})"
    return out


# A PASSING CHECK SAYS SO ON THE PAGE (2026-11-03). A failing check rewrote
# the report as WITHHELD, but a passing one left no trace a reader could see:
# nothing on DOMAIN_REPORT.md or REPORT.md said the numbers had been made
# twice. summary() reads the check's record back for the pages, the run state
# and the study record; page_line() is the one sentence both reports print.
# A record older than the table it checked is not this table's check, and is
# reported as none.
CHECK_REL = f"04_analysis/{CHECK_JSON}"


def summary(run_dir) -> Optional[Dict[str, Any]]:
    """The check of the run's current table, from 04_analysis/numbers_check.json:
    {"pass", "n_compared", "n_mismatched", "largest_relative_difference",
    "checked_at", "record", "error"}; None when there is no record, or the
    record is older than 03_results/regional.json (an earlier table's)."""
    rd = Path(run_dir)
    p, t = rd / "04_analysis" / CHECK_JSON, rd / TABLE
    rec = _read_json(p) if p.is_file() else {}
    if not isinstance(rec, dict) or rec.get("kind") != "numbers_check":
        return None
    try:
        if t.is_file() and p.stat().st_mtime < t.stat().st_mtime:
            return None
    except OSError:
        return None
    w = rec.get("largest_relative_difference") or {}
    return {"pass": bool(rec.get("pass")), "n_compared": rec.get("n_compared"),
            "n_mismatched": rec.get("n_mismatched"),
            "largest_relative_difference": w.get("rel_diff"),
            "largest_at": _where(w) if w else None,
            "checked_at": rec.get("checked_at"), "record": CHECK_REL,
            "error": rec.get("error")}


def _count(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def page_line(s: Optional[Dict[str, Any]], lead: str) -> str:
    """The sentence a report prints near its top about the check of its numbers.
    `lead` says how the page's numbers come from the table (the numbers report:
    "Every number in the answer and under The numbers is read from
    03_results/regional.json"); the check's own record supplies the rest."""
    if not s:
        return (f"**Numbers check: none on record.** There is no {CHECK_REL} for the "
                f"current {TABLE}, so its numbers were not made a second time.")
    if not s.get("pass"):
        return (f"**Numbers check: did not pass.** {_count(s.get('n_mismatched'))} of "
                f"{_count(s.get('n_compared'))} numbers in {TABLE} did not reproduce "
                f"({s['record']}).")
    w = s.get("largest_relative_difference")
    return (f"**Numbers check: passed.** {lead}, and every number in that table was made "
            f"again from the model output by an independent reading before this page was "
            f"written: {_count(s.get('n_compared'))} numbers compared, the largest relative "
            f"difference {f'{float(w):.3g}' if w is not None else 'none'} ({s['record']}).")


# AN EARLIER ANALYSIS IS SET ASIDE WHEN THE NUMBERS ARE WITHHELD (2026-11-03).
# A run analysed again (--resume, --finalize) whose numbers now fail kept the
# REPORT.md, analysis.json, deck, domain_report.json and figures of the earlier
# pass, which present the numbers that just failed to reproduce. They are
# moved, not deleted, into 04_analysis/superseded/<time>/, so nothing a person
# wrote or read is lost and nothing beside the withheld page presents them.
SUPERSEDED = "superseded"
PRODUCTS = ("REPORT.md", "analysis.json", "analysis.pptx", "domain_report.json",
            "domain_fig_*", "comparison.json", "investigation.json",
            "investigation_plan.md", "interpretation.json", "step*_round*_*.txt")


def set_aside(out_dir, names: Sequence[str] = PRODUCTS, figures: bool = True) -> List[str]:
    """Move the files out_dir holds by these names out of it, into
    out_dir/superseded/<time>/, and with `figures` every figure an Analyzer
    script made (a .png beside the .py of the same name); the moved names."""
    d = Path(out_dir)
    if not d.is_dir():
        return []
    hits = {p for pat in names for p in d.glob(pat) if p.is_file()}
    for py in (d.glob("*.py") if figures else ()):
        figs = [q for q in d.glob(py.stem + ".*") if q.suffix in (".png", ".pdf", ".svg")]
        if figs:
            hits |= {py, *figs}
    if not hits:
        return []
    dest = d / SUPERSEDED / datetime.now().strftime("%Y%m%d_%H%M%S")
    k = 1
    while dest.exists():
        dest = dest.with_name(dest.name.split("_v")[0] + f"_v{k}")
        k += 1
    dest.mkdir(parents=True)
    for p in sorted(hits):
        p.replace(dest / p.name)
    return [str(dest.relative_to(d) / p.name) for p in sorted(hits)]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: numbers_check.py <run_dir>")
    r = check(sys.argv[1])
    w = r.get("largest_relative_difference") or {}
    print(json.dumps({"pass": r["pass"], "n_compared": r["n_compared"],
                      "n_mismatched": r["n_mismatched"], "error": r.get("error"),
                      "largest_relative_difference": w.get("rel_diff"),
                      "at": _where(w) if w else None}, indent=1))
    sys.exit(0 if r["pass"] else 1)
