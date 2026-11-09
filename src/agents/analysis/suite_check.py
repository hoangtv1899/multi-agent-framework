#!/usr/bin/env python3
"""The suite numbers a report cites, made again from the model output (package F2).

A study with a suite reaches its values only through the logged `suite` calls
(domain_access), and each answered call leaves its result in
04_analysis/lookups.jsonl. The claim trace (claim_trace.trace_evidence) holds
each claim's numbers to those records, so a claim that misquotes its record
fails; it cannot see a record that is itself wrong. This module makes every
number of each record a claim cites again, from each run's own history files,
and compares; numbers_check.check_evidence runs it and step 4 withholds the
report on any mismatch.

INDEPENDENCE (plan section 3.7). Nothing here imports domain_methods,
domain_suite, domain_access, gridded, regions or domain_analysis (a test
checks the imports). The history is read with netCDF4 through numbers_check's
own reader (its file listing, its restart rule and each record's own
time_bounds); the masks are numbers_check's (the surface file the control's
lnd_in names, matplotlib's point-in-polygon test, a field region's rule on its
own file); and every method is written here again from its contract
(docs/ANALYSIS_METHODS.md, plan section 3.4): region values by the regional
table's rules (a rate's total, a state's time-weighted mean, a water year and
a season from the records over their months), the anomaly and the difference
with the percent policy, the screen's rows, the overlap, the elevation split
and the six timing measures with their reasons, flags, distributions, shifts,
changes of eligibility and coverage, and their range across settings on the
comparable cells with its reversal flag (the user's decisions on G part 1,
2026-11-08). What is shared is the definitions, the
field notes (each field's type and percent policy, read as JSON), and where
the inputs are: the manager's records name each run's output directory (never
a suite's manifest), the control's registry entry names the regions, the band
edges and the start months, and the analysis plan names the normal years.

WHAT IS READ. Only the fields, runs and years the cited records name: a
record's own field and period with its normal years, and for a screen only the
rows a claim states (the claim trace's matched values). An overlap or an
elevation split names its maps by their hashes: each must be an array a logged
lookup returned (get_map, or a map of a per-cell anomaly or difference), and
it is made again from that lookup's own call. A map made by a script's own
arithmetic cannot be made again, so its record fails and says why.

PRECISION AND TOLERANCE (plan section 3.2). The history is float32 and is
reduced here in float64. A per-cell map is rounded to float32, the precision
the suite stores maps in, before anything is made from it, so an equal-area
quarter, a sign or a band holds the same cells. A number is reproduced when
|logged - recomputed| <= RTOL x |recomputed| + EPS x scale + ATOL, the numbers
check's own rule (RTOL 1e-5, EPS 1e-9 of the values a difference, anomaly or
percent is made from, ATOL 1e-9). Counts, reason codes, statuses, marks and
flags must be equal, and a date or a shift in days must agree within 1e-9 of
a day.

  check(study_dir, claims, registry=None, write=True)   the comparison record
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.analysis import numbers_check as NC

EVIDENCE_JSON = "evidence_numbers_check.json"
LOOKUPS = "lookups.jsonl"
PLAN_JSON = "analysis_plan.json"
NOTES_JSON = Path(__file__).resolve().parents[3] / "mcp" / "elm-mcp" / "domain_field_notes.json"
DAY = 86400.0
VEG = NC.VEGETATED
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
SEASON_FIRST = {"DJF": 12, "MAM": 3, "JJA": 6, "SON": 9}
DAY_TOL = 1e-9
# The methods' defaults, from their contracts (docs/ANALYSIS_METHODS.md)
DEFAULTS = {"anomaly": {"near_zero_share": 0.01}, "difference": {"near_zero_share": 0.01},
            "screen": {"near_constant_share": 0.01}, "overlap": {"sign_share_limit": 0.8},
            "elevation_split": {"small_share": 0.01}}
TIMING_KIND = {"snow_peak": "wateryear", "snow_gone": "wateryear", "half_runoff": "wateryear",
               "leaf_green_up": "year", "photosynthesis_onset": "year", "growth_peak": "year"}
TIMING_FIELD = {"snow_peak": "H2OSNO", "snow_gone": "H2OSNO", "leaf_green_up": "TLAI",
                "photosynthesis_onset": "GPP", "growth_peak": "GPP", "half_runoff": "QRUNOFF"}
# Each outcome's name (the contract's) and the code timing.nc stores
REASON = {"date": 0, "records_missing": 1, "before_record": 2, "no_snow": 3, "snow_persists": 4,
          "no_leaves": 5, "little_leaf_change": 6, "no_dormant_season": 7,
          "no_photosynthesis": 8, "no_runoff": 9, "insufficient_follow_up": 10,
          "cell_values_missing": 11}
FLAG = {"snow_returns": 1, "partial_record": 2, "two_peaks": 4, "runoff_falls": 8}
METHOD_CALLS = ("anomaly", "difference", "screen", "overlap", "elevation_split") + \
    tuple(TIMING_KIND)
NO_NUMBER_CALLS = ("get_map", "get_series", "get_timing")
SCREEN_COLS = ("value", "reference", "change", "percent", "score", "status", "mark",
               "n_normal", "near_constant")
ROW_VALUES = ("value", "reference", "change", "percent", "score")
MAP_PATHS = {"anomaly": ("anomaly", "value", "normal", "percent"),
             "difference": ("difference", "run_value", "control_value", "percent")}
_CACHE: Dict[str, Any] = {}


class Absent(Exception):
    """A period the run's records do not cover whole."""


CheckError = NC.CheckError


# ─────────────────────────────────────────────────────────────────────────────
# PERIODS
# ─────────────────────────────────────────────────────────────────────────────
def parse_period(label) -> Tuple[str, int, Any]:
    s = str(label).strip()
    if re.fullmatch(r"WY\d{4}", s):
        return "wateryear", int(s[2:]), None
    if re.fullmatch(r"\d{4}", s):
        return "year", int(s), None
    m = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return "month", int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(\d{4})-(DJF|MAM|JJA|SON)", s)
    if m:
        return "season", int(m.group(1)), m.group(2)
    raise CheckError(f"{label!r} is not a period label")


def label_of(kind: str, year: int, part: Any = None) -> str:
    if kind == "month":
        return f"{year}-{int(part):02d}"
    return {"year": f"{year}", "wateryear": f"WY{year}", "season": f"{year}-{part}"}[kind]


def _add_months(y: int, m: int, k: int) -> Tuple[int, int]:
    t = y * 12 + (m - 1) + k
    return t // 12, t % 12 + 1


def months_of(kind: str, year: int, part: Any = None) -> List[Tuple[int, int]]:
    """(year, month) of each month a period holds: a water year from October of
    the year before, DJF with the December before."""
    if kind == "year":
        return [(year, m) for m in range(1, 13)]
    if kind == "wateryear":
        return [_add_months(year - 1, 10, k) for k in range(12)]
    if kind == "season":
        return [_add_months(*((year - 1, 12) if part == "DJF" else (year, SEASON_FIRST[part])), k)
                for k in range(3)]
    return [(year, int(part))]


def span(kind: str, year: int, part: Any = None):
    ms = months_of(kind, year, part)
    end = _add_months(*ms[-1], 1)
    return (ms[0][0], ms[0][1], 1), (end[0], end[1], 1)


def left_out(kind: str, part: Any, normal_year: int, event_years: Sequence[int]) -> bool:
    """A normal year's period holding a month of an event year other than
    itself (the DJF or the water year after an event year) stays out."""
    yrs = {y for y, _ in months_of(kind, normal_year, part)} - {int(normal_year)}
    return bool(yrs & {int(e) for e in event_years})


def kind_parts(kind: str) -> List[Any]:
    return list(range(1, 13)) if kind == "month" else list(SEASON_ORDER)


# ─────────────────────────────────────────────────────────────────────────────
# ONE RUN'S HISTORY
# ─────────────────────────────────────────────────────────────────────────────
class Run:
    """One run's history files, read with netCDF4 through numbers_check's
    reader; values are per cell, flattened row by row over the grid."""

    def __init__(self, run_dir, like: Optional["Run"] = None):
        self.dir = str(run_dir)
        self.h = NC.History(run_dir)
        g = self.h.grid()
        if like is not None:
            NC._same_cells(g["lat"], g["lon"], like.lat, like.lon, f"the history of {run_dir}")
        self.lat, self.lon = g["lat"], g["lon"]
        self.w = np.asarray(g["weight"], dtype=np.float64).ravel()
        self._tb: Dict[str, List[Tuple[Path, str, str, np.ndarray]]] = {}
        self._x: Dict[Tuple[str, str], np.ndarray] = {}
        self._topw: Optional[np.ndarray] = None
        self._days: Dict[str, Any] = {}
        self._units: Dict[str, str] = {}
        self.reads: set = set()

    def forget(self) -> None:
        self._x.clear()

    def stamp(self) -> List[Any]:
        return sorted([p.name, p.stat().st_size, p.stat().st_mtime_ns]
                      for fs in self.h.files.values() for p in fs)

    def bounds(self, stream: str):
        if stream not in self._tb:
            import netCDF4
            out, prev = [], None
            for p in self.h.files[stream]:
                with netCDF4.Dataset(p) as d:
                    t = d["time"]
                    bn = getattr(t, "bounds", None) or "time_bounds"
                    if bn not in d.variables:
                        raise CheckError(f"{p.name} has no time bounds")
                    units, cal = str(t.units), str(getattr(t, "calendar", "standard"))
                    tb = np.asarray(d[bn][:], dtype=np.float64)
                tb, prev = NC.restart_clip(tb, prev, units, cal)
                out.append((p, units, cal, tb))
            self._tb[stream] = out
        return self._tb[stream]

    def units(self, var: str) -> str:
        if var not in self._units:
            st = self.h.stream_for(var)
            self._units[var] = self.h.units(st, var) if st else ""
        return self._units[var]

    def rate(self, var: str) -> bool:
        return NC.per_second(var, self.units(var))

    def _top_weights(self) -> np.ndarray:
        """(levels, cells): each layer's thickness inside the top metre over
        their sum, from DZSOI of the first file that holds it."""
        if self._topw is None:
            import netCDF4
            dz = None
            for st in self.h.streams():
                with netCDF4.Dataset(self.h.files[st][0]) as d:
                    if "DZSOI" in d.variables:
                        dz = NC._clean(d["DZSOI"][:])
                        break
            if dz is None:
                raise CheckError("no history file carries DZSOI, so a top-1 m mean "
                                 "cannot be made")
            dz = dz.reshape(dz.shape[0], -1)
            top = np.cumsum(dz, axis=0) - dz
            with np.errstate(invalid="ignore"):
                inside = np.clip(np.minimum(top + dz, 1.0) - top, 0.0, None)
            tot = inside.sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                w = inside / tot
            w[:, ~(np.isfinite(dz).all(axis=0) & (tot > 0))] = np.nan
            self._topw = w
        return self._topw

    def raw(self, path: Path, var: str) -> np.ndarray:
        """Every record of var in one file, (records, cells) float32, NaN for
        no value; a layered field as its top-1 m mean."""
        key = (str(path), var)
        if key not in self._x:
            import netCDF4
            with netCDF4.Dataset(path) as d:
                v = d[var]
                v.set_auto_maskandscale(True)
                a = NC._clean(v[:])
            self.reads.add((var, path.name))
            if a.ndim == 4:
                if var != "H2OSOI":
                    raise CheckError(f"{var} has levels; only H2OSOI's top metre is made here")
                w = self._top_weights()
                a = a.reshape(a.shape[0], a.shape[1], -1)
                use = np.isfinite(w) & (w > 0)
                y = np.einsum("nlc,lc->nc", np.where(use[None], a, 0.0), np.where(use, w, 0.0))
                bad = (use[None] & ~np.isfinite(a)).any(axis=1) | \
                    ~np.isfinite(w).all(axis=0)[None]
                y[bad] = np.nan
                a = y
            elif a.ndim == 3:
                a = a.reshape(a.shape[0], -1)
            else:
                raise CheckError(f"{var} has {a.ndim} dimensions; (time, lat, lon) expected")
            self._x[key] = a.astype(np.float32)
        return self._x[key]

    # ── periods from the records ───────────────────────────────────────
    def records(self, var: str, start, end) -> Tuple[np.ndarray, np.ndarray]:
        """(values (records, cells) float64, each record's days inside
        [start, end)), the records tiling the interval by their own
        time_bounds, from the first stream that holds var (Absent if not)."""
        st = self.h.stream_for(var)
        if st is None:
            raise CheckError(f"no history stream of {self.dir} holds {var}")
        got = []
        length = None
        for p, units, cal, tb in self.bounds(st):
            a, b = NC._day(start, units, cal), NC._day(end, units, cal)
            length = b - a
            lo, hi = np.maximum(tb[:, 0], a), np.minimum(tb[:, 1], b)
            idx = np.flatnonzero(hi - lo > NC.TIME_TOL_DAYS)
            if not idx.size:
                continue
            x = self.raw(p, var)
            got += [(float(lo[k] - a), float(hi[k] - a), x[k]) for k in idx]
        if not got:
            raise Absent(f"no record of {var} in it")
        got.sort(key=lambda r: r[0])
        edge = 0.0
        for lo, hi, _ in got:
            if abs(lo - edge) > NC.TIME_TOL_DAYS:
                raise Absent(f"the records of {var} leave a gap or overlap at day {edge:.4f}")
            edge = hi
        if abs(edge - length) > NC.TIME_TOL_DAYS:
            raise Absent(f"the records of {var} end at day {edge:.4f} of {length:g}")
        return (np.stack([r[2] for r in got]).astype(np.float64),
                np.array([r[1] - r[0] for r in got]))

    def period_map(self, var: str, label: str) -> np.ndarray:
        """The period's per-cell value, float32 as the suite stores maps, then
        float64: a rate's total, a state's time-weighted mean; a cell with no
        value in any record of the period has none (Absent when no cell has)."""
        kind, y, part = parse_period(label)
        x, days = self.records(var, *span(kind, y, part))
        with np.errstate(invalid="ignore"):
            tot = (x * days[:, None]).sum(axis=0)
        v = tot * DAY if self.rate(var) else tot / days.sum()
        v = v.astype(np.float32).astype(np.float64)
        if not np.isfinite(v).any():
            raise Absent(f"no cell has a value of {var} in {label}")
        return v

    def region_value(self, var: str, label: str, mask: np.ndarray
                     ) -> Tuple[float, float]:
        """(the region's value, the share of its area covered: the smallest
        over the period's records). Each record's value is the area-weighted
        mean over the region's cells with a value."""
        kind, y, part = parse_period(label)
        x, days = self.records(var, *span(kind, y, part))
        m = np.asarray(mask, dtype=bool).ravel()
        W = self.w[m]
        xs = x[:, m]
        ok = np.isfinite(xs)
        num = (np.where(ok, xs, 0.0) * W).sum(axis=1)
        den = (ok * W).sum(axis=1)
        tot = float(W.sum())
        if tot <= 0 or not (den > 0).all():
            raise Absent(f"a record of {label} has no value of {var} in the region")
        means = num / den
        s = float((means * days).sum())
        v = s * DAY if self.rate(var) else s / float(days.sum())
        return v, float((den / tot).min())

    # ── days, for the timing measures ──────────────────────────────────
    def daily_stream(self, var: str) -> str:
        for st in self.h.streams():
            if var not in self.h.variables(st):
                continue
            tb = self.bounds(st)[0][3]
            L = tb[:, 1] - tb[:, 0]
            if L.size and abs(float(np.median(L)) - 1.0) < 0.01:
                return st
        raise CheckError(f"no daily history stream of {self.dir} holds {var}")

    def day_index(self, var: str) -> Tuple[Dict[int, List], str, str, str]:
        """{calendar year: [(file, record, day of the year - 1)]}: each daily
        record by the day it ends in."""
        if var not in self._days:
            self._days[var] = self._day_index(var)
        return self._days[var]

    def _day_index(self, var: str):
        import netCDF4
        st = self.daily_stream(var)
        out: Dict[int, List] = {}
        units = cal = None
        for p, units, cal, tb in self.bounds(st):
            L = tb[:, 1] - tb[:, 0]
            good = np.flatnonzero((L > 0.5) & (L < 1.5))
            if not good.size:
                continue
            y0 = netCDF4.num2date(tb[good[0], 1] - 0.5, units, calendar=cal).year
            y1 = netCDF4.num2date(tb[good[-1], 1] - 0.5, units, calendar=cal).year
            for y in range(y0, y1 + 1):
                a = NC._day((y, 1, 1), units, cal)
                n = int(round(NC._day((y + 1, 1, 1), units, cal) - a))
                idx = np.round(tb[good, 1] - a).astype(np.int64) - 1
                for k, j in zip(good, idx):
                    if 0 <= j < n:
                        out.setdefault(y, []).append((p, int(k), int(j)))
        return out, st, units, cal

    def days(self, var: str, year: int) -> Optional[np.ndarray]:
        """(days of the year, cells) float64 of one calendar year, NaN where a
        day has no record or a cell no value; a rate as the day's total (the
        rate times 86,400 s). None when the run holds no record of the year."""
        idx, st, units, cal = self.day_index(var)
        if year not in idx:
            return None
        n = int(round(NC._day((year + 1, 1, 1), units, cal) - NC._day((year, 1, 1), units, cal)))
        out = np.full((n, self.w.size), np.nan)
        for p, k, j in idx[year]:
            out[j] = self.raw(p, var)[k]
        if NC.per_second(var, self.h.units(st, var)):
            out = out * DAY
        return out

    def calendar_info(self, var: str):
        idx, st, units, cal = self.day_index(var)
        return sorted(idx), units, cal


# ─────────────────────────────────────────────────────────────────────────────
# SMALL STATISTICS (written from the contracts)
# ─────────────────────────────────────────────────────────────────────────────
def wquantile(values, weights, q: float) -> Optional[float]:
    """The smallest value at which the cumulative share of the area reaches
    q; the midpoint of it and the next value where it reaches q exactly."""
    x = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return None
    x, w = x[keep], w[keep]
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    c = np.cumsum(w)
    goal, tol = q * c[-1], 1e-12 * c[-1]
    k = int(np.argmax(c >= goal - tol)) if (c >= goal - tol).any() else x.size - 1
    if abs(c[k] - goal) <= tol and k + 1 < x.size:
        return float((x[k] + x[k + 1]) / 2.0)
    return float(x[k])


def share(w: np.ndarray, part: np.ndarray, whole: np.ndarray) -> float:
    t = float(w[whole].sum())
    return float(w[part].sum()) / t if t > 0 else 0.0


def missing(x: np.ndarray, region: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
    reg = region & (w > 0)
    gone = reg & ~np.isfinite(x)
    return {"cells": int(gone.sum()), "area_share": share(w, gone, reg)}


def wcorr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> Optional[float]:
    p = w / w.sum()
    da, db = a - (p * a).sum(), b - (p * b).sum()
    va, vb = (p * da * da).sum(), (p * db * db).sum()
    if not (va > 0 and vb > 0):
        return None
    return float((p * da * db).sum() / math.sqrt(va * vb))


def quarters(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> List[Dict[str, float]]:
    """The first map's mean in each quarter of the area, cells ranked by the
    second (ties in cell order), a cell across a quarter's edge split."""
    o = np.argsort(b, kind="stable")
    ws = w[o]
    hi = np.cumsum(ws)
    lo = hi - ws
    tot = hi[-1]
    out = []
    for k in range(4):
        part = np.clip(np.minimum(hi, tot * (k + 1) / 4.0) - np.maximum(lo, tot * k / 4.0),
                       0.0, None)
        has = part > 0
        bb, aa = b[o][has], a[o][has]
        out.append({"area_km2": float(part.sum()), "second_map_from": float(bb.min()),
                    "second_map_to": float(bb.max()),
                    "first_map_mean": float((part[has] * aa).sum() / part.sum())})
    return out


def concentration(x: np.ndarray, w: np.ndarray) -> Optional[float]:
    """The smallest share of the area that holds half of the total absolute
    value: cells largest first, the last by the part of its area needed."""
    ax = np.abs(x)
    tot = float((ax * w).sum())
    if not tot > 0:
        return None
    o = np.argsort(-ax, kind="stable")
    held = np.cumsum((ax * w)[o])
    area = np.cumsum(w[o])
    k = int(np.argmax(held >= 0.5 * tot))
    held0, area0 = (held[k - 1], area[k - 1]) if k > 0 else (0.0, 0.0)
    return float((area0 + (0.5 * tot - held0) / ax[o][k]) / w.sum())


def percent_ok(note: Dict[str, Any]) -> bool:
    """The field notes' percent policy: never for an event or a bounded index,
    and only where the notes allow it."""
    if note.get("type") in ("event", "bounded"):
        return False
    pc = note.get("percent")
    return bool(isinstance(pc, dict) and pc.get("allowed") is True)


def far_from_zero(ref, rr, frac: float):
    r = np.asarray(ref, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(r) & (r != 0) & ~(np.abs(r) < frac * np.asarray(rr, dtype=np.float64))


# ─────────────────────────────────────────────────────────────────────────────
# THE CONTEXT: RUNS, MASKS, NOTES, PLAN
# ─────────────────────────────────────────────────────────────────────────────
class Context:
    def __init__(self, study_dir, registry: Optional[Dict[str, Any]] = None):
        self.study = Path(study_dir)
        self.out = self.study / "04_analysis"
        self.inp = NC.inputs(self.study, registry)
        self.entry = registry if registry is not None else NC._registry(self.inp["control"])
        self.lines = _read_lines(self.out / LOOKUPS)
        self.plan = _read_json(self.out / PLAN_JSON)
        self.notes = (_read_json(NOTES_JSON).get("fields") or {})
        self.ws = int(self.entry.get("water_year_start_month") or 10)
        self.ys = int(self.entry.get("year_start_month") or 1)
        self._runs: Dict[str, Run] = {}
        self._dirs: Dict[str, str] = {}
        self._masks: Dict[str, np.ndarray] = {}
        self._mcache: Dict[str, np.ndarray] = {}
        self._surface = None
        self.memo: Dict[Tuple, Any] = {}

    # where each run's output is: the manager's records, never a manifest
    def run_dir(self, run: str) -> str:
        if run in self._dirs:
            return self._dirs[run]
        if run == "control":
            d = self.inp.get("control_run_dir")
            if not d:
                raise CheckError("no record of the manager names the control's output "
                                 "directory (01_inputs/domain_control_facts.json)")
            reg = self.entry.get("run_dir")
            if reg and not NC._same_dir(reg, d):
                raise CheckError(f"the control registry names {reg}, the manager's record {d}")
        else:
            if "/" in run or run in (".", ".."):
                raise CheckError(f"{run!r} is not a run name")
            found = sorted((self.study / "domain_case" / run / "checks").glob("y*.json"))
            named = sorted({str(_read_json(p).get("run_dir")) for p in found
                            if _read_json(p).get("run_dir")})
            if not named:
                raise CheckError(f"no record of the manager names run {run}'s output directory "
                                 f"(domain_case/{run}/checks/y*.json)")
            if len(named) > 1:
                raise CheckError(f"run {run}'s year checks name different directories: "
                                 f"{', '.join(named)}")
            d = named[0]
        self._dirs[run] = str(d)
        return str(d)

    def run(self, run: str) -> Run:
        if run not in self._runs:
            base = self._runs.get("control") if run != "control" else None
            if run != "control" and base is None:
                base = self.run("control")
            self._runs[run] = Run(self.run_dir(run), like=base)
        return self._runs[run]

    @property
    def w(self) -> np.ndarray:
        return self.run("control").w

    def surface(self):
        if self._surface is None:
            c = self.run("control")
            g = {"lat": c.lat, "lon": c.lon}
            self._surface = NC.read_surface(
                NC._fsurdat(self.inp.get("case_dir"), self.run_dir("control")), g)
        return self._surface

    def mask(self, name: str) -> np.ndarray:
        """A region's cells, rebuilt from its name by numbers_check's rules;
        'not <region>' is no region of its own (elevation_split's complement)."""
        if name not in self._masks:
            c = self.run("control")
            m = NC.region_mask(name, {"lat": c.lat, "lon": c.lon}, self.surface(),
                               self.inp["regions"], self._mcache)
            self._masks[name] = np.asarray(m, dtype=bool).ravel()
        return self._masks[name]

    def band_names(self) -> Optional[List[str]]:
        edges = self.entry.get("elevation_band_edges")
        if not edges:
            return None
        e = [float(x) for x in edges]
        return ([f"below {e[0]:g} m"] + [f"{a:g}-{b:g} m" for a, b in zip(e, e[1:])]
                + [f"{e[-1]:g} m and above"])

    def other_regions(self) -> List[str]:
        return [f"{k}{NC.VEG_SUFFIX}" for k in self.inp["regions"]]

    def normal_years(self, keys: Dict[str, Any]) -> List[int]:
        ny = keys.get("normal_years")
        if ny is None:
            ny = (self.plan.get("baseline") or {}).get("normal_years")
        if not ny:
            raise CheckError("the normal years this lookup used are on no record the check "
                             "reads (its keys name none and the analysis plan names none)")
        return [int(y) for y in ny]

    def plan_normal_years(self) -> List[int]:
        return [int(y) for y in (self.plan.get("baseline") or {}).get("normal_years") or []]

    def note(self, field: str) -> Dict[str, Any]:
        return self.notes.get(field) or {}

    def stamps(self, runs: Sequence[str]) -> List[Any]:
        return [[r, self.run_dir(r), self.run(r).stamp()] for r in sorted(set(runs))]


def _read_json(p) -> Dict[str, Any]:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}


def _read_lines(p) -> List[Dict[str, Any]]:
    try:
        return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    except OSError:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# REGION VALUES, NORMALS, ANOMALY AND DIFFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def _getter(ctx: Context, run: str, field: str, region: Optional[str]):
    """label -> (value or map, covered share) or (None, None) where absent;
    each made once per check (ctx.memo)."""
    r = ctx.run(run)

    def get(label):
        key = (run, field, str(label), region)
        if key not in ctx.memo:
            try:
                ctx.memo[key] = (r.period_map(field, label), None) if region is None else \
                    r.region_value(field, label, ctx.mask(region))
            except Absent:
                ctx.memo[key] = (None, None)
        return ctx.memo[key]
    return get


def normal_of(get, kind, part, normal_years, event_years) -> Dict[str, Any]:
    used, left, miss, incomplete, cov = {}, [], [], [], {}
    for y in normal_years:
        if left_out(kind, part, y, event_years):
            left.append(int(y))
            continue
        v, c = get(label_of(kind, int(y), part))
        if v is None:
            miss.append(int(y))
            continue
        if c is not None and c < 1.0:
            incomplete.append(int(y))
        cov[int(y)] = c
        used[int(y)] = v
    return {"used": used, "left": left, "missing": miss, "incomplete": incomplete, "cov": cov}


def _mean(vals: Dict[int, Any]):
    if not vals:
        return None
    a = np.stack([np.asarray(v, dtype=np.float64) for v in vals.values()])
    return a.mean(axis=0) if a.ndim > 1 else float(a.mean())


def ref_range(get, kind, normal_years, event_years):
    """The range of the field's normals over the periods of the kind (twelve
    months for a month, the four seasons for anything else)."""
    rk = "month" if kind == "month" else "season"
    ns = []
    for part in kind_parts(kind):
        m = _mean(normal_of(get, rk, part, normal_years, event_years)["used"])
        if m is not None:
            ns.append(np.asarray(m, dtype=np.float64))
    if not ns:
        return None
    a = np.stack(ns)
    return a.max(axis=0) - a.min(axis=0) if a.ndim > 1 else float(a.max() - a.min())


def _support(used: Dict[int, Any]) -> Dict[str, Any]:
    a = np.array([float(v) for v in used.values()])
    return {"n": int(a.size), "years": sorted(used), "sd": float(a.std(ddof=1)) if a.size > 1
            else None, "min": float(a.min()) if a.size else None,
            "max": float(a.max()) if a.size else None}


def _coverage(out, shares: Dict[str, Optional[float]], lacking: List[str]) -> None:
    for k, v in shares.items():
        out[f"coverage.area_share.{k}"] = v
    have = [v for v in shares.values() if v is not None]
    out["coverage.least_area_share"] = min(have) if have else None
    out["coverage.missing_cell_values"] = [k for k, v in shares.items()
                                           if v is not None and v < 1.0]
    out["coverage.missing_time_records"] = list(lacking)


def cover_mask(ctx: Context) -> np.ndarray:
    return ctx.mask(VEG) & (ctx.w > 0)


def anomaly(ctx: Context, keys: Dict[str, Any], settings: Dict[str, Any]
            ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    field, period, region = keys["field"], str(keys["period"]), keys.get("region")
    run = keys.get("run") or "control"
    kind, year, part = parse_period(period)
    ny = ctx.normal_years(keys)
    ev = [year]
    get = _getter(ctx, run, field, region)
    out: Dict[str, Any] = {"normal_years": ny, "event_years": ev}
    tol: Dict[str, Any] = {}
    maps: Dict[str, np.ndarray] = {}
    value, cov = get(period)
    n = normal_of(get, kind, part, ny, ev)
    normal = _mean(n["used"])
    out.update({"normal_left_out": sorted(n["left"]), "normal_missing": n["missing"],
                "normal_incomplete": n["incomplete"]})
    lab = lambda y: label_of(kind, y, part)                      # noqa: E731
    if region is None:
        cm = cover_mask(ctx)
        sh = lambda x: share(ctx.w, cm & np.isfinite(x), cm)     # noqa: E731
        shares = {period: sh(value)} if value is not None else {}
        shares.update({lab(y): sh(x) for y, x in n["used"].items()})
        out["coverage.over"] = VEG
    else:
        out.update(value=value, covered=cov, normal=normal)
        shares = {period: cov} if value is not None else {}
        shares.update({lab(y): n["cov"].get(y) for y in n["used"]})
    _coverage(out, shares, ([] if value is not None else [period]) +
              [lab(y) for y in n["missing"]])
    if value is None or normal is None:
        out.update(anomaly=None, percent=None)
        return out, tol, maps
    a = np.asarray(value, dtype=np.float64) - np.asarray(normal, dtype=np.float64)
    rr = ref_range(get, kind, ny, ev)
    ok = percent_ok(ctx.note(field))
    frac = settings["near_zero_share"]
    if region is None:
        cm = cover_mask(ctx)
        out["coverage.result.cells"] = int(cm.sum())
        out["coverage.result.missing_cells"] = int((cm & ~np.isfinite(a)).sum())
        out["coverage.result.area_share"] = share(ctx.w, cm & np.isfinite(a), cm)
        out["support.n"] = len(n["used"])
        out["support.years"] = sorted(n["used"])
        if len(n["used"]) < 2:
            out["support.sd"] = None
        pct = None
        if ok and rr is not None:
            good = far_from_zero(normal, rr, frac)
            with np.errstate(invalid="ignore", divide="ignore"):
                pct = np.where(good, 100.0 * a / np.where(good, normal, 1.0), np.nan)
        else:
            out["percent"] = None
        if rr is None:
            out["reference_range"] = None
        maps.update(anomaly=a, value=np.asarray(value), normal=np.asarray(normal))
        if pct is not None:
            maps["percent"] = pct
        return out, tol, maps
    a = float(a)
    pct = None
    if ok and rr is not None and bool(far_from_zero(normal, rr, frac)):
        pct = 100.0 * a / float(normal)
    sup = _support(n["used"])
    out.update({"anomaly": a, "percent": pct, "reference_range": rr,
                **{f"support.{k}": v for k, v in sup.items()}})
    sc = max(abs(value), abs(normal))
    tol.update(anomaly=sc, percent=100.0 * sc / abs(normal) if normal else 0.0,
               reference_range=sc)
    return out, tol, maps


def difference(ctx: Context, keys: Dict[str, Any], settings: Dict[str, Any]
               ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    field, period, region = keys["field"], str(keys["period"]), keys.get("region")
    run, ctl = keys["run"], keys.get("control") or "control"
    kind, year, part = parse_period(period)
    gr, gc = _getter(ctx, run, field, region), _getter(ctx, ctl, field, region)
    rv, rcov = gr(period)
    cv, ccov = gc(period)
    if rv is None or cv is None:            # a period either run lacks: neither is compared
        rv = cv = rcov = ccov = None
    out: Dict[str, Any] = {}
    tol: Dict[str, Any] = {}
    maps: Dict[str, np.ndarray] = {}
    if region is not None:
        out.update(run_value=rv, control_value=cv)
    shares, lacking = {}, []
    cm = cover_mask(ctx) if region is None else None
    for who, v, c in (("run", rv, rcov), ("control", cv, ccov)):
        if v is None:
            lacking.append(f"{who} {period}")
        else:
            shares[f"{who} {period}"] = share(ctx.w, cm & np.isfinite(v), cm) \
                if region is None else c
    _coverage(out, shares, lacking)
    if region is None:
        out["coverage.over"] = VEG
    if rv is None or cv is None:
        out.update(difference=None, percent=None)
        return out, tol, maps
    d = np.asarray(rv, dtype=np.float64) - np.asarray(cv, dtype=np.float64)
    ny = ctx.plan_normal_years()
    if ny:
        rr = ref_range(gc, kind, ny, [year])
        out["reference_range_from"] = "the control's normals"
    else:
        rk = "month" if kind == "month" else "season"
        vals = [gc(label_of(rk, year, p))[0] for p in kind_parts(kind)]
        vals = [np.asarray(v, dtype=np.float64) for v in vals if v is not None]
        rr = (np.stack(vals).max(axis=0) - np.stack(vals).min(axis=0)) if vals else None
        out["reference_range_from"] = f"the control's {rk} values of {year}"
    ok = percent_ok(ctx.note(field))
    frac = settings["near_zero_share"]
    if region is None:
        out["coverage.result.cells"] = int(cm.sum())
        out["coverage.result.missing_cells"] = int((cm & ~np.isfinite(d)).sum())
        out["coverage.result.area_share"] = share(ctx.w, cm & np.isfinite(d), cm)
        maps.update(difference=d, run_value=np.asarray(rv), control_value=np.asarray(cv))
        if ok and rr is not None:
            good = far_from_zero(cv, rr, frac)
            with np.errstate(invalid="ignore", divide="ignore"):
                maps["percent"] = np.where(good, 100.0 * d / np.where(good, cv, 1.0), np.nan)
        else:
            out["percent"] = None
        if rr is None:
            out["reference_range"] = None
        return out, tol, maps
    pct = None
    if ok and rr is not None and bool(far_from_zero(cv, rr, frac)):
        pct = 100.0 * float(d) / float(cv)
    out.update(difference=float(d), percent=pct, reference_range=rr)
    sc = max(abs(rv), abs(cv))
    tol.update(difference=sc, percent=100.0 * sc / abs(cv) if cv else 0.0, reference_range=sc)
    return out, tol, maps


# ─────────────────────────────────────────────────────────────────────────────
# THE SCREEN'S ROWS
# ─────────────────────────────────────────────────────────────────────────────
def screen_row(ctx: Context, run: str, control: Optional[str], field: str, period: str,
               region: str, normal_years: List[int], frac: float) -> Dict[str, Any]:
    """One row of the screen by its contract: the anomaly against the normal
    years (the control) or the difference from the control (another run),
    scored by the field's type."""
    note = ctx.note(field)
    typ = note.get("type")
    kind, year, part = parse_period(period)
    base = control or run
    m = ctx.mask(region)
    area = float(ctx.w[m].sum())
    gb = _getter(ctx, base, field, region)
    row: Dict[str, Any] = dict.fromkeys(SCREEN_COLS + ("_sd",))
    v, cov = _getter(ctx, run, field, region)(period)
    if control is not None and gc_absent(ctx, control, field, region, period):
        row["status"] = "period absent"
        return row
    if v is None:
        row["status"] = "period absent"
        return row
    n = normal_of(gb, kind, part, normal_years, [year])
    vals = [float(x) for x in n["used"].values()]
    sup = _support(n["used"])
    rr = ref_range(gb, kind, normal_years, [year])
    row["n_normal"] = sup["n"]
    row["_sd"] = sup["sd"]
    if typ == "event":
        conv = (lambda x: None if x is None else float(x) * area) \
            if field == NC.BURNED_FRACTION else (lambda x: None if x is None else float(x))
        vv = conv(v)
        nv = [conv(x) for x in vals]
        ref = conv(gb(period)[0]) if control is not None else \
            (float(np.mean(nv)) if nv else None)
        row.update(value=vv, reference=ref)
        if vv is None or ref is None:
            row["status"] = "insufficient support"
            return row
        row["change"] = vv - ref
        if row["change"] == 0.0:
            row["status"] = "no change"
        elif control is not None:
            row.update(status="marked", mark="differs_from_control")
        elif sup["n"] < 2 or n["incomplete"]:
            row["status"] = "insufficient support"
        elif vv > max(nv):
            row.update(status="marked", mark="above_normal_maximum")
        else:
            row["status"] = "within the normal years' range"
        return row
    ref = gb(period)[0] if control is not None else (float(np.mean(vals)) if vals else None)
    row.update(value=v, reference=ref)
    if ref is None:
        row["status"] = "insufficient support"
        return row
    ch = v - ref
    row["change"] = ch
    if percent_ok(note) and rr is not None and bool(far_from_zero(ref, rr, 0.01)):
        row["percent"] = 100.0 * ch / ref
    if ch == 0.0:
        row["status"] = "no change"
    elif typ in ("store", "bounded"):
        row["status"] = "not scored yet"
    elif typ not in ("flux", "state"):
        row["status"] = "no type in the field notes"
    elif sup["n"] < 2 or n["incomplete"]:
        row["status"] = "insufficient support"
    else:
        sd = sup["sd"]
        near = sd == 0.0 or (rr is not None and sd <= frac * rr)
        row["near_constant"] = bool(near)
        if near:
            row["status"] = "near-constant normal"
        else:
            row.update(status="scored", score=ch / sd)
    return row


def gc_absent(ctx: Context, control: str, field: str, region: str, period: str) -> bool:
    return _getter(ctx, control, field, region)(period)[0] is None


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAP AND ELEVATION SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def _pair(a, b, w, limit) -> Dict[str, Any]:
    if a.size < 3:
        return {"cells": int(a.size), "correlation": None}
    tot = float(w.sum())
    out = {"cells": int(a.size), "area_km2": tot, "correlation": wcorr(a, b, w)}
    for k, q in enumerate(quarters(a, b, w)):
        for kk, v in q.items():
            out[f"quarters_by_second_map.{k}.{kk}"] = v
    sh = {"first_positive": float(w[a > 0].sum() / tot), "first_negative": float(w[a < 0].sum() / tot),
          "second_positive": float(w[b > 0].sum() / tot), "second_negative": float(w[b < 0].sum() / tot)}
    out.update({f"sign_shares.{k}": v for k, v in sh.items()})
    if max(sh.values()) > limit:
        out["same_sign_share"] = None
    else:
        same = ((a > 0) & (b > 0)) | ((a < 0) & (b < 0))
        out["same_sign_share"] = float(w[same].sum() / tot)
    return out


def overlap(ctx: Context, a: np.ndarray, b: np.ndarray, keys: Dict[str, Any],
            settings: Dict[str, Any]) -> Dict[str, Any]:
    w = ctx.w
    m = ctx.mask(keys["region"])
    both = m & np.isfinite(a) & np.isfinite(b) & (w > 0)
    out: Dict[str, Any] = {"covered_share": share(w, both, m)}
    for who, x in (("first", a), ("second", b)):
        for k, v in missing(x, m, w).items():
            out[f"missing_cell_values.{who}.{k}"] = v
    out.update(_pair(a[both], b[both], w[both], settings["sign_share_limit"]))
    if keys.get("by_band"):
        names = ctx.band_names()
        if names is None:
            raise CheckError("the control's registry entry names no band edges")
        for name in names:
            k = both & ctx.mask(name)
            for p, v in _pair(a[k], b[k], w[k], settings["sign_share_limit"]).items():
                out[f"by_band.{name}.{p}"] = v
    return out


def _part(x, w, base, pm, region_area, tot_abs, small) -> Dict[str, Any]:
    inside = base & pm
    k = inside & np.isfinite(x)
    ra = float(w[inside].sum())
    rec = {"region_cells": int(inside.sum()), "cells": int(k.sum()),
           "region_area_share": ra / region_area if region_area > 0 else 0.0,
           "empty": not bool(inside.any()),
           "small": bool(inside.any() and ra < small * region_area)}
    for kk, v in missing(x, inside, w).items():
        rec[f"missing_cell_values.{kk}"] = v
    if not k.any():
        rec["area_share"] = 0.0
        return rec
    a = float(w[k].sum())
    area_all = float(w[base & np.isfinite(x)].sum())
    rec.update({"area_km2": a, "area_share": a / area_all,
                "mean": float((x[k] * w[k]).sum() / a),
                "positive_x_area": float((np.clip(x[k], 0, None) * w[k]).sum()),
                "negative_x_area": float((np.clip(x[k], None, 0) * w[k]).sum()),
                "abs_change_share": (float((np.abs(x[k]) * w[k]).sum()) / tot_abs
                                     if tot_abs > 0 else None),
                "concentration": concentration(x[k], w[k])})
    return rec


def elevation_split(ctx: Context, x: np.ndarray, keys: Dict[str, Any],
                    settings: Dict[str, Any], logged: Dict[str, Any]) -> Dict[str, Any]:
    w = ctx.w
    region = keys["region"]
    base = ctx.mask(region) & (w > 0)
    ra = float(w[base].sum())
    m = base & np.isfinite(x)
    area = float(w[m].sum())
    tot_abs = float((np.abs(x[m]) * w[m]).sum())
    out: Dict[str, Any] = {"region_cells": int(base.sum()), "region_area_km2": ra,
                           "covered_share": share(w, base & np.isfinite(x), base),
                           "area_km2": area,
                           "mean": float((x[m] * w[m]).sum() / area) if area > 0 else None,
                           "total_abs_change_x_area": tot_abs,
                           "concentration": concentration(x[m], w[m])}
    for k, v in missing(x, base, w).items():
        out[f"missing_cell_values.{k}"] = v
    names = ctx.band_names()
    if names is None:      # no edges on record: the bands the record names, by name
        names = sorted({p.split(".")[1] for p in logged if p.startswith("parts.")})
    small = settings["small_share"]
    for name in names:
        rec = _part(x, w, base, ctx.mask(name), ra, tot_abs, small)
        out.update({f"parts.{name}.{k}": v for k, v in rec.items()})
        if rec["empty"] or rec["small"]:
            out[f"flagged.{name}"] = "empty" if rec["empty"] else "small"
    for name in ctx.other_regions():
        if name == region:
            continue
        om = ctx.mask(name)
        for nm, pm in ((name, om), (f"not {name}", ~om)):
            rec = _part(x, w, base, pm, ra, tot_abs, small)
            out.update({f"other_regions.{nm}.{k}": v for k, v in rec.items()})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# THE TIMING MEASURES (each written from its contract)
# ─────────────────────────────────────────────────────────────────────────────
def _inputs_missing(x: np.ndarray) -> np.ndarray:
    """Per cell: 0, records_missing in every cell when a day of the period has
    no value in any cell (a missing time record), else cell_values_missing
    where the cell alone lacks a value on a day."""
    fin = np.isfinite(x)
    if x.shape[0] and not fin.any(axis=1).all():
        return np.full(x.shape[1], REASON["records_missing"])
    return np.where(fin.all(axis=0), 0, REASON["cell_values_missing"])


def _first_max(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.where(np.isfinite(x), x, -np.inf)
    return y.max(axis=0), y.argmax(axis=0)


def _windows(x: np.ndarray, n: int) -> np.ndarray:
    """Row j: the mean of days j..j+n-1, each window summed on its own (a
    window with a missing day has none)."""
    if n < 1 or n % 2 == 0:
        raise CheckError(f"a running mean needs an odd window, not {n}")
    if x.shape[0] < n:
        return np.full((0, x.shape[1]), np.nan)
    return np.stack([x[j:j + n].mean(axis=0) for j in range(x.shape[0] - n + 1)])


def _all_run(b: np.ndarray, n: int) -> np.ndarray:
    """Row k: True where days k..k+n-1 are all True."""
    if b.shape[0] < n:
        return np.zeros((0, b.shape[1]), dtype=bool)
    c = np.vstack([np.zeros((1, b.shape[1]), dtype=np.int64), np.cumsum(b, axis=0)])
    return (c[n:] - c[:-n]) == n


def _first_where(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if c.shape[0] == 0:
        z = np.zeros(c.shape[1], dtype=bool)
        return z, np.full(c.shape[1], -1)
    f = c.any(axis=0)
    return f, np.where(f, c.argmax(axis=0), -1)


def _dated(reason: np.ndarray, at: np.ndarray, first_day: int) -> np.ndarray:
    return np.where(reason == 0, at + first_day, np.nan).astype(np.float64)


def m_snow_peak(x, first_day, lead, threshold_mm):
    c = x.shape[1]
    if lead > 0:
        return {"date": np.full(c, np.nan), "reason": np.full(c, REASON["before_record"]),
                "flags": np.zeros(c, dtype=np.int64)}
    big, at = _first_max(x)
    r = _inputs_missing(x)
    r = np.where(r != 0, r, np.where(big < threshold_mm, REASON["no_snow"], 0))
    return {"date": _dated(r, at, first_day), "reason": r, "flags": np.zeros(c, dtype=np.int64)}


def m_snow_gone(x, first_day, lead, threshold_mm, days):
    days = int(days)
    xs = x[lead:]
    n, c = xs.shape
    big, at = _first_max(xs)
    below = np.isfinite(xs) & (xs < threshold_mm)
    above = np.isfinite(xs) & (xs >= threshold_mm)
    rb, ra = _all_run(below, days), _all_run(above, days)
    k = np.arange(rb.shape[0])[:, None]
    found, j = _first_where(rb & (k >= at[None, :]))
    # the trailing days below the threshold: a run that starts after the peak
    # and reaches the end of the days searched before it is long enough
    tail = np.zeros(c, dtype=np.int64)
    alive = np.ones(c, dtype=bool)
    for i in range(n - 1, -1, -1):
        alive &= below[i]
        tail += alive
        if not alive.any():
            break
    short = (tail > 0) & (tail < days) & (n - tail > at)
    r = _inputs_missing(xs)
    r = np.where(r != 0, r, np.where(big < threshold_mm, REASON["no_snow"],
                                     np.where(found, 0, np.where(short, REASON["insufficient_follow_up"],
                                                                 REASON["snow_persists"]))))
    date = _dated(r, j + lead, first_day)
    back = (ra & (np.arange(ra.shape[0])[:, None] > j[None, :])).any(axis=0) if ra.shape[0] \
        else np.zeros(c, dtype=bool)
    flags = np.where(back & (r == 0), FLAG["snow_returns"], 0)
    if lead > 0:
        flags = flags | FLAG["partial_record"]
    return {"date": date, "reason": r, "flags": flags.astype(np.int64)}


def _low_before(x, upto):
    i = np.arange(x.shape[0])[:, None]
    y = np.where((i <= upto[None, :]) & np.isfinite(x), x, np.inf)
    return y.min(axis=0), y.argmin(axis=0)


def m_leaf_green_up(x, first_day, lead, fraction, rise_mark):
    c = x.shape[1]
    peak, at = _first_max(x)
    low, lowd = _low_before(x, at)
    rise = peak - low
    mark = low + fraction * rise
    i = np.arange(x.shape[0])[:, None]
    with np.errstate(invalid="ignore"):
        found, j = _first_where(np.isfinite(x) & (x >= mark[None, :]) & (i > lowd[None, :]))
        r = _inputs_missing(x)
        r = np.where(r != 0, r, np.where(~(peak > 0), REASON["no_leaves"],
                                         np.where(rise < rise_mark * peak,
                                                  REASON["little_leaf_change"], 0)))
    r = np.where((r == 0) & ~found, REASON["little_leaf_change"], r)
    return {"date": _dated(r, j, first_day), "reason": r, "flags": np.zeros(c, dtype=np.int64)}


def m_photosynthesis_onset(x, first_day, lead, fraction, window):
    c = x.shape[1]
    window = int(window)
    m = _windows(x, window)
    peak, at = _first_max(m)
    low, lowj = _low_before(m, at)
    mark = fraction * peak
    i = np.arange(m.shape[0])[:, None]
    with np.errstate(invalid="ignore"):
        found, j = _first_where(np.isfinite(m) & (m >= mark[None, :]) & (i > lowj[None, :]))
        r = _inputs_missing(x)
        r = np.where(r != 0, r, np.where(~(peak > 0), REASON["no_photosynthesis"],
                                         np.where(~(low < mark), REASON["no_dormant_season"], 0)))
    r = np.where((r == 0) & ~found, REASON["no_dormant_season"], r)
    return {"date": _dated(r, j + window // 2, first_day), "reason": r,
            "flags": np.zeros(c, dtype=np.int64)}


def m_growth_peak(x, first_day, lead, window, near, dip):
    window = int(window)
    m = _windows(x, window)
    top, i0 = _first_max(m)
    r = _inputs_missing(x)
    r = np.where(r != 0, r, np.where(~(top > 0), REASON["no_photosynthesis"], 0))
    date = _dated(r, i0 + window // 2, first_day)
    # another peak: a day whose mean is within `near` of the largest, the
    # mean between the two falling below that day's own by more than `dip`
    # of the largest
    k = m.shape[0]
    mm = np.where(np.isfinite(m), m, -np.inf)
    i = np.arange(k)[:, None]
    thr = (1.0 - near) * top
    after = np.where(i >= i0[None, :], mm, np.inf)
    low_after = np.minimum.accumulate(after, axis=0)
    fwd = ((i > i0[None, :]) & (mm >= thr[None, :]) &
           (low_after < mm - dip * top[None, :])).any(axis=0)
    before = np.where(i <= i0[None, :], mm, np.inf)[::-1]
    low_before = np.minimum.accumulate(before, axis=0)[::-1]
    bwd = ((i < i0[None, :]) & (mm >= thr[None, :]) &
           (low_before < mm - dip * top[None, :])).any(axis=0)
    flags = np.where((fwd | bwd) & (r == 0), FLAG["two_peaks"], 0).astype(np.int64)
    return {"date": date, "reason": r, "flags": flags}


def m_half_runoff(x, first_day, lead, fraction):
    c = x.shape[1]
    if lead > 0:
        return {"date": np.full(c, np.nan), "reason": np.full(c, REASON["before_record"]),
                "flags": np.zeros(c, dtype=np.int64)}
    cum = np.cumsum(np.where(np.isfinite(x), x, 0.0), axis=0)
    total = cum[-1]
    found, j = _first_where(cum >= fraction * total[None, :])
    r = _inputs_missing(x)
    r = np.where(r != 0, r, np.where(~(total > 0), REASON["no_runoff"], 0))
    complete = np.isfinite(x).all(axis=0)
    falls = (np.isfinite(x) & (x < 0)).any(axis=0) & complete
    return {"date": _dated(r, j, first_day), "reason": r,
            "flags": np.where(falls, FLAG["runoff_falls"], 0).astype(np.int64)}


MEASURES: Dict[str, Callable] = {"snow_peak": m_snow_peak, "snow_gone": m_snow_gone,
                                 "leaf_green_up": m_leaf_green_up,
                                 "photosynthesis_onset": m_photosynthesis_onset,
                                 "growth_peak": m_growth_peak, "half_runoff": m_half_runoff}


def period_days(r: Run, var: str, kind: str, year: int, ys: int, ws: int
                ) -> Tuple[np.ndarray, int, int]:
    """(days of the period x cells, the date number of its first row, the
    rows before the run's first record): a year from its start month, a water
    year from its start month in the year before (31 December = 0)."""
    held, units, cal = r.calendar_info(var)
    if not held:
        raise CheckError(f"the run holds no daily record of {var}")
    if kind == "year":
        a, b = (year, ys), (year + 1, ys)
    elif ws == 1:
        a, b = (year, 1), (year + 1, 1)
    else:
        a, b = (year - 1, ws), (year, ws)
    rows, lead = [], 0
    for cy in range(a[0], (b[0] if b[1] > 1 else b[0] - 1) + 1):
        lo = NC._day((cy, a[1], 1), units, cal) - NC._day((cy, 1, 1), units, cal) \
            if cy == a[0] else 0.0
        hi = (NC._day((cy, b[1], 1), units, cal) - NC._day((cy, 1, 1), units, cal)
              if (cy == b[0] and b[1] > 1) else
              NC._day((cy + 1, 1, 1), units, cal) - NC._day((cy, 1, 1), units, cal))
        lo, hi = int(round(lo)), int(round(hi))
        d = r.days(var, cy)
        rows.append(d[lo:hi] if d is not None else np.full((hi - lo, r.w.size), np.nan))
        if d is None and cy < held[0]:
            lead += hi - lo
    first = 1
    if kind == "wateryear" and ws != 1:
        first = 1 - int(round(NC._day((year, 1, 1), units, cal) -
                              NC._day((year - 1, ws, 1), units, cal)))
    return np.concatenate(rows), first, lead


def timing_dates(ctx: Context, run: str, measure: str, year: int,
                 variants: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, np.ndarray]]:
    r = ctx.run(run)
    x, first, lead = period_days(r, TIMING_FIELD[measure], TIMING_KIND[measure], int(year),
                                 ctx.ys, ctx.ws)
    return {lab: MEASURES[measure](x, first, lead, **st) for lab, st in variants.items()}


def distribution(v: Dict[str, np.ndarray], m: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
    d, r, f, wm = v["date"][m], v["reason"][m], v["flags"][m], w[m]
    tot = float(wm.sum())
    sh = lambda b: float(wm[b].sum() / tot) if tot > 0 else 0.0      # noqa: E731
    has = np.isfinite(d)
    rec = np.isin(r, (REASON["records_missing"], REASON["before_record"]))
    cell = r == REASON["cell_values_missing"]
    dated = float(wm[has].sum())
    out = {"cells": int(m.sum()), "dates": int(has.sum()),
           "inputs_area_share": sh(~rec & ~cell),
           "missing_time_records.cells": int(rec.sum()), "missing_time_records.area_share": sh(rec),
           "missing_cell_values.cells": int(cell.sum()),
           "missing_cell_values.area_share": sh(cell),
           "date_area_share": sh(has),
           "median": wquantile(d[has], wm[has], 0.5), "q25": wquantile(d[has], wm[has], 0.25),
           "q75": wquantile(d[has], wm[has], 0.75)}
    for name, code in REASON.items():
        if name != "date" and (r == code).any():
            out[f"reasons.{name}"] = int((r == code).sum())
    for name, bit in FLAG.items():
        fk = (f & bit) > 0
        if fk.any():
            a = float(wm[fk].sum())
            out[f"flags.{name}"] = int(fk.sum())
            pre = f"flag_area.{name}."
            out.update({pre + "cells": int(fk.sum()), pre + "area_km2": a,
                        pre + "region_area_km2": tot,
                        pre + "share_of_region_area": a / tot if tot > 0 else 0.0,
                        pre + "cells_with_a_date": int((fk & has).sum()),
                        pre + "dated_area_km2": dated,
                        pre + "share_of_dated_area": (float(wm[fk & has].sum()) / dated
                                                      if dated > 0 else None)})
    return out


def eligible(first, second, m, w, names: Tuple[str, str]) -> Dict[str, Any]:
    a, b = m & np.isfinite(first), m & np.isfinite(second)
    parts = {"both": a & b, f"only_{names[0]}": a & ~b, f"only_{names[1]}": ~a & b,
             "neither": m & ~a & ~b}
    out = {}
    for k, x in parts.items():
        out[f"{k}.cells"] = int(x.sum())
        out[f"{k}.area_share"] = share(w, x, m)
    out["region_area_km2"] = float(w[m].sum())
    return out


def timing(ctx: Context, measure: str, keys: Dict[str, Any], logged: Dict[str, Any]
           ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    year, run = int(keys["year"]), keys.get("run") or "control"
    against = keys.get("against") or "normal"
    m = ctx.mask(keys["region"])
    w = ctx.w
    variants = _logged_variants(logged)
    if not variants:
        raise CheckError("the record names no variant and its settings")
    by = {year: timing_dates(ctx, run, measure, year, variants)}
    ref: Dict[str, Optional[np.ndarray]] = {}
    out: Dict[str, Any] = {}
    if against == "control":
        ctl = timing_dates(ctx, "control", measure, year, variants)
        ref = {lab: ctl[lab]["date"] for lab in variants}
        out["against"] = "the control's same year"
    else:
        ny = ctx.normal_years(keys)
        for y in ny:
            if y not in by:
                try:
                    by[y] = timing_dates(ctx, run, measure, y, variants)
                except CheckError:
                    by[y] = None
        for lab in variants:
            rows = []
            for y in ny:
                v = (by.get(y) or {}).get(lab)
                if v is None:
                    rows = None
                    break
                rows.append(np.where((v["flags"] & FLAG["partial_record"]) > 0, np.nan, v["date"]))
            if rows is None:
                ref[lab] = np.full(w.size, np.nan)
            else:
                a = np.stack(rows)
                ref[lab] = np.where(np.isfinite(a).all(axis=0), a.mean(axis=0), np.nan)
        out["against"] = f"the normal years {list(ny)}"
    tol: Dict[str, Any] = {}
    partial = False
    for lab, v in by[year].items():
        pre = f"variants.{lab}."
        for k, x in distribution(v, m, w).items():
            out[pre + "distribution." + k] = x
        sh = v["date"] - ref[lab]
        ok = m & np.isfinite(sh)
        s = {"cells": int(ok.sum()), "area_share": share(w, ok, m),
             "area_km2": float(w[ok].sum()), "median": wquantile(sh[ok], w[ok], 0.5),
             "q25": wquantile(sh[ok], w[ok], 0.25), "q75": wquantile(sh[ok], w[ok], 0.75)}
        s.update({f"eligible.{k}": x for k, x in
                  eligible(v["date"], ref[lab], m, w, ("this", "reference")).items()})
        if ((v["flags"][ok] & FLAG["partial_record"]) > 0).any():
            s["comparability"] = "restricted comparability"
            partial = True
        out.update({pre + "shift." + k: x for k, x in s.items()})
        for k in ("median", "q25", "q75"):
            tol[pre + "distribution." + k] = tol[pre + "shift." + k] = "day"
    d0 = by[year]["default"]["date"] if "default" in by[year] else None
    for i in range(len(variants)):
        lab = logged.get(f"sensitivity.{i}.variant")
        if lab is None:
            break
        if d0 is None or lab not in by[year]:
            raise CheckError(f"the sensitivity row {i} names a variant the record lacks")
        v = by[year][lab]["date"]
        ch = v - d0
        ok = m & np.isfinite(ch)
        pre = f"sensitivity.{i}."
        out.update({pre + "median_date": out[f"variants.{lab}.distribution.median"],
                    pre + "default_median_date": out["variants.default.distribution.median"],
                    pre + "median_shift": out.get(f"variants.{lab}.shift.median"),
                    pre + "default_median_shift": out.get("variants.default.shift.median"),
                    pre + "median_cell_change": wquantile(ch[ok], w[ok], 0.5),
                    pre + "cells_with_both": int(ok.sum()),
                    pre + "cell_change_area_share": share(w, ok, m),
                    pre + "cell_change_area_km2": float(w[ok].sum())})
        out.update({pre + "eligible." + k: x for k, x in
                    eligible(d0, v, m, w, ("default", "alternative")).items()})
        for k in ("median_date", "default_median_date", "median_shift", "default_median_shift",
                  "median_cell_change"):
            tol[pre + k] = "day"
    if any(p.startswith(ACROSS) for p in logged) or \
            (_num(logged.get("version")) and logged["version"] >= RANGE_FROM_VERSION):
        a, at = across(by[year], ref, m, w, list(variants))
        out.update(a)
        tol.update(at)
    if partial:
        out["comparability"] = "restricted comparability: a partial-record date is in a shift"
    return out, tol


# THE RANGE ACROSS SETTINGS (the user's decisions on G part 1, 2026-11-08),
# made here on its own: the cells with a shift under every setting the record
# names (a date in the year and in its baseline at each), each setting's
# median shift on them, the smallest and the largest (a tie keeps the one
# named first: the default, then the record's order), their spread against
# 7 days, and whether one setting gives an earlier date and another a later
# one. A timing record made by version 3 or later holds it.
ACROSS = "across_settings."
RANGE_FROM_VERSION = 3
RANGE_DAYS = 7.0
NOISE_DAYS = 1e-9


def across(year: Dict[str, Dict[str, np.ndarray]], ref: Dict[str, Optional[np.ndarray]],
           m: np.ndarray, w: np.ndarray, named: Sequence[str]
           ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    order = ["default"] if "default" in named else []
    order += [lab for lab in named if lab != "default"]
    order = [lab for lab in order if lab in year and ref.get(lab) is not None]
    shifts = {lab: year[lab]["date"] - ref[lab] for lab in order}
    both = m.copy()
    for lab in order:
        both = both & np.isfinite(shifts[lab])
    out: Dict[str, Any] = {ACROSS + "comparable.cells": int(both.sum()),
                           ACROSS + "comparable.area_km2": float(w[both].sum()),
                           ACROSS + "comparable.area_share": share(w, both, m),
                           ACROSS + "limit_days": RANGE_DAYS}
    tol: Dict[str, Any] = {}
    med = {}
    for i, lab in enumerate(order):
        med[lab] = wquantile(shifts[lab][both], w[both], 0.5)
        out[f"{ACROSS}evaluated.{i}.variant"] = lab
        out[f"{ACROSS}evaluated.{i}.median_shift"] = med[lab]
        tol[f"{ACROSS}evaluated.{i}.median_shift"] = "day"
    got = [lab for lab in order if med[lab] is not None]
    if not got:
        out.update({ACROSS + "spread_days": None, ACROSS + "range_required": False,
                    ACROSS + "direction_reversal": False})
        return out, tol
    small = large = got[0]
    for lab in got:
        if med[lab] < med[small]:
            small = lab
        if med[lab] > med[large]:
            large = lab
    parts = {"minimum": small, "maximum": large}
    if got[0] == "default":
        parts["default"] = "default"
    for part, lab in parts.items():
        out[f"{ACROSS}{part}.variant"] = lab
        out[f"{ACROSS}{part}.median_shift"] = med[lab]
        tol[f"{ACROSS}{part}.median_shift"] = "day"
    gap = med[large] - med[small]
    out[ACROSS + "spread_days"] = gap
    tol[ACROSS + "spread_days"] = "day"
    out[ACROSS + "range_required"] = bool(gap > RANGE_DAYS + NOISE_DAYS)
    out[ACROSS + "direction_reversal"] = bool(med[small] < -NOISE_DAYS and
                                              med[large] > NOISE_DAYS)
    return out, tol


def _logged_variants(logged: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{variant: settings} as the record names them (the inputs of each
    variant, in the record's order)."""
    out: Dict[str, Dict[str, Any]] = {}
    for p, v in logged.items():
        m = re.fullmatch(r"variants\.(.+?)\.settings\.([A-Za-z_]+)", p)
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = v
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAPS NAMED BY THEIR HASHES
# ─────────────────────────────────────────────────────────────────────────────
def map_source(ctx: Context, sha: str, before: int) -> Dict[str, Any]:
    """The logged lookup (before line `before`) that returned the array of
    this hash: get_map whole, or a map of a per-cell anomaly or difference."""
    for i in range(min(before, len(ctx.lines)) - 1, -1, -1):
        ln = ctx.lines[i]
        if "sha256" not in ln or ln.get("refused") or ln.get("failed"):
            continue
        if ln.get("call") == "get_map" and ln.get("sha256") == sha:
            return {"line": i + 1, "call": "get_map", "keys": ln.get("keys") or {}, "path": None}
        for p, h in (ln.get("arrays") or {}).items():
            if h == sha and ln.get("call") in MAP_PATHS and p in MAP_PATHS[ln["call"]]:
                return {"line": i + 1, "call": ln["call"], "keys": ln.get("keys") or {},
                        "settings": ln.get("settings"), "path": p}
    raise CheckError(f"its map {sha[:12]} is no array a logged lookup returned (a map a script "
                     f"made by its own arithmetic cannot be made again); a map from "
                     f"suite.get_map or from a per-cell anomaly or difference can")


def remake_map(ctx: Context, src: Dict[str, Any]) -> np.ndarray:
    k = src["keys"]
    if src["call"] == "get_map":
        run, field, period = k.get("run") or "control", k["field"], str(k["period"])
        r = ctx.run(run)
        if period == "normal" or period.startswith("normal-"):
            part = period[len("normal") + 1:]
            kind, pt = ("year", None) if not part else ("wateryear", None) if part == "WY" \
                else ("season", part)
            ny = ctx.plan_normal_years()
            if not ny:
                raise CheckError("a normal map's years are named by no analysis plan")
            used = {}
            for y in ny:
                try:
                    used[y] = r.period_map(field, label_of(kind, y, pt))
                except Absent:
                    pass
            if not used:
                raise CheckError(f"no normal year holds {field} for {period}")
            return _mean(used)
        return r.period_map(field, period)
    st = dict(DEFAULTS[src["call"]], **(src.get("settings") or {}))
    fn = anomaly if src["call"] == "anomaly" else difference
    _, _, maps = fn(ctx, dict(k, region=None), st)
    if src["path"] not in maps:
        raise CheckError(f"the {src['call']} of line {src['line']} has no map {src['path']}")
    return np.asarray(maps[src["path"]], dtype=np.float64).ravel()


# ─────────────────────────────────────────────────────────────────────────────
# ONE RECORD
# ─────────────────────────────────────────────────────────────────────────────
def _row_places(logged: Dict[str, Any]) -> Dict[Tuple[str, str], List[str]]:
    """{(field, period): [path prefix]} of every row the screen record holds,
    in its table and in its display."""
    out: Dict[Tuple[str, str], List[str]] = {}
    for p, v in logged.items():
        if p.endswith(".field"):
            pre = p[:-len("field")]
            per = logged.get(pre + "period")
            if per is not None and (pre.startswith("rows.") or pre.startswith("display.")):
                out.setdefault((str(v), str(per)), []).append(pre)
    return out


def recompute(ctx: Context, i: int, cited_paths: Sequence[str]
              ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """(recomputed values by path, tolerance by path, notes) of line i."""
    ln = ctx.lines[i - 1]
    call, keys = ln.get("call"), dict(ln.get("keys") or {})
    logged = ln.get("result") or {}
    named = dict(ln.get("settings") or {})
    if call in ("anomaly", "difference"):
        st = dict(DEFAULTS[call], **named)
        out, tol, _ = (anomaly if call == "anomaly" else difference)(ctx, keys, st)
        return out, tol, []
    if call in TIMING_KIND:
        out, tol = timing(ctx, call, keys, logged)
        return out, tol, []
    if call == "screen":
        st = dict(DEFAULTS["screen"], **named)
        places = _row_places(logged)
        # a row is cited when a claim states one of its values (a count of
        # normal years alone, a label, does not cite a row)
        want = sorted({k for k, pres in places.items() for p in cited_paths for pre in pres
                       if p.startswith(pre) and p[len(pre):] in ROW_VALUES})
        run = keys.get("run") or "control"
        ctl = None if run == "control" else "control"
        ny = ctx.normal_years(keys)
        out, tol = {}, {}
        last = None
        for field, period in want:
            if last is not None and field != last:       # one field's files held at a time
                for r in ctx._runs.values():
                    r.forget()
            last = field
            row = screen_row(ctx, run, ctl, field, period, keys["region"], ny,
                             st["near_constant_share"])
            sc = max(abs(row["value"] or 0.0), abs(row["reference"] or 0.0))
            for pre in places[(field, period)]:
                for col in SCREEN_COLS:
                    if pre.startswith("display.") and col == "near_constant":
                        continue                      # the display shows no such column
                    out[pre + col] = row[col]
                tol[pre + "change"] = sc
                tol[pre + "percent"] = 100.0 * sc / abs(row["reference"]) \
                    if row["reference"] else 0.0
                tol[pre + "score"] = sc / row["_sd"] if row.get("_sd") else 0.0
        notes = [] if want else ["no claim states a value of this screen, so no row was "
                                 "made again"]
        return out, tol, notes
    if call in ("overlap", "elevation_split"):
        st = dict(DEFAULTS[call], **named)
        names = ("a", "b") if call == "overlap" else ("x",)
        arrays = []
        notes = []
        for nm in names:
            src = map_source(ctx, str(keys.get(nm)), i)
            arrays.append(remake_map(ctx, src))
            notes.append(f"map {nm}: line L{src['line']} ({src['call']}"
                         + (f", its {src['path']}" if src["path"] else "") + ")")
        if call == "overlap":
            return overlap(ctx, arrays[0], arrays[1], keys, st), {}, notes
        return elevation_split(ctx, arrays[0], keys, st, logged), {}, notes
    raise CheckError(f"the check makes no {call} again")


# ─────────────────────────────────────────────────────────────────────────────
# THE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _norm_logged(logged: Dict[str, Any]) -> Dict[str, Any]:
    """The logged result with its left-out normal years as a list (they are
    logged as {year: why})."""
    out = {k: v for k, v in logged.items() if not k.startswith("normal_left_out.")}
    lo = sorted(int(k.split(".", 1)[1]) for k in logged if k.startswith("normal_left_out."))
    if lo or "normal_left_out" in logged or any(k.startswith("normal_") for k in logged):
        out["normal_left_out"] = lo
    return out


def compare(rid: str, call: str, where: str, logged: Dict[str, Any], mine: Dict[str, Any],
            tol: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]], List[str],
                                          Optional[Dict[str, Any]]]:
    """(numbers compared, mismatch rows, numeric paths the record holds that
    were not made again, the comparison with the largest relative difference)."""
    L = _norm_logged(logged)
    n, bad, worst = 0, [], None
    for p, m in mine.items():
        has = p in L
        lv = L.get(p)
        if not has and m is None:
            continue
        if isinstance(m, np.generic):
            m = m.item()
        n += 1
        row = {"record": rid, "call": call, "where": where, "path": p, "logged": lv,
               "recomputed": m}
        if not has:
            bad.append(dict(row, why="the record has no value where the check made one"))
            continue
        if m is None or lv is None:
            if not (m is None and lv is None):
                bad.append(dict(row, why="one side has no value"))
            continue
        if isinstance(m, bool) or isinstance(lv, bool) or isinstance(m, str) or \
                isinstance(lv, str) or isinstance(m, list) or isinstance(lv, list):
            same = (m == lv) if not isinstance(m, list) else \
                [_plain(x) for x in m] == [_plain(x) for x in (lv or [])]
            if not same:
                bad.append(dict(row, why="the record and the check differ"))
            continue
        if not (_num(m) and _num(lv)):
            bad.append(dict(row, why="the record's entry is not a number"))
            continue
        t, r = float(lv), float(m)
        d = abs(t - r)
        kind = tol.get(p)
        if isinstance(m, int):
            allowed = 0.0                                  # a count is equal or it is not
        elif kind == "day":
            allowed = DAY_TOL
        else:
            s = abs(r) + (NC.EPS / NC.RTOL) * abs(float(kind) if _num(kind) else 0.0)
            allowed = NC.RTOL * s + NC.ATOL
        rel = d / abs(r) if r != 0 else (0.0 if d == 0 else math.inf)
        row.update(abs_diff=d, rel_diff=rel)
        if worst is None or rel > worst["rel_diff"]:
            worst = dict(row)
        if not (d <= allowed) or not math.isfinite(t):
            bad.append(dict(row, allowed=allowed,
                            why="outside the tolerance" if allowed else "not equal"))
    left = sorted(p for p, v in L.items() if _num(v) and p not in mine and not _label(p))
    return n, bad, left, worst


def _plain(x):
    return float(x) if _num(x) else x


_LABELS = re.compile(r"(^|\.)(settings|version|text_revision|year|years|first_day|status|"
                     r"normal_years|event_years|normal_years_used|n_normal)(\.|$)")


def _label(p: str) -> bool:
    """A path that names what was computed, not a value of it (a screen row's
    count of normal years, a label, cites no row by itself)."""
    return bool(_LABELS.search(p))


def _where(ln: Dict[str, Any]) -> str:
    k = ln.get("keys") or {}
    bits = [ln.get("call")] + [str(k[x]) for x in ("run", "field", "period", "year", "region")
                               if k.get(x) is not None]
    return ", ".join(str(b) for b in bits if b)


def _cited(claims: Sequence[Dict[str, Any]], lines: Sequence[Dict[str, Any]]
           ) -> Dict[str, List[str]]:
    """{record id: [the paths of the values the claims state]}: every record
    a claim names, with the claim trace's matched values (a claim the trace
    has not seen: the paths of its records holding the values it gives)."""
    out: Dict[str, List[str]] = {}
    for c in claims or []:
        ids = [str(r) for r in c.get("evidence") or []]
        for rid in ids:
            out.setdefault(rid, [])
        if "evidence_trace" in c:
            for m in (c["evidence_trace"] or {}).get("matched") or []:
                rid, _, path = str(m).partition(":")
                out.setdefault(rid, []).append(path)
            continue
        for rid in ids:
            n = int(rid[1:]) if rid[1:].isdigit() else 0
            res = (lines[n - 1].get("result") or {}) if 1 <= n <= len(lines) else {}
            for v in c.get("values") or []:
                if _num(v):
                    out[rid] += [p for p, x in res.items() if _num(x) and
                                 abs(float(x) - float(v)) <= 1e-9 * max(1.0, abs(float(x)))]
    return out


def _key(ctx: Context, i: int, cited_paths: Sequence[str]) -> str:
    ln = ctx.lines[i - 1]
    runs = {"control"} | {str(v) for k, v in (ln.get("keys") or {}).items()
                          if k in ("run", "control") and v}
    srcs = []
    for nm in ("a", "b", "x"):
        h = (ln.get("keys") or {}).get(nm)
        if ln.get("call") in ("overlap", "elevation_split") and h:
            try:
                s = map_source(ctx, str(h), i)
                srcs.append([s["call"], s["keys"], s.get("settings"), s["path"]])
                runs |= {str(s["keys"].get(k)) for k in ("run", "control") if s["keys"].get(k)}
            except CheckError:
                srcs.append(None)
    blob = [ln.get("call"), ln.get("keys"), ln.get("settings"),
            sorted(cited_paths) if ln.get("call") == "screen" else None, srcs,
            ctx.plan.get("baseline"), ctx.inp.get("regions"), ctx.entry.get("elevation_band_edges"),
            ctx.stamps(sorted(runs)), sorted(_logged_variants(ln.get("result") or {}).items()),
            [ln.get("result", {}).get(f"sensitivity.{k}.variant") for k in range(8)]]
    return json.dumps(blob, sort_keys=True, default=str)


def check(study_dir, claims: Sequence[Dict[str, Any]], registry: Optional[Dict[str, Any]] = None,
          write: bool = True) -> Dict[str, Any]:
    """Every number of each record the claims cite, made again from the
    model output and compared. Writes 04_analysis/evidence_numbers_check.json."""
    rd = Path(study_dir)
    rec: Dict[str, Any] = {
        "kind": "suite_numbers_check", "checked_at": datetime.now().isoformat(timespec="seconds"),
        "log": f"04_analysis/{LOOKUPS}", "pass": False, "n_compared": 0, "n_mismatched": 0,
        "mismatches": [], "records": {}, "not_recomputed": [], "largest_relative_difference": None,
        "tolerance": {"relative": NC.RTOL, "operands": NC.EPS, "absolute": NC.ATOL,
                      "days": DAY_TOL,
                      "rule": "|logged - recomputed| <= relative x |recomputed| + operands x "
                              "scale + absolute; counts, statuses, marks, reason codes and flags "
                              "equal; dates and shifts in days within 1e-9 of a day"},
        "method": METHOD}
    try:
        ctx = Context(rd, registry)
        cited = _cited(claims, ctx.lines)
        rec["cited"] = sorted(cited, key=lambda r: int(r[1:]) if r[1:].isdigit() else 0)
        worst = None
        for rid in rec["cited"]:
            if not (rid.startswith("L") and rid[1:].isdigit()) or \
                    not 1 <= int(rid[1:]) <= len(ctx.lines):
                continue                       # the claim trace fails a record that is not there
            i = int(rid[1:])
            ln = ctx.lines[i - 1]
            call = ln.get("call")
            if "sha256" not in ln or ln.get("refused") or ln.get("failed") or ln.get("support"):
                continue
            if call in NO_NUMBER_CALLS:
                rec["records"][rid] = {"call": call, "compared": 0,
                                       "note": "it logs no number (a map or a series is an "
                                               "array the log does not keep); its arrays are "
                                               "made again where an overlap or split uses them"}
                continue
            if call not in METHOD_CALLS:
                rec["not_recomputed"].append({"record": rid, "call": call,
                                              "why": "the check makes no such call again"})
                continue
            where = _where(ln)
            t0 = datetime.now()
            try:
                k = _key(ctx, i, cited[rid])
                if k not in _CACHE:
                    _CACHE[k] = recompute(ctx, i, cited[rid])
                    for r in ctx._runs.values():
                        r.forget()
                    ctx.memo = {mk: mv for mk, mv in ctx.memo.items() if mk[3] is not None}
                mine, tol, notes = _CACHE[k]
            except (CheckError, Absent) as e:
                rec["mismatches"].append({"record": rid, "call": call, "where": where,
                                          "path": None, "logged": "its numbers",
                                          "recomputed": None,
                                          "why": f"could not be made again: {e}"})
                rec["records"][rid] = {"call": call, "compared": 0, "error": str(e)}
                continue
            n, bad, left, top = compare(rid, call, where, ln.get("result") or {}, mine, tol)
            if top is not None and (worst is None or top["rel_diff"] > worst["rel_diff"]):
                worst = top
            rec["n_compared"] += n
            rec["mismatches"] += bad
            rec["records"][rid] = {"call": call, "where": where, "compared": n,
                                   "mismatched": len(bad), "notes": notes,
                                   "seconds": round((datetime.now() - t0).total_seconds(), 2)}
            if left:
                rec["not_recomputed"].append({"record": rid, "call": call, "n": len(left),
                                              "paths": left[:20],
                                              "why": "numbers of the record the check does not "
                                                     "make again (a value the claims cite is "
                                                     "always made again)"})
                stated = [p for p in cited[rid] if p in left]
                for p in stated:
                    rec["mismatches"].append({
                        "record": rid, "call": call, "where": where, "path": p,
                        "logged": (ln.get("result") or {}).get(p), "recomputed": None,
                        "why": "a claim states this number and the check cannot make it again"})
        rec["n_mismatched"] = len(rec["mismatches"])
        rec["largest_relative_difference"] = worst
        rec["sources"] = {"runs": dict(ctx._dirs), "control": ctx.inp.get("control"),
                          "reads": sorted({f"{run}: {v} from {f}" for run, r in ctx._runs.items()
                                           for v, f in r.reads})}
        rec["pass"] = not rec["mismatches"]
    except Exception as e:                                          # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["pass"] = False
    if write:
        out = rd / "04_analysis"
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / EVIDENCE_JSON).write_text(json.dumps(rec, indent=1, default=str))
        except OSError as e:
            rec.setdefault("notes", []).append(f"the record could not be written: {e}")
    return rec


METHOD = (
    "Each number of every lookup a claim cites (04_analysis/lookups.jsonl), made again from "
    "each run's own history by code written apart from domain_methods, domain_suite, "
    "domain_access, gridded and regions: the history read with netCDF4 through the numbers "
    "check's reader (each record weighted by the days of its own time_bounds inside the "
    "period, the records required to tile it), each run's output directory from the manager's "
    "records, the regions rebuilt from the surface file and the registry, the field notes' "
    "type and percent policy, the plan's normal years; region values by the regional "
    "table's rules (a rate's total, a state's time-weighted mean, the smallest covered share "
    "over the records), maps rounded to float32 as the suite stores them, and every method "
    "(anomaly, difference, the screen's cited rows, overlap, elevation_split and the six "
    "timing measures from the daily history, with their range across settings on the "
    "comparable cells and its reversal flag) by its contract.")
