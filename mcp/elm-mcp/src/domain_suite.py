#!/usr/bin/env python3
"""One run's results store (the suite): maps and region series, made once.

THE SUITE (2026-11-04). Every number the analysis reports was made by reading
the run's history again, a gigabyte a year per stream, for each question. The
suite reads each year file of each stream once and keeps what the analyses are
made from, as NetCDF on the grid's own lat and lon axes, beside the regions
(masks.nc, regions.MaskSet.to_netcdf) and a manifest (suite.json):

  maps_year.nc        (year, lat, lon) per field: a rate's total over the
                      calendar year, a state's time-weighted mean   float32
  maps_wateryear.nc   (wateryear, lat, lon): October to September, named by
                      the year it ends; a water year whose October-December
                      the run lacks is absent, with the reason        float32
  maps_season.nc      (year, season, lat, lon): DJF (with the December
                      before), MAM, JJA, SON; the first year's DJF is absent,
                      its maps fill, with the reason                  float32
  series_day.nc       (region, day): area-weighted region means per day from
                      the daily stream; a rate's day total (the rate times
                      86,400 s, in per-day units), a state's day mean float64
  series_month.nc, series_season.nc, series_year.nc, series_wateryear.nc
                      (region, period)                                float64
  timing.nc           (lat, lon, year) per timing measure of domain_methods
                      and per setting its contract names: the date, the
                      reason code where there is none, the quality flags;
                      water-year measures under the year they end
                                                         float32, int8

Each series field X has X_covered beside it: the share of the region's area
(cell area x land fraction) that had a value, the smallest over the records a
period is made of. Month maps are not built: the monthly history holds them
and is referenced, not copied. Normals are not stored: a normal is the mean of
the plan's normal years, made from these at lookup time.

WHICH STREAM. Month, season and year values, as maps and as series, come from
the monthly stream for every field it holds, by the regional table's own code:
the month, season and year series ARE domain_analysis.run_values' numbers (the
same stream choice, STREAM_ORDER and _stream_for, the same rule, the same
masks), so they equal the table by construction. A field only the daily
stream holds (COL_FIRE_CLOSS) takes them from the daily stream, as the table
does. A water year is made from its months as a season is: a rate's total is
their sum, a state's mean is weighted by their days. The maps follow the same
rule cell by cell, from the same records.

ONE READ PER YEAR FILE. run_values reads each year of each stream it needs in
one pass (gridded.Stream.read_block_pass). The suite hands it the streams
wrapped (_Tee): each of those passes also reads the fields only the suite
needs, over the whole grid, and every block goes to the suite's running sums
before the table's own callback gets its window of it, values unchanged. A
year no table field needs from a stream (the daily fields' days, when no field
is daily-only) is read by the suite itself, once. A cell without a value
leaves the numerator and the denominator alike; a map cell is filled when any
record of its period has no value there, since a total over part of a period
is not that period's total.

H2OSOI (15 levels): its products are the top-1 m mean, each layer weighted by
its thickness inside the top metre, from the layer thicknesses the files
carry (DZSOI), a layer crossing 1 m by its part above it. Its series are made
by the table's rule from its own region means (regions.RegionSeries.by_year,
domain_analysis.months_and_seasons), since the table makes no number from a
field with levels.

VERSIONING. The manifest records what the products depend on: SUITE_VERSION
(raised whenever this code makes different products), the run directory, each
history file read (name, size, mtime_ns), the regions (their content hash, and
names, cells and area), the band edges and the fields asked for. A build is
reused when all of it matches, and nothing is written. When the only change is
new history files that complete the years after the last one, those years are
added: their year, their seasons (DJF holds the December before) and their
water year, the products that cross into them, and nothing else is remade.
Anything else rebuilds every product. The code's git commit and the build time
are recorded as provenance, not compared: a commit that leaves the products
alone does not rebuild them. The manifest is deleted before any product is
touched and written last, so a build cut off leaves no manifest and the next
build starts again.

  build(control, out_dir, run_dir=None, fields=None)   the suite of one run,
                       reused, extended by new years, or built, as above
  load(out_dir)        its manifest, or None
  top_metre_weights(dz)  the layers' weights of the top-1 m mean

Measured on compy01 (login node), 2026-11-04, the Naches control
(naches_bgc_fix, 2015-2023; 52 fields, 15 regions; the h0 and h1 year files
read once each, nine passes per stream): the first build took 307 s (max
resident memory 819 MB) and wrote 198 MB (197,654,755 bytes: maps_season
104 MB, maps_year 26 MB, maps_wateryear 23 MB, series_day 38 MB, the other
series 1-2 MB each, masks 0.3 MB; suite.json 89 KB); a second build reads no
year file and writes nothing. A region's mean of a year map, made from the
float32 cells, was within 1.6e-7 relative of the year series.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import domain_analysis as DA
import gridded as G
import regions as R

SUITE_VERSION = 1
MANIFEST = "suite.json"
MASKS_FILE = "masks.nc"
TIMING_FILE = "timing.nc"
MAP_KINDS = ("year", "wateryear", "season")
SERIES_KINDS = ("day", "month", "season", "year", "wateryear")
PRODUCTS = tuple(f"maps_{k}" for k in MAP_KINDS) + tuple(f"series_{k}" for k in SERIES_KINDS)
SEASON_NAMES = tuple(n for n, _ in DA.SEASONS)
WY_MONTHS = ((-1, 10), (-1, 11), (-1, 12)) + tuple((0, m) for m in range(1, 10))
# The suite's one rule for a field with levels, and the fields it is defined
# for: a thickness-weighted mean is right for a volumetric content, wrong for
# an amount per layer (SOILLIQ, kg/m2 in each layer, sums instead).
TOP_METRE_FIELDS = ("H2OSOI",)
TOP_METRE_M = 1.0
THICKNESS_VAR = "DZSOI"
COVERED = "_covered"


class SuiteError(Exception):
    """A suite cannot be built, and the message says why."""


# ─────────────────────────────────────────────────────────────────────────────
# SMALL RULES
# ─────────────────────────────────────────────────────────────────────────────
def top_metre_weights(dz, depth: float = TOP_METRE_M) -> np.ndarray:
    """(levels, cells) weights of the top-`depth` mean: each layer's thickness
    inside the top `depth` metres (a layer crossing it by its part above it),
    over their sum. NaN for a cell whose thicknesses are not all given."""
    d = np.asarray(dz, dtype=np.float64)
    d = d.reshape(d.shape[0], -1)
    top = np.vstack([np.zeros((1, d.shape[1])), np.cumsum(d, axis=0)[:-1]])
    with np.errstate(invalid="ignore"):
        inside = np.clip(np.minimum(top + d, depth) - top, 0.0, None)
    tot = inside.sum(axis=0)
    bad = ~np.all(np.isfinite(d), axis=0) | ~(tot > 0)
    w = inside / np.where(tot > 0, tot, 1.0)
    w[:, bad] = np.nan
    return w


def _top_metre(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(records, levels, *grid) to (records, cells): the weighted mean of the
    layers inside the top metre; NaN where one of those layers has no value."""
    n, lev = x.shape[0], x.shape[1]
    a = x.reshape(n, lev, -1)
    with np.errstate(invalid="ignore"):
        use = w > 0
    y = np.einsum("nlc,lc->nc", np.where(use, a, 0.0), np.where(use, w, 0.0))
    bad = np.any(use[None] & ~np.isfinite(a), axis=1) | ~np.all(np.isfinite(w), axis=0)[None]
    y[bad] = np.nan
    return y


def _record_days(stream, keys) -> np.ndarray:
    """Days each record counts for, as regions.RegionSeries.record_days: a
    month's nominal length (the run's first step folded into January is not
    counted, as the table does not count it)."""
    if stream.cadence == "daily":
        return np.ones(len(keys))
    if stream.cadence == "subdaily":
        return np.full(len(keys), 1.0 / float(stream.steps_per_day))
    return np.array([float(G.days_in_month(k[0], k[1], stream.calendar)) for k in keys])


def _expected_per_month(stream, year: int) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for k in G.expected_keys(year, stream.cadence, stream.calendar, stream.steps_per_day):
        out[k[1]] = out.get(k[1], 0) + 1
    return out


def _months_held(stream, year: int, months: Sequence[int]) -> bool:
    """Whether the stream holds every record of these months of `year`."""
    if year not in stream.years():
        return False
    got: Dict[int, int] = {}
    for r in stream.year_records(year)[0]:
        got[r.key[1]] = got.get(r.key[1], 0) + 1
    want = _expected_per_month(stream, year)
    return all(got.get(m, 0) == want[m] for m in months)


def _combine(parts: List[Optional[np.ndarray]], keys: Sequence[Tuple[int, int]], rate: bool,
             calendar: str):
    """A period from its months, as months_and_seasons makes a season: a
    rate's total is their sum, a state's mean is weighted by their days. NaN
    where a month has no value. Works on numbers and on arrays alike."""
    if any(p is None for p in parts):
        return None
    if rate:
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out
    d = [float(G.days_in_month(y, m, calendar)) for y, m in keys]
    num = parts[0] * d[0]
    for p, w in zip(parts[1:], d[1:]):
        num = num + p * w
    return num / sum(d)


def _wy_keys(y: int) -> List[Tuple[int, int]]:
    return [(y + dy, m) for dy, m in WY_MONTHS]


def wy_label(y: int) -> str:
    """A water year as the suite names it: WY2016, October 2015 to September 2016."""
    return f"WY{int(y)}"


def masks_sha256(ms: R.MaskSet) -> str:
    """The regions' content: names, grid shape and every cell in or out."""
    h = hashlib.sha256(json.dumps(list(ms.grid.shape)).encode())
    for n in ms.names():
        h.update(n.encode() + b"\0")
        h.update(np.packbits(np.ascontiguousarray(ms[n], dtype=bool).ravel()).tobytes())
    return h.hexdigest()


def _git() -> Dict[str, Any]:
    here = str(Path(__file__).resolve().parent)
    out: Dict[str, Any] = {"git_commit": None, "uncommitted_changes": None}
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, universal_newlines=True, timeout=30)
        if r.returncode == 0:
            out["git_commit"] = r.stdout.strip()
        r = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           universal_newlines=True, timeout=60)
        if r.returncode == 0:
            out["uncommitted_changes"] = bool(r.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def load(out_dir) -> Optional[Dict[str, Any]]:
    """The suite's manifest, or None when none is there (or it cannot be read)."""
    p = Path(str(out_dir)) / MANIFEST
    if not p.is_file():
        return None
    try:
        m = json.loads(p.read_text())
    except ValueError:
        return None
    return m if isinstance(m, dict) else None


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# WHAT A BUILD READS
# ─────────────────────────────────────────────────────────────────────────────
def note_rows(notes_path=None) -> List[str]:
    """The catalog's rows (domain_field_notes.json, row true): the fields a
    suite holds by default."""
    import domain_catalog as C
    notes = C.load_notes(Path(notes_path) if notes_path else C.NOTES_PATH)
    return sorted(n for n, e in notes.items() if C.is_row(e))


def _thickness(streams: Dict[str, Any], first: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """DZSOI from the first file of `first`, else of another stream (the same
    cells): the daily files of the Naches control do not carry it."""
    import netCDF4
    for s in [first] + [x for x in streams if x != first]:
        p = streams[s].files[0]
        with netCDF4.Dataset(p) as d:
            if THICKNESS_VAR in d.variables:
                return G.read_clean(d[THICKNESS_VAR]), p.name
    return None, None


class _Context:
    """Everything a build reads before any product is made, and what the
    manifest records of it."""

    def __init__(self, control: Dict[str, Any], run_dir: Path, fields: Sequence[str]):
        self.control = control
        self.run_dir = run_dir
        have = G.discover(run_dir)
        self.fields_asked = sorted(dict.fromkeys(str(f) for f in fields))
        # THE STREAMS OPENED (2026-11-04): in STREAM_ORDER, until every field
        # asked for is in one of them and a daily stream is open, so each
        # field's stream is the first that holds it, as the table chooses, and
        # the 3-hourly stream of the Naches control (150 s to index cold) is
        # not opened for nothing.
        self.streams: Dict[str, Any] = {}
        for s in DA.STREAM_ORDER:
            if s not in have:
                continue
            if self.streams and any(x.cadence == "daily" for x in self.streams.values()) \
                    and all(any(v in x.variables for x in self.streams.values())
                            for v in self.fields_asked):
                break
            self.streams[s] = G.open_stream(run_dir, s)
        if not self.streams:
            raise SuiteError(f"{run_dir} holds none of the streams {DA.STREAM_ORDER}")
        self.plan: Dict[str, Dict[str, Any]] = {}
        self.left_out: Dict[str, str] = {}
        self.problems: List[str] = []
        self.thickness: Dict[str, np.ndarray] = {}
        self.layers_note: Dict[str, Any] = {}
        for v in self.fields_asked:
            self._plan_field(v)
        self.period_streams = sorted({p["stream"] for p in self.plan.values()},
                                     key=DA.STREAM_ORDER.index)
        self.used = sorted(set(self.period_streams) |
                           {p["day_stream"] for p in self.plan.values() if p["day_stream"]},
                           key=DA.STREAM_ORDER.index)
        if not self.used:
            raise SuiteError("none of the fields asked for can be made: "
                             + "; ".join(f"{k}: {w}" for k, w in self.left_out.items()))
        whole = [set(self.streams[s].complete_years()) for s in self.used]
        self.years = sorted(set.intersection(*whole))
        for s in self.used:
            for y in sorted(set(self.streams[s].complete_years()) - set(self.years)):
                self.problems.append(f"{y} is whole in {s} but not in every stream the suite "
                                     f"reads ({', '.join(self.used)}): left out")
        if not self.years:
            raise SuiteError(f"no year is whole in every stream the suite reads "
                             f"({', '.join(self.used)})")
        self.grid = self.streams[self.used[0]].grid
        self.calendar = self.streams[self.used[0]].calendar
        self._regions()
        self.history = {s: [self._facts(p) for p in self.streams[s].files] for s in self.used}

    @staticmethod
    def _facts(p: Path) -> Dict[str, Any]:
        st = os.stat(p)
        return {"name": p.name, "bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}

    def _plan_field(self, v: str) -> None:
        st = DA._stream_for(self.streams, v)
        if st is None:
            self.left_out[v] = f"none of the run's streams ({', '.join(self.streams)}) holds it"
            return
        s = self.streams[st]
        cm = s._vars[v].get("cell_methods") or ""
        if s.cadence != "monthly" and "time: point" in cm:
            self.left_out[v] = (f"its only stream {st} holds it as an instantaneous value, which "
                                f"the reader does not label by day")
            return
        top = False
        if s.levels(v):
            if v not in TOP_METRE_FIELDS:
                self.left_out[v] = (f"it has a value at each of its levels "
                                    f"({', '.join(s.levels(v))}), and the suite's one rule for "
                                    f"levels, the top-1 m mean, is defined for "
                                    f"{', '.join(TOP_METRE_FIELDS)} only")
                return
            top = True
        day = next((x for x in DA.STREAM_ORDER if x in self.streams
                    and self.streams[x].cadence == "daily" and v in self.streams[x].variables),
                   None)
        no_day = None
        if day is None:
            no_day = "no daily stream holds it"
        elif "time: point" in (self.streams[day]._vars[v].get("cell_methods") or ""):
            day, no_day = None, f"{day} holds it as an instantaneous value"
        for x in ([st] + ([day] if day else [])) if top else []:
            if x in self.thickness:
                continue
            dz, src = _thickness(self.streams, x)
            nlev = int(self.streams[x]._vars[v]["shape"][0])
            if dz is None or dz.shape[0] != nlev:
                self.left_out[v] = (f"its top-1 m mean needs the layer thicknesses "
                                    f"({THICKNESS_VAR}) for its {nlev} levels, and the files "
                                    f"{'do not carry them' if dz is None else 'carry ' + str(dz.shape[0])}")
                return
            self.thickness[x] = top_metre_weights(dz)
            self.layers_note[x] = src
        rate = bool(s.per_second(v))
        self.plan[v] = {"stream": st, "cadence": s.cadence, "top_metre": top,
                        "day_stream": day, "no_day": no_day, "rate": rate,
                        "units": s.units(v)}

    def _regions(self) -> None:
        reg = self.control.get("regions") or {}
        names = ([DA.VEGETATED] + sorted(reg) + [DA.BANDS]
                 + [{"name": DA.BANDS, "within": r} for r in sorted(reg)])
        self.surface = DA._surface(self.control["case_dir"])
        mk = DA.masks(self.grid, self.surface, names, reg,
                      self.control.get(DA.BAND_EDGES_KEY))
        self.ms: R.MaskSet = mk["masks"]
        self.problems += [f"regions: {p}" for p in mk["problems"]]
        self.band_edges = [float(e) for e in mk["band_edges"]]
        self.band_edges_from = mk["band_edges_from"]
        self.masks_sha = masks_sha256(self.ms)
        self.regions = [{"name": n, "cells": int(m["n_cells"]),
                         "area_km2": float(m["area_km2"]), "rule": m["rule"]}
                        for n, m in self.ms.summary().items()]
        union = np.zeros(self.grid.n_cells, dtype=bool)
        for n in self.ms.names():
            union |= self.ms[n].ravel()
        self.union = np.flatnonzero(union)
        w = self.grid.weight_km2.ravel()[self.union]
        self.W = np.stack([np.where(self.ms[n].ravel()[self.union], w, 0.0)
                           for n in self.ms.names()], axis=1)
        self.wtot = self.W.sum(axis=0)
        if np.any(self.wtot <= 0):
            raise SuiteError("a region has cells but no weight (land fraction 0)")
        veg = self.ms[DA.VEGETATED].ravel()
        self.veg_w = np.where(veg, self.grid.weight_km2.ravel(), 0.0)

    # the period axes ────────────────────────────────────────────────────────
    def djf_present(self, y: int) -> bool:
        return all(_months_held(self.streams[s], y - 1, (12,)) for s in self.period_streams)

    def wy_present(self, y: int) -> bool:
        """A water year is made when the year before is whole in every stream
        read: its October-December are then months of the suite, in the
        maps' sums and in the table's months alike."""
        return (int(y) - 1) in self.years

    def absent_why(self, y: int, what: str) -> str:
        s = self.period_streams[0]
        first = min(self.streams[s].years()) if self.streams[s].years() else None
        if what == "DJF":
            return (f"DJF {y} needs December {y - 1}, which the run's {s} stream does not hold "
                    f"({DA._no_december_why(y, first)}), so it {DA.SEASON_LEFT_OUT}")
        held = (f"its output starts in {first}" if first is None or first > y - 1
                else f"{y - 1} is not a whole year in every stream it reads")
        return (f"{wy_label(y)} needs October-December {y - 1}, which the run does not hold "
                f"whole ({held}), so it is left out, not made from nine months")

    def axes(self, years: Sequence[int]) -> Dict[str, List[Any]]:
        days = []
        for y in years:
            days += [f"{k[0]:04d}-{k[1]:02d}-{k[2]:02d}"
                     for k in G.expected_keys(y, "daily", self.calendar)]
        return {"day": days,
                "month": [DA.month_key(y, m) for y in years for m in range(1, 13)],
                "season": [DA.season_key(y, n) for y in years for n in SEASON_NAMES
                           if n != "DJF" or self.djf_present(y)],
                "year": [str(y) for y in years],
                "wateryear": [wy_label(y) for y in years if self.wy_present(y)]}


# ─────────────────────────────────────────────────────────────────────────────
# THE RUNNING SUMS AND THE TEE
# ─────────────────────────────────────────────────────────────────────────────
class _Sums:
    """What the suite keeps of every block: per field, each month's per-cell
    total (a rate) or time-weighted mean (a state) with how many records gave
    it; each month's covered share per region; the top-1 m fields' region
    means per record; and the daily fields' region means per day."""

    def __init__(self, ctx: _Context, targets: Sequence[int], context: Sequence[int], maps):
        self.ctx, self.maps = ctx, maps
        self.targets, self.context = set(int(y) for y in targets), set(int(y) for y in context)
        self.seen: set = set()
        self.msum: Dict[str, Dict[Tuple[int, int], List[np.ndarray]]] = {}
        self.month: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}
        self.cov: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}
        self.top_means: Dict[str, Dict[Tuple, Tuple[np.ndarray, np.ndarray]]] = {}
        self.day: Dict[str, Dict[Tuple, Tuple[np.ndarray, np.ndarray]]] = {}
        self.map_cover: Dict[str, float] = {}
        self.timing = getattr(ctx, "timing", None)

    def fields_for(self, s: str, y: int) -> List[str]:
        out = []
        if y in self.targets or y in self.context:
            out += [v for v, p in self.ctx.plan.items() if p["stream"] == s]
        if y in self.targets:
            out += [v for v, p in self.ctx.plan.items() if p["day_stream"] == s and v not in out]
        t = self.timing
        if t is not None and s == t.stream and (y in self.targets or y in self.context):
            out += [v for v in t.fields if v not in out]      # a context year: its Oct-Dec
        return out

    def _values(self, s: str, v: str, pb: G.PassBlock) -> np.ndarray:
        a = pb.data[v]
        if self.ctx.plan[v]["top_metre"]:
            a = _top_metre(a, self.ctx.thickness[s])
        a = a.reshape(len(pb.keys), -1)
        return a * G.SECONDS_PER_DAY if pb.per_second[v] else a

    def take(self, s: str, pb: G.PassBlock) -> None:
        st = self.ctx.streams[s]
        y = int(pb.year)
        if self.timing is not None and s == self.timing.stream:
            self.timing.take(pb)
        days = _record_days(st, pb.keys)
        for v in pb.data:
            p = self.ctx.plan.get(v)
            if p is None:
                continue
            want_map = p["stream"] == s and (y in self.targets or y in self.context)
            want_day = p["day_stream"] == s and y in self.targets
            if not (want_map or want_day):
                continue
            x = self._values(s, v, pb)
            xu = x[:, self.ctx.union]
            ok = np.isfinite(xu)
            den = ok.astype(np.float64) @ self.ctx.W
            share = den / self.ctx.wtot
            if want_map:
                self._months(v, pb.keys, x, days, share, y)
                if p["top_metre"]:
                    num = np.where(ok, xu, 0.0) @ self.ctx.W
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mean = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
                    tm = self.top_means.setdefault(v, {})
                    for r, k in enumerate(pb.keys):
                        tm[k] = (mean[r], share[r])
            if want_day:
                num = np.where(ok, xu, 0.0) @ self.ctx.W
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
                dd = self.day.setdefault(v, {})
                for r, k in enumerate(pb.keys):
                    dd[k] = (mean[r], share[r])

    def _months(self, v, keys, x, days, share, y) -> None:
        ok = np.isfinite(x)
        ms = self.msum.setdefault(v, {})
        cv = self.cov.setdefault(v, {})
        for r, k in enumerate(keys):
            ym = (int(k[0]), int(k[1]))
            if y not in self.targets and ym[1] < 10:
                continue                 # a year read for its October-December only
            acc = ms.get(ym)
            if acc is None:
                acc = ms[ym] = [np.zeros(x.shape[1]), np.zeros(x.shape[1]), 0]
            acc[0] += np.where(ok[r], x[r], 0.0) * days[r]
            acc[1] += ok[r]
            acc[2] += days[r]
            cv[ym] = share[r] if ym not in cv else np.minimum(cv[ym], share[r])

    def finish(self, s: str, y: int) -> None:
        """A pass over (s, y) is done: its months become values, and a target
        year's maps are made and written."""
        y = int(y)
        self.seen.add((s, y))
        if self.timing is not None and s == self.timing.stream:
            self.timing.year_done(y)
        fields = [v for v, p in self.ctx.plan.items() if p["stream"] == s]
        want = _expected_per_month(self.ctx.streams[s], y)
        for v in fields:
            ms = self.msum.get(v) or {}
            for m in range(1, 13):
                acc = ms.pop((y, m), None)
                if acc is None:
                    continue
                tot, n, d = acc
                val = tot if self.ctx.plan[v]["rate"] else tot / d
                val[n != want[m]] = np.nan
                self.month.setdefault(v, {})[(y, m)] = val
        if y in self.targets:
            self.maps.year_done(self, fields, y)
        for v in fields:
            mon = self.month.get(v) or {}
            for k in [k for k in mon if k[0] < y or (k[0] == y and k[1] < 10)]:
                del mon[k]

    def note_cover(self, v: str, val: np.ndarray) -> None:
        """The smallest share of the vegetated area a map of v covers."""
        w = self.ctx.veg_w
        tot = float(w.sum())
        c = float(w[np.isfinite(val)].sum()) / tot if tot > 0 else 0.0
        self.map_cover[v] = min(self.map_cover.get(v, 1.0), c)


class _Tee:
    """A stream as run_values reads it, whose passes also feed the suite.

    Everything but read_block_pass is the stream's own. A pass reads the
    table's fields and the suite's over the whole grid; the suite takes each
    block first, then the table's callback gets its fields cut to its window,
    the values it would have read itself."""

    def __init__(self, stream, sums: _Sums):
        self._s, self._sums = stream, sums

    def __getattr__(self, name):
        return getattr(self._s, name)

    def read_block_pass(self, vars, year, block_records=30, reduce=None, window=None,
                        layer=None, require_complete=True, read_ahead=None):
        names = [vars] if isinstance(vars, str) else [str(v) for v in vars]
        s, y = self._s.name, int(year)
        if layer is not None or (s, y) in self._sums.seen:
            return self._s.read_block_pass(vars, year, block_records=block_records,
                                           reduce=reduce, window=window, layer=layer,
                                           require_complete=require_complete,
                                           read_ahead=read_ahead)
        extra = [v for v in self._sums.fields_for(s, y) if v not in names]
        have = self._s.year_records(y)[0]
        lack = self._s.lacking(extra, have) if (extra and have) else {}
        for v, files in sorted(lack.items()):
            self._sums.ctx.problems.append(f"{v} is missing from {', '.join(files)}: its "
                                           f"{y} products have no value")
        extra = [v for v in extra if v not in lack]

        def both(pb):
            self._sums.take(s, pb)
            if reduce is not None:
                reduce(_part(pb, names, window))
        out = self._s.read_block_pass(names + extra, year, block_records=block_records,
                                      reduce=both, window=None, layer=None,
                                      require_complete=require_complete, read_ahead=read_ahead)
        self._sums.finish(s, y)
        return out


def _part(pb: G.PassBlock, names: Sequence[str], window) -> G.PassBlock:
    """The block as a pass of `names` over `window` would have given it."""
    if window is None:
        data = {v: pb.data[v] for v in names}
    else:
        r0, r1, c0, c1 = window
        data = {v: pb.data[v][..., r0:r1, c0:c1] for v in names}
    return G.PassBlock(year=pb.year, first=pb.first, keys=pb.keys, labels=pb.labels,
                       data=data, units={v: pb.units[v] for v in names},
                       per_second={v: pb.per_second[v] for v in names}, missing=pb.missing)


# ─────────────────────────────────────────────────────────────────────────────
# THE PRODUCTS ON DISK
# ─────────────────────────────────────────────────────────────────────────────
def _units(plan: Dict[str, Any], kind: str) -> str:
    u = plan["units"]
    if not plan["rate"]:
        return u
    per_day = G.per_day_units(plan.get("name") or "", u)
    if kind == "day":
        return per_day
    yr = per_day.replace("/d", "/yr")
    return DA.period_units(yr, True, {"month": "month", "season": "season", "year": "yr",
                                      "wateryear": "water year"}[kind])


def _how(plan: Dict[str, Any], kind: str) -> str:
    words = {"day": "day", "month": "month", "season": "season", "year": "calendar year",
             "wateryear": "water year (October to September, named by the year it ends)"}
    base = DA.how_text(plan["rate"], words[kind])
    if plan["top_metre"]:
        base += ("; the top-1 m mean of its layers, each weighted by its thickness inside "
                 "the top metre")
    return base


def _spatial(d, grid: G.Grid) -> Tuple[str, ...]:
    for name, n in zip(grid.dims, grid.shape):
        d.createDimension(name, n)
    if grid.regular:
        d.createVariable("lat", "f8", ("lat",))[:] = grid.lat_axis
        d.createVariable("lon", "f8", ("lon",))[:] = grid.lon_axis
    else:
        d.createVariable("lat", "f8", grid.dims)[:] = grid.lat
        d.createVariable("lon", "f8", grid.dims)[:] = grid.lon
    d["lon"].note = "degrees east in [-180, 180)"
    d.createVariable("weight_km2", "f8", grid.dims)[:] = grid.weight_km2
    d["weight_km2"].note = "cell area x land fraction, the weight of every region number"
    return tuple(grid.dims)


def _globals(d, ctx: _Context, what: str) -> None:
    d.title = f"ELM whole-domain suite: {what}"
    d.SUITE_VERSION = np.int32(SUITE_VERSION)
    d.run_dir = str(ctx.run_dir)
    d.manifest = MANIFEST


class _Maps:
    """maps_year.nc, maps_wateryear.nc and maps_season.nc, created for a full
    build or opened to add years, each field's slice written as its year is
    done."""

    def __init__(self, folder: Path, ctx: _Context, years: Sequence[int], add: bool):
        import netCDF4
        self.ctx, self.add = ctx, add
        self.ds = {}
        self.index: Dict[str, Dict[Any, int]] = {}
        self.made: Dict[str, List[str]] = {"year": [], "wateryear": [], "season": []}
        fields = sorted(ctx.plan)
        for kind in MAP_KINDS:
            p = folder / f"maps_{kind}.nc"
            if add:
                d = netCDF4.Dataset(p, "a")
            else:
                d = netCDF4.Dataset(p, "w", format="NETCDF4")
                _globals(d, ctx, f"maps per {kind}")
                sp = _spatial(d, ctx.grid)
                tdim = "wateryear" if kind == "wateryear" else "year"
                d.createDimension(tdim, None)
                tv = d.createVariable(tdim, "i4", (tdim,))
                tv.note = ("named by the year it ends: October of the year before to September"
                           if kind == "wateryear" else "calendar year")
                dims: Tuple[str, ...] = (tdim,) + sp
                chunks = [1] + list(ctx.grid.shape)
                if kind == "season":
                    d.createDimension("season", len(SEASON_NAMES))
                    sv = d.createVariable("season", str, ("season",))
                    sv[:] = np.array(SEASON_NAMES, dtype=object)
                    sv.note = "DJF holds the December of the year before"
                    dims = (tdim, "season") + sp
                    chunks = [1, 1] + list(ctx.grid.shape)
                for v in fields:
                    pl = dict(ctx.plan[v], name=v)
                    x = d.createVariable(v, "f4", dims, fill_value=np.float32(np.nan),
                                         chunksizes=chunks)
                    x.units = _units(pl, "season" if kind == "season" else kind)
                    x.how = _how(pl, kind)
                    x.stream = pl["stream"]
            tname = "wateryear" if kind == "wateryear" else "year"
            have = [int(t) for t in np.asarray(d[tname][:]).ravel()] if add else []
            self.index[kind] = {t: i for i, t in enumerate(have)}
            self.ds[kind] = d
        for y in years:
            self._slot("year", y)
            self._slot("season", y)
            if ctx.wy_present(y):
                self._slot("wateryear", y)

    def _slot(self, kind: str, t: int) -> int:
        ix = self.index[kind]
        if t not in ix:
            ix[t] = len(ix)
            tname = "wateryear" if kind == "wateryear" else "year"
            self.ds[kind][tname][ix[t]] = np.int32(t)
        return ix[t]

    def year_done(self, sums: _Sums, fields: Sequence[str], y: int) -> None:
        ctx = self.ctx
        shape = ctx.grid.shape
        cal = ctx.calendar
        for v in fields:
            mon = sums.month.get(v) or {}
            rate = ctx.plan[v]["rate"]
            ks = [(y, m) for m in range(1, 13)]
            val = _combine([mon.get(k) for k in ks], ks, rate, cal)
            val = np.full(ctx.grid.n_cells, np.nan) if val is None else val
            sums.note_cover(v, val)
            self.ds["year"][v][self.index["year"][y]] = val.reshape(shape).astype(np.float32)
            for j, (name, months) in enumerate(DA.SEASONS):
                parts = DA.season_parts(y, name, months)
                val = None
                if name != "DJF" or ctx.djf_present(y):
                    val = _combine([mon.get(k) for k in parts], parts, rate, cal)
                    val = np.full(ctx.grid.n_cells, np.nan) if val is None else val
                    sums.note_cover(v, val)
                    out = val
                else:
                    out = np.full(ctx.grid.n_cells, np.nan)
                self.ds["season"][v][self.index["season"][y], j] = \
                    out.reshape(shape).astype(np.float32)
            if ctx.wy_present(y):
                ks = _wy_keys(y)
                val = _combine([mon.get(k) for k in ks], ks, rate, cal)
                val = np.full(ctx.grid.n_cells, np.nan) if val is None else val
                sums.note_cover(v, val)
                self.ds["wateryear"][v][self.index["wateryear"][y]] = \
                    val.reshape(shape).astype(np.float32)
        lab = str(y)
        if lab not in self.made["year"]:
            self.made["year"].append(lab)
            self.made["season"] += [DA.season_key(y, n) for n in SEASON_NAMES
                                    if n != "DJF" or ctx.djf_present(y)]
            if ctx.wy_present(y):
                self.made["wateryear"].append(wy_label(y))

    def close(self) -> None:
        for d in self.ds.values():
            d.close()


def _series_file(path: Path, ctx: _Context, kind: str, labels: Sequence[str], add: bool,
                 values: Dict[str, np.ndarray], covered: Dict[str, np.ndarray]) -> None:
    """Write (full build) or append (add) one series file: per field the
    (region, period) values and their covered shares."""
    import netCDF4
    regions = ctx.ms.names()
    if add:
        d = netCDF4.Dataset(path, "a")
        n0 = len(d.dimensions[kind])
    else:
        d = netCDF4.Dataset(path, "w", format="NETCDF4")
        _globals(d, ctx, f"region series per {kind}")
        d.createDimension("region", len(regions))
        rv = d.createVariable("region", str, ("region",))
        rv[:] = np.array(regions, dtype=object)
        d.createDimension(kind, None)
        if kind in ("year", "wateryear"):
            tv = d.createVariable(kind, "i4", (kind,))
        else:
            tv = d.createVariable(kind, str, (kind,))
        tv.note = {"day": "the day, labelled by its start",
                   "month": "the month, as the regional table names it",
                   "season": "the season, as the regional table names it (DJF: the year of "
                             "its January)",
                   "year": "calendar year",
                   "wateryear": "October of the year before to September, named by the year "
                                "it ends"}[kind]
        chunk = [len(regions), 512 if kind == "day" else 64]
        for v in sorted(values):
            pl = dict(ctx.plan[v], name=v)
            x = d.createVariable(v, "f8", ("region", kind), fill_value=np.nan,
                                 chunksizes=chunk)
            x.units = _units(pl, kind)
            x.how = _how(pl, kind) + ", the area-weighted mean over the region's cells"
            x.stream = pl["day_stream"] if kind == "day" else pl["stream"]
            c = d.createVariable(v + COVERED, "f8", ("region", kind), fill_value=np.nan,
                                 chunksizes=chunk)
            c.units = "1"
            c.how = ("the share of the region's area (cell area x land fraction) with a "
                     "value, the smallest over the records of the period")
        n0 = 0
    n = len(labels)
    if n:
        if kind in ("year", "wateryear"):
            d[kind][n0:n0 + n] = np.array([int(str(t).replace("WY", "")) for t in labels],
                                          dtype=np.int32)
        else:
            d[kind][n0:n0 + n] = np.array(list(labels), dtype=object)
        for v in sorted(values):
            d[v][:, n0:n0 + n] = values[v]
            d[v + COVERED][:, n0:n0 + n] = covered[v]
    d.close()


def _read_months(path: Path, fields: Sequence[str], labels: Sequence[str]
                 ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Months already in a suite's series_month.nc: (values, covered) per
    field as (region, month) in the order of `labels`."""
    import netCDF4
    with netCDF4.Dataset(path) as d:
        d.set_auto_mask(False)
        have = [str(x) for x in np.asarray(d["month"][:], dtype=object)]
        pos = [have.index(lab) for lab in labels]
        vals = {v: np.asarray(d[v][:, pos], dtype=np.float64) for v in fields if v in d.variables}
        cov = {v: np.asarray(d[v + COVERED][:, pos], dtype=np.float64)
               for v in fields if v in d.variables}
    return vals, cov


# ─────────────────────────────────────────────────────────────────────────────
# THE BUILD
# ─────────────────────────────────────────────────────────────────────────────
def _decide(old: Optional[Dict[str, Any]], ctx: _Context, out: Path
            ) -> Tuple[str, str, List[int]]:
    """("reuse" | "add" | "build", why, the years to add)."""
    if old is None:
        return "build", "no suite manifest is there", []
    if old.get("SUITE_VERSION") != SUITE_VERSION:
        return "build", (f"the builder makes version {SUITE_VERSION}, the suite is "
                         f"{old.get('SUITE_VERSION')}"), []
    for key, now in (("run_dir", str(ctx.run_dir)), ("masks_sha256", ctx.masks_sha),
                     ("band_edges_m", ctx.band_edges), ("fields_asked", ctx.fields_asked)):
        if old.get(key) != now:
            return "build", f"{key} changed", []
    gone = [f"{p}.nc" for p in PRODUCTS if not (out / f"{p}.nc").is_file()]
    if gone or not (out / MASKS_FILE).is_file():
        return "build", f"product files are missing: {', '.join(gone) or MASKS_FILE}", []
    oh = old.get("history") or {}
    if sorted(oh) != sorted(ctx.history):
        return "build", "the streams read changed", []
    added = False
    for s, now in ctx.history.items():
        was = {f["name"]: f for f in oh[s]}
        nf = {f["name"]: f for f in now}
        lost = sorted(set(was) - set(nf))
        if lost:
            return "build", f"history file {lost[0]} is gone", []
        for name, f in was.items():
            if f != nf[name]:
                return "build", f"history file {name} changed (size or modification time)", []
        added |= bool(set(nf) - set(was))
    if not added:
        return "reuse", "every history file, region, field and the version are as recorded", []
    old_years = [int(y) for y in (old.get("periods") or {}).get("year") or []]
    if [y for y in old_years if y not in ctx.years] or not old_years:
        return "build", "a year the suite holds is no longer whole", []
    new = [y for y in ctx.years if y not in old_years]
    if not new:
        return "reuse", "new history files hold no new whole year", []
    if new != list(range(max(old_years) + 1, max(old_years) + 1 + len(new))):
        return "build", f"the new years {new} do not follow {max(old_years)}", []
    return "add", f"new history files complete {', '.join(map(str, new))}", new


# WHAT A REUSE MAY UPDATE (2026-11-04). Some of what the manifest records
# describes the products without changing them: where the band edges come from
# (moved from a constant into the registry with the same values), a region's
# rule in words, the control's name, the surface file's path. When only these
# differ, the products are reused and the manifest alone is brought up to
# date; when nothing differs, nothing is written.
DESCRIPTIVE = ("band_edges_from", "regions", "control", "surface_file")


def _refreshed(old: Dict[str, Any], ctx: "_Context") -> Optional[Dict[str, Any]]:
    now = {"band_edges_from": ctx.band_edges_from, "regions": ctx.regions,
           "control": ctx.control.get("name"), "surface_file": str(ctx.surface)}
    changed = [k for k in DESCRIPTIVE if old.get(k) != now[k]]
    if not changed:
        return None
    m = dict(old, **{k: now[k] for k in changed})
    m["manifest_refreshed"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "keys": changed}
    return m


def build(control: Dict[str, Any], out_dir, run_dir=None, fields: Optional[Sequence[str]] = None,
          notes_path=None) -> Dict[str, Any]:
    """The suite of one run in out_dir, reused, extended or built.

    control   the control's registry entry (domain_controls.get): case_dir
              (its surface file), regions, and run_dir
    run_dir   the run to build, when not the control's own (an experiment's
              run on the control's grid)
    fields    the fields to hold (default: the catalog's rows, note_rows)

    Returns {"action": "reused" | "added" | "built", "why", "made": {kind:
    [labels]}, "seconds", "manifest"}.
    """
    t0 = time.time()
    rd = Path(str(run_dir or control["run_dir"]))
    out = Path(str(out_dir))
    ctx = _Context(control, rd, list(fields) if fields is not None else note_rows(notes_path))
    old = load(out)
    action, why, new_years = _decide(old, ctx, out)
    if action == "reuse":
        fresh = _refreshed(old, ctx)
        if fresh is not None:
            why += (f"; the manifest's {', '.join(fresh['manifest_refreshed']['keys'])} "
                    f"brought up to date")
        due = _timing_due(old, ctx, out)
        if due:
            # the other products stay as they are: only timing.nc is made
            rec = _timing_pass(ctx, out, [int(y) for y in old["periods"]["year"]], due)
            fresh = _with_timing(fresh or dict(old), rec, out)
            why += f"; {TIMING_FILE} made ({due})"
        if fresh is not None:
            _write_json(out / MANIFEST, fresh)
        return {"action": "reused", "why": why, "made": {}, "seconds": time.time() - t0,
                "manifest": str(out / MANIFEST)}
    out.mkdir(parents=True, exist_ok=True)
    if (out / MANIFEST).exists():
        (out / MANIFEST).unlink()
    add = action == "add"
    targets = new_years if add else list(ctx.years)
    work = out if add else out / ".building"
    if not add:
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
    timing_now = add and not _timing_due(old, ctx, out)
    ctx.timing = _Timing.make(ctx, targets) if (not add or timing_now) else None
    made = _make(ctx, work, targets, add, old, out)
    rec = None
    if ctx.timing is not None:
        ctx.timing.finish()
        rec = _write_timing(work / TIMING_FILE, ctx, ctx.timing,
                            out / TIMING_FILE if add else None)
    if not add:
        for f in [f"{p}.nc" for p in PRODUCTS] + ([TIMING_FILE] if rec else []):
            os.replace(work / f, out / f)
        ctx.ms.to_netcdf(out / MASKS_FILE)
        shutil.rmtree(work, ignore_errors=True)
    elif not timing_now:
        # the timing record no longer matches the methods: every year is made again
        rec = _timing_pass(ctx, out, list(ctx.years), _timing_due(old, ctx, out))
    seconds = time.time() - t0
    made["timing"] = rec if rec is not None else _timing_none(ctx)
    man = _manifest(ctx, made, old if add else None, "added" if add else "built", why,
                    seconds, out)
    _write_json(out / MANIFEST, man)
    return {"action": man["last_build"]["action"], "why": why, "made": made["periods"],
            "seconds": seconds, "manifest": str(out / MANIFEST)}


def _make(ctx: _Context, work: Path, targets: Sequence[int], add: bool,
          old: Optional[Dict[str, Any]], out: Path) -> Dict[str, Any]:
    """Read the passes, write the maps as each year is done, then the series."""
    targets = sorted(int(y) for y in targets)
    context = sorted({y - 1 for y in targets} - set(targets))
    maps = _Maps(work, ctx, targets, add)
    try:
        sums = _Sums(ctx, targets, context, maps)
        tees = {s: _Tee(st, sums) for s, st in ctx.streams.items()}
        table_fields = [v for v, p in ctx.plan.items() if not p["top_metre"]]
        rv = DA.run_values(ctx.run_dir, ctx.ms, table_fields, targets, streams=tees)
        ctx.problems += [f"table: {p}" for p in rv.get("problems") or []]
        # the passes the table did not make, each (stream, year) once, in order
        for s in ctx.used:
            for y in sorted(set(targets) | set(context)):
                if (s, y) in sums.seen or y not in ctx.streams[s].years():
                    continue
                vs = sums.fields_for(s, y)
                if not vs:
                    continue
                ctx.streams[s].read_block_pass(vs, y, reduce=lambda pb, s=s: sums.take(s, pb),
                                               require_complete=y in targets)
                sums.finish(s, y)
    finally:
        maps.close()
    tm = _top_metre_tables(ctx, sums, targets, context)
    return _write_series(ctx, sums, rv, tm, targets, context, work, add, old, out, maps.made)


def _top_metre_tables(ctx: _Context, sums: _Sums, targets, context) -> Dict[str, Any]:
    """The top-1 m fields by the table's own rule, from their region means per
    record: what run_values would give, had it a value per cell to start from."""
    out: Dict[str, Any] = {}
    regions = ctx.ms.names()
    for v, p in ctx.plan.items():
        if not p["top_metre"]:
            continue
        st = ctx.streams[p["stream"]]
        got = sums.top_means.get(v) or {}
        keys = sorted(got)
        if not keys:
            continue
        mean = np.array([got[k][0] for k in keys])
        share = np.array([got[k][1] for k in keys])
        units = _units(dict(p, name=v), "day") if p["rate"] else p["units"]
        rs = R.RegionSeries(stream=st.name, cadence=st.cadence, calendar=st.calendar,
                            steps_per_day=st.steps_per_day, keys=keys, regions=regions,
                            fields=[v], units={v: units},
                            values={(r, v): mean[:, j] for j, r in enumerate(regions)},
                            valid_fraction={(r, v): share[:, j] for j, r in enumerate(regions)},
                            region_table={}, per={v: "day" if p["rate"] else None})
        how = "sum" if p["rate"] else "mean"
        o: Dict[str, Any] = {"values": {}, "missing": {}}
        for reg in regions:
            vals, why = rs.by_year(reg, v, targets, how=how)
            o["values"].setdefault(reg, {})[v] = {
                str(y): (None if vals.get(y) is None else float(vals[y])) for y in targets}
            if why:
                o["missing"].setdefault(reg, {})[v] = {str(y): w for y, w in why.items()}
        prev = [y for y in context if y in st.years()]
        DA.months_and_seasons(o, rs, st.name, regions, [v], targets, prev,
                              first_year=min(st.years()) if st.years() else None)
        out[v] = o
    return out


def _write_series(ctx, sums, rv, tm, targets, context, work, add, old, out, made_maps
                  ) -> Dict[str, Any]:
    regions = ctx.ms.names()
    nreg = len(regions)
    ax = ctx.axes(targets)
    fields = sorted(ctx.plan)
    missing: Dict[str, Any] = {}

    def table(v):
        return tm.get(v) if ctx.plan[v]["top_metre"] else rv

    def cov_of(v, keys):
        cs = [(sums.cov.get(v) or {}).get(k) for k in keys]
        if any(c is None for c in cs):
            return np.full(nreg, np.nan)
        return np.min(np.vstack(cs), axis=0)

    # months, seasons and years: the table's own numbers
    vals: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in ("month", "season", "year")}
    covs: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in ("month", "season", "year")}
    for v in fields:
        t = table(v) or {}
        for kind, src, labels in (("month", "monthly", ax["month"]),
                                  ("season", "seasonal", ax["season"]),
                                  ("year", "values", ax["year"])):
            a = np.full((nreg, len(labels)), np.nan)
            c = np.full((nreg, len(labels)), np.nan)
            for i, lab in enumerate(labels):
                if kind == "month":
                    y, m = int(lab[:4]), int(lab[5:])
                    keys = [(y, m)]
                elif kind == "season":
                    y, n = int(lab[:4]), lab[5:]
                    keys = DA.season_parts(y, n, dict(DA.SEASONS)[n])
                else:
                    keys = [(int(lab), m) for m in range(1, 13)]
                c[:, i] = cov_of(v, keys)
                for j, reg in enumerate(regions):
                    x = (((t.get(src) or {}).get(reg) or {}).get(v) or {}).get(lab)
                    a[j, i] = np.nan if x is None else float(x)
            vals[kind][v], covs[kind][v] = a, c
        for key in ("missing", "monthly_missing", "seasonal_missing"):
            for reg, fv in (t.get(key) or {}).items():
                if v in fv:
                    missing.setdefault(key, {}).setdefault(reg, {})[v] = fv[v]
    # water years: the months' totals or day-weighted means
    wv: Dict[str, np.ndarray] = {}
    wc: Dict[str, np.ndarray] = {}
    mlab = {lab: i for i, lab in enumerate(ax["month"])}
    before = [DA.month_key(min(targets) - 1, m) for m in (10, 11, 12)]
    old_v, old_c = ({}, {})
    if add and any(lab in ax["wateryear"] for lab in [wy_label(min(targets))]):
        old_v, old_c = _read_months(out / "series_month.nc", fields, before)
    for v in fields:
        rate = ctx.plan[v]["rate"]
        a = np.full((nreg, len(ax["wateryear"])), np.nan)
        c = np.full((nreg, len(ax["wateryear"])), np.nan)
        for i, lab in enumerate(ax["wateryear"]):
            y = int(lab[2:])
            ks = _wy_keys(y)
            cols, ccols = [], []
            for (yy, m) in ks:
                ml = DA.month_key(yy, m)
                if ml in mlab:
                    cols.append(vals["month"][v][:, mlab[ml]])
                    ccols.append(covs["month"][v][:, mlab[ml]])
                elif v in old_v and ml in before:
                    cols.append(old_v[v][:, before.index(ml)])
                    ccols.append(old_c[v][:, before.index(ml)])
                else:
                    cols.append(np.full(nreg, np.nan))
                    ccols.append(np.full(nreg, np.nan))
            for j in range(nreg):
                xs = [float(col[j]) for col in cols]
                if all(np.isfinite(xs)):
                    a[j, i] = float(_combine(xs, ks, rate, ctx.calendar))
            c[:, i] = np.min(np.vstack(ccols), axis=0)
        wv[v], wc[v] = a, c
    # days
    dv: Dict[str, np.ndarray] = {}
    dc: Dict[str, np.ndarray] = {}
    dkeys = []
    for y in targets:
        dkeys += [k for k in G.expected_keys(y, "daily", ctx.calendar)]
    day_fields = [v for v in fields if ctx.plan[v]["day_stream"]]
    for v in day_fields:
        got = sums.day.get(v) or {}
        a = np.full((nreg, len(dkeys)), np.nan)
        c = np.full((nreg, len(dkeys)), np.nan)
        for i, k in enumerate(dkeys):
            if k in got:
                a[:, i], c[:, i] = got[k]
        dv[v], dc[v] = a, c
    _series_file(work / "series_day.nc", ctx, "day", ax["day"], add, dv, dc)
    _series_file(work / "series_month.nc", ctx, "month", ax["month"], add, vals["month"],
                 covs["month"])
    _series_file(work / "series_season.nc", ctx, "season", ax["season"], add, vals["season"],
                 covs["season"])
    _series_file(work / "series_year.nc", ctx, "year", ax["year"], add, vals["year"],
                 covs["year"])
    _series_file(work / "series_wateryear.nc", ctx, "wateryear", ax["wateryear"], add, wv, wc)
    return {"periods": {"day": ax["day"], "month": ax["month"], "season": ax["season"],
                        "year": ax["year"], "wateryear": ax["wateryear"]},
            "maps": made_maps, "missing": missing, "map_cover": dict(sums.map_cover)}


def _manifest(ctx: _Context, made: Dict[str, Any], old: Optional[Dict[str, Any]], action: str,
              why: str, seconds: float, out: Path) -> Dict[str, Any]:
    """suite.json: SUITE_VERSION, the code's commit, what the products depend
    on, the fields and periods present and absent with the reasons."""
    per = {k: list(v) for k, v in made["periods"].items()}
    if old:
        oldp = old.get("periods") or {}
        per = {k: list(oldp.get(k) or []) + per[k] for k in per}
    years = [int(y) for y in per["year"]]
    absent: Dict[str, Dict[str, str]] = {"season": {}, "wateryear": {}}
    for y in years:
        if not ctx.djf_present(y):
            absent["season"][DA.season_key(y, "DJF")] = ctx.absent_why(y, "DJF")
        if not ctx.wy_present(y):
            absent["wateryear"][wy_label(y)] = ctx.absent_why(y, "wateryear")
    fields: Dict[str, Any] = {}
    cover_old = {v: (e or {}).get("maps_min_covered_share")
                 for v, e in ((old or {}).get("fields") or {}).items()}
    for v in sorted(ctx.plan):
        p = dict(ctx.plan[v], name=v)
        prods = ["maps_year", "maps_wateryear", "maps_season"]
        if p["day_stream"]:
            prods.append("series_day")
        prods += ["series_month", "series_season", "series_year", "series_wateryear"]
        cov = made["map_cover"].get(v)
        if cover_old.get(v) is not None:
            cov = min(cover_old[v], cov) if cov is not None else cover_old[v]
        e: Dict[str, Any] = {
            "products": prods, "stream": p["stream"], "day_stream": p["day_stream"],
            "how": "total" if p["rate"] else "mean",
            "units": {k: _units(p, k) for k in SERIES_KINDS if k != "day" or p["day_stream"]},
            "maps_min_covered_share": None if cov is None else round(float(cov), 6)}
        if not p["day_stream"]:
            e["no_series_day"] = p["no_day"]
        if p["top_metre"]:
            w = ctx.thickness[p["stream"]]
            ok = np.all(np.isfinite(w), axis=0)
            same = bool(ok.any() and np.allclose(w[:, ok], w[:, ok][:, :1], rtol=0, atol=1e-12))
            e["levels"] = {
                "rule": "the top-1 m mean of its layers, each weighted by its thickness inside "
                        "the top metre, a layer crossing 1 m by its part above it",
                "thicknesses_from": {s: ctx.layers_note[s] for s in ctx.thickness},
                "layer_weights": ([round(float(x), 6) for x in w[:, ok][:, 0]] if same
                                  else "they differ between cells")}
        fields[v] = e
    files = {f.name: int(f.stat().st_size) for f in sorted(out.glob("*.nc"))}
    edges_note = ctx.band_edges_from
    m = {
        "SUITE_VERSION": SUITE_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_build": {"action": action, "why": why, "seconds": round(seconds, 1),
                       "made": made["periods"] if action == "added" else
                       {k: f"all {len(v)}" for k, v in made["periods"].items()}},
        "first_build_seconds": (old or {}).get("first_build_seconds", round(seconds, 1)),
        "code": _git(),
        "control": ctx.control.get("name"),
        "run_dir": str(ctx.run_dir),
        "history": ctx.history,
        "masks_file": MASKS_FILE,
        "masks_sha256": ctx.masks_sha,
        "regions": ctx.regions,
        "surface_file": str(ctx.surface),
        "band_edges_m": ctx.band_edges,
        "band_edges_from": edges_note,
        "fields_asked": ctx.fields_asked,
        "fields": fields,
        "fields_left_out": dict(ctx.left_out),
        "periods": per,
        "periods_absent": {k: v for k, v in absent.items() if v},
        "month_maps": ("not built: the monthly history files listed under history hold them "
                       "and are referenced, not copied"),
        "normals": ("not stored: a normal is the mean of the plan's normal years, made from "
                    "these maps or series when it is asked for"),
        "files": files,
        "total_bytes": int(sum(files.values())),
        "missing": _merged((old or {}).get("missing") or {}, made["missing"]),
        "problems": list(dict.fromkeys(list((old or {}).get("problems") or []) + ctx.problems)),
        "timing": made.get("timing"),
    }
    return m


def _merged(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """b's nested keys added to a's: the reasons of the years already held and
    of the years added."""
    out = dict(a)
    for k, v in b.items():
        out[k] = _merged(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) \
            else v
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: domain_suite.py <control name> <suite dir>")
        sys.exit(2)
    import domain_controls
    r = build(domain_controls.get(sys.argv[1]), sys.argv[2])
    print(json.dumps({k: (v if k != "made" else {a: len(b) for a, b in v.items()})
                      for k, v in r.items()}, indent=1, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# THE TIMING MEASURES (timing.nc)
# ─────────────────────────────────────────────────────────────────────────────
# TIMING.NC (2026-11-05). The six timing measures of domain_methods (snow_peak,
# snow_gone, leaf_green_up, photosynthesis_onset, growth_peak, half_runoff),
# per cell and year, every setting the contracts name: the date, the reason
# code where there is no date, and the quality flags. Made from the suite's
# own passes of the daily stream (the timing fields ride on the pass of each
# year, and a year read only for its October-December gives a water year its
# start), so no year file is read again. A suite built before timing.nc
# existed, or whose timing record no longer matches the methods' versions,
# settings or the registry's start months, gets it from a pass of its own over
# the daily year files, the timing fields only, and nothing else is remade.
# Variables are (lat, lon, year): the year axis last, as the series' period
# axis, so a year added is appended where the others' new periods go.
class _Timing:
    """The timing measures' collector: each daily block's timing fields per
    cell and day, a calendar year at a time, handed to the methods' period
    assembler; each period's measures computed as soon as it is whole."""

    @classmethod
    def make(cls, ctx: "_Context", years: Sequence[int]) -> Optional["_Timing"]:
        spec = _timing_spec(ctx)
        return cls(ctx, years, spec) if spec["measures"] else None

    def __init__(self, ctx: "_Context", years: Sequence[int], spec: Dict[str, Any]):
        import domain_methods as M
        self.M, self.spec = M, spec
        self.stream, self.fields = spec["stream"], list(spec["fields"])
        st = ctx.streams[self.stream]
        self.cal = st.calendar
        kinds = sorted({M.TIMING_KIND[m] for m in spec["measures"]})
        per = {k: tuple(dict.fromkeys(M.TIMING_FIELD[m] for m in spec["measures"]
                                      if M.TIMING_KIND[m] == k)) for k in kinds}
        self.years = sorted(int(y) for y in years)
        self.asm = M.PeriodAssembler(self.years, st.years(), self.cal, tuple(ctx.grid.shape),
                                     ctx.control, kinds, per)
        self.nc = int(np.prod(ctx.grid.shape))
        self.cur: Dict[int, Dict[str, np.ndarray]] = {}
        self.results: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]] = {
            m: {} for m in spec["measures"]}

    def take(self, pb: G.PassBlock) -> None:
        y = int(pb.year)
        arr = self.cur.get(y)
        if arr is None:
            nd = G.days_in_year(y, self.cal)
            arr = self.cur[y] = {f: np.full((nd, self.nc), np.nan) for f in self.fields}
        for f in self.fields:
            if f not in pb.data:
                continue
            v = pb.data[f].reshape(len(pb.keys), -1)
            if pb.per_second[f]:
                v = v * G.SECONDS_PER_DAY            # a rate's day total, as read_year makes it
            for i, k in enumerate(pb.keys):
                arr[f][R.day_of_year(k, self.cal) - 1] = v[i]

    def year_done(self, y: int) -> None:
        arr = self.cur.pop(int(y), None)
        if arr is not None:
            for p in self.asm.add(int(y), arr):
                self._compute(p)

    def finish(self) -> None:
        for y in sorted(self.cur):
            self.year_done(y)
        for p in self.asm.finish():
            self._compute(p)

    def _compute(self, p) -> None:
        for mid in self.spec["measures"]:
            if self.M.TIMING_KIND[mid] == p.kind and p.year in self.years:
                r = self.M.METHODS[mid].run(p)
                self.results[mid][p.year] = {lab: {k: v[k] for k in ("date", "reason", "flags")}
                                             for lab, v in r["variants"].items()}


def _timing_spec(ctx: "_Context") -> Dict[str, Any]:
    """What timing.nc holds for this suite: the measures whose field has a
    daily series here, each with its version and every setting."""
    import domain_methods as M
    have = [f for f in M.TIMING_FIELDS if f in ctx.plan and ctx.plan[f]["day_stream"]]
    streams = [ctx.plan[f]["day_stream"] for f in have]
    stream = max(sorted(set(streams)), key=streams.count) if streams else None
    fields = [f for f in have if ctx.plan[f]["day_stream"] == stream]
    measures = {m: {"version": M.METHODS[m].version,
                    "variants": {lab: st for lab, st in M.METHODS[m].variants()}}
                for m in M.TIMING_IDS if M.TIMING_FIELD[m] in fields}
    ys, ws = M.start_months(ctx.control)
    return {"file": TIMING_FILE, "stream": stream, "fields": fields, "measures": measures,
            "not_built": {m: f"{M.TIMING_FIELD[m]} has no daily series in this suite"
                          for m in M.TIMING_IDS if m not in measures},
            "year_start_month": ys, "water_year_start_month": ws}


TIMING_KEYS = ("stream", "fields", "measures", "year_start_month", "water_year_start_month")


def _timing_due(old: Optional[Dict[str, Any]], ctx: "_Context", out: Path) -> Optional[str]:
    """Why timing.nc must be made, or None when it is current (or cannot be)."""
    spec = _timing_spec(ctx)
    if not spec["measures"]:
        return None
    have = (old or {}).get("timing") or {}
    if not (out / TIMING_FILE).is_file() or not have.get("years"):
        return f"{TIMING_FILE} is not there"
    if any(json.loads(json.dumps(spec[k])) != have.get(k) for k in TIMING_KEYS):
        return "the timing measures' versions, settings, fields or start months changed"
    if [int(y) for y in have["years"]] != [int(y) for y in (old.get("periods") or {}).get("year")
                                           or []]:
        return "its years are not the suite's"
    return None


def _timing_none(ctx: "_Context") -> Dict[str, Any]:
    spec = _timing_spec(ctx)
    return {"file": None, "not_built": spec["not_built"],
            "why": "no field a timing measure reads has a daily series in this suite"}


def _timing_pass(ctx: "_Context", out: Path, years: Sequence[int], why: str) -> Dict[str, Any]:
    """timing.nc from a pass of its own over the daily year files (the timing
    fields only, each year file once), when the suite's passes did not make it."""
    import domain_methods as M
    t0 = time.time()
    t = _Timing.make(ctx, years)
    if t is None:
        return _timing_none(ctx)
    st = ctx.streams[t.stream]
    for c in t.asm.needed():
        t.cur[c] = M.read_year(st, c, t.fields)
        t.year_done(c)
    t.finish()
    rec = _write_timing(out / TIMING_FILE, ctx, t, None)
    rec["made_by"] = f"a pass of its own over the {t.stream} year files: {why}"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def _timing_name(mid: str, label: str) -> str:
    return f"{mid}__" + re.sub(r"[^A-Za-z0-9_]", "_", label.replace(".", "p").replace("=", "_"))


def _write_timing(path: Path, ctx: "_Context", t: _Timing,
                  old_path: Optional[Path]) -> Dict[str, Any]:
    """timing.nc: the years of `t` beside those of old_path (an add keeps the
    years already there, value for value)."""
    import netCDF4
    M = t.M
    old: Dict[str, np.ndarray] = {}
    old_years: List[int] = []
    if old_path is not None and Path(old_path).is_file():
        with netCDF4.Dataset(old_path) as d:
            d.set_auto_mask(False)
            old_years = [int(y) for y in np.asarray(d["year"][:])]
            old = {vn: np.asarray(v[:]) for vn, v in d.variables.items()
                   if v.dimensions[-1:] == ("year",) and vn != "year"}
    years = old_years + [y for y in t.years if y not in old_years]
    tmp = Path(str(path) + ".tmp")
    nvar = 0
    with netCDF4.Dataset(tmp, "w") as d:
        dims = _spatial(d, ctx.grid)
        _globals(d, ctx, "timing measures per cell (domain_methods)")
        d.calendar = str(t.cal)
        d.year_start_month = np.int32(t.spec["year_start_month"])
        d.water_year_start_month = np.int32(t.spec["water_year_start_month"])
        d.reasons = json.dumps({str(c): txt for c, txt in M.REASONS.values()})
        d.flags = json.dumps({str(b): f"{k}: {txt}" for k, (b, txt) in M.FLAGS.items()})
        d.createDimension("year", len(years))
        yv = d.createVariable("year", "i4", ("year",))
        yv[:] = years
        yv.note = ("the year a year measure belongs to; for snow_peak, snow_gone and "
                   "half_runoff, the year their water year ends in")
        for mid, info in t.spec["measures"].items():
            wy = M.TIMING_KIND[mid] == "wateryear"
            for lab, st in info["variants"].items():
                base = _timing_name(mid, lab)
                for part, typ in (("date", "f4"), ("reason", "i1"), ("flags", "i1")):
                    vn = f"{base}__{part}"
                    cols = []
                    for y in years:
                        r = (t.results.get(mid) or {}).get(y)
                        if y in t.years and r is not None:
                            cols.append(np.asarray(r[lab][part]).reshape(ctx.grid.shape))
                        else:
                            cols.append(old[vn][..., old_years.index(y)])
                    kw = {"fill_value": np.float32(np.nan)} if typ == "f4" else {}
                    v = d.createVariable(vn, typ, dims + ("year",), **kw)
                    v[:] = np.stack(cols, axis=-1).astype(np.float32 if typ == "f4" else np.int8)
                    v.method, v.version, v.setting = mid, np.int32(info["version"]), lab
                    v.settings = json.dumps(st)
                    if part == "date":
                        v.units = ("day of the year the water year ends in, 1 = 1 January, "
                                   "earlier days counted back (31 December = 0)" if wy else
                                   "day of the year, 1 = its first day")
                    elif part == "reason":
                        v.note = "0 = a date; the codes are the file's reasons attribute"
                    else:
                        v.note = "bits of the file's flags attribute"
                    nvar += 1
    os.replace(tmp, path)
    rec = {k: t.spec[k] for k in ("file", "not_built") + TIMING_KEYS}
    rec.update(years=years, variables=nvar, made_by="the suite's own passes of the "
               f"{t.stream} year files")
    return rec


def _with_timing(man: Dict[str, Any], rec: Dict[str, Any], out: Path) -> Dict[str, Any]:
    m = dict(man, timing=rec)
    files = {f.name: int(f.stat().st_size) for f in sorted(out.glob("*.nc"))}
    m.update(files=files, total_bytes=int(sum(files.values())))
    return m
