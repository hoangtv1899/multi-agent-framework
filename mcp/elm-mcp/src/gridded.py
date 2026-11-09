#!/usr/bin/env python3
"""Map-shaped ELM history: every cell of a whole-domain run, labelled right.

The companion of extract.py. That module reads ONE column's daily series; this
one reads a gridded run (a 144 x 96 box, a watershed, a region) and refuses a
single column, while extract.py now refuses a grid, so neither reader can
quietly misread the other's files. READS AND RETURNS: no derived fields, no
plots, nothing written beside the history. The one thing it writes anywhere is
each history file's index, to a cache outside the run (THE INDEX ON DISK).
Regions and statistics over them live in regions.py.

FOUR CONVENTIONS, each a silent error when missed. All four were measured on
the Naches ELM-BGC control run (naches_bgc_fix, 2015-2021) on 2026-10-14:

  * A RECORD IS LABELLED BY THE START OF ITS INTERVAL, time_bounds[:, 0]. The
    decoded `time` coordinate, and mcdate beside it, is the END stamp. A reader
    that trusts it shifts every daily value by one day and every monthly value
    by one month, and nothing raises.
  * A FILE DOES NOT HOLD ITS OWN YEAR. A daily or sub-daily file named for
    1 January Y opens with the last day (or step) of Y-1, and the last day of
    Y arrives as the first record of the file named for Y+1. Calendar years
    are assembled across that boundary. When the Y+1 file is absent, the year
    is reported as missing its 31 December, never quietly returned as 364 days.
  * THE FIRST RECORD OF A RUN IS NOT A DAY. Daily and sub-daily streams open
    with one model step on its own (30 minutes here, time_bounds [-1/48, 0],
    nstep 0). Monthly streams fold that step into January, which is then 31.02
    days long. The first is dropped and the second kept; both are listed on
    the stream. The step is found by its nstep of 0, not by its length, so a
    one-day run and output written every step are read right too.
  * FILL IS 1e36. Anything above 1e30 in magnitude, non-finite, or equal to the
    variable's _FillValue or missing_value is NaN; packed variables are tested
    before they are unpacked.

Files are read one at a time with netCDF4, only the variables and records
asked for, so a 7-year run never sits in memory whole. The time decoding uses
netCDF4's own num2date, so no dependency is added to the server.

    from gridded import open_stream
    h1 = open_stream("/path/to/case/run", "h1")
    block = h1.read("GPP", 2016)          # (365, 96, 144), NaN for fill
    h1.read_block_pass(["GPP", "TLAI"], 2016, reduce=f)   # every field, 30 days at a time
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

FILL_ABS = 1e30
SECONDS_PER_DAY = 86400.0

# <case>.elm.h<N>.<stamp>.nc, the stamp being YYYY-MM-DD-SSSSS, or YYYY-MM for
# ELM's default monthly files (one month per file, hist_mfilt = 1). A
# multi-instance run writes <case>.elm_0001.h0..., which is found and refused
# by name. Restart history (.rh0.) and anything else does not match.
HISTORY_NAME = re.compile(
    r"^(?P<case>.+?)\.elm(?P<inst>_\d{4})?\.(?P<stream>h\d+)\."
    r"(?P<stamp>\d{4}-\d{2}(?:-\d{2}-\d{5})?)\.nc$")

# FIELDS STORED PER SECOND WHOSE UNITS DO NOT SAY SO. FAREA_BURNED is labelled
# 'proportion' and is a fraction per second: read that way, the Naches run loses
# 966-1,176 gC of fire carbon per m2 burned, against 1,874-1,954 gC standing.
# Read per 30-minute step instead, it loses 1.7-2.1 million gC per m2 burned,
# 1,000 times what stands there (verify_model_modis.json units_check, 2026-10-15).
PER_SECOND_DESPITE_UNITS = {"FAREA_BURNED": "proportion/s"}

# WHAT A FIELD'S NAME OR LABEL GETS WRONG, for whoever reads the field next. Each
# was found by hand on the Naches run and checked against the ELM source.
FIELD_NOTES = {
    "FAREA_BURNED": "a fraction of the cell burned PER SECOND (FireMod.F90), although "
                    "its label says 'timestep fractional area burned' in 'proportion'; "
                    "a year's burned fraction is the rate times each record's seconds",
    "FIRE": "emitted infrared (longwave) radiation in W/m^2, not fire; the fire "
            "fields are FAREA_BURNED and COL_FIRE_CLOSS",
}

# A rate per second, in every spelling the Naches history uses: gC/m^2/s,
# kg m-2 s-1, mm s^-1, counts/km2/sec (NFIRE), proportion/sec (BAF_*), per sec
# (LFC2) and umol/m2s (FPSN). NOT kg/m/s^2 (TAUX, wind stress) or s-2: a second
# to any other power is not a rate to put on a per-day footing.
_PER_SECOND = re.compile(
    r"/\s*(?:s|sec|second)\b(?!\s*(?:\^|\*\*)\s*-?\s*\d)"
    r"|\bs\s*(?:\^|\*\*)?\s*-\s*1\b"
    r"|\bper\s+(?:s|sec|second)\b"
    r"|m\^?2\s*s\b", re.IGNORECASE)

# Interval widths are compared with this tolerance, in days (about 0.1 s).
_TOL_DAYS = 1e-6

# The longest model step a month may carry folded in (6 hours; ELM runs 30 or
# 60 minutes). A month longer than that is not a month plus a start step.
_MAX_STEP_DAYS = 0.25

# CF calendar names that mean the same calendar.
_CALENDAR_ALIASES = {"gregorian": "standard", "365_day": "noleap",
                     "366_day": "all_leap"}


class GriddedReadError(Exception):
    """The history cannot be read as a gridded run, and the message says why."""


def _calendar(name: Optional[str]) -> str:
    c = (name or "standard").strip().lower()
    return _CALENDAR_ALIASES.get(c, c)


# ─────────────────────────────────────────────────────────────────────────────
# CALENDARS
# ─────────────────────────────────────────────────────────────────────────────
_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def days_in_month(year: int, month: int, calendar: str) -> int:
    """Days in one month of a CF calendar. ELM writes 'noleap'."""
    cal = _calendar(calendar)
    if cal == "noleap":
        return _MONTH_DAYS[month - 1]
    if cal == "all_leap":
        return 29 if month == 2 else _MONTH_DAYS[month - 1]
    if cal == "360_day":
        return 30
    if cal in ("standard", "proleptic_gregorian", "julian"):
        if month != 2:
            return _MONTH_DAYS[month - 1]
        if cal == "julian":
            leap = year % 4 == 0
        else:
            leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    raise GriddedReadError(f"calendar {calendar!r} is not one this reader knows")


def days_in_year(year: int, calendar: str) -> int:
    return sum(days_in_month(year, m, calendar) for m in range(1, 13))


Key = Tuple[int, int, int, int, int, int]       # year, month, day, hour, minute, second


def _to_second(t):
    """A decoded time rounded to the nearest whole second.

    1/24 of a day has no exact binary form, so far from the reference date an
    hourly bound decodes to 00:59:59.999997; read by its fields that is the
    wrong hour and fails the step-grid test. Model steps are whole seconds, so
    nothing real is lost.
    """
    from datetime import timedelta
    us = int(getattr(t, "microsecond", 0) or 0)
    if not us:
        return t
    return t + timedelta(microseconds=(1_000_000 - us) if us >= 500_000 else -us)


def _key(t) -> Key:
    return (int(t.year), int(t.month), int(t.day),
            int(t.hour), int(t.minute), int(t.second))


def label_text(key: Key, cadence: str) -> str:
    """'2016-01-31' daily, '2016-01-31 21:00' sub-daily, '2016-01' monthly."""
    y, m, d, hh, mm, _ = key
    if cadence == "monthly":
        return f"{y:04d}-{m:02d}"
    if cadence == "subdaily":
        return f"{y:04d}-{m:02d}-{d:02d} {hh:02d}:{mm:02d}"
    return f"{y:04d}-{m:02d}-{d:02d}"


def expected_keys(year: int, cadence: str, calendar: str,
                  steps_per_day: Optional[int] = None) -> List[Key]:
    """Every record label a complete calendar year has, in order."""
    if cadence == "monthly":
        return [(year, m, 1, 0, 0, 0) for m in range(1, 13)]
    n = 1 if cadence == "daily" else int(steps_per_day or 0)
    if n < 1:
        raise GriddedReadError("a sub-daily stream needs its steps per day")
    step_s = 86400 // n
    out = []
    for m in range(1, 13):
        for d in range(1, days_in_month(year, m, calendar) + 1):
            for k in range(n):
                s = k * step_s
                out.append((year, m, d, s // 3600, (s % 3600) // 60, s % 60))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FILL AND UNITS
# ─────────────────────────────────────────────────────────────────────────────
def clean(x, var=None) -> np.ndarray:
    """float64 copy with fill as NaN: non-finite, |x| > 1e30, or the variable's
    own _FillValue / missing_value. For values already read and unpacked; to
    read a variable from a file, use read_clean."""
    a = np.array(x, dtype=np.float64, copy=True)
    bad = ~np.isfinite(a) | (np.abs(a) > FILL_ABS)
    if var is not None:
        for name in ("_FillValue", "missing_value"):
            if name in var.ncattrs():
                fv = np.asarray(var.getncattr(name), dtype=np.float64).ravel()
                for f in fv:
                    if np.isfinite(f):
                        bad |= a == f
    a[bad] = np.nan
    return a


def read_clean(var, index=Ellipsis) -> np.ndarray:
    """A netCDF4 variable read as float64 with every no-data value as NaN.

    PACKING IS UNDONE AFTER THE NO-DATA TEST. netCDF4 scales by default even
    with masking off, which turns a packed _FillValue of -32767 into -3.2767,
    a value no fill test recognises. So fill, missing_value and anything
    outside valid_min / valid_max / valid_range are found on the RAW numbers,
    and only then are scale_factor and add_offset applied.
    """
    var.set_auto_maskandscale(False)
    a = np.asarray(var[index]).astype(np.float64)
    attrs = var.ncattrs()
    bad = ~np.isfinite(a)
    for name in ("_FillValue", "missing_value"):
        if name in attrs:
            for f in np.asarray(var.getncattr(name), dtype=np.float64).ravel():
                if np.isfinite(f):
                    bad |= a == f
    if "valid_range" in attrs:
        lo, hi = np.asarray(var.getncattr("valid_range"), dtype=np.float64).ravel()[:2]
        bad |= (a < lo) | (a > hi)
    if "valid_min" in attrs:
        bad |= a < float(np.asarray(var.getncattr("valid_min")).ravel()[0])
    if "valid_max" in attrs:
        bad |= a > float(np.asarray(var.getncattr("valid_max")).ravel()[0])
    if "scale_factor" in attrs:
        a = a * float(np.asarray(var.getncattr("scale_factor")).ravel()[0])
    if "add_offset" in attrs:
        a = a + float(np.asarray(var.getncattr("add_offset")).ravel()[0])
    bad |= np.abs(a) > FILL_ABS
    a[bad] = np.nan
    return a


def is_per_second(var_name: str, units: str) -> bool:
    """True for a rate stored per second, from the units or the known list."""
    if var_name in PER_SECOND_DESPITE_UNITS:
        return True
    return bool(_PER_SECOND.search(units or ""))


def per_day_units(var_name: str, units: str) -> str:
    """The units of a per-second rate once multiplied by 86400 s."""
    u = PER_SECOND_DESPITE_UNITS.get(var_name, units or "")
    u = re.sub(r"/\s*(?:s|sec|second)\b(?!\s*(?:\^|\*\*)\s*-?\s*\d)", "/d", u,
               flags=re.IGNORECASE)
    u = re.sub(r"\bs(\s*(?:\^|\*\*)?\s*-\s*1)\b", r"d\1", u, flags=re.IGNORECASE)
    u = re.sub(r"\bper\s+(?:s|sec|second)\b", "per day", u, flags=re.IGNORECASE)
    u = re.sub(r"(m\^?2)\s*s\b", r"\1/d", u, flags=re.IGNORECASE)
    return u


# ─────────────────────────────────────────────────────────────────────────────
# THE GRID
# ─────────────────────────────────────────────────────────────────────────────
def to_pm180(lon) -> np.ndarray:
    """Longitudes in [-180, 180). The history writes 0-360 east."""
    a = np.asarray(lon, dtype=np.float64)
    return ((a + 180.0) % 360.0) - 180.0


@dataclass(frozen=True)
class Grid:
    """Where the cells are and what each one weighs.

    Arrays are grid-shaped: (lat, lon) for a regular grid, (lndgrid,) for an
    unstructured one. Rows of a regular grid run in the file's order (south
    to north on the Naches run). `weight_km2` is area * landfrac, zero where
    either is fill. Frozen, with read-only arrays, because every Stream that
    opens the same files shares one.
    """
    lat: np.ndarray
    lon: np.ndarray
    area_km2: np.ndarray
    landfrac: np.ndarray
    dims: Tuple[str, ...]
    lat_axis: Optional[np.ndarray] = None
    lon_axis: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.lat.shape)

    @property
    def n_cells(self) -> int:
        return int(self.lat.size)

    @property
    def regular(self) -> bool:
        return self.dims == ("lat", "lon")

    @property
    def weight_km2(self) -> np.ndarray:
        w = self.area_km2 * self.landfrac
        return np.where(np.isfinite(w), w, 0.0)

    def same_cells(self, other: "Grid", tol_deg: float = 1e-9) -> bool:
        if self.shape != other.shape:
            return False
        dlat = np.nanmax(np.abs(self.lat - other.lat))
        dlon = np.nanmax(np.abs(to_pm180(self.lon - other.lon)))
        return bool(dlat <= tol_deg and dlon <= tol_deg)

    def check_coordinates(self, lat, lon, what: str,
                          tol_deg: float = 1e-4) -> Dict[str, float]:
        """Max |difference| against another file's cell centres; raises when
        they describe different cells. Longitudes compare modulo 360."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if lat.ndim == 1 and lon.ndim == 1 and self.regular \
                and lat.size == self.shape[0] and lon.size == self.shape[1]:
            lon, lat = np.meshgrid(lon, lat)
        if lat.shape != self.shape or lon.shape != self.shape:
            raise GriddedReadError(
                f"{what}: cell centres have shape {lat.shape}, the history "
                f"grid {self.shape}")
        dlat = float(np.nanmax(np.abs(lat - self.lat)))
        dlon = float(np.nanmax(np.abs(to_pm180(lon - self.lon))))
        if dlat > tol_deg or dlon > tol_deg:
            raise GriddedReadError(
                f"{what}: cell centres differ from the history grid by "
                f"{dlat:.3g} deg in latitude and {dlon:.3g} deg in longitude "
                f"(tolerance {tol_deg} deg); they are not the same cells")
        return {"max_abs_lat_diff_deg": dlat, "max_abs_lon_diff_deg": dlon}


def _grid_from(ds) -> Grid:
    if "lat" not in ds.variables or "lon" not in ds.variables:
        raise GriddedReadError("no lat/lon variables: not an ELM history file")
    lat_v, lon_v = ds["lat"], ds["lon"]
    if "lndgrid" in lat_v.dimensions:
        lat = read_clean(lat_v)
        lon = to_pm180(read_clean(lon_v))
        dims: Tuple[str, ...] = ("lndgrid",)
        lat_axis = lon_axis = None
    elif lat_v.dimensions == ("lat",) and lon_v.dimensions == ("lon",):
        lat_axis = read_clean(lat_v)
        lon_axis = to_pm180(read_clean(lon_v))
        lon, lat = np.meshgrid(lon_axis, lat_axis)
        dims = ("lat", "lon")
    else:
        raise GriddedReadError(
            f"lat{lat_v.dimensions} / lon{lon_v.dimensions}: neither a regular "
            f"(lat, lon) grid nor an unstructured lndgrid")

    def _cellvar(name):
        if name not in ds.variables:
            raise GriddedReadError(f"no {name} variable: cells cannot be weighted")
        a = read_clean(ds[name])
        if a.shape != lat.shape:
            raise GriddedReadError(f"{name} has shape {a.shape}, the grid {lat.shape}")
        return a

    area = _cellvar("area")
    units = (getattr(ds["area"], "units", "") or "").replace(" ", "")
    if units not in ("km^2", "km2", "km**2"):
        raise GriddedReadError(
            f"area is in {units!r}; this reader weights in km^2 and will not guess")
    return Grid(lat=lat, lon=lon, area_km2=area, landfrac=_cellvar("landfrac"),
                dims=dims, lat_axis=lat_axis, lon_axis=lon_axis)


# ─────────────────────────────────────────────────────────────────────────────
# FINDING THE FILES
# ─────────────────────────────────────────────────────────────────────────────
def discover(run_dir) -> Dict[str, List[Path]]:
    """{stream: [files]} for the ELM history in one run directory.

    ONE CASE AND ONE INSTANCE PER DIRECTORY. Two cases' files side by side
    would be stitched into one record, and a multi-instance run's members
    likewise, so both are refused rather than sorted out by guesswork.
    """
    rd = Path(run_dir)
    if not rd.is_dir():
        raise GriddedReadError(f"no such run directory: {rd}")
    streams: Dict[str, List[Path]] = {}
    cases, insts = set(), set()
    for p in sorted(rd.iterdir()):
        m = HISTORY_NAME.match(p.name)
        if not m or not p.is_file():
            continue
        cases.add(m.group("case"))
        if m.group("inst"):
            insts.add("elm" + m.group("inst"))
        streams.setdefault(m.group("stream"), []).append(p)
    if len(cases) > 1:
        raise GriddedReadError(
            f"{rd} holds history from {len(cases)} cases ({sorted(cases)}); "
            f"point the reader at one case's run directory")
    if insts:
        raise GriddedReadError(
            f"{rd} holds multi-instance history ({', '.join(sorted(insts))}); "
            f"this reader takes a single-instance run")
    return {k: sorted(v) for k, v in sorted(streams.items())}


# ─────────────────────────────────────────────────────────────────────────────
# A STREAM: ONE HISTORY TAPE ACROSS ALL ITS FILES
# ─────────────────────────────────────────────────────────────────────────────
# WHAT EACH FILE HOLDS, KEPT FOR THE LIFE OF THE PROCESS. ELM writes history in
# the classic netCDF layout, where a record variable is interleaved with every
# other one: the 365 time bounds of a daily file sit about 3 MB apart, so
# reading them is 365 separate reads, about 20 ms each from a cold Lustre
# cache (55 s for the seven Naches h1 files, 150 s for h2). A second open of
# the same file in the same process costs nothing; in a new process the index
# comes from the cache on disk (THE INDEX ON DISK, below). Keyed by path,
# inode, size and the modification and change times: Lustre keeps mtime in
# whole seconds and every full-year file of a stream has the same size, so
# mtime and size alone could miss a rewrite; any rewrite moves ctime.
_FILE_INDEX: Dict[Tuple, Dict] = {}


def _file_key(path: Path) -> Tuple[str, int, int, int, int]:
    st = path.stat()
    return (str(path.resolve()), int(st.st_ino), int(st.st_size),
            int(st.st_mtime_ns), int(st.st_ctime_ns))


def _file_index(path: Path) -> Dict:
    key = _file_key(path)
    hit = _FILE_INDEX.get(key)
    if hit is not None:
        return hit
    entry = _index_entry(key[0])
    info = _index_load(entry, key) if entry is not None else None
    if info is None:
        info = _index_from_file(path)
        if entry is not None:
            _index_save(entry, key, path, info)
    # Shared by every Stream that opens this file, so nobody may write to it.
    grid = info["grid"]
    for a in (grid.lat, grid.lon, grid.area_km2, grid.landfrac,
              grid.lat_axis, grid.lon_axis, info["tb"]):
        if a is not None:
            a.flags.writeable = False
    _FILE_INDEX[key] = info
    return info


def _index_from_file(path: Path) -> Dict:
    """What _file_index keeps of one history file, read from the file."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        grid = _grid_from(ds)
        if "time" not in ds.variables:
            raise GriddedReadError(f"{path.name}: no time variable")
        tv = ds["time"]
        bname = getattr(tv, "bounds", None) or "time_bounds"
        if bname not in ds.variables:
            raise GriddedReadError(
                f"{path.name}: no {bname}; without interval bounds a record "
                f"cannot be labelled by its start")
        bv = ds[bname]
        bv.set_auto_maskandscale(False)
        tb = np.asarray(bv[:], dtype=np.float64)
        if tb.ndim != 2 or tb.shape[1] != 2:
            raise GriddedReadError(f"{path.name}: {bname} is not (time, 2)")
        # The run's start step carries nstep 0. Only the first record of a
        # file can be it, so one value per file is read, not the whole column.
        first_nstep = None
        if "nstep" in ds.variables and tb.shape[0] > 0:
            nv = ds["nstep"]
            nv.set_auto_maskandscale(False)
            first_nstep = int(np.asarray(nv[0]).ravel()[0])
        variables, metadata = {}, []
        for vname, v in ds.variables.items():
            if not v.dimensions or v.dimensions[0] != "time" \
                    or vname in ("time", bname):
                continue
            on_grid = len(v.dimensions) > len(grid.dims) and \
                tuple(v.dimensions[-len(grid.dims):]) == grid.dims
            if not on_grid:
                metadata.append(vname)          # nstep, mcdate, date_written ...
                continue
            variables[vname] = {
                "dims": tuple(v.dimensions), "shape": tuple(v.shape[1:]),
                "units": str(getattr(v, "units", "") or ""),
                "cell_methods": str(getattr(v, "cell_methods", "") or "")}
        return {"grid": grid, "tb": tb, "time_units": str(tv.units),
                "calendar": _calendar(getattr(tv, "calendar", "standard")),
                "first_nstep": first_nstep, "variables": variables,
                "metadata": sorted(metadata)}


# THE INDEX ON DISK (2026-11-04). _FILE_INDEX lasts as long as the process,
# and every analysis is a new process that opens every stream its run has
# (domain_analysis.run_values: h0, h1 and h2), so each one paid the
# per-record reads again before it read a single value: 40.9 s and 57.1 s to
# open the Naches control's h1 cold (compy01, 2026-11-04). So the index of
# each file, everything _index_from_file reads (the grid, the time bounds,
# the first nstep, the calendar and time units, and each variable's
# dimensions, shape, units and cell_methods), is kept on disk as well, one
# entry per file, under the key _FILE_INDEX uses: a file changed in any way
# is read again.
#
#   where    $ELM_STREAM_INDEX_CACHE (an absolute path, or ~/...), else
#            ~/.cache/elm-mcp/stream_index/; ELM_STREAM_INDEX_CACHE=off
#            switches it off (the tests run with it off, tests/conftest.py)
#   never    in a run or case directory: a cache directory at or under the
#            history file's own directory is not used at all, nor a relative
#            one, which would land wherever a process happens to run
#   whole    an entry is written to a temporary file and renamed into place,
#            so two processes cannot leave a torn one
#   trusted  only when it loads, carries this format and the file's own key,
#            and holds what an index holds, in the types and shapes it holds
#            them; anything else is ignored and the entry made again from the
#            file
#
# Measured on compy01 (login node), 2026-11-04, the Naches control's h1
# (10 files): see open_stream. The numbers check
# (src/agents/analysis/numbers_check.py) reads each run's history with its
# own code and does not import this module, so it never reads the cache.
INDEX_CACHE_ENV = "ELM_STREAM_INDEX_CACHE"
_INDEX_CACHE_OFF = ("off", "0", "false", "no", "none")
# Raise it whenever the making of an index changes (_index_from_file and what
# it calls, or the entry's encoding): an entry of an older format is then read
# again from its file. tests/test_gridded_index_cache.py fails until it is.
_INDEX_FORMAT = 1


def index_cache_dir() -> Optional[Path]:
    """The directory of the index cache, or None when it is switched off.

    A relative path is not used: where it landed would depend on the
    directory each process runs in, a case directory among them."""
    v = (os.environ.get(INDEX_CACHE_ENV) or "").strip()
    if v.lower() in _INDEX_CACHE_OFF:
        return None
    if v:
        p = Path(v).expanduser()
        return p if p.is_absolute() else None
    try:
        return Path.home() / ".cache" / "elm-mcp" / "stream_index"
    except (KeyError, RuntimeError):                # no home directory to find
        return None


def _index_entry(real: str) -> Optional[Path]:
    """Where the index of the history file at `real` (its resolved path) is
    kept, or None: the cache is off, or would sit in the file's directory."""
    d = index_cache_dir()
    if d is None:
        return None
    try:
        d = d.resolve()
    except (OSError, RuntimeError):
        return None
    run = Path(real).parent
    if d == run or run in d.parents:
        return None
    name = hashlib.sha256(real.encode("utf-8", "surrogateescape")).hexdigest()[:40]
    return d / f"{name}.npz"


def _index_save(entry: Path, key: Tuple, path: Path, info: Dict) -> None:
    """One file's index into the cache, whole or not at all. Never raises: a
    cache that cannot be written leaves reading as it was. A file that changed
    while it was being read is not kept."""
    import tempfile
    tmp = None
    try:
        if _file_key(path) != key:
            return
        g = info["grid"]
        meta = {"format": _INDEX_FORMAT, "key": list(key), "dims": list(g.dims),
                "time_units": info["time_units"], "calendar": info["calendar"],
                "first_nstep": info["first_nstep"],
                "variables": {n: {"dims": list(d["dims"]), "shape": list(d["shape"]),
                                  "units": d["units"], "cell_methods": d["cell_methods"]}
                              for n, d in info["variables"].items()},
                "metadata": list(info["metadata"])}
        arrays = {"tb": info["tb"], "lat": g.lat, "lon": g.lon, "area_km2": g.area_km2,
                  "landfrac": g.landfrac}
        if g.lat_axis is not None:
            arrays.update(lat_axis=g.lat_axis, lon_axis=g.lon_axis)
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        entry.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=entry.parent, prefix=".tmp-", suffix=".npz")
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, entry)
        tmp = None
    except Exception:                                               # noqa: BLE001
        pass
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _index_load(entry: Path, key: Tuple) -> Optional[Dict]:
    """One file's index from the cache, or None when there is none to trust."""
    try:
        with np.load(entry, allow_pickle=False) as z:
            meta = json.loads(z["meta"].tobytes().decode("utf-8"))
            if not isinstance(meta, dict) or meta.get("format") != _INDEX_FORMAT \
                    or meta.get("key") != list(key):
                return None
            arrays = {n: z[n] for n in z.files if n != "meta"}
        return _index_checked(meta, arrays)
    except Exception:                                               # noqa: BLE001
        return None


def _index_checked(meta: Dict, arrays: Dict[str, np.ndarray]) -> Optional[Dict]:
    """The index a cache entry holds, rebuilt with the types _index_from_file
    gives, or None when any part of it is not what that function makes."""
    dims = tuple(meta.get("dims") or ())
    regular = dims == ("lat", "lon")
    if not regular and dims != ("lndgrid",):
        return None
    want = {"tb", "lat", "lon", "area_km2", "landfrac"}
    want |= {"lat_axis", "lon_axis"} if regular else set()
    if set(arrays) != want or any(a.dtype != np.float64 for a in arrays.values()):
        return None
    tb, lat = arrays["tb"], arrays["lat"]
    if tb.ndim != 2 or tb.shape[1] != 2 or lat.ndim != len(dims) or any(
            arrays[n].shape != lat.shape for n in ("lon", "area_km2", "landfrac")):
        return None
    if regular and (arrays["lat_axis"].shape != (lat.shape[0],)
                    or arrays["lon_axis"].shape != (lat.shape[1],)):
        return None
    nstep, md, vs = meta.get("first_nstep"), meta.get("metadata"), meta.get("variables")
    if not (nstep is None or type(nstep) is int) \
            or not isinstance(meta.get("time_units"), str) \
            or not isinstance(meta.get("calendar"), str) \
            or not (isinstance(md, list) and all(isinstance(x, str) for x in md)) \
            or not isinstance(vs, dict):
        return None
    variables = {}
    for n, d in vs.items():
        if not (isinstance(d, dict) and set(d) == {"dims", "shape", "units", "cell_methods"}
                and isinstance(d["dims"], list) and all(isinstance(x, str) for x in d["dims"])
                and isinstance(d["shape"], list) and all(type(x) is int for x in d["shape"])
                and isinstance(d["units"], str) and isinstance(d["cell_methods"], str)):
            return None
        variables[n] = {"dims": tuple(d["dims"]), "shape": tuple(d["shape"]),
                        "units": d["units"], "cell_methods": d["cell_methods"]}
    grid = Grid(lat=lat, lon=arrays["lon"], area_km2=arrays["area_km2"],
                landfrac=arrays["landfrac"], dims=dims, lat_axis=arrays.get("lat_axis"),
                lon_axis=arrays.get("lon_axis"))
    return {"grid": grid, "tb": tb, "time_units": meta["time_units"],
            "calendar": meta["calendar"], "first_nstep": nstep, "variables": variables,
            "metadata": list(md)}


# ─────────────────────────────────────────────────────────────────────────────
# READING AHEAD OF A PASS
# ─────────────────────────────────────────────────────────────────────────────
# READ-AHEAD (2026-11-04). A pass over a year (Stream.read_block_pass) waits on
# small cold reads, one per field and record: about 15,700 of them for the 43
# daily fields of a Naches year, 28.8 s against 12.3 s to read the file
# straight through. A thread that reads the same records straight through in
# large chunks while the pass runs puts them in the node's page cache before
# the pass asks for them, and the pass's small reads are then served from
# memory. It is a hint and nothing more: it reads through its own file handle
# into a scratch buffer and throws the bytes away (netCDF4 is not called from
# two threads), so every value the pass hands over is read exactly as without
# it; an error in it is recorded, never raised; and it is told to stop and
# joined when the pass returns or raises, so it never outlives the pass.
# Measured: see Stream.read_block_pass. Chunks of 16 MB, as measured there:
# large enough to stream, and stopping waits for at most one of them.
_READ_AHEAD_CHUNK = 16 * 2 ** 20


def record_span(ds, path, first: int, last: int) -> Optional[Tuple[int, int]]:
    """The bytes [start, end) that records first..last of an open classic
    netCDF file occupy, or None for any other layout.

    A classic file (CDF-1, 2 or 5) ends with its records, each holding every
    record variable's values at that record, each variable's share padded to
    4 bytes (none when there is one record variable). So a record's size
    follows from the variables' shapes, and the records begin that many
    records before the end of the file. A netCDF-4 file keeps its chunks
    elsewhere and has no span. A file still being written may end short of
    its last record, which shifts the span by what is missing: for a hint
    that costs time, never a value."""
    if not str(getattr(ds, "data_model", "")).startswith("NETCDF3"):
        return None
    unlimited = [n for n, d in ds.dimensions.items() if d.isunlimited()]
    if len(unlimited) != 1:
        return None
    rec = unlimited[0]
    sizes = [int(np.prod(v.shape[1:], dtype=np.int64)) * v.dtype.itemsize
             for v in ds.variables.values() if v.dimensions and v.dimensions[0] == rec]
    nrec = len(ds.dimensions[rec])
    if not sizes or not 0 <= first <= last < nrec:
        return None
    recsize = sizes[0] if len(sizes) == 1 else sum(-(-n // 4) * 4 for n in sizes)
    begin = os.path.getsize(path) - nrec * recsize
    if recsize <= 0 or begin < 0:
        return None
    return begin + first * recsize, begin + (last + 1) * recsize


class _ReadAhead:
    """The read-ahead of one pass: each (path, start, end) in turn, read
    straight through in a daemon thread; stop() ends it and waits for it."""

    def __init__(self, spans: Sequence[Tuple[str, int, int]], chunk: Optional[int] = None):
        self.spans = list(spans)
        self.chunk = int(chunk or _READ_AHEAD_CHUNK)
        self.bytes_read = 0
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gridded-read-ahead",
                                        daemon=True)

    def _run(self) -> None:
        try:
            buf = memoryview(bytearray(self.chunk))
            for path, start, end in self.spans:
                if self._stop.is_set():
                    return
                with open(path, "rb", buffering=0) as f:
                    f.seek(start)
                    pos = start
                    while pos < end and not self._stop.is_set():
                        n = self._read(f, buf[:min(self.chunk, end - pos)])
                        if not n:
                            break
                        pos += n
                        self.bytes_read += n
        except Exception as e:                                      # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"

    @staticmethod
    def _read(f, view) -> int:
        return f.readinto(view)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()


@dataclass
class Record:
    key: Key
    file: int          # index into Stream.files
    index: int         # record index inside that file
    start: Any = None  # decoded interval start and end
    end: Any = None


@dataclass
class Block:
    """One variable's records for one calendar year (or a part of one)."""
    var: str
    year: int
    keys: List[Key]
    labels: List[str]
    data: np.ndarray                 # (records, [levels,] *grid or window)
    units: str
    per_second: bool
    missing: List[str] = field(default_factory=list)


@dataclass
class PassBlock:
    """One block of a one-pass read (Stream.read_block_pass): consecutive
    records of one calendar year, with every field the pass was asked for,
    each exactly as read() gives it at these records."""
    year: int
    first: int                       # the block's first record, counted in the year's records
    keys: List[Key]
    labels: List[str]
    data: Dict[str, np.ndarray]      # field -> (records, [levels,] *grid or window)
    units: Dict[str, str]
    per_second: Dict[str, bool]
    missing: List[str] = field(default_factory=list)   # the year's absent labels, first block

    def block(self, var: str) -> Block:
        """One field of this block, in the shape iter_read gives its pieces."""
        return Block(var=var, year=self.year, keys=self.keys, labels=self.labels,
                     data=self.data[var], units=self.units[var],
                     per_second=self.per_second[var], missing=self.missing)


class Stream:
    """One history tape (h0, h1, h2, ...) of one gridded run.

    Opening it reads only the time bounds of each file (and one nstep) and the
    grid, or for a file indexed before, its index from the cache on disk (THE
    INDEX ON DISK); variables are read on request, one calendar year at a time.

    Attributes worth reading before trusting a number:
        cadence         'monthly', 'daily' or 'subdaily'
        steps_per_day   records per day for a fixed-step stream
        dropped         records left out, each with the reason
        extended        monthly records kept although they start before the
                        1st (the run's first step folded into January)
    A label that appears twice, or records that overlap, raise: overlapping
    files are not merged.
    """

    def __init__(self, name: str, files: Sequence[Path]):
        import netCDF4
        if not files:
            raise GriddedReadError(f"stream {name}: no files")
        self.name = name
        self.files = [Path(f) for f in files]
        self.dropped: List[Dict] = []
        self.extended: List[Dict] = []
        self.calendar: Optional[str] = None
        self._vars: Dict[str, Dict] = {}
        self._meta: List[str] = []
        raw = []                 # (file, record, start, end, width, is_start_step)
        for fi, path in enumerate(self.files):
            info = _file_index(path)
            g = info["grid"]
            if fi == 0:
                self.grid = g
                if g.n_cells == 1:
                    raise GriddedReadError(
                        f"{path.name} is a single-cell history: read it with "
                        f"extract.extract_column, not as a grid")
                self._vars = dict(info["variables"])
                self._meta = list(info["metadata"])
            elif not g.same_cells(self.grid):
                raise GriddedReadError(
                    f"{path.name} is on a different grid from {self.files[0].name}")
            cal = info["calendar"]
            if self.calendar is None:
                self.calendar = cal
            elif cal != self.calendar:
                raise GriddedReadError(
                    f"{path.name} uses calendar {cal}, {self.files[0].name} "
                    f"{self.calendar}")
            tb = info["tb"]
            if tb.shape[0] == 0:
                continue
            ok = np.all(np.isfinite(tb), axis=1) & np.all(np.abs(tb) < FILL_ABS, axis=1)
            for k in np.flatnonzero(~ok):
                self.dropped.append({"file": path.name, "record": int(k), "start": None,
                                     "length_days": None,
                                     "reason": "time bounds not written (fill); the "
                                               "file may still have been in progress"})
            idx = np.flatnonzero(ok)
            if not idx.size:
                continue
            starts = netCDF4.num2date(tb[idx, 0], info["time_units"], calendar=cal)
            ends = netCDF4.num2date(tb[idx, 1], info["time_units"], calendar=cal)
            for j, k in enumerate(idx):
                s, e = _to_second(starts[j]), _to_second(ends[j])
                w = (e - s).total_seconds() / SECONDS_PER_DAY
                raw.append((fi, int(k), s, e, w, k == 0 and info["first_nstep"] == 0))
        if not raw:
            raise GriddedReadError(f"stream {name}: the files hold no records")
        self._classify(raw)

    # ── cadence and labels ───────────────────────────────────────────────────
    def _drop(self, fi, k, s, w, why) -> None:
        self.dropped.append({"file": self.files[fi].name, "record": k,
                             "start": str(s), "length_days": round(w, 6),
                             "reason": why})

    def _classify(self, raw) -> None:
        body = []
        for r in raw:
            if r[5]:
                self._drop(r[0], r[1], r[2], r[4], "the run's start step (nstep 0)")
            else:
                body.append(r)
        if not body:
            raise GriddedReadError(
                f"stream {self.name}: its only record is the run's start step")
        raw_w = np.array([r[4] for r in body])
        widths = np.round(raw_w, 6)
        vals, counts = np.unique(widths, return_counts=True)
        # The most common width, taken at full precision from its own records:
        # rounded to six decimals, a 30-minute step (1/48 day) reads as 48.0008
        # steps per day and fails the whole-fraction test below.
        modal = float(raw_w[widths == vals[np.argmax(counts)]].mean())
        records: List[Record] = []
        extra_of: Dict[Tuple[int, int], float] = {}     # extended record -> extra days
        # MONTHLY when at least half the records are a month long (28-31 days,
        # plus one step at the start of a run). Asking only for the single most
        # common width would call a short run with one partial month anything.
        month_like = float(np.mean((raw_w >= 27.5) & (raw_w < 32.5)))
        if month_like >= 0.5:
            self.cadence, self.steps_per_day = "monthly", None
            for fi, k, s, e, w, _ in body:
                why = None
                if (e.day, e.hour, e.minute, e.second) != (1, 0, 0, 0):
                    why = "does not end at the start of a month"
                else:
                    y, m = (e.year, e.month - 1) if e.month > 1 else (e.year - 1, 12)
                    nominal = days_in_month(y, m, self.calendar)
                    if w < nominal - _TOL_DAYS:
                        why = f"covers {w:.4f} of the month's {nominal} days"
                    elif w - nominal > _MAX_STEP_DAYS:
                        why = (f"is {w:.4f} days long, more than a month plus one "
                               f"model step")
                    elif w - nominal > _TOL_DAYS:
                        # A run's first model step folded into its first month:
                        # January of the run, or the first month of a later
                        # segment started from a restart (MidA_fdrought, 2019).
                        self.extended.append({"file": self.files[fi].name, "record": k,
                                              "month": f"{y:04d}-{m:02d}",
                                              "extra_days": round(w - nominal, 6)})
                        extra_of[(fi, k)] = w - nominal
                if why:
                    self._drop(fi, k, s, w, why)
                    continue
                records.append(Record((y, m, 1, 0, 0, 0), fi, k, s, e))
        elif 0.0 < modal <= 1.0 + _TOL_DAYS and \
                abs(modal * round(1.0 / modal) - 1.0) < 1e-5:
            n = int(round(1.0 / modal))
            self.steps_per_day = n
            self.cadence = "daily" if n == 1 else "subdaily"
            step_s = 86400 // n
            for fi, k, s, e, w, _ in body:
                why = None
                if abs(w - modal) > _TOL_DAYS:
                    why = (f"interval of {w:.6f} days, not the stream's "
                           f"{modal:.6f}")
                elif (s.hour * 3600 + s.minute * 60 + s.second) % step_s != 0:
                    why = "does not start on the stream's step grid"
                if why:
                    self._drop(fi, k, s, w, why)
                    continue
                records.append(Record(_key(s), fi, k, s, e))
        else:
            raise GriddedReadError(
                f"stream {self.name}: records are {modal} days long, neither "
                f"monthly nor a whole fraction of a day; not a cadence this reader "
                f"handles")
        if not records:
            reasons: Dict[str, int] = {}
            for d in self.dropped:
                reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
            raise GriddedReadError(
                f"stream {self.name}: all {len(self.dropped)} records were dropped: "
                + "; ".join(f"{n} {r}" for r, n in reasons.items()))
        records.sort(key=lambda r: r.key)
        prev = None
        for r in records:
            if prev is not None and r.key == prev.key:
                raise GriddedReadError(
                    f"stream {self.name}: {label_text(r.key, self.cadence)} appears "
                    f"twice, in {self.files[prev.file].name} (record {prev.index}) and "
                    f"{self.files[r.file].name} (record {r.index}); overlapping "
                    f"files are not merged by guesswork")
            if prev is not None:
                ov = (prev.end - r.start).total_seconds() / SECONDS_PER_DAY
                extra = extra_of.get((r.file, r.index))
                if ov > _TOL_DAYS and extra is not None and ov <= extra + _TOL_DAYS:
                    # A later run segment's start step, folded into its first
                    # month, overlaps the month before by that one step.
                    for e in self.extended:
                        if (e["file"], e["record"]) == (self.files[r.file].name, r.index):
                            e["overlaps_previous_days"] = round(ov, 6)
                elif ov > _TOL_DAYS:
                    # Defensive: the rules above leave no way for two kept
                    # records with different labels to overlap otherwise.
                    raise GriddedReadError(
                        f"stream {self.name}: {label_text(r.key, self.cadence)} "
                        f"({self.files[r.file].name}, record {r.index}) starts before "
                        f"{label_text(prev.key, self.cadence)} ends; the records overlap")
            prev = r
        self.records = records

    # ── what is there ────────────────────────────────────────────────────────
    @property
    def variables(self) -> List[str]:
        """Fields on the grid. Per-record values (nstep, mcdate, ...) are not
        listed; they are in metadata_variables."""
        return sorted(self._vars)

    @property
    def metadata_variables(self) -> List[str]:
        return list(self._meta)

    def lacking(self, names: Sequence[str],
                records: Optional[Sequence[Record]] = None) -> Dict[str, List[str]]:
        """{name: files without it}, over the files that hold these records
        (every file when none are given). `variables` lists the first file's
        fields only; a later segment's files can carry other ones."""
        files = sorted({r.file for r in records}) if records else range(len(self.files))
        out: Dict[str, List[str]] = {}
        for fi in files:
            have = _file_index(self.files[fi])["variables"]
            for n in names:
                if n not in have:
                    out.setdefault(n, []).append(self.files[fi].name)
        return out

    def units(self, var: str) -> str:
        self._need(var)
        return self._vars[var]["units"]

    def per_second(self, var: str) -> bool:
        return is_per_second(var, self.units(var))

    def levels(self, var: str) -> Tuple[str, ...]:
        """Dimensions between time and the grid, e.g. ('levgrnd',)."""
        self._need(var)
        dims = self._vars[var]["dims"]
        return tuple(dims[1:len(dims) - len(self.grid.dims)])

    def years(self) -> List[int]:
        return sorted({r.key[0] for r in self.records})

    def year_records(self, year: int) -> Tuple[List[Record], List[Key]]:
        """(records present, labels missing) for one calendar year."""
        have = [r for r in self.records if r.key[0] == year]
        want = expected_keys(year, self.cadence, self.calendar, self.steps_per_day)
        got = {r.key for r in have}
        wanted = set(want)
        extra = [r for r in have if r.key not in wanted]
        if extra:
            raise GriddedReadError(
                f"stream {self.name}: {label_text(extra[0].key, self.cadence)} is "
                f"not a label the {self.calendar} calendar has at this cadence")
        return have, [k for k in want if k not in got]

    def complete_years(self) -> List[int]:
        return [y for y in self.years() if not self.year_records(y)[1]]

    def labels(self, year: Optional[int] = None) -> List[str]:
        rs = self.records if year is None else self.year_records(year)[0]
        return [label_text(r.key, self.cadence) for r in rs]

    def _need(self, var: str) -> None:
        if var in self._vars:
            return
        if var in self._meta:
            raise GriddedReadError(
                f"{var} on stream {self.name} is a per-record value, not a field "
                f"on the grid; this reader reads grid fields")
        raise GriddedReadError(
            f"{var} is not on stream {self.name}; it carries "
            f"{len(self._vars)} grid variables (see Stream.variables)")

    # ── reading ──────────────────────────────────────────────────────────────
    def _missing_message(self, year: int, missing: List[Key]) -> str:
        want = expected_keys(year, self.cadence, self.calendar, self.steps_per_day)
        first = label_text(missing[0], self.cadence)
        msg = (f"stream {self.name}, {year}: {len(missing)} of {len(want)} records "
               f"missing (first {first})")
        if self.cadence != "monthly" and missing[-1] == want[-1]:
            msg += (f"; the year's last record is written as the first record of "
                    f"the next year's file, so that file is absent or short")
        return msg

    def _layer(self, var: str, layer: Optional[int]) -> Optional[int]:
        lev = self.levels(var)
        if layer is None:
            return None
        if not lev:
            raise GriddedReadError(f"{var} has no levels to pick layer {layer} from")
        n = int(self._vars[var]["shape"][0])
        k = int(layer) + (n if int(layer) < 0 else 0)
        if not 0 <= k < n:
            raise GriddedReadError(f"{var} has {n} levels ({lev[0]}); layer {layer} "
                                   f"is not one of them")
        return k

    def read_records(self, var: str, records: Sequence[Record],
                     window: Optional[Tuple[int, int, int, int]] = None,
                     layer: Optional[int] = None) -> np.ndarray:
        """The variable at these records, in the order given, fill as NaN.

        `window` = (row0, row1, col0, col1) reads part of a regular grid.
        `layer` picks one level of a layered variable (negative counts from
        the bottom); without it the levels are kept as the second axis.

        AN INSTANTANEOUS FIELD IS REFUSED on a daily or sub-daily stream. Its
        value is a snapshot at the END of each interval (cell_methods 'time:
        point'), and every record here is labelled by its START, so it would
        sit one step early. On a monthly stream the month label describes it.
        """
        import netCDF4
        k_layer, sl = self._index(var, window, layer)
        out_parts, order = [], []
        by_file = self._by_file(records)
        for fi in sorted(by_file):
            items = by_file[fi]
            with netCDF4.Dataset(self.files[fi]) as ds:
                out_parts.append(self._read_part(ds, fi, var, items, k_layer, sl))
            order += [p for p, _ in items]
        return self._assemble(out_parts, order)

    # read_records in three parts (2026-11-04), so read_block_pass refuses what
    # it refuses and reads every value with the same code: the checks and the
    # index of a record's values, one open file's share of the records, and
    # the parts put back in the order the records were asked for.
    def _index(self, var: str, window: Optional[Tuple[int, int, int, int]],
               layer: Optional[int]) -> Tuple[Optional[int], List]:
        """What read_records checks before reading `var`, and the index of one
        record's values: (the layer's position or None, the slices after the
        record axis)."""
        self._need(var)
        lev = self.levels(var)
        k_layer = self._layer(var, layer)
        if self.cadence != "monthly" and \
                "time: point" in self._vars[var]["cell_methods"]:
            raise GriddedReadError(
                f"{var} is an instantaneous field (cell_methods 'time: point'): its "
                f"value is a snapshot at the end of each interval, and this reader "
                f"labels records by their start")
        if window is not None and not self.grid.regular:
            raise GriddedReadError("a window applies to a regular (lat, lon) grid only")
        sl: List = []
        if lev:
            sl += [slice(k_layer, k_layer + 1) if k_layer is not None else slice(None)]
            sl += [slice(None)] * (len(lev) - 1)
        if window is not None:
            r0, r1, c0, c1 = window
            sl += [slice(r0, r1), slice(c0, c1)]
        else:
            sl += [slice(None)] * len(self.grid.dims)
        return k_layer, sl

    @staticmethod
    def _by_file(records: Sequence[Record]) -> Dict[int, List[Tuple[int, int]]]:
        """{file: [(position among the records, record index in the file)]}."""
        by_file: Dict[int, List[Tuple[int, int]]] = {}
        for pos, r in enumerate(records):
            by_file.setdefault(r.file, []).append((pos, r.index))
        return by_file

    def _read_part(self, ds, fi: int, var: str, items: List[Tuple[int, int]],
                   k_layer: Optional[int], sl: List) -> np.ndarray:
        """`var` at one open file's share of the records, in the order of
        items, fill as NaN."""
        idx = np.array([i for _, i in items])
        lo, hi = int(idx.min()), int(idx.max()) + 1
        if var not in ds.variables:
            raise GriddedReadError(
                f"{var} is in {self.files[0].name} but missing from "
                f"{self.files[fi].name}")
        a = read_clean(ds[var], tuple([slice(lo, hi)] + sl))
        a = a[idx - lo]
        if k_layer is not None:
            a = a[:, 0]
        return a

    @staticmethod
    def _assemble(parts: List[np.ndarray], order: List[int]) -> np.ndarray:
        """The files' parts as one array, in the order the records were asked for."""
        data = np.concatenate(parts, axis=0)
        inv = np.empty(len(order), dtype=int)
        inv[np.array(order)] = np.arange(len(order))
        return data[inv]

    def read(self, var: str, year: int,
             window: Optional[Tuple[int, int, int, int]] = None,
             layer: Optional[int] = None,
             require_complete: bool = True) -> Block:
        """One calendar year of one variable, assembled across files.

        A year with a record missing raises, naming the first absent label and,
        when it is 31 December, the file that should have carried it. Pass
        require_complete=False to take what is there; Block.missing lists the
        rest, so a short year is never silent.
        """
        have, missing = self.year_records(year)
        if not have:
            raise GriddedReadError(f"stream {self.name}: no records in {year}")
        if missing and require_complete:
            raise GriddedReadError(self._missing_message(year, missing))
        data = self.read_records(var, have, window=window, layer=layer)
        return Block(var=var, year=year, keys=[r.key for r in have],
                     labels=[label_text(r.key, self.cadence) for r in have],
                     data=data, units=self.units(var), per_second=self.per_second(var),
                     missing=[label_text(k, self.cadence) for k in missing])

    def iter_read(self, var: str, year: int, max_records: int,
                  window: Optional[Tuple[int, int, int, int]] = None,
                  layer: Optional[int] = None,
                  require_complete: bool = True) -> Iterator[Block]:
        """read() in consecutive pieces of at most max_records, for sub-daily
        streams whose year would not fit in memory at once. Refuses what
        read() refuses, before yielding anything."""
        n = int(max_records)
        if n < 1:
            raise GriddedReadError(f"max_records must be at least 1, not {max_records}")
        have, missing = self.year_records(year)
        if not have:
            raise GriddedReadError(f"stream {self.name}: no records in {year}")
        if missing and require_complete:
            raise GriddedReadError(self._missing_message(year, missing))
        self._need(var)
        for a in range(0, len(have), n):
            part = have[a:a + n]
            yield Block(var=var, year=year, keys=[r.key for r in part],
                        labels=[label_text(r.key, self.cadence) for r in part],
                        data=self.read_records(var, part, window=window, layer=layer),
                        units=self.units(var), per_second=self.per_second(var),
                        missing=[label_text(k, self.cadence) for k in missing]
                        if a == 0 else [])

    # ONE PASS OVER A YEAR (2026-11-04). read() takes one field over a whole
    # year, and in ELM's classic netCDF every record holds all fields of its
    # day side by side, so one field's year is spread over the whole file:
    # the 43 daily fields of the Naches control read one at a time took 89.3 s
    # for a year file that reads straight through in 12.3 s (compy01,
    # 2026-11-03/04). Here the records are walked in blocks and every field
    # asked for is taken from a block before the next, so the reads stay
    # within one block of records at a time (about 100 MB of a daily Naches
    # file) instead of crossing the whole file once per field. Within a block
    # the order does not matter: a cold block of 30 records cost 1.30 s field
    # after field and 1.24 s record after record (2019, compy01, 2026-11-04),
    # so each block's fields are read by read_records' own code, each file
    # opened once per pass, and every value is the one read() gives.
    def read_block_pass(self, vars: Sequence[str], year: int, block_records: int = 30,
                        reduce: Optional[Callable[[PassBlock], Any]] = None,
                        window: Optional[Tuple[int, int, int, int]] = None,
                        layer: Optional[int] = None,
                        require_complete: bool = True,
                        read_ahead: Optional[bool] = None) -> Dict[str, Any]:
        """Every field in `vars` over one calendar year, in one pass: the
        year's records in blocks of at most block_records, every field read
        from a block before the next, each block handed to reduce(PassBlock).
        Nothing is kept: what the callback does not keep is gone when it
        returns.

        The values are read()'s, record for record: the same records (the
        year's labels assembled across files, the run's start step left out),
        float64 with fill as NaN, the same window and layer (one layer for
        every field), the same units and per-second flag, and the same
        refusals (a short year unless require_complete=False, a field the
        stream lacks, holds per record, or holds as an instantaneous value, a
        layer of a field without levels, a field a later file lacks), all made
        before the first block is handed over. PassBlock.missing lists the
        year's absent labels on the first block, as iter_read does.

        read_ahead: a thread reads the year's records straight through while
        the pass runs (READ-AHEAD, above), so the pass's small reads find them
        in memory. None takes read_ahead_default(); True or False forces it.
        The values are the same either way, bit for bit.

        Measured on compy01 (login node), 2026-11-04, on the Naches control
        (h1 daily, 1.17 GB a year; h0 monthly, 1.03 GB a year, 12 records),
        blocks of 30, the pages of the files a year's records live in dropped
        from the node's cache first and checked empty, one measurement each:

            43 daily fields    read ahead 10.6 s (2016)   without 18.3 s (2015),
                                                          and 28.8 s (2022) earlier that day
            1 daily field      read ahead  4.5 s (2018)   without 11.5 s (2017)
            5 monthly fields   read ahead  2.2 s (2016)   without  1.4 s (2017)

        against 12.3 s to read a daily year file straight through (2020) and
        89.3 s to read its 43 fields one at a time (2019), both measured on
        2026-11-03/04. Without the thread the waiting is
        on the cold reads, one per field and record; the same pass over a warm
        synthetic copy of the layout takes 4.1 s. Hence read_ahead_default: a
        daily year is spread over 365 records, and reading them straight
        through wins even for one field, although the thread reads every byte
        of each record (58 times what one field needs). A monthly year is 12
        records: the pass's few reads were done when the thread had read 0.2
        of its 1.0 GB, and the thread only competed with them. Sub-daily
        streams follow the daily rule (the same layout, more and smaller
        records); they were not timed.

        Returns {"year", "records", "blocks", "missing", "read_ahead"}: counts
        and labels, no values. read_ahead says whether the thread ran ("on"),
        the bytes it was to read and did read, and the error that stopped it
        or kept it from starting, if any.
        """
        import netCDF4
        from contextlib import ExitStack
        names = [vars] if isinstance(vars, str) else [str(v) for v in vars]
        if not names:
            raise GriddedReadError("read_block_pass needs at least one field")
        if len(set(names)) != len(names):
            raise GriddedReadError(f"fields repeat: {names}")
        n = int(block_records)
        if n < 1:
            raise GriddedReadError(f"block_records must be at least 1, not {block_records}")
        if not callable(reduce):
            raise GriddedReadError("read_block_pass hands each block to a reduce callback "
                                   "and keeps nothing itself; none was given")
        have, missing = self.year_records(year)
        if not have:
            raise GriddedReadError(f"stream {self.name}: no records in {year}")
        if missing and require_complete:
            raise GriddedReadError(self._missing_message(year, missing))
        index = {v: self._index(v, window, layer) for v in names}
        units = {v: self.units(v) for v in names}
        per_s = {v: self.per_second(v) for v in names}
        lost = [label_text(k, self.cadence) for k in missing]
        files = sorted({r.file for r in have})
        blocks = 0
        with ExitStack() as stack:
            ds = {fi: stack.enter_context(netCDF4.Dataset(self.files[fi])) for fi in files}
            in_file_order = {}
            for fi in files:
                for v in names:
                    if v not in ds[fi].variables:
                        raise GriddedReadError(
                            f"{v} is in {self.files[0].name} but missing from "
                            f"{self.files[fi].name}")
                # the fields in the file's own order, so the reads of a block
                # move forward through its records
                at = {nm: k for k, nm in enumerate(ds[fi].variables)}
                in_file_order[fi] = sorted(names, key=lambda v: at[v])
            # Started only now, once nothing is left to refuse; stopped and
            # joined by the ExitStack as the pass returns or raises.
            ahead = {"on": False, "bytes_planned": 0, "bytes_read": 0, "error": None}
            ra = None
            if self.read_ahead_default() if read_ahead is None else bool(read_ahead):
                ra = self._start_read_ahead(have, ds, ahead)
                if ra is not None:
                    stack.callback(ra.stop)
            for a in range(0, len(have), n):
                part = have[a:a + n]
                by_file = self._by_file(part)
                parts: Dict[str, List[np.ndarray]] = {v: [] for v in names}
                order: List[int] = []
                for fi in sorted(by_file):
                    for v in in_file_order[fi]:
                        k_layer, sl = index[v]
                        parts[v].append(self._read_part(ds[fi], fi, v, by_file[fi],
                                                        k_layer, sl))
                    order += [p for p, _ in by_file[fi]]
                reduce(PassBlock(year=year, first=a, keys=[r.key for r in part],
                                 labels=[label_text(r.key, self.cadence) for r in part],
                                 data={v: self._assemble(parts.pop(v), order)
                                       for v in names},
                                 units=dict(units), per_second=dict(per_s),
                                 missing=list(lost) if a == 0 else []))
                blocks += 1
        if ra is not None:
            ahead.update(bytes_read=ra.bytes_read, error=ra.error)
        return {"year": year, "records": len(have), "blocks": blocks, "missing": lost,
                "read_ahead": ahead}

    def read_ahead_default(self) -> bool:
        """Whether read_block_pass reads ahead when not told: on a daily or
        sub-daily stream, not on a monthly one (measured in read_block_pass:
        10.6 against 18.3 s for 43 daily fields, 4.5 against 11.5 s for one,
        2.2 against 1.4 s for 5 monthly fields)."""
        return self.cadence != "monthly"

    def _start_read_ahead(self, have: Sequence[Record], ds: Dict[int, Any],
                          ahead: Dict[str, Any]) -> Optional[_ReadAhead]:
        """The read-ahead of a pass over these records, started: in each file
        the span from its first record to its last, the files in the order the
        pass reaches them. None, with the reason in `ahead`, when there is no
        span to read or it cannot start; the pass goes on without it."""
        try:
            spans = []
            for fi in dict.fromkeys(r.file for r in have):
                idx = [r.index for r in have if r.file == fi]
                sp = record_span(ds[fi], self.files[fi], min(idx), max(idx))
                if sp is not None:
                    spans.append((str(self.files[fi]), sp[0], sp[1]))
            ahead["bytes_planned"] = sum(e - s for _, s, e in spans)
            if not spans:
                ahead["error"] = "no span to read: the files are not classic netCDF"
                return None
            ra = _ReadAhead(spans)
            ra.start()
        except Exception as e:                                      # noqa: BLE001
            ahead["error"] = f"{type(e).__name__}: {e}"
            return None
        ahead["on"] = True
        return ra

    def describe(self) -> Dict:
        """What the stream holds, for a record or a log line."""
        return {
            "stream": self.name, "n_files": len(self.files),
            "first_file": self.files[0].name, "last_file": self.files[-1].name,
            "cadence": self.cadence, "steps_per_day": self.steps_per_day,
            "calendar": self.calendar, "n_records": len(self.records),
            "first_label": label_text(self.records[0].key, self.cadence),
            "last_label": label_text(self.records[-1].key, self.cadence),
            "complete_years": self.complete_years(),
            "dropped": self.dropped, "extended": self.extended,
            "grid": {"dims": list(self.grid.dims), "shape": list(self.grid.shape),
                     "n_cells": self.grid.n_cells,
                     "area_km2_total": float(self.grid.weight_km2.sum())},
            "n_variables": len(self._vars),
        }


def open_stream(run_dir, stream: str = "h1") -> Stream:
    """The named history stream of a gridded run. Raises when it is absent.

    Measured on compy01 (login node), 2026-11-04, the Naches control's h1 (ten
    files, 3,285 records), the files' pages dropped from the node's cache and
    checked empty before each open: 47.4 s with an empty index cache (THE
    INDEX ON DISK), then 0.63 s from the cache in a new process, which read
    not one page of the files. The ten entries take 62 KB."""
    found = discover(run_dir)
    if stream not in found:
        raise GriddedReadError(
            f"no {stream} history in {run_dir}; streams present: "
            f"{', '.join(found) or 'none'}")
    return Stream(stream, found[stream])
