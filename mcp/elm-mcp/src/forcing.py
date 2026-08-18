#!/usr/bin/env python3
"""
Prescribed weather for one column
mcp/elm-mcp/src/forcing.py

    in   a column's coordinates, a year, and how to fill each variable
    out  single-gridcell DATM files, a stream that points at them, and the
         namelist lines that make ELM read them

WHY THIS EXISTS. Every ELM run this framework drives reads gridded NLDAS-2 and
therefore takes the weather of a real place. That is right for a site study and
it is a hidden assumption in a conceptual one: a soil sweep answers only for the
climate it borrowed. Writing the weather instead makes the assumption a setting.

UNIFORM IS A SPECIAL CASE OF TIME-VARYING. There is one writer; the fills differ.
A constant is a flat series, a warming experiment is a real series plus an
offset, and a storm-structure experiment is a real series redistributed. Nothing
below treats "idealised" as a different kind of thing from "modified".

    copy         the real cell, extracted. Looks pointless and is the most
                 useful fill here — see verify_against_real() below.
    uniform      constants
    scale        real weather, one variable multiplied
    offset       real weather, one variable shifted
    (fills are functions; adding one does not change anything else)

────────────────────────────────────────────────────────────────────────
HOW ELM IS POINTED AT THESE FILES — resolved from the E3SM source
────────────────────────────────────────────────────────────────────────
The first version of this module guessed at three things and said so. All
three are now settled by reading the tree rather than by building a case.

  * ONE DATA STREAM, not three. namelist_definition_datm.xml:204 —
        <value datm_mode="CLMMOSARTTEST">CLMMOSARTTEST</value>
    The Qian modes split forcing into Solar/Precip/TPQW; CLMMOSARTTEST does
    not, because the NLDAS file carries all seven variables together. So the
    generated stream file is `datm.streams.txt.CLMMOSARTTEST`.

  * `mapalgo = "nn"` AS ONE VALUE IS ALREADY CORRECT. The list is positional
    and the data stream is FIRST — CIME appends presaero and topo after it.
    So elm_wrapper's single "nn" lands on the forcing stream, which is the
    one that matters. presaero and topo keep CIME's bilinear default, which
    is right: they are separate datasets on their own grids and never read
    the file written here.

  * NO NEW RUNTIME KEY, AND NO `streams` OVERRIDE. CIME reads a user-supplied
    stream file if one exists — nmlgen.py:551,

        user_stream_path = os.path.join(caseroot,
                                        "user_" + os.path.basename(stream_path))

    so writing `user_datm.streams.txt.CLMMOSARTTEST` into the case directory
    replaces the generated one. This is much safer than setting `streams` in
    user_nl_datm, which would replace the WHOLE list and silently drop the
    presaero and topo streams that ELM needs.

  * DATM_MODE AND DIN_LOC_ROOT ARE BOTH LEFT ALONE. DIN_LOC_ROOT would have
    been the wrong tool regardless: it moves where surface data and restarts
    are looked up too.

  * taxmode = "cycle" repeats a time axis — how one synthetic year drives a
    multi-year spin-up.

WIRED IN 2026-08-15. `install()` below is the one call a case build makes, and
elm_wrapper._write_prescribed_forcing makes it, from the spec that travelled as
JSON on the column. What remains unproven is only the last link: no ELM case has
yet RUN against a written stream, so a failure to read these files would show up
at model init rather than here. The writer itself round-trips a real NLDAS cell
with zero differences, which is what makes that a narrow risk rather than a
broad one.

ONE BEHAVIOUR THAT SURPRISES PEOPLE: tintalgo = "coszen" on the solar stream.
DATM redistributes shortwave by the cosine of the solar zenith angle, so a
constant FSDS in the file STILL produces a diurnal cycle. Physically sensible,
and not what someone asking for uniform light expects — so flattening it is
opt-in and named, rather than happening because the file said so.
"""
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

# Where the REAL cell is read from when a fill modifies it rather than
# inventing it. There is no second copy of these two constants any more: the
# framework's core/forcing_availability.py held the same pair and opened the
# same directory, and was deleted on 2026-08-17 when its window scan moved
# below — ELM knowledge belongs behind this boundary, not in front of it.
DEFAULT_DIN_LOC_ROOT = "/compyfs/inputdata"
NLDAS_SUBPATH = "atm/datm7/NLDAS"


def real_nldas_dir(root: Optional[str] = None) -> Path:
    """The directory DATM would have read. $DIN_LOC_ROOT wins if it is set."""
    root = root or os.environ.get("DIN_LOC_ROOT") or DEFAULT_DIN_LOC_ROOT
    return Path(root) / NLDAS_SUBPATH


# ─────────────────────────────────────────────────────────────────────────────
# WHICH YEARS CAN ACTUALLY BE SIMULATED
#
# MOVED HERE FROM src/core/forcing_availability.py ON 2026-08-17. That module
# sat in the FRAMEWORK and opened this directory, parsed these filenames, and
# knew DATM_MODE=CLMMOSARTTEST — ELM knowledge on the wrong side of the MCP
# boundary, and a second copy of the two constants above. Reception pasted its
# output into every request, including ones that chose a different model, so a
# PFLOTRAN study of 2024 was refused for an ELM forcing gap.
#
# READ FROM DISK, NEVER ASSERTED, and that is the whole point. The prompt used
# to carry the window as prose — "NLDAS-2 available 1980-2018" — and it had
# drifted in both directions: it stopped at 2018 while Compy holds complete
# years through 2023, and it offered a Qian fallback that DATM_MODE makes
# impossible. A directory listing cannot drift.
# ─────────────────────────────────────────────────────────────────────────────
MONTHS_PER_YEAR = 12
_NLDAS_FILE = re.compile(r"^clmforc\.nldas\.(\d{4})-(\d{2})\.nc$")


def month_counts(directory) -> Dict[int, int]:
    """{year: months present}.

    Filenames that do not match are ignored — the tree carries a couple of
    strays (clmforc.nldas.0016-06.nc) that would otherwise invent a year 16 AD.
    """
    counts: Dict[int, int] = {}
    directory = Path(directory)
    if not directory.is_dir():
        return counts
    for name in os.listdir(directory):
        m = _NLDAS_FILE.match(name)
        if m:
            counts[int(m.group(1))] = counts.get(int(m.group(1)), 0) + 1
    return counts


def contiguous_span(years) -> List[int]:
    """Longest run of consecutive years. Pure arithmetic, no filesystem.

    A run needs an UNBROKEN window: a gap year mid-record would abort the
    simulation partway rather than at submit time, so the longest gap-free run
    is what may be offered, not first..last.
    """
    ys = sorted(set(int(y) for y in years))
    if not ys:
        return []
    best = run = [ys[0]]
    for y in ys[1:]:
        run = run + [y] if y == run[-1] + 1 else [y]
        if len(run) > len(best):
            best = run
    return best


def scan_window(root: Optional[str] = None) -> Dict[str, Any]:
    """What the forcing tree holds. The only function here that touches disk."""
    d = real_nldas_dir(root)
    counts = month_counts(d)
    complete = [y for y, n in counts.items() if n >= MONTHS_PER_YEAR]
    span = contiguous_span(complete)
    partial = {y: n for y, n in counts.items() if n < MONTHS_PER_YEAR}
    return {
        "dataset": "NLDAS-2",
        "resolution": "0.125 deg (~12 km)",
        "datm_mode": STREAM_NAME,
        "path": str(d),
        "exists": d.is_dir(),
        "n_files": sum(counts.values()),
        "yr_first": span[0] if span else None,
        "yr_last": span[-1] if span else None,
        "n_years": len(span),
        "partial_years": dict(sorted(partial.items())),
        "excluded": sorted(set(complete) - set(span)),
    }

# The seven DATM reads, with the units the real files carry. Verified by
# opening /compyfs/inputdata/atm/datm7/NLDAS/clmforc.nldas.1995-01.nc.
VARIABLES: Dict[str, str] = {
    "PRECTmms": "mm/s",
    "TBOT":     "K",
    "WIND":     "m/s",
    "QBOT":     "kg/kg",
    "FLDS":     "W/m2",
    "FSDS":     "W/m2",
    "PSRF":     "Pa",
}

# The real files are hourly and monthly: January is 744 steps.
# WHAT DATM CALLS EACH VARIABLE INTERNALLY. Copied from a stream file CIME
# GENERATED for a working site case, not derived — the first version built these
# by lowercasing the file's variable name, which is right for exactly two of the
# seven and wrong for the other five:
#
#     (datm_comp_run) max values = 265.94, 0.0, 0.0
#     ERROR: (datm_comp_run) ERROR: cannot compute shum
#
# The tell was `flds_strm = strm_tbot:strm_wind` in the log — TBOT and WIND
# mapped because tbot/wind happen to be their internal names, and the other
# five silently mapped to nothing at all. DATM does not complain about a field
# it was never given; it fails later computing something from the zeros.
DATM_FIELD: Dict[str, str] = {
    "TBOT":     "tbot",
    "WIND":     "wind",
    "QBOT":     "shum",
    "PSRF":     "pbot",
    "FLDS":     "lwdn",
    "PRECTmms": "precn",
    "FSDS":     "swdn",
}

STEPS_PER_DAY = 24
TIME_UNITS = "days since 1979-01-01 00:00:00"
FILE_PATTERN = "clmforc.nldas.{year:04d}-{month:02d}.nc"

# NLDAS-2, matching inputs._nldas_cell.
NLDAS_DEG = 0.125

# The stream CLMMOSARTTEST defines — one, carrying all seven variables.
# namelist_definition_datm.xml:204. CIME names the generated file after it, and
# looks for `user_` + that name in the case directory first.
STREAM_NAME = "CLMMOSARTTEST"
STREAM_FILE = f"datm.streams.txt.{STREAM_NAME}"
USER_STREAM_FILE = f"user_{STREAM_FILE}"

_missing_field = sorted(set(VARIABLES) - set(DATM_FIELD))
if _missing_field:                                          # pragma: no cover
    raise ImportError(
        f"no DATM field name for {_missing_field}. A variable written into the "
        f"file but absent from the stream's variableNames is read by nothing, "
        f"and DATM reports that as a failure to compute something else "
        f"entirely — see the note on DATM_FIELD.")

_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def days_in_month(year: int, month: int) -> int:
    """ELM's calendar is NOLEAP — every February has 28 days.

    Using a real calendar here would put 29 steps of February into a file the
    model reads with 28, and the offset would carry silently through the rest
    of the year.
    """
    return _DAYS_IN_MONTH[month - 1]


# ─────────────────────────────────────────────────────────────────────
# FILLS
# ─────────────────────────────────────────────────────────────────────
# A fill takes (variable, the real series for this cell, n_steps) and returns
# the series to write. Real values arrive as a plain list so a fill never needs
# netCDF, and a new fill is a function rather than a branch in the writer.
def fill_copy(_var: str, real: Optional[List[float]], n: int) -> List[float]:
    """The real cell, unchanged. The fill that proves the machinery."""
    if real is None:
        raise ValueError("copy needs the real series; none was read")
    return list(real[:n])


def fill_uniform(values: Dict[str, float]) -> Callable:
    """Constants, one per variable. Anything unnamed keeps its real series."""
    def _f(var: str, real: Optional[List[float]], n: int) -> List[float]:
        if var in values:
            return [float(values[var])] * n
        if real is None:
            raise ValueError(f"uniform gave no value for {var} and there is no "
                             f"real series to fall back on")
        return list(real[:n])
    return _f


def fill_scale(factors: Dict[str, float]) -> Callable:
    """Real weather with one or more variables multiplied.

    Keeps the diurnal cycle, the seasonality and the storm structure, which is
    what makes a +20% precipitation experiment interpretable where a constant
    drizzle is not: rainfall INTENSITY drives the infiltration/runoff split,
    and a flat series removes intensity by construction.
    """
    def _f(var: str, real: Optional[List[float]], n: int) -> List[float]:
        if real is None:
            raise ValueError(f"scale needs the real series for {var}")
        k = float(factors.get(var, 1.0))
        return [v * k for v in real[:n]]
    return _f


def fill_offset(deltas: Dict[str, float]) -> Callable:
    """Real weather with one or more variables shifted — a +2 K experiment."""
    def _f(var: str, real: Optional[List[float]], n: int) -> List[float]:
        if real is None:
            raise ValueError(f"offset needs the real series for {var}")
        d = float(deltas.get(var, 0.0))
        return [v + d for v in real[:n]]
    return _f


FILLS = {"copy": fill_copy, "uniform": fill_uniform,
         "scale": fill_scale, "offset": fill_offset}

# Which fills need the real cell read first. `copy`, `scale` and `offset` are
# all modifications OF the real series; only `uniform` can invent one — and even
# then only for the variables it names, so it too reads the real cell whenever
# it leaves any of the seven unspecified.
NEEDS_REAL = {"copy", "scale", "offset"}


def spec_to_fill(spec: Any) -> Dict[str, Any]:
    """A JSON fill spec → the callable, plus what reading it implies.

    THE TRANSLATION THAT HAS TO EXIST SOMEWHERE. A design travels as JSON
    through the planner, columns.json and case_inputs.json; a fill is a Python
    function. Doing the conversion here rather than at each of those boundaries
    means one definition of what a spec may say, and one error message when it
    says something else.

    Accepts the compact form too — the string "copy" is {"fill": "copy"} — so a
    level that carries no parameters does not have to be written as a dict.

    Returns {fill, kind, needs_real, values, flat_solar}. `needs_real` is the
    caller's instruction to pass src_dir; getting it wrong raises inside the
    fill rather than writing a wrong file, but the caller should not have to
    find out that way.
    """
    if isinstance(spec, str):
        spec = {"fill": spec}
    if not isinstance(spec, dict):
        raise ValueError(f"a fill spec is a string or an object, not {spec!r}")

    kind = str(spec.get("fill") or "").strip().lower()
    if kind not in FILLS:
        raise ValueError(f"unknown fill {kind!r}; this module has "
                         f"{sorted(FILLS)}")

    values = dict(spec.get("values") or {})
    unknown = sorted(set(values) - set(VARIABLES))
    if unknown:
        raise ValueError(
            f"{kind} names variable(s) {unknown}, which DATM does not read "
            f"here. The seven are {sorted(VARIABLES)}. A misspelt name would "
            f"otherwise be applied to nothing and the run would look like the "
            f"experiment succeeded with no effect.")
    if kind != "copy" and not values:
        raise ValueError(f"{kind} was given no variables to act on, so it "
                         f"would write the real weather under an experiment's "
                         f"name. Use fill 'copy' if that is what you want.")

    # AN IDENTITY IS A COPY, whatever it was called. Scaling by 1.0 and
    # offsetting by 0.0 reproduce the source byte for byte — the check above
    # already refuses the empty spelling of exactly this, and these are the
    # same statement with a number in it.
    #
    # NORMALISED RATHER THAN REFUSED, because inside a factor list an identity
    # is the CONTROL and belongs there. What must not happen is a design record
    # calling it an experiment, or a level list holding both `copy` and
    # `scale 1.0` as though they were two different columns.
    normalised_from = None
    if kind == "scale" and all(float(v) == 1.0 for v in values.values()):
        normalised_from, kind, values = f"scale by {values}", "copy", {}
    elif kind == "offset" and all(float(v) == 0.0 for v in values.values()):
        normalised_from, kind, values = f"offset by {values}", "copy", {}

    fill = fill_copy if kind == "copy" else FILLS[kind](values)
    # uniform still needs the real cell for anything it did not name.
    needs_real = kind in NEEDS_REAL or set(values) != set(VARIABLES)
    return {"fill": fill, "kind": kind, "values": values,
            "needs_real": needs_real,
            "normalised_from": normalised_from,
            "flat_solar": bool(spec.get("flat_solar", False))}


# ─────────────────────────────────────────────────────────────────────
# READING THE REAL CELL
# ─────────────────────────────────────────────────────────────────────
# WHY ONE POINT OUT OF ONE MONTH IS EXPENSIVE, and what actually fixes it
# ─────────────────────────────────────────────────────────────────────
# The source files are the whole CONUS grid for every hour of the month:
# 464 x 224 cells x 744 steps x 7 variables x float32 = 2.17 GB.
#
# `time` IS THE UNLIMITED DIMENSION, and that is the fact that governs
# everything here. NETCDF3_CLASSIC interleaves record variables BY RECORD, so
# the file is 744 records of 2.91 MB, each holding all seven variables' maps for
# one hour. One cell's series is therefore 744 values spaced a whole record
# apart, spread across the entire file — and reading each variable separately
# means SEVEN traversals of 2.17 GB, about 15 GB, to extract 21 KB.
#
# Measured on cold 1997 files:
#     raw sequential (dd)               2.17 GB in 12.4 s  = 174 MB/s
#     netCDF4, 7 separate reads                     65.3 s
#     scipy netcdf_file(mmap=True)                  16.2 s
#
# The mmap version faults in only the pages it touches — 7 small reads inside
# each 2.91 MB record instead of seven passes over all of them — and returns
# BIT-IDENTICAL values for all seven variables and both coordinate arrays.
# There is no scale_factor or add_offset on these variables to mishandle.
#
# An earlier note here claimed the values sat 415,744 bytes apart and that
# reading sequentially bought nothing. Both were wrong, and wrong for the same
# reason: they assumed contiguous per-variable storage, which a record
# dimension does not give you.
_CELL_CACHE: Dict[tuple, Dict[str, List[float]]] = {}


def clear_cache() -> None:
    """Forget the read cells. For tests; a run reads each month once anyway."""
    _CELL_CACHE.clear()


def _locate(lats, lons, lat: float, lon: float) -> tuple:
    """(j, i) of the nearest cell, on the file's OWN coordinate arrays.

    Nearest-index on what the file says rather than recomputed from the
    published grid geometry, so a change in the forcing dataset cannot make
    this disagree with the cell DATM would have picked.
    """
    import numpy as np
    # LONGXY may be 0..360 or -180..180; compare in the file's own frame.
    lon_q = lon % 360.0 if float(np.nanmax(lons)) > 180.0 else lon
    dist = (np.asarray(lats) - lat) ** 2 + (np.asarray(lons) - lon_q) ** 2
    return tuple(int(x) for x in
                 np.unravel_index(int(np.argmin(dist)), dist.shape))


def _read_cell_mmap(path: Path, lat: float, lon: float):
    """(j, i, series) by memory-mapping the file, or None if that cannot work.

    Returns None rather than raising so the caller falls back to netCDF4: scipy
    reads NETCDF3 only, and if the forcing dataset is ever rewritten as
    NETCDF4/HDF5 this stops being possible with no other warning.
    """
    try:
        import numpy as np
        from scipy.io import netcdf_file
    except ImportError:                                     # pragma: no cover
        return None
    try:
        f = netcdf_file(str(path), "r", mmap=True)
    except Exception:                                       # noqa: BLE001
        return None
    try:
        j, i = _locate(np.asarray(f.variables["LATIXY"].data),
                       np.asarray(f.variables["LONGXY"].data), lat, lon)
        # THE CACHE IS CHECKED HERE, not by the caller, because finding (j, i)
        # is the cheap part and reading the series is the whole cost.
        hit = _CELL_CACHE.get((str(path), j, i))
        if hit is not None:
            return j, i, {k: list(v) for k, v in hit.items()}
        out = {v: [float(x) for x in np.array(f.variables[v].data[:, j, i])]
               for v in VARIABLES if v in f.variables}
        return j, i, out
    except Exception:                                       # noqa: BLE001
        return None
    finally:
        # After close() the mapping is gone; every array above was copied out
        # with np.array/float first, so nothing here is a view into it.
        f.close()


def _read_cell_netcdf4(path: Path, lat: float, lon: float):
    """The same thing through netCDF4. Four times slower, and always available."""
    import netCDF4 as nc
    import numpy as np

    with nc.Dataset(str(path)) as d:
        j, i = _locate(np.asarray(d.variables["LATIXY"][:]),
                       np.asarray(d.variables["LONGXY"][:]), lat, lon)
        hit = _CELL_CACHE.get((str(path), j, i))
        if hit is not None:
            return j, i, {k: list(v) for k, v in hit.items()}
        out = {v: [float(x) for x in np.asarray(d.variables[v][:, j, i])]
               for v in VARIABLES if v in d.variables}
    return j, i, out


def read_real_cell(src_dir, year: int, month: int,
                   lat: float, lon: float) -> Dict[str, List[float]]:
    """The seven series at the cell containing (lat, lon), as plain lists."""
    p = Path(src_dir) / FILE_PATTERN.format(year=year, month=month)

    got = _read_cell_mmap(p, lat, lon) or _read_cell_netcdf4(p, lat, lon)
    j, i, out = got

    # A CELL WITH NO DATA IS NOT A CELL OF 1e20. Every variable carries
    # _FillValue = 1e20, so a point over water or off the domain comes back as
    # a full series of it — and would be written into a forcing file, handed to
    # ELM, and produce numbers. Refusing here is the only place this is still
    # obviously wrong rather than merely surprising.
    missing = sorted(v for v, s in out.items()
                     if s and all(x >= 1e19 for x in s))
    if missing:
        raise ValueError(
            f"{p.name} cell ({j}, {i}) for ({lat}, {lon}) is fill value in "
            f"{missing} — there is no weather at this point. A forcing file "
            f"written from it would run and mean nothing.")

    _CELL_CACHE[(str(p), j, i)] = {k: list(v) for k, v in out.items()}
    # Copied out, because a fill is handed the list and a caller that modified
    # it in place would silently change the next column's weather.
    return {k: list(v) for k, v in out.items()}


# ─────────────────────────────────────────────────────────────────────
# WRITING
# ─────────────────────────────────────────────────────────────────────
def write_month(out_dir, year: int, month: int, lat: float, lon: float,
                fill, real: Optional[Dict[str, List[float]]] = None) -> str:
    """One single-gridcell DATM file, in the format the real files use.

    THE CELL IS PLACED AT THE COLUMN'S OWN COORDINATES, and that is not a
    convenience. elm_wrapper writes mapalgo="nn", so DATM takes the NEAREST
    forcing cell to the model point — it does not check the point is inside the
    forcing domain. A single-cell file written somewhere else would be used
    anyway, silently, and the run would finish with weather from a place nobody
    chose. Taking the coordinates from the column rather than as a separate
    argument is what makes that impossible to get wrong.
    """
    import netCDF4 as nc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / FILE_PATTERN.format(year=year, month=month)

    n = days_in_month(year, month) * STEPS_PER_DAY
    # Days since the epoch, at the START of this month, NOLEAP.
    day0 = 365 * (year - 1979) + sum(_DAYS_IN_MONTH[:month - 1])
    times = [day0 + k / STEPS_PER_DAY for k in range(n)]
    half = NLDAS_DEG / 2.0

    with nc.Dataset(str(p), "w", format="NETCDF3_CLASSIC") as d:
        d.createDimension("time", n)
        d.createDimension("lat", 1)
        d.createDimension("lon", 1)
        d.createDimension("scalar", 1)

        t = d.createVariable("time", "f8", ("time",))
        t.units = TIME_UNITS
        t.calendar = "noleap"
        t[:] = times

        xy = {"LONGXY": (lon, "degrees_east"), "LATIXY": (lat, "degrees_north")}
        for name, (val, units) in xy.items():
            v = d.createVariable(name, "f8", ("lat", "lon"))
            v.units = units
            v[:] = [[val]]

        for name, val, units in (("EDGEE", lon + half, "degrees_east"),
                                 ("EDGEW", lon - half, "degrees_east"),
                                 ("EDGEN", lat + half, "degrees_north"),
                                 ("EDGES", lat - half, "degrees_north")):
            v = d.createVariable(name, "f8", ("scalar",))
            v.units = units
            v[:] = [val]

        for name, units in VARIABLES.items():
            series = fill(name, (real or {}).get(name), n)
            if len(series) != n:
                raise ValueError(
                    f"{name}: fill returned {len(series)} steps for a month "
                    f"needing {n} — a short series would be cycled by DATM "
                    f"and the mismatch would not be reported")
            v = d.createVariable(name, "f4", ("time", "lat", "lon"))
            v.units = units
            v[:] = [[[x]] for x in series]

        d.title = "single-gridcell prescribed forcing (IDEAS conceptual run)"
        d.note = ("Written for one column. mapalgo='nn' means DATM takes the "
                  "nearest cell without checking containment, so this file's "
                  "coordinates ARE the column's.")
    return str(p)


def write_year(out_dir, year: int, lat: float, lon: float, fill,
               src_dir=None, progress=None) -> List[str]:
    """Twelve months. Reads the real cell first when the fill needs it.

    SAYS WHAT IT IS DOING, because a cold year takes about twelve minutes and
    the alternative is twelve minutes of silence in the middle of a case build,
    which is indistinguishable from a hang. Cached months return instantly, so
    the line also shows when the reading stopped being the cost.
    """
    # FLUSHED, or it is not progress. Python block-buffers stdout when it is a
    # file rather than a terminal, and a batch job's stdout is always a file —
    # so these lines sat in the buffer and appeared only when the job ended,
    # which is exactly when nobody needs them. Observed on the 10-year build:
    # seventeen minutes in, the log held nothing but its two startup lines.
    def say(msg):
        (progress or print)(msg, **({} if progress else {"flush": True}))
    written = []
    for month in range(1, 13):
        real = None
        if src_dir is not None:
            t0 = time.time()
            real = read_real_cell(src_dir, year, month, lat, lon)
            dt = time.time() - t0
            say(f"   forcing {year}-{month:02d}: read the real cell in "
                f"{dt:5.1f}s" + ("  (cached)" if dt < 1.0 else ""))
        written.append(write_month(out_dir, year, month, lat, lon, fill, real))
    return written


# ─────────────────────────────────────────────────────────────────────
# POINTING ELM AT IT
# ─────────────────────────────────────────────────────────────────────
def _year_list(years) -> List[int]:
    """One year or many, always as a sorted list of ints."""
    if isinstance(years, int):
        return [years]
    out = sorted({int(y) for y in years})
    if not out:
        raise ValueError("no years given — a stream with no files in it makes "
                         "DATM fail at init with a message about the domain")
    return out


def stream_text(data_dir, domain_file: str, years) -> str:
    """The stream file DATM reads, in the format a generated one uses.

    EVERY YEAR OF THE RUN GETS LISTED. A 5-year run against a stream naming one
    year does not fail: DATM cycles the axis it was given, so the run finishes
    having repeated 1995 five times, and nothing in the log says so.

    THE DOMAIN IS THE FORCING FILE ITSELF, not the case's CIME domain file, and
    getting this wrong is what made the first real run fail at init:

        (shr_strdata_init_streams) fileName = Domainfile_47.1106_-121.3851.nc
        (shr_strdata_init_streams)  lonName = LONGXY
        ERROR: (shr_ncread_varDimNum) ERROR inq varid: LONGXY

    `domainInfo` names the variables DATM reads the STREAM's grid from, and
    those are the names in the file being read — LATIXY/LONGXY, which every
    forcing file here carries because write_month puts them there along with
    the EDGE* bounds. The CIME domain file describes the MODEL's grid and uses
    xc/yc; pointing at it while asking for LONGXY asks one file for another
    file's variables.

    `domain_file` is kept in the signature and deliberately unused: callers
    pass the case's domain and it is the wrong file for this field. Taking it
    and ignoring it is clearer than a parameter that silently means something
    else, and the caller still needs its own domain for xmlchange.
    """
    yrs = _year_list(years)
    names = "\n".join("    " + FILE_PATTERN.format(year=y, month=m)
                      for y in yrs for m in range(1, 13))
    # The first month, which exists whenever any of them do.
    first = FILE_PATTERN.format(year=yrs[0], month=1)
    variables = "\n".join(f"     {v} {DATM_FIELD[v]}" for v in VARIABLES)
    return f"""<?xml version="1.0"?>
<file id="stream" version="1.0">
<dataSource>
   GENERIC
</dataSource>
<domainInfo>
  <variableNames>
     time    time
     LONGXY  lon
     LATIXY  lat
  </variableNames>
  <filePath>
     {Path(data_dir)}
  </filePath>
  <fileNames>
     {first}
  </fileNames>
</domainInfo>
<fieldInfo>
   <variableNames>
{variables}
   </variableNames>
   <filePath>
     {Path(data_dir)}
   </filePath>
   <fileNames>
{names}
   </fileNames>
   <offset>
      0
   </offset>
</fieldInfo>
</file>
"""


def install_stream(case_dir, data_dir, domain_file: str, years) -> str:
    """Write the user stream file CIME reads instead of its generated one.

    THE WHOLE POINTING MECHANISM, and it is one file in the case directory.
    CIME checks for `user_` + the stream's own filename before generating one
    (nmlgen.py:551), so this replaces the forcing source without touching
    DATM_MODE, DIN_LOC_ROOT, or the `streams` list.

    NOT AN OVERRIDE OF `streams` IN user_nl_datm, deliberately. That variable
    is the WHOLE list — the data stream plus presaero plus topo — so setting it
    to one entry would silently drop the aerosol and topography streams ELM
    needs. This replaces one stream's contents and leaves the list alone.
    """
    p = Path(case_dir) / USER_STREAM_FILE
    p.write_text(stream_text(data_dir, domain_file, years))
    return str(p)


def namelist_lines(flat_solar: bool = False) -> str:
    """Extra user_nl_datm lines. Usually none — the stream file does the work.

    `mapalgo` is NOT written here. elm_wrapper already writes a single "nn",
    the list is positional, and the data stream is first — so the forcing
    stream is already nearest-neighbour, which is what a single-cell file
    needs. Writing more entries would only reach presaero and topo, which are
    on their own grids and should keep CIME's default.

    `flat_solar` pins tintalgo, and is the one thing a caller may want. DATM
    redistributes shortwave by the cosine of the solar zenith angle, so a
    constant FSDS in the file still produces a diurnal cycle — correct physics,
    and not what someone asking for uniform light expects. Opt-in and named
    rather than silently applied.
    """
    if not flat_solar:
        return ""
    return ('! uniform light: stop DATM reshaping FSDS by solar zenith angle\n'
            'tintalgo = "linear"\n')


# ─────────────────────────────────────────────────────────────────────
# THE ONE CALL A CASE BUILD MAKES
# ─────────────────────────────────────────────────────────────────────
def install(case_dir, data_dir, spec: Any, lat: float, lon: float,
            years, domain_file: str, src_dir=None) -> Dict[str, Any]:
    """Write the weather, the stream that points at it, and say what to add.

    EVERYTHING OR NOTHING. The three products only work together — files
    without a stream are ignored, a stream without files fails at init, and
    either half succeeding would leave a case that runs on the WRONG weather
    rather than one that refuses to run. So this raises on the first problem
    and the caller has nothing to unwind.

    Returns the paths plus `namelist` — lines the caller appends to
    user_nl_datm. Returned rather than written because this module does not own
    that file; elm_wrapper composes it from several sources and appending here
    would race its own write.
    """
    resolved = spec_to_fill(spec)
    yrs = _year_list(years)
    if src_dir is None and resolved["needs_real"]:
        raise ValueError(
            f"fill {resolved['kind']!r} is a modification of the real cell "
            f"(or leaves some of the seven variables unnamed) but no source "
            f"directory was given, so there is nothing to modify")

    written: List[str] = []
    for y in yrs:
        written += write_year(data_dir, y, lat, lon, resolved["fill"],
                              src_dir=src_dir if resolved["needs_real"] else None)

    stream = install_stream(case_dir, data_dir, domain_file, yrs)
    return {
        "fill": resolved["kind"],
        "values": resolved["values"],
        "years": yrs,
        "lat": lat, "lon": lon,
        "n_files": len(written),
        "data_dir": str(Path(data_dir)),
        "stream_file": stream,
        "namelist": namelist_lines(resolved["flat_solar"]),
    }


# ─────────────────────────────────────────────────────────────────────
# THE TEST WITH A KNOWN ANSWER
# ─────────────────────────────────────────────────────────────────────
def verify_against_real(out_dir, src_dir, year: int, month: int,
                        lat: float, lon: float) -> Dict[str, Any]:
    """Write the real cell out and read it back. Every value must survive.

    THE ONLY CHECK HERE WITH AN ANSWER KNOWN IN ADVANCE. Every other fill
    produces weather nobody has seen, so a mistake in the writer, the time
    axis, the units or the grid would look exactly like a result. Extracting
    the real cell and round-tripping it is the one case where a disagreement is
    unambiguous.

    Float32 storage is the only permitted difference: the source is float32
    too, so the comparison is exact rather than tolerant.
    """
    real = read_real_cell(src_dir, year, month, lat, lon)
    path = write_month(out_dir, year, month, lat, lon, fill_copy, real)

    import netCDF4 as nc
    bad: Dict[str, Any] = {}
    with nc.Dataset(path) as d:
        n_expected = days_in_month(year, month) * STEPS_PER_DAY
        if len(d.dimensions["time"]) != n_expected:
            bad["time"] = (f"{len(d.dimensions['time'])} steps written, "
                           f"{n_expected} expected")
        for v in VARIABLES:
            if v not in real:
                continue
            got = [float(x) for x in d.variables[v][:, 0, 0]]
            src = real[v][:len(got)]
            diffs = sum(1 for a, b in zip(got, src) if a != b)
            if diffs:
                bad[v] = f"{diffs} of {len(got)} values differ"
    return {"ok": not bad, "file": path, "differences": bad,
            "n_steps": days_in_month(year, month) * STEPS_PER_DAY}
