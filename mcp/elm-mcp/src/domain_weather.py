#!/usr/bin/env python3
"""A run's own weather: the data atmosphere's anomaly forcing (weather_scale)
and a recorded year read in another model year (weather_repeat_year).

The weather tool of docs/ELM_DOMAIN_EXPERIMENTS.md: the changes
`weather_scale` and `weather_repeat_year` of domain_changes.py. The second
is at the end of this file (REPEATING A YEAR).

The part of E3SM that hands recorded weather to the land model, the data
atmosphere (DATM, here in its CLMNCEP mode), can add a fixed amount to the
air temperature and multiply the precipitation, cell by cell and at every
model step, after the recorded weather has been mapped onto the model's
cells. E3SM calls this anomaly forcing (namelist `anomaly_forcing`;
components/data_comps/datm/src/datm_comp_mod.F90). No forcing file is copied
or rewritten. This module writes what a run needs for it:

  write_anomaly(...)  one netCDF on the model's cells: the amount added to the
                      temperature (tas, K) and the precipitation factor (pr)
                      for every month of the run's years; it carries the
                      cells' own coordinates, so it describes its own grid
  stream_text(...)    the stream file through which DATM reads it
  datm_lines(...)     the user_nl_datm lines: the anomaly streams, and how
                      DATM reads between their records ("lower")
  inputs(...)         all of it for one run: the files written under the
                      run's inputs folder, the user files of its recipe, its
                      user_nl_datm lines, and the checks its jobs apply (CIME
                      copies a user stream file into the case unchanged,
                      CIME/nmlgen.py, so the case stage compares checksums)
  check_year(...)     the year check: the run's daily history against the
                      control's, under the file

WHAT THE LAND MODEL RECEIVES (read in the code, 2026-10-25/26). DATM splits
precipitation into rain and snow, caps the air's moisture (specific
humidity) at saturation and sets the potential temperature from the
RECORDED temperature, before the anomaly is applied; then it adds the amount
to the temperature and multiplies rain and snow, each, by the factor. A
warmer run therefore does not turn snow into rain. The land model computes
the air's relative humidity and density itself, from the temperature it
receives and the recorded moisture (cpl/lnd_import_export.F90): a warmer
run is also drier air to it, relative humidity falling by about 6-7 % of
its value per kelvin, and its fire model reads that. In a cooler run the
recorded moisture can exceed saturation for the cooled air, so dew and frost
form that the recorded weather would not make. The change's not-meaning
says all of this.

THE TIME AXIS. DATM refuses a stream whose record spacing varies by more
than dtlimit (1.5), so there is one record per month, at the month's first
instant, read with "lower" interpolation: the earlier of the two records
around the time, at full weight. CIME's own default for these streams is
"nearest" (seen on a scratch clone of the Naches control resolved by
preview_namelists), which would give the second half of every month the
next month's record: hence the tintalgo line.

WHEN A MONTH'S VALUE TAKES EFFECT (read in the code, 2026-10-25). Each land
step uses the weather DATM computed at the step's start: the driver advances
the clock, runs the land on what the atmosphere sent before, then runs the
atmosphere (driver-mct/main/cime_comp_mod.F90). DATM moves to the next pair
of records only when the time is strictly past the later one
(shr_dmodel_readLBUB), and "lower" keeps the earlier record
(shr_tInterp_getFactors). With records exactly at midnight, the step that
starts at 00:00 on the 1st would still take the previous month's value, one
step in 48 of the day. So the stream file moves every record LEAD_SECONDS
earlier (its <offset>, added to each record's time in shr_stream_readTCoord):
each step takes its own month's value, for any model step longer than that
lead. The proof on the Naches control (2026-10-26) found 1 June and
1 September exact.

THE YEARS. The anomaly streams' default years are 2006-2300, with the data
year equal to the model year (namelist_definition_datm.xml); DATM reads no
record dated before 1 January 2006. The first year's January record, moved
by the lead, takes effect just before that year begins, and the last time
DATM is called in a year's job is 00:00 on 1 January after the run's last
year: both must lie inside 2006-2300, so a run's years lie in 2007-2299
(STREAM_YEARS, year_problem). The file also holds the year after the run's
last, whose January closes the run's December, and the year before its
first, which DATM never needs but which does no harm.

THE CELLS. The file carries the domain file's own xc, yc and mask, so DATM
finds the stream's grid equal to the model's (within 0.01 degrees) and maps
nothing: each cell takes its own value (shr_strdata_mod.F90,
shr_strdata_init_mapping, domaps false). Only active cells (domain mask 1)
are changed; a change that reaches none is refused.

THE FILE'S NAME is its content's (weather_anomaly_<sha256[:12]>.nc), so
building a run again never overwrites a file an armed or finished chain
reads.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ANOMALY = {
    "temperature_K": {"stream": "Anomaly.Forcing.Temperature", "var": "tas",
                      "model_var": "tbot_af", "how": "offset", "identity": 0.0,
                      "units": "K", "history": ["TBOT"], "tolerance": 1e-3,
                      "why": "the daily air temperature the land model receives is the "
                             "control's plus the amount in the chosen months and cells, "
                             "and the control's elsewhere",
                      "unchanged": "the daily air temperature the land model receives "
                                   "is the control's"},
    "precip_factor": {"stream": "Anomaly.Forcing.Precip", "var": "pr",
                      "model_var": "prec_af", "how": "factor", "identity": 1.0,
                      "units": "1", "history": ["RAIN", "SNOW"], "tolerance": 1e-5,
                      "why": "daily rain and daily snow are each the control's times the "
                             "factor in the chosen months and cells, and the control's "
                             "elsewhere",
                      "unchanged": "daily rain and daily snow are each the control's"},
}
# The tolerances sit far above what float32 history values make (the proof:
# 0.0 K and 1.04e-7) and below what a one-step slip at a month boundary makes
# for a change of 0.05 K or a factor 0.0005 from 1 (amount / 48 per day).
FILE_STEM = "weather_anomaly"
INTERP = "lower"
ALL_MONTHS = list(range(1, 13))
# Below this a day's precipitation (mm/s) is too small for its ratio to mean
# anything: 1e-9 mm/s is 0.0000864 mm in a day.
PRECIP_FLOOR = 1e-9
EARTH_RADIUS_KM = 6371.22
# How far (degrees) a history cell's centre may sit from the anomaly file's:
# ELM writes lat and lon as float32 (about 8e-6 degrees of rounding near 238),
# and a cell of the Naches grid is 0.0083 degrees wide.
CELL_MATCH_DEG = 1e-4
# The anomaly streams' years (strm_year_align, _start, _end in
# namelist_definition_datm.xml); inside them the data year is the model year.
STREAM_YEARS = (2006, 2300)
# How far before its month's first instant each record takes effect (the
# stream file's offset): shorter than any model step, so the step that starts
# at midnight on the 1st takes the new month (WHEN A MONTH'S VALUE TAKES
# EFFECT above).
LEAD_SECONDS = 60
# DATM's per-stream namelist entries (per_stream_entry="true" in
# namelist_definition_datm.xml): CIME appends one entry to each for every
# stream, so a run's lists are the control's with the anomaly streams' added.
PER_STREAM_KEYS = ("dtlimit", "fillalgo", "fillmask", "fillread", "fillwrite", "mapalgo",
                   "mapmask", "mapread", "mapwrite", "readmode", "taxmode", "tintalgo")


class WeatherError(Exception):
    """A weather change cannot be written, and the message says why."""


def active(params: Dict[str, Any]) -> List[str]:
    """The anomaly parameters that change something, in ANOMALY's order. A
    value is compared as the file stores it (float32), so one that rounds to
    the identity changes nothing."""
    return [k for k, a in ANOMALY.items()
            if float(np.float32(params.get(k, a["identity"]))) != a["identity"]]


def year_problem(years: Sequence[int]) -> Optional[str]:
    """Why the anomaly streams cannot cover these model years, or None (THE
    YEARS above)."""
    lo, hi = STREAM_YEARS
    out = sorted(int(y) for y in years if not lo < int(y) < hi)
    if out:
        return (f"the data atmosphere reads its anomaly streams for {lo}-{hi} only: the "
                f"first year's January record takes effect just before that year begins, "
                f"and the last call falls on 1 January after the last year, so a run's "
                f"years must lie in {lo + 1}-{hi - 1}; this one has {out}")
    return None


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# THE CONTROL'S GRID, AS DATM SEES IT
# ─────────────────────────────────────────────────────────────────────────────
def control_datm(case_dir) -> Dict[str, list]:
    """The control's resolved datm_in (CaseDocs/datm_in), read."""
    from run_gate import read_namelist
    p = Path(case_dir) / "CaseDocs" / "datm_in"
    if not p.is_file():
        raise WeatherError(f"{p} is missing: the control's case has no resolved datm_in")
    return read_namelist(p)


def domain_file(case_dir) -> Path:
    """The land domain file the control's data atmosphere maps onto
    (CaseDocs/datm_in, shr_strdata_nml domainfile)."""
    d = (control_datm(case_dir).get("domainfile") or [None])[0]
    if not d or not Path(str(d)).is_file():
        raise WeatherError(f"the control's datm_in names no readable domainfile ({d!r})")
    return Path(str(d))


def read_domain(path) -> Dict[str, np.ndarray]:
    """xc, yc, area, mask and frac of a domain file, each (nj, ni)."""
    import netCDF4
    with netCDF4.Dataset(path) as f:
        out = {}
        for v in ("xc", "yc", "area", "mask", "frac"):
            if v not in f.variables:
                raise WeatherError(f"{Path(path).name} has no {v}")
            out[v] = np.asarray(f[v][:])
    if out["xc"].ndim != 2 or out["xc"].shape != out["yc"].shape:
        raise WeatherError(f"{Path(path).name}: xc and yc are not one 2-D grid")
    return out


def grid_from_domain(dom: Dict[str, np.ndarray]):
    """The gridded.Grid of a domain file, longitudes put in -180..180 as the
    history reader does, so a region's mask is the one the analysis makes.
    The cells must be a regular latitude-longitude grid (the reader's own
    kind): its axes are what a polygon's cell edges are drawn from."""
    import gridded as G
    lat = np.asarray(dom["yc"], float)
    lon = np.asarray(G.to_pm180(np.asarray(dom["xc"], float)), float)
    lat_axis, lon_axis = lat[:, 0], lon[0, :]
    if not (np.allclose(lat, lat_axis[:, None], rtol=0.0, atol=1e-9)
            and np.allclose(lon, lon_axis[None, :], rtol=0.0, atol=1e-9)):
        raise WeatherError("the control's domain is not a regular latitude-longitude "
                           "grid: a region cannot be drawn on it")
    # domain area is in square radians
    area_km2 = np.asarray(dom["area"], float) * EARTH_RADIUS_KM ** 2
    return G.Grid(lat=lat, lon=lon, area_km2=area_km2,
                  landfrac=np.asarray(dom["frac"], float), dims=("lat", "lon"),
                  lat_axis=lat_axis, lon_axis=lon_axis)


def region_mask(grid, region: Optional[str], registered: Optional[Dict]) -> np.ndarray:
    """Every cell when region is None; else the cells of a region registered
    for the control (a polygon, or a gridded field and a rule), on every
    cell of it, not only the vegetated ones: the weather falls on all of it."""
    if not region:
        return np.ones(grid.shape, dtype=bool)
    import regions as R
    r = (registered or {}).get(region)
    if not r:
        raise WeatherError(f"region {region!r} is not registered for the control "
                           f"({sorted(registered or {})})")
    if r.get("polygon"):
        m, _ = R.polygon_mask(grid, r["polygon"])
    elif r.get("field"):
        m = R.mask_from_field(r["field"], r["var"], grid, r.get("op", ">"),
                              float(r.get("value", 0.0)), over=r.get("over", "any"))
    else:
        raise WeatherError(f"region {region!r} names neither a polygon nor a field")
    m = np.asarray(m, dtype=bool)
    if not m.any():
        raise WeatherError(f"region {region!r} holds no cell of the grid")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# THE FILES
# ─────────────────────────────────────────────────────────────────────────────
def month_starts(years: Sequence[int]):
    """(t, ym, y0): one record per month from the year before the run's first
    to the year after its last; t is each month's first instant in days since
    1 January of y0, that first year (noleap), ym its (year, month)."""
    import gridded as G
    y0, y1 = min(years) - 1, max(years) + 1
    t, ym, day = [], [], 0.0
    for y in range(y0, y1 + 1):
        for m in ALL_MONTHS:
            t.append(day)
            ym.append((y, m))
            day += G.days_in_month(y, m, "noleap")
    return np.asarray(t, dtype=np.float64), ym, y0


def write_anomaly(path, dom: Dict[str, np.ndarray], params: Dict[str, Any],
                  years: Sequence[int], mask: np.ndarray) -> Dict[str, Any]:
    """The anomaly file: tas (K added) and pr (factor) on the domain's grid,
    one record per month; the change in the chosen months and cells, the
    identity (0, 1) everywhere else."""
    import netCDF4
    t, ym, y0 = month_starts(years)
    months = set(params.get("months") or ALL_MONTHS)
    on = np.array([m in months for _, m in ym])
    shape = dom["xc"].shape
    if mask.shape != shape:
        raise WeatherError(f"the region's mask is {mask.shape}, the grid {shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as f:
        f.createDimension("time", None)
        f.createDimension("nj", shape[0])
        f.createDimension("ni", shape[1])
        tv = f.createVariable("time", "f8", ("time",))
        tv.units = f"days since {y0:04d}-01-01 00:00:00"
        tv.calendar = "noleap"
        tv.long_name = "the first instant of each month"
        tv[:] = t
        for v in ("xc", "yc", "area", "mask"):
            x = f.createVariable(v, "f8" if v != "mask" else "i4", ("nj", "ni"))
            x[:] = dom[v]
        f["xc"].units, f["yc"].units = "degrees_east", "degrees_north"
        f["area"].units = "radian^2"
        for key, a in ANOMALY.items():
            val = float(params.get(key, a["identity"]))
            x = f.createVariable(a["var"], "f4", ("time", "nj", "ni"))
            x.units = a["units"]
            x.long_name = (f"{a['how']} applied to the data atmosphere's "
                           f"{'temperature' if key == 'temperature_K' else 'precipitation'}")
            field = np.where(mask, val, a["identity"]).astype(np.float32)
            ident = np.full(shape, a["identity"], dtype=np.float32)
            x[:] = np.stack([field if o else ident for o in on])
        f.title = "a whole-domain run's anomaly forcing (domain_weather.py)"
        f.params = json.dumps({k: params.get(k) for k in ("temperature_K", "precip_factor",
                                                          "months", "region")})
    return {"path": str(path), "sha256": sha256(path), "records": int(t.size),
            "years": [ym[0][0], ym[-1][0]]}


def stream_text(stream: str, data_dir, data_file: str) -> str:
    """The user stream file through which DATM reads one anomaly field: the
    file describes its own grid (xc, yc, area, mask on the model's cells),
    and every record takes effect LEAD_SECONDS before its month begins."""
    var, model_var = next((a["var"], a["model_var"]) for a in ANOMALY.values()
                          if a["stream"] == stream)
    return f"""<?xml version="1.0"?>
<file id="stream" version="1.0">
<dataSource>
   GENERIC
</dataSource>
<domainInfo>
  <variableNames>
     time    time
     xc      lon
     yc      lat
     area    area
     mask    mask
  </variableNames>
  <filePath>
     {data_dir}
  </filePath>
  <fileNames>
     {data_file}
  </fileNames>
</domainInfo>
<fieldInfo>
   <variableNames>
     {var}   {model_var}
   </variableNames>
   <filePath>
     {data_dir}
   </filePath>
   <fileNames>
     {data_file}
   </fileNames>
   <offset>
      -{LEAD_SECONDS}
   </offset>
</fieldInfo>
</file>
"""


def _quoted(values: Sequence[str]) -> str:
    return ", ".join(f'"{v}"' for v in values)


def datm_lines(control_case_dir, streams: Sequence[str]) -> List[tuple]:
    """(component, key, value) for user_nl_datm: the anomaly streams, and
    tintalgo as the control's list with "lower" for each of them (the
    namelist takes the whole list, so the control's entries are kept)."""
    p = Path(control_case_dir) / "CaseDocs" / "datm_in"
    nl = control_datm(control_case_dir) if p.is_file() else {}
    tint = [str(x) for x in (nl.get("tintalgo") or [])]
    n = len(nl.get("streams") or [])
    if not tint or len(tint) != n:
        raise WeatherError(f"the control's datm_in has {len(tint)} tintalgo entries for "
                           f"{n} streams: cannot extend it")
    return [("datm", "anomaly_forcing", ", ".join(f"'{s}'" for s in streams)),
            ("datm", "tintalgo", _quoted(tint + [INTERP] * len(streams)))]


def settings_expected(control_case_dir, streams: Sequence[str]) -> Dict[str, Dict]:
    """What the run's resolved datm_in must hold beside the control's: the
    anomaly streams named, and every per-stream list grown by them alone."""
    nl = control_datm(control_case_dir)
    n = len(streams)
    want: Dict[str, Any] = {
        "anomaly_forcing": list(streams),
        "streams": {"appends": n, "starts": [f"datm.streams.txt.{s}" for s in streams]},
        "tintalgo": {"appends": n, "values": [INTERP] * n}}
    for k in PER_STREAM_KEYS:
        if k != "tintalgo" and k in nl:
            want[k] = {"appends": n}
    return {"datm_in": want}


def inputs(params: Dict[str, Any], years: Sequence[int], inputs_dir,
           control_case_dir, control_entry: Optional[Dict] = None) -> Dict[str, Any]:
    """Everything one run's weather_scale change needs, written under
    inputs_dir/weather/: the anomaly file (named by its content) and the user
    stream files. Returns the recipe's user files, the user_nl_datm lines,
    and the checks."""
    on = active(params)
    if not on:
        raise WeatherError("weather_scale changes nothing: temperature_K is 0 and "
                           "precip_factor is 1")
    bad = year_problem(years)
    if bad:
        raise WeatherError(bad)
    have = [str(s) for s in control_datm(control_case_dir).get("anomaly_forcing") or []
            if str(s).strip()]
    if have:
        raise WeatherError(f"the control's data atmosphere already reads anomaly streams "
                           f"({have}): weather_scale would replace them")
    dom = read_domain(domain_file(control_case_dir))
    if dom["xc"].shape[0] == 1:
        raise WeatherError("the control's domain is one row of cells, which ELM writes "
                           "as an unstructured (lndgrid) history; the year check reads a "
                           "regular latitude-longitude grid only")
    region = params.get("region")
    mask = (region_mask(grid_from_domain(dom), region, (control_entry or {}).get("regions"))
            if region else np.ones(dom["xc"].shape, dtype=bool))
    mask = mask & (np.asarray(dom["mask"]) == 1)
    if not mask.any():
        raise WeatherError("weather_scale reaches no active cell of the control's domain"
                           + (f" in region {region!r}" if region else ""))
    wdir = Path(inputs_dir).resolve() / "weather"
    tmp = wdir / f"{FILE_STEM}.writing.nc"
    anom = write_anomaly(tmp, dom, params, years, mask)
    final = wdir / f"{FILE_STEM}_{anom['sha256'][:12]}.nc"
    os.replace(tmp, final)
    anom["path"] = str(final)
    streams = [ANOMALY[k]["stream"] for k in on]
    user_files = {}
    for s in streams:
        p = wdir / f"user_datm.streams.txt.{s}"
        p.write_text(stream_text(s, wdir, final.name))
        user_files[p.name] = {"from": str(p), "sha256": sha256(p)}
    # Both fields are checked every year: the file holds the identity for a
    # parameter left at it, so its field must come out the control's. Rain
    # and snow are compared one by one, so a change of phase is caught.
    output = [{"stream": "h1", "fields": a["history"], "rule": "matches_anomaly",
               "how": a["how"], "anomaly_file": anom["path"],
               "anomaly_sha256": anom["sha256"], "anomaly_var": a["var"],
               "tolerance": a["tolerance"], "floor": PRECIP_FLOOR,
               "why": a["why"] if k in on else a["unchanged"]} for k, a in ANOMALY.items()]
    return {"anomaly": dict(anom, streams=streams, cells=int(mask.sum()),
                            region=params.get("region")),
            "user_files": user_files,
            "user_nl": datm_lines(control_case_dir, streams),
            "settings": settings_expected(control_case_dir, streams),
            "case_files": {f"CaseDocs/{n[len('user_'):]}": v["sha256"]
                           for n, v in user_files.items()},
            "output": output}


# ─────────────────────────────────────────────────────────────────────────────
# THE CHECK, ON A YEAR OF OUTPUT (run_gate's matches_anomaly rule)
# ─────────────────────────────────────────────────────────────────────────────
def anomaly_by_month(path, var: str, year: int):
    """{month: field (nj, ni)} of the anomaly file's records in `year`, and
    the file's xc, yc. The records must be the consecutive month starts
    write_anomaly writes; any other time axis is refused, not guessed."""
    import netCDF4
    import gridded as G
    with netCDF4.Dataset(path) as f:
        t = np.asarray(f["time"][:], float)
        base = int(str(f["time"].units).split("since")[1].strip()[:4])
        data = np.asarray(f[var][:], float)
        xc, yc = np.asarray(f["xc"][:], float), np.asarray(f["yc"][:], float)
    out, day = {}, 0.0
    for k in range(t.size):
        y, m = base + k // 12, k % 12 + 1
        if abs(t[k] - day) > 1e-6:
            raise WeatherError(f"record {k} of {Path(path).name} is not the first "
                               f"instant of {y}-{m:02d}")
        if y == year:
            out[m] = data[k]
        day += G.days_in_month(y, m, "noleap")
    return out, xc, yc


def check_year(run_stream, ref_stream, rule: Dict[str, Any], year: int) -> Dict[str, Any]:
    """One year of a run's daily history against the control's, under the
    anomaly file the run was built with, field by field.

    how=offset: the run minus the control equals the file's amount.
    how=factor: the run over the control equals the file's factor wherever
    the control's day is wetter than the floor; on a drier day the run stays
    at or below the floor times the factor (at least the floor itself).
    Each field is compared on its own (rain and snow each), so a change of
    phase is caught.

    Fails, saying why (`reason`), when a cell-day has a value in one run and
    not the other, when nothing could be compared, when the file changes the
    field this year but no compared cell-day falls where it does, or when the
    difference exceeds the tolerance."""
    fields = list(rule["fields"])
    offset = rule["how"] == "offset"
    keys = None
    run_data, ref_data = {}, {}
    for f in fields:
        r_blk, c_blk = run_stream.read(f, year), ref_stream.read(f, year)
        rk, ck = [tuple(k) for k in r_blk.keys], [tuple(k) for k in c_blk.keys]
        if rk != ck:
            raise WeatherError(f"the run's and the control's {year} records of {f} differ "
                               f"({len(rk)} against {len(ck)})")
        if keys is None:
            keys = rk
        elif rk != keys:
            raise WeatherError(f"{f}'s {year} records differ from {fields[0]}'s")
        run_data[f], ref_data[f] = r_blk.data.astype(float), c_blk.data.astype(float)
    by_month, xc, yc = anomaly_by_month(rule["anomaly_file"], rule["anomaly_var"], year)
    g = run_stream.grid
    lon = np.where(xc > 180.0, xc - 360.0, xc)
    glon = np.where(g.lon > 180.0, g.lon - 360.0, g.lon)
    if g.lat.shape != yc.shape or not (
            np.allclose(g.lat, yc, rtol=0.0, atol=CELL_MATCH_DEG)
            and np.allclose(glon, lon, rtol=0.0, atol=CELL_MATCH_DEG)):
        raise WeatherError("the anomaly file's cells are not the run's history cells")
    missing = sorted({int(k[1]) for k in keys} - set(by_month))
    if missing:
        raise WeatherError(f"the anomaly file holds no {year} record for months {missing}")
    exp = np.stack([by_month[int(k[1])] for k in keys])
    ident = 0.0 if offset else 1.0
    changes = bool(np.any(exp != ident))
    tol = float(rule.get("tolerance", 0.0))
    floor = float(rule.get("floor", PRECIP_FLOOR))
    per: Dict[str, Dict[str, Any]] = {}
    for f in fields:
        a, b = run_data[f], ref_data[f]
        fa, fb = np.isfinite(a), np.isfinite(b)
        ok = fa & fb
        row: Dict[str, Any] = {"unpaired": int((fa != fb).sum())}
        if offset:
            err = np.abs((a - b) - exp)[ok]
            row.update(compared=int(ok.sum()), changed_compared=int((ok & (exp != ident)).sum()),
                       max_error=float(err.max()) if err.size else None)
        else:
            wet = ok & (b > floor)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(wet, a / np.where(wet, b, 1.0), np.nan)
            # a factor of 0 leaves nothing to divide by: its error is the
            # share of the control's precipitation the run kept
            err = (np.abs(ratio - exp) / np.where(exp > 0, exp, 1.0))[wet]
            dry_bad = ok & ~wet & (a > floor * np.maximum(exp, 1.0) + 1e-12)
            row.update(compared=int(wet.sum()), changed_compared=int((wet & (exp != ident)).sum()),
                       max_relative_error=float(err.max()) if err.size else None,
                       dry_days_with_precipitation=int(dry_bad.sum()))
        per[f] = row
    measure = "max_error" if offset else "max_relative_error"
    worsts = [r[measure] for r in per.values() if r[measure] is not None]
    worst = max(worsts) if worsts else None
    out: Dict[str, Any] = {
        "how": rule["how"], "fields": fields, "tolerance": tol, measure: worst,
        "compared": sum(r["compared"] for r in per.values()),
        "changed_compared": sum(r["changed_compared"] for r in per.values()),
        "unpaired": sum(r["unpaired"] for r in per.values()), "per_field": per}
    if not offset:
        out["dry_days_with_precipitation"] = sum(r["dry_days_with_precipitation"]
                                                 for r in per.values())
    reasons = []
    if out["unpaired"]:
        reasons.append(f"{out['unpaired']} cell-day(s) have a value in one run and not the "
                       f"other (" + ", ".join(f"{f} {r['unpaired']}" for f, r in per.items()
                                             if r["unpaired"]) + ")")
    if not out["compared"]:
        reasons.append(f"no cell-day of {year} could be compared")
    elif changes and not out["changed_compared"]:
        reasons.append(f"the file changes {'+'.join(fields)} in {year}, but no compared "
                       f"cell-day falls where it does: the change reached no cell the "
                       f"history reports")
    if worst is not None and worst > tol:
        reasons.append(f"{'off by' if offset else 'relative error'} up to {worst:.3g} "
                       f"(tolerance {tol:g})" + ("" if offset else ", " + ", ".join(
                           f"{f} {r[measure]:.3g}" for f, r in per.items()
                           if r[measure] is not None)))
    if not offset and out["dry_days_with_precipitation"]:
        reasons.append(f"{out['dry_days_with_precipitation']} dry cell-day(s) of the "
                       f"control have precipitation in the run")
    out["pass"] = not reasons
    if reasons:
        out["reason"] = "; ".join(reasons)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# REPEATING A YEAR (weather_repeat_year)
# ─────────────────────────────────────────────────────────────────────────────
# A model year reads another recorded year's weather when its job sets which
# years of the forcing DATM reads (DATM_CLMNCEP_YR_START, _END, _ALIGN) before
# it runs: each chain year is its own job, and CIME writes the namelists and
# the stream's file list anew when a run starts. No file is written for the
# run. Read in E3SM's code, and tested on a clone of the Naches control
# (2026-10-26, job 791045: three 5-day runs restarted on 6 and 11 January
# 2015, inside one year; records in workflow_outputs/elm_restart_test_20261026):
#
#   - after a restart DATM takes the years from the namelist, not from its
#     restart file: shr_stream_restRead reads the stored years and discards
#     them. It restores its stored record positions unless a file whose times
#     it had read has another name at the same place in the new list, and
#     finds them afresh otherwise. Both paths were run; each run's largest
#     daily difference from the control's recorded year (0.36 K and 0.074 K
#     in TBOT) is what one carried step on the first day gives.
#   - the first land step after a restart takes the atmosphere's last output
#     from the coupler's restart (driver-mct/main/seq_rest_mod.F90, a2x_ax),
#     which the previous job computed at the restart instant with ITS years.
#     At a year boundary that is the record for 00:00 on 1 January of the
#     year after the previous year's recorded year: for 2016 reading 2015
#     after 2015, 2016's own first record (the control's value), not 2015's.
#     One model step (30 minutes on the Naches) is affected; 1 January of
#     such a year is reported, not held to the control's (carried_first_days).
#   - restored positions are wrong when the new file list starts with the old
#     one but is shorter (they point past its end) or longer (the old end caps
#     it, and the year wraps without an error). So every year of a run reads
#     the same two-year form: its recorded year W and W+1, aligned so that the
#     model year reads W. Two years' lists then have the same length, and
#     differ in their first file (always read) unless they are the same list.
#     The year after W also closes W's December as the control's run does.
#
# The form needs a forcing whose first record of each year is at 00:00 on 1
# January (NLDAS, as the Naches control reads). Where the first record comes
# later (GSWP3 at 01:30, CRUNCEP at 03:00), DATM would take the hours before
# it from the end of W+1 (findBounds' cycle branch), and every 1 January would
# differ from the control's: such a forcing is refused (a three-year form,
# W-1 to W+1, would serve it, after its own proof).
YEAR_KEYS = ("DATM_CLMNCEP_YR_START", "DATM_CLMNCEP_YR_END", "DATM_CLMNCEP_YR_ALIGN")
REPEAT_FIELDS = ("TBOT", "RAIN", "SNOW")
_STRDATA_STREAM = re.compile(r"\(shr_strdata_print\)\s+streams\s*\(\s*\d+\)\s*=\s*(\S+)"
                             r"\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)")


def year_map(params: Dict[str, Any], years: Sequence[int]) -> Dict[int, int]:
    """{model year: the recorded year it reads} for every year of the run; a
    year the map does not name reads its own."""
    m = {int(k): int(v) for k, v in (params.get("map") or {}).items()}
    return {int(y): m.get(int(y), int(y)) for y in years}


def repeat_settings(ymap: Dict[int, int]) -> Dict[int, List[List[Any]]]:
    """Each model year's DATM settings: its recorded year and the one after,
    aligned so that the model year reads its recorded year (the two-year
    form above)."""
    return {y: [["DATM_CLMNCEP_YR_START", w], ["DATM_CLMNCEP_YR_END", w + 1],
                ["DATM_CLMNCEP_YR_ALIGN", y]] for y, w in sorted(ymap.items())}


def carried_first_days(ymap: Dict[int, int]) -> List[int]:
    """Model years whose first step has another year's weather: the previous
    job's last call read the record after its recorded year, and this year's
    recorded year is not that one. The run's first year starts afresh."""
    ys = sorted(ymap)
    return [y for prev, y in zip(ys, ys[1:]) if ymap[y] != ymap[prev] + 1]


def _case_value(case_dir: Path, name: str) -> Optional[str]:
    """A case setting as its env files hold it, with CIME's unresolved
    references ($NAME or ${NAME}, as compset defaults are stored) followed."""
    import domain_case as DC
    v = DC._env_value(case_dir, name)
    for _ in range(5):
        m = re.fullmatch(r"\$\{?(\w+)\}?", str(v or "").strip())
        if not m:
            break
        v = DC._env_value(case_dir, m.group(1))
    return v


def forcing_streams(control_case_dir) -> Dict[str, Any]:
    """The control's forcing years and the stream files they set, checked
    for what the two-year form needs: model years that are recorded years
    (ALIGN equal to START), a file list CIME makes from the years (no user
    stream file), no bias correction or anomaly forcing on the model year,
    and each year's first record at 00:00 on 1 January."""
    cd = Path(control_case_dir)
    raw = {k: _case_value(cd, k) for k in YEAR_KEYS}
    cur = {k: _int(v) for k, v in raw.items()}
    if any(v is None for v in cur.values()):
        raise WeatherError("the control's case does not say, as whole years, which forcing "
                           "years it reads: " + ", ".join(f"{k}={raw[k]!r}" for k in YEAR_KEYS))
    align, first, last = (cur["DATM_CLMNCEP_YR_ALIGN"], cur["DATM_CLMNCEP_YR_START"],
                          cur["DATM_CLMNCEP_YR_END"])
    if align != first:
        raise WeatherError(f"the control's model year Y reads recorded year {first} + "
                           f"(Y - {align}), not Y, so its output years are not its "
                           f"recorded years (DATM_CLMNCEP_YR_ALIGN {align}, _START {first})")
    nl = control_datm(cd)
    names = []
    for entry in nl.get("streams") or []:
        parts = str(entry).split()
        if len(parts) == 4 and [_int(x) for x in parts[1:]] == [align, first, last]:
            names.append(parts[0])
    if not names:
        raise WeatherError(f"no stream of the control's datm_in reads the forcing years its "
                           f"case sets ({align} {first} {last})")
    for n in names:
        if (cd / f"user_{n}").exists():
            raise WeatherError(f"the control's case holds user_{n}: its file list stays fixed "
                               f"whatever years are set, so a year cannot be repeated")
    nl_user = cd / "user_nl_datm"
    if nl_user.is_file() and re.search(r"^\s*streams\s*=", nl_user.read_text(errors="replace"),
                                       re.M | re.I):
        raise WeatherError("the control's user_nl_datm sets the streams itself, so the "
                           "forcing years each year's job sets would not take effect")
    for key in ("bias_correct", "anomaly_forcing"):
        if [str(s) for s in nl.get(key) or [] if str(s).strip() and str(s) != "null"]:
            raise WeatherError(f"the control's data atmosphere also reads {key} streams, "
                               f"which stay on the model year: a repeated year would mix "
                               f"two years")
    for n in names:
        late = _first_record_after_midnight(cd / "CaseDocs" / n)
        if late:
            raise WeatherError(f"{n}'s records of each year start at {late} on 1 January, "
                               f"not 00:00: DATM would take the hours before from the end of "
                               f"the next recorded year, so every 1 January would differ")
    return {"align": align, "first": first, "last": last, "streams": names}


def _first_record_after_midnight(stream_file: Path) -> Optional[str]:
    """When a stream's first record of its first file falls after 00:00 on
    that day (its time plus the stream's <offset>), as HH:MM; else None."""
    import xml.etree.ElementTree as ET
    import netCDF4
    if not stream_file.is_file():
        raise WeatherError(f"{stream_file} is missing: the forcing's records cannot be read")
    root = ET.fromstring(stream_file.read_text(errors="replace"))
    fi = root.find("fieldInfo")
    path = (fi.find("filePath").text or "").strip()
    files = (fi.find("fileNames").text or "").split()
    off = _int(((fi.find("offset").text if fi.find("offset") is not None else "0") or "0")
               .strip()) or 0
    if not files:
        raise WeatherError(f"{stream_file.name} names no forcing file")
    p = Path(path) / files[0]
    try:
        with netCDF4.Dataset(p) as f:
            t0 = float(np.asarray(f["time"][:1], float)[0])
            units = str(f["time"].units)
    except (OSError, KeyError, IndexError) as e:
        raise WeatherError(f"cannot read the first record of {p.name}: {e}") from None
    m = re.match(r"\s*(days|hours|minutes|seconds)\s+since\s+\S+(?:[ T](\d+):(\d+)(?::(\d+))?)?",
                 units)
    if not m:
        raise WeatherError(f"{p.name}: time units {units!r} are not read here")
    scale = {"days": 86400.0, "hours": 3600.0, "minutes": 60.0, "seconds": 1.0}[m.group(1)]
    base = sum(int(x or 0) * s for x, s in zip(m.groups()[1:], (3600, 60, 1)))
    secs = (t0 * scale + base + off) % 86400.0
    if abs(secs) < 1e-3 or abs(secs - 86400.0) < 1e-3:
        return None
    return f"{int(secs // 3600):02d}:{int(secs % 3600 // 60):02d}"


def _int(x) -> Optional[int]:
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


def repeat_inputs(params: Dict[str, Any], years: Sequence[int], control_case_dir,
                  control_entry: Optional[Dict] = None) -> Dict[str, Any]:
    """Everything one run's weather_repeat_year change needs. No file is
    written for the run: each year's DATM settings go to the chain's year
    jobs (year_settings), and two rules check every year: the forcing years
    DATM's log says the year's run read, and its daily TBOT, RAIN and SNOW
    against the control's for the recorded year."""
    ymap = year_map(params, years)
    if all(w == y for y, w in ymap.items()):
        raise WeatherError("weather_repeat_year changes nothing: every model year reads "
                           "its own year")
    f = forcing_streams(control_case_dir)
    bad = sorted({w for w in ymap.values() if not f["first"] <= w < f["last"]})
    if bad:
        raise WeatherError(f"the control's forcing holds {f['first']}-{f['last']}, and a "
                           f"year read needs the year after it there too: {bad} do not have "
                           f"it")
    settings = repeat_settings(ymap)
    carried = carried_first_days(ymap)
    rules = [
        {"stream": "datm_in", "rule": "datm_stream_years", "streams": f["streams"],
         "years": {str(y): [y, w, w + 1] for y, w in ymap.items()},
         "why": "DATM's log for the year's job names the forcing with the year's recorded "
                "year and the next, aligned so the model year reads it"},
        {"stream": "h1", "rule": "equals_control_year", "fields": list(REPEAT_FIELDS),
         "years": {str(y): w for y, w in ymap.items()}, "carried_first_days": carried,
         "tolerance": 0.0,
         "why": "daily TBOT, RAIN and SNOW are the control's for the recorded year, day by "
                "day and cell by cell (1 January of a year whose recorded year does not "
                "follow the previous year's is reported, not held)"}]
    return {"repeat": {"map": {str(y): w for y, w in ymap.items()}, "carried_first_days":
                       carried, "streams": f["streams"]},
            "year_settings": {str(y): v for y, v in settings.items()},
            "settings": {}, "output": rules}


def _recorded(rule: Dict[str, Any], year: int):
    try:
        return rule["years"][str(year)]
    except KeyError:
        raise WeatherError(f"the rule names no recorded year for model year {year}") from None


def check_stream_years(run_dir, rule: Dict[str, Any], year: int, job=None) -> Dict[str, Any]:
    """The forcing years the year's run read, from DATM's own log for the
    year's job (atm.log.<job>.*: its "streams" lines): each forcing stream
    named with the expected alignment and years. The run directory's datm_in
    is not used: the next year's job writes it anew."""
    from run_gate import _job_file, _read_text
    want = [int(x) for x in _recorded(rule, year)]
    if not job:
        raise WeatherError("no job id: the year's DATM log cannot be found")
    log = _job_file(Path(run_dir), "atm.log", str(job))
    if log is None:
        raise WeatherError(f"no atm.log.{job}.* in {run_dir}: the year's forcing years "
                           f"cannot be read")
    got: Dict[str, List[int]] = {}
    for m in _STRDATA_STREAM.finditer(_read_text(log)):
        got[m.group(1)] = [int(m.group(i)) for i in (2, 3, 4)]
    seen = {n: got.get(n) for n in rule["streams"]}
    wrong = {n: v for n, v in seen.items() if v != want}
    out: Dict[str, Any] = {"expected": want, "streams": seen, "log": log.name,
                           "pass": not wrong}
    if wrong:
        out["reason"] = "; ".join(
            f"{n} reads {' '.join(map(str, v)) if v else 'nothing'}, not "
            f"{' '.join(map(str, want))} (align, first, last)" for n, v in wrong.items())
    return out


def check_control_year(run_stream, ref_stream, rule: Dict[str, Any], year: int
                       ) -> Dict[str, Any]:
    """One year of a run's daily history against the control's for the
    recorded year it reads, field by field, day of the year by day of the
    year: equal within the tolerance (0: exactly), every cell-day reported by
    both. 1 January of a carried year is reported apart."""
    w = int(_recorded(rule, year))
    carried = year in [int(y) for y in rule.get("carried_first_days") or []]
    tol = float(rule.get("tolerance", 0.0))
    per: Dict[str, Dict[str, Any]] = {}
    for f in rule["fields"]:
        a_blk, b_blk = run_stream.read(f, year), ref_stream.read(f, w)
        ka = [tuple(k)[1:] for k in a_blk.keys]
        kb = [tuple(k)[1:] for k in b_blk.keys]
        if ka != kb:
            raise WeatherError(f"the run's {year} records of {f} are not the control's "
                               f"{w} records day for day ({len(ka)} against {len(kb)})")
        a, b = a_blk.data.astype(float), b_blk.data.astype(float)
        fa, fb = np.isfinite(a), np.isfinite(b)
        ok = fa & fb
        first = np.array([k[0] == 1 and k[1] == 1 for k in ka])
        first = first.reshape((-1,) + (1,) * (a.ndim - 1))
        held = ok & ~(first if carried else np.zeros_like(first))
        diff = np.abs(a - b)
        row: Dict[str, Any] = {"unpaired": int((fa != fb).sum()), "compared": int(held.sum()),
                               "max_abs_difference": float(diff[held].max())
                               if held.any() else None}
        if carried:
            fd = ok & first
            row["first_day_max_abs_difference"] = float(diff[fd].max()) if fd.any() else None
        per[f] = row
    worsts = [r["max_abs_difference"] for r in per.values()
              if r["max_abs_difference"] is not None]
    worst = max(worsts) if worsts else None
    out: Dict[str, Any] = {"how": "equals", "fields": list(rule["fields"]), "reads": w,
                           "tolerance": tol, "max_abs_difference": worst,
                           "compared": sum(r["compared"] for r in per.values()),
                           "unpaired": sum(r["unpaired"] for r in per.values()),
                           "first_day_carried": carried, "per_field": per}
    reasons = []
    if out["unpaired"]:
        reasons.append(f"{out['unpaired']} cell-day(s) have a value in one run and not the "
                       f"other (" + ", ".join(f"{f} {r['unpaired']}" for f, r in per.items()
                                             if r["unpaired"]) + ")")
    if not out["compared"]:
        reasons.append(f"no cell-day of {year} could be compared with the control's {w}")
    if worst is not None and worst > tol:
        reasons.append(f"differs from the control's {w} by up to {worst:.3g} "
                       f"(tolerance {tol:g}): " + ", ".join(
                           f"{f} {r['max_abs_difference']:.3g}" for f, r in per.items()
                           if r["max_abs_difference"] is not None))
    out["pass"] = not reasons
    if reasons:
        out["reason"] = "; ".join(reasons)
    return out
