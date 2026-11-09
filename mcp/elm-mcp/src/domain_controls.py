#!/usr/bin/env python3
"""The prepared whole-domain controls this machine has, and each one's facts.

A control is a finished whole-domain run that experiments are built on: the
case its recipe is read from, the run directory its output is in, how it was
shown to reproduce, and the regions and gridded observations prepared for its
grid (docs/ELM_DOMAIN_EXPERIMENTS.md). The list is a local JSON file, found
like the other bulk data (paths.py key `domain_controls`), and git-ignored
because it names local paths; domain_controls.json.example shows its shape.

  load()            the registry, with every path checked
  facts(name)       the control's facts, from the cache beside the registry
  compute_facts()   read them from the control's own output (minutes)
  setup_facts()     its fire, grid and configuration, from its case's files
                    (seconds), each with a sentence to quote
  summary(entry)    what reception and the planner are shown, kept small

THE FACTS ARE MEASURED, NEVER WRITTEN BY HAND. They are what keeps a design
honest about its control: that its fire is the model's own and burns where no
observed fire did, which years its weather covers, which fields it wrote. A
design that reads them cannot describe a no-fire twin as an observed fire's
effect.
"""
from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

FACTS_DIR = "domain_control_facts"
# The year facts reported per control, from run_gate.year_check.
YEAR_FACTS = ("gpp_gC_m2_yr", "peak_tlai", "precip_mm", "et_mm", "runoff_mm",
              "swe_apr1_mm", "water_closure_frac_of_precip", "totecosysc_change_frac")
# The settings a change may alter, read from the control's resolved namelist.
NAMELIST_SETTINGS = ("use_nofire", "co2_ppmv", "co2_type", "stream_year_first_ndep",
                     "stream_year_last_ndep", "suplnitro", "suplphos")
XML_SETTINGS = ("CCSM_CO2_PPMV", "ELM_CO2_TYPE", "DATM_MODE", "DATM_CLMNCEP_YR_ALIGN",
                "DATM_CLMNCEP_YR_START", "DATM_CLMNCEP_YR_END", "RUN_STARTDATE")
REQUIRED = ("case_dir", "run_dir")


class ControlError(Exception):
    """A control cannot be used, and the message says why."""


def _path(override: Optional[str] = None):
    import paths
    return paths.resolve("domain_controls", override)


def load(override: Optional[str] = None) -> Dict[str, Any]:
    """The registry: {"path", "source", "controls": {name: entry}, "problems"}.
    A missing file is not an error: it is a machine with no prepared control."""
    p, source = _path(override)
    out: Dict[str, Any] = {"path": str(p), "source": source, "controls": {},
                           "problems": []}
    if not p.is_file():
        out["problems"].append(f"no registry at {p}: this machine has no prepared "
                               f"whole-domain control (see domain_controls.json.example)")
        return out
    try:
        d = json.loads(p.read_text())
    except ValueError as e:
        out["problems"].append(f"{p} cannot be read: {e}")
        return out
    for name, e in sorted((d.get("controls") or {}).items()):
        bad = [f"{name}: no {k}" for k in REQUIRED if not e.get(k)]
        for k in REQUIRED:
            if e.get(k) and not Path(e[k]).is_dir():
                bad.append(f"{name}: {k} {e[k]} is not a directory")
        for kind in ("regions", "observations"):
            for rn, r in (e.get(kind) or {}).items():
                f = r.get("polygon") or r.get("field") or r.get("path")
                if not f or not Path(f).is_file():
                    bad.append(f"{name}: {kind} {rn}: {f} is missing")
        if bad:
            out["problems"] += bad
        out["controls"][name] = dict(e, name=name, usable=not bad)
    return out


def get(name: str, override: Optional[str] = None) -> Dict[str, Any]:
    reg = load(override)
    if name not in reg["controls"]:
        raise ControlError(f"no prepared control {name!r}; this machine has "
                           f"{sorted(reg['controls']) or 'none'}"
                           + (f" ({'; '.join(reg['problems'])})" if reg["problems"] else ""))
    e = reg["controls"][name]
    if not e["usable"]:
        raise ControlError(f"control {name} cannot be used: "
                           + "; ".join(p for p in reg["problems"] if p.startswith(name)))
    return dict(e, registry=reg["path"])


def _facts_path(entry: Dict[str, Any]) -> Path:
    return Path(entry["registry"]).parent / FACTS_DIR / f"{entry['name']}.json"


def facts(name: str, override: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The cached facts of a control, or None when they were never computed or
    the control's output changed since (a history file newer than the cache)."""
    e = get(name, override)
    p = _facts_path(e)
    if not p.is_file():
        return None
    try:
        f = json.loads(p.read_text())
    except ValueError:
        return None
    newest = max((q.stat().st_mtime for q in Path(e["run_dir"]).glob("*.elm.h*.nc")),
                 default=0.0)
    if newest > p.stat().st_mtime:
        return None
    if f.get("version") != FACTS_VERSION or _setup_failed(f):
        f = _upgrade(f, e, p)
    return f


SETUP_PARTS = ("fire", "grid", "configuration")


def _setup_failed(f: Dict[str, Any]) -> bool:
    """True when a setup part is missing or holds an error."""
    return any(not isinstance(f.get(k), dict) or f[k].get("error") for k in SETUP_PARTS)


# A SETUP PART THAT FAILED IS READ AGAIN; A CACHE THAT CANNOT BE WRITTEN IS
# NOT AN ERROR (2026-11-03). The setup is read in about 2 s, so a part that
# could not be read (lnd_in briefly away, inputdata not mounted) is not
# written down: the cache keeps its old content and the next call reads the
# case again, where a stored error would stay until the full refresh. A cache
# folder the server may not write (a shared or read-only install, a full
# disk) leaves the facts in memory only; the caller still gets them, so
# describe_domain_controls does not report the control as unusable. Each
# process writes its own temporary file, so two servers upgrading at once
# cannot rename each other's.
def _upgrade(f: Dict[str, Any], e: Dict[str, Any], p: Path) -> Optional[Dict[str, Any]]:
    """A cache of an older format brought to FACTS_VERSION. A cache from
    before the version key (format 1) gains the setup parts, read from the
    case in seconds; its year facts are kept, since their format did not
    change. A cache of this format whose setup parts failed reads them again.
    The rewritten cache keeps its old modification time, so a history file
    written after the year facts still makes it stale; it is rewritten only
    when every setup part was read. A format this code does not know is not
    used: refresh_domain_control_facts."""
    import os
    if f.get("version") not in (None, 1, FACTS_VERSION):
        return None
    f = json.loads(json.dumps(dict(f, version=FACTS_VERSION, **setup_facts(e["case_dir"])),
                              default=str))
    if _setup_failed(f):
        return f
    f["setup_computed"] = datetime.datetime.now().isoformat(timespec="seconds")
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    try:
        st = p.stat()
        tmp.write_text(json.dumps(f, indent=1))
        os.utime(tmp, (st.st_atime, st.st_mtime))
        tmp.replace(p)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
    return f


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTING THE FACTS
# ─────────────────────────────────────────────────────────────────────────────
def _settings(case_dir: Path) -> Dict[str, Any]:
    import domain_case as DC
    from run_gate import read_namelist
    nl = read_namelist(case_dir / "CaseDocs" / "lnd_in")
    s: Dict[str, Any] = {k: (nl.get(k) or [None])[0] for k in NAMELIST_SETTINGS}
    for k in XML_SETTINGS:
        s[k] = DC._env_value(case_dir, k)
    m = re.search(r"simyr(\d{4})-(\d{4})", str((nl.get("stream_fldfilename_ndep") or [""])[0]))
    s["ndep_stream_years"] = [int(m.group(1)), int(m.group(2))] if m else None
    s["requested_fields"] = {f"h{k}": list(nl.get(f"hist_fincl{k + 1}") or [])
                             for k in range(6) if nl.get(f"hist_fincl{k + 1}")}
    return s


def _forcing_years(s: Dict[str, Any]) -> Optional[List[int]]:
    try:
        a, b = int(s["DATM_CLMNCEP_YR_START"]), int(s["DATM_CLMNCEP_YR_END"])
        al = int(s.get("DATM_CLMNCEP_YR_ALIGN") or a)
    except (TypeError, ValueError, KeyError):
        return None
    return [al, al + (b - a)]


def burned_area_km2(run_dir, year: int) -> Dict[str, float]:
    """The model's own burned area in one calendar year, from h0 FAREA_BURNED,
    a fraction PER SECOND (gridded.PER_SECOND_DESPITE_UNITS) integrated over
    each record's seconds inside the year, weighted by area x landfrac. The
    run's first model step, folded into its first January, is the year
    before's (as regions.RegionSeries.by_year counts it)."""
    import numpy as np
    import gridded as G
    s = G.open_stream(run_dir, "h0")
    recs, _ = s.year_records(year)
    blk = s.read("FAREA_BURNED", year)
    if not blk.per_second:
        raise ControlError("FAREA_BURNED is not read as a rate per second")
    secs = []
    for r in recs:
        lo, hi = type(r.start)(year, 1, 1), type(r.start)(year + 1, 1, 1)
        secs.append(max(0.0, (min(r.end, hi) - max(r.start, lo)).total_seconds()))
    secs = np.array(secs)
    frac = np.nansum(blk.data * secs[:, None, None], axis=0)
    w = s.grid.weight_km2
    return {"km2": float(np.nansum(frac * w)),
            "cells_over_1pct": int(np.sum(frac > 0.01)),
            "max_cell_fraction": float(np.nanmax(frac))}


def observed_burned_area_km2(path, var: str, grid) -> Dict[int, float]:
    """An observed burned fraction per cell and year on the model grid (the
    regridded MODIS MCD64A1 of the hand experiment), totalled per year."""
    import netCDF4
    import numpy as np
    from gridded import read_clean
    with netCDF4.Dataset(path) as f:
        grid.check_coordinates(read_clean(f["lat"]), read_clean(f["lon"]),
                               what=Path(path).name)
        years = [int(y) for y in f["year"][:]]
        bf = read_clean(f[var])
    w = grid.weight_km2
    return {y: float(np.nansum(np.nan_to_num(bf[k]) * w)) for k, y in enumerate(years)}


def _hours_per_year(entry: Dict[str, Any]) -> Optional[float]:
    """Hours of one node per model year, from the reproduction's STATUS lines."""
    rd = (entry.get("reproduced") or {}).get("record_dir")
    if not rd or not (Path(rd) / "STATUS.log").is_file():
        return None
    vals = [float(m.group(1)) for m in re.finditer(
        r"stage=y\d{4} CHECK \w+ hours_per_sim_year=([0-9.]+)",
        (Path(rd) / "STATUS.log").read_text())]
    return round(sum(vals) / len(vals), 3) if vals else None


# ─────────────────────────────────────────────────────────────────────────────
# THE CONTROL'S SETUP, READ FROM ITS OWN FILES (2026-11-03)
# ─────────────────────────────────────────────────────────────────────────────
# A reader of a report could not tell from it how the control was set up: its
# fire model, its grid, its nutrients, its weather. Those facts were in the
# case, scattered over lnd_in, datm_in, the env XML and the domain and surface
# files, and a person had to type them into the registry's description by
# hand. They are now read from those files, each part with one plain sentence
# the card, the design page and the reports quote word for word.
#
# FACTS_VERSION marks the cache's format. A cache written before these parts
# existed (no version) is upgraded in place by facts(): only the setup parts
# are added, since the year facts it holds have not changed format, and the
# setup is read in seconds where the year facts take minutes.
FACTS_VERSION = 2
# ELM's earth radius (SHR_CONST_REARTH), by which a domain file's area in
# square radians becomes km2, as the model itself does.
EARTH_RADIUS_KM = 6371.22
# Land-namelist keys that set up the fire model.
FIRE_KEY = re.compile(r"fire|popdens|lightng")
# A key that would prescribe a burned area, if a namelist named one.
PRESCRIBED_FIRE_KEY = re.compile(r"burn|farea")
# The fire model's inputs ELM reads from the surface data (CNStateType.F90).
SURFACE_FIRE_FIELDS = {"gdp": "gross domestic product",
                       "peatf": "peatland fraction",
                       "abm": "month of peak crop-residue burning"}
SUPPLEMENT_WORDS = {"NONE": "not supplemented",
                    "ALL": "supplemented to meet plant demand everywhere",
                    "PROG_CROP_ONLY": "supplemented for prognostic crops only"}
NUTRIENT_WORDS = {"cnp": "carbon, nitrogen and phosphorus", "cn": "carbon and nitrogen"}
# THE IDEA BEFORE THE CODE (2026-11-03). The setup sentences are quoted on
# the card, the page and every report, so a scheme's code (RD, CTC) comes
# after what it does, never alone.
NU_COM_WORDS = {"RD": "relative demand", "ECA": "equilibrium chemistry approximation"}
DECOMP_WORDS = {"ctc": "converging trophic cascade", "century": "CENTURY"}


def _first(nl: Dict[str, list], key: str):
    return (nl.get(key) or [None])[0]


def _years(a, b) -> Optional[List[int]]:
    try:
        return [int(a), int(b)]
    except (TypeError, ValueError):
        return None


def _word(v) -> str:
    """A namelist value as a reader sees it: true or false, 367 not 367.0."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _held(ys: Optional[List[int]]) -> str:
    """How a stream's years are used. A single year below 1000 is a stream
    of one year with no calendar date (the lightning climatology's year 1),
    which a reader would otherwise take for a calendar year."""
    if not ys:
        return "for years not stated"
    if ys[0] == ys[1] and ys[0] < 1000:
        return f"one year of data used the same way every year, stream year {ys[0]}"
    return f"held at year {ys[0]}" if ys[0] == ys[1] else f"read for {ys[0]}-{ys[1]}"


def _stream(nl: Dict[str, list], tag: str) -> Dict[str, Any]:
    f = _first(nl, f"stream_fldfilename_{tag}")
    return {"file": str(f) if f else None, "file_name": Path(str(f)).name if f else None,
            "years": _years(_first(nl, f"stream_year_first_{tag}"),
                            _first(nl, f"stream_year_last_{tag}"))}


def _surface_range(surface, fields) -> Dict[str, Any]:
    """min and max of each named surface field, or a note that it is absent."""
    import netCDF4
    import numpy as np
    out: Dict[str, Any] = {}
    with netCDF4.Dataset(str(surface)) as f:
        for v, what in fields.items():
            if v not in f.variables:
                out[v] = {"what": what, "present": False}
                continue
            a = np.ma.filled(np.ma.masked_invalid(f[v][:].astype(float)), np.nan)
            out[v] = {"what": what, "present": True,
                      "min": round(float(np.nanmin(a)), 4),
                      "max": round(float(np.nanmax(a)), 4)}
    return out


def fire_setup(nl: Dict[str, list]) -> Dict[str, Any]:
    """The fire model's setup as the land namelist states it, with the
    surface data's fire inputs, and one sentence a reader can quote."""
    nofire = _first(nl, "use_nofire")
    use_cn, fates = _first(nl, "use_cn"), _first(nl, "use_fates") is True
    spit = _first(nl, "fates_spitfire_mode")
    try:
        spit_on = None if spit is None else int(spit) > 0
    except (TypeError, ValueError):
        spit_on = None
    on = (spit_on if fates else
          nofire is False or (nofire is None and use_cn is True))
    pop, light = _stream(nl, "popdens"), _stream(nl, "lightng")
    for s, tag in ((pop, "popdens"), (light, "lightng")):
        s["map"] = _first(nl, f"{tag}mapalgo")
    out: Dict[str, Any] = {
        "on": on, "use_nofire": nofire,
        "population_density": pop, "lightning": light,
        "namelist": {k: (v[0] if len(v) == 1 else v) for k, v in sorted(nl.items())
                     if FIRE_KEY.search(k)},
        "prescribed_burned_area": sorted(k for k in nl if PRESCRIBED_FIRE_KEY.search(k))}
    if _first(nl, "fsurdat") and Path(str(_first(nl, "fsurdat"))).is_file():
        try:
            out["surface"] = _surface_range(_first(nl, "fsurdat"), SURFACE_FIRE_FIELDS)
        except Exception as e:                                      # noqa: BLE001
            out["surface"] = {"error": f"{type(e).__name__}: {e}"}
    observed = ("no observed fire is prescribed: the land namelist names no burned-area "
                "input" if not out["prescribed_burned_area"] else
                "the land namelist names " + ", ".join(out["prescribed_burned_area"]))
    # WHOSE FIRE, AND ONLY "NOTHING BURNS" WHEN THE CASE SAYS SO (2026-11-03).
    # Under FATES the fire is FATES's own (fates_spitfire_mode), not ELM's.
    # Without the carbon-nutrient cycle ELM's fire model does not run, which
    # is not the same as a switch set off. A namelist that states none of the
    # switches is said to be unread, never fire-free.
    nf = "use_nofire not set" if nofire is None else f"use_nofire {_word(nofire)}"
    if fates:
        out["sentence"] = (
            (f"the fire is FATES's own fire model, not ELM's (fates_spitfire_mode "
             f"{_word(spit)})" if spit_on else
             f"FATES runs the vegetation with its fire model off (fates_spitfire_mode "
             f"{_word(spit)})" if spit_on is False else
             "FATES runs the vegetation and the land namelist does not say whether its "
             "fire model is on (no fates_spitfire_mode)") + f"; {observed}")
    elif nofire is True:
        out["sentence"] = (f"the fire model is off ({nf}), so nothing burns in the run; "
                           f"{observed}")
    elif not on and use_cn is False:
        out["sentence"] = (f"ELM's fire model does not run: it is part of the carbon-nutrient "
                           f"cycle, which is off (use_cn false, {nf}); {observed}")
    elif not on:
        out["sentence"] = (f"the fire setup could not be read: the land namelist sets "
                           f"neither use_nofire nor use_cn; {observed}")
    else:
        lit = (f"lit by lightning from {light['file_name']} ({_held(light['years'])})"
               if light["file_name"] else "with no lightning file named")
        people = (f" with population density {_held(pop['years'])} "
                  f"({pop['file_name']})" if pop["file_name"] else
                  " with no population-density file named")
        out["sentence"] = f"the fire is ELM's own fire model ({nf}), {lit}{people}; {observed}"
    return out


def _spacing(a, axis: int) -> float:
    import numpy as np
    d = np.abs(np.diff(np.asarray(a, dtype=float), axis=axis))
    return float(np.median(d)) if d.size else float("nan")


def _degrees_words(d: float) -> str:
    """1/120 degree when the spacing is one over a whole number, else the value."""
    if not d or d != d:
        return "unknown"
    n = round(1.0 / d)
    return f"1/{n} degree" if n > 1 and abs(1.0 / n - d) <= 1e-3 * d else f"{d:.6g} degree"


def grid_setup(domain, surface=None) -> Dict[str, Any]:
    """The grid as the domain file holds it: rows, columns, cells, land cells,
    the latitude and longitude of the cell centres, the cell size in degrees
    and in km at the box's mean latitude, and the land area (area x frac over
    the land cells, by ELM's earth radius). The surface data must have the
    same rows and columns."""
    import math
    import numpy as np
    import domain_weather as W
    d = W.read_domain(domain)
    xc, yc = np.asarray(d["xc"], float), np.asarray(d["yc"], float)
    mask = np.asarray(d["mask"]) > 0
    nj, ni = xc.shape
    lon = np.where(xc > 180.0, xc - 360.0, xc)
    dx, dy = _spacing(xc, 1), _spacing(yc, 0)
    mean_lat = float(np.mean(yc))
    km_deg = math.pi * EARTH_RADIUS_KM / 180.0
    km_ew, km_ns = dx * km_deg * math.cos(math.radians(mean_lat)), dy * km_deg
    land_km2 = float(np.sum(np.where(mask, d["area"] * d["frac"], 0.0))) * EARTH_RADIUS_KM ** 2
    out: Dict[str, Any] = {
        "file": str(domain), "rows": int(nj), "columns": int(ni), "cells": int(nj * ni),
        "land_cells": int(mask.sum()),
        "lat_range": [round(float(yc.min()), 4), round(float(yc.max()), 4)],
        "lon_range": [round(float(lon.min()), 4), round(float(lon.max()), 4)],
        "cell_size_deg": {"east_west": round(dx, 6), "north_south": round(dy, 6)},
        "cell_size_km": {"east_west": round(km_ew, 3), "north_south": round(km_ns, 3)},
        "mean_lat": round(mean_lat, 3), "land_area_km2": round(land_km2, 1),
        "earth_radius_km": EARTH_RADIUS_KM}
    if surface and Path(str(surface)).is_file():
        import netCDF4
        with netCDF4.Dataset(str(surface)) as f:
            dims = {k: len(v) for k, v in f.dimensions.items()}
        out["surface_shape"] = [dims.get("lsmlat"), dims.get("lsmlon")]
        out["surface_matches"] = out["surface_shape"] == [nj, ni]
    wx, wy = _degrees_words(dx), _degrees_words(dy)
    size = wx if wx == wy else f"{wx} east-west by {wy} north-south"
    land = ("every one of them land" if out["land_cells"] == out["cells"] else
            f"{out['land_cells']:,} of them land")
    out["sentence"] = (f"{ni} x {nj} cells (columns by rows), {nj * ni:,} in all, {land}; "
                       f"cells of {size}, about {km_ew:.2f} km east-west by {km_ns:.2f} km "
                       f"north-south at the box's mean latitude, {mean_lat:.1f} degrees; "
                       f"{land_km2:,.0f} km2 of land")
    return out


def _bldnml_flags(opts: Optional[str]) -> Dict[str, Any]:
    """ELM_BLDNML_OPTS as flags: '-bgc bgc -methane' -> {bgc: bgc, methane: True}."""
    out: Dict[str, Any] = {}
    toks = str(opts or "").split()
    for k, t in enumerate(toks):
        if t.startswith("-"):
            nxt = toks[k + 1] if k + 1 < len(toks) else None
            out[t.lstrip("-")] = nxt if nxt and not nxt.startswith("-") else True
    return out


def configuration_setup(case_dir: Path, nl: Dict[str, list]) -> Dict[str, Any]:
    """The compset, the carbon-nutrient setup the namelist states, CO2,
    nitrogen and phosphorus deposition, the nutrient supplements and the
    weather, each with a sentence."""
    import domain_case as DC
    env = {k: DC._env_value(case_dir, k) for k in (
        "COMPSET", "ELM_NML_USE_CASE", "ELM_BLDNML_OPTS", "CCSM_CO2_PPMV", "ELM_CO2_TYPE",
        "DATM_MODE", "DATM_PRESAERO", "DATM_TOPO", "DATM_CO2_TSERIES",
        "DATM_CLMNCEP_YR_ALIGN", "DATM_CLMNCEP_YR_START", "DATM_CLMNCEP_YR_END")}
    flags = _bldnml_flags(env["ELM_BLDNML_OPTS"])
    cn = {k: _first(nl, k) for k in ("use_cn", "use_fates", "nu_com", "use_century_decomp",
                                     "use_vertsoilc", "use_lch4", "fsoilordercon")}
    cn["build_options"] = env["ELM_BLDNML_OPTS"]
    cn["build_flags"] = flags
    words = []
    if cn["use_cn"]:
        nut = NUTRIENT_WORDS.get(str(flags.get("nutrient")).lower())
        words.append(f"{nut} cycles" if nut else "carbon and nutrient cycles")
        words[-1] += " (use_cn true)"
    elif cn["use_cn"] is False:
        words.append("no carbon-nutrient cycle (use_cn false)")
    if cn["use_fates"]:
        words.append("FATES vegetation")
    if cn["nu_com"]:
        how = NU_COM_WORDS.get(str(cn["nu_com"]).upper())
        words.append(f"plants and microbes share nutrients by {how} (nu_com {cn['nu_com']})"
                     if how else f"nutrient competition set by nu_com {cn['nu_com']}")
    decomp = flags.get("soil_decomp") or ({True: "century", False: "ctc"}
                                          .get(cn["use_century_decomp"]))
    if decomp:
        name, code = DECOMP_WORDS.get(str(decomp).lower()), str(decomp).upper()
        words.append(f"soil decomposition by the {name} scheme"
                     + (f" ({code})" if name != code else "") if name else
                     f"soil decomposition by the {code} scheme")
    if cn["use_vertsoilc"]:
        words.append("soil carbon resolved by depth")
    if cn["use_lch4"]:
        words.append("methane on")
    co2 = {"type": _first(nl, "co2_type") or env["ELM_CO2_TYPE"],
           "ppmv": _first(nl, "co2_ppmv") if _first(nl, "co2_ppmv") is not None
           else env["CCSM_CO2_PPMV"], "datm_co2_tseries": env["DATM_CO2_TSERIES"]}
    co2_words = (f"held constant at {_word(co2['ppmv'])} ppmv"
                 if co2["type"] == "constant"
                 else f"of type {co2['type']} (co2_ppmv {_word(co2['ppmv'])})")
    dep = {"nitrogen": _stream(nl, "ndep"), "phosphorus": _stream(nl, "pdep")}
    dep_words = "; ".join(f"{k} deposition {_held(v['years'])} from {v['file_name']}"
                          if v["file_name"] else f"no {k} deposition file named"
                          for k, v in dep.items())
    sup = {"nitrogen": _first(nl, "suplnitro"), "phosphorus": _first(nl, "suplphos")}
    sup_words = "; ".join(
        f"{k} {SUPPLEMENT_WORDS.get(str(v).upper(), f'supplement setting {v}')} "
        f"(supl{'nitro' if k == 'nitrogen' else 'phos'} {v})"
        for k, v in sup.items() if v is not None) or "no supplement setting named"
    weather: Dict[str, Any] = {"mode": env["DATM_MODE"],
                               "years": _years(env["DATM_CLMNCEP_YR_START"],
                                               env["DATM_CLMNCEP_YR_END"]),
                               "align": int(env["DATM_CLMNCEP_YR_ALIGN"])
                               if str(env["DATM_CLMNCEP_YR_ALIGN"] or "").isdigit() else None,
                               "aerosols": env["DATM_PRESAERO"], "topography": env["DATM_TOPO"]}
    try:
        import domain_weather as W
        datm = W.control_datm(case_dir)
        weather["data_mode"] = _first(datm, "datamode")
        weather["streams"] = W.forcing_summary(case_dir)
        tax = datm.get("taxmode") or []
        main = next((s for s in weather["streams"] if "error" not in s
                     and {"TBOT", "PRECTmms"} & set(s.get("fields") or [])), None)
        if main:
            weather["main_stream"] = main["stream"]
            weather["source"] = Path(main.get("data_dir") or "").name or None
            weather["files"] = [main.get("first"), main.get("last"), main.get("n_files")]
            k = weather["streams"].index(main)
            weather["time_axis"] = tax[k] if k < len(tax) else None
    except Exception as e:                                          # noqa: BLE001
        weather["error"] = f"{type(e).__name__}: {e}"
    ys, al = weather["years"], weather["align"]
    w = ("weather read from "
         + (f"{weather['source']} files" if weather.get("source") else "files")
         + " by the data atmosphere"
         + (f", years {ys[0]}-{ys[1]}" if ys else "")
         + ((", model year Y reading weather year Y" if al == ys[0] else
             f", year {ys[0]} read as model year {al}") if ys and al is not None else "")
         + (f", repeated in a {weather['time_axis']} past the last year"
            if weather.get("time_axis") == "cycle" else "")
         + (f" (DATM mode {weather['mode']}"
            + (f", data mode {weather['data_mode']}" if weather.get("data_mode") else "")
            + ")" if weather.get("mode") else ""))
    out = {"compset": env["COMPSET"], "use_case": env["ELM_NML_USE_CASE"],
           "carbon_nutrient": cn, "co2": co2, "deposition": dep, "supplements": sup,
           "weather": weather,
           # keyed by what each says, so a reader labels them: "co2: held
           # constant at 367 ppmv"
           "sentences": {"compset": (env["COMPSET"] or "not stated in the case's env XML")
                         + (f" (land namelist use case {env['ELM_NML_USE_CASE']})"
                            if env["ELM_NML_USE_CASE"] else ""),
                         "carbon_nutrient": ", ".join(words) or "not stated in lnd_in",
                         "co2": co2_words, "deposition": dep_words,
                         "supplements": sup_words, "weather": w}}
    return out


def setup_facts(case_dir) -> Dict[str, Any]:
    """The fire, grid and configuration parts of a control's facts, from its
    CaseDocs/lnd_in and datm_in, env XML, domain and surface files. A part
    that cannot be read says why in its own "error"; the others stand."""
    from run_gate import read_namelist
    cd = Path(case_dir)
    out: Dict[str, Any] = {}
    try:
        nl = read_namelist(cd / "CaseDocs" / "lnd_in")
    except Exception as e:                                          # noqa: BLE001
        err = {"error": f"the case's CaseDocs/lnd_in cannot be read: {type(e).__name__}: {e}"}
        return {"fire": dict(err), "grid": dict(err), "configuration": dict(err)}
    for part, fn in (("fire", lambda: fire_setup(nl)),
                     ("grid", lambda: grid_setup(_domain_path(cd, nl), _first(nl, "fsurdat"))),
                     ("configuration", lambda: configuration_setup(cd, nl))):
        try:
            out[part] = fn()
        except Exception as e:                                      # noqa: BLE001
            out[part] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _domain_path(cd: Path, nl: Dict[str, list]) -> Path:
    """The land domain file: lnd_in's fatmlndfrc, else the env XML's
    LND_DOMAIN_PATH/LND_DOMAIN_FILE."""
    import domain_case as DC
    p = _first(nl, "fatmlndfrc")
    if p and Path(str(p)).is_file():
        return Path(str(p))
    d, f = DC._env_value(cd, "LND_DOMAIN_PATH"), DC._env_value(cd, "LND_DOMAIN_FILE")
    if d and f and (Path(d) / f).is_file():
        return Path(d) / f
    raise ControlError(f"no readable land domain file: lnd_in fatmlndfrc {p!r}, env XML "
                       f"{d!r}/{f!r}")


def compute_facts(name: str, override: Optional[str] = None,
                  years: Optional[List[int]] = None) -> Dict[str, Any]:
    """Every fact of one control, read from its own case and output, written
    to the cache beside the registry. Minutes: each year's check reads a year
    of daily history."""
    import gridded as G
    import run_gate as RG
    e = get(name, override)
    cd, rd = Path(e["case_dir"]), Path(e["run_dir"])
    s = _settings(cd)
    h0 = G.open_stream(rd, "h0")
    out_years = h0.complete_years()
    ys = years or out_years
    per_year: Dict[str, Dict] = {}
    model_fire: Dict[str, float] = {}
    pattern: Dict[str, Dict] = {}
    for y in ys:
        try:
            yc = RG.year_check(rd, cd, y)
            per_year[str(y)] = {k: yc.get(k) for k in YEAR_FACTS}
            per_year[str(y)]["checks_pass"] = yc.get("pass")
        except Exception as ex:                                     # noqa: BLE001
            per_year[str(y)] = {"error": f"{type(ex).__name__}: {ex}"}
        if "FAREA_BURNED" in h0.variables:
            b = burned_area_km2(rd, y)
            model_fire[str(y)] = round(b["km2"], 1)
            pattern[str(y)] = {k: b[k] for k in ("cells_over_1pct", "max_cell_fraction")}
    fire: Dict[str, Any] = {}
    if model_fire:
        fire = {"model": model_fire, "model_cells": pattern,
                "pattern": "spread over "
                           f"{min(p['cells_over_1pct'] for p in pattern.values())}-"
                           f"{max(p['cells_over_1pct'] for p in pattern.values())} "
                           "cells burning more than 1 % in a year, the most burned cell "
                           f"{max(p['max_cell_fraction'] for p in pattern.values()):.0%}"}
        for on, o in (e.get("observations") or {}).items():
            if o.get("var") == "burned_fraction_year":
                obs = observed_burned_area_km2(o["path"], o["var"], h0.grid)
                fire["observed"] = {str(y): round(v, 1) for y, v in obs.items()
                                    if y in ys}
                fire["observed_source"] = o.get("what") or on
    streams = {}
    for st in sorted(G.discover(rd)):
        try:
            ss = G.open_stream(rd, st)
            streams[st] = {"fields": sorted(ss.variables), "years": ss.complete_years()}
        except Exception as ex:                                     # noqa: BLE001
            streams[st] = {"error": f"{type(ex).__name__}: {ex}"}
    start = s.get("RUN_STARTDATE")
    f = {"name": name, "version": FACTS_VERSION, "case": cd.name,
         "computed": datetime.datetime.now().isoformat(timespec="seconds"),
         "start_year": int(start[:4]) if start else None,
         "forcing_years": _forcing_years(s),
         "output_years": [out_years[0], out_years[-1]] if out_years else None,
         "settings": s, "per_year": per_year, "burned_area_km2": fire,
         "hours_per_model_year": _hours_per_year(e), "streams": streams,
         "field_notes": {k: v for k, v in G.FIELD_NOTES.items()
                         if any(k in (x.get("fields") or []) for x in streams.values())}}
    f.update(setup_facts(cd))
    p = _facts_path(e)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(f, indent=1, default=str))
    tmp.replace(p)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# WHAT RECEPTION AND THE PLANNER ARE SHOWN
# ─────────────────────────────────────────────────────────────────────────────
def not_in_it(f: Optional[Dict[str, Any]]) -> List[str]:
    """What the control does not contain that a question may assume, in plain
    words, from its facts: the observed fires, when the facts set the model's
    burned area beside an observed one (Reception tells the user before a
    design is settled)."""
    fire = (f or {}).get("burned_area_km2") or {}
    m, o = fire.get("model") or {}, fire.get("observed") or {}
    ys = sorted(set(m) & set(o))
    if not ys:
        return []
    src = fire.get("observed_source") or "the observation"
    pairs = "; ".join(f"{y} {m[y]:.0f} against {o[y]:.0f} km2" for y in ys)
    if not any(m[y] for y in ys):
        return [f"{_FIRE_NONE}; burned area observed by {src}: "
                + "; ".join(f"{y} {o[y]:.0f} km2" for y in ys)]
    return [f"{_FIRE_MODEL_ONLY}; burned area over the grid, the model against {src}: "
            f"{pairs}"]


# THE SAME ITEMS WITHOUT THE OBSERVATIONS' NUMBERS (2026-11-03), for the
# Planner: its caveats go to the Analyzer, which reads the model's output
# only, so the observed burned area must not ride along in them.
_FIRE_MODEL_ONLY = ("the observed fires: the run's fire is the model's own, lit and spread "
                    "in its own places and years, so a fire that happened is not in it")
_FIRE_NONE = "the observed fires: the run has no fire"


def not_in_it_words(f: Optional[Dict[str, Any]]) -> List[str]:
    """not_in_it's items in words only, with no observed number or source."""
    fire = (f or {}).get("burned_area_km2") or {}
    m, o = fire.get("model") or {}, fire.get("observed") or {}
    ys = sorted(set(m) & set(o))
    if not ys:
        return []
    return [_FIRE_MODEL_ONLY if any(m[y] for y in ys) else _FIRE_NONE]


# THE SUITE IN ONE LINE (2026-11-04). Reception is told whether the control's
# results store (domain_suite: maps and region series made once from its
# history) is built: its version, products, size and build date, about 200
# characters, read from the suite's manifest, or "not built". The store itself
# is read by the analysis, never shown here.
SUITE_MANIFEST = "suite.json"


def suite_line(entry: Dict[str, Any]) -> str:
    d = entry.get("suite_dir")
    if not d:
        return "not built (the registry names no suite_dir)"
    p = Path(str(d)) / SUITE_MANIFEST
    if not p.is_file():
        return "not built"
    try:
        m = json.loads(p.read_text())
    except ValueError:
        return "not built (its manifest cannot be read)"
    years = (m.get("periods") or {}).get("year") or []
    span = f"{years[0]}-{years[-1]}" if years else "no whole year"
    built = str(m.get("built_at") or "?")[:16].replace("T", " ")
    return (f"version {m.get('SUITE_VERSION')}, built {built} UTC, "
            f"{(m.get('total_bytes') or 0) / 1e9:.2f} GB: {len(m.get('fields') or {})} fields, "
            f"{span}; maps by year, water year and season; region series by day, month, "
            f"season, year and water year")


def summary(entry: Dict[str, Any], f: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One control as reception sees it: small enough for a tool result (cut
    at 12,000 characters), so field lists are counts plus the fields the case
    asked for. The year numbers stay whole: a run's difference from the
    control is computed from them."""
    out = {"name": entry["name"], "description": entry.get("description"),
           "usable": entry.get("usable"),
           "reproduced": entry.get("reproduced"),
           # how well it matches observations, measured outside the server and
           # entered with its source, as `reproduced` is
           "evaluation": entry.get("evaluation"),
           "regions": {k: v.get("what") for k, v in (entry.get("regions") or {}).items()},
           "observations": {k: v.get("what")
                            for k, v in (entry.get("observations") or {}).items()},
           "suite": suite_line(entry)}
    if not f:
        out["facts"] = None
        out["facts_note"] = ("not computed yet: refresh_domain_control_facts reads them "
                             "from the control's output")
        return out
    s = f.get("settings") or {}
    out["facts"] = {
        "case": f.get("case"), "start_year": f.get("start_year"),
        "forcing_years": f.get("forcing_years"), "output_years": f.get("output_years"),
        "settings": {k: s.get(k) for k in NAMELIST_SETTINGS + ("CCSM_CO2_PPMV",
                                                              "ELM_CO2_TYPE",
                                                              "ndep_stream_years")},
        "per_year": f.get("per_year"),
        "burned_area_km2": {k: v for k, v in (f.get("burned_area_km2") or {}).items()
                            if k != "model_cells"},
        "hours_per_model_year": f.get("hours_per_model_year"),
        "streams": {st: {"n_fields": len(d.get("fields") or []), "years": d.get("years")}
                    for st, d in (f.get("streams") or {}).items()},
        "requested_fields": s.get("requested_fields"),
        "field_notes": f.get("field_notes"),
        "computed": f.get("computed")}
    out["facts"].update(_setup_summary(f))
    out["not_in_it"] = not_in_it(f)
    out["not_in_it_words"] = not_in_it_words(f)
    return out


# WHAT OF THE SETUP RECEPTION IS SHOWN (2026-11-03): each part's sentence and
# the few numbers a design is sized by. The whole parts stay in the cache; the
# tool result is cut at 12,000 characters, and the parts in full would push
# the control's facts past it.
SETUP_SUMMARY = {"fire": ("on", "sentence"),
                 "grid": ("cells", "land_cells", "land_area_km2", "lat_range", "lon_range",
                          "sentence"),
                 "configuration": ("sentences",)}


def _setup_summary(f: Dict[str, Any]) -> Dict[str, Any]:
    """The setup parts as reception sees them (SETUP_SUMMARY). A cache
    without them says so in each part's error."""
    out: Dict[str, Any] = {}
    for part, keys in SETUP_SUMMARY.items():
        v = f.get(part)
        if not isinstance(v, dict):
            v = {"error": "not in these facts: refresh_domain_control_facts reads it"}
        out[part] = ({"error": v["error"]} if v.get("error") else
                     json.loads(json.dumps({k: v.get(k) for k in keys}, default=str)))
    return out


def describe(override: Optional[str] = None) -> Dict[str, Any]:
    reg = load(override)
    rows = []
    for name, e in reg["controls"].items():
        try:
            fa = facts(name, override) if e["usable"] else None
        except ControlError:
            fa = None
        rows.append(summary(dict(e, registry=reg["path"]), fa))
    return {"registry": reg["path"], "source": reg["source"], "controls": rows,
            "problems": reg["problems"]}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "refresh":
        print(json.dumps(summary(get(sys.argv[2]), compute_facts(sys.argv[2])), indent=1))
    else:
        print(json.dumps(describe(), indent=1))
