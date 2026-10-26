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
    f = {"name": name, "case": cd.name, "computed": datetime.datetime.now()
         .isoformat(timespec="seconds"),
         "start_year": int(start[:4]) if start else None,
         "forcing_years": _forcing_years(s),
         "output_years": [out_years[0], out_years[-1]] if out_years else None,
         "settings": s, "per_year": per_year, "burned_area_km2": fire,
         "hours_per_model_year": _hours_per_year(e), "streams": streams,
         "field_notes": {k: v for k, v in G.FIELD_NOTES.items()
                         if any(k in (x.get("fields") or []) for x in streams.values())}}
    p = _facts_path(e)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(f, indent=1, default=str))
    tmp.replace(p)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# WHAT RECEPTION AND THE PLANNER ARE SHOWN
# ─────────────────────────────────────────────────────────────────────────────
def summary(entry: Dict[str, Any], f: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One control as reception sees it: small enough for a tool result (cut
    at 12,000 characters), so field lists are counts plus the fields the case
    asked for. The year numbers stay whole: a run's difference from the
    control is computed from them."""
    out = {"name": entry["name"], "description": entry.get("description"),
           "usable": entry.get("usable"),
           "reproduced": entry.get("reproduced"),
           "regions": {k: v.get("what") for k, v in (entry.get("regions") or {}).items()},
           "observations": {k: v.get("what")
                            for k, v in (entry.get("observations") or {}).items()}}
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
