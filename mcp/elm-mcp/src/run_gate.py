#!/usr/bin/env python3
"""Did this job's model year run, and is it the run that was meant?

Four gates for a whole-domain ELM run, each returning a JSON-ready record that
carries the numbers behind its verdict ('pass') and names every failure
('reasons'). Ported from the hand checks of the Naches ELM-BGC experiment
(check_run.py and compare_replication.py, 2026-10-12/14), whose records they
reproduce.

  scan_logs        THIS job's lnd and cpl logs and timing, found by the job id
                   in their names (CIME's LID is <job id>.<yymmdd-HHMMSS>).
                   case.submit --no-batch exits 0 when the model aborts, and a
                   newest-file lookup once judged a dead job by an older run's
                   logs (2026-10-13), so a job id is required. The coupler log
                   also says which model dates the job ran, so a year check
                   can be tied to this job's own simulation.
  year_check       model facts for one calendar year: every history tape
                   complete with the fields its namelist asks for, a restart
                   for the next year, no missing values over vegetated cells,
                   and sanity bounds on GPP, peak LAI, water closure and the
                   carbon stock (YearLimits). Read with gridded.py and
                   regions.py, so the conventions are theirs.
  carry_over       a warm start really started from its saved state: the first
                   day's means against the restart, over vegetated cells. A
                   cold start differs by about 100 %, so 5 % separates cleanly.
  reference_gate   a year against a reference run of the same setup: per-field
                   bias and daily correlation (ReferenceLimits). The phenology
                   bug of 2026-10-13 passed every model fact and year 1 of this
                   gate; year 2 caught it (GPP -10.1 %, TLAI r 0.878).

A number that cannot be computed (a missing value, a zero mean) fails its gate
and the reason says so; records hold None, never NaN. The surface data and
saved state are read from the case's own lnd_in (fsurdat, finidat) unless
given. Every function is imported directly; the command line at the bottom is
for job scripts, and always writes its record and STATUS line:

    python run_gate.py year --case-dir C --run-dir R --job $SLURM_JOB_ID \\
        --year 2016 --out C/checks --status C/STATUS.log \\
        [--reference REF_RUN_DIR [--reference-report-only]]

With --reference, each year is also set against that run with reference_gate,
gated unless --reference-report-only (for an experiment expected to differ).
domain_case.py writes the job scripts that call this.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import math
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import gridded as G
import regions as R

SECONDS_PER_DAY = 86400.0


class GateError(Exception):
    """A gate cannot be run, and the message says why."""


# What a gate turns into a reason rather than a crash.
_GATE_ERRORS = (GateError, G.GriddedReadError, R.RegionError, OSError)


def _num(x) -> Optional[float]:
    """A float for the record, None when it is not a number."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _fmt(x, spec: str) -> str:
    return "n/a" if _num(x) is None else format(float(x), spec)


# ─────────────────────────────────────────────────────────────────────────────
# THE CASE'S NAMELIST
# ─────────────────────────────────────────────────────────────────────────────
def _strip_comment(line: str) -> str:
    out, q = [], None
    for ch in line:
        if q:
            out.append(ch)
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
            out.append(ch)
        elif ch == "!":
            break
        else:
            out.append(ch)
    return "".join(out)


def _split_values(text: str) -> List[str]:
    vals, cur, q = [], [], None
    for ch in text:
        if q:
            cur.append(ch)
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
            cur.append(ch)
        elif ch in ",\n":
            if "".join(cur).strip():
                vals.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        vals.append("".join(cur).strip())
    return vals


def _value(tok: str):
    t = tok.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        return t[1:-1]
    low = t.lower()
    if low in (".true.", "t", ".t."):
        return True
    if low in (".false.", "f", ".f."):
        return False
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(low.replace("d", "e"))
    except ValueError:
        return t


def read_namelist(path) -> Dict[str, list]:
    """{key: [values]} from a Fortran namelist file such as lnd_in.

    Values continue over lines until the next 'key =' or the group's closing
    '/'; strings lose their quotes, numbers and logicals are converted. Keys
    are lower-cased. Enough for the history and input keys read here, which
    CIME writes whole: an indexed entry such as hist_nhtfrq(2) = -24 is
    refused rather than misread.
    """
    text = "\n".join(_strip_comment(l) for l in Path(path).read_text().splitlines())
    out: Dict[str, list] = {}
    for grp in re.finditer(r"&\w+(.*?)^\s*/", text, re.S | re.M):
        body = grp.group(1)
        idx = re.search(r"^\s*([A-Za-z_]\w*)\s*\(([^)]*)\)\s*=", body, re.M)
        if idx:
            raise GateError(f"{path}: the indexed entry {idx.group(1)}({idx.group(2)}) is "
                            f"not read by this parser")
        keys = list(re.finditer(r"^\s*([A-Za-z_]\w*)\s*=", body, re.M))
        for k, m in enumerate(keys):
            end = keys[k + 1].start() if k + 1 < len(keys) else len(body)
            out[m.group(1).lower()] = [_value(v) for v in _split_values(body[m.end():end])]
    return out


def case_namelist(case_dir, run_dir=None) -> Tuple[Dict[str, list], Path]:
    """The land namelist the run read: run_dir/lnd_in, else CaseDocs/lnd_in."""
    for p in ([Path(run_dir) / "lnd_in"] if run_dir else []) + \
             [Path(case_dir) / "CaseDocs" / "lnd_in"]:
        if p.is_file():
            return read_namelist(p), p
    raise GateError(f"no lnd_in in {run_dir} or {Path(case_dir) / 'CaseDocs'}")


def _nhtfrq(nl: Dict[str, list], stream: str) -> int:
    """hist_nhtfrq of one tape. A tape the list does not reach keeps ELM's
    default: 0 (monthly) for h0, -24 (daily) for the others (histFileMod)."""
    k = int(stream[1:])
    freq = nl.get("hist_nhtfrq") or []
    return int(freq[k]) if k < len(freq) else (0 if k == 0 else -24)


def tape_interval_days(nl: Dict[str, list], stream: str) -> float:
    """How often a tape writes, in days, from hist_nhtfrq: 0 is monthly (30
    days here; first_write_days says when a monthly file first appears), a
    negative value is hours, a positive one model steps of dtime seconds."""
    n = _nhtfrq(nl, stream)
    if n == 0:
        return 30.0
    if n < 0:
        return -n / 24.0
    dtime = float((nl.get("dtime") or [1800])[0])
    return n * dtime / SECONDS_PER_DAY


def first_write_days(nl: Dict[str, list], stream: str, start, calendar: str) -> float:
    """Days from a run's start to a tape's first record: the end of the
    starting month for a monthly tape (ELM opens a monthly file only then),
    one interval otherwise."""
    if _nhtfrq(nl, stream) != 0:
        return tape_interval_days(nl, stream)
    into = (start.day - 1) + (start.hour * 3600 + start.minute * 60 + start.second) / \
        SECONDS_PER_DAY
    return G.days_in_month(start.year, start.month, calendar) - into


def requested_tapes(nl: Dict[str, list]) -> Dict[str, List[str]]:
    """{stream: fields the namelist adds to it}: hist_fincl1 -> h0, 2 -> h1 ..."""
    out = {}
    for key, vals in sorted(nl.items()):
        m = re.fullmatch(r"hist_fincl(\d+)", key)
        if m:
            out[f"h{int(m.group(1)) - 1}"] = [str(v).split(":")[0] for v in vals if v]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# THE JOB'S LOGS
# ─────────────────────────────────────────────────────────────────────────────
def _read_text(path) -> str:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        return f.read()


def _job_file(folder: Path, prefix: str, job: str) -> Optional[Path]:
    """The newest file <prefix>.<job>.<yymmdd-HHMMSS>[.gz] in folder, by the
    timestamp in its name (a resubmitted job writes a second one)."""
    hits = sorted(p for p in folder.glob(f"{prefix}.{job}.*") if p.is_file())
    return hits[-1] if hits else None


_BALANCE_NOT_A_PROBLEM = re.compile(r"^\s*\d+\s+\w+_BALANCE_ERROR|skipping CN balance check")
_MODEL_DATE = re.compile(r"model date =\s*(\d+)\s+\d+")
_END_DATE = re.compile(r"SUCCESSFUL TERMINATION[^\n]*\n[^\n]*at YMD,TOD =\s*(\d+)\s+\d+")
_SIM_DAYS = re.compile(r"# simulated days \(this run\) =\s*([0-9.]+)")
_MODEL_LOG_ERROR = re.compile(r"ENDRUN|MPI_Abort|forrtl: severe|SIGSEGV|Segmentation fault")


def _date(ymd: str) -> str:
    v = int(ymd)
    return f"{v // 10000:04d}-{v // 100 % 100:02d}-{v % 100:02d}"


def scan_logs(run_dir, case_dir, job_id) -> Dict:
    """THIS job's land, coupler and model logs and timing. A record.

    Gated: the coupler log says SUCCESSFUL TERMINATION, and the land log has
    no ENDRUN. Recorded only: ERROR lines, balance warnings (not field names
    such as CMASS_BALANCE_ERROR, nor the first-step notice), NaN lines, the
    model dates the job ran (first and last in the coupler log, and the end
    date of a finished run), abort lines in e3sm.log, and throughput from the
    timing file.
    """
    job = str(job_id or "").strip()
    if not job:
        raise GateError("a log scan needs the job id: without it the newest logs are "
                        "read, which may belong to an older run")
    rd, cd = Path(run_dir), Path(case_dir)
    lnd, cpl = _job_file(rd, "lnd.log", job), _job_file(rd, "cpl.log", job)
    e3 = _job_file(rd, "e3sm.log", job)
    rec: Dict = {"job": job, "lnd_log": str(lnd) if lnd else None,
                 "cpl_log": str(cpl) if cpl else None, "e3sm_log": str(e3) if e3 else None}
    reasons = []
    ctext = _read_text(cpl) if cpl else ""
    rec["successful_termination"] = "SUCCESSFUL TERMINATION" in ctext
    dates = _MODEL_DATE.findall(ctext)
    m = _END_DATE.search(ctext)
    rec["model_start"] = _date(dates[0]) if dates else None
    rec["model_last"] = _date(dates[-1]) if dates else None
    rec["model_end"] = _date(m.group(1)) if m else None
    m = _SIM_DAYS.search(ctext)
    rec["simulated_days"] = float(m.group(1)) if m else None
    if not cpl:
        reasons.append(f"no cpl.log from job {job}")
    elif not rec["successful_termination"]:
        reasons.append("the coupler log has no SUCCESSFUL TERMINATION" +
                       (f" (last model date {rec['model_last']})" if rec["model_last"] else ""))
    rec["error_lines"], rec["abort_lines"], rec["abort_context"] = [], [], None
    rec["balance_lines"] = rec["nan_lines"] = None
    if lnd:
        lines = _read_text(lnd).splitlines()
        rec["error_lines"] = [l.strip()[:200] for l in lines if re.search(r"\bERROR\b", l)][:10]
        hits = [k for k, l in enumerate(lines) if re.search(r"ENDRUN|endrun", l)]
        rec["abort_lines"] = [lines[k].strip()[:200] for k in hits][:10]
        if hits:
            # ENDRUN: is printed alone; the cause follows on the next lines
            k = hits[0]
            after = [l.strip() for l in lines[k + 1:k + 20] if l.strip()][:2]
            rec["abort_context"] = " ".join([lines[k].strip()] + after)[:300]
            reasons.append(f"the land log stops the run: {rec['abort_context']}")
        rec["balance_lines"] = sum(1 for l in lines if re.search(r"balance", l, re.I)
                                   and re.search(r"warn|error", l, re.I)
                                   and not _BALANCE_NOT_A_PROBLEM.search(l))
        rec["nan_lines"] = sum(1 for l in lines if re.search(r"\bNaN\b", l))
    else:
        reasons.append(f"no lnd.log from job {job}")
    rec["e3sm_error_lines"] = [l.strip()[:200] for l in _read_text(e3).splitlines()
                               if _MODEL_LOG_ERROR.search(l)][:5] if e3 else []
    if cpl and not rec["successful_termination"] and not rec["abort_lines"] and \
            rec["e3sm_error_lines"]:
        reasons.append(f"the model log says: {rec['e3sm_error_lines'][0]}")
    tim = _job_file(cd / "timing", "e3sm_timing.*", job)
    rec["timing_file"] = str(tim) if tim else None
    if tim:
        txt = _read_text(tim)
        m = re.search(r"Model Throughput:\s*([0-9.]+)\s*simulated_years/day", txt)
        if m and float(m.group(1)) > 0:
            rec["sim_years_per_day"] = float(m.group(1))
            rec["hours_per_sim_year"] = 24.0 / float(m.group(1))
        for k in ("Init Time", "Run Time"):
            m = re.search(rf"{k}\s*:\s*([0-9.]+)\s*seconds", txt)
            if m:
                rec[k.lower().replace(" ", "_") + "_s"] = float(m.group(1))
    rec["pass"], rec["reasons"] = not reasons, reasons
    return rec


def job_ran_year(logs: Dict, year: int) -> Optional[str]:
    """None when the job's logs show it simulated the whole calendar year,
    else why not. The run folder can hold an earlier job's files for the
    same year (a resubmission, a chain restarted from an older pointer)."""
    if not logs.get("successful_termination"):
        return None                     # scan_logs has failed the job already
    start, end = logs.get("model_start"), logs.get("model_end")
    if not (start and end):
        return "the coupler log gives no model dates, so the year cannot be tied to this job"
    if not (start <= f"{year:04d}-01-01" and end >= f"{year + 1:04d}-01-01"):
        return (f"this job simulated {start} to {end}, not all of {year}: the year's files "
                f"are not all its own")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ONE YEAR'S MODEL FACTS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class YearLimits:
    """Sanity bounds that catch a broken run, not a judgement of the climate.

    The defaults are the Naches hand check's: a vegetated mountain basin.
    A desert or a tundra domain may need other GPP and LAI bounds.
    """
    gpp_gC_m2_yr: Tuple[float, float] = (200.0, 3000.0)
    peak_tlai: Tuple[float, float] = (0.5, 15.0)
    water_closure_frac_max: float = 0.05
    carbon_stock_change_frac_max: float = 0.05


# The daily fields the year check reads, and how it combines them. A vegetated
# cell missing any of them at any day fails the year: the means are taken over
# the cells with values, so a lost cell would otherwise go unseen.
_YEAR_FIELDS = (R.Field("GPP"), R.Field("TLAI"), R.Field("ET", ("QVEGT", "QVEGE", "QSOIL")),
                R.Field("P", ("RAIN", "SNOW")), R.Field("QRUNOFF"), R.Field("TWS"),
                R.Field("H2OSNO"), R.Field("TOTECOSYSC"))


def _vegetated(grid: G.Grid, surface) -> np.ndarray:
    return R.vegetated(R.read_surface(surface, grid, ("PCT_NATVEG",))["PCT_NATVEG"])


def _daily_stream(run_dir, tapes: Sequence[str]) -> G.Stream:
    for name in tapes:
        try:
            s = G.open_stream(run_dir, name)
        except G.GriddedReadError:
            continue
        if s.cadence == "daily":
            return s
    raise GateError(f"no daily history stream among {list(tapes)} in {run_dir}")


def _lack_reason(name: str, lack: Dict[str, List[str]], n_files: int) -> str:
    miss = sorted(lack)
    if all(len(v) == n_files for v in lack.values()):
        return f"{name} lacks requested fields {miss}"
    first = min(lack.values(), key=len)[0]
    return f"{name} lacks requested fields {miss} in some of its files (first {first})"


def year_check(run_dir, case_dir, year: int, surface=None,
               limits: YearLimits = YearLimits()) -> Dict:
    """Model facts for one calendar year of a whole-domain run. A record.

    Gated: every tape the namelist adds fields to holds the whole year and,
    in every file holding it, those fields; the restart for 1 January of the
    next year exists; no vegetated cell misses a value of a field read here;
    and the YearLimits bounds hold. Numbers are area-weighted means over
    vegetated cells (PCT_NATVEG = 100), summed over the year for fluxes.
    """
    nl, nl_path = case_namelist(case_dir, run_dir)
    surface = surface or (nl.get("fsurdat") or [None])[0]
    if not surface:
        raise GateError(f"no fsurdat in {nl_path} and no surface given")
    year = int(year)
    reasons: List[str] = []
    tapes: Dict[str, Dict] = {}
    for name, fields in requested_tapes(nl).items():
        try:
            st = G.open_stream(run_dir, name)
            have, missing = st.year_records(year)
            lack = st.lacking(fields, have)
        except G.GriddedReadError as e:
            tapes[name] = {"ok": False, "error": str(e)}
            reasons.append(f"{name}: {e}")
            continue
        n_files = len({r.file for r in have}) if have else len(st.files)
        tapes[name] = {"cadence": st.cadence, "records": len(have),
                       "missing_records": len(missing),
                       "first_missing": G.label_text(missing[0], st.cadence) if missing else None,
                       "missing_fields": sorted(lack), "ok": not missing and not lack}
        if missing:
            reasons.append(f"{name}: {st._missing_message(year, missing)}")
        if lack:
            reasons.append(_lack_reason(name, lack, n_files))
    rec: Dict = {"year": year, "case": None, "namelist": str(nl_path),
                 "surface": str(surface), "tapes": tapes, "limits": asdict(limits),
                 "restart_next_year": False}
    try:
        files = G.discover(run_dir)
        for fs in files.values():
            m = G.HISTORY_NAME.match(fs[0].name)
            rec["case"] = m.group("case") if m else None
            break
        case = rec["case"]
        restart = Path(run_dir) / f"{case}.elm.r.{year + 1}-01-01-00000.nc" if case else None
        rec["restart_next_year"] = bool(restart and restart.is_file())
        if not rec["restart_next_year"]:
            reasons.append(f"no restart for 1 January {year + 1}")
        daily = _daily_stream(run_dir, sorted(files))
        veg = _vegetated(daily.grid, surface)
        ms = R.MaskSet(daily.grid)
        ms.add("vegetated", veg, "PCT_NATVEG >= 100", [Path(surface).name])
        rs = R.reduce_regions(daily, _YEAR_FIELDS, ms, [year])
    except _GATE_ERRORS as e:
        reasons.append(f"year numbers: {e}")
        rec["pass"], rec["reasons"] = False, reasons
        return rec
    v = {f: rs.series("vegetated", f) for f in rs.fields}
    labels = rs.labels
    flux = v["P"] - v["ET"] - v["QRUNOFF"]
    P = float(v["P"].sum())
    tws_change = float(v["TWS"][-1] - v["TWS"][0])
    apr1 = [k for k, lab in enumerate(labels) if lab == f"{year}-04-01"]
    eco = v["TOTECOSYSC"]
    # the fluxes of days 2..N against the change between day 1 and day N
    wc = (flux[1:].sum() - tws_change) / P if P else float("nan")
    cs = (eco[-1] - eco[0]) / eco[0] if eco[0] else float("nan")
    gpp, lai = float(v["GPP"].sum()), float(np.max(v["TLAI"]))
    by_field = {f: int(rs.nan_cells[("vegetated", f)]) for f in rs.fields}
    rec.update({
        "stream": daily.name, "days": len(labels), "cells": int(veg.sum()),
        "gpp_gC_m2_yr": _num(gpp), "peak_tlai": _num(lai),
        "precip_mm": _num(P), "et_mm": _num(v["ET"].sum()),
        "runoff_mm": _num(v["QRUNOFF"].sum()), "tws_change_mm": _num(tws_change),
        "water_closure_frac_of_precip": _num(wc),
        "swe_apr1_mm": _num(v["H2OSNO"][apr1[0]]) if apr1 else None,
        "totecosysc_change_frac": _num(cs),
        "nan_cells": sum(by_field.values()), "nan_cells_by_field": by_field,
    })
    lo, hi = limits.gpp_gC_m2_yr
    if not lo <= gpp <= hi:
        reasons.append(f"GPP {gpp:.1f} gC m-2 yr-1 is outside {lo:g}-{hi:g}")
    lo, hi = limits.peak_tlai
    if not lo <= lai <= hi:
        reasons.append(f"peak TLAI {lai:.3f} is outside {lo:g}-{hi:g}")
    if P == 0:
        reasons.append("water closure undefined: no precipitation over vegetated cells")
    elif not abs(wc) < limits.water_closure_frac_max:
        reasons.append(f"water does not close: {_fmt(wc, '.4g')} of precipitation"
                       + ("" if math.isfinite(wc) else " (a field it is computed from "
                                                        "misses values)"))
    if eco[0] == 0:
        reasons.append("carbon stock change undefined: no ecosystem carbon on the first day")
    elif not abs(cs) < limits.carbon_stock_change_frac_max:
        reasons.append(f"the ecosystem carbon stock changed by {_fmt(cs, '.4g')} in a year"
                       + ("" if math.isfinite(cs) else " (TOTECOSYSC misses values)"))
    if rec["nan_cells"]:
        bad = {f: n for f, n in by_field.items() if n}
        reasons.append(f"{rec['nan_cells']} vegetated cell-fields miss values ({bad})")
    rec["pass"], rec["reasons"] = not reasons, reasons
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# A WARM START'S FIRST DAY
# ─────────────────────────────────────────────────────────────────────────────
# (history field, restart field, floor for the relative difference)
CARRY_FIELDS = (("TOTSOMC", "totsomc", 10.0), ("TOTVEGC", "totvegc", 10.0),
                ("LEAFC", "leafc", 1.0), ("TLAI", "tlai", 0.05))
# Gated: LEAFC (a primary pool) and TOTSOMC. TOTVEGC and TLAI are reported
# only: their definitions can differ between the code that wrote the restart
# and the code that reads it.
CARRY_MAX_MEDIAN = {"TOTSOMC": 0.05, "LEAFC": 0.05}


def carry_over(run_dir, case_dir=None, restart=None, surface=None,
               fields=CARRY_FIELDS, gates=CARRY_MAX_MEDIAN) -> Dict:
    """The run's first daily means against its saved state. A record.

    The restart's natural-vegetation values are put on the grid (PFT values
    weighted by their land-unit weight, column values as they are) and set
    against the history's first day over vegetated cells: median and 99th
    percentile of |history - restart| / max(|restart|, floor).
    """
    import netCDF4
    if restart is None or surface is None:
        if case_dir is None:
            raise GateError("carry_over needs the case directory, or both restart and surface")
        nl, nl_path = case_namelist(case_dir, run_dir)
        restart = restart or (nl.get("finidat") or [None])[0]
        surface = surface or (nl.get("fsurdat") or [None])[0]
        if not restart:
            raise GateError(f"{nl_path} names no finidat: a cold start has nothing to carry over")
    daily = _daily_stream(run_dir, sorted(G.discover(run_dir)))
    first = daily.records[0]
    veg = _vegetated(daily.grid, surface)
    t = netCDF4.Dataset(restart)
    reasons: List[str] = []
    res: Dict = {}
    try:
        def raw(n):
            v = t[n]
            v.set_auto_maskandscale(False)
            return np.asarray(v[:])
        gi, gj = raw("grid1d_ixy").astype(int) - 1, raw("grid1d_jxy").astype(int) - 1
        pl, pg, pw = raw("pfts1d_ityplun"), raw("pfts1d_gridcell_index").astype(int) - 1, \
            raw("pfts1d_wtlnd")
        cl, cg = raw("cols1d_ityplun"), raw("cols1d_gridcell_index").astype(int) - 1
        for hname, rname, floor in fields:
            if rname not in t.variables:
                res[hname] = {"skipped": f"restart has no {rname}"}
                continue
            x = raw(rname).astype(np.float64)
            dim = t[rname].dimensions[0]
            if dim == "pft":
                sel = pl == 1
                grid = np.zeros(veg.shape)
                np.add.at(grid, (gj[pg[sel]], gi[pg[sel]]), x[sel] * pw[sel])
            elif dim == "column":
                sel = cl == 1
                grid = np.full(veg.shape, np.nan)
                grid[gj[cg[sel]], gi[cg[sel]]] = x[sel]
            else:
                res[hname] = {"skipped": f"restart {rname} is on {dim}"}
                continue
            if hname not in daily.variables:
                res[hname] = {"skipped": f"the {daily.name} history has no {hname}"}
                continue
            h = daily.read_records(hname, [first])[0]
            m = veg & np.isfinite(h) & np.isfinite(grid) & (np.abs(grid) < 1e30)
            nan_h = int((veg & ~np.isfinite(h)).sum())
            if not m.any():
                res[hname] = {"cells": 0, "history_nan_cells": nan_h,
                              "skipped": "no vegetated cell has both a first-day value and "
                                         "a saved one"}
                continue
            rel = np.abs(h[m] - grid[m]) / np.maximum(np.abs(grid[m]), floor)
            res[hname] = {"cells": int(m.sum()), "median_rel_diff": float(np.median(rel)),
                          "p99_rel_diff": float(np.percentile(rel, 99)),
                          "history_nan_cells": nan_h,
                          "domain_mean_history": float(h[m].mean()),
                          "domain_mean_restart": float(grid[m].mean())}
    finally:
        t.close()
    for k, lim in gates.items():
        r = res.get(k) or {}
        if "median_rel_diff" not in r:
            reasons.append(f"{k}: {r.get('skipped', 'not compared')}")
        elif not r["median_rel_diff"] <= lim:
            reasons.append(f"{k} on the first day differs from the saved state by a median "
                           f"{r['median_rel_diff']:.3f} (limit {lim}): not a warm start from it")
        elif r["history_nan_cells"]:
            reasons.append(f"{k}: {r['history_nan_cells']} vegetated cells miss the first day")
    return {"restart": str(restart), "surface": str(surface), "stream": daily.name,
            "first_day": G.label_text(first.key, daily.cadence), "fields": res,
            "gates": dict(gates), "pass": not reasons, "reasons": reasons}


# ─────────────────────────────────────────────────────────────────────────────
# A YEAR AGAINST A REFERENCE RUN
# ─────────────────────────────────────────────────────────────────────────────
REFERENCE_FIELDS = ("GPP", "NPP", "NEE", "TLAI", "ELAI", "LEAFC", "TOTVEGC", "TOTSOMC",
                    "H2OSNO", "BTRAN", "TSA")
# Cells of a reference run count as ours within this distance: float32
# centres from another source sit up to 4.6e-5 deg off on the Naches grid.
CELL_TOL_DEG = 1e-4


@dataclass(frozen=True)
class ReferenceLimits:
    """Max |bias| in percent and min daily correlation of the vegetated-cell
    means, per gated field. The Naches limits: they catch a broken setup,
    not the code-version and forcing-interpolation differences, which gave
    GPP +0.5 %, TLAI -0.7 %, pools 0.0 % and r >= 0.995 on the first test.
    A field named here and not in REFERENCE_FIELDS is compared as well."""
    fields: Tuple[Tuple[str, float, float], ...] = (
        ("GPP", 3.0, 0.99), ("TLAI", 3.0, 0.99), ("LEAFC", 1.0, 0.99),
        ("TOTVEGC", 1.0, 0.99), ("TOTSOMC", 1.0, 0.99), ("H2OSNO", 5.0, 0.99),
        ("BTRAN", 3.0, 0.99), ("TSA", 0.1, 0.999))
    min_days: int = 360


def _compare(a: np.ndarray, b: np.ndarray, lim) -> Dict:
    """Bias and daily correlation of our series a against the reference b.

    Where a number is undefined the record says why: a reference mean of 0
    gives no percentage (both 0 is agreement), and a constant series gives no
    correlation (both constant and equal is agreement).
    """
    fin = np.isfinite(a) & np.isfinite(b)
    gaps = int((~fin).sum())
    a, b = a[fin], b[fin]
    out: Dict = {"days_without_value": gaps, "notes": []}
    if not a.size:
        out.update(ours=None, reference=None, bias_pct=None, daily_r=None,
                   ok=None if lim is None else False)
        out["notes"].append("no day with a value in both runs")
        return out
    ma, mb = float(a.mean()), float(b.mean())
    if mb != 0.0:
        bias = 100.0 * (ma - mb) / abs(mb)
    elif ma == 0.0:
        bias = 0.0
    else:
        bias = None
        out["notes"].append("the reference mean is 0, ours is not: no percentage bias")
    r, r_ok = None, False
    if a.std() > 0 and b.std() > 0:
        r = float(np.corrcoef(a, b)[0, 1])
        r_ok = lim is not None and r >= lim[1]
    elif np.array_equal(a, b):
        r_ok = True
        out["notes"].append("both series constant and equal: no correlation, and none needed")
    else:
        side = "both series are" if a.std() == 0 and b.std() == 0 else \
            ("ours is" if a.std() == 0 else "the reference is")
        out["notes"].append(f"{side} constant: no correlation")
    out.update(ours=ma, reference=mb, bias_pct=bias, daily_r=r)
    if lim is None:
        out["ok"] = None
    else:
        out["ok"] = bool(gaps == 0 and bias is not None and abs(bias) <= lim[0] and r_ok)
        if gaps:
            out["notes"].append(f"{gaps} days without a value in one run")
    return out


def reference_gate(ours: G.Stream, reference: G.Stream, year: int, surface,
                   limits: ReferenceLimits = ReferenceLimits(),
                   band_edges: Sequence[float] = (1000.0, 1500.0, 2000.0)) -> Dict:
    """One calendar year of our run against a reference run on the same
    cells. A record, with 'pass' meaning every gated field is within its
    limits over at least min_days common days."""
    year = int(year)
    if not ours.grid.same_cells(reference.grid, tol_deg=CELL_TOL_DEG):
        raise GateError("the reference run is on other cells")
    s = R.read_surface(surface, ours.grid, ("PCT_NATVEG", "TOPO"))
    veg = R.vegetated(s["PCT_NATVEG"])
    # Both runs are weighted with OUR cells' area: the reference may carry
    # another source's (the Naches reference run's files hold a 1/8-degree area).
    w = ours.grid.weight_km2
    ms = R.MaskSet(ours.grid)
    ms.add("vegetated", veg, "PCT_NATVEG >= 100", [Path(surface).name])
    bands = R.elevation_bands(s["TOPO"], band_edges)
    for name, m in bands.items():
        if (veg & m).any():
            ms.add(name, veg & m, f"vegetated, TOPO {name}", [Path(surface).name])
    gated = {f: (b, r) for f, b, r in limits.fields}
    names = list(REFERENCE_FIELDS) + [f for f in gated if f not in REFERENCE_FIELDS]
    both = [f for f in names if f in ours.variables and f in reference.variables]
    ro = R.reduce_regions(ours, both, ms, [year], require_complete=False, weights=w)
    rr = R.reduce_regions(reference, both, ms, [year], require_complete=False, weights=w,
                          grid_tol_deg=CELL_TOL_DEG)
    common = sorted(set(ro.keys) & set(rr.keys))
    io = np.array([ro.keys.index(k) for k in common], dtype=int)
    ir = np.array([rr.keys.index(k) for k in common], dtype=int)
    rec: Dict = {"year": year, "days_compared": len(common), "limits": asdict(limits),
                 "fields": {}}
    reasons: List[str] = []
    for f in names:
        lim = gated.get(f)
        if f not in both:
            side = "ours" if f not in ours.variables else "reference"
            rec["fields"][f] = {"missing": side, "ok": False if lim else None}
            if lim:
                reasons.append(f"{f} missing from {side}")
            continue
        c = _compare(ro.series("vegetated", f)[io], rr.series("vegetated", f)[ir], lim)
        rec["fields"][f] = {"units": ro.units[f], "ours": _num(c["ours"]),
                            "reference": _num(c["reference"]), "bias_pct": _num(c["bias_pct"]),
                            "daily_r": _num(c["daily_r"]), "limit": list(lim) if lim else None,
                            "ok": c["ok"]}
        if c["notes"]:
            rec["fields"][f]["notes"] = c["notes"]
        if c["ok"] is False:
            bias = f"{c['bias_pct']:+.2f} %" if c["bias_pct"] is not None else "undefined"
            r = f"{c['daily_r']:.3f}" if c["daily_r"] is not None else "undefined"
            reasons.append(f"{f}: bias {bias} (limit {lim[0]}), r {r} (limit {lim[1]})"
                           + (f"; {'; '.join(c['notes'])}" if c["notes"] else ""))
    if "GPP" in both:
        rec["annual_gpp_gC_m2"] = {"ours": _num(ro.series("vegetated", "GPP")[io].sum()),
                                   "reference": _num(rr.series("vegetated", "GPP")[ir].sum())}
    rec["bands"] = {}
    for name in bands:
        if name not in ms or "GPP" not in both:
            continue
        rec["bands"][name] = {"cells": ms.mask(name).n_cells,
                              "gpp_ours": _num(ro.series(name, "GPP")[io].sum()),
                              "gpp_reference": _num(rr.series(name, "GPP")[ir].sum())}
        if "H2OSNO" in both:
            rec["bands"][name].update(
                swe_peak_ours=_num(ro.series(name, "H2OSNO")[io].max()),
                swe_peak_reference=_num(rr.series(name, "H2OSNO")[ir].max()))
    if len(common) < limits.min_days:
        reasons.append(f"only {len(common)} common days (at least {limits.min_days})")
    rec["pass"], rec["reasons"] = not reasons, reasons
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND LINE, FOR JOB SCRIPTS
# ─────────────────────────────────────────────────────────────────────────────
def _jsonable(x):
    """The record as strict JSON: NaN and infinities become None."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return _num(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _write(rec: Dict, out_dir, stage: str, job: str, status, summary: str) -> Path:
    out = Path(out_dir) / f"{stage}_{job}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(rec), indent=1, default=str, allow_nan=False))
    tmp.replace(out)
    if status:
        with open(status, "a") as f:
            f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} job={job} stage={stage} "
                    f"CHECK {'PASS' if rec['pass'] else 'FAIL'} {summary} ({out.name})\n")
    return out


def _smoke(a, rec: Dict, reasons: List[str]) -> str:
    try:
        daily = _daily_stream(a.run_dir, sorted(G.discover(a.run_dir)))
    except _GATE_ERRORS as e:
        reasons.append(f"no daily history to check: {e}")
        return ""
    rec["daily_records"] = len(daily.records)
    if rec["daily_records"] != a.days:
        reasons.append(f"{rec['daily_records']} daily records, not {a.days}")
    rec["missing_fields"] = {}
    try:
        nl, _ = case_namelist(a.case_dir, a.run_dir)
        start = daily.records[0].start
        for name, fields in requested_tapes(nl).items():
            try:
                st = G.open_stream(a.run_dir, name)
            except G.GriddedReadError as e:
                if first_write_days(nl, name, start, daily.calendar) > a.days:
                    rec["missing_fields"][name] = f"no file expected in a {a.days}-day run"
                else:
                    rec["missing_fields"][name] = [str(e)]
                    reasons.append(f"{name}: {e}")
                continue
            lack = st.lacking(fields)
            rec["missing_fields"][name] = sorted(lack)
            if lack:
                reasons.append(f"{name} lacks {sorted(lack)}")
    except _GATE_ERRORS as e:
        reasons.append(f"history tapes: {e}")
    try:
        rec["carry_over"] = carry_over(a.run_dir, a.case_dir, restart=a.restart,
                                       surface=a.surface)
        reasons += rec["carry_over"]["reasons"]
    except _GATE_ERRORS as e:
        rec["carry_over"] = {"pass": False, "reasons": [str(e)]}
        reasons.append(f"carry-over: {e}")
    return " ".join(f"{k}:med={v['median_rel_diff']:.4f}"
                    for k, v in rec["carry_over"].get("fields", {}).items()
                    if "median_rel_diff" in v)


def _reference_stream(a) -> G.Stream:
    """The reference run named on the command line: a run directory, or a
    glob of files inside it."""
    if a.reference_pattern:
        import glob
        files = sorted(glob.glob(str(Path(a.reference) / a.reference_pattern)))
        if not files:
            raise GateError(f"no reference files {a.reference_pattern} in {a.reference}")
        return G.Stream("reference", files)
    return G.open_stream(a.reference, a.reference_stream)


def _reference_year(a, y: int, surface, ref: List) -> Dict:
    """One year against the reference run: gated, or recorded only."""
    try:
        if not ref:
            ref.append(_reference_stream(a))
        if not surface:
            nl, nl_path = case_namelist(a.case_dir, a.run_dir)
            surface = (nl.get("fsurdat") or [None])[0]
        rr = reference_gate(G.open_stream(a.run_dir, a.reference_stream), ref[0], y, surface)
    except _GATE_ERRORS as e:
        rr = {"pass": False, "reasons": [f"cannot compare: {e}"]}
    rr["gated"] = not a.reference_report_only
    rr["reference"] = a.reference + (f" ({a.reference_pattern})" if a.reference_pattern else "")
    return rr


def _years(a, rec: Dict, reasons: List[str]) -> str:
    rec["years"] = {}
    ref: List = []
    for y in a.year:
        try:
            rec["years"][y] = year_check(a.run_dir, a.case_dir, y, surface=a.surface)
        except _GATE_ERRORS as e:
            rec["years"][y] = {"year": y, "pass": False, "reasons": [f"cannot check: {e}"]}
        why = job_ran_year(rec["logs"], y)
        if why:
            rec["years"][y]["pass"] = False
            rec["years"][y]["reasons"].append(why)
        if a.reference:
            rr = _reference_year(a, y, a.surface or rec["years"][y].get("surface"), ref)
            rec["years"][y]["reference"] = rr
            if rr["gated"] and not rr["pass"]:
                rec["years"][y]["pass"] = False
                rec["years"][y]["reasons"] += [f"reference: {r}" for r in rr["reasons"]]
    reasons += [f"{y}: {r}" for y, v in rec["years"].items() for r in v["reasons"]]

    def ref_note(v):
        rr = v.get("reference")
        if not rr:
            return ""
        gpp = ((rr.get("fields") or {}).get("GPP") or {}).get("bias_pct")
        return (f" ref={'pass' if rr['pass'] else 'FAIL'}{'' if rr['gated'] else '(report)'}"
                f" refGPP={_fmt(gpp, '+.2f')}%")
    return " | ".join(f"{y}: GPP={_fmt(v.get('gpp_gC_m2_yr'), '.1f')} "
                      f"peakLAI={_fmt(v.get('peak_tlai'), '.3f')} "
                      f"closure={_fmt(v.get('water_closure_frac_of_precip'), '.4g')}"
                      + ref_note(v)
                      for y, v in rec["years"].items())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("year", "smoke"))
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--job", default=os.environ.get("SLURM_JOB_ID"))
    ap.add_argument("--year", type=int, action="append", default=[])
    ap.add_argument("--days", type=int, default=5, help="smoke: daily records expected")
    ap.add_argument("--surface")
    ap.add_argument("--restart")
    ap.add_argument("--out", required=True, help="folder for the check record")
    ap.add_argument("--status", help="a STATUS.log to append one line to")
    ap.add_argument("--stage", help="name in the record's file name (default: the mode)")
    ap.add_argument("--reference", help="year: a reference run's directory, compared with "
                                        "reference_gate")
    ap.add_argument("--reference-pattern", help="year: a glob of the reference's files "
                                                "inside --reference")
    ap.add_argument("--reference-stream", default="h1",
                    help="year: the stream compared, ours and the reference run's")
    ap.add_argument("--reference-report-only", action="store_true",
                    help="year: record the comparison without gating on it")
    a = ap.parse_args(argv)
    if not a.job:
        ap.error("--job (or SLURM_JOB_ID) is required: logs are found by it")
    if a.mode == "year" and not a.year:
        ap.error("year mode needs --year")
    stage = a.stage or a.mode
    rec: Dict = {"mode": a.mode, "case_dir": a.case_dir, "run_dir": a.run_dir,
                 "time": datetime.datetime.now().isoformat(timespec="seconds")}
    reasons: List[str] = []
    summary = ""
    try:
        rec["logs"] = scan_logs(a.run_dir, a.case_dir, a.job)
        reasons += rec["logs"]["reasons"]
        summary = _years(a, rec, reasons) if a.mode == "year" else _smoke(a, rec, reasons)
    except Exception as e:           # still leave a record that names the failure
        reasons.append(f"the gate itself failed: {type(e).__name__}: {e}")
        rec["traceback"] = traceback.format_exc()
    rec["pass"], rec["reasons"] = not reasons, reasons
    hp = (rec.get("logs") or {}).get("hours_per_sim_year")
    summary = (f"hours_per_sim_year={hp:.3f} " if hp else "") + summary
    _write(rec, a.out, stage, str(a.job), a.status, summary)
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
