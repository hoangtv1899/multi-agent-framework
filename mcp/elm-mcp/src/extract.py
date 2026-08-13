#!/usr/bin/env python3
"""One ELM column's daily series, straight from its history files.

READS AND RETURNS. Writes nothing, plots nothing, derives nothing. The caller
decides where the answer goes and what it means.

NO DERIVED FIELDS, EVER. `recharge_fraction`, `runoff_fraction`,
`water_budget` and the rest of the FIELD_SEMANTICS block are deliberately
absent. That block is the reason extraction was kept off the far side of the
MCP boundary on 2026-08-06: `recharge_fraction` is the recharge-vs-RUNOFF
split, not a fraction of precipitation, and reading it the obvious way was
wrong by P/(QCHARGE+QOVER) — columns draining ~0 mm/yr reported 1.00. A raw
series carries its own units and means what its name says; a ratio does not.
Anything that needs interpreting stays where the interpretation lives.

Standalone on purpose: it imports nothing from elm_results_analyzer, whose
42 kB bundled the derived metrics and plotting that this does not want. The variable definitions below are the CANONICAL ones — that
module imports them from here, so the two lists cannot drift apart.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# WHAT IS EXTRACTED
# ─────────────────────────────────────────────────────────────────────
TARGET_VARIABLES = ['QOVER', 'QCHARGE', 'TWS', 'SOILLIQ', 'ZWT', 'RAIN',
                    'H2OSNO', 'SNOW', 'QINFL', 'QDRAI', 'QSOIL', 'QVEGE',
                    'QVEGT', 'QSNOMELT', 'H2OSOI']

VARIABLE_UNITS = {
    'QOVER':    'mm/s',      # surface runoff
    'QCHARGE':  'mm/s',      # aquifer recharge (vegetated landunits only)
    'TWS':      'mm',        # total water storage
    'SOILLIQ':  'kg/m2',     # soil liquid water, PER LAYER (veg only)
    'ZWT':      'm',         # water table depth (veg only)
    'RAIN':     'mm/s',      # atmospheric forcing, not model skill
    'H2OSNO':   'mm',        # snow water equivalent, despite the ELM long_name
    'SNOW':     'mm/s',      # atmospheric forcing
    'QINFL':    'mm/s',      # infiltration
    'QDRAI':    'mm/s',      # sub-surface drainage (baseflow)
    'QSOIL':    'mm/s',      # ground evaporation      ─┐
    'QVEGE':    'mm/s',      # canopy evaporation       ├─ ET is their sum;
    'QVEGT':    'mm/s',      # canopy transpiration    ─┘  no QFLX_EVAP_TOT here
    'QSNOMELT': 'mm/s',      # snow melt — links SWE to streamflow
    'H2OSOI':   'mm3/mm3',   # volumetric soil water, PER LAYER (veg only)
}

# Fluxes are stored per SECOND and returned per DAY. Every observation this is
# compared against is daily, and a series in mm/s beside one in mm/day is how a
# factor of 86400 gets into a bias.
FLUX_VARIABLES = ('QOVER', 'QCHARGE', 'RAIN', 'SNOW', 'QINFL', 'QDRAI',
                  'QSOIL', 'QVEGE', 'QVEGT', 'QSNOMELT')
SECONDS_PER_DAY = 86400.0

# THESE KEEP THEIR LAYER DIMENSION. Flattening (time x levgrnd) into one list
# was the old behaviour and it destroys the only thing these variables are for:
# H2OSOI exists here to be compared against a soil-moisture probe or SMAP, both
# of which measure the TOP layer. Flattened, there is no way to say which of
# 5,280 numbers is the top layer on day one without assuming an ordering.
LAYERED_VARIABLES = ('SOILLIQ', 'H2OSOI')
LAYER_DEPTH_VAR = 'ZSOI'              # (levgrnd,) node depth, time-invariant
LAYER_THICKNESS_VAR = 'DZSOI'         # (levgrnd,) layer thickness, likewise

# ─────────────────────────────────────────────────────────────────────
# WARM-START RELAXATION
# ─────────────────────────────────────────────────────────────────────
# A warm start inherits storage from the CONUS spin-up, and that state is not in
# equilibrium with THIS domain's forcing. The column drains hard while it
# settles and the transient is enormous — measured on the 2019 Upper Gunnison
# run, column-mean QOVER+QDRAI was 45.82 mm/day on day 1 against a 0.179 mm/day
# Feb-Jun baseline, 256x. Averaging that into an annual number is not a
# measurement of anything.
#
# Dropped BEFORE any statistic, so the series and anything computed from it
# agree about what period they cover. The cost is that the record no longer
# starts on 1 January, which is why `date_range` travels with every column.
SPINUP_DAYS = 14

HISTORY_GLOB = 'run/*.elm.h0.*.nc'


def _sigfig(x: float, n: int = 4) -> float:
    """Round to n SIGNIFICANT figures, not n decimal places.

    Decimal places are the wrong instrument when one series is a water-table
    depth near 71.67 m and another is snow water equivalent at 0.00002 mm.
    Three decimals would zero out small-but-real recharge fluxes — 0.0005
    mm/day is 0.18 mm/yr, which matters on a column that barely recharges.
    """
    if x == 0 or not np.isfinite(x):
        return 0.0 if x == 0 else x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def _clean(v: Any) -> Optional[float]:
    """One value, JSON-safe. NaN becomes None — a gap, never a zero."""
    f = float(v)
    return None if not np.isfinite(f) else _sigfig(f)


def history_files(case_dir: str) -> List[Path]:
    """This column's h0 files, in time order. Empty if the run produced none.

    A LITERAL `$PSCRATCH` IS EXPANDED. Case directories are written into
    case_inputs.json with the variable unexpanded by one path and expanded by
    another, and an unexpanded one globs nothing — which reads as "the column
    produced no output" rather than "the path was never resolved". Folded in
    here from the copy elm_results_analyzer kept, so both callers get it.
    """
    d = Path(str(case_dir).replace("$PSCRATCH", os.environ.get("PSCRATCH", "")))
    return sorted(d.glob(HISTORY_GLOB)) if d.is_dir() else []


def open_history(files: List[Path]):
    """The h0 files as one dataset, on the settings both readers must share.

    ONE OPEN, ONE SET OF KWARGS. This call was written twice, identically, in
    this file and in elm_results_analyzer — and `combine`, `coords` and
    `compat` decide how monthly files are concatenated, so two copies drifting
    would not raise, it would quietly change what a year contains. The explicit
    values are also what silence xarray's FutureWarnings without changing
    today's behaviour.
    """
    import xarray as xr
    return xr.open_mfdataset([str(f) for f in files],
                             combine='by_coords', decode_times=True,
                             engine='netcdf4', data_vars='all',
                             coords='different', compat='no_conflicts',
                             join='outer')


def _drop_partial_days(ds) -> Tuple[Any, List[str]]:
    """Drop calendar days that were not fully sampled. Returns (ds, dates).

    A DAY WITH ONE TIMESTEP IS NOT A DAY. ELM writes 3-hourly here, so a
    complete day is eight steps — but the last day of every run holds exactly
    one, because a history timestamp marks the END of its averaging window and
    the final average of 31 December lands on 1 January. Resampling to daily
    means then gives that stub the same weight as every full day beside it.

    Measured on Brandywine col_01: dropping the single 1-step day moves QOVER
    from 139.978 to 140.320 mm/yr, and — the point of the exercise — makes the
    mean of the daily series EQUAL the mean of the raw 3-hourly record to six
    decimal places. The whole 0.2% gap between "computed while the NetCDF was
    open" and "computed from the published series" was this one day. With it
    gone, anything downstream of extracted.json can reproduce the numbers
    exactly, which is the only way a wrong number can be traced.

    COMPLETE IS THE MODAL COUNT, not a hardcoded 8, so a run written hourly or
    six-hourly is judged against its own cadence. A daily-output run has one
    step per day everywhere, the mode is 1, and nothing is dropped.
    """
    if ds is None or 'time' not in getattr(ds, 'dims', ()):
        return ds, []
    t = np.asarray(ds['time'].values)
    if t.size == 0:
        return ds, []
    days = [str(x)[:10] for x in t]
    counts: Dict[str, int] = {}
    for d in days:
        counts[d] = counts.get(d, 0) + 1
    if not counts:
        return ds, []
    # The cadence this run actually wrote, as the most common day length.
    per_day = max(set(counts.values()), key=list(counts.values()).count)
    short = sorted(d for d, n in counts.items() if n < per_day)
    if not short:
        return ds, []
    keep = np.asarray([d not in set(short) for d in days])
    if not keep.any():
        # Every day is short — the cadence is not what it looks like, and
        # dropping everything would turn a readable run into an empty one.
        return ds, []
    return ds.isel(time=keep), short


def _drop_spinup(ds, days: int) -> Tuple[Any, int]:
    """Trim the first `days` of record. Returns (ds, n_timesteps_dropped)."""
    if not days or ds is None or 'time' not in getattr(ds, 'dims', ()):
        return ds, 0
    t = np.asarray(ds['time'].values)
    if t.size == 0:
        return ds, 0
    start = np.datetime64(str(t[0])[:10]) + np.timedelta64(int(days), 'D')
    keep = np.asarray([np.datetime64(str(x)[:10]) >= start for x in t])
    if not keep.any():
        # A record shorter than the window: keep all of it rather than return
        # nothing. A short run is a finding; an empty series is a bug.
        return ds, 0
    return ds.isel(time=keep), int((~keep).sum())


def _layer_geometry(ds) -> Dict[str, Optional[List[float]]]:
    """Soil layer node depths and THICKNESSES in metres, from the history file.

    Both are in every h0 file as time-invariant (levgrnd,) fields, and both are
    needed for different questions. ZSOI says where a layer's node SITS, which
    is what a moisture probe at 10 cm is compared against. DZSOI says how THICK
    it is, which is the only way to weight layers into a column mean — ELM's
    grid runs 1.75 cm at the surface to 13.85 m at the bottom, so an unweighted
    mean over layers is dominated by the deepest one and means nothing.

    Carrying thickness is what makes "the active soil column" computable rather
    than a constant: the first ten layers sum to 3.8020 m, which is the 3.8 m
    the water-table caveat has been quoting all along.
    """
    out: Dict[str, Optional[List[float]]] = {"depth": None, "thickness": None}
    for key, var in (("depth", LAYER_DEPTH_VAR),
                     ("thickness", LAYER_THICKNESS_VAR)):
        if var not in ds:
            continue
        try:
            z = np.asarray(ds[var].squeeze().values, dtype=float)
            out[key] = [_clean(v) for v in np.atleast_1d(z)]
        except Exception:                                       # noqa: BLE001
            continue
    return out


def _daily(da, var: str) -> Tuple[List[str], Any]:
    """(dates, values) resampled from the native 3-hourly step to daily means.

    Layered variables come back (n_days, n_layers); everything else (n_days,).
    """
    d = da.squeeze().resample(time='1D').mean()
    dates = [str(t)[:10] for t in np.asarray(d['time'].values)]
    vals = np.asarray(d.values, dtype=float)
    if var in LAYERED_VARIABLES:
        vals = vals.reshape(len(dates), -1)
    else:
        vals = vals.reshape(len(dates))
    if var in FLUX_VARIABLES:
        vals = vals * SECONDS_PER_DAY
    return dates, vals


def extract_column(case_dir: str,
                   variables: Optional[List[str]] = None,
                   spinup_days: int = SPINUP_DAYS) -> Tuple[Dict, Dict]:
    """One column's daily series. Returns (data, meta).

    data — {"dates": [...], "variables": {VAR: {values, units}}}, and for a
           layered variable {values, units, layer_depth_m, n_layers} where
           `values` is one row per day, one entry per soil layer.

           ONE DATE LIST PER COLUMN, not one per variable. Every variable here
           is the same resample of the same dataset, so they share a time axis
           by construction; carrying it fifteen times cost 1.17 MB of repeated
           date strings per run against 0.08 MB hoisted. A variable whose axis
           somehow differs is NOT quietly reshaped onto the shared one — it is
           dropped and named in meta.time_axis_mismatch.

    meta — what the extraction FOUND, never what it means:
           status, n_history_files, n_timesteps, date_range,
           absent_variables, empty_variables, time_axis_mismatch.

    ABSENT AND EMPTY ARE DIFFERENT FINDINGS. A variable missing from the file
    was never written — it is not in hist_fincl1, and the fix is to request it.
    A variable that is present and entirely fill value DID run and produced
    nothing: Naches col_02 is 100% urban, and ZWT/QCHARGE/SOILLIQ/H2OSOI are
    vegetated-landunit diagnostics there, so ELM correctly wrote 1e36 for every
    timestep. Reporting both as `null` hid a column that could never take part
    in a water-table comparison, and the comparison simply said "16 columns"
    with no mention of the seventeenth.
    """
    want = list(variables or TARGET_VARIABLES)
    files = history_files(case_dir)
    meta: Dict[str, Any] = {
        'case_dir': str(case_dir),
        'n_history_files': len(files),
        'variables_requested': want,
    }
    if not Path(case_dir).is_dir():
        meta['status'] = 'case directory missing'
        return {}, meta
    if not files:
        # The model did not run, or ran and wrote nothing. Either way there is
        # no series here, and that is a fact about the RUN, not about the model.
        meta['status'] = 'no history files'
        return {}, meta

    try:
        ds = open_history(files)
    except Exception as e:                                      # noqa: BLE001
        meta['status'] = f'unreadable: {type(e).__name__}: {str(e)[:120]}'
        return {}, meta

    series: Dict[str, Any] = {}
    dates: Optional[List[str]] = None
    absent, empty, mismatch = [], [], []
    try:
        # The first day the model actually wrote, BEFORE anything is trimmed.
        # Without it the record cannot say what it dropped from, only how much
        # — and "the series starts 15 January" means nothing on its own.
        record_start = (str(np.asarray(ds['time'].values)[0])[:10]
                        if 'time' in getattr(ds, 'dims', ()) and
                        np.asarray(ds['time'].values).size else None)
        ds, dropped = _drop_spinup(ds, spinup_days)
        # AFTER the spin-up trim, because trimming can itself expose a partial
        # day at the front — and before anything is resampled, so every
        # variable is built from the same complete days.
        ds, partial = _drop_partial_days(ds)
        geom = _layer_geometry(ds)
        depths, thick = geom["depth"], geom["thickness"]
        for var in want:
            if var not in ds:
                absent.append(var)
                continue
            try:
                d, vals = _daily(ds[var], var)
            except Exception as e:                              # noqa: BLE001
                absent.append(var)
                meta.setdefault('read_errors', {})[var] = \
                    f'{type(e).__name__}: {str(e)[:80]}'
                continue
            if vals.size == 0 or not np.isfinite(vals).any():
                # Present, ran, and every value is fill. See the docstring.
                empty.append(var)
                continue
            if dates is None:
                dates = d
            elif d != dates:
                # Should be impossible — same files, same resample. Dropped
                # rather than aligned, because silently reindexing one variable
                # onto another's axis is how a series ends up describing days
                # it was not measured on.
                mismatch.append(var)
                continue
            block: Dict[str, Any] = {
                'units': ('mm/day' if var in FLUX_VARIABLES
                          else VARIABLE_UNITS.get(var, '')),
            }
            if var in LAYERED_VARIABLES:
                if depths is not None and len(depths) == vals.shape[1]:
                    block['layer_depth_m'] = depths
                if thick is not None and len(thick) == vals.shape[1]:
                    block['layer_thickness_m'] = thick
                block['n_layers'] = int(vals.shape[1])
                block['values'] = [[_clean(v) for v in row] for row in vals]
            else:
                block['values'] = [_clean(v) for v in vals]
            series[var] = block

        data = {'dates': dates or [], 'variables': series}
        meta.update({
            'status': 'ok' if series else 'no requested variable had values',
            'n_timesteps': len(dates or []),
            'date_range': [dates[0], dates[-1]] if dates else None,
            'record_start': record_start,
            'n_timesteps_dropped_spinup': dropped,
            # Named, not just counted: a reader comparing this series to a
            # gauge record has to know which calendar days are absent and why.
            'partial_days_dropped': partial,
            'absent_variables': absent,
            'empty_variables': empty,
            'time_axis_mismatch': mismatch,
        })
    finally:
        ds.close()
    return data, meta


# ─────────────────────────────────────────────────────────────────────
# THE WHOLE RUN
# ─────────────────────────────────────────────────────────────────────
EXTRACTED = 'extracted.json'
RESULTS_DIR = '03_results'


def _identity(rd: Path) -> Dict[str, Dict[str, Any]]:
    """Per-column identity, assembled from the files that already hold it.

    Coordinates come from case_inputs.json rather than columns.json even though
    both carry them: case_inputs is what the cases were BUILT from, so it is
    what the model actually ran with. columns.json is the sampling design, and
    is the only place the pin fields exist — so that is all it is read for.
    """
    out: Dict[str, Dict[str, Any]] = {}
    spec = rd / '01_inputs' / 'case_inputs.json'
    if spec.is_file():
        try:
            for c in json.loads(spec.read_text()) or []:
                if c.get('case_name'):
                    out[c['case_name']] = {
                        k: c.get(k) for k in
                        ('lat', 'lon', 'elevation_m', 'band',
                         'forcing_start', 'forcing_end', 'scenario_name',
                         # WHICH FORCING PERIOD. Added 2026-08-13, when the
                         # rows started being built from this artifact rather
                         # than from the manager's experiment list — that list
                         # carried it and the artifact did not, so every row
                         # came out `forcing_period: None`.
                         'forcing_period')}
        except Exception:                                       # noqa: BLE001
            pass
    colf = rd / 'columns.json'
    if colf.is_file():
        try:
            raw = json.loads(colf.read_text())
            for c in (raw if isinstance(raw, list) else raw.get('columns') or []):
                cid = c.get('id')
                if cid in out:
                    out[cid].update({
                        'pinned': bool(c.get('pinned')),
                        'station_id': c.get('station_id'),
                        'station_variable': c.get('station_variable')})
        except Exception:                                       # noqa: BLE001
            pass
    return out


def write_extracted(run_dir, cols_meta: Dict[str, Any], data: Dict[str, Any],
                    want_vars: List[str], spinup_days: int) -> Path:
    """Write 03_results/extracted.json. THE artifact, and the only writer.

    Called by extract_run and by ELMResultsAnalyzer, because both extract the
    same columns from the same files and only one of them used to leave a
    record. That gap was not theoretical: on 2026-08-13 a Naches run carried an
    extracted.json from the previous day beside an experiment.json from that
    morning — 352 days against 351, the partial day still in one and gone from
    the other — and nothing in either file said they disagreed. Metrics
    recomputed from the stale one came out 8x further from closing.

    "Every computation is downstream of the extraction" only means something if
    the extraction artifact is CURRENT. One writer, called by every path that
    reads history files, is what makes that true.
    """
    rd = Path(run_dir).resolve()
    out_path = rd / RESULTS_DIR / EXTRACTED
    failed = [{'case_name': c, 'reason': m.get('status')}
              for c, m in cols_meta.items() if m.get('status') != 'ok']
    empty = [{'case_name': c, 'variables': m['empty_variables']}
             for c, m in cols_meta.items() if m.get('empty_variables')]
    payload = {
        'metadata': {
            'extracted_at': datetime.now().isoformat(timespec='seconds'),
            'run_dir': str(rd),
            'variables_requested': want_vars,
            'spinup_days': spinup_days,
            'n_columns': len(cols_meta),
            'columns': cols_meta,
            'failed': failed,
            'empty': empty,
            'note': ('daily means from the native 3-hourly history files; '
                     'fluxes converted to mm/day. Raw series only — no derived '
                     'fields, by the decision of 2026-08-06.'),
        },
        'data': data,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ATOMIC. A crash mid-write must not leave a half artifact that parses.
    tmp = out_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out_path)
    return out_path


def extract_run(run_dir: str,
                columns: Optional[List[str]] = None,
                variables: Optional[List[str]] = None,
                spinup_days: int = SPINUP_DAYS,
                overwrite: bool = False) -> Dict[str, Any]:
    """Every column's daily series into 03_results/extracted.json.

    THIS IS WHAT ESTABLISHES WHAT RAN. Nothing else in the run directory
    reliably says: run_elm_ensemble returns a submission receipt written before
    the model starts, RUN_SUMMARY.json still said "pending, 0 success" for a
    run where all 17 columns finished, and check_elm_job reports an ensemble
    job as a build job whenever built_cases.json exists. The history files on
    disk are the only ground truth, and this is what reads them.

    Reuse has three cases: no artifact or overwrite -> extract everything;
    artifact and no `columns` -> reuse it untouched; artifact and `columns`
    -> extract those and MERGE, which is how one failed column is redone
    without re-reading sixteen good ones.
    """
    rd = Path(run_dir).resolve()
    out_path = rd / RESULTS_DIR / EXTRACTED
    if not rd.is_dir():
        return {'ok': False, 'error': f'no such run dir: {rd}'}

    built = rd / '01_inputs' / 'built_cases.json'
    if not built.is_file():
        return {'ok': False,
                'error': f'no built_cases.json in {rd / "01_inputs"} — the '
                         f'ensemble has not been built. Run run_elm_ensemble '
                         f'first; this reads what it produced.'}
    try:
        cases = {c['case_name']: c['case_dir']
                 for c in (json.loads(built.read_text()).get('cases') or [])
                 if c.get('case_name') and c.get('case_dir')}
    except Exception as e:                                      # noqa: BLE001
        return {'ok': False, 'error': f'unreadable built_cases.json: {e}'}
    if not cases:
        return {'ok': False, 'error': 'built_cases.json names no case dirs'}

    want_cols = [c for c in (columns or list(cases))]
    unknown = [c for c in want_cols if c not in cases]
    want_cols = [c for c in want_cols if c in cases]
    want_vars = list(variables or TARGET_VARIABLES)

    existing: Dict[str, Any] = {}
    if out_path.is_file() and not overwrite:
        try:
            existing = json.loads(out_path.read_text())
        except Exception:                                       # noqa: BLE001
            existing = {}
        if existing and not columns:
            md = existing.get('metadata') or {}
            return {'ok': True, 'reused': True,
                    'extracted_json': str(out_path),
                    'n_columns': len(md.get('columns') or {}),
                    'note': 'artifact already present; pass overwrite=true to '
                            're-read, or columns= to redo specific ones'}

    ident = _identity(rd)
    data = dict((existing.get('data') or {})) if existing else {}
    cols_meta = dict(((existing.get('metadata') or {}).get('columns') or {})) \
        if existing else {}

    for name in want_cols:
        d, m = extract_column(cases[name], variables=want_vars,
                              spinup_days=spinup_days)
        m.update(ident.get(name) or {})
        cols_meta[name] = m
        if d.get('variables'):
            data[name] = d
        else:
            data.pop(name, None)      # a redo that found nothing removes stale

    ok_cols = [c for c, m in cols_meta.items() if m.get('status') == 'ok']
    failed = [{'case_name': c, 'reason': m.get('status')}
              for c, m in cols_meta.items() if m.get('status') != 'ok']
    empty = [{'case_name': c, 'variables': m['empty_variables']}
             for c, m in cols_meta.items() if m.get('empty_variables')]
    ranges = [m['date_range'] for m in cols_meta.values() if m.get('date_range')]

    write_extracted(rd, cols_meta, data, want_vars, spinup_days)

    summary: Dict[str, Any] = {
        'ok': True, 'reused': False,
        'extracted_json': str(out_path),
        'n_columns': len(cols_meta), 'n_ok': len(ok_cols),
        'n_failed': len(failed),
        'variables': want_vars,
        'period': ([min(r[0] for r in ranges), max(r[1] for r in ranges)]
                   if ranges else None),
        'size_mb': round(out_path.stat().st_size / 1e6, 2),
    }
    if failed:
        summary['failed'] = failed
    if empty:
        # NOT a failure. col_02 is 100% urban, so its vegetated-landunit
        # variables are fill for every timestep — the column ran fine and
        # simply cannot answer a water-table question. Named so a later count
        # of "16 columns" has somewhere to point.
        summary['columns_with_empty_variables'] = empty
    if unknown:
        summary['unknown_columns'] = unknown
    if not ok_cols:
        summary['warning'] = ('no column produced a series — this is a fact '
                              'about the run, not about the model')
    return summary
