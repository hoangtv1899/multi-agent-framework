#!/usr/bin/env python3
"""
Per-column rows, built from the extraction artifact
mcp/elm-mcp/src/column_rows.py

    in   03_results/extracted.json  (written by extract.extract_run)
    out  one row per column — the list experiment.json publishes as `columns`

WHY THIS EXISTS AS A FUNCTION AND NOT A CLASS. It replaces ELMResultsAnalyzer,
which was 383 lines by the end and did three separable things: it looped over
the manager's experiment list, it read the history files, and it computed the
metrics. The last two were already delegated — to extract.py and
column_metrics.py — so what remained was a stateful class wrapped around an
assembly step, holding `results`, `_blocks`, `_meta`, `spinup_dropped` and
`extra_summary` so that a later method could find what an earlier one left.

THE REAL REASON IT WENT: TWO WRITERS OF ONE FILE. Both extract_run and
ELMResultsAnalyzer.extract_all called write_extracted, so `extracted.json` had
two producers reached by two paths. They did not disagree — same writer, same
spin-up constant — but that is the shape of the bug that already cost a day:
an extracted.json from one morning beside an experiment.json from the next,
351 days against 352, and nothing saying so. One producer now writes it, and
this reads what was written.

WHICH MEANS THE METRICS ARE DOWNSTREAM OF THE FILE, not of an open NetCDF
handle. That was the rule already ("every computation is downstream of the
extraction", 2026-08-13) and it was true of the arithmetic; it was not yet true
of the assembly, which still ran in the same pass as the read. Now the whole
row — identity, series, metrics — is reconstructible from one artifact by
anyone holding the run directory, whether or not scratch still exists.

NO FIELD IS INVENTED HERE. Identity comes from extract._identity, which reads
case_inputs.json (what the cases were BUILT from) and columns.json (the only
place the pin fields live). Band, soil profile, forcing_cell and soil_summary
are joined on later by the manager's _merge_column_metadata, out of
columns.json — this module does not need them and must not guess them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from extract import (EXTRACTED, RESULTS_DIR, TARGET_VARIABLES)
from column_metrics import column_metrics


# The identity fields extract._identity puts on every column's metadata. Listed
# rather than copied wholesale so a new key in the artifact cannot silently
# become a row field nobody declared.
_IDENTITY = ("scenario_name", "forcing_period", "forcing_start", "forcing_end",
             "lat", "lon", "elevation_m",
             "pinned", "station_id", "station_variable")

# `soil` IS NOT A ROW FIELD ANY MORE (2026-08-13). ELMResultsAnalyzer set it
# from the manager's experiment list, where nothing ever populated it: measured
# across every packaged run on disk, `soil` is null on 100% of columns, while
# `soil_profile` beside it is complete. Two readers were testing the null one —
# figure_registry's soil-capability probe and analyze_run's clay panels — so a
# real capability read as absent on every run. They read soil_profile now, and
# carrying a field that has never held a value is worse than not carrying it.


def _keep_last_full_year(block: Dict[str, Any]) -> Dict[str, Any]:
    """The block narrowed to the last calendar year with a full record of days.

    Returns it unchanged when no year qualifies. 300 days, not 365: the
    warm-start trim takes a fortnight off the front and the partial-day drop
    takes one off the back, so a whole simulated year is never 365 days on
    disk. The threshold only has to tell a simulated year from a stub.
    """
    dates = (block or {}).get("dates") or []
    if not dates:
        return block
    years: Dict[str, int] = {}
    for d in dates:
        years[d[:4]] = years.get(d[:4], 0) + 1
    full = [y for y, n in years.items() if n >= 300]
    if not full or len(years) < 2:
        return block
    keep = max(full)
    idx = [i for i, d in enumerate(dates) if d[:4] == keep]
    out = dict(block, dates=[dates[i] for i in idx], variables={})
    for var, blk in (block.get("variables") or {}).items():
        vals = blk.get("values") or []
        out["variables"][var] = dict(
            blk, values=[vals[i] for i in idx if i < len(vals)])
    return out


def _empty_row(name: str, meta: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """A column that produced nothing still gets a row, and says why.

    Omitting it would make a failed ensemble look like a smaller one.
    """
    row = {k: meta.get(k) for k in _IDENTITY}
    row.update({"case_name": name, "status": "failed", "error": reason,
                "variables": {}, "metrics": {},
                "case_dir": meta.get("case_dir"),
                "n_history_files": meta.get("n_history_files")})
    row.setdefault("scenario_name", name)
    return row


def load_extracted(run_dir) -> Optional[Dict[str, Any]]:
    """The artifact, or None when it is absent or unreadable."""
    p = Path(run_dir).resolve() / RESULTS_DIR / EXTRACTED
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:                                           # noqa: BLE001
        return None


def build_rows(run_dir,
               last_year_only: bool = False,
               variables: Optional[List[str]] = None
               ) -> Dict[str, Dict[str, Any]]:
    """extracted.json -> {case_name: row}, metrics computed from the series.

    Keyed by case name because that is the shape _extract_rows and _package
    have accepted since PFLOTRAN landed, and the key is the join column for
    _merge_column_metadata.
    """
    payload = load_extracted(run_dir)
    if not payload:
        return {}
    meta = (payload.get("metadata") or {})
    cols = meta.get("columns") or {}
    data = payload.get("data") or {}
    want = list(variables or meta.get("variables_requested")
                or TARGET_VARIABLES)

    rows: Dict[str, Dict[str, Any]] = {}
    for name, m in cols.items():
        if m.get("status") != "ok":
            rows[name] = _empty_row(name, m, m.get("status") or "no data")
            continue
        block = data.get(name)
        if not (block or {}).get("variables"):
            rows[name] = _empty_row(name, m, "no series in the artifact")
            continue

        if last_year_only:
            block = _keep_last_full_year(block)

        derived = column_metrics(block)
        dates = block.get("dates") or []

        # The row's `variables` keeps the shape every consumer already reads —
        # summary stats at the top, the daily series under `daily` — and both
        # halves come from the published series, so a number in the package can
        # be checked against the artifact it was computed from.
        variables_out: Dict[str, Any] = {}
        for var in want:
            blk = (block.get("variables") or {}).get(var)
            if not blk:
                variables_out[var] = None
                continue
            entry = dict((derived.get("stats") or {}).get(var) or {})
            entry["daily"] = {"units": blk.get("units"),
                              "dates": dates,
                              "values": blk.get("values")}
            for k in ("n_layers", "layer_depth_m", "layer_thickness_m"):
                if blk.get(k) is not None:
                    entry["daily"][k] = blk[k]
            variables_out[var] = entry

        row = {k: m.get(k) for k in _IDENTITY}
        row.update({
            "case_name":       name,
            "status":          "ok",
            "variables":       variables_out,
            "metrics":         derived["metrics"],
            # WHERE IT WAS READ FROM, as a directory and a count. The row used
            # to carry the full list of ~17 absolute history-file paths per
            # column — 200-odd paths into purgeable scratch, in a file meant to
            # outlive it, read by nothing. The case directory answers "which
            # run" and the count answers "how much"; the artifact's own
            # metadata holds the rest.
            "case_dir":        m.get("case_dir"),
            "n_history_files": m.get("n_history_files"),
        })
        row.setdefault("scenario_name", name)
        rows[name] = row
    return rows


def spinup_dropped(run_dir) -> Dict[str, Any]:
    """What the warm-start trim removed, for the package to record.

    A series that does not start where the simulation did must say so — a
    reader comparing this to a gauge record needs to know. Read off the
    artifact rather than remembered from the extraction pass.
    """
    payload = load_extracted(run_dir)
    if not payload:
        return {}
    meta = payload.get("metadata") or {}
    days = int(meta.get("spinup_days") or 0)
    if not days:
        return {}
    starts, firsts, dropped = [], [], 0
    for m in (meta.get("columns") or {}).values():
        if m.get("status") != "ok":
            continue
        dropped += int(m.get("n_timesteps_dropped_spinup") or 0)
        if m.get("record_start"):
            starts.append(m["record_start"])
        rng = m.get("date_range")
        if rng:
            firsts.append(rng[0])
    if not dropped:
        return {}
    return {"days": days,
            "from": min(starts) if starts else None,
            "to": min(firsts) if firsts else None,
            "timesteps_dropped": dropped,
            "reason": "warm-start relaxation; storage inherited from the CONUS "
                      "spin-up is not in equilibrium with this domain's forcing"}
