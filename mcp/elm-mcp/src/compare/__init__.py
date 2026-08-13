#!/usr/bin/env python3
"""Model vs observations, for ELM columns. MEASUREMENTS ONLY.

    swe          H2OSNO                  vs snow pillow   mm
    water_table  ZWT                     vs well          m below surface
    streamflow   QOVER + QDRAI           vs gauge         mm/day
    et           QSOIL + QVEGE + QVEGT   vs flux tower    mm/day

ONE MODULE PER OBSERVABLE, and this file is only the dispatcher. Each module is
the same small shape — a `SPEC`, a `compare()`, a `plot()` — so adding a fifth
observable means adding a file and one line in OBSERVABLES, never a branch in
shared code. The shared half (pairing, metrics, quality accounting) lives once
in _common, because two copies of an NSE would drift.

The split follows what is actually specific. A peak-date axis means nothing for
a water table; a log depth scale means nothing for snow; only ET has a
gap-filling model behind half its record. Those belong in their own files. An
inner join on dates does not.

WHAT THIS RETURNS AND WHAT IT DOES NOT. Numbers: paired series, per-station
metrics, and diagnostics about the pairing itself. No verdicts. "bias = -41 mm"
is a measurement; "the model underestimates snowpack" is an interpretation, and
interpretation belongs to whoever reads this.

EVERY COMPARISON IS CONTEXT, NONE IS A SKILL CLAIM (decided 2026-08-10). It is
what makes streamflow admissible: a 1-D column's point runoff and a gauge's
routed discharge are not the same quantity, so no metric between them scores
the model — but the hydrograph shape is worth seeing. Each record carries
`model_comparand`, `obs_quantity` and `colocated` so a reader can see what was
put beside what.

WHERE THE OBSERVATIONS COME FROM — reception.json, and nowhere else. Reception
queries the four data servers once, after the period is fixed, and persists the
whole payload under `observations`: streamflow, water_table, swe and et, each
station with its coordinates and its daily series. This package reads that file
in the shape reception wrote it. To refresh the observations, RE-RUN RECEPTION —
it is the component that reaches outside, and routing the refresh through it is
what keeps the observations a run is judged against identical to the ones its
brief was written from.

A FILE, not an argument, for the reason this server already applies elsewhere:
a few short strings travel inline and results never do, and one basin's gauges
are 10k rows before ET is even in the picture.

`quality` is load-bearing. See et.py, which computes its metrics twice because
of it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import _common as C
from . import et, maps, streamflow, swe, water_table

# The registry. A new observable is a module plus a line here.
OBSERVABLES = {m.SPEC.name: m for m in (swe, water_table, streamflow, et)}

# Re-exported so callers and tests have one import site for the shared pieces.
load_observations = C.load_observations
model_series = C.model_series
pair_stations = C.pair_stations
metrics = C.metrics


def compare_all(rows: List[Dict], reception_json: str,
                observables: Optional[List[str]] = None,
                figure_dir: str = "") -> Dict[str, Any]:
    """Every requested observable, whether or not it has an observation.

    reception_json: the run's reception.json, or a payload already narrowed to
    its `observations` block.

    `references` is GONE (2026-08-12). It carried static per-column priors for
    the water table — {case_name: {"fan": 12.3, "parflow_clm": 10.1}} — from a caller that
    had fetched them itself. water_table now reads Fan and ParFlow CONUS2 out of
    reception.json, which is the file the observations already come from, and
    compares them as DISTRIBUTIONS rather than per-column priors. The argument
    had no reader left; the shape it described no longer exists.
    """
    series, meta, dropped = C.load_observations(reception_json)
    domain = C.load_domain(reception_json)
    want = observables or list(OBSERVABLES)

    out: Dict[str, Any] = {
        "observables": {}, "n_observation_rows_dropped": dropped,
        "domain": domain,
        "note": ("measurements only — no verdict is offered on any of these, "
                 "and every comparison is context rather than a skill claim"),
    }
    figures: Dict[str, Any] = {}
    for name in want:
        mod = OBSERVABLES.get(name)
        if mod is None:
            out["observables"][name] = {
                "error": f"unknown observable '{name}'; have "
                         f"{sorted(OBSERVABLES)}"}
            continue
        # reception_json is passed to EVERY module and used by water_table alone:
        # Fan wells and the modelled water-table raster live in that file and
        # reach no module through `series`, because BLOCKS loads only station
        # tables and a GeoTIFF is not one. The others absorb it in **kw.
        rec = mod.compare(rows, series, meta, domain=domain,
                          reception_json=reception_json)
        out["observables"][name] = rec
        if figure_dir:
            # DRAWN EVEN WHEN THE RECORD CARRIES AN `error` (2026-08-12). The
            # guard used to be `not rec.get("error")`, which was right when an
            # error meant an empty record — and wrong the moment water_table and
            # streamflow began returning model-side findings ALONGSIDE the
            # message that no station was available. That is exactly the basin
            # where the figure is the only water-table or runoff picture there
            # is, and it was the one being skipped: Naches has no recorder well
            # in any year, so its water-table panel of three distributions was computed
            # and then never rendered. Each plot() returns None when it truly
            # has nothing to draw, so the decision belongs to the module that
            # knows what it has, not to a key that means several things.
            #
            # NON-FATAL. The numbers are the product; a figure that will not
            # render must not take the comparison down with it.
            try:
                # reception_json reaches plot() for the same reason it reaches
                # compare(): streamflow's map panel draws the DEM samples and
                # the watershed ring, and both live in that file rather than in
                # `series`. The other plots absorb it in **kw.
                p = mod.plot(rec, rows, series,
                             str(Path(figure_dir) / f"compare_{name}.png"),
                             reception_json=reception_json)
                if p:
                    figures[name] = p
            except Exception as e:                              # noqa: BLE001
                figures[name] = f"failed: {type(e).__name__}: {e}"[:200]

    if figure_dir:
        try:
            # comparison_spatial_map.png, not compare_map.png — the file is a
            # different figure now: values, model beside observation, one row
            # per observable. The old name described a siting map that no
            # longer exists.
            p = maps.plot_all(out["observables"], rows, meta,
                              str(Path(figure_dir) /
                                  "comparison_spatial_map.png"),
                              series=series, reception_json=reception_json)
            if p:
                figures["map"] = p
        except Exception as e:                                  # noqa: BLE001
            figures["map"] = f"failed: {type(e).__name__}: {e}"[:200]
    out["figures"] = figures
    return out


FILENAME = "comparison.json"


def compare_run(rows: List[Dict], reception_json: str, out_dir: str,
                observables: Optional[List[str]] = None,
                draw: bool = True) -> Dict[str, Any]:
    """compare_all, written down, plus the summary. The whole thing, once.

    TWO CALLERS, ONE BODY (2026-08-12). The MCP tool `compare_to_obs` and the
    Analyzer's step 1 both want the same four moves — compare, draw, write
    comparison.json, summarise — and both used to do them themselves. Nothing
    forced the two to agree: the file name lived in this package as one
    constant and in the Analyzer as another, and the Analyzer's own comparison
    was a whole second implementation of the comparison itself. That is the
    duplication this package was split out to end, so the last four lines of it
    end here too.

    `draw` exists for a caller that only needs the numbers — a step 3 re-run
    against an archived study, a test — and does not want to pay for five
    renders it will not look at.

    Returns {comparison, summary, path, figures}. The full record is on disk at
    `path`; `summary` is the part small enough to travel or to put in a prompt.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = compare_all(rows, reception_json, observables=observables,
                      figure_dir=str(out_dir) if draw else "")
    path = out_dir / FILENAME
    path.write_text(json.dumps(out, indent=2, default=str))
    return {"comparison": out, "summary": summarise(out),
            "path": str(path), "figures": out.get("figures") or {}}


# ── what travels back to the caller ──────────────────────────────────────────
_MAX_STR = 160          # a note belongs in the file, not in a return value
_MAX_LIST = 24          # column names travel; a daily series does not
_MAX_DEPTH = 2          # a headline block, and the sub-blocks that carry it

# PER-COLUMN TABLES AND RAW SERIES NEVER TRAVEL, whatever their size. They are
# named rather than caught by a length rule because the rule cannot tell a
# 16-column table from a 16-name list of findings, and both are useful — one on
# disk, one inline.
_BULK = {"values", "per_column", "range_m_per_column",
         "frac_days_below_per_column", "towers", "dates"}


def _compact(value: Any, depth: int = 0) -> Any:
    """One record block, shrunk to what can travel.

    The full record is on disk and its path travels with this; what comes back
    inline has to stay small enough to read. Numbers and short strings are
    kept, long prose is dropped (it is in the file), a short list of scalars
    survives — `columns: [col_01, col_04, …]` is the useful half of a finding —
    and nesting is followed two levels: a headline block, then the sub-blocks
    that carry its numbers. TWO, not one: at one level `distributions` kept its
    prose and dropped model/fan_2013/parflow entirely, and `how_water_leaves`
    came back as a unit string and a note. The findings live one level in.
    """
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_STR else None
    if isinstance(value, list):
        if len(value) > _MAX_LIST or any(isinstance(v, (dict, list))
                                         for v in value):
            return None
        return value
    if isinstance(value, dict) and depth < _MAX_DEPTH:
        kept = {k: (None if k in _BULK else _compact(v, depth + 1))
                for k, v in value.items()}
        return {k: v for k, v in kept.items() if v is not None and v != {}}
    return None


def summarise(out: Dict[str, Any]) -> Dict[str, Any]:
    """The comparison as a few dozen numbers — for a caller, not for a file.

    LIVES HERE, NOT IN THE SERVER (2026-08-12). The MCP tool used to build this
    itself, reaching into `rec["pairs"][i]["columns"]` and `rec["assignment"]`
    — a second reader of a shape only this package defines. When every
    observable matched first and stopped emitting nested columns, that loop
    broke on a KeyError, and nothing in the package could have told it so. One
    reader, next to the writer.

    THE MODEL-SIDE FINDINGS TRAVEL, AND THAT IS THE POINT. `error` used to be
    the whole summary for an observable with no station, so a caller was told
    "no gauge in this basin" and never that ten of seventeen columns drained
    nothing all year. Each module declares which of its blocks are headlines;
    they come back compacted, beside the error rather than instead of it.
    """
    summary: Dict[str, Any] = {}
    for name, rec in (out.get("observables") or {}).items():
        mod = OBSERVABLES.get(name)
        spec = getattr(mod, "SPEC", None)
        entry: Dict[str, Any] = {
            "units": rec.get("units"), "colocated": rec.get("colocated"),
            # WHAT WAS PUT BESIDE WHAT — the two facts that make every number
            # below readable, and they used to stay behind in the file. Added
            # 2026-08-12 after a live Naches analysis: the Analyzer's own
            # figure step, seeing only "streamflow, mm/day", defined the
            # drainage limb as QDRAI + QCHARGE, reported an ensemble total of
            # 639.8 mm beside this record's 254.723 mm, and called the same
            # column both "no drainage at all" and "824.7 mm". QCHARGE is
            # recharge INTO the aquifer store, not water leaving the column.
            # The comparand says so in one line; nothing downstream could have
            # known it otherwise.
            "model_comparand": rec.get("model_comparand"),
            "obs_quantity": rec.get("obs_quantity"),
            "n_columns": rec.get("n_columns_with_series"),
            "n_stations": rec.get("n_stations"),
        }
        if rec.get("error"):
            entry["error"] = rec["error"]
        if rec.get("skipped"):
            entry["skipped"] = rec["skipped"]

        # `pairs` for the three that match one column to one station, `gauges`
        # for streamflow, which matches nothing and stands the ensemble mean
        # against each gauge in turn.
        matched = []
        for e in (rec.get("pairs") or rec.get("gauges") or []):
            if not e.get("n_days"):
                continue
            m = e.get("metrics") or {}
            matched.append({
                "station_id": e.get("station_id"),
                **({"column": e["case_name"]} if e.get("case_name") else {}),
                "n_days": e.get("n_days"), "overlap": e.get("overlap"),
                **({"separation_km": e["separation_km"]}
                   if e.get("separation_km") is not None else {}),
                **{k: m.get(k) for k in ("bias", "rmse", "nse", "kge")},
                **({"frac_measured": (e.get("obs_quality") or {})
                    .get("frac_measured")} if e.get("obs_quality") else {}),
                # THE KEY CARRIES THE SIGN CONVENTION (2026-08-12). The record
                # calls this `best_offset_days` and explains beside it that a
                # positive offset means the model's water arrives AFTER the
                # gauge saw it — but the explanation stays in the file, and the
                # summary is what travels. A live Naches analysis read +22 and
                # +17 and reported the model as "17-22 days early", inverting
                # a melt-timing result. A name that states the direction cannot
                # be read backwards.
                **({"model_days_later_than_gauge": (e.get("timing") or {})
                    .get("best_offset_days")} if e.get("timing") else {}),
            })
        if matched:
            entry["matched"] = matched
        for key, n in (("unpaired_stations", "n_unpaired_stations"),
                       ("unmatched_columns", "n_unmatched_columns")):
            if rec.get(key):
                entry[n] = len(rec[key])
        if rec.get("stations_excluded_outside_basin"):
            entry["stations_excluded_outside_basin"] = \
                rec["stations_excluded_outside_basin"]

        for key in (getattr(spec, "headlines", ()) or ()):
            block = _compact(rec.get(key))
            if block not in (None, {}, []):
                entry[key] = block
        summary[name] = {k: v for k, v in entry.items() if v is not None}
    return summary
