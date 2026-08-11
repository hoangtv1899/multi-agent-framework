#!/usr/bin/env python3
"""Model vs observations, for ELM columns. MEASUREMENTS ONLY.

    swe          H2OSNO                  vs snow pillow   mm
    wtd          ZWT                     vs well          m below surface
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

THE OBSERVATION CONTRACT — the caller writes these, this package reads them:

    observations.csv        station_id,variable,time,value,quality
        SNOTEL:663,swe,2019-01-01,241.3,measured
        US-NR1,et,2019-01-01,0.42,filled

    observations_meta.json  one entry per (station_id, variable):
        station_id · variable · units · lat · lon · elevation_m · source
        · in_basin · licence · drainage_area_km2 (gauges only)

CSV, not parquet: neither pyarrow nor fastparquet is installed here, and a
format the host cannot read is not a format. A FILE, not an argument, for the
reason this server already applies elsewhere — a few short strings travel
inline and results never do, and one flux tower is 490k rows.

`quality` is load-bearing. See et.py, which computes its metrics twice because
of it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from . import _common as C
from . import et, maps, streamflow, swe, wtd

# The registry. A new observable is a module plus a line here.
OBSERVABLES = {m.SPEC.name: m for m in (swe, wtd, streamflow, et)}

# Re-exported so callers and tests have one import site for the shared pieces.
load_observations = C.load_observations
model_series = C.model_series
pair_stations = C.pair_stations
metrics = C.metrics


def compare_all(rows: List[Dict], obs_csv: str, meta_json: str = "",
                observables: Optional[List[str]] = None,
                figure_dir: str = "",
                references: Optional[Dict[str, Dict[str, float]]] = None
                ) -> Dict[str, Any]:
    """Every requested observable that has both a model series and observations.

    references: static per-column priors for wtd, {case_name: {"fan": 12.3,
    "parflow_clm": 10.1}}. Ignored by the others.
    """
    series, meta, dropped = C.load_observations(obs_csv, meta_json)
    want = observables or list(OBSERVABLES)

    out: Dict[str, Any] = {
        "observables": {}, "n_observation_rows_dropped": dropped,
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
        rec = mod.compare(rows, series, meta, references=references)
        out["observables"][name] = rec
        if figure_dir and not rec.get("error"):
            # NON-FATAL. The numbers are the product; a figure that will not
            # render must not take the comparison down with it.
            try:
                p = mod.plot(rec, rows, series,
                             str(Path(figure_dir) / f"compare_{name}.png"))
                if p:
                    figures[name] = p
            except Exception as e:                              # noqa: BLE001
                figures[name] = f"failed: {type(e).__name__}: {e}"[:200]

    if figure_dir:
        try:
            p = maps.plot_all(out["observables"], rows, meta,
                              str(Path(figure_dir) / "compare_map.png"))
            if p:
                figures["map"] = p
        except Exception as e:                                  # noqa: BLE001
            figures["map"] = f"failed: {type(e).__name__}: {e}"[:200]
    out["figures"] = figures
    return out
