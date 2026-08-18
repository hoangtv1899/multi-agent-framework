#!/usr/bin/env python3
"""Model vs observations, for ELM columns. MEASUREMENTS ONLY.

    swe          H2OSNO                  vs snow pillow   mm
    water_table  ZWT                     vs well          m below surface
    streamflow   QOVER + QDRAI           vs gauge         mm/day
    et           QSOIL + QVEGE + QVEGT   vs flux tower    mm/day

ONE MODULE PER OBSERVABLE, and this file is only the registry. The dispatcher
and the shared half (loading, pairing, metrics, figures) moved to the
framework's agents/analysis/compare_common on 2026-08-18, because none of it
knows ELM and PFLOTRAN's package needed the same. Each module is
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

from typing import Any, Dict, List, Optional

from agents.analysis import compare_common as C
from . import et, maps, streamflow, swe, water_table

# The registry. A new observable is a module plus a line here.
OBSERVABLES = {m.SPEC.name: m for m in (swe, water_table, streamflow, et)}

# Re-exported so callers and tests have one import site for the shared pieces.
load_observations = C.load_observations
model_series = C.model_series
pair_stations = C.pair_stations
metrics = C.metrics
summarise_all = C.summarise
FILENAME = C.COMPARISON_FILENAME

# THE DRIVER LIVES IN THE FRAMEWORK NOW (agents/analysis/compare_common,
# 2026-08-18) — compare, draw, write comparison.json, summarise — generic over
# a registry. This package is the registry, ELM's observables, and its map;
# these wrappers keep the names its two callers use (the Analyzer's step 1 and
# the server's compare_to_obs tool).


def compare_all(rows: List[Dict], reception_json: str,
                observables: Optional[List[str]] = None,
                figure_dir: str = "") -> Dict[str, Any]:
    return C.compare_all(OBSERVABLES, rows, reception_json,
                         observables=observables, figure_dir=figure_dir,
                         maps=maps)


def compare_run(rows: List[Dict], reception_json: str, out_dir: str,
                observables: Optional[List[str]] = None,
                draw: bool = True) -> Dict[str, Any]:
    return C.compare_run(OBSERVABLES, rows, reception_json, out_dir,
                         observables=observables, draw=draw, maps=maps)


def summarise(out: Dict[str, Any]) -> Dict[str, Any]:
    return C.summarise(OBSERVABLES, out)
