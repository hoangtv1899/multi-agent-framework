#!/usr/bin/env python3
"""Model vs observations, for PFLOTRAN columns. MEASUREMENTS ONLY.

    water_table   the column's water table (a boundary condition, from CONUS2)
                  and the modelled pressure crossing        vs well   m below surface

THE SAME SHAPE AS ELM's PACKAGE (mcp/elm-mcp/src/compare): a registry of
observable modules, each a `SPEC`, a `compare()` and a `plot()`. The dispatcher
and the shared half — reading reception's observations, pairing stations to
columns, metrics, figures — are the framework's (agents/analysis/compare_common)
and are not repeated here. What is here is what only this model knows: that its
water table is prescribed at the bottom of every column, that the water table
in its output is the depth where liquid pressure crosses atmospheric, and that
its snapshots sit at a handful of output times rather than every day.

WHERE THE OBSERVATIONS COME FROM — reception.json, and nowhere else, in the
shape reception wrote them. A synthetic well written for a check must carry
`source: synthetic` and `synthetic: true` on its record; both pass through to
every pair and every figure label so nothing downstream can mistake it for a
measurement.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.analysis import compare_common as C
from . import water_table

# The registry. A new observable is a module plus a line here.
OBSERVABLES = {m.SPEC.name: m for m in (water_table,)}

load_observations = C.load_observations
pair_stations = C.pair_stations
metrics = C.metrics
FILENAME = C.COMPARISON_FILENAME


def compare_all(rows: List[Dict], reception_json: str,
                observables: Optional[List[str]] = None,
                figure_dir: str = "") -> Dict[str, Any]:
    return C.compare_all(OBSERVABLES, rows, reception_json,
                         observables=observables, figure_dir=figure_dir)


def compare_run(rows: List[Dict], reception_json: str, out_dir: str,
                observables: Optional[List[str]] = None,
                draw: bool = True) -> Dict[str, Any]:
    return C.compare_run(OBSERVABLES, rows, reception_json, out_dir,
                         observables=observables, draw=draw)


def summarise(out: Dict[str, Any]) -> Dict[str, Any]:
    return C.summarise(OBSERVABLES, out)
