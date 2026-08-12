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


def compare_all(rows: List[Dict], reception_json: str,
                observables: Optional[List[str]] = None,
                figure_dir: str = "",
                references: Optional[Dict[str, Dict[str, float]]] = None
                ) -> Dict[str, Any]:
    """Every requested observable that has both a model series and observations.

    reception_json: the run's reception.json, or a payload already narrowed to
    its `observations` block.

    references: static per-column priors for wtd, {case_name: {"fan": 12.3,
    "parflow_clm": 10.1}}. Ignored by the others.
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
        # reception_json is passed to EVERY module and used by wtd alone: the
        # Fan wells and the modelled water-table raster live in that file and
        # reach no module through `series`, because BLOCKS loads only station
        # tables and a GeoTIFF is not one. The others absorb it in **kw.
        rec = mod.compare(rows, series, meta, references=references,
                          domain=domain, reception_json=reception_json)
        out["observables"][name] = rec
        if figure_dir:
            # DRAWN EVEN WHEN THE RECORD CARRIES AN `error` (2026-08-12). The
            # guard used to be `not rec.get("error")`, which was right when an
            # error meant an empty record — and wrong the moment wtd and
            # streamflow began returning model-side findings ALONGSIDE the
            # message that no station was available. That is exactly the basin
            # where the figure is the only water-table or runoff picture there
            # is, and it was the one being skipped: Naches has no recorder well
            # in any year, so its wtd panel of three distributions was computed
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
            p = maps.plot_all(out["observables"], rows, meta,
                              str(Path(figure_dir) / "compare_map.png"))
            if p:
                figures["map"] = p
        except Exception as e:                                  # noqa: BLE001
            figures["map"] = f"failed: {type(e).__name__}: {e}"[:200]
    out["figures"] = figures
    return out
