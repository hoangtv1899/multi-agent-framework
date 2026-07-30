#!/usr/bin/env python3
"""
Analyzer step 1d — the combined spatial figure
src/agents/analysis/step1_maps.py

    in   the three step-1 validator records + ctx
    out  one grid: a row per observable, non-ELM source left, ELM right

WHY THIS IS ITS OWN MODULE. It reads all three validator records, so it cannot
live inside any one of them without that validator importing its siblings.
Every point it draws comes from the validators' own map_points(), so this file
decides LAYOUT and nothing else — it never recomputes a mean, a coordinate, or
an exclusion. If the combined figure and a standalone figure ever disagree, the
bug is in the validator they share, not here.

WHAT THE ROWS ARE, AND WHAT THEY ARE NOT.

    SWE          SNOTEL vs ELM, both mm. A real comparison: same quantity,
                 same units, collocated to within an elevation band.
    streamflow   USGS vs ELM, both mm/day of specific discharge. The units
                 match and the MEANING does not — a gauge integrates a routed
                 catchment, a column is 1 m2 of local generation with no
                 routing and no run-on. See the validator's
                 unrouted_columns_vs_integrated_gauge caveat.
    WTD          Fan 2013 vs ELM, both m. Fan is a modelled equilibrium prior,
                 NOT a measurement. In-basin USGS wells are overlaid on the
                 Fan panel when they exist; on the 2019 Upper Gunnison run all
                 ten lay outside the watershed, so there is no measured
                 water-table depth for that basin at all.

That is why the columns carry no "observed"/"modelled" headers. Two of the
three left-hand panels are observations and the third is a model, and a header
claiming otherwise would misrepresent exactly the row with the weakest
evidence. Each panel names its own source instead.

A SCALE PER ROW. mm, mm/day and m cannot share a colourbar. Within a row the
two panels do share, which is the entire reason to draw them adjacent.

Rows whose validators returned nothing are dropped rather than drawn empty.
"""
from typing import Any, Dict, List, Optional

from agents.analysis import step1_validate_swe as _swe          # noqa: E402
from agents.analysis import step1_validate_streamflow as _flow  # noqa: E402
from agents.analysis import step1_validate_wtd as _wtd          # noqa: E402
from agents.analysis.step1_geo import plot_grid                 # noqa: E402


def build_rows(ctx,
               swe: Optional[Dict[str, Any]] = None,
               streamflow: Optional[Dict[str, Any]] = None,
               wtd: Optional[Dict[str, Any]] = None
               ) -> List[Dict[str, Any]]:
    """The row specification, separated from drawing so it can be asserted on.

    A test can check that the WTD row carries a well overlay, or that the
    streamflow gauges are sized, without rendering a figure and reading pixels.
    """
    rows: List[Dict[str, Any]] = []

    if swe:
        obs, mod = _swe.map_points(swe, ctx)
        rows.append({
            "label": "mean SWE (mm)", "log": False,
            "panels": [{"title": "SNOTEL", "points": obs},
                       {"title": "ELM", "points": mod}]})

    if streamflow:
        obs, mod, sizes = _flow.map_points(streamflow, ctx)
        rows.append({
            # log: mean specific discharge spans orders of magnitude across a
            # basin, and linear would flatten everything below the largest
            # gauge into one colour.
            "label": "mean runoff (mm/day)", "log": True,
            "panels": [{"title": "USGS", "points": obs, "sizes": sizes},
                       {"title": "ELM", "points": mod}]})

    if wtd:
        wells, fan, model, wsizes = _wtd.map_points(wtd, ctx)
        rows.append({
            "label": "water-table depth (m)", "log": True,
            "panels": [
                # Wells ride on the Fan panel rather than taking a third
                # column: three sources, two slots, and a column that is empty
                # on most basins reads as "measured nothing".
                {"title": "Fan 2013", "points": fan,
                 "overlay": ({"title": "USGS wells", "points": wells,
                              "marker": "D"} if wells else None)},
                {"title": "ELM", "points": model}]})

    return rows


def plot_all(ctx, out_path,
             swe: Optional[Dict[str, Any]] = None,
             streamflow: Optional[Dict[str, Any]] = None,
             wtd: Optional[Dict[str, Any]] = None,
             **kw) -> str:
    """One figure: every observable that has data, over the same ground."""
    return plot_grid(build_rows(ctx, swe=swe, streamflow=streamflow, wtd=wtd),
                     ctx.data.get("boundary") or [], out_path, **kw)
