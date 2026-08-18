#!/usr/bin/env python3
"""Evapotranspiration: model QSOIL + QVEGE + QVEGT against a flux tower.

THE OBSERVABLE BEST MATCHED TO A 1-D COLUMN. ET is vertical, local to the tower
footprint (hundreds of metres, the same order as a column), and ELM computes it
directly. Where a stream gauge cannot be co-located with a column however it is
labelled, a flux tower can.

QSOIL + QVEGE + QVEGT, not QFLX_EVAP_TOT: this ELM build registers the
components — ground evaporation, canopy evaporation, transpiration — and does
not write the total. Summing the wrong subset silently under-reports ET.

AND YET THIS COMPARISON HAS NEVER RUN ON DATA (measured 2026-08-12). Reception
fetches TOWERS, not series, and three links of that chain are open:

  data_gather.py calls get_et with `with_values` off, from when there were no
      credentials — "asking for values would return ok=false and lose the tower
      list with it";
  get_et with values ON refuses even with credentials — "fetching series
      through get_et is not wired yet — call request_flux_data instead";
  request_flux_data submits a download in the account holder's name, emails the
      PIs of LEGACY sites, returns file URLs rather than series, and has never
      been run against a real account.

So `values_available: false` travels with every tower list, load_observations
finds no dates and drops every tower, and this module reports an absence. That
is the truth, but "no observations in the table" is the WRONG WORDS for it: it
reads as an empty basin when the usual state is towers found and never
requested. This module now separates the three cases — none in the basin, none
operating, or found and unfetched — because a coverage fact and a credentials
gap are acted on differently.

WHAT THIS MODULE ADDS WHEN THE SERIES DO ARRIVE: metrics computed TWICE, once on
the measured pairs alone and once on everything. That is the US-NR1 lesson made
operational rather than written in a docstring — gap-filled annual ET there is
464 mm in 2016 against 89 mm from the measured half-hours alone, because 36% of
the year was observed.

BUT THE SHAPE UPSTREAM CANNOT YET CARRY IT, and saying so here is the point of
knowing. load_observations reads ONE `quality` per station and stamps it on
every point, while measured and gap-filled alternate DAY BY DAY. Until reception
carries quality per day, `metrics_measured_only` will come back either identical
to the all-pairs metrics or empty — never wrong, and never the finding it was
built to make. Left in place deliberately: the day the series arrive this is the
half that already works.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.analysis import compare_common as C

SPEC = C.Spec(
    name="et", model_vars=["QSOIL", "QVEGE", "QVEGT"], units="mm/day",
    comparand=("QSOIL + QVEGE + QVEGT — ground evaporation, canopy "
               "evaporation, transpiration. This ELM build registers the "
               "components, not a single QFLX_EVAP_TOT"),
    obs_quantity="eddy-covariance latent heat flux, expressed as water depth",
    colocated=True, pair_on="distance",
    # REASONED, NOT MEASURED, and the record says which. Every other limit in
    # this package was set by measuring the alternatives on a real basin; here
    # there is nothing to measure on — two of thirteen chain-eval basins had a
    # tower operating in their simulation year, and neither returned a series.
    # The number comes from the footprint argument instead: a tower integrates
    # hundreds of metres, an ELM column is a 1 km cell, and ET varies with
    # vegetation over that distance. Leaving it undeclared is not the safe
    # option — that is exactly what handed streamflow a gauge 51.9 km from its
    # column. Re-measure this the first time a basin pairs.
    max_km=10.0,
    headlines=("where_et_comes_from", "tower_discovery"))

# Water leaving a column over the WHOLE RUN, below which it has not left. The
# same threshold streamflow.py uses, for the same reason: exact zero is a knife
# edge a recompile can cross.
NEGLIGIBLE_MM = 0.01


# ═════════════════════════════════════════════════════════════════════════════
# 1.  MODEL-SIDE FINDINGS — true whether or not a single tower exists
# ═════════════════════════════════════════════════════════════════════════════
def _where_et_comes_from(model_columns: List[Dict]) -> Dict[str, Any]:
    """Which of the three paths the water left by, per column, over the run.

        QSOIL   evaporation from the ground
        QVEGE   evaporation of water sitting on the canopy
        QVEGT   transpiration, water the plants moved

    THE HALF OF THE WATER BALANCE THAT HOLDS TOGETHER, and that is worth
    knowing next to the other two modules. Naches 1979, the same seventeen
    columns whose runoff spanned 0.0 to 1,123 mm and ten of which drained
    nothing at all:

        ET total          129 mm (col_02)  ...  533 mm (col_10),  median 402
        transpiration     19.5% (col_15)   ...  60.1% (col_01),   median 49%
        columns with no transpiration                             0 of 17

    A four-fold band with a sensible transpiration share throughout, against
    three orders of magnitude and ten dead columns in the drainage. Same model,
    same year, same columns — so whatever is wrong with the subsurface is not
    wrong with the surface, and a reader should meet that before any metric.

    COUNTED, NEVER CUT, and computed before anything is compared: ET is the
    observable most likely to have no observation at all, so this is usually
    the whole record.
    """
    parts = {v: C.model_series(model_columns, [v]) for v in SPEC.model_vars}
    per_column, totals, no_transp, no_ground = {}, [], [], []
    for case in sorted(set().union(*(set(p) for p in parts.values()))):
        got = {v: (sum(p[case]["values"]) if case in p else None)
               for v, p in parts.items()}
        if any(v is None for v in got.values()):
            per_column[case] = {
                "note": "one of QSOIL/QVEGE/QVEGT is missing for this column"}
            continue
        total = sum(got.values())
        totals.append(total)
        per_column[case] = {
            "ground_mm": round(got["QSOIL"], 3),
            "canopy_mm": round(got["QVEGE"], 3),
            "transpiration_mm": round(got["QVEGT"], 3),
            "total_mm": round(total, 3),
            "frac_transpiration": (round(got["QVEGT"] / total, 4)
                                   if total > 0 else None)}
        if got["QVEGT"] < NEGLIGIBLE_MM:
            no_transp.append(case)
        if got["QSOIL"] < NEGLIGIBLE_MM:
            no_ground.append(case)
    return {
        "units": "mm over the run",
        "negligible_below_mm": NEGLIGIBLE_MM,
        "per_column": per_column,
        "spread_between_columns": _quantiles(totals),
        "columns_with_no_transpiration": {
            "n": len(no_transp), "of": len(totals), "columns": no_transp,
            "note": ("the plants in these columns moved no water all run. "
                     "Nothing is excluded on this basis — but a basin ET or "
                     "water-use number should be read knowing how many of "
                     "these went into it")},
        "columns_with_no_ground_evaporation": {
            "n": len(no_ground), "of": len(totals), "columns": no_ground},
        "note": ("where each column's ET came from, over the whole run. True "
                 "with no tower in the basin, which is why it is computed "
                 "before anything is compared — and for this observable that "
                 "is nearly always the case"),
    }


def _quantiles(values: List[float], nd: int = 3) -> Dict[str, Any]:
    """n, range and quartiles. The shape streamflow.py and water_table.py use."""
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0}
    n = len(v)
    return {"n": n, "min": round(v[0], nd), "p25": round(v[n // 4], nd),
            "median": round(v[n // 2], nd), "p75": round(v[(3 * n) // 4], nd),
            "max": round(v[-1], nd), "mean": round(sum(v) / n, nd)}


# ═════════════════════════════════════════════════════════════════════════════
# 2.  WHY THERE IS NOTHING TO COMPARE — three different answers
# ═════════════════════════════════════════════════════════════════════════════
def _tower_discovery(reception_json: Optional[str]) -> Dict[str, Any]:
    """What reception found, as distinct from what it fetched.

    READ FROM reception.json DIRECTLY, because it cannot arrive any other way:
    load_observations keeps a station only if it has dates, and a tower with no
    series has none — so by the time this module sees `observations`, a basin
    with three towers and a basin with none look identical.

    That identity is the bug this fixes. "No observations of 'et' in the table"
    was reported for both, and they are not the same finding: one is a basin
    with no eddy-covariance site in it, the other is a fetch that was never
    made. The first is a fact about the world and the second is a to-do.
    """
    if not reception_json:
        return {}
    try:
        obs = (json.loads(Path(reception_json).read_text())
               .get("observations") or {})
    except Exception:                                           # noqa: BLE001
        return {}
    block = obs.get("et")
    if not isinstance(block, dict):
        return {}
    towers = block.get("towers") or []
    return {
        "n_in_bbox": block.get("n_in_bbox"),
        "n_operating": block.get("n_operating"),
        "n_with_released_data": block.get("n_with_released_data"),
        "values_available": block.get("values_available"),
        "values_note": block.get("values_note"),
        "fetch_ok": block.get("ok"), "fetch_error": block.get("error"),
        "towers": [{k: t.get(k) for k in
                    ("id", "name", "lat", "lon", "igbp", "elevation_m",
                     "in_basin", "operating")
                    if t.get(k) is not None} for t in towers],
    }


def _no_series_reason(disc: Dict[str, Any]) -> str:
    """The message that fits what actually happened."""
    if not disc:
        return ("no tower series reached this comparison, and no reception "
                "file was given, so whether any tower exists in this domain "
                "could not be checked")
    n_box = disc.get("n_in_bbox")
    n_run = disc.get("n_operating")
    towers = disc.get("towers") or []
    if disc.get("fetch_ok") is False:
        return (f"the AmeriFlux fetch itself failed, so nothing is known about "
                f"towers here: {str(disc.get('fetch_error'))[:200]}. This is a "
                f"failed fetch, not an absence of towers.")
    if not n_box:
        return ("no AmeriFlux tower sits in this bounding box, so there is no "
                "ET observation to compare against. A fact about the basin "
                "and the period, not a failed fetch — AmeriFlux began in the "
                "1990s, so any earlier simulation year has none anywhere.")
    if not n_run:
        return (f"{n_box} AmeriFlux tower(s) sit in this bounding box but none "
                f"was operating in the simulated period, so none can validate "
                f"it. A coverage fact, not a failed fetch.")
    return (
        f"{len(towers) or n_run} tower(s) were operating here, and reception "
        f"fetched NO SERIES for them: it discovers towers only "
        f"(values_available: {disc.get('values_available')}). AmeriFlux flux "
        f"data needs a registered account, and the fetch is not wired through "
        f"to the series — get_et refers callers to request_flux_data, which "
        f"submits a download in the account holder's name. THIS IS AN "
        f"UNFETCHED OBSERVATION, NOT AN ABSENT ONE: the towers are listed "
        f"under `tower_discovery` with their coordinates and land cover.")


# ═════════════════════════════════════════════════════════════════════════════
# 3.  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def compare(model_columns: List[Dict], observations: Dict, station_meta: Dict,
            reception_json: Optional[str] = None, **kw) -> Dict[str, Any]:
    """One column to one tower, then the numbers — plus where the ET came from.

    model_columns  the extracted rows, one dict per column
    observations   {(station_id, variable): {dates, values, quality}}
    station_meta   {(station_id, variable): {lat, lon, igbp, in_basin…}}
    reception_json the run's reception.json, for what was discovered but not
                   fetched — the usual state of this observable
    """
    model = C.model_series(model_columns, SPEC.model_vars)
    rec: Dict[str, Any] = {
        "observable": SPEC.name, "units": SPEC.units,
        "model_comparand": SPEC.comparand, "obs_quantity": SPEC.obs_quantity,
        "colocated": SPEC.colocated,
        "n_columns_with_series": len(model),
        "pairs": [],
    }
    if not model:
        rec["error"] = (
            f"no column has a daily series for {'+'.join(SPEC.model_vars)} — "
            f"the extraction either has not run or predates daily series. This "
            f"is an absence of model output, not of agreement.")
        return rec

    # ── BEFORE ANY EARLY RETURN ─────────────────────────────────────────────
    rec["where_et_comes_from"] = _where_et_comes_from(model_columns)
    rec["model_period"] = C.span(sorted({d for m in model.values()
                                         for d in m["dates"]}))

    # ── OUTSIDE THE DIVIDE, OUT OF THE COMPARISON ───────────────────────────
    # Reception's tag, applied and never recomputed. ONLY AN EXPLICIT False
    # EXCLUDES: None means nobody could check, and unchecked is not outside.
    stations = C.stations_for(observations, SPEC.name)
    tags = [(station_meta.get((sid, SPEC.name)) or {}).get("in_basin")
            for sid in stations]
    outside = sorted(sid for sid in stations
                     if (station_meta.get((sid, SPEC.name)) or {}).get("in_basin")
                     is False)
    for sid in outside:
        stations.pop(sid, None)
    rec["stations_excluded_outside_basin"] = outside
    rec["n_stations"] = len(stations)
    if sum(1 for t in tags if t is None):
        rec["in_basin_unchecked"] = {
            "n_stations": sum(1 for t in tags if t is None),
            "note": ("these towers carry no in_basin flag, so none was "
                     "excluded. Unchecked is not outside — but nor is it "
                     "inside. Re-run reception for this domain to tag them.")}

    if not stations:
        # THE THREE CASES, TOLD APART. What reception discovered is reported
        # either way, so a basin with towers and no series never again reads
        # like a basin with no towers.
        disc = _tower_discovery(reception_json)
        if disc:
            rec["tower_discovery"] = disc
        rec["error"] = (
            (f"all {len(outside)} tower(s) fell outside the watershed and were "
             f"excluded, so nothing was compared: {', '.join(outside)}")
            if outside else _no_series_reason(disc))
        rec["skipped"] = "no tower series"
        return rec

    # ── MATCH FIRST ─────────────────────────────────────────────────────────
    station_rows = [{"station_id": sid,
                     **{k: (station_meta.get((sid, SPEC.name)) or {}).get(k)
                        for k in ("lat", "lon", "elevation_m")}}
                    for sid in stations]
    column_rows = [{"case_name": case,
                    **{k: m.get(k) for k in
                       ("lat", "lon", "elevation_m", "pinned", "station_id",
                        "station_variable")}}
                   for case, m in model.items()]
    matched, unpaired, unmatched = C.pair_stations(
        station_rows, column_rows, on=SPEC.pair_on,
        max_delta_m=SPEC.max_delta_m, max_km=SPEC.max_km, observable=SPEC.name)
    rec.update({
        "matched_on": SPEC.pair_on, "max_separation_km": SPEC.max_km,
        "limit_source": ("declared for 'et' on its Spec from the footprint "
                         "argument, NOT measured on a paired basin — no basin "
                         "has yet returned a tower series to measure on"),
        "n_pinned": sum(1 for a in matched if a.get("matched_on") == "pinned"),
        "matched_in_order": ("one pass over the columns in name order; a tower "
                             "leaves the pool when a column takes it, so "
                             "col_01 has first refusal"),
        "unpaired_stations": unpaired, "unmatched_columns": unmatched})

    # ── THEN COMPARE, ONLY WHAT WAS MATCHED ─────────────────────────────────
    for a in matched:
        sid, case = a.get("station_id"), a.get("case_name")
        obs, m = stations.get(sid), model.get(case)
        if not (obs and m):
            continue
        st = station_meta.get((sid, SPEC.name), {})
        entry: Dict[str, Any] = {
            "station_id": sid, "case_name": case,
            "matched_on": a.get("matched_on"),
            "separation_km": a.get("separation_km"),
            "obs_period": C.span(obs["dates"]), "n_obs": len(obs["dates"]),
            "in_basin": st.get("in_basin"), "source": st.get("source"),
            "licence": st.get("licence"),
            # LAND COVER TRAVELS, AND DOES NOT FILTER. ET is governed by
            # vegetation more than by distance, so the tower's IGBP class is
            # the natural analogue of SWE's elevation limit — and it cannot be
            # used as one: the columns carry elevation, band and soil texture,
            # and no plant functional type at all. Reported so a reader can see
            # a grassland tower standing against a forest column; not applied,
            # because nothing on the model side could be compared with it.
            "tower_igbp": st.get("igbp"),
        }
        if st.get("units") and st["units"] != SPEC.units:
            entry["units_mismatch"] = (
                f"station reports {st['units']}, this comparison is in "
                f"{SPEC.units} — NOT converted; fix the table")
        dates, mm, oo, qq = C.pair(m, obs)      # inner join, never interpolated
        if not dates:
            entry.update({"n_days": 0, "note": "no shared dates"})
            rec["pairs"].append(entry)
            continue
        meas, _ = C.split_quality(qq)
        entry.update({
            "n_days": len(dates), "overlap": C.span(dates),
            "metrics": C.metrics(mm, oo), "obs_quality": C.quality(qq),
            # THE SAME COMPARISON TWICE. If these two disagree, the difference
            # IS the finding — it says how much of the agreement is the
            # tower's and how much is its gap-filling model's. See the module
            # docstring for why this cannot yet differ: quality arrives once
            # per station, not once per day.
            "metrics_measured_only": (
                C.metrics([mm[i] for i in meas], [oo[i] for i in meas])
                if meas else
                {"n": 0, "note": "no measured pairs — every point is filled"}),
        })
        rec["pairs"].append(entry)

    rec["n_pairs_total"] = sum(e.get("n_days", 0) for e in rec["pairs"])
    if not rec["pairs"]:
        rec["error"] = (
            f"no tower could be matched to a column within {SPEC.max_km:.0f} "
            f"km. {len(unpaired)} tower(s) and {len(unmatched)} column(s) went "
            f"unmatched, each carrying the number that disqualified it. This "
            f"is a siting result, not a disagreement.")
    return rec


# ═════════════════════════════════════════════════════════════════════════════
# 4.  THE FIGURE
# ═════════════════════════════════════════════════════════════════════════════
GROUND_COLOUR = "#A4522A"       # evaporation from the ground
CANOPY_COLOUR = "#3A5B78"       # water evaporating off the leaves
TRANSP_COLOUR = "#2C6A5C"       # water the plants moved
FILLED_COLOUR = "#C1440E"       # gap-filled observations, hollow


def _colour(i: int) -> str:
    import matplotlib.pyplot as plt
    cyc = plt.rcParams["axes.prop_cycle"].by_key().get("color") or ["#1f77b4"]
    return cyc[i % len(cyc)]


def plot(rec: Dict, model_columns: List[Dict], observations: Dict,
         out_path: str, reception_json: Optional[str] = None,
         **kw) -> Optional[str]:
    """Three panels, and the last two are drawn with no tower at all.

      1  ET AGAINST TIME. Every column in grey, matched ones in their tower's
         colour, the tower's own days as points — filled where the instrument
         measured, HOLLOW where the gap-filling model supplied the number. That
         distinction is the reason this observable has its own figure.

      2  WHERE EACH COLUMN'S ET CAME FROM. Ground evaporation, canopy
         evaporation and transpiration stacked per column, ordered by total.

      3  WHERE THOSE COLUMNS ARE. Mean ET in place, named, over the same ground
         the sampling design uses.

    Panels 2 and 3 are why this draws at all. ET is the observable most often
    left with nothing to stand beside — no basin has yet returned a tower
    series — and under the old shape that meant `return None` and no figure.
    """
    model = C.model_series(model_columns, SPEC.model_vars)
    came_from = (rec.get("where_et_comes_from") or {}).get("per_column") or {}
    if not (model or came_from):
        return None
    import datetime as dt
    fig, (ax1, ax2, ax3) = C.new_figure(ncols=3, width=18.0)
    d = lambda s: dt.date.fromisoformat(s)                      # noqa: E731
    stations = C.stations_for(observations, SPEC.name)
    pairs = [e for e in (rec.get("pairs") or []) if e.get("n_days")]

    # ── panel 1: ET against time ────────────────────────────────────────────
    matched_cases = {e["case_name"] for e in pairs}
    for i, (case, m) in enumerate(model.items()):
        if case in matched_cases:
            continue
        ax1.plot([d(x) for x in m["dates"]], m["values"], lw=0.7,
                 color="#9AA7B0", alpha=0.7, zorder=1,
                 label=(f"{len(model) - len(matched_cases)} unmatched columns"
                        if i == 0 else None))
    for i, e in enumerate(pairs):
        col = _colour(i)
        m = model.get(e["case_name"]) or {}
        if m:
            ax1.plot([d(x) for x in m["dates"]], m["values"], lw=1.6,
                     color=col, zorder=2, label=f"{e['case_name']} (model)")
        obs = stations.get(e["station_id"]) or {}
        meas, fill = C.split_quality(obs.get("quality") or [])
        od, ov = obs.get("dates") or [], obs.get("values") or []
        if meas:
            ax1.scatter([d(od[k]) for k in meas], [ov[k] for k in meas], s=14,
                        color=col, zorder=3, label=f"{e['station_id']}")
        if fill:
            ax1.scatter([d(od[k]) for k in fill], [ov[k] for k in fill], s=14,
                        facecolors="none", edgecolors=FILLED_COLOUR,
                        linewidths=0.7, zorder=3,
                        label=f"{e['station_id']} (gap-filled)")
    ax1.set_ylabel(f"ET  [{SPEC.units}]")
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_title(f"{len(model)} columns, {len(pairs)} matched to a tower"
                  if pairs else
                  f"{len(model)} columns — no tower series to compare")
    C.legend(ax1)

    # ── panel 2: where each column's ET came from ───────────────────────────
    rows = [(c, v.get("ground_mm") or 0.0, v.get("canopy_mm") or 0.0,
             v.get("transpiration_mm") or 0.0)
            for c, v in came_from.items() if v.get("total_mm") is not None]
    rows.sort(key=lambda t: t[1] + t[2] + t[3], reverse=True)
    if rows:
        x = range(len(rows))
        g = [r[1] for r in rows]
        c_ = [r[2] for r in rows]
        t_ = [r[3] for r in rows]
        ax2.bar(x, g, color=GROUND_COLOUR, label="from the ground (QSOIL)")
        ax2.bar(x, c_, bottom=g, color=CANOPY_COLOUR,
                label="off the canopy (QVEGE)")
        ax2.bar(x, t_, bottom=[a + b for a, b in zip(g, c_)],
                color=TRANSP_COLOUR, label="through the plants (QVEGT)")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels([r[0] for r in rows], rotation=90, fontsize=8)
        C.legend(ax2)
    ax2.set_ylabel("water leaving over the run  [mm]")
    ax2.set_title("where each column's ET came from")

    # ── panel 3: where those columns are ────────────────────────────────────
    C.column_map(fig, ax3, model,
                 lambda c, m: (sum(m["values"]) / len(m["values"])
                               if m["values"] else None),
                 f"mean ET  [{SPEC.units}]",
                 reception_json or kw.get("reception_json"),
                 # NOT A GREEN RAMP, though ET asks for one. YlGn and Greens
                 # both start at near-white, and the driest columns then
                 # disappear into a grey hillshade — measured on Naches, where
                 # col_01, col_02 and col_05 rendered as white dots on white
                 # ground. viridis is dark at the low end and legible on both.
                 cmap="viridis")
    return C.save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAP POINTS — this module's own values, in place. LAYOUT BELONGS TO maps.py.
# ─────────────────────────────────────────────────────────────────────────────
def map_points(rec, model_columns, observations, station_meta, **kw):
    """One row: flux towers beside the columns, both mean ET.

    The left panel is usually empty and the row is drawn anyway. Two of
    thirteen chain-eval basins had a tower operating, so an absent ET
    observation is the normal case — and a row that shows the model alone,
    labelled, states that far better than a row silently dropped.
    """
    model = C.model_series(model_columns, SPEC.model_vars)
    obs = []
    for sid, s in C.stations_for(observations, SPEC.name).items():
        m = station_meta.get((sid, SPEC.name)) or {}
        if m.get("lat") is None or m.get("in_basin") is False or not s["values"]:
            continue
        obs.append((m["lon"], m["lat"], sum(s["values"]) / len(s["values"]), sid))
    mod = [(v["lon"], v["lat"], sum(v["values"]) / len(v["values"]), c)
           for c, v in model.items()
           if v.get("lat") is not None and v["values"]]
    return {"label": f"mean ET  [{SPEC.units}]", "log": False,
            "panels": [{"title": "AmeriFlux towers", "points": obs},
                       {"title": "ELM columns", "points": mod}]}
