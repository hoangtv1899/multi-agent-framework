#!/usr/bin/env python3
"""Water table: model ZWT against wells, and against two static fields.

THREE SOURCES, TWO KINDS OF QUESTION, AND THEY ARE NOT MIXED.

    a recorder well   a daily SERIES, this year, at a point   ->  PAIRED
    Fan et al. 2013   one 1927-2009 mean per site             ->  DISTRIBUTION
    ParFlow CONUS2    a simulated steady state, any point     ->  DISTRIBUTION

Only the first can be matched to a column and compared day against day. The
other two say where the water table SITS in this basin, not what it did this
year, and matching them one to one would invent a correspondence that is not
there — Fan has 131 sites inside Brandywine against 12 columns, and choosing 12
of them would be a decision dressed up as a measurement.

So they are compared as DISTRIBUTIONS: does the model's water table live in the
range this basin's water tables live in? That needs no pairing, and it is the
only water-table answer available in a basin with no recorder well — which is
most of them. At Naches, where nothing was compared at all before, the three
distributions read:

    model    n=16   p25   5.47   median  21.46   p75   58.91   max  70.24
    fan_2013 n=50   p25   6.00   median  11.66   p75   27.04   max  73.76
    parflow  n=17   p25   0.04   median  66.36   p75 171.04   max 295.15

MATCHED FIRST, THEN COMPARED (2026-08-12), as in swe.py: every well used to be
compared against every column and the one-to-one match computed afterwards,
which leaves every discarded comparison in the record where the flattering one
is always available.

NOTHING HERE FILTERS ANYTHING. Two ways a column's water table can be
untrustworthy are counted and reported, and every column stays in every
comparison. A column frozen at 65 m while its well rises two metres is the
clearest evidence the model has a problem; dropping it would delete the finding
and leave only the columns that behave.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.analysis import compare_common as C

SPEC = C.Spec(
    name="water_table", model_vars=["ZWT"], units="m",
    comparand="ZWT, diagnosed water-table depth (positive down)",
    obs_quantity="depth to water in a well (positive down)",
    colocated=True, pair_on="distance",
    # DISTANCE ONLY, 5 km, set by the user after measuring the alternatives on
    # Naches: with no limit the column loop hands every column SOME well, and
    # col_09 stood against one 37.1 km away while two wells served seven and six
    # columns between them. At 5 km: 3 pairs, worst separation 4.3 km, no well
    # used twice. 10 km buys two more pairs at 8.8 km, which is a long way to
    # carry a water table.
    #
    # THE WELLS DO CARRY ELEVATION NOW — 21 of 21 at Brandywine, spanning
    # 1-89 m — which they did not when this Spec was written; the fetch gained
    # it when it became recorder-only. An elevation filter is therefore possible
    # where it was impossible before. Deliberately not added: the number would
    # have to be measured the way max_km was, and SWE's 150 m is a snow lapse
    # rate with nothing to say about water tables.
    max_km=5.0,
    headlines=("columns_with_unmoving_water_table", "below_active_soil",
               "distributions"))

# ELM's hydrologically active soil column. Below this, ZWT is diagnosed from an
# unconfined aquifer rather than simulated — a fact about where the model's
# answer came from, counted rather than averaged away.
ACTIVE_SOIL_DEPTH_M = 3.8

# A water table moving less than this across the whole run is not participating.
UNMOVING_RANGE_M = 0.01


# ═════════════════════════════════════════════════════════════════════════════
# 1.  MODEL-SIDE FINDINGS — true whether or not a single well exists
# ═════════════════════════════════════════════════════════════════════════════
def _unmoving_water_table(model: Dict) -> Dict[str, Any]:
    """Columns whose water table is the same number every day. COUNTED, not cut.

    Was `static_water_table_columns`, which collided with `water_table_static`
    — reception's name for Fan's long-term well data. One word, two unrelated
    meanings, one file.

    ELM computes soil water down to about 3.8 m; below that the water table sits
    in a store the soil barely exchanges with. A column whose water table starts
    at 65 m therefore never hears about the weather: measured on Naches 1979,
    col_01 sat at exactly 65.49 m for all 365 days while col_14, at 4.2 m, moved
    3.05 m over the same year. Ask the first one how much water drained to
    groundwater and the answer is a fixed boundary, not physics — it comes back
    as a number and looks like a result.

    Four of sixteen Naches columns are in that state, which no basin mean would
    ever show. That is the whole reason to count them.
    """
    unmoving, ranges = [], {}
    for case, m in model.items():
        vals = [v for v in m["values"] if v is not None]
        if not vals:
            continue
        rng = max(vals) - min(vals)
        ranges[case] = round(rng, 4)
        if rng < UNMOVING_RANGE_M:
            unmoving.append(case)
    return {
        "n": len(unmoving), "of": len(ranges), "columns": sorted(unmoving),
        "threshold_m": UNMOVING_RANGE_M, "range_m_per_column": ranges,
        "note": ("the water table in these columns varies by less than the "
                 "threshold across the whole run, so the soil column and the "
                 "water table never interact. Nothing is excluded on this "
                 "basis — a drainage or recharge number averaged over the "
                 "basin should be read knowing how many of these went into it")}


def _below_active_soil(model: Dict) -> Dict[str, Any]:
    """How much of the model's own water table sits below the simulated soil.

    A HEADLINE, not a footnote. This used to be computed per pair and buried
    inside it, so a reader met the metrics first. At Brandywine the observed
    depths run 0.43-35.8 m with a median of 12.5 m and only 5 of 21 wells sit
    inside 3.8 m — so for most of that basin the model side of the comparison is
    a DIAGNOSED aquifer level rather than a simulated one, and that belongs in
    front of any bias or RMSE rather than behind it.
    """
    frac, deep_mean = {}, []
    for case, m in model.items():
        vals = [v for v in m["values"] if v is not None]
        if not vals:
            continue
        frac[case] = round(sum(1 for v in vals if v > ACTIVE_SOIL_DEPTH_M)
                           / len(vals), 4)
        if sum(vals) / len(vals) > ACTIVE_SOIL_DEPTH_M:
            deep_mean.append(case)
    return {
        "active_soil_depth_m": ACTIVE_SOIL_DEPTH_M,
        "n_columns_mean_below": len(deep_mean), "of": len(frac),
        "columns_mean_below": sorted(deep_mean),
        "frac_days_below_per_column": frac,
        "note": ("below the active soil depth ELM DIAGNOSES the water table "
                 "from an unconfined aquifer instead of simulating it, so a "
                 "comparison there is against a different kind of quantity "
                 "than one in the top few metres")}


# ═════════════════════════════════════════════════════════════════════════════
# 2.  THE THREE DISTRIBUTIONS — no pairing, no scoring
# ═════════════════════════════════════════════════════════════════════════════
def _quantiles(values: List[float]) -> Optional[Dict[str, Any]]:
    """n, range, quartiles, and how many sit inside the simulated soil."""
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    n = len(v)
    return {"n": n, "min": round(v[0], 3), "p25": round(v[n // 4], 3),
            "median": round(v[n // 2], 3), "p75": round(v[(3 * n) // 4], 3),
            "max": round(v[-1], 3),
            "n_inside_active_soil": sum(1 for x in v if x <= ACTIVE_SOIL_DEPTH_M),
            "values": [round(x, 3) for x in v]}


def _distributions(model: Dict, model_columns: List[Dict],
                   reception_json: Optional[str]) -> Dict[str, Any]:
    """Where the water table sits in this basin, three ways.

        model      mean ZWT per column                     N = n_columns
        fan_2013   one long-term mean per in-basin well     N = 131 at Brandywine
        parflow    CONUS2 steady state at each column       N = n_columns

    MODEL AND PARFLOW ARE AT THE SAME POINTS, so those two are the same places
    measured twice and the difference IS meaningful column by column. FAN IS
    NOT — it is wherever somebody drilled — so it stays a basin-level
    distribution and nothing may be differenced against it per column.

    MEASURED AND SIMULATED STAY APART. Fan is real well records; ParFlow is a
    continental hydrologic simulation. They are separate keys and are never
    merged into one "reference", which is what the code this replaces did.

    reception.json is read directly, because neither block reaches this module
    any other way: `BLOCKS` in _common loads only the recorder wells, and the
    modelled field is a GeoTIFF rather than a table.
    """
    out: Dict[str, Any] = {
        "note": ("where the water table SITS, not what it did this year. No "
                 "pairing and no score: a 1927-2009 mean and a simulated "
                 "steady state cannot be matched to one modelled year at a "
                 "point, but they can say whether the model's water table "
                 "lives in the range this basin's water tables live in"),
        "same_points": ("model and parflow are sampled at the SAME column "
                        "locations, so their difference is meaningful per "
                        "column; fan_2013 is wherever wells exist and is a "
                        "basin distribution only"),
    }

    # a) the model's own water table, one number per column
    means = []
    for m in model.values():
        vals = [v for v in m["values"] if v is not None]
        if vals:
            means.append(sum(vals) / len(vals))
    out["model"] = {"kind": "modelled, mean ZWT per column over the run",
                    **(_quantiles(means) or {"n": 0})}

    if not reception_json:
        out["unavailable"] = ("no reception.json was passed, so neither the "
                              "Fan wells nor the modelled field could be read")
        return out
    rj = Path(reception_json)

    # b) Fan's wells — MEASURED, and only those inside the divide
    try:
        obs = (json.loads(rj.read_text()).get("observations") or {})
        block = obs.get("water_table_static") or {}
        wells = [w for w in (block.get("wells") or [])
                 if w.get("in_basin") is not False and w.get("wtd_m") is not None]
        out["fan_2013"] = {
            "kind": "measured, one long-term mean per well site",
            "period": block.get("period"),
            "n_wells_in_bbox": block.get("n_wells"),
            "min_records_filter": block.get("min_records"),
            **(_quantiles([w["wtd_m"] for w in wells]) or {"n": 0})}
    except Exception as e:                                      # noqa: BLE001
        out["fan_2013"] = {"error": f"{type(e).__name__}: {e}"[:200]}

    # c) ParFlow CONUS2 — SIMULATED, at each column's own point.
    # Read from the GeoTIFF reception wrote, through the one reader, so this
    # cannot drift from the cell the rest of the framework would have got.
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
        from core.static_wtd import describe as _describe
        lats = [r.get("lat") for r in model_columns if r.get("lat") is not None]
        lons = [r.get("lon") for r in model_columns if r.get("lon") is not None]
        d = _describe(rj, lats, lons)
        vals = [p["wtd_m"] for p in d.get("points") or []
                if p.get("wtd_m") is not None]
        out["parflow_conus2"] = {
            "kind": "simulated, ParFlow CONUS2 steady state at each column",
            "raster": d.get("path"), "resolution_m": d.get("resolution_m"),
            "n_outside_raster": d.get("n_outside"),
            "n_nodata": d.get("n_nodata"),
            **(_quantiles(vals) or {"n": 0})}
        if not d.get("ok"):
            out["parflow_conus2"] = {"error": d.get("error")}
    except Exception as e:                                      # noqa: BLE001
        out["parflow_conus2"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 3.  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def compare(model_columns: List[Dict], observations: Dict, station_meta: Dict,
            reception_json: Optional[str] = None, **kw) -> Dict[str, Any]:
    """One column to one well, then the numbers — plus where the basin sits.

    model_columns  the extracted rows, one dict per column
    observations   {(station_id, variable): {dates, values, quality}}
    station_meta   {(station_id, variable): {lat, lon, elevation_m, in_basin…}}
    reception_json the run's reception.json, for the Fan wells and the raster
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
            "no column has a daily series for ZWT — the extraction either has "
            "not run or predates daily series. This is an absence of model "
            "output, not of agreement.")
        return rec

    # ── BEFORE ANY EARLY RETURN ─────────────────────────────────────────────
    # These need no observation to be true, and the basin most likely to have
    # no recorder well is exactly the basin where they are the whole answer.
    # Gunnison is the case that forces it: all ten wells fell outside the
    # divide, so the paired comparison had nothing — and 14 of its 19 columns
    # still had a water table that never moved.
    rec["columns_with_unmoving_water_table"] = _unmoving_water_table(model)
    rec["below_active_soil"] = _below_active_soil(model)
    rec["distributions"] = _distributions(model, model_columns, reception_json)
    rec["model_period"] = C.span(sorted({d for m in model.values()
                                         for d in m["dates"]}))

    # ── OUTSIDE THE DIVIDE, OUT OF THE COMPARISON ───────────────────────────
    # Applying reception's tag, never recomputing it. ONLY AN EXPLICIT False
    # EXCLUDES: None means nobody could check, and unchecked must not read as
    # failed.
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
            "note": ("these wells carry no in_basin flag, so none was "
                     "excluded. Unchecked is not outside — but nor is it "
                     "inside. Re-run reception for this domain to tag them.")}

    if not stations:
        # WHY THERE IS NOTHING TO COMPARE, in the words that fit this
        # observable. wtd is fetched from RECORDER wells only, so an empty
        # table usually means the basin has no well logging a daily level
        # rather than that the fetch failed. Naches has none in any year;
        # Brandywine 2010 has 21.
        rec["error"] = (
            (f"all {len(outside)} well(s) fell outside the watershed and were "
             f"excluded, so no series was compared: {', '.join(outside)}")
            if outside else
            "no well in this domain records a daily water-table series for "
            "this period, so no series was compared. A recorder well is what "
            "says how the water table MOVED; where it typically SITS is in "
            "`distributions` above, which does not need one. This is a fact "
            "about the basin, not a failed comparison.")
        rec["skipped"] = "no recorder wells"
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
        max_delta_m=SPEC.max_delta_m, max_km=SPEC.max_km,
        observable=SPEC.name)
    rec.update({
        "matched_on": SPEC.pair_on, "max_separation_km": SPEC.max_km,
        "max_delta_elevation_m": SPEC.max_delta_m,
        "n_pinned": sum(1 for a in matched if a.get("matched_on") == "pinned"),
        "matched_in_order": ("one pass over the columns in name order; a well "
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
            "delta_elevation_m": a.get("delta_elevation_m"),
            "obs_period": C.span(obs["dates"]), "n_obs": len(obs["dates"]),
            "in_basin": st.get("in_basin"), "source": st.get("source"),
            "licence": st.get("licence"),
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
        entry.update({"n_days": len(dates), "overlap": C.span(dates),
                      "metrics": C.metrics(mm, oo),
                      "obs_quality": C.quality(qq)})
        # HOW MUCH OF THIS PAIR CAME FROM BELOW THE SIMULATED SOIL. The
        # basin-wide version is in `below_active_soil`; this is the same
        # question asked of the days actually compared.
        entry["frac_below_active_soil"] = round(
            sum(1 for v in mm if v is not None and v > ACTIVE_SOIL_DEPTH_M)
            / len(mm), 4)
        rec["pairs"].append(entry)

    rec["n_pairs_total"] = sum(e.get("n_days", 0) for e in rec["pairs"])
    if not rec["pairs"]:
        rec["error"] = (
            f"no well could be matched to a column within {SPEC.max_km:.0f} km. "
            f"{len(unpaired)} well(s) and {len(unmatched)} column(s) went "
            f"unmatched, each carrying the number that disqualified it. This "
            f"is a siting result, not a disagreement.")
    return rec


# ═════════════════════════════════════════════════════════════════════════════
# 4.  THE FIGURE
# ═════════════════════════════════════════════════════════════════════════════
def _yscale(values: List[float]) -> str:
    """'log' when the depths span orders of magnitude, else 'linear'.

    A warm-started column can legitimately sit at 60 m while a valley well sits
    at 2 m. On a linear axis the shallow half of the basin is a flat line at the
    bottom of the panel and nothing about it can be read.
    """
    vs = [v for v in values if v is not None and v > 0]
    if len(vs) < 2:
        return "linear"
    return "log" if max(vs) / max(min(vs), 1e-6) > 50 else "linear"


def _colour(i: int) -> str:
    import matplotlib.pyplot as plt
    cyc = plt.rcParams["axes.prop_cycle"].by_key().get("color") or ["#1f77b4"]
    return cyc[i % len(cyc)]


DIST_STYLE = [("model", "modelled\nZWT per column", "#2C6A5C"),
              ("fan_2013", "measured\nFan wells", "#3A5B78"),
              ("parflow_conus2", "simulated\nParFlow CONUS2", "#6B4A86")]


def plot(rec: Dict, model_columns: List[Dict], observations: Dict,
         out_path: str, **kw) -> Optional[str]:
    """Depth against time, where the basin's water table sits, and where the
    columns are.

    NEITHER OF THE LAST TWO NEEDS A WELL, and that is the point. The water table
    is the observable most often left with nothing to stand beside — Naches has
    no recorder well in any year — and these are the only water-table panels
    such a basin gets.

    THE 1:1 SCATTER OF MATCHED DAYS IS GONE (2026-08-12, user's call). It
    answered "how close are the pairs" for the two or three columns that had a
    well, using the whole middle of the figure to do it; the map that replaced
    it answers "what did the other fourteen do, and where". The pair metrics are
    all still in the record, per pair, which is where a number belongs.
    """
    import random
    model = C.model_series(model_columns, SPEC.model_vars)
    pairs = [e for e in (rec.get("pairs") or []) if e.get("n_days")]
    dist = rec.get("distributions") or {}
    have_dist = any((dist.get(k) or {}).get("values") for k, _, _ in DIST_STYLE)
    if not (model or pairs or have_dist):
        return None
    import datetime as dt
    fig, (ax1, ax2, ax3) = C.new_figure(ncols=3, width=18.0)
    d = lambda s: dt.date.fromisoformat(s)                      # noqa: E731
    stations = C.stations_for(observations, SPEC.name)

    # ── panel 1: every column, matched ones in their well's colour ──────────
    allv, matched_cases = [], {e["case_name"] for e in pairs}
    for i, (case, m) in enumerate(model.items()):
        allv += [v for v in m["values"] if v is not None]
        if case in matched_cases:
            continue
        ax1.plot([d(x) for x in m["dates"]], m["values"], lw=0.7,
                 color="#9AA7B0", alpha=0.7, zorder=1,
                 label=f"{len(model) - len(matched_cases)} unmatched columns"
                       if i == 0 else None)
    for i, e in enumerate(pairs):
        col = _colour(i)
        m = model.get(e["case_name"]) or {}
        if m:
            ax1.plot([d(x) for x in m["dates"]], m["values"], lw=1.6,
                     color=col, zorder=2, label=f"{e['case_name']} (model)")
        obs = stations.get(e["station_id"]) or {}
        if obs.get("dates"):
            ax1.scatter([d(x) for x in obs["dates"]], obs["values"], s=13,
                        color=col, zorder=3, label=e["station_id"])
            allv += [v for v in obs["values"] if v is not None]
    ax1.axhline(ACTIVE_SOIL_DEPTH_M, ls=":", lw=1.2, color="#A4522A", zorder=2,
                label=f"active soil {ACTIVE_SOIL_DEPTH_M} m")
    ax1.set_yscale(_yscale(allv))
    ax1.invert_yaxis()                      # depth: down the page is deeper
    ax1.set_ylabel(f"depth to water  [{SPEC.units}]")
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_title(f"{len(model)} columns, {len(pairs)} matched to a well")
    C.legend(ax1)

    # ── panel 2: the three distributions ────────────────────────────────────
    # A BOX AND ITS OWN POINTS. A histogram of twelve columns is noise, and the
    # three groups differ in size by an order of magnitude (12 columns against
    # 131 wells), which a shared-bin histogram would misrepresent.
    rng = random.Random(0)              # deterministic jitter: same figure twice
    ticks, labels, seen = [], [], []
    for i, (key, label, colour) in enumerate(DIST_STYLE):
        vals = (dist.get(key) or {}).get("values") or []
        ticks.append(i)
        labels.append(f"{label}\nn={len(vals)}" if vals else f"{label}\n—")
        if not vals:
            continue
        seen += vals
        ax2.boxplot([vals], positions=[i], widths=0.42, showfliers=False,
                    medianprops=dict(color=colour, lw=1.8),
                    boxprops=dict(color=colour), whiskerprops=dict(color=colour),
                    capprops=dict(color=colour))
        ax2.scatter([i + rng.uniform(-0.17, 0.17) for _ in vals], vals,
                    s=14, alpha=0.55, color=colour, zorder=3)
    if seen:
        ax2.axhline(ACTIVE_SOIL_DEPTH_M, ls=":", lw=1.2, color="#A4522A",
                    zorder=1, label=f"active soil {ACTIVE_SOIL_DEPTH_M} m")
        ax2.set_yscale(_yscale(seen))
        ax2.invert_yaxis()
        C.legend(ax2)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_xlim(-0.6, len(DIST_STYLE) - 0.4)
    ax2.set_ylabel(f"depth to water  [{SPEC.units}]")
    ax2.set_title("where the water table sits")

    # ── panel 3: where those columns are ────────────────────────────────────
    # THE SAME MODEL NUMBERS AS THE FIRST BOX IN PANEL 2, put back in place.
    # Panel 2 says the model's water tables span 4 m to 70 m at Naches; this
    # says which column is which, and whether the deep ones share a corner of
    # the basin or sit next to the shallow ones.
    C.column_map(fig, ax3, model,
                 lambda c, m: (sum(m["values"]) / len(m["values"])
                               if m["values"] else None),
                 f"mean depth to water  [{SPEC.units}]", kw.get("reception_json"),
                 cmap="cividis_r")
    return C.save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAP POINTS — this module's own values, in place. LAYOUT BELONGS TO maps.py.
# ─────────────────────────────────────────────────────────────────────────────
def map_points(rec, model_columns, observations, station_meta,
               reception_json=None, **kw):
    """One row: where the water table is documented, beside where ELM puts it.

    THREE SOURCES, TWO PANELS — the same squeeze the distributions panel has.
    Fan takes the left slot because it exists nearly everywhere (299 sites at
    Naches, which has no recorder well in any year), and the recorder wells
    ride on top of it as an overlay in their own marker. A basin with no well
    simply gets no overlay, rather than an empty third panel that reads as
    "measured nothing".
    """
    model = C.model_series(model_columns, SPEC.model_vars)
    fan = []
    if reception_json:
        try:
            obs = (json.loads(Path(reception_json).read_text())
                   .get("observations") or {})
            for w in ((obs.get("water_table_static") or {}).get("wells") or []):
                if (w.get("in_basin") is not False and w.get("wtd_m") is not None
                        and w.get("lat") is not None):
                    fan.append((w["lon"], w["lat"], w["wtd_m"], w.get("id")))
        except Exception:                                       # noqa: BLE001
            pass
    wells = []
    for sid, s in C.stations_for(observations, SPEC.name).items():
        m = station_meta.get((sid, SPEC.name)) or {}
        if m.get("lat") is None or m.get("in_basin") is False or not s["values"]:
            continue
        wells.append((m["lon"], m["lat"], sum(s["values"]) / len(s["values"]), sid))
    mod = [(v["lon"], v["lat"], sum(v["values"]) / len(v["values"]), c)
           for c, v in model.items()
           if v.get("lat") is not None and v["values"]]
    return {"label": f"depth to water  [{SPEC.units}]", "log": True,
            "panels": [{"title": "Fan 2013 long-term mean", "points": fan,
                        "overlay": ({"title": "USGS recorder wells",
                                     "points": wells, "marker": "D"}
                                    if wells else None)},
                       {"title": "ELM columns", "points": mod}]}
