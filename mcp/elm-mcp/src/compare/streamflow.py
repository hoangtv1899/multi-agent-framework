#!/usr/bin/env python3
"""Streamflow: the ensemble's mean runoff against a gauge's discharge per area.

THE ONE OBSERVABLE THAT DOES NOT PAIR (2026-08-12).

A snow pillow and a well measure a point, and a column IS a point, so "find the
nearest one" is a real question and swe.py and wtd.py ask it. A gauge measures
every drop of water that came off the ground upstream of it, after the river has
carried it there. That measurement has no location — it has an AREA — so there
is no nearest column to find, and picking one is picking an answer.

Measured on Naches 1979, where the code this replaces did exactly that: the
outlet gauge drains 2,437 km², 85% of the basin, and was handed col_01 because
col_01 sits 8.5 km away. Over their 258 shared days:

    gauge                        263.9 mm      r  --
    mean of all 17 columns       217.0 mm      r  +0.55
    col_01 alone                   1.4 mm      r  -0.06

col_01 produces almost no runoff. The number in the record was a negative
correlation and a 190-fold gap, and neither is about the model — both are about
which of seventeen columns happened to be near the gauge. The second gauge got
a column 51.9 km away, because no distance limit was ever declared here.

SO: ONE MODEL SERIES, the mean over every column, against each in-basin gauge.
No pairing, no bijection, no 10,353-entry cross product. Same shape as wtd.py —
the model-side findings run FIRST and survive a basin with no gauge at all.

THIS IS WHAT THE PLANNER ALREADY PROMISES, and compare never delivered.
planner.txt: "Never pin a streamflow gauge… Streamflow is still validated — as a
basin-aggregate comparison against the ensemble, which is what it always was.
Emit it with `stations: []`". The Naches plan did exactly that, emitted no
stations, and standard_compare then matched a gauge to col_01 by distance anyway.

STILL CONTEXT, NOT A SKILL CLAIM, and more so than before. A mean over sampled
columns is not routed discharge either; it is merely the same KIND of quantity,
which one column was not. `colocated: false` travels with every number here.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, List, Optional, Tuple

from . import _common as C

SPEC = C.Spec(
    name="streamflow", model_vars=["QOVER", "QDRAI"], units="mm/day",
    comparand=("QOVER + QDRAI averaged over every column — point surface "
               "runoff plus subsurface drainage, unrouted"),
    obs_quantity="gauge discharge per unit contributing area",
    colocated=False,
    # NO PAIRING, AND THEREFORE NO PAIRING LIMIT. Every other Spec declares
    # max_km because the question is "which station is close enough"; here the
    # question does not arise. Written as "none" rather than left absent, so
    # nothing downstream reads a missing limit as an unlimited one — which is
    # precisely how a gauge ended up matched to a column 51.9 km away.
    pair_on="none")

COLUMN_AREA_M2 = 1.0

# How far to slide the two series past each other looking for the offset that
# lines them up. 30 days: on Naches the best offset for the outlet gauge was
# already at 14 d and still improving at the edge of a ±15 d search, and a
# melt-driven basin can run a fortnight late without anything being broken.
MAX_OFFSET_DAYS = 30

# An offset is only considered when it still has this many days under it, so a
# correlation cannot be won on a handful of days at the end of the record.
MIN_OFFSET_DAYS = 30

# How often flow of each size happens: the flow exceeded on 5% of days is the
# flood end, on 95% the recession end. (Points on the flow-duration curve.)
PERCENT_OF_DAYS = (5, 10, 25, 50, 75, 90, 95)

# Water leaving a column over the WHOLE RUN, below which it has not left. A
# threshold rather than a test for exactly zero, for the reason wtd.py uses one
# for an unmoving water table: at Naches six columns drain exactly 0.0 mm and
# four more drain under a thousandth of a millimetre in a year, and those four
# are the same finding. Exact zero is also a knife edge a recompile can cross.
NEGLIGIBLE_MM = 0.01


def _quantiles(values: List[float], nd: int = 3) -> Dict[str, Any]:
    """n, range and quartiles. The same shape wtd.py reports."""
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0}
    n = len(v)
    return {"n": n, "min": round(v[0], nd), "p25": round(v[n // 4], nd),
            "median": round(v[n // 2], nd), "p75": round(v[(3 * n) // 4], nd),
            "max": round(v[-1], nd), "mean": round(sum(v) / n, nd)}


def _r(a: List[float], b: List[float]) -> Optional[float]:
    """Pearson correlation, or None when either side never varies."""
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(va * vb)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  MODEL-SIDE FINDINGS — true whether or not a single gauge exists
# ═════════════════════════════════════════════════════════════════════════════
def _how_water_leaves(model_columns: List[Dict]) -> Dict[str, Any]:
    """Where each column's water goes — over the surface, or through the soil.

    THE FINDING THAT WAS BURIED. Naches 1979, totals over the run:

        col_15   167.1 mm over   +   956.2 mm through   =  1123.3 mm
        col_08   172.0           +   873.7              =  1045.7
        ...
        col_10     0.0           +     0.0              =     0.0
        col_11     0.0           +     0.0              =     0.0

    Ten of seventeen columns drain nothing at all — six of them exactly 0.0 mm,
    four more under a thousandth of a millimetre in the year — and the totals
    span 0.0 to 1,123 mm, three orders of magnitude between columns in the same
    basin in the same year. Same finding as the frozen water tables in wtd.py,
    arriving through a different variable, and invisible under 10,353
    station-by-column pairs.

    ONE FUNCTION, NOT TWO. The spread between columns is a summary of this same
    table rather than a separate finding, so it comes back in the same record.
    It is what says whether the ensemble mean summarises agreement or averages
    two unrelated populations. On Naches it is the second: mean 254.7 mm/yr,
    median about 4.6, and no column near either.

    COUNTED, NEVER CUT. A column producing no runoff stays in the ensemble mean;
    dropping it would raise the mean and delete the reason it needed raising.

    QOVER and QDRAI are read SEPARATELY — the SPEC's own series is their sum,
    and the split is the whole point here.
    """
    over = C.model_series(model_columns, ["QOVER"])
    soil = C.model_series(model_columns, ["QDRAI"])
    per_column, totals, no_drainage, no_runoff = {}, [], [], []
    exactly_zero = 0        # counted from the RAW sum, not the rounded one
    for case in sorted(set(over) | set(soil)):
        s = sum(over[case]["values"]) if case in over else None
        d = sum(soil[case]["values"]) if case in soil else None
        if s is None or d is None:
            per_column[case] = {"note": "QOVER or QDRAI missing for this column"}
            continue
        total = s + d
        totals.append(total)
        per_column[case] = {
            "surface_mm": round(s, 3), "through_soil_mm": round(d, 3),
            "total_mm": round(total, 3),
            "frac_through_soil": round(d / total, 4) if total > 0 else None}
        if d < NEGLIGIBLE_MM:
            no_drainage.append(case)
            exactly_zero += (d == 0)
        if total < NEGLIGIBLE_MM:
            no_runoff.append(case)
    return {
        "units": "mm over the run",
        "negligible_below_mm": NEGLIGIBLE_MM,
        "per_column": per_column,
        "spread_between_columns": _quantiles(totals),
        "columns_with_no_drainage": {
            "n": len(no_drainage), "of": len(totals),
            "columns": no_drainage,
            "n_exactly_zero": exactly_zero,
            "note": ("QDRAI over the whole run is below the threshold in these "
                     "columns, so effectively nothing reaches groundwater in "
                     "them. Nothing is excluded on this basis — but a recharge "
                     "or drainage number averaged over the basin should be read "
                     "knowing how many of these went into it")},
        "columns_with_no_runoff_at_all": {
            "n": len(no_runoff), "of": len(totals), "columns": no_runoff,
            "note": ("neither over the surface nor through the soil. These "
                     "columns still count in the ensemble mean: dropping them "
                     "would raise it and delete the reason it needed raising")},
        "note": ("where the water leaves each column, over the whole run. True "
                 "with no gauge in the basin, which is why it is computed "
                 "before anything is compared"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2.  THE MODEL'S ONE SERIES — formed once, and said out loud
# ═════════════════════════════════════════════════════════════════════════════
def _ensemble_mean(model: Dict) -> Dict[str, Any]:
    """Mean runoff over the columns, day by day, plus the spread across them.

    NAMED FOR HOW IT IS FORMED, not for the role it plays. `basin_aggregate`
    would overclaim: that phrase implies an area-weighted integration over the
    watershed, and this is a plain mean over the columns the planner sampled.
    The planner's own word for that set is the ensemble, and this is its mean.

    ON THE DAYS EVERY COLUMN HAS, not on every date any column has. A mean over
    whichever columns happened to report is a different set of columns each day,
    and its ups and downs would be the membership changing rather than the water.

    THE MEAN, NOT THE MEDIAN, and that is a decision. Naches is two populations
    — ten columns near zero, five large — so the mean (254.7 mm/yr) and the
    median (about 4.6) answer different questions. Water AMOUNT is what a gauge
    asks, and an equal-weight mean is the closest thing to an area average
    available; the median would describe the typical column and understate the
    water by fifty times. The median is in `spread_between_columns`, where the
    skew is visible rather than hidden.

    Returns the daily mean, and nothing else. A per-day quartile band was tried
    and removed: with ten of seventeen columns producing nothing, p25 sits on
    the floor every day, and on the log axis panel 1 needs the fill became a
    grey block covering the panel. Panel 1 draws the columns themselves instead,
    the way swe.py and wtd.py do, so the two populations are visible rather than
    summarised into a shape that hides them.
    """
    cases = sorted(model)
    if not cases:
        return {"dates": [], "values": [], "n_columns": 0}
    shared = set(model[cases[0]]["dates"])
    for c in cases[1:]:
        shared &= set(model[c]["dates"])
    dates = sorted(shared)
    by_case = {c: dict(zip(model[c]["dates"], model[c]["values"])) for c in cases}
    values = [sum(by_case[c][d] for c in cases) / len(cases) for d in dates]
    return {"dates": dates, "values": values, "n_columns": len(cases),
            "n_dates_dropped": len(set().union(
                *(set(model[c]["dates"]) for c in cases))) - len(dates)}


def _ensemble_summary(ens: Dict) -> Dict[str, Any]:
    """What goes in the record — the series itself stays out of it.

    The daily arrays are 352 numbers three times over and every reader that
    needs them can rebuild them from the extracted columns, which is what plot()
    does. The record carries measurements, not a second copy of the model.
    """
    v = ens["values"]
    return {
        "n_columns": ens["n_columns"], "n_days": len(v),
        "period": C.span(ens["dates"]),
        "total_mm": round(sum(v), 3) if v else None,
        "mean_mm_day": round(sum(v) / len(v), 5) if v else None,
        "n_dates_dropped": ens.get("n_dates_dropped", 0),
        "formed_from": ("the plain mean over every column, on the days every "
                        "column has. A day missing from any column is dropped "
                        "rather than averaged over the rest, so the ups and "
                        "downs are the water changing and not the membership"),
        "not_area_weighted": (
            "NOT an area-weighted basin average. The columns were sampled to "
            "SPAN the basin's range — its elevation bands, its soils — not to "
            "tile it, and no column carries an area weight to do better with. "
            "This is the mean of the sampling design. It is the same KIND of "
            "quantity as the gauge's, which is what makes it admissible; it is "
            "not a basin water balance"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3.  AGAINST EACH GAUGE — amount, timing, and how often
# ═════════════════════════════════════════════════════════════════════════════
def _timing_offset(model_by_date: Dict[str, float],
                   obs_by_date: Dict[str, float],
                   max_offset: int = MAX_OFFSET_DAYS) -> Dict[str, Any]:
    """The offset that lines the two series up best, and how much it helps.

    A gauge sees water after the hillslope released it and the channel carried
    it; a column releases it where it stands. Comparing the two on the same
    calendar day therefore measures the routing delay as if it were an error.
    Sliding one past the other separates them: Naches outlet, ±15 d search, r
    rose from 0.55 SAME DAY to 0.64 with the MODEL FOURTEEN DAYS LATE — a
    melt-timing statement, and one a same-day correlation cannot make.

    SHIFTED IN DAYS, NOT IN ARRAY POSITIONS. The outlet gauge reported 272 of
    365 days, so the shared series has holes in it and stepping one place along
    an array is not stepping one day. Every offset is applied to the DATE and
    the pair is kept only when both sides have that date.

    A POSITIVE OFFSET MEANS THE MODEL IS LATE: obs on day d against model on
    day d + offset.

    REPORTS WHETHER THE BEST OFFSET SAT AT THE EDGE of the search. At ±15 d
    Naches was still improving when it ran out of window, so the true offset was
    unknown and "14" would have been a limit of the search reported as a result.
    That is why the window is 30.

    Both numbers travel: `r_same_day` is context for `r_at_best_offset`, never
    replaced by it. An offset that improves r describes when the water arrives;
    it is not a correction applied to make the model look better.
    """
    def at(offset: int) -> Tuple[Optional[float], int]:
        a, b = [], []
        for d, o in obs_by_date.items():
            k = (dt.date.fromisoformat(d) + dt.timedelta(days=offset)).isoformat()
            m = model_by_date.get(k)
            if m is not None:
                a.append(o)
                b.append(m)
        return _r(a, b), len(a)

    r0, n0 = at(0)
    floor = max(MIN_OFFSET_DAYS, 0.5 * n0)
    best_off, best_r = None, None
    for off in range(-max_offset, max_offset + 1):
        r, n = at(off)
        if r is None or n < floor:
            continue
        if best_r is None or r > best_r:
            best_off, best_r = off, r
    return {
        "best_offset_days": best_off,
        "r_at_best_offset": None if best_r is None else round(best_r, 4),
        "r_same_day": None if r0 is None else round(r0, 4),
        "n_days_same_day": n0,
        "searched_days": max_offset,
        "best_offset_at_edge": best_off is not None and abs(best_off) == max_offset,
        "sign": ("a positive offset means the model's water arrives that many "
                 "days AFTER the gauge saw it"),
        "note": ("the offset describes when the water arrives; it is not "
                 "applied to anything. Every other number in this entry is on "
                 "the same calendar day"),
    }


def _how_often_each_flow(values: List[float]) -> Dict[str, Any]:
    """The flow exceeded on 5%, 10%, 25% … 95% of days. No dates involved.

    Sort every day from wettest to driest and read the curve at those points.
    It asks whether flows of each SIZE happen as often in the model as at the
    gauge, which is the honest amount question for two quantities that are not
    co-located and cannot be expected to line up day by day. A model that is
    right about every flow but a fortnight late scores badly on bias and well
    here, and the gap between those two answers is worth having.
    """
    v = sorted((x for x in values if x is not None), reverse=True)
    if not v:
        return {}
    n = len(v)
    return {f"{p}%": round(v[min(n - 1, int(p / 100.0 * n))], 5)
            for p in PERCENT_OF_DAYS}


def _compare_to_gauge(ens: Dict, obs: Dict, meta: Dict,
                      basin_km2: Optional[float]) -> Dict[str, Any]:
    """One gauge, one entry. Everything measured, nothing scored."""
    entry: Dict[str, Any] = {
        "station_id": meta.get("station_id"), "name": meta.get("name"),
        "obs_period": C.span(obs["dates"]), "n_obs": len(obs["dates"]),
        "in_basin": meta.get("in_basin"), "source": meta.get("source"),
        "licence": meta.get("licence"),
    }
    # A units mismatch is REPORTED, never silently converted: assuming "mm"
    # meant "mm/day" is how a factor of 86400 gets in.
    if meta.get("units") and meta["units"] != SPEC.units:
        entry["units_mismatch"] = (
            f"station reports {meta['units']}, this comparison is in "
            f"{SPEC.units} — NOT converted; fix the table")

    # ── WHAT IT DRAINS ──────────────────────────────────────────────────────
    area = meta.get("drainage_area_km2")
    scale: Dict[str, Any] = {"column_area_m2": COLUMN_AREA_M2}
    if area:
        scale["drainage_area_km2"] = area
        # Orders of magnitude between the two footprints. NOTHING is scaled by
        # this — it is here so the mismatch is a number rather than an
        # adjective. A 2,437 km² catchment against a 1 m² column is 9.4 orders.
        scale["area_ratio_orders"] = round(
            math.log10(float(area) * 1e6 / COLUMN_AREA_M2), 1)
        if basin_km2:
            scale["basin_area_km2"] = basin_km2
            scale["area_fraction_of_basin"] = round(float(area) / basin_km2, 3)
            # AGAINST THE MODELLED BASIN, not only against the column. A
            # drainage area alone is uninterpretable — 10,285 km² is either this
            # basin or fifty times it — and a gauge draining MORE than was
            # simulated carries water from catchments the study never modelled,
            # so a bias against it is not a model error. In-basin tagging does
            # not catch this: the gauge sits inside the divide and still
            # integrates ground beyond it.
            scale["exceeds_modelled_domain"] = float(area) > float(basin_km2)
    entry["scale"] = scale

    # ── AMOUNT, ON THE SHARED DAYS ──────────────────────────────────────────
    dates, mm, oo, qq = C.pair({"dates": ens["dates"], "values": ens["values"]},
                               obs)                 # inner join, no interpolation
    if not dates:
        entry.update({"n_days": 0, "note": "no shared dates"})
        return entry
    entry.update({
        "n_days": len(dates), "overlap": C.span(dates),
        "metrics": C.metrics(mm, oo), "obs_quality": C.quality(qq),
        # A bias in mm/day hides how much water that is over a season.
        "totals_over_overlap_mm": {"gauge": round(sum(oo), 2),
                                   "ensemble_mean": round(sum(mm), 2)},
    })

    # ── TIMING, IN DAYS ─────────────────────────────────────────────────────
    entry["timing"] = _timing_offset(
        dict(zip(ens["dates"], ens["values"])),
        {d: v for d, v in zip(obs["dates"], obs["values"])})

    # ── HOW OFTEN EACH FLOW SIZE HAPPENS, over the same days ────────────────
    entry["how_often_each_flow"] = {
        "gauge": _how_often_each_flow(oo),
        "ensemble_mean": _how_often_each_flow(mm),
        "note": ("the flow exceeded on that percentage of the shared days. No "
                 "dates are involved, so a model that is right about every "
                 "flow but late still lines up here")}
    return entry


# ═════════════════════════════════════════════════════════════════════════════
# 4.  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def compare(model_columns: List[Dict], observations: Dict, station_meta: Dict,
            domain: Optional[Dict[str, Any]] = None, **kw) -> Dict[str, Any]:
    """The record: `gauges`, not `pairs`.

    NO `pairs` AND NO `assignment` KEY, deliberately — there is no pairing to
    report, and an empty `pairs` list would read as a pairing that found nothing
    rather than one that was never attempted.

    model_columns  the extracted rows, one dict per column
    observations   {(station_id, variable): {dates, values, quality}}
    station_meta   {(station_id, variable): {lat, lon, drainage_area_km2…}}
    domain         the modelled basin, for the area the gauges are read against
    """
    model = C.model_series(model_columns, SPEC.model_vars)
    rec: Dict[str, Any] = {
        "observable": SPEC.name, "units": SPEC.units,
        "model_comparand": SPEC.comparand, "obs_quantity": SPEC.obs_quantity,
        "colocated": SPEC.colocated,
        "n_columns_with_series": len(model),
        "compared_as": ("one model series — the ensemble mean — against each "
                        "gauge. No column is matched to a gauge: a gauge "
                        "measures an AREA, so there is no nearest column to "
                        "find, and picking one picks the answer"),
        "gauges": [],
    }
    if not model:
        rec["error"] = (
            f"no column has a daily series for {'+'.join(SPEC.model_vars)} — "
            f"the extraction either has not run or predates daily series. This "
            f"is an absence of model output, not of agreement.")
        return rec

    # ── BEFORE ANY EARLY RETURN ─────────────────────────────────────────────
    # These need no observation to be true, and a basin with no gauge in its
    # modelled year is exactly where they are the whole answer.
    rec["how_water_leaves"] = _how_water_leaves(model_columns)
    ens = _ensemble_mean(model)
    rec["ensemble_mean"] = _ensemble_summary(ens)
    rec["model_period"] = C.span(sorted({d for m in model.values()
                                         for d in m["dates"]}))

    # ── OUTSIDE THE DIVIDE, OUT OF THE COMPARISON ───────────────────────────
    # Applying reception's tag, never recomputing it. ONLY AN EXPLICIT False
    # EXCLUDES: None means nobody could check, and unchecked must not read as
    # failed. A gauge beyond the watershed integrates tributaries this study
    # never modelled, so its hydrograph is not this basin's — and "context" is
    # not a licence to put an unrelated catchment beside the columns.
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
            "note": ("these gauges carry no in_basin flag, so none was "
                     "excluded. Unchecked is not outside — but nor is it "
                     "inside. Re-run reception for this domain to tag them.")}

    if not stations:
        rec["error"] = (
            (f"all {len(outside)} gauge(s) fell outside the watershed and were "
             f"excluded, so nothing was compared: {', '.join(outside)}")
            if outside else
            "no gauge in this domain reported daily discharge for this period, "
            "so nothing was compared. Where the model's own water went is in "
            "`how_water_leaves` above, which needs no gauge. This is a fact "
            "about the basin, not a failed comparison.")
        rec["skipped"] = "no gauges"
        return rec

    if not ens["dates"]:
        rec["error"] = ("the columns share no date, so no ensemble mean could "
                        "be formed. This is a model-output problem, not a "
                        "disagreement.")
        return rec

    # ── EACH GAUGE AGAINST THE ONE ENSEMBLE MEAN ────────────────────────────
    basin_km2 = (domain or {}).get("area_km2")
    for sid in sorted(stations):
        meta = dict(station_meta.get((sid, SPEC.name)) or {})
        meta.setdefault("station_id", sid)
        rec["gauges"].append(_compare_to_gauge(ens, stations[sid], meta,
                                               basin_km2))

    rec["n_days_compared"] = sum(g.get("n_days", 0) for g in rec["gauges"])
    exceed = sorted(g["station_id"] for g in rec["gauges"]
                    if (g.get("scale") or {}).get("exceeds_modelled_domain"))
    if basin_km2:
        rec["gauges_exceeding_modelled_domain"] = {
            "n": len(exceed), "of": len(rec["gauges"]),
            "basin_area_km2": basin_km2, "station_ids": exceed,
            "note": ("these gauges drain more area than was simulated, so "
                     "their flow includes water never modelled")}
    return rec


# ═════════════════════════════════════════════════════════════════════════════
# 5.  THE FIGURE
# ═════════════════════════════════════════════════════════════════════════════
DECADES_SHOWN = 4       # a log axis below this much of its own peak is noise
SURFACE_COLOUR = "#6B4A86"
SOIL_COLOUR = "#2C6A5C"


def _log_limits(ax, values: List[float]) -> None:
    """Log axis, floored so an eight-decade tail cannot squash the data.

    Model runoff drops to 1e-8 mm/day in the columns that produce none, and an
    axis honouring that draws every real number in the top eighth of the panel.
    """
    v = [x for x in values if x is not None and x > 0]
    if len(v) < 2 or max(v) / min(v) <= 50:
        return
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(min(v), max(v) / 10 ** DECADES_SHOWN) * 0.8,
                top=max(v) * 1.6)


def plot(rec: Dict, model_columns: List[Dict], observations: Dict,
         out_path: str, reception_json: Optional[str] = None) -> Optional[str]:
    """Three panels, and the last two are drawn with no gauge at all.

      1  THE HYDROGRAPH. Every column in grey, the ensemble mean over them, each
         gauge in its own colour.

      2  WHERE EACH COLUMN'S WATER GOES. A stacked bar per column, surface
         runoff on through-the-soil drainage, ordered by total. On Naches it
         shows ten bars at zero.

      3  WHERE THOSE COLUMNS ARE. The same runoff as panel 2, in place: each
         column at its own coordinates, coloured by mean runoff and named, over
         the terrain and watershed outline sampling_design.png uses. Panel 2
         says the ensemble splits in two; this says whether the split is
         geographic. At Naches it is NOT: the six wet columns span 120.84W to
         121.48W and the four dead ones sit at 121.02W to 121.33W, inside that
         range, and the dead ones average 1,462 m against the wet ones' 1,266 m.
         Neither where a column sits nor how high it is separates them — which
         points at the column's own state rather than at its place in the basin,
         and is worth knowing before a basin mean is read.

    THE SORTED-FLOW PANEL WAS DROPPED (2026-08-12, user's call): it plotted the
    flow exceeded on each percent of days for gauge and model, and read as
    confusing next to a hydrograph. The NUMBERS stay in the record under
    `how_often_each_flow`, where they are seven values rather than a curve that
    has to be learned before it can be read.

    No 1:1 scatter either. For a comparison about shape and amount rather than
    day-matched magnitude, a scatter of two quantities that are not co-located
    invites exactly the reading the record refuses to make.
    """
    model = C.model_series(model_columns, SPEC.model_vars)
    leaves = (rec.get("how_water_leaves") or {}).get("per_column") or {}
    if not (model or leaves):
        return None
    ens = _ensemble_mean(model)
    stations = C.stations_for(observations, SPEC.name)
    gauges = [g for g in (rec.get("gauges") or []) if g.get("n_days")]

    fig, (ax1, ax2, ax3) = C.new_figure(ncols=3, width=18.0)
    d = lambda s: dt.date.fromisoformat(s)                      # noqa: E731

    # ── panel 1: the hydrograph ─────────────────────────────────────────────
    seen: List[float] = []
    for i, m in enumerate(model.values()):
        ax1.plot([d(x) for x in m["dates"]], m["values"], lw=0.7,
                 color="#9AA7B0", alpha=0.7, zorder=1,
                 label=f"{len(model)} columns" if i == 0 else None)
        seen += m["values"]
    if ens["dates"]:
        ax1.plot([d(x) for x in ens["dates"]], ens["values"], lw=1.8,
                 color="#2F3B42", zorder=3,
                 label=f"ensemble mean ({ens['n_columns']} columns)")
        seen += ens["values"]
    for i, g in enumerate(gauges):
        obs = stations.get(g["station_id"]) or {}
        if not obs.get("dates"):
            continue
        ax1.plot([d(x) for x in obs["dates"]], obs["values"], lw=1.4,
                 color=_colour(i), zorder=4, label=g["station_id"])
        seen += obs["values"]
    _log_limits(ax1, seen)
    ax1.set_ylabel(f"runoff  [{SPEC.units}]")
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_title(f"ensemble mean and {len(gauges)} gauge(s) — not co-located"
                  if gauges else
                  f"ensemble mean of {len(model)} columns — no gauge to compare")
    C.legend(ax1)

    # ── panel 2: where each column's water goes ─────────────────────────────
    rows = [(c, v.get("surface_mm") or 0.0, v.get("through_soil_mm") or 0.0)
            for c, v in leaves.items() if v.get("total_mm") is not None]
    rows.sort(key=lambda t: t[1] + t[2], reverse=True)
    if rows:
        x = range(len(rows))
        surf = [r[1] for r in rows]
        soil = [r[2] for r in rows]
        ax2.bar(x, surf, color=SURFACE_COLOUR, label="over the surface (QOVER)")
        ax2.bar(x, soil, bottom=surf, color=SOIL_COLOUR,
                label="through the soil (QDRAI)")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels([r[0] for r in rows], rotation=90, fontsize=8)
        C.legend(ax2)
    ax2.set_ylabel("water leaving over the run  [mm]")
    ax2.set_title("where each column's water goes")

    # ── panel 3: where those columns are ────────────────────────────────────
    # THE SAME NUMBER AS PANEL 2, IN PLACE. Panel 2 orders the columns by how
    # much water left them and loses where they were; a basin whose runoff is
    # all in one corner and one whose runoff is scattered draw the same bars.
    C.column_map(fig, ax3, model,
                 lambda c, m: (sum(m["values"]) / len(m["values"])
                               if m["values"] else None),
                 f"mean runoff  [{SPEC.units}]", reception_json, cmap="plasma")
    return C.save(fig, out_path)


def _colour(i: int) -> str:
    import matplotlib.pyplot as plt
    cyc = plt.rcParams["axes.prop_cycle"].by_key().get("color") or ["#1f77b4"]
    return cyc[i % len(cyc)]
