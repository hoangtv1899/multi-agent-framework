#!/usr/bin/env python3
"""SWE: model H2OSNO against a snow pillow.

MATCHED FIRST, THEN COMPARED (2026-08-12). Every station used to be compared
against every column and the one-to-one match computed afterwards, which on a
17-column Naches run produced 8,772 paired days to keep 516 of them — 94% of the
record was a comparison already ruled out. The cost was never the arithmetic;
it was that `comparison.json` then carried 34 complete metric sets beside one
"assignment", and the flattering one is always in there somewhere. A comparison
you have decided is invalid should not be computed, because computing it is what
makes it available.

So: drop the stations outside the divide, match one column to one station, and
compare only those. An unmatched station and an unmatched column each keep the
number that disqualified them — that is the part worth reading when a basin
comes back thin.

WHAT THIS ADDS over generic metrics: a thin pack lasting months and a deep one
melting fast produce the same mean. Peak agreeing while duration disagrees is a
different finding from both disagreeing, and bias/RMSE cannot express either. So
the season's TIMING is reported for both sides, as numbers, for the caller to
difference — never differenced here.

Pairing is on ELEVATION, not distance. See _common.pair_stations.

This module was the first to stop calling _common.standard_compare. All four
have now been through it and that function is deleted, along with the
assigned_pairs helper that read its output — 157 lines implementing the
compare-everything-then-choose approach the package rejected. What is genuinely
shared stayed in _common: loading, pairing, metrics, quality, the plot helpers.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.analysis import compare_common as C

SPEC = C.Spec(
    name="swe", model_vars=["H2OSNO"], units="mm",
    comparand="H2OSNO, snow water equivalent on the column",
    obs_quantity="snow water equivalent at a snow pillow",
    colocated=True, pair_on="elevation",
    # Set 2026-08-11. Narrower than the limits the four observables used to
    # share: a snow pillow measures the snow ON IT, and a column 10 km away
    # across a ridge is not that snowpack. Elevation FILTERS, distance
    # DECIDES — see pair_stations.
    max_delta_m=150.0, max_km=5.0,
    headlines=("shared_window", "onset_censored_columns"))

THRESHOLD_MM = 25.0     # ~1 inch SWE: a pack, not a dusting

# HOW LONG THE GROUND MUST BE BARE TO END A SNOW SEASON.
# A mid-winter thaw can drop a pack under the threshold for a week, and calling
# that the end of the season would put melt-out in January. A month with no
# pack is not a thaw.
SEASON_GAP_DAYS = 30


# ─────────────────────────────────────────────────────────────────────────────
# THE SEASON
# ─────────────────────────────────────────────────────────────────────────────
def _seasons(values: Sequence[Optional[float]], threshold: float,
             gap_days: int) -> List[List[int]]:
    """Index runs of above-threshold days, split where the pack is long gone.

    A CALENDAR YEAR CAN HOLD TWO PARTIAL SEASONS and this is what notices.
    Reception fetches SWE on the calendar year since 2026-08-12, so a series
    runs melt-limb, bare summer, then the NEXT season accumulating in November
    and December. The old code took the last above-threshold day of the whole
    series as melt-out, which in a full-year record lands just before New Year
    — and the model series has the same shape, so both sides would have agreed
    on a midwinter melt-out. The water year used to prevent this by construction.
    """
    above = [i for i, v in enumerate(values)
             if v is not None and not (isinstance(v, float) and math.isnan(v))
             and v > threshold]
    if not above:
        return []
    runs, cur = [], [above[0]]
    for a, b in zip(above, above[1:]):
        if b - a > gap_days:
            runs.append(cur)
            cur = [b]
        else:
            cur.append(b)
    runs.append(cur)
    return runs


def snow_season_timing(dates: List[str], values: List[float],
                       threshold: float = THRESHOLD_MM) -> Dict[str, Any]:
    """WHEN the snowpack happened, not only how much of it there was.

    Was called `phenology`, which is a botanist's word for the timing of
    seasonal events and needed translating every time it was read.

    THE TIMING BELONGS TO ONE SEASON — the one holding the peak. Depth and mean
    are still taken over the whole window, because both sides share that window
    and "how much snow was on the ground over this year" is a fair question.
    `n_seasons` says when the window held more than one, so a reader is never
    guessing which one the dates describe.
    """
    keep = [(d, v) for d, v in zip(dates or [], values or [])
            if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not keep:
        return {"available": False, "reason": "no SWE series"}
    ds = [d for d, _ in keep]
    vs = [v for _, v in keep]

    peak = max(vs)
    peak_i = vs.index(peak)
    runs = _seasons(vs, threshold, SEASON_GAP_DAYS)
    # The season containing the peak. A year whose peak never crosses the
    # threshold has no season at all, which is a real answer for a bare year.
    season = next((r for r in runs if r[0] <= peak_i <= r[-1]), [])

    out: Dict[str, Any] = {
        "available": True,
        "peak_swe_mm": round(peak, 1),
        "mean_swe_mm": round(sum(vs) / len(vs), 1),
        "peak_date": ds[peak_i],
        "peak_dowy": day_of_water_year(ds[peak_i]),
        "threshold_mm": threshold,
        "has_snowpack": bool(runs),
        "n_seasons": len(runs),
        "days_above_threshold": len(season),
    }
    if len(runs) > 1:
        out["n_seasons_note"] = (
            f"this window holds {len(runs)} separate snow seasons; the dates "
            f"below describe the one containing the peak")
    if season:
        # ONSET IS CENSORED WHEN THE RECORD OPENS WITH SNOW ALREADY DOWN.
        # A run starting 1 January has a pack on day one, and calling that day
        # the first snow reports WHEN THE RUN BEGAN, not when snow arrived — a
        # date that looks like a measurement and is an artefact of the start
        # date. Since reception moved SWE to the calendar year this is the
        # normal case in a snow basin rather than the exception, and it is the
        # honest answer: ELM was not running the previous November, so this run
        # cannot observe its own snow onset.
        out["first_snow_censored"] = season[0] == 0
        out["first_snow_date"] = None if season[0] == 0 else ds[season[0]]
        # Melt-out is the LAST day above threshold IN THIS SEASON, not the first
        # day below: a mid-winter thaw dipping under the line for a week would
        # otherwise end the season in January.
        out["melt_out_date"] = ds[season[-1]]
        out["melt_out_dowy"] = day_of_water_year(ds[season[-1]])
    return out


def day_of_water_year(d: Optional[str]) -> Optional[int]:
    """Oct 1 = 1. Snow years do not respect January.

    A calendar day-of-year puts a 15 December peak at 349 and a 5 January peak
    at 5 — adjacent events 344 apart, which makes the peak-date panel unreadable
    and any mean of it meaningless. Unaffected by which window was FETCHED:
    peak timing on a snow-year axis is right either way.
    """
    if not d:
        return None
    import datetime as dt
    try:
        day = dt.date.fromisoformat(d[:10])
    except ValueError:
        return None
    start = dt.date(day.year - (1 if day.month < 10 else 0), 10, 1)
    return (day - start).days + 1


def shared_window(model: Dict, stations: Dict) -> Optional[List[str]]:
    """The window both sides actually cover.

    Was `water_year_window`, named for a mismatch that no longer arrives from
    that direction — reception fetches SWE on the calendar year now. It is still
    needed: a run can start mid-year, as Gunnison's did on 2019-01-15, and
    metrics over the union would count months where one side has nothing.
    """
    m_dates = sorted({d for m in model.values() for d in m["dates"]})
    o_dates = sorted({d for s in stations.values() for d in s["dates"]})
    if not (m_dates and o_dates):
        return None
    lo, hi = max(m_dates[0], o_dates[0]), min(m_dates[-1], o_dates[-1])
    return [lo, hi] if lo <= hi else None


def _clip(dates, values, window) -> Tuple[List, List]:
    """The part of a series inside `window`, or all of it if there is no window.

    The season timing is computed from a WHOLE series rather than from the
    paired days, so it is the one block that has to be clipped by hand. It was
    not, until 2026-08-10: shared_window was computed, stored, and then ignored,
    so at Gunnison the observed record ran 2018-10-01 to 2020-09-30 while the
    model ran 2019-01-15 to 2020-09-30, and the observed mean and days-above
    were counted over 106 extra days of snow the model was never asked to
    produce. Peak survives that; a mean over two different denominators is not
    a comparison.
    """
    if not window:
        return list(dates or []), list(values or [])
    lo, hi = window[0], window[1]
    kept = [(d, v) for d, v in zip(dates or [], values or []) if lo <= d <= hi]
    return [d for d, _ in kept], [v for _, v in kept]


# ─────────────────────────────────────────────────────────────────────────────
# THE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def compare(model_columns: List[Dict], observations: Dict, station_meta: Dict,
            **kw) -> Dict[str, Any]:
    """One column to one station, then the numbers. Measurements, not verdicts.

    model_columns  the extracted rows, one dict per column, each carrying a
                   daily series per variable
    observations   {(station_id, variable): {dates, values, quality}}
    station_meta   {(station_id, variable): {lat, lon, elevation_m, in_basin,
                   source, licence, units}}
    """
    model = C.model_series(model_columns, SPEC.model_vars)
    stations = C.stations_for(observations, SPEC.name)

    rec: Dict[str, Any] = {
        "observable": SPEC.name, "units": SPEC.units,
        "model_comparand": SPEC.comparand, "obs_quantity": SPEC.obs_quantity,
        "colocated": SPEC.colocated,
        "n_columns_with_series": len(model),
        "threshold_mm": THRESHOLD_MM,
        "season_gap_days": SEASON_GAP_DAYS,
        "pairs": [],
    }

    # ── OUTSIDE THE DIVIDE, OUT OF THE COMPARISON ───────────────────────────
    # Applying reception's tag, never recomputing it. A station beyond the
    # watershed measures ground the study does not model, and the matcher
    # cannot know that, so it will spend a column on one: measured 2026-08-10
    # at Gunnison, Red Mountain Pass, outside, claimed col_15 and left Wager
    # Gulch, inside, with no column at all.
    #
    # ONLY AN EXPLICIT False EXCLUDES. None means nobody could check — no
    # polygon, or a station with no coordinates — and unchecked must never read
    # as failed. Refusing untested stations instead cost naches_1979 and
    # brandywine_2010 all four of their pinned columns on 2026-08-08, when one
    # elevation request timed out and took the polygon with it.
    tags = [(station_meta.get((sid, SPEC.name)) or {}).get("in_basin")
            for sid in stations]
    outside = sorted(sid for sid in stations
                     if (station_meta.get((sid, SPEC.name)) or {}).get("in_basin")
                     is False)
    for sid in outside:
        stations.pop(sid, None)
    rec["stations_excluded_outside_basin"] = outside
    rec["n_stations"] = len(stations)
    unchecked = sum(1 for t in tags if t is None)
    if unchecked:
        rec["in_basin_unchecked"] = {
            "n_stations": unchecked,
            "note": ("these stations carry no in_basin flag, so none of them "
                     "was excluded. Unchecked is not outside — but nor is it "
                     "inside. Re-run reception for this domain to tag them.")}

    if not model:
        rec["error"] = (
            "no column has a daily series for H2OSNO — the extraction either "
            "has not run or predates daily series. This is an absence of model "
            "output, not of agreement.")
        return rec

    rec["model_period"] = C.span(sorted({d for m in model.values()
                                         for d in m["dates"]}))
    win = shared_window(model, stations) if stations else None
    rec["shared_window"] = win

    # BEFORE THE EARLY RETURN. Whether a run can observe its own snow onset is
    # decided by its start date, not by whether a SNOTEL site happened to be in
    # the basin — so a basin with no snow stations at all is still told that its
    # onset dates are artefacts.
    censored = [case for case, m in model.items()
                if snow_season_timing(*_clip(m["dates"], m["values"], win))
                .get("first_snow_censored")]
    rec["onset_censored_columns"] = {
        "n": len(censored), "of": len(model), "columns": sorted(censored),
        "note": ("these columns already had snow above threshold on the first "
                 "day of the record, so accumulation onset is not observable "
                 "in this run and no onset date is reported for them")}

    if not stations:
        # "None were inside the basin" and "none were fetched" are different
        # findings, and the caller acts on them differently — one is a siting
        # problem, the other a coverage problem.
        rec["error"] = (
            (f"all {len(outside)} swe station(s) fell outside the watershed "
             f"and were excluded, so nothing was compared: "
             f"{', '.join(outside)}")
            if outside else
            "no observations of 'swe' in the table — nothing was compared. "
            "This is not a disagreement.")
        return rec

    # ── MATCH FIRST ─────────────────────────────────────────────────────────
    # One column, one station. Elevation FILTERS, distance DECIDES, and a
    # station leaves the pool once a column takes it — so col_01 has first
    # refusal. The limits are the ones declared on this Spec and nothing else:
    # an observable never borrows a number from outside its own definition.
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

    rec["matched_on"] = SPEC.pair_on
    rec["max_separation_km"] = SPEC.max_km
    rec["max_delta_elevation_m"] = SPEC.max_delta_m
    rec["n_pinned"] = sum(1 for a in matched if a.get("matched_on") == "pinned")
    rec["matched_in_order"] = ("one pass over the columns in name order; a "
                               "station leaves the pool when a column takes "
                               "it, so col_01 has first refusal")
    rec["unpaired_stations"] = unpaired
    rec["unmatched_columns"] = unmatched

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
        # A units mismatch is REPORTED, never silently converted: assuming "mm"
        # meant "mm/day" is how a factor of 86400 gets in.
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
        # Both sides over the SAME window, so peak, mean and days-above are one
        # measurement taken twice rather than two different ones.
        entry["season_obs"] = snow_season_timing(
            *_clip(obs["dates"], obs["values"], win))
        entry["season_model"] = snow_season_timing(
            *_clip(m["dates"], m["values"], win))
        rec["pairs"].append(entry)

    rec["n_pairs_total"] = sum(e.get("n_days", 0) for e in rec["pairs"])
    if not rec["pairs"]:
        rec["error"] = (
            f"no station could be matched to a column within "
            f"{SPEC.max_delta_m:.0f} m of elevation and {SPEC.max_km:.0f} km. "
            f"{len(unpaired)} station(s) and {len(unmatched)} column(s) went "
            f"unmatched; each carries the number that disqualified it. This is "
            f"a siting result, not a disagreement.")
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# THE FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def _colour(i: int) -> str:
    import matplotlib.pyplot as plt
    cyc = plt.rcParams["axes.prop_cycle"].by_key().get("color") or ["#1f77b4"]
    return cyc[i % len(cyc)]


def plot(rec: Dict, model_columns: List[Dict], observations: Dict,
         out_path: str, **kw) -> Optional[str]:
    """Three panels: the basin's snow, the season's timing, and where it lay.

    EVERY COLUMN IS DRAWN, not only the matched ones. At Naches 2 of 17 columns
    got a station — the other 15 are the study, and the observations are a thin
    check on a small corner of it. Dropping them to tidy the figure would throw
    away the result to make room for the test of it.

    A matched column is drawn in ITS STATION'S COLOUR on top of that grey
    (2026-08-12). Before, the observations sat against an undifferentiated cloud
    and the one thing the panel exists to show — which line is being checked
    against which dots — was the one thing invisible.

    THE 1:1 SCATTER OF MATCHED DAYS IS GONE (2026-08-12, user's call), and the
    map took its place. The scatter spent the middle of the figure on the two
    columns that had a station; the map covers all seventeen, and a basin with
    no SNOTEL site in it still gets a snow figure — which under the old shape
    returned None and drew nothing at all.
    """
    model = C.model_series(model_columns, SPEC.model_vars)
    pairs = [e for e in (rec.get("pairs") or []) if e.get("n_days")]
    if not (model or pairs):
        return None
    import datetime as dt
    fig, (ax1, ax2, ax3) = C.new_figure(ncols=3, width=18.0)
    d = lambda s: dt.date.fromisoformat(s)                      # noqa: E731

    # ── panel 1: every column, matched ones picked out ──────────────────────
    matched_cases = {e["case_name"] for e in pairs}
    for i, (case, m) in enumerate(model.items()):
        if case in matched_cases:
            continue
        ax1.plot([d(x) for x in m["dates"]], m["values"], lw=0.7,
                 color="#9AA7B0", alpha=0.6, zorder=1,
                 label=f"{len(model) - len(matched_cases)} unmatched columns"
                       if i == 0 else None)
    for i, e in enumerate(pairs):
        col = _colour(i)
        m = model.get(e["case_name"]) or {}
        if m:
            ax1.plot([d(x) for x in m["dates"]], m["values"], lw=1.6,
                     color=col, zorder=2, label=f"{e['case_name']} (model)")
        obs = C.stations_for(observations, SPEC.name).get(e["station_id"]) or {}
        meas, fill = C.split_quality(obs.get("quality") or [])
        od = obs.get("dates") or []
        ov = obs.get("values") or []
        if meas:
            ax1.scatter([d(od[k]) for k in meas], [ov[k] for k in meas],
                        s=13, color=col, zorder=3, label=f"{e['station_id']}")
        if fill:
            ax1.scatter([d(od[k]) for k in fill], [ov[k] for k in fill], s=13,
                        facecolors="none", edgecolors=col, linewidths=0.7,
                        zorder=3, label=f"{e['station_id']} (gap-filled)")
    ax1.set_ylabel(f"SWE  [{SPEC.units}]")
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_title(f"{len(model)} columns, {len(pairs)} matched to a station")
    C.legend(ax1)

    # ── panel 2: the season's timing, both events ───────────────────────────
    # PEAK AND MELT-OUT TOGETHER. Peak alone answers "does it melt at the right
    # time" only halfway: a pack can peak on the right day and linger a month.
    drew = False
    for i, e in enumerate(pairs):
        po = e.get("season_obs") or {}
        pm = e.get("season_model") or {}
        col = _colour(i)
        for key, marker, tag in (("peak_dowy", "o", "peak"),
                                 ("melt_out_dowy", "s", "melt-out")):
            if po.get(key) and pm.get(key):
                ax2.scatter([po[key]], [pm[key]], s=70, marker=marker,
                            color=col, zorder=3,
                            label=f"{e['station_id']} · {tag}")
                drew = True
    if drew:
        vals = [v for e in pairs for s in ("season_obs", "season_model")
                for k in ("peak_dowy", "melt_out_dowy")
                if (v := (e.get(s) or {}).get(k))]
        C.one_to_one(ax2, vals, vals)
        C.legend(ax2)
    else:
        ax2.text(0.5, 0.5, "no station was matched to a column", ha="center",
                 va="center", transform=ax2.transAxes, color="#77837F",
                 fontsize=9)
    ax2.set_xlabel("observed  (day of water year)")
    ax2.set_ylabel("model  (day of water year)")
    ax2.set_title("season timing")

    # ── panel 3: where the snow lay ─────────────────────────────────────────
    # MEAN SWE OVER THE RUN, per column, in place. A basin's snow is an
    # elevation story before it is anything else, and panel 1 draws seventeen
    # lines with no way to tell which is high and which is low.
    C.column_map(fig, ax3, model,
                 lambda c, m: (sum(m["values"]) / len(m["values"])
                               if m["values"] else None),
                 f"mean SWE  [{SPEC.units}]", kw.get("reception_json"),
                 cmap="viridis")
    return C.save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAP POINTS — this module's own values, in place. LAYOUT BELONGS TO maps.py.
# ─────────────────────────────────────────────────────────────────────────────
def map_points(rec, model_columns, observations, station_meta, **kw):
    """One row of the spatial comparison: SNOTEL beside ELM, both mean SWE.

    THE VALUES ARE COMPUTED HERE, NOT IN maps.py, which is the rule the legacy
    figure kept and the reason its two maps could never disagree about a
    station: the module that knows what H2OSNO means is the module that
    averages it. maps.py decides where the panels go and nothing else.
    """
    model = C.model_series(model_columns, SPEC.model_vars)
    stations = C.stations_for(observations, SPEC.name)
    obs = []
    for sid, s in stations.items():
        m = station_meta.get((sid, SPEC.name)) or {}
        if m.get("lat") is None or m.get("in_basin") is False or not s["values"]:
            continue
        obs.append((m["lon"], m["lat"], sum(s["values"]) / len(s["values"]), sid))
    mod = [(v["lon"], v["lat"], sum(v["values"]) / len(v["values"]), c)
           for c, v in model.items()
           if v.get("lat") is not None and v["values"]]
    return {"label": f"mean SWE  [{SPEC.units}]", "log": False,
            "panels": [{"title": "SNOTEL", "points": obs},
                       {"title": "ELM", "points": mod}]}
