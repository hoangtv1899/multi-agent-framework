#!/usr/bin/env python3
"""
Model vs observations, for ELM columns. MEASUREMENTS ONLY.

    swe          H2OSNO                  vs SNOTEL / snow pillow      mm
    wtd          ZWT                     vs USGS wells                m below surface
    streamflow   QOVER + QDRAI           vs USGS gauge                mm/day
    et           QSOIL + QVEGE + QVEGT   vs AmeriFlux tower           mm/day

WHAT THIS RETURNS, AND WHAT IT DOES NOT. It returns numbers: paired series,
per-station metrics, and diagnostics ABOUT the pairing — how many pairs, over
what window, how much of the observation was measured rather than gap-filled.
It returns no verdicts. "bias = -41 mm" is a measurement; "the model
underestimates snowpack" is an interpretation, and interpretation belongs to
whoever reads this, not to the thing that computed it.

EVERY COMPARISON HERE IS CONTEXT, NONE IS A SKILL CLAIM. That is a decision
(2026-08-10), not a hedge, and it is what makes the streamflow comparison
admissible at all: a 1-D column produces point runoff and a gauge measures
routed discharge over an upstream area, so the two are not co-located and no
metric between them scores the model. The hydrograph SHAPE is still worth
seeing. Each record carries `model_comparand` and `obs_quantity` as plain
strings so a reader can see for themselves what was put beside what.

WHY THE RAW SERIES AND NOT THE DERIVED METRICS. This reads only the `daily`
blocks the extraction wrote. It never touches the summary fields —
recharge_fraction and friends — because those carry semantics that live on the
framework side (recharge_fraction is the recharge-vs-runoff SPLIT, not a
fraction of precipitation; read the obvious way it made columns draining
~0 mm/yr report 1.00). Reading a NetCDF variable onto a time axis involves no
judgement; deciding what the ratio of two of them MEANS does. Only the first is
here.

THE OBSERVATION CONTRACT — the caller writes these, this module reads them:

    observations.csv        long format, one row per station per timestamp
        station_id,variable,time,value,quality
        SNOTEL:663,swe,2019-01-01,241.3,measured
        US-NR1,et,2019-01-01,0.42,filled

    observations_meta.json  one entry per (station_id, variable)
        [{"station_id": "US-NR1", "variable": "et", "units": "mm/day",
          "lat": 40.03, "lon": -105.55, "elevation_m": 3050,
          "source": "AmeriFlux BASE", "in_basin": false,
          "licence": "CCBY4.0"}]

CSV rather than parquet on purpose: neither pyarrow nor fastparquet is
installed on Compy, and a format the host cannot read is not a format. A file
rather than an argument for the same reason case_inputs.json is a file — this
server's own rule is that a few short strings travel inline and results never
do, and one tower is 490k rows.

`quality` IS LOAD-BEARING, not decoration. Measured 2026-08-10 at US-NR1: the
gap-filled annual ET is 464 mm in 2016 against 89 mm from the measured
half-hours alone, because only 36% of that year was actually observed. A
comparison handed values with no provenance will compare a gap-filling model's
output to a land model's output and call the agreement a result. So every
record reports the quality breakdown of the pairs it used, and does not decide
what to do about it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# What each observable is compared as. `model` names the ELM history variables
# that are summed; `units` is what BOTH sides are put in before pairing.
OBSERVABLES: Dict[str, Dict[str, Any]] = {
    "swe": {
        "model": ["H2OSNO"], "units": "mm",
        "model_comparand": "H2OSNO, snow water equivalent on the column",
        "obs_quantity": "snow water equivalent at a snow pillow",
        "colocated": True,
    },
    "wtd": {
        "model": ["ZWT"], "units": "m",
        "model_comparand": "ZWT, diagnosed water-table depth (positive down)",
        "obs_quantity": "depth to water in a well (positive down)",
        "colocated": True,
    },
    "streamflow": {
        "model": ["QOVER", "QDRAI"], "units": "mm/day",
        "model_comparand": "QOVER + QDRAI, point surface runoff plus drainage",
        "obs_quantity": "gauge discharge per unit contributing area",
        # NOT co-located, and the record says so as a fact rather than a
        # judgement: a column is 1 m^2 and unrouted, the gauge integrates a
        # basin. The reader decides what that is worth.
        "colocated": False,
    },
    "et": {
        "model": ["QSOIL", "QVEGE", "QVEGT"], "units": "mm/day",
        "model_comparand": ("QSOIL + QVEGE + QVEGT — ground evaporation, "
                            "canopy evaporation, transpiration. This ELM "
                            "build registers the components, not a single "
                            "QFLX_EVAP_TOT"),
        "obs_quantity": "eddy-covariance latent heat flux, as water depth",
        "colocated": True,
    },
}

# ELM's hydrologically active soil column. Deeper than this, ZWT is diagnosed
# from an unconfined aquifer rather than simulated — a fact about the model
# worth counting, which is why the wtd record reports how many pairs sit below
# it instead of quietly averaging them in.
ACTIVE_SOIL_DEPTH_M = 3.8

# WHICH COLUMN A STATION IS COMPARED TO. Elevation for SWE, because elevation
# is what governs snowpack; geography for the rest.
#
# THIS IS PORTED, NOT INVENTED, and the first version of this module got it
# wrong by inventing it. step1_compare_swe.pair_by_elevation records that
# matching SWE stations on horizontal distance was tried and rejected: it
# "produced offsets up to 846 m, with five stations collapsed onto two
# columns", where matching on elevation left a worst offset of 162 m and every
# station paired. Reaching for nearest-by-lat/lon here reintroduced a rule
# somebody had already measured and thrown out.
PAIR_ON = {"swe": "elevation", "wtd": "distance",
           "streamflow": "distance", "et": "distance"}
MAX_PAIR_DELTA_M = 200.0        # elevation, see pair_stations
SWE_THRESHOLD_MM = 25.0         # ~1 inch SWE; a pack, not a dusting


# ─────────────────────────────────────────────────────────────────────────────
# READING WHAT THE CALLER WROTE
# ─────────────────────────────────────────────────────────────────────────────
def load_observations(obs_csv: str, meta_json: str = "") -> Tuple[Dict, Dict]:
    """The observation table and its metadata, keyed for pairing.

    Returns ({(station_id, variable): {"dates": [...], "values": [...],
    "quality": [...]}}, {(station_id, variable): meta}).

    Rows with an unparseable value are DROPPED and counted, never coerced to
    zero: a missing snow measurement is not zero snow.
    """
    import csv as _csv
    series: Dict[Tuple[str, str], Dict[str, List]] = {}
    dropped = 0
    with open(obs_csv, newline="") as fh:
        for row in _csv.DictReader(fh):
            sid = (row.get("station_id") or "").strip()
            var = (row.get("variable") or "").strip().lower()
            t = (row.get("time") or "").strip()[:10]
            if not (sid and var and t):
                dropped += 1
                continue
            try:
                v = float(row.get("value"))
            except (TypeError, ValueError):
                dropped += 1
                continue
            if math.isnan(v):
                dropped += 1
                continue
            s = series.setdefault((sid, var),
                                  {"dates": [], "values": [], "quality": []})
            s["dates"].append(t)
            s["values"].append(v)
            s["quality"].append((row.get("quality") or "unknown").strip().lower())

    meta: Dict[Tuple[str, str], Dict] = {}
    if meta_json and Path(meta_json).is_file():
        for m in json.loads(Path(meta_json).read_text()) or []:
            key = (str(m.get("station_id")), str(m.get("variable", "")).lower())
            meta[key] = m
    for s in series.values():                       # chronological, for spans
        order = sorted(range(len(s["dates"])), key=lambda i: s["dates"][i])
        for k in ("dates", "values", "quality"):
            s[k] = [s[k][i] for i in order]
    series["_dropped"] = dropped                    # reported, not hidden
    return series, meta


def _daily(row: Dict, var: str) -> Dict[str, List]:
    """One variable's daily series off an extracted row, or empty."""
    return ((row.get("variables") or {}).get(var) or {}).get("daily") or {}


def model_series(rows: List[Dict], observable: str) -> Dict[str, Dict]:
    """{case_name: {"dates": [...], "values": [...]}} for one observable.

    Sums the named variables date-wise (ET is three components, streamflow is
    two). A date is kept only when EVERY component has a value there — a
    partial sum is a smaller number, not a missing one, and it would read as
    the model drying out rather than as an absent field.
    """
    spec = OBSERVABLES[observable]
    out: Dict[str, Dict] = {}
    for row in rows or []:
        name = row.get("case_name")
        if not name:
            continue
        parts = []
        for var in spec["model"]:
            d = _daily(row, var)
            if not (d.get("dates") and d.get("values")):
                parts = []
                break
            parts.append(dict(zip(d["dates"], d["values"])))
        if not parts:
            continue
        common = set(parts[0])
        for p in parts[1:]:
            common &= set(p)
        dates = sorted(d for d in common
                       if all(p.get(d) is not None for p in parts))
        if not dates:
            continue
        out[name] = {"dates": dates,
                     "values": [sum(p[d] for p in parts) for d in dates],
                     "units": (_daily(row, spec["model"][0]) or {}).get("units"),
                     "lat": row.get("lat"), "lon": row.get("lon"),
                     "elevation_m": row.get("elevation_m")}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PAIRING AND METRICS
# ─────────────────────────────────────────────────────────────────────────────
def _pair(m: Dict, o: Dict) -> Tuple[List, List, List, List]:
    """Model and observation on their SHARED dates. Returns (dates, m, o, q).

    An inner join on the date, not interpolation. Filling a gap in either
    series to make the arrays line up invents a measurement, and the count of
    pairs — which is reported — would then describe the invention.
    """
    mv = dict(zip(m["dates"], m["values"]))
    dates, mm, oo, qq = [], [], [], []
    for i, d in enumerate(o["dates"]):
        if d in mv:
            dates.append(d)
            mm.append(mv[d])
            oo.append(o["values"][i])
            qq.append(o["quality"][i] if i < len(o.get("quality", [])) else "unknown")
    return dates, mm, oo, qq


def _metrics(m: List[float], o: List[float]) -> Dict[str, Any]:
    """Standard goodness-of-fit numbers. No thresholds, no verdicts.

    NSE and KGE are undefined when the observation has zero variance (a well
    that never moved, a season with no snow), and returning 0.0 there would be
    a made-up score. They come back None with n reported, so a reader can see
    the difference between "no skill" and "not computable".
    """
    n = len(m)
    if n == 0:
        return {"n": 0}
    mean_m = sum(m) / n
    mean_o = sum(o) / n
    bias = mean_m - mean_o
    mae = sum(abs(a - b) for a, b in zip(m, o)) / n
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(m, o)) / n)
    var_o = sum((b - mean_o) ** 2 for b in o)
    var_m = sum((a - mean_m) ** 2 for a in m)
    cov = sum((a - mean_m) * (b - mean_o) for a, b in zip(m, o))
    r = (cov / math.sqrt(var_m * var_o)) if var_m > 0 and var_o > 0 else None
    nse = (1.0 - sum((a - b) ** 2 for a, b in zip(m, o)) / var_o) \
        if var_o > 0 else None
    kge = None
    if r is not None and mean_o != 0 and var_o > 0:
        alpha = math.sqrt(var_m / n) / math.sqrt(var_o / n)
        beta = mean_m / mean_o
        kge = 1.0 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {"n": n,
            "model_mean": round(mean_m, 4), "obs_mean": round(mean_o, 4),
            "bias": round(bias, 4), "mae": round(mae, 4), "rmse": round(rmse, 4),
            "pearson_r": None if r is None else round(r, 4),
            "nse": None if nse is None else round(nse, 4),
            "kge": None if kge is None else round(kge, 4)}


def _quality_breakdown(q: List[str]) -> Dict[str, Any]:
    """How much of the paired observation was actually measured.

    The reason this exists is on the record: at US-NR1 in 2016, gap-filled ET
    is 464 mm and the measured half-hours alone give 89 mm, because 36% of the
    year was observed. Both numbers are true and they are not the same claim.
    """
    counts: Dict[str, int] = {}
    for x in q:
        counts[x] = counts.get(x, 0) + 1
    n = len(q) or 1
    measured = counts.get("measured", 0) + counts.get("approved", 0)
    return {"counts": counts,
            "frac_measured": round(measured / n, 4),
            "frac_gap_filled": round(counts.get("filled", 0) / n, 4)}


def _span(dates: List[str]) -> Optional[List[str]]:
    return [dates[0], dates[-1]] if dates else None


def swe_phenology(dates: List[str], values: List[float],
                  threshold: float = SWE_THRESHOLD_MM) -> Dict[str, Any]:
    """WHEN the snowpack happened, not just how much of it there was.

    Ported from step1_compare_swe.swe_metrics, and the reason it is here rather
    than left to bias/RMSE is in that module's own comment: a thin pack lasting
    months and a deep one melting fast can produce the same mean. Peak agreeing
    while duration disagrees is a different finding from both disagreeing, and
    a generic goodness-of-fit number cannot express either.

    Measurements only, on both sides, so the caller can difference them.
    """
    pairs = [(d, v) for d, v in zip(dates or [], values or [])
             if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not pairs:
        return {"available": False, "reason": "no SWE series"}
    ds = [d for d, _ in pairs]
    vs = [v for _, v in pairs]
    peak = max(vs)
    above = [i for i, v in enumerate(vs) if v > threshold]
    out = {"available": True,
           "peak_swe_mm": round(peak, 1),
           "mean_swe_mm": round(sum(vs) / len(vs), 1),
           "peak_date": ds[vs.index(peak)],
           "first_date": ds[0], "last_date": ds[-1],
           "threshold_mm": threshold,
           "days_above_threshold": len(above),
           "has_snowpack": bool(above)}
    if above:
        out["first_snow_date"] = ds[above[0]]
        # Melt-out is the LAST day above threshold, not the first day below:
        # a mid-winter thaw dipping under the line for a week would otherwise
        # end the season in January.
        out["melt_out_date"] = ds[above[-1]]
    return out


def pair_stations(stations: List[Dict], columns: List[Dict],
                  on: str = "distance",
                  max_delta_m: float = MAX_PAIR_DELTA_M) -> Tuple[List, List]:
    """One station to one column. A BIJECTION, closest claims first.

    Ported from step1_compare_swe.pair_by_elevation, including the property the
    first version of this module lost: two stations cannot share a column.
    Allowing it "would put two points at the same y on a 1:1 plot, and a
    repeated model value cannot carry the comparison it appears to make."

    `on` is 'elevation' for SWE — elevation governs snowpack, and matching on
    horizontal distance was measured at up to 846 m of elevation offset with
    five stations collapsing onto two columns — and 'distance' otherwise.

    Returns (pairs, unpaired). An unpaired station carries its REASON: excluded
    by a rule and never reported are different findings, and a list that merges
    them cannot tell you which happened.
    """
    def _delta(st, col):
        if on == "elevation":
            if st.get("elevation_m") is None or col.get("elevation_m") is None:
                return None
            return abs(col["elevation_m"] - st["elevation_m"])
        if st.get("lat") is None or col.get("lat") is None:
            return None
        return math.hypot((col.get("lat") or 0) - st["lat"],
                          (col.get("lon") or 0) - (st.get("lon") or 0))

    cands = []
    for st in stations:
        best, delta = None, None
        for col in columns:
            d = _delta(st, col)
            if d is None:
                continue
            if delta is None or d < delta:
                best, delta = col, d
        if best is None:
            continue
        cands.append((delta, st, best))

    cands.sort(key=lambda t: t[0])          # closest claims its column first
    pairs, unpaired, taken = [], [], {}
    for delta, st, col in cands:
        if on == "elevation" and delta > max_delta_m:
            unpaired.append({"station_id": st["station_id"],
                             "reason": f"nearest column is {delta:.0f} m away "
                                       f"in elevation, beyond the "
                                       f"{max_delta_m:.0f} m limit"})
            continue
        cid = col["case_name"]
        if cid in taken:
            unpaired.append({"station_id": st["station_id"],
                             "reason": f"{cid} is already paired with "
                                       f"{taken[cid]}, which is closer"})
            continue
        taken[cid] = st["station_id"]
        p = {"station_id": st["station_id"], "case_name": cid, "matched_on": on}
        if on == "elevation":
            p["station_elevation_m"] = st.get("elevation_m")
            p["column_elevation_m"] = col.get("elevation_m")
            p["delta_elevation_m"] = round(
                (col.get("elevation_m") or 0) - (st.get("elevation_m") or 0), 1)
        pairs.append(p)
    return pairs, unpaired


# ─────────────────────────────────────────────────────────────────────────────
# ONE OBSERVABLE
# ─────────────────────────────────────────────────────────────────────────────
def compare_one(rows: List[Dict], series: Dict, meta: Dict,
                observable: str) -> Dict[str, Any]:
    """Pair every station against every column, and report the numbers.

    EVERY COLUMN AGAINST EVERY STATION, deliberately. Which column a station
    should be compared to is a design question — nearest, same elevation band,
    the pinned one — and answering it here would bury a choice the caller
    should make. The pairs are returned per (station, column) so the caller can
    select. A one-to-one assignment is computed over the whole set afterwards
    and reported as `assignment`, but it is offered, not imposed.
    """
    spec = OBSERVABLES[observable]
    model = model_series(rows, observable)
    stations = {k: v for k, v in series.items()
                if isinstance(k, tuple) and k[1] == observable}

    rec: Dict[str, Any] = {
        "observable": observable,
        "units": spec["units"],
        "model_comparand": spec["model_comparand"],
        "obs_quantity": spec["obs_quantity"],
        "colocated": spec["colocated"],
        "n_columns_with_series": len(model),
        "n_stations": len(stations),
        "pairs": [],
    }
    if not model:
        rec["error"] = (
            f"no column has a daily series for {'+'.join(spec['model'])}. The "
            f"extraction either has not run or predates daily series — this is "
            f"not an absence of agreement, it is an absence of model output.")
        return rec
    if not stations:
        rec["error"] = (f"no observations of '{observable}' in the table — "
                        f"nothing was compared. This is not a disagreement.")
        return rec

    rec["model_period"] = _span(sorted(
        {d for m in model.values() for d in m["dates"]}))

    for (sid, _), obs in stations.items():
        st_meta = meta.get((sid, observable), {})
        entry: Dict[str, Any] = {
            "station_id": sid,
            "obs_period": _span(obs["dates"]),
            "n_obs": len(obs["dates"]),
            "in_basin": st_meta.get("in_basin"),
            "source": st_meta.get("source"),
            "licence": st_meta.get("licence"),
            "obs_units": st_meta.get("units"),
            "columns": [],
        }
        # A units mismatch is REPORTED, never silently converted: guessing that
        # "mm" meant "mm/day" is exactly how a factor of 86400 gets in.
        if st_meta.get("units") and st_meta["units"] != spec["units"]:
            entry["units_mismatch"] = (
                f"station reports {st_meta['units']}, this comparison is in "
                f"{spec['units']} — NOT converted; convert before writing the "
                f"table")
        for case, m in model.items():
            dates, mm, oo, qq = _pair(m, obs)
            if not dates:
                entry["columns"].append(
                    {"case_name": case, "n_pairs": 0,
                     "note": "no shared dates with this station"})
                continue
            col = {"case_name": case,
                   "n_pairs": len(dates),
                   "overlap": _span(dates),
                   "metrics": _metrics(mm, oo),
                   "obs_quality": _quality_breakdown(qq),
                   "lat": m.get("lat"), "lon": m.get("lon"),
                   "elevation_m": m.get("elevation_m")}
            if observable == "wtd":
                # A measured fact about where the model's answer came from.
                deep = sum(1 for v in mm if v is not None
                           and v > ACTIVE_SOIL_DEPTH_M)
                col["frac_below_active_soil"] = round(deep / len(mm), 4)
                col["active_soil_depth_m"] = ACTIVE_SOIL_DEPTH_M
            entry["columns"].append(col)
        paired = [c for c in entry["columns"] if c.get("n_pairs")]
        entry["n_columns_paired"] = len(paired)
        if observable == "swe":
            entry["swe_phenology_obs"] = swe_phenology(obs["dates"], obs["values"])
        rec["pairs"].append(entry)

    rec["n_pairs_total"] = sum(c.get("n_pairs", 0) for e in rec["pairs"]
                               for c in e["columns"])

    # THE ASSIGNMENT, as a bijection over all stations at once. It cannot be
    # decided per station in the loop above — "closest claims first" is a
    # property of the whole set, and a per-station nearest lets every station
    # pick the same column. Reported alongside the full grid rather than
    # replacing it: the caller may have a better rule, and every pair is
    # already there to use.
    on = PAIR_ON.get(observable, "distance")
    sts = [{"station_id": e["station_id"],
            "lat": (meta.get((e["station_id"], observable)) or {}).get("lat"),
            "lon": (meta.get((e["station_id"], observable)) or {}).get("lon"),
            "elevation_m": (meta.get((e["station_id"], observable))
                            or {}).get("elevation_m")}
           for e in rec["pairs"]]
    cols = [{"case_name": c, **{k: v for k, v in m.items()
                                if k in ("lat", "lon", "elevation_m")}}
            for c, m in model.items()]
    assigned, unpaired = pair_stations(sts, cols, on=on)
    rec["assignment"] = {"matched_on": on, "pairs": assigned,
                         "unpaired": unpaired,
                         "note": ("one station to one column, closest first, "
                                  "no column claimed twice. Offered, not "
                                  "imposed — every station-column pair is in "
                                  "`pairs` above")}
    by_station = {a["station_id"]: a["case_name"] for a in assigned}
    for e in rec["pairs"]:
        e["assigned_column"] = by_station.get(e["station_id"])
        if observable == "swe" and e["assigned_column"]:
            m = model.get(e["assigned_column"])
            if m:
                e["swe_phenology_model"] = swe_phenology(m["dates"], m["values"])
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def plot_observable(rec: Dict[str, Any], rows: List[Dict], series: Dict,
                    out_path: str) -> Optional[str]:
    """Two panels for one observable: the series, and model against obs.

    NOTHING ON THE PLOT INTERPRETS. Axis labels name the quantity and its
    units; the title names the observable and the station count. No verdict, no
    "good agreement", no shaded skill bands — the numbers are in the record and
    what they mean is the reader's call.

    The model is drawn as a thin line PER COLUMN rather than an ensemble mean.
    A mean of 19 columns spanning 1500 m of relief is a number no column
    experienced, and drawing it invites reading the spread as uncertainty when
    it is design.

    Returns the path written, or None when there was nothing to draw — which is
    not a failure and is reported as such by the caller.
    """
    pairs = [e for e in (rec.get("pairs") or [])
             if any(c.get("n_pairs") for c in e["columns"])]
    if not pairs:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        import datetime as _dt
    except Exception:                                           # noqa: BLE001
        return None

    obs_name = rec["observable"]
    units = rec["units"]
    model = model_series(rows, obs_name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for kw in ("font.size", "axes.labelsize", "axes.titlesize"):
        plt.rcParams[kw] = 13

    def _d(s):
        return _dt.date.fromisoformat(s)

    for i, m in enumerate(model.values()):
        ax1.plot([_d(x) for x in m["dates"]], m["values"],
                 lw=0.7, color="#9AA7B0", alpha=0.75, zorder=1,
                 label="model columns" if i == 0 else None)
    # Observations as POINTS, not a line: they are samples at a station, and a
    # line between them would draw values nobody measured. Gap-filled points
    # are hollow — the distinction the quality column exists to carry, made
    # visible rather than left in the record for someone to look up.
    for e in pairs:
        sid = e["station_id"]
        obs = series.get((sid, obs_name))
        if not obs:
            continue
        xs = [_d(x) for x in obs["dates"]]
        meas = [i for i, q in enumerate(obs["quality"])
                if q in ("measured", "approved")]
        fill = [i for i in range(len(xs)) if i not in set(meas)]
        if meas:
            ax1.scatter([xs[i] for i in meas], [obs["values"][i] for i in meas],
                        s=16, zorder=3, label=f"{sid} (measured)")
        if fill:
            ax1.scatter([xs[i] for i in fill], [obs["values"][i] for i in fill],
                        s=16, zorder=3, facecolors="none",
                        edgecolors="#C1440E", linewidths=0.8,
                        label=f"{sid} (gap-filled)")
    # FRAMED, on purpose. A frameless legend inside the axes draws its sample
    # markers on the same white ground as the data, and the first version of
    # this figure showed a legend swatch that read as a second observation.
    ax1.legend(fontsize=9, frameon=True, facecolor="white", framealpha=0.92,
               edgecolor="#CCCCCC", loc="best")
    ax1.set_ylabel(f"{obs_name}  [{units}]")
    ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_title(f"{obs_name}: {len(model)} columns, "
                  f"{rec['n_stations']} station(s)")
    if rec.get("colocated") is False:
        # A statement of geometry, not of quality.
        ax1.set_title(ax1.get_title() + "  ·  not co-located")

    # Panel 2 — EVERY PAIR, not the period mean. Two means collapse a season
    # into one dot and hide the thing worth seeing: whether the model tracks
    # the observation across its range or only agrees on average. The pairs are
    # re-derived here rather than carried in the record, which stays numbers.
    lim = []
    for e in pairs:
        sid = e["station_id"]
        obs = series.get((sid, obs_name))
        want = e.get("assigned_column") or next(
            c["case_name"] for c in e["columns"] if c.get("n_pairs"))
        m = model.get(want)
        if not (obs and m):
            continue
        _, mm, oo, qq = _pair(m, obs)
        if not mm:
            continue
        meas = [i for i, q in enumerate(qq) if q in ("measured", "approved")]
        fill = [i for i in range(len(mm)) if i not in set(meas)]
        if meas:
            ax2.scatter([oo[i] for i in meas], [mm[i] for i in meas], s=18,
                        alpha=0.75, zorder=3, label=f"{sid} · {want}")
        if fill:
            ax2.scatter([oo[i] for i in fill], [mm[i] for i in fill], s=18,
                        facecolors="none", edgecolors="#C1440E",
                        linewidths=0.7, zorder=3,
                        label=f"{sid} · {want} (gap-filled)")
        lim += mm + oo
    if lim:
        lo, hi = min(lim), max(lim)
        pad = 0.06 * ((hi - lo) or 1.0)
        lo, hi = lo - pad, hi + pad
        ax2.plot([lo, hi], [lo, hi], ls="--", lw=1, color="#666", zorder=1)
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)
        ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel(f"observed  [{units}]")
    ax2.set_ylabel(f"model  [{units}]")
    ax2.set_title("every pair, assigned column")
    ax2.legend(fontsize=9, frameon=True, facecolor="white", framealpha=0.92,
               edgecolor="#CCCCCC", loc="best")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def compare_all(rows: List[Dict], obs_csv: str, meta_json: str = "",
                observables: Optional[List[str]] = None,
                figure_dir: str = "") -> Dict[str, Any]:
    """Every observable that has both a model series and observations."""
    series, meta = load_observations(obs_csv, meta_json)
    dropped = series.pop("_dropped", 0)
    want = observables or list(OBSERVABLES)
    out: Dict[str, Any] = {
        "observables": {},
        "n_observation_rows_dropped": dropped,
        "note": ("measurements only — no verdict is offered on any of these, "
                 "and every comparison here is context rather than a skill "
                 "claim"),
    }
    figures: Dict[str, Any] = {}
    for name in want:
        if name not in OBSERVABLES:
            out["observables"][name] = {"error": f"unknown observable '{name}'"}
            continue
        rec = compare_one(rows, series, meta, name)
        out["observables"][name] = rec
        if figure_dir:
            # NON-FATAL. The numbers are the product; a figure that will not
            # render must not take the comparison down with it.
            try:
                p = plot_observable(rec, rows, series,
                                    str(Path(figure_dir) / f"compare_{name}.png"))
                if p:
                    figures[name] = p
            except Exception as e:                              # noqa: BLE001
                figures[name] = f"failed: {type(e).__name__}: {e}"[:200]
    out["figures"] = figures
    return out
