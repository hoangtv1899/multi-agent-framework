#!/usr/bin/env python3
"""The half of comparison that is the same for every observable.

Pairing, metrics and quality accounting do not depend on what is being
compared, so they live once here. Everything that DOES depend on the
observable — what a peak date means, whether a water table needs a log axis,
whether a gauge is co-located at all — lives in that observable's own module.

The split is the point. Two copies of an NSE would drift; two copies of "what
snowpack phenology means" never existed to begin with.
"""
from __future__ import annotations

import csv as _csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class Spec:
    """What one observable is, in the few facts every stage needs.

    Small and declarative on purpose: a new observable should be a SPEC plus a
    compare() and a plot(), not a new branch in shared code.
    """

    def __init__(self, name: str, model_vars: List[str], units: str,
                 comparand: str, obs_quantity: str,
                 colocated: bool = True, pair_on: str = "distance"):
        self.name = name
        self.model_vars = model_vars
        self.units = units
        self.comparand = comparand
        self.obs_quantity = obs_quantity
        self.colocated = colocated
        self.pair_on = pair_on


MAX_PAIR_DELTA_M = 200.0        # elevation pairing limit; see pair_stations


# ── reading what the caller wrote ────────────────────────────────────────────
def load_observations(obs_csv: str, meta_json: str = "") -> Tuple[Dict, Dict, int]:
    """(series, meta, n_dropped) from the long CSV and its metadata.

    series is {(station_id, variable): {dates, values, quality}}, chronological.
    Rows with an unparseable value are DROPPED AND COUNTED, never coerced to
    zero: a missing snow measurement is not zero snow.
    """
    series: Dict[Tuple[str, str], Dict[str, List]] = {}
    dropped = 0
    with open(obs_csv, newline="") as fh:
        for row in _csv.DictReader(fh):
            sid = (row.get("station_id") or "").strip()
            var = (row.get("variable") or "").strip().lower()
            t = (row.get("time") or "").strip()[:10]
            try:
                v = float(row.get("value"))
            except (TypeError, ValueError):
                v = float("nan")
            if not (sid and var and t) or math.isnan(v):
                dropped += 1
                continue
            s = series.setdefault((sid, var),
                                  {"dates": [], "values": [], "quality": []})
            s["dates"].append(t)
            s["values"].append(v)
            s["quality"].append((row.get("quality") or "unknown").strip().lower())
    for s in series.values():
        order = sorted(range(len(s["dates"])), key=lambda i: s["dates"][i])
        for k in ("dates", "values", "quality"):
            s[k] = [s[k][i] for i in order]

    meta: Dict[Tuple[str, str], Dict] = {}
    if meta_json and Path(meta_json).is_file():
        for m in json.loads(Path(meta_json).read_text()) or []:
            meta[(str(m.get("station_id")),
                  str(m.get("variable", "")).lower())] = m
    return series, meta, dropped


def stations_for(series: Dict, name: str) -> Dict[str, Dict]:
    """{station_id: series} for one observable."""
    return {k[0]: v for k, v in series.items()
            if isinstance(k, tuple) and k[1] == name}


def model_series(rows: List[Dict], variables: List[str]) -> Dict[str, Dict]:
    """{case_name: {dates, values, ...}}, the named ELM variables summed.

    A date survives only when EVERY component has a value there. A partial sum
    is a smaller number, not a missing one, and it would read as the model
    drying out rather than as an absent field.
    """
    out: Dict[str, Dict] = {}
    for row in rows or []:
        name = row.get("case_name")
        if not name:
            continue
        parts = []
        for var in variables:
            d = ((row.get("variables") or {}).get(var) or {}).get("daily") or {}
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
                     "lat": row.get("lat"), "lon": row.get("lon"),
                     "elevation_m": row.get("elevation_m")}
    return out


# ── pairing ──────────────────────────────────────────────────────────────────
def pair(m: Dict, o: Dict) -> Tuple[List, List, List, List]:
    """Model and observation on their SHARED dates: (dates, model, obs, quality).

    An inner join, never interpolation. Filling a gap to make the arrays line up
    invents a measurement, and the pair count would then describe the invention.
    """
    mv = dict(zip(m["dates"], m["values"]))
    dates, mm, oo, qq = [], [], [], []
    for i, d in enumerate(o["dates"]):
        if d in mv:
            dates.append(d)
            mm.append(mv[d])
            oo.append(o["values"][i])
            qq.append(o["quality"][i] if i < len(o.get("quality") or []) else "unknown")
    return dates, mm, oo, qq


def pair_stations(stations: List[Dict], columns: List[Dict],
                  on: str = "distance",
                  max_delta_m: float = MAX_PAIR_DELTA_M) -> Tuple[List, List]:
    """One station to one column. A BIJECTION, closest claiming first.

    Ported from step1_compare_swe.pair_by_elevation. `on='elevation'` for SWE
    because elevation governs snowpack — matching on horizontal distance was
    measured at up to 846 m of elevation offset with five stations collapsing
    onto two columns. Two stations may not share a column: "a repeated model
    value cannot carry the comparison it appears to make."

    An unpaired station carries its REASON. Excluded by a rule and never
    reported are different findings.
    """
    def delta(st, col):
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
        best, d_best = None, None
        for col in columns:
            d = delta(st, col)
            if d is not None and (d_best is None or d < d_best):
                best, d_best = col, d
        if best is not None:
            cands.append((d_best, st, best))

    cands.sort(key=lambda t: t[0])
    pairs, unpaired, taken = [], [], {}
    for d, st, col in cands:
        if on == "elevation" and d > max_delta_m:
            unpaired.append({"station_id": st["station_id"],
                             "reason": f"nearest column is {d:.0f} m away in "
                                       f"elevation, beyond the {max_delta_m:.0f} m limit"})
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
            p["delta_elevation_m"] = round(
                (col.get("elevation_m") or 0) - (st.get("elevation_m") or 0), 1)
        pairs.append(p)
    return pairs, unpaired


# ── metrics ──────────────────────────────────────────────────────────────────
def metrics(m: List[float], o: List[float]) -> Dict[str, Any]:
    """bias · MAE · RMSE · r · NSE · KGE. No thresholds, no verdicts.

    NSE and KGE are undefined on a zero-variance observation — a well that never
    moved, a season with no snow — and returning 0.0 would be a made-up score.
    They come back None, so "no skill" and "not computable" stay distinct.
    """
    n = len(m)
    if n == 0:
        return {"n": 0}
    mean_m, mean_o = sum(m) / n, sum(o) / n
    var_m = sum((a - mean_m) ** 2 for a in m)
    var_o = sum((b - mean_o) ** 2 for b in o)
    cov = sum((a - mean_m) * (b - mean_o) for a, b in zip(m, o))
    r = cov / math.sqrt(var_m * var_o) if var_m > 0 and var_o > 0 else None
    sse = sum((a - b) ** 2 for a, b in zip(m, o))
    nse = 1.0 - sse / var_o if var_o > 0 else None
    kge = None
    if r is not None and mean_o != 0 and var_o > 0:
        kge = 1.0 - math.sqrt((r - 1) ** 2
                              + (math.sqrt(var_m / n) / math.sqrt(var_o / n) - 1) ** 2
                              + (mean_m / mean_o - 1) ** 2)
    rnd = lambda x: None if x is None else round(x, 4)          # noqa: E731
    return {"n": n, "model_mean": rnd(mean_m), "obs_mean": rnd(mean_o),
            "bias": rnd(mean_m - mean_o),
            "mae": rnd(sum(abs(a - b) for a, b in zip(m, o)) / n),
            "rmse": rnd(math.sqrt(sse / n)),
            "pearson_r": rnd(r), "nse": rnd(nse), "kge": rnd(kge)}


MEASURED = ("measured", "approved")


def quality(q: List[str]) -> Dict[str, Any]:
    """How much of the paired observation was actually measured.

    At US-NR1 in 2016 the gap-filled annual ET is 464 mm and the measured
    half-hours alone give 89 mm, because 36% of the year was observed. Both are
    true; they are not the same claim.
    """
    counts: Dict[str, int] = {}
    for x in q:
        counts[x] = counts.get(x, 0) + 1
    n = len(q) or 1
    return {"counts": counts,
            "frac_measured": round(sum(counts.get(k, 0) for k in MEASURED) / n, 4),
            "frac_gap_filled": round(counts.get("filled", 0) / n, 4)}


def span(dates: List[str]) -> Optional[List[str]]:
    return [dates[0], dates[-1]] if dates else None


# ── the standard comparison every observable starts from ─────────────────────
def standard_compare(spec: Spec, rows: List[Dict], series: Dict,
                     meta: Dict) -> Dict[str, Any]:
    """Every station against every column, plus the one-to-one assignment.

    EVERY station against EVERY column, deliberately: which column a station
    ought to be compared to is a design question, and answering it only inside
    here would bury a choice the caller should be able to override. The
    bijection is computed over the whole set afterwards and reported as
    `assignment` — offered, not imposed.

    An observable's own module calls this and then adds what only it knows.
    """
    model = model_series(rows, spec.model_vars)
    stations = stations_for(series, spec.name)
    rec: Dict[str, Any] = {
        "observable": spec.name, "units": spec.units,
        "model_comparand": spec.comparand, "obs_quantity": spec.obs_quantity,
        "colocated": spec.colocated,
        "n_columns_with_series": len(model), "n_stations": len(stations),
        "pairs": [],
    }
    if not model:
        rec["error"] = (f"no column has a daily series for "
                        f"{'+'.join(spec.model_vars)} — the extraction either "
                        f"has not run or predates daily series. This is an "
                        f"absence of model output, not of agreement.")
        return rec
    if not stations:
        rec["error"] = (f"no observations of '{spec.name}' in the table — "
                        f"nothing was compared. This is not a disagreement.")
        return rec

    rec["model_period"] = span(sorted({d for m in model.values()
                                       for d in m["dates"]}))
    for sid, obs in stations.items():
        st = meta.get((sid, spec.name), {})
        entry: Dict[str, Any] = {
            "station_id": sid, "obs_period": span(obs["dates"]),
            "n_obs": len(obs["dates"]), "in_basin": st.get("in_basin"),
            "source": st.get("source"), "licence": st.get("licence"),
            "columns": [],
        }
        # A units mismatch is REPORTED, never silently converted: assuming "mm"
        # meant "mm/day" is how a factor of 86400 gets in.
        if st.get("units") and st["units"] != spec.units:
            entry["units_mismatch"] = (
                f"station reports {st['units']}, this comparison is in "
                f"{spec.units} — NOT converted; fix the table")
        for case, m in model.items():
            dates, mm, oo, qq = pair(m, obs)
            if not dates:
                entry["columns"].append({"case_name": case, "n_pairs": 0,
                                         "note": "no shared dates"})
                continue
            entry["columns"].append({
                "case_name": case, "n_pairs": len(dates),
                "overlap": span(dates), "metrics": metrics(mm, oo),
                "obs_quality": quality(qq)})
        entry["n_columns_paired"] = sum(1 for c in entry["columns"]
                                        if c.get("n_pairs"))
        rec["pairs"].append(entry)

    rec["n_pairs_total"] = sum(c.get("n_pairs", 0) for e in rec["pairs"]
                               for c in e["columns"])
    sts = [{"station_id": e["station_id"],
            **{k: (meta.get((e["station_id"], spec.name)) or {}).get(k)
               for k in ("lat", "lon", "elevation_m")}} for e in rec["pairs"]]
    cols = [{"case_name": c, **{k: m.get(k) for k in
                                ("lat", "lon", "elevation_m")}}
            for c, m in model.items()]
    assigned, unpaired = pair_stations(sts, cols, on=spec.pair_on)
    rec["assignment"] = {"matched_on": spec.pair_on, "pairs": assigned,
                         "unpaired": unpaired}
    by_station = {a["station_id"]: a["case_name"] for a in assigned}
    for e in rec["pairs"]:
        e["assigned_column"] = by_station.get(e["station_id"])
    return rec


def assigned_pairs(rec: Dict, model: Dict, series: Dict, name: str):
    """Yield (station_id, case_name, dates, model, obs, quality) for the
    assigned column of each station. The shape every plot() needs."""
    for e in rec.get("pairs") or []:
        case = e.get("assigned_column")
        obs = series.get((e["station_id"], name))
        m = model.get(case) if case else None
        if not (m and obs):
            continue
        dates, mm, oo, qq = pair(m, obs)
        if dates:
            yield e["station_id"], case, dates, mm, oo, qq


# ── plotting helpers ─────────────────────────────────────────────────────────
def new_figure(ncols: int = 2, width: float = 13.0, height: float = 4.6):
    """A figure with this project's plot conventions already applied.

    Large type, and NOTHING interpretive on the canvas: axis labels name the
    quantity and its units, titles name what is drawn. What a panel MEANS
    belongs in the record, which is numbers, and in whatever reads it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 13,
                         "axes.titlesize": 13, "legend.fontsize": 9})
    return plt.subplots(1, ncols, figsize=(width, height))


def legend(ax):
    """Framed, always. A frameless legend inside the axes draws its sample
    markers on the same ground as the data, and one early version of these
    figures showed a legend swatch that read as an extra observation."""
    ax.legend(frameon=True, facecolor="white", framealpha=0.92,
              edgecolor="#CCCCCC", loc="best")


def save(fig, out_path: str) -> str:
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(out_path)


def one_to_one(ax, xs: List[float], ys: List[float]) -> None:
    """A 1:1 line over the data's own range, with equal axes."""
    vals = [v for v in list(xs) + list(ys) if v is not None]
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    pad = 0.06 * ((hi - lo) or 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            ls="--", lw=1, color="#666", zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")


def split_quality(qq: List[str]) -> Tuple[List[int], List[int]]:
    """(indices measured, indices gap-filled or unknown)."""
    meas = [i for i, q in enumerate(qq) if q in MEASURED]
    return meas, [i for i in range(len(qq)) if i not in set(meas)]
