#!/usr/bin/env python3
"""The basin's baseflow recession timescale from the gauges already on disk.

    python tools/recession_tau.py <run_dir> --out <json> [--exclude <gauge id>]

Section 4 of docs/coupling/lateral_sink_design.md: sink_tau_days is REQUIRED
when a sink is requested and has no hidden default; it comes from here, out
of the run's own reception.json. Two methods on the same recession segments,
both reported:

  master  the master recession: a pooled fixed-effect slope of ln Q against
          time over every kept segment (each segment demeaned, so only the
          slope is shared), tau = -1 / slope;
  dqdt    -dQ/dt against Q on consecutive segment days, binned in log10 Q;
          the binned medians give tau = Q / (-dQ/dt) with the exponent
          b = 1, and b is also fitted free for inspection.

Segment rule: falling days (Q[i] <= Q[i-1]); a rise day and the 3 days
after it are dropped; days with basin-mean forcing precipitation (RAIN +
SNOW) above 1 mm/day, today or the day before, are dropped when
experiment.json carries the forcing (days the forcing does not cover are
dropped too); days on which any SNOTEL station's daily SWE fell by more
than 2 mm are dropped when reception.json carries daily SWE. Runs of at
least 5 kept consecutive days that fall overall are segments. Regulated
gauges are excluded by id (--exclude): still reported, never pooled.

The pooled tau per basin is the same fixed-effect slope over the kept
segments of every kept in-basin gauge. Q is reception's mm/day; tau does
not depend on the unit.
"""
import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

RAIN_THRESHOLD_MM_DAY = 1.0
MELT_THRESHOLD_MM_DAY = 2.0
RISE_SHADOW_DAYS = 3
MIN_SEGMENT_DAYS = 5

# The dQ/dt method needs enough pairs to bin: at least MIN_PAIRS pairs,
# N_BINS equal bins in log10 Q, a bin counts with MIN_PER_BIN pairs, and a
# fit needs MIN_BINS bins.
MIN_PAIRS = 10
N_BINS = 8
MIN_PER_BIN = 5
MIN_BINS = 3


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def _tau(k):
    """1/k in days, or None when the slope does not describe a recession."""
    return _r(1.0 / k) if k > 0 else None


def _floats(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


# ── what is on disk ───────────────────────────────────────────────────────

def load_gauges(reception):
    """Every streamflow station with a daily record, in reception's order."""
    out = []
    for s in reception["observations"]["streamflow"]["stations"]:
        mm = s.get("mm_day") or {}
        if not mm:
            continue
        keys = sorted(mm)
        out.append({"id": s["id"], "name": s.get("name"),
                    "in_basin": bool(s.get("in_basin")),
                    "drainage_area_km2": s.get("drainage_area_km2"),
                    "dates": [date.fromisoformat(k) for k in keys],
                    "q": _floats([mm[k] for k in keys])})
    return out


def load_swe(reception):
    """SNOTEL daily SWE per station, {name: (dates, mm)}; empty when absent."""
    out = {}
    swe = reception["observations"].get("swe") or {}
    for s in swe.get("stations") or []:
        d = s.get("daily") or {}
        if d.get("dates"):
            out[s.get("name") or s.get("triplet")] = (
                [date.fromisoformat(x) for x in d["dates"]], _floats(d["values"]))
    return out


def load_forcing(experiment):
    """Basin-mean daily precipitation, RAIN + SNOW in mm/day, over the
    columns that ran; None when experiment.json carries no forcing."""
    series, dates = [], None
    for c in experiment.get("columns") or []:
        v = c.get("variables") or {}
        if c.get("status") != "ok" or "RAIN" not in v or "SNOW" not in v:
            continue
        rain, snow = v["RAIN"]["daily"], v["SNOW"]["daily"]
        if rain["dates"] != snow["dates"]:
            raise ValueError(f"{c.get('case_name')}: RAIN and SNOW daily "
                             f"dates differ")
        if dates is None:
            dates = rain["dates"]
        elif rain["dates"] != dates:
            raise ValueError(f"{c.get('case_name')}: its forcing dates differ "
                             f"from the first column's")
        series.append(_floats(rain["values"]) + _floats(snow["values"]))
    if not series:
        return None
    return {"dates": [date.fromisoformat(x) for x in dates],
            "mm_day": np.nanmean(np.array(series), axis=0),
            "n_columns": len(series)}


# ── the segment rule ──────────────────────────────────────────────────────

def day_flags(dates, forcing=None, swe=None):
    """Per gauge day: wet, known (the forcing covers it), melt, and drop.

    drop is what the segment rule removes: melt days always; with forcing
    on disk also wet days and days the forcing does not cover.
    """
    n = len(dates)
    wet, known, melt = (np.zeros(n, bool) for _ in range(3))
    if forcing is not None:
        p = dict(zip(forcing["dates"], forcing["mm_day"]))
        for i, d in enumerate(dates):
            today, yday = p.get(d, np.nan), p.get(d - timedelta(days=1), np.nan)
            if not np.isfinite(today):
                continue
            known[i] = True
            wet[i] = (today > RAIN_THRESHOLD_MM_DAY or
                      (np.isfinite(yday) and yday > RAIN_THRESHOLD_MM_DAY))
    for _name, (sd, sv) in (swe or {}).items():
        s = dict(zip(sd, sv))
        for i, d in enumerate(dates):
            a, b = s.get(d - timedelta(days=1), np.nan), s.get(d, np.nan)
            if np.isfinite(a) and np.isfinite(b) and a - b > MELT_THRESHOLD_MM_DAY:
                melt[i] = True
    drop = melt | ((wet | ~known) if forcing is not None else False)
    return {"wet": wet, "known": known, "melt": melt, "drop": drop}


def segments(q, drop=None, dates=None):
    """Recession segments as lists of indices, after the segment rule.

    A day is kept when it falls (Q[i] <= Q[i-1]), is positive, is not a
    rise day nor within RISE_SHADOW_DAYS after one, is not dropped, and
    (when dates are given) follows the previous day on the calendar. Runs
    of MIN_SEGMENT_DAYS kept days that fall overall are segments; a flat
    run is not a recession.
    """
    q = np.asarray(q, dtype=float)
    n = q.size
    drop = np.zeros(n, bool) if drop is None else np.asarray(drop, bool)
    keep = np.zeros(n, bool)
    shadow = -1                      # last index shadowed by a rise
    for i in range(1, n):
        if q[i] > q[i - 1]:
            shadow = i + RISE_SHADOW_DAYS
        gap = dates is not None and dates[i] - dates[i - 1] != timedelta(days=1)
        keep[i] = (q[i] <= q[i - 1] and q[i] > 0 and i > shadow
                   and not drop[i] and not gap)
    out, cur = [], []
    for i in range(n + 1):
        if i < n and keep[i]:
            cur.append(i)
            continue
        if len(cur) >= MIN_SEGMENT_DAYS and q[cur[0]] > q[cur[-1]]:
            out.append(cur)
        cur = []
    return out


# ── the two methods ───────────────────────────────────────────────────────

def _segment_tau(t, y):
    slope = np.polyfit(t, y, 1)[0]
    return -1.0 / slope if slope < 0 else np.inf


def _demeaned(q, segs):
    """(t, ln Q) demeaned per segment, ready to pool; plus per-segment tau."""
    xs, ys, taus = [], [], []
    for s in segs:
        t = np.arange(len(s), dtype=float)
        y = np.log(np.asarray(q, dtype=float)[s])
        xs.append(t - t.mean())
        ys.append(y - y.mean())
        taus.append(_segment_tau(t, y))
    return xs, ys, taus


def _pooled_slope(xs, ys, taus):
    """The fixed-effect slope with its 2-se band, as tau."""
    n_seg = len(xs)
    if n_seg == 0:
        return {"n_segments": 0, "n_days": 0, "tau_days": None}
    x, y = np.concatenate(xs), np.concatenate(ys)
    sxx = float((x * x).sum())
    slope = float((x * y).sum() / sxx)
    resid = y - slope * x
    se = float(np.sqrt((resid ** 2).sum() / max(x.size - n_seg - 1, 1) / sxx))
    k = -slope
    finite = [t for t in taus if np.isfinite(t)]
    return {"n_segments": n_seg, "n_days": int(x.size),
            "k_per_day": _r(k, 5), "k_se_per_day": _r(se, 5),
            "tau_days": _tau(k), "tau_lo_days": _tau(k + 2 * se),
            "tau_hi_days": _tau(k - 2 * se),
            "segment_tau_median_days": _r(np.median(finite)) if finite else None,
            "segment_tau_iqr_days": ([_r(np.percentile(finite, 25)),
                                      _r(np.percentile(finite, 75))]
                                     if finite else None)}


def master_recession(q, segs):
    """Pooled fixed-effect slope of ln Q against day over the segments."""
    return _pooled_slope(*_demeaned(q, segs))


def dqdt_binned(q, segs):
    """-dQ/dt against Q on consecutive segment days, binned in log10 Q."""
    q = np.asarray(q, dtype=float)
    qm, dq = [], []
    for s in segs:
        for a, b in zip(s[:-1], s[1:]):
            qm.append(0.5 * (q[a] + q[b]))
            dq.append(q[a] - q[b])
    qm, dq = np.array(qm), np.array(dq)
    ok = dq > 0                      # a flat pair carries no rate
    qm, dq = qm[ok], dq[ok]
    out = {"n_pairs": int(qm.size), "n_bins": 0, "tau_days": None}
    if qm.size < MIN_PAIRS:
        return out
    lq = np.log10(qm)
    edges = np.linspace(lq.min(), lq.max() + 1e-9, N_BINS + 1)
    bq, bd, bn = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lq >= lo) & (lq < hi)
        if m.sum() >= MIN_PER_BIN:
            bq.append(10 ** lq[m].mean())
            bd.append(np.median(dq[m]))
            bn.append(int(m.sum()))
    out.update({"n_bins": len(bq), "pairs_per_bin": bn,
                "tau_pointwise_median_days": _r(1.0 / np.median(dq / qm))})
    if len(bq) < MIN_BINS:
        return out
    lbq, lbd = np.log10(bq), np.log10(bd)
    b, loga = np.polyfit(lbq, lbd, 1)
    k = float(10 ** np.mean(lbd - lbq))
    out.update({"b_free": _r(b, 3), "a_free": _r(10 ** loga, 5),
                "k_per_day": _r(k, 5), "tau_days": _tau(k)})
    return out


# ── per gauge and per basin ───────────────────────────────────────────────

def analyse_gauge(g, flags):
    """One gauge's segments and both methods."""
    segs = segments(g["q"], flags["drop"], g["dates"])
    return {"id": g["id"], "name": g["name"],
            "drainage_area_km2": g["drainage_area_km2"],
            "n_days": int(g["q"].size),
            "first": str(g["dates"][0]), "last": str(g["dates"][-1]),
            "q_mean_mm_day": _r(np.nanmean(g["q"]), 4),
            "n_wet_days": int(flags["wet"].sum()),
            "n_unknown_forcing_days": int((~flags["known"]).sum()),
            "n_melt_days": int(flags["melt"].sum()),
            "n_segments": len(segs), "n_segment_days": sum(len(s) for s in segs),
            "segments": [{"start": str(g["dates"][s[0]]),
                          "end": str(g["dates"][s[-1]]), "n_days": len(s),
                          "tau_days": _r(t) if np.isfinite(t) else None}
                         for s, t in zip(segs, _demeaned(g["q"], segs)[2])],
            "master": master_recession(g["q"], segs),
            "dqdt": dqdt_binned(g["q"], segs),
            "_segs": segs}


def pooled_tau(gauges, analysed):
    """The basin's fixed-effect slope over every kept gauge's segments."""
    xs, ys, taus, used = [], [], [], []
    for g, a in zip(gauges, analysed):
        if a["excluded"] or not a["_segs"]:
            continue
        x, y, t = _demeaned(g["q"], a["_segs"])
        xs += x
        ys += y
        taus += t
        used.append(g["id"])
    out = _pooled_slope(xs, ys, taus)
    out.update({"n_gauges": len(used), "gauges": used})
    return out


def compute(run_dir, exclude=()):
    """The report for a run: per in-basin gauge and pooled per basin."""
    run_dir = Path(run_dir).resolve()
    reception = json.loads((run_dir / "reception.json").read_text())
    exp_path = run_dir / "experiment.json"
    experiment = json.loads(exp_path.read_text()) if exp_path.is_file() else {}
    gauges = [g for g in load_gauges(reception) if g["in_basin"]]
    if not gauges:
        raise ValueError(f"{run_dir}: reception.json holds no in-basin gauge "
                         f"with a daily record, so tau cannot be computed")
    ids = [g["id"] for g in gauges]
    unknown = [x for x in exclude if x not in ids]
    if unknown:
        raise ValueError(f"--exclude names {unknown} but the run's in-basin "
                         f"gauges are {ids}")
    gauges.sort(key=lambda g: -(g["drainage_area_km2"] or 0.0))
    forcing = load_forcing(experiment)
    swe = load_swe(reception)

    analysed = []
    for g in gauges:
        row = analyse_gauge(g, day_flags(g["dates"], forcing, swe))
        row["excluded"] = g["id"] in exclude
        analysed.append(row)
    pooled = pooled_tau(gauges, analysed)
    for row in analysed:
        del row["_segs"]

    melt_rule = (f"any SNOTEL daily SWE falling by more than "
                 f"{MELT_THRESHOLD_MM_DAY} mm/day ({sorted(swe)})"
                 if swe else "no daily SWE on disk: no melt exclusion")
    rain_rule = (f"basin-mean RAIN + SNOW over {forcing['n_columns']} columns "
                 f"above {RAIN_THRESHOLD_MM_DAY} mm/day today or the day before, "
                 f"{forcing['dates'][0]}..{forcing['dates'][-1]}; days outside "
                 f"the forcing dropped" if forcing else
                 "no RAIN/SNOW in experiment.json: no rain exclusion")
    domain = reception.get("brief", {}).get("domain", {})
    return {
        "run_dir": str(run_dir), "basin": domain.get("name"),
        "method": {
            "segment_rule": (f"falling days; a rise day and the "
                             f"{RISE_SHADOW_DAYS} days after it dropped; runs "
                             f"of >= {MIN_SEGMENT_DAYS} consecutive kept days "
                             f"that fall overall"),
            "rain_rule": rain_rule, "melt_rule": melt_rule,
            "master": "pooled fixed-effect slope of ln Q against day, "
                      "segments demeaned; tau = -1/slope",
            "dqdt": (f"-dQ/dt against Q on consecutive segment days, "
                     f"{N_BINS} log10 Q bins of >= {MIN_PER_BIN} pairs, "
                     f">= {MIN_BINS} bins; tau = 1/geometric mean of "
                     f"median(-dQ/dt)/Q per bin (b = 1)"),
            "pooled": "master method over the kept segments of every "
                      "in-basin gauge not excluded",
        },
        "constants": {"rain_threshold_mm_day": RAIN_THRESHOLD_MM_DAY,
                      "melt_threshold_mm_day": MELT_THRESHOLD_MM_DAY,
                      "rise_shadow_days": RISE_SHADOW_DAYS,
                      "min_segment_days": MIN_SEGMENT_DAYS},
        "forcing": ({"available": True, "n_columns": forcing["n_columns"],
                     "first": str(forcing["dates"][0]),
                     "last": str(forcing["dates"][-1]),
                     "n_days": len(forcing["dates"])}
                    if forcing else {"available": False}),
        "swe": {"available": bool(swe), "stations": sorted(swe)},
        "excluded": list(exclude),
        "n_gauges_in_basin": len(gauges),
        "gauges": analysed,
        "pooled": pooled,
        "sink_tau_days": pooled["tau_days"],
        "sink_tau_source": (f"tools/recession_tau.py master recession pooled "
                            f"over {pooled['n_gauges']} in-basin gauge(s) "
                            f"{pooled['gauges']}: {pooled['n_segments']} "
                            f"segments, {pooled['n_days']} days; excluded "
                            f"{list(exclude)}; rain rule: {rain_rule}; melt "
                            f"rule: {melt_rule}"),
    }


def outside_run(out, run_dir):
    """Refuse an --out inside the run: the run is read-only to this tool."""
    out, run_dir = Path(out).resolve(), Path(run_dir).resolve()
    if out == run_dir or run_dir in out.parents:
        raise ValueError(f"--out {out} lies inside the run {run_dir}, which "
                         f"this tool never writes into")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", action="append", default=[],
                    help="gauge id to keep out of the pool (regulated); repeatable")
    a = ap.parse_args()
    try:
        out = outside_run(a.out, a.run_dir)
        report = compute(a.run_dir, a.exclude)
    except ValueError as e:
        sys.exit(str(e))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    for g in report["gauges"]:
        m, d = g["master"], g["dqdt"]
        print(f"  {g['id']} {g['name']} ({g['drainage_area_km2']} km^2)"
              f"{' EXCLUDED' if g['excluded'] else ''}: {g['n_segments']} "
              f"segments / {g['n_segment_days']} days; master tau "
              f"{m['tau_days']} d ({m.get('tau_lo_days')}..{m.get('tau_hi_days')}); "
              f"dQ/dt tau {d['tau_days']} d (b {d.get('b_free')}, "
              f"{d['n_pairs']} pairs, {d['n_bins']} bins)")
    p = report["pooled"]
    print(f"{report['basin']}: pooled tau {p['tau_days']} d "
          f"({p.get('tau_lo_days')}..{p.get('tau_hi_days')}) over "
          f"{p['n_gauges']} gauges, {p['n_segments']} segments, "
          f"{p['n_days']} days")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
