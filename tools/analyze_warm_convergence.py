#!/usr/bin/env python3
"""
Year-resolved warm-start convergence analysis.

Answers: does the first-year transplant adjustment decay, and what do the
settled (final-year) recharge and water-table numbers look like next to the
cold start? Reads the daily h0 history of a COLD run (1 yr) and a WARM run
(multi-year, CONUS-transplant finidat), aggregates per calendar year, and
reports per-column: annual recharge (QCHARGE), end-of-year water table (ZWT),
and the year-over-year deltas that measure transient decay.

    python3 tools/analyze_warm_convergence.py \
        --cold <cold_run_dir> --warm <warm3yr_run_dir> --label upper_gunnison

Writes <warm>/04_analysis/warm_convergence.json and prints the table.
"""
import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

SECS_YR = 86400.0 * 365.0


def yearly_series(case_dir):
    """Per-year {year: {recharge_mm_yr, zwt_end_m, zwt_mean_m}} from daily h0."""
    files = sorted(glob.glob(f"{case_dir}/run/*.elm.h0.*.nc"))
    if not files:
        return {}
    qc, zwt, pr, dates = [], [], [], []
    for f in files:
        with Dataset(f) as d:
            t = d.variables["time"]
            base = datetime.strptime(str(t.units).split("since")[1].strip()[:10],
                                     "%Y-%m-%d")
            # daily averages are stamped at interval end; center by -12 h
            dates += [base + timedelta(days=float(v) - 0.5) for v in t[:]]
            qc.append(np.asarray(d.variables["QCHARGE"][:]).reshape(len(t), -1)[:, 0])
            zwt.append(np.asarray(d.variables["ZWT"][:]).reshape(len(t), -1)[:, 0])
            pr.append(np.asarray(d.variables["RAIN"][:]).reshape(len(t), -1)[:, 0]
                      + np.asarray(d.variables["SNOW"][:]).reshape(len(t), -1)[:, 0])
    qc, zwt, pr = np.concatenate(qc), np.concatenate(zwt), np.concatenate(pr)
    by_year = defaultdict(list)
    for i, dt in enumerate(dates):
        by_year[dt.year].append(i)
    out = {}
    for y, idx in sorted(by_year.items()):
        if len(idx) < 300:            # partial year (spinup stub/rollover) — skip
            continue
        idx = np.array(idx)
        out[y] = {"recharge_mm_yr": float(np.mean(qc[idx]) * SECS_YR),
                  "precip_mm_yr": float(np.mean(pr[idx]) * SECS_YR),
                  "zwt_end_m": float(zwt[idx][-1]),
                  "zwt_mean_m": float(np.mean(zwt[idx]))}
    return out


def collect(run_dir):
    cases = json.loads((Path(run_dir) / "cases.json").read_text())
    return {c.split(".")[-1]: yearly_series(c) for c in cases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", required=True, help="cold run dir (1 yr)")
    ap.add_argument("--warm", required=True, help="warm run dir (multi-year)")
    ap.add_argument("--label", default="basin")
    args = ap.parse_args()

    cold, warm = collect(args.cold), collect(args.warm)
    cols = sorted(set(cold) & set(warm))

    rows, decayed = [], 0
    print(f"\n{args.label}: cold 1-yr vs CONUS-warm by year   (recharge mm/yr; "
          f"ZWT end-of-year m)")
    print(f"{'col':8s} {'cold_R':>8s} | {'warm_R_y1':>9s} {'y2':>8s} {'y3':>8s} "
          f"| {'d21':>7s} {'d32':>7s} {'settling':>9s} | {'ZWT_y3':>7s}")
    for c in cols:
        yc = list(cold[c].values())[0] if cold[c] else None
        yw = [v for _, v in sorted(warm[c].items())]
        if yc is None or len(yw) < 3:
            continue
        r = [v["recharge_mm_yr"] for v in yw[:3]]
        p = [v["precip_mm_yr"] for v in yw[:3]]
        d21, d32 = r[1] - r[0], r[2] - r[1]
        # 3-way call with a 1 mm/yr noise floor; forcing varies year to year,
        # so "growing" may be climate response, not transient — see precip.
        if abs(d21) < 1.0 and abs(d32) < 1.0:
            settling = "settled"
        elif abs(d32) < abs(d21):
            settling = "decaying"
        else:
            settling = "growing"
        decayed += settling in ("settled", "decaying")
        rows.append({"col": c, "cold_recharge_mm_yr": round(yc["recharge_mm_yr"], 2),
                     "warm_recharge_mm_yr_by_year": [round(x, 2) for x in r],
                     "precip_mm_yr_by_year": [round(x, 0) for x in p],
                     "delta_y2_y1": round(d21, 2), "delta_y3_y2": round(d32, 2),
                     "settling": settling,
                     "warm_zwt_end_y3_m": round(yw[2]["zwt_end_m"], 2),
                     "cold_zwt_end_m": round(yc["zwt_end_m"], 2)})
        print(f"{c:8s} {yc['recharge_mm_yr']:8.2f} | {r[0]:9.2f} {r[1]:8.2f} "
              f"{r[2]:8.2f} | {d21:7.2f} {d32:7.2f} {settling:>9s} "
              f"| {yw[2]['zwt_end_m']:7.2f}")

    n = len(rows)
    mean_y3 = float(np.mean([r["warm_recharge_mm_yr_by_year"][2] for r in rows]))
    mean_cold = float(np.mean([r["cold_recharge_mm_yr"] for r in rows]))
    print(f"\n  {decayed}/{n} columns show |y3-y2| < |y2-y1| (transient decaying)")
    print(f"  mean recharge: cold {mean_cold:+.2f}  warm-y3 {mean_y3:+.2f} mm/yr")

    out = Path(args.warm) / "04_analysis" / "warm_convergence.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"label": args.label, "columns": rows,
                               "n_decaying": decayed, "n": n,
                               "mean_cold": round(mean_cold, 2),
                               "mean_warm_y3": round(mean_y3, 2)}, indent=1))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
