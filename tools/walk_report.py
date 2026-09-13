#!/usr/bin/env python3
"""
Read a finished walk directory and print what happened, column by column.

    python tools/walk_report.py WALK_DIR [--against OTHER_WALK_DIR] [--json OUT]

For each column: the water table at the start and the end of the year, its
span, what the ELM legs sent down (the forward flux), what the lateral sink
took out (when there is one), the e-folding time of the outflow over the
first two windows against the recession dial, and whether the column left the
walk early. With --against, the same columns from another walk (a sealed twin,
say) are shown beside it. Nothing here runs a model or reads a deck; it reads
walk.json, walk_log.jsonl and walk_summary.json and does arithmetic on them.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


def load(wd):
    wd = Path(wd)
    w = json.loads((wd / "walk.json").read_text())
    s = json.loads((wd / "walk_summary.json").read_text())
    rows = [json.loads(l) for l in (wd / "walk_log.jsonl").read_text().splitlines()
            if l.strip()] if (wd / "walk_log.jsonl").is_file() else []
    return w, s, rows


def per_column(w, s, rows):
    out = {}
    tau = ((w.get("sink") or {}).get("tau_days"))
    start = {c["id"]: c.get("water_table_m") for c in w["columns"]}
    for cid, t in (s.get("solved_trajectories") or {}).items():
        wt = [x for x in t.get("wt_solved_m") or [] if x is not None]
        q = [x for x in (t.get("lateral_outflow_window_mm") or []) if x is not None]
        fwd = [r["columns"].get(cid, {}).get("forward_mm_window", 0.0) for r in rows
               if isinstance(r["columns"].get(cid), dict)]
        clip = [r["columns"].get(cid, {}).get("forward_clipped_mm", 0.0) for r in rows
                if isinstance(r["columns"].get(cid), dict)]
        surf = any(r["columns"].get(cid, {}).get("saturated_to_surface") for r in rows
                   if isinstance(r["columns"].get(cid), dict))
        efold = None
        if len(q) >= 2 and q[0] > 0 and q[1] > 0 and q[1] < q[0]:
            d0 = (t.get("day") or [31, 59])[:2]
            span = (d0[1] - d0[0]) if len(d0) == 2 else 28
            efold = round(span / math.log(q[0] / q[1]), 1)
        out[cid] = {
            "wt_start_m": start.get(cid), "wt_first_window_m": wt[0] if wt else None,
            "wt_end_m": wt[-1] if wt else None,
            "wt_span_m": round(max(wt) - min(wt), 3) if wt else None,
            "windows": len(wt), "reached_surface": surf,
            "forward_total_mm": round(sum(fwd), 1) if fwd else None,
            "forward_clipped_mm": round(sum(clip), 1) if clip else None,
            "outflow_total_mm": round(sum(q), 1) if q else None,
            "outflow_first_window_mm": round(q[0], 1) if q else None,
            "outflow_last_window_mm": round(q[-1], 2) if q else None,
            "outflow_efold_days": efold, "tau_dial_days": tau,
            "left_the_walk": (s.get("columns_left_the_walk") or {}).get(cid),
        }
    return out


def elm_monthly(s, var):
    out = {}
    for cid, d in (s.get("elm_daily") or {}).items():
        by = {}
        for dt, v in zip(d.get("dates") or [], d.get(var) or []):
            if v is not None:
                by.setdefault(dt[5:7], []).append(v)
        out[cid] = {m: round(mean(v), 3) for m, v in sorted(by.items())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("walk_dir")
    ap.add_argument("--against", default=None, help="another walk dir to show beside it")
    ap.add_argument("--json", default=None, help="write the table as JSON here")
    a = ap.parse_args()
    w, s, rows = load(a.walk_dir)
    table = per_column(w, s, rows)
    other = per_column(*load(a.against)) if a.against else {}
    sink = w.get("sink") or {}
    print(f"{Path(a.walk_dir).name}: year {w.get('year')}, forward {s.get('forward_var')} "
          f"({s.get('negative_forward')}), bottom {s.get('pf_bottom')}, sink {sink.get('datum')} "
          f"tau {sink.get('tau_days')} d, {len(table)} columns, "
          f"{len(s.get('columns_left_the_walk') or {})} left early")
    hdr = f"{'col':7s}{'wt start':>9s}{'wt end':>8s}{'span':>7s}{'in mm':>8s}{'clip mm':>8s}{'out mm':>9s}{'efold d':>8s} surf"
    if other:
        hdr += f"   | {Path(a.against).name}: wt end  span"
    bal = {cid: ((d.get("balance") or {}).get("year") or {})
           for cid, d in (s.get("elm_daily") or {}).items() if isinstance(d, dict)}
    if any(bal.values()):
        hdr += "   | ELM year: P  residual  stamps"
    print(hdr)
    for cid, r in sorted(table.items()):
        line = (f"{cid:7s}{(r['wt_start_m'] or 0):9.2f}{(r['wt_end_m'] or 0):8.2f}"
                f"{(r['wt_span_m'] or 0):7.2f}{(r['forward_total_mm'] or 0):8.1f}"
                f"{(r['forward_clipped_mm'] or 0):8.1f}{(r['outflow_total_mm'] or 0):9.1f}"
                f"{(r['outflow_efold_days'] or 0):8.1f} {'YES' if r['reached_surface'] else 'no '}")
        if r["left_the_walk"]:
            line += f"  LEFT w{r['left_the_walk'].get('window')}: {str(r['left_the_walk'].get('reason'))[:40]}"
        if other and cid in other:
            o = other[cid]
            line += f"   | {(o['wt_end_m'] or 0):8.2f} {(o['wt_span_m'] or 0):5.2f}"
        if bal.get(cid):
            b = bal[cid]
            line += (f"   | {b.get('P_mm', 0):6.0f} {b.get('residual_mm', 0):+8.0f}"
                     f" {b.get('stamp_jumps_mm', 0):+7.0f}")
        print(line)
    if a.json:
        Path(a.json).write_text(json.dumps({"columns": table, "against": other,
                                            "elm_monthly_QCHARGE": elm_monthly(s, "QCHARGE"),
                                            "elm_monthly_QDRAI": elm_monthly(s, "QDRAI"),
                                            "elm_monthly_ZWT": elm_monthly(s, "ZWT")}, indent=1))
        print(f"written {a.json}")


if __name__ == "__main__":
    main()
