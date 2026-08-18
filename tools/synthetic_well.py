#!/usr/bin/env python3
"""
A synthetic well beside one column — to exercise the water-table comparison
tools/synthetic_well.py

    python tools/synthetic_well.py RUN_DIR [--column col_03] [--bias 0.8]
                                   [--amp 0.6] [--noise 0.05] [--seed 0]
                                   [--out DIR] [--compare]

WHAT IT DOES. Reads a packaged run (experiment.json for the columns,
reception.json for the observations), invents ONE well at the chosen column's
coordinates with a daily depth-to-water series through the run's forcing year,
and writes a COPY of the run's three Analyzer files — reception.json with the
well added under observations.water_table.wells, strategy.json, experiment.json
— into a new directory. The run itself is never touched. With --compare it
then runs the Analyzer's step 0 and step 1 on the copy and prints the
water-table summary and where the figure went.

WHY A COPY, AND WHY THE LABELS. Naches has no recorder well in any year, so a
PFLOTRAN study there has nothing to stand a column beside and step 1 cannot be
seen working. A synthetic series lets the machinery run end to end. It is not
a measurement, and three things make sure nothing downstream forgets that: the
well's `source` says synthetic and `synthetic: true` is on the record (both
pass through compare_common into every pair and every label); its `quality` is
"synthetic", so frac_measured is 0; and the copy's provenance carries an entry
saying a well was invented, by what, and why. Nothing computed against it says
anything about the basin.

THE SERIES. The column's own water table (CONUS2, `water_table_m`) plus a
constant offset, a one-year cosine that is shallowest around day 120 (late
April — after melt in a snow basin), and small Gaussian noise; clipped at
0.05 m below ground. Every knob is an argument, so a reader of the copy's
provenance can see exactly what was invented.
"""
import argparse
import datetime as dt
import json
import math
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DAYS = 365


def _pick_column(rows, want):
    if want:
        for r in rows:
            if r.get("case_name") == want or r.get("id") == want:
                return r
        raise SystemExit(f"no column named {want!r} in experiment.json")
    # the TYPICAL transient column: the one whose unsaturated depth is the
    # median of the transient columns' — not the shallowest, which is usually
    # saturated, and not the deepest, which sits hundreds of metres down
    tr = [r for r in rows if r.get("transient")
          and isinstance(r.get("water_table_m"), (int, float))
          and isinstance(r.get("unsaturated_m"), (int, float))]
    if tr:
        tr.sort(key=lambda r: r["unsaturated_m"])
        return tr[len(tr) // 2]
    for r in rows:
        if isinstance(r.get("water_table_m"), (int, float)):
            return r
    raise SystemExit("no column carries water_table_m — nothing to invent a well around")


def make_well(col, year, bias, amp, noise, seed, doy_min=120):
    rng = random.Random(seed)
    jan1 = dt.date(int(year), 1, 1)
    base = float(col["water_table_m"]) + bias
    dates, values = [], []
    for d in range(DAYS):
        v = base - amp * math.cos(2 * math.pi * (d - doy_min) / DAYS) + rng.gauss(0, noise)
        dates.append((jan1 + dt.timedelta(days=d)).isoformat())
        values.append(round(max(v, 0.05), 3))
    cid = col.get("case_name") or col.get("id")
    return {
        "id": f"SYN-{cid}",
        "name": f"SYNTHETIC well at {cid} — not a measurement",
        "source": "synthetic — tools/synthetic_well.py",
        "synthetic": True,
        "quality": "synthetic",
        "lat": col.get("lat"), "lon": col.get("lon"),
        "elevation_m": col.get("elevation_m"),
        "in_basin": True,
        "observation_kind": "series", "parameter_kind": "depth",
        "daily": {"dates": dates, "values": values},
        "n_obs": len(values), "n_days": len(values),
        "wtd_m": round(sum(values) / len(values), 3),
        "min_depth_m": round(min(values), 3), "max_depth_m": round(max(values), 3),
        "recipe": {"given_water_table_m": col["water_table_m"], "bias_m": bias,
                   "amplitude_m": amp, "noise_sd_m": noise, "seed": seed,
                   "shallowest_day_of_year": doy_min, "year": int(year)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("run_dir")
    ap.add_argument("--column", default=None, help="case_name to put the well at "
                    "(default: the first transient column with a water table)")
    ap.add_argument("--bias", type=float, default=0.8, help="offset from the column's water table, m (+ deeper)")
    ap.add_argument("--amp", type=float, default=0.6, help="seasonal amplitude, m")
    ap.add_argument("--noise", type=float, default=0.05, help="Gaussian noise, m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="the copy's directory (default: RUN_DIR_synthetic)")
    ap.add_argument("--compare", action="store_true", help="run Analyzer steps 0 and 1 on the copy")
    a = ap.parse_args()

    run = Path(a.run_dir).resolve()
    exp = json.loads((run / "experiment.json").read_text())
    rows = exp.get("columns") or []
    col = _pick_column(rows, a.column)
    year = col.get("forcing_start") or (exp.get("period") or {}).get("yr_start")
    if not year:
        raise SystemExit("the column has no forcing_start and the run no period — which year?")
    well = make_well(col, year, a.bias, a.amp, a.noise, a.seed)

    out = Path(a.out).resolve() if a.out else run.parent / (run.name + "_synthetic")
    out.mkdir(parents=True, exist_ok=True)
    (out / "04_analysis").mkdir(exist_ok=True)
    for f in ("strategy.json", "experiment.json"):
        shutil.copy(run / f, out / f)
    rec = json.loads((run / "reception.json").read_text())
    obs = rec.setdefault("observations", {})
    wt = obs.setdefault("water_table", {})
    wells = wt.setdefault("wells", [])
    wells.append(well)
    wt["ok"] = True
    wt["n_with_records"] = (wt.get("n_with_records") or 0) + 1
    wt["synthetic_wells"] = (wt.get("synthetic_wells") or []) + [well["id"]]
    wt["note"] = ((wt.get("note") or "") + " | ONE SYNTHETIC WELL ADDED by "
                  "tools/synthetic_well.py to exercise the comparison — see "
                  "synthetic_wells; it is not a measurement.").strip(" |")
    rec.setdefault("provenance", []).append({
        "tool": None, "args": {"synthetic_well": well["id"], **well["recipe"]},
        "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
        "ok": True, "error": None,
        "note": (f"SYNTHETIC: a well was invented at column "
                 f"{col.get('case_name')} by tools/synthetic_well.py so the "
                 f"water-table comparison could be exercised on a basin with no "
                 f"recorder well. Nothing computed against it is about the basin."),
    })
    (out / "reception.json").write_text(json.dumps(rec, indent=2, default=str))
    print(f"✓ {well['id']} at ({well['lat']}, {well['lon']}) — mean {well['wtd_m']} m, "
          f"{well['min_depth_m']}–{well['max_depth_m']} m over {year}; "
          f"column given {col['water_table_m']} m")
    print(f"✓ copy written: {out}  (reception.json + strategy.json + experiment.json)")

    if a.compare:
        from agents.analysis import step0_context, step1_compare
        ctx = step0_context.load(str(out))
        print(f"✓ step 0: {len(ctx.columns)} columns, {len(ctx.caveats)} caveats")
        record = step1_compare.compare_all(ctx, str(out / "04_analysis"), draw=True)
        if record.get("skipped"):
            print("step 1 skipped:", record["skipped"]); return
        s = (record.get("summary") or {}).get("water_table") or {}
        print("✓ step 1 water_table summary:")
        print(json.dumps(s, indent=1, default=str)[:3000])
        print("figures:", record.get("figures"))


if __name__ == "__main__":
    main()
