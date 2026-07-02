#!/usr/bin/env python3
"""
Observation validation for a completed ELM run — fetch the IN-DOMAIN observations
via the MCP servers (+ the USGS OGC API for daily discharge) and confront them
with the model, honestly labelling the strength of each comparison:

  • water-table depth  — model ZWT vs Fan (2013) regional WTD at the columns AND
    observed USGS wells with records (distribution comparison, not co-located).
  • streamflow         — FIRST-ORDER water balance: modeled yield (runoff+recharge,
    mm/yr) vs observed specific discharge (gauge mean flow ÷ drainage area) for
    in-domain gauges with daily records in the simulated year.
  • snow (SWE)         — observed peak SWE per SNOTEL station for the water year;
    CONTEXT-ONLY (the run carries no SWE output to compare against).

Reads reception_brief.json (bbox), columns.json (Fan WTD per column), the run
plan (simulation year) and 04_analysis/hydro_summary.json (model state — run
analyze_run.py first). Writes 04_analysis/validation.json + validation.png.

    module load pytorch/2.8.0
    python3 tools/validate_run.py --run-dir <dir>
"""
import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

OGC_DAILY = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
CFS_TO_M3S = 0.0283168
CFS_TO_M3YR = CFS_TO_M3S * 86400 * 365.25
MI2_TO_M2 = 2.58999e6


def _n(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _bbox_str(bb):
    return f'{bb["min_lon"]},{bb["min_lat"]},{bb["max_lon"]},{bb["max_lat"]}'


# ── observation fetchers ─────────────────────────────────────────────────────
def fetch_wells(clients, bb, max_sample=10):
    """Observed water-table depths at USGS wells that actually have records."""
    out = {"n_sites": None, "sampled": 0, "wells": []}
    try:
        gw = clients["usgs_water"].call_tool_json(
            "get_groundwater_sites", {"bbox": _bbox_str(bb), "limit": 200}) or {}
        sites = gw.get("sites", [])
        out["n_sites"] = gw.get("n_sites", len(sites))
        for s in sites[:max_sample]:
            sid = s.get("id") or s.get("monitoring_location_id")
            if not sid:
                continue
            out["sampled"] += 1
            w = clients["usgs_water"].call_tool_json(
                "get_water_table_depth", {"monitoring_location_id": sid, "limit": 50}) or {}
            summ = w.get("summary") or {}
            if not summ.get("n_obs"):
                continue
            d = next((_n(summ.get(k)) for k in
                      ("mean_depth_m", "median_depth_m", "latest_depth_m")
                      if _n(summ.get(k)) is not None), None)
            if d is not None:
                out["wells"].append({"id": sid, "wtd_m": round(d, 2),
                                     "n_obs": summ.get("n_obs")})
    except Exception as e:
        out["error"] = str(e)[:100]
    return out


def fetch_gauge_yields(clients, bb, year, max_gauges=4):
    """Observed specific discharge (mm/yr) for in-domain gauges: mean daily flow
    in the simulated year ÷ drainage area. The honest routing-free comparison.
    Discovers which gauges HAVE data via one bbox query on the OGC daily
    collection, then matches their drainage areas from monitoring-locations."""
    import httpx
    out = {"n_gauges_bbox": None, "gauges": []}
    try:
        sg = clients["usgs_water"].call_tool_json("get_monitoring_locations", {
            "bbox": _bbox_str(bb), "site_type_code": "ST", "limit": 200}) or {}
        feats = sg.get("features") or []
        out["n_gauges_bbox"] = sg.get("numberReturned") or len(feats)
        meta = {f"USGS-{f['properties'].get('monitoring_location_number')}":
                (f["properties"].get("monitoring_location_name", ""),
                 _n(f["properties"].get("drainage_area"))) for f in feats}

        with httpx.Client(timeout=90, follow_redirects=True) as cx:
            # one spatial query tells us which gauges actually have records
            disc = cx.get(OGC_DAILY, params={
                "bbox": _bbox_str(bb), "parameter_code": "00060",
                "datetime": f"{year}-01-01/{year}-12-31",
                "limit": 1000, "f": "json"}).json()
            site_ids = sorted({f["properties"]["monitoring_location_id"]
                               for f in disc.get("features", [])})
            for sid in site_ids[:max_gauges * 2]:
                name, da_mi2 = meta.get(sid, (sid, None))
                if not da_mi2:
                    continue                               # no drainage area → no yield
                r = cx.get(OGC_DAILY, params={
                    "monitoring_location_id": sid, "parameter_code": "00060",
                    "datetime": f"{year}-01-01/{year}-12-31",
                    "limit": 400, "f": "json"})
                pairs = [(f["properties"].get("time"), _n(f["properties"].get("value")))
                         for f in r.json().get("features", [])]
                pairs = [(t, v) for t, v in pairs if v is not None and t]
                if len(pairs) < 300:                       # need (most of) the year
                    continue
                vals = [v for _, v in pairs]
                q_mm = sum(vals) / len(vals) * CFS_TO_M3YR / (da_mi2 * MI2_TO_M2) * 1000
                out["gauges"].append({"id": sid, "name": name.title()[:38],
                                      "drainage_mi2": da_mi2, "n_days": len(vals),
                                      "specific_discharge_mm_yr": round(q_mm, 1)})
                if "daily_best" not in out:                # keep one daily series
                    to_mm_day = CFS_TO_M3S * 86400 / (da_mi2 * MI2_TO_M2) * 1000
                    out["daily_best"] = {
                        "gauge": name.title()[:38], "id": sid, "drainage_mi2": da_mi2,
                        "mm_day": {t[:10]: round(v * to_mm_day, 4) for t, v in pairs}}
                if len(out["gauges"]) >= max_gauges:
                    break
    except Exception as e:
        out["error"] = str(e)[:100]
    return out


def model_daily_runoff(run_dir: Path, cases_file: str):
    """Column-mean daily total runoff (QOVER+QDRAI, mm/day) keyed by date string.
    Returns None when the case dirs no longer exist (scratch purged)."""
    import numpy as np
    import xarray as xr
    cf = run_dir / cases_file
    if not cf.exists():
        return None
    per_day = {}
    n_cols = 0
    for cd in json.loads(cf.read_text()):
        fs = sorted(glob.glob(cd + "/run/*.elm.h0.*.nc"))
        if not fs:
            continue
        ds = xr.open_mfdataset(fs, combine="by_coords", decode_times=True,
                               engine="netcdf4", data_vars="all",
                               coords="different", compat="no_conflicts", join="outer")
        q = ((ds["QOVER"] + ds["QDRAI"]) * 86400.0).squeeze()   # mm/day, 3-hourly
        days = ds["time"].dt.strftime("%Y-%m-%d").values
        vals = np.asarray(q.values, dtype=float)
        ds.close()
        agg = {}
        for d, v in zip(days, vals):
            agg.setdefault(d, []).append(v)
        n_cols += 1
        for d, vs in agg.items():
            per_day.setdefault(d, []).append(float(np.mean(vs)))
    if not per_day:
        return None
    return {"n_columns": n_cols,
            "mm_day": {d: round(float(sum(v) / len(v)), 4)
                       for d, v in sorted(per_day.items())}}


def flow_metrics(obs: dict, mod: dict):
    """NSE and KGE between two {date: mm/day} series on their common days."""
    import numpy as np
    days = sorted(set(obs) & set(mod))
    if len(days) < 100:
        return None
    o = np.array([obs[d] for d in days], float)
    m = np.array([mod[d] for d in days], float)
    nse = 1 - float(np.sum((m - o) ** 2)) / float(np.sum((o - o.mean()) ** 2))
    r = float(np.corrcoef(o, m)[0, 1])
    alpha = float(m.std() / o.std()) if o.std() > 0 else np.nan
    beta = float(m.mean() / o.mean()) if o.mean() > 0 else np.nan
    kge = 1 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return {"n_days": len(days), "NSE": round(nse, 3), "KGE": round(kge, 3),
            "r": round(r, 3), "alpha_var_ratio": round(alpha, 3),
            "beta_bias_ratio": round(beta, 3), "days": days,
            "obs": o.tolist(), "mod": m.tolist()}


def fetch_peak_swe(clients, bb, year):
    """Observed peak SWE per SNOTEL station for the water year (Oct–Sep)."""
    out = {"n_stations": None, "stations": []}
    try:
        sn = clients["snotel"].call_tool_json("get_snotel_stations", {
            "min_lon": bb["min_lon"], "min_lat": bb["min_lat"],
            "max_lon": bb["max_lon"], "max_lat": bb["max_lat"]}) or {}
        stations = sn.get("stations", [])
        out["n_stations"] = sn.get("n_stations", len(stations))
        for s in stations[:8]:
            trip = s.get("triplet") or s.get("station_triplet")
            if not trip:
                continue
            sw = clients["snotel"].call_tool_json("get_snotel_swe", {
                "station_triplet": trip,
                "start_date": f"{year - 1}-10-01",
                "end_date": f"{year}-09-30"}) or {}
            summ = sw.get("summary") or {}
            if _n(summ.get("peak_swe_mm")) is not None:
                out["stations"].append({
                    "name": (s.get("name") or trip)[:24], "triplet": trip,
                    "elevation_ft": s.get("elevation_ft") or s.get("elevation"),
                    "peak_swe_mm": round(_n(summ["peak_swe_mm"]), 1),
                    "peak_date": summ.get("peak_date")})
    except Exception as e:
        out["error"] = str(e)[:100]
    return out


# ── assemble the validation ──────────────────────────────────────────────────
def sim_year(rd: Path):
    for f in ("run_plan.json", "phase3_plan.json", "soilsweep_plan.json"):
        p = rd / f
        if p.exists():
            cc = (json.loads(p.read_text()).get("CONDITIONS_COUPLERS") or [{}])[0]
            y = _n(cc.get("DATM_CLMNCEP_YR_START"))
            if y:
                return int(y)
    return 1995


def build_validation(run_dir: Path, clients, cases_file="cases.json"):
    import numpy as np
    brief = json.loads((run_dir / "reception_brief.json").read_text())
    dom = brief.get("domain", {})
    bb = dom["bbox"]
    year = sim_year(run_dir)
    cols = json.loads((run_dir / "columns.json").read_text())
    cols = cols["columns"] if isinstance(cols, dict) else cols
    fan = {c["id"]: _n(c.get("fan_wtd_m")) for c in cols}
    hs = json.loads((run_dir / "04_analysis" / "hydro_summary.json").read_text())

    ok = [r for r in hs.get("experiments", []) if r.get("status") == "ok"]
    model_zwt = [_n(r["metrics"].get("water_table_depth_m")) for r in ok]
    model_zwt = [z for z in model_zwt if z is not None]
    fan_at_cols = [fan[r["case_name"]] for r in ok if fan.get(r["case_name"]) is not None]
    yields = [(_n(r["metrics"].get("annual_recharge_mm_yr")) or 0)
              + (_n(r["metrics"].get("annual_runoff_mm_yr")) or 0) for r in ok]
    precip = [_n(r["metrics"].get("precip_mm_yr")) for r in ok]
    precip = [p for p in precip if p is not None]

    print("  fetching observations (wells, gauge discharge, SNOTEL SWE)…")
    wells = fetch_wells(clients, bb)
    gauges = fetch_gauge_yields(clients, bb, year)
    swe = fetch_peak_swe(clients, bb, year)

    med = lambda v: round(float(np.median(v)), 2) if v else None
    obs_wtd = [w["wtd_m"] for w in wells["wells"]]
    mean_yield = round(float(np.mean(yields)), 1) if yields else None
    obs_q = [g["specific_discharge_mm_yr"] for g in gauges["gauges"]]
    peak_swes = [s["peak_swe_mm"] for s in swe["stations"]]

    # daily hydrograph comparison (model total runoff vs best gauge), NSE/KGE
    print("  building daily hydrograph comparison (NSE/KGE)…")
    hydro = None
    daily_best = gauges.get("daily_best")
    mod_daily = model_daily_runoff(run_dir, cases_file)
    if daily_best and mod_daily:
        fm = flow_metrics(daily_best["mm_day"], mod_daily["mm_day"])
        if fm:
            hydro = {"gauge": daily_best["gauge"], "gauge_id": daily_best["id"],
                     "drainage_mi2": daily_best["drainage_mi2"],
                     "model_columns": mod_daily["n_columns"], **fm}

    # model SWE (present in runs since H2OSNO joined hist_fincl1)
    model_peak_swe = [_n(r["metrics"].get("peak_swe_mm")) for r in ok]
    model_peak_swe = [s for s in model_peak_swe if s is not None]

    targets = [
        {"variable": "water-table depth", "status": "compared",
         "obs": f"Fan 2013 at columns + {len(obs_wtd)} USGS wells with records",
         "result": (f"model ZWT median {med(model_zwt)} m vs Fan {med(fan_at_cols)} m "
                    f"vs observed wells {med(obs_wtd)} m (median; n={len(obs_wtd)})"),
         "note": "distribution comparison (wells are not co-located with columns); the "
                 "column ZWT is a shallow/perched table — offsets from the deep regional "
                 "WTD are expected without groundwater coupling + spin-up."},
        {"variable": "streamflow (water yield)", "status": "compared" if obs_q else "context-only",
         "obs": f"{len(obs_q)} in-domain gauges with {year} daily records ÷ drainage area",
         "result": (f"modeled yield {mean_yield} mm/yr vs observed specific discharge "
                    f"{', '.join(str(q) for q in obs_q)} mm/yr "
                    f"({', '.join(g['name'] for g in gauges['gauges'])})"
                    if obs_q else f"no in-domain gauge had {year} daily records"),
         "note": "first-order water-balance check — unweighted column mean vs gauge "
                 "subcatchments (which may drain the wetter headwaters); routing/area-"
                 "weighting would sharpen this.",
         "needs": None if obs_q else "runoff routing / column aggregation"},
        {"variable": "streamflow (daily hydrograph)",
         "status": "compared" if hydro else "context-only",
         "obs": (f"daily specific discharge at {hydro['gauge']} "
                 f"({hydro['drainage_mi2']:.0f} mi²), {hydro['n_days']} common days"
                 if hydro else "needs surviving history files + a gauge with daily records"),
         "result": (f"NSE={hydro['NSE']}, KGE={hydro['KGE']} (r={hydro['r']}, "
                    f"variability α={hydro['alpha_var_ratio']}, bias β={hydro['beta_bias_ratio']}) — "
                    f"column-mean QOVER+QDRAI vs gauge"
                    if hydro else "daily comparison unavailable"),
         "note": "unrouted, unweighted columns vs an integrated gauge — timing errors are "
                 "expected; poor NSE/KGE quantifies exactly what routing + snow + spin-up "
                 "would need to fix." if hydro else None},
        {"variable": "snow water equivalent",
         "status": "compared" if (model_peak_swe and peak_swes) else "context-only",
         "obs": f"{len(peak_swes)} SNOTEL stations, water year {year}",
         "result": ((f"model peak SWE {min(model_peak_swe):.0f}–{max(model_peak_swe):.0f} mm "
                     f"across columns vs observed {min(peak_swes):.0f}–{max(peak_swes):.0f} mm "
                     "across stations (coarse uniform forcing — expect underestimation)")
                    if (model_peak_swe and peak_swes) else
                    (f"observed peak SWE {min(peak_swes):.0f}–{max(peak_swes):.0f} mm "
                     f"across stations vs column forcing precip "
                     f"{', '.join(str(round(p)) for p in sorted(set(round(p) for p in precip)))} mm/yr — "
                     "this run predates SWE (H2OSNO) in the history output"
                     if peak_swes else "no SNOTEL SWE retrieved")),
         "needs": None if (model_peak_swe and peak_swes)
                  else "snow/SWE history output (default since 2026-07) + elevation-downscaled forcing"},
    ]

    return {"domain": {"name": dom.get("name"), "huc": dom.get("huc"),
                       "area_km2": dom.get("area_km2")},
            "sim_year": year,
            "observation_inventory": {
                "snotel": {"n": swe["n_stations"], "with_swe": len(peak_swes)},
                "streamgages": {"n": gauges["n_gauges_bbox"], "with_year_records": len(obs_q)},
                "gwwells": {"n": wells["n_sites"], "sampled": wells["sampled"],
                            "with_records": len(obs_wtd)}},
            "wtd_comparison": {"model_zwt_m": model_zwt, "fan_at_columns_m": fan_at_cols,
                               "observed_wells_m": obs_wtd},
            "streamflow_comparison": {"modeled_yield_mm_yr": mean_yield,
                                      "gauges": gauges["gauges"]},
            "hydrograph": hydro,
            "swe_context": swe["stations"],
            "model_peak_swe_mm": model_peak_swe,
            "targets": targets}


# ── figure ───────────────────────────────────────────────────────────────────
def plot_validation(val, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    import numpy as np

    hydro = val.get("hydrograph")
    if hydro:
        fig = plt.figure(figsize=(13.2, 8.6))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=.42, wspace=.3)
        axh = fig.add_subplot(gs[0, :])
        ax = [fig.add_subplot(gs[1, i]) for i in range(3)]
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in hydro["days"]]
        axh.plot(dates, hydro["obs"], color="#d95f0e", lw=1.4,
                 label=f"observed — {hydro['gauge']} ({hydro['drainage_mi2']:.0f} mi²)")
        axh.plot(dates, hydro["mod"], color="#2c7fb8", lw=1.4,
                 label=f"model — column-mean QOVER+QDRAI ({hydro['model_columns']} cols)")
        axh.set_ylabel("specific discharge (mm/day)")
        axh.set_title(f"Daily hydrograph — NSE={hydro['NSE']}  KGE={hydro['KGE']}  "
                      f"(r={hydro['r']}, α={hydro['alpha_var_ratio']}, β={hydro['beta_bias_ratio']})",
                      fontweight="bold")
        axh.legend(frameon=False, fontsize=9)
        axh.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axh.spines[["top", "right"]].set_visible(False); axh.grid(alpha=.25)
    else:
        fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.3))

    # 1 · WTD distributions: model vs Fan vs observed wells
    wtd = val["wtd_comparison"]
    groups = [("model ZWT", wtd["model_zwt_m"], "#2c7fb8"),
              ("Fan 2013\n(at columns)", wtd["fan_at_columns_m"], "#8856a7"),
              ("USGS wells\n(observed)", wtd["observed_wells_m"], "#31a354")]
    rng = np.random.default_rng(0)
    for i, (lab, vals, c) in enumerate(groups):
        v = np.array([x for x in vals if x is not None and x > 0], float)
        if not len(v):
            continue
        x = np.full(len(v), i) + rng.uniform(-.12, .12, len(v))
        ax[0].scatter(x, v, s=42, color=c, edgecolor="#222", zorder=3, alpha=.85)
        ax[0].hlines(np.median(v), i - .25, i + .25, color=c, lw=2.6, zorder=4)
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels([g[0] for g in groups], fontsize=9)
    ax[0].set_yscale("log"); ax[0].invert_yaxis()
    ax[0].set_ylabel("water-table depth (m, log)")
    ax[0].set_title("Water table — model vs Fan vs wells", fontweight="bold")

    # 2 · water yield: modeled vs gauge specific discharge
    sf = val["streamflow_comparison"]
    labels = ["model\n(yield)"] + [g["name"][:16] + f"\n({g['drainage_mi2']:.0f} mi²)"
                                   for g in sf["gauges"]]
    vals = [sf["modeled_yield_mm_yr"]] + [g["specific_discharge_mm_yr"] for g in sf["gauges"]]
    colors = ["#2c7fb8"] + ["#d95f0e"] * len(sf["gauges"])
    ax[1].bar(range(len(vals)), [v or 0 for v in vals], color=colors, edgecolor="#222")
    for i, v in enumerate(vals):
        if v is not None:
            ax[1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("mm / yr")
    ax[1].set_title(f"Water yield vs gauges ({val['sim_year']})", fontweight="bold")

    # 3 · peak SWE — observed stations (+ model range when the run has H2OSNO)
    swe = val["swe_context"]
    mswe = val.get("model_peak_swe_mm") or []
    if swe:
        names = [s["name"][:14] for s in swe]
        peaks = [s["peak_swe_mm"] for s in swe]
        ax[2].bar(range(len(peaks)), peaks, color="#756bb1", edgecolor="#222",
                  label="SNOTEL observed")
        ax[2].set_xticks(range(len(names)))
        ax[2].set_xticklabels(names, fontsize=8, rotation=30, ha="right")
        ax[2].set_ylabel("peak SWE (mm)")
    if mswe:
        ax[2].axhspan(min(mswe), max(mswe) + 1, color="#2c7fb8", alpha=.3,
                      label="model columns (range)")
        ax[2].legend(frameon=False, fontsize=8)
        ax[2].set_title(f"Peak SWE, WY{val['sim_year']} — model vs SNOTEL",
                        fontweight="bold", fontsize=11)
    else:
        ax[2].set_title(f"Observed peak SWE, WY{val['sim_year']}\n"
                        "(context — run predates SWE output)", fontweight="bold", fontsize=11)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.25)
    fig.suptitle(f"Observation validation — {val['domain'].get('name')} "
                 f"(HUC {val['domain'].get('huc')})", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Validate a run against in-domain observations")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cases-file", default="cases.json",
                    help="case-dir list, for the daily hydrograph (default cases.json)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "04_analysis" / "hydro_summary.json").exists():
        sys.exit("run analyze_run.py first (no 04_analysis/hydro_summary.json)")
    if not (run_dir / "reception_brief.json").exists():
        sys.exit("no reception_brief.json — controlled runs have no domain to validate against")

    from core.mcp_manager import MCPManager
    clients = MCPManager("mcp_config.json").get_all_clients()
    val = build_validation(run_dir, clients, cases_file=args.cases_file)

    print("\n" + "=" * 80)
    print(f"OBSERVATION VALIDATION — {val['domain'].get('name')} "
          f"(HUC {val['domain'].get('huc')}, sim year {val['sim_year']})")
    print("=" * 80)
    inv = val["observation_inventory"]
    print(f"in-domain obs: SNOTEL {inv['snotel']['n']} ({inv['snotel']['with_swe']} w/ SWE) · "
          f"gauges {inv['streamgages']['n']} ({inv['streamgages']['with_year_records']} w/ records) · "
          f"wells {inv['gwwells']['n']} ({inv['gwwells']['with_records']}/{inv['gwwells']['sampled']} sampled w/ records)")
    print("-" * 80)
    for t in val["targets"]:
        tag = {"compared": "✓ COMPARED", "context-only": "· context"}.get(t["status"], t["status"])
        print(f"{tag:<12} {t['variable']}  [{t['obs']}]")
        print(f"             {t['result']}")
        if t.get("note"):
            print(f"             note: {t['note']}")
        if t.get("needs"):
            print(f"             needs: {t['needs']}")
    print("=" * 80)

    out = run_dir / "04_analysis"
    (out / "validation.json").write_text(json.dumps(val, indent=2, default=str))
    if not args.no_plot:
        plot_validation(val, out / "validation.png")
    print(f"\nwritten to {out}/validation.json")


if __name__ == "__main__":
    main()
