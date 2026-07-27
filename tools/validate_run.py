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

    source /qfs/people/tran289/IDEAS/env_compy.sh
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
# USGS reports drainage area in square miles and SNOTEL elevation in feet.
# Everything this module EMITS is metric — convert at the boundary, once.
MI2_TO_KM2 = 2.58999
FT_TO_M = 0.3048

# Days discarded before any streamflow metric. A warm-started column flushes its
# prescribed initial storage in the first days — on the 1995 Naches run that was
# a single 405 mm/day spike contributing ~400 of the model's 1011 mm annual
# total, which alone drove alpha to 6.1 and NSE to -36.9. Scoring it measures
# initialization, not hydrology.
WARMUP_DAYS = 30
NLDI_BASIN = ("https://api.water.usgs.gov/nldi/linked-data/nwissite/"
              "{site}/basin")


def _rings(geom):
    """GeoJSON Polygon/MultiPolygon -> list of exterior rings [(lon,lat), ...]."""
    if not geom:
        return []
    if geom.get("type") == "Polygon":
        return [geom["coordinates"][0]]
    if geom.get("type") == "MultiPolygon":
        return [poly[0] for poly in geom["coordinates"]]
    return []


def _in_polygon(lat, lon, rings):
    """Ray-casting point-in-polygon. Rings are (lon, lat) as GeoJSON stores them."""
    if lat is None or lon is None or not rings:
        return False
    inside = False
    for ring in rings:
        n = len(ring)
        for i in range(n):
            x1, y1 = ring[i][0], ring[i][1]
            x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
            if (y1 > lat) != (y2 > lat):
                xin = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
                if lon < xin:
                    inside = not inside
    return inside


def fetch_gauge_basin(gauge_id):
    """The gauge's contributing-area polygon from the USGS NLDI.

    Without it, a basin-wide ensemble is scored against whatever sub-catchment
    the gauge happens to drain — the dominant error in the previous comparison.
    """
    import httpx
    site = gauge_id if gauge_id.startswith("USGS-") else f"USGS-{gauge_id}"
    try:
        r = httpx.get(NLDI_BASIN.format(site=site), timeout=45,
                      follow_redirects=True)
        if r.status_code != 200:
            return None
        feats = (r.json() or {}).get("features") or []
        rings = _rings(feats[0].get("geometry")) if feats else []
        return rings or None
    except Exception:
        return None


def band_weights(columns, bands):
    """{col_id: area weight}, summing to 1.

    Bands carry `grid_points` — the DEM sample count in that elevation band.
    On a regular sampling grid that is proportional to band AREA, so a band's
    share of the basin is grid_points / total, split evenly among the columns
    allocated to it.

    Why it matters: the sampler allocates N proportional to area but with a
    floor of >=1 per band, so a small high band gets the same single column as
    a large valley band. An unweighted mean then over-represents the small
    band. Falls back to equal weights when the band metadata is absent.
    """
    from collections import defaultdict
    per_band = defaultdict(list)
    for c in columns:
        per_band[c.get("band")].append(c["id"])

    gp = {b.get("band"): _n(b.get("grid_points")) or 0.0 for b in (bands or [])}
    total = sum(gp.get(b, 0.0) for b in per_band)
    if not total:
        n = len(columns) or 1
        return {c["id"]: 1.0 / n for c in columns}

    w = {}
    for b, ids in per_band.items():
        share = gp.get(b, 0.0) / total
        for cid in ids:
            w[cid] = share / len(ids)
    s = sum(w.values()) or 1.0
    return {k: v / s for k, v in w.items()}


def weighted_mean(values_by_id, weights):
    """Area-weighted mean over whatever ids are present in BOTH dicts."""
    ids = [i for i in values_by_id if weights.get(i) is not None
           and values_by_id[i] is not None]
    if not ids:
        return None
    wsum = sum(weights[i] for i in ids)
    if wsum <= 0:
        return None
    return sum(values_by_id[i] * weights[i] for i in ids) / wsum


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
                # Location and the individual measurements: needed to pair a
                # well with its nearest column and to plot it through time.
                # These are discrete field measurements, not a logger series.
                series = []
                for m in (w.get("measurements") or w.get("observations") or []):
                    dt = m.get("time") or m.get("date") or m.get("datetime")
                    dv = next((_n(m.get(k)) for k in
                               ("depth_to_water_m", "depth_m", "value")
                               if _n(m.get(k)) is not None), None)
                    if dt and dv is not None:
                        series.append({"date": str(dt)[:10], "wtd_m": round(dv, 3)})
                out["wells"].append({
                    "id": sid, "wtd_m": round(d, 2), "n_obs": summ.get("n_obs"),
                    "lat": _n(s.get("lat") or s.get("latitude")),
                    "lon": _n(s.get("lon") or s.get("longitude")),
                    "series": sorted(series, key=lambda r: r["date"])})
    except Exception as e:
        out["error"] = str(e)[:100]
    return out


def fetch_gauge_yields(clients, bb, year, max_gauges=4):
    """Observed specific discharge (mm/yr) for in-domain gauges: mean daily flow
    in the simulated year ÷ drainage area. The honest routing-free comparison.

    Which gauges HAVE records this year — and how large their catchments are —
    comes from `usgs_water.get_streamflow_availability`, the same MCP call
    Reception can make BEFORE a study is designed. That query used to be inlined
    here, which is why "the only gauge with 1995 data drains 7% of the basin"
    was a post-run discovery rather than a design input.
    """
    import httpx
    out = {"n_gauges_bbox": None, "gauges": []}
    try:
        cov = clients["usgs_water"].call_tool_json("get_streamflow_availability", {
            "bbox": _bbox_str(bb), "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31", "min_days": 300, "limit": 200}) or {}
        out["n_gauges_bbox"]        = cov.get("n_sites")
        out["n_gauges_with_records"] = cov.get("n_available")

        with httpx.Client(timeout=90, follow_redirects=True) as cx:
            for site in (cov.get("available") or [])[:max_gauges * 2]:
                sid, da_km2 = site["id"], site.get("drainage_area_km2")
                if not da_km2:
                    continue                               # no drainage area → no yield
                name  = (site.get("name") or sid)
                da_m2 = da_km2 * 1e6
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
                q_mm = sum(vals) / len(vals) * CFS_TO_M3YR / da_m2 * 1000
                out["gauges"].append({"id": sid, "name": name.title()[:38],
                                      "drainage_area_km2": da_km2,
                                      "n_days": len(vals),
                                      "specific_discharge_mm_yr": round(q_mm, 1)})
                if "daily_best" not in out:                # keep one daily series
                    to_mm_day = CFS_TO_M3S * 86400 / da_m2 * 1000
                    out["daily_best"] = {
                        "gauge": name.title()[:38], "id": sid,
                        "drainage_area_km2": da_km2,
                        "mm_day": {t[:10]: round(v * to_mm_day, 4) for t, v in pairs}}
                if len(out["gauges"]) >= max_gauges:
                    break
    except Exception as e:
        out["error"] = str(e)[:100]
    return out


def model_daily_runoff(run_dir: Path, cases_file: str, weights=None,
                       only_columns=None):
    """Daily total runoff (QOVER+QDRAI, mm/day) keyed by date.

    weights       {col_id: area weight} — AREA-weighted mean instead of a plain
                  one. The sampler allocates >=1 column per band regardless of
                  band size, so an unweighted mean over-weights small bands.
    only_columns  restrict to a set of column ids (the gauge's catchment).

    Also returns the per-column series, so a subset can be re-aggregated
    without re-reading 13 multi-file datasets.
    Returns None when the case dirs no longer exist (scratch purged).
    """
    import numpy as np
    import xarray as xr
    cf = run_dir / cases_file
    if not cf.exists():
        return None
    per_col = {}
    for cd in json.loads(cf.read_text()):
        name = cd.split(".")[-1]
        if only_columns is not None and name not in only_columns:
            continue
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
        per_col[name] = {d: float(np.mean(vs)) for d, vs in agg.items()}
    if not per_col:
        return None

    all_days = sorted({d for s in per_col.values() for d in s})
    w = {c: (weights or {}).get(c, 1.0) for c in per_col}
    tot = sum(w.values()) or 1.0
    mean = {}
    for d in all_days:
        num = sum(s[d] * w[c] for c, s in per_col.items() if d in s)
        den = sum(w[c] for c, s in per_col.items() if d in s) or tot
        mean[d] = round(num / den, 4)
    return {"n_columns": len(per_col), "columns": sorted(per_col),
            "weighted": bool(weights),
            "mm_day": mean, "per_column": per_col}


def model_daily_context(run_dir: Path, cases_file: str):
    """Column-mean daily precipitation, ET and SWE — {var: {date: value}}.

    Context only: precipitation is the FORCING (comparing it to a station
    validates NLDAS, not ELM) and there is no in-basin flux tower for ET. They
    are plotted so the water-balance terms can be read in time, never scored.

        P        = RAIN + SNOW                 mm/day
        ET       = QVEGE + QVEGT + QSOIL       mm/day  (canopy + transpiration + soil)
        runoff   = QOVER                       mm/day
        recharge = QCHARGE                     mm/day
        SWE      = H2OSNO                      mm      (state)
        ZWT      = water-table depth           m       (state)
    """
    import numpy as np
    import xarray as xr
    cf = run_dir / cases_file
    if not cf.exists():
        return None
    GROUPS = {"P": ("RAIN", "SNOW"), "ET": ("QVEGE", "QVEGT", "QSOIL"),
              "runoff": ("QOVER",), "recharge": ("QCHARGE",)}
    STATE = {"SWE": ("H2OSNO",), "ZWT": ("ZWT",)}
    acc = {k: {} for k in list(GROUPS) + list(STATE)}
    for cd in json.loads(cf.read_text()):
        fs = sorted(glob.glob(cd + "/run/*.elm.h0.*.nc"))
        if not fs:
            continue
        try:
            ds = xr.open_mfdataset(fs, combine="by_coords", decode_times=True,
                                   engine="netcdf4", data_vars="all",
                                   coords="different", compat="no_conflicts",
                                   join="outer")
        except Exception:
            continue
        days = ds["time"].dt.strftime("%Y-%m-%d").values
        for name, vs in list(GROUPS.items()) + list(STATE.items()):
            present = [v for v in vs if v in ds]
            if not present:
                continue
            tot = sum(ds[v] for v in present).squeeze()
            if name in GROUPS:
                tot = tot * 86400.0                      # mm/s -> mm/day
            arr = np.asarray(tot.values, dtype=float)
            per = {}
            for d, v in zip(days, arr):
                per.setdefault(d, []).append(v)
            for d, vv in per.items():
                acc[name].setdefault(d, []).append(float(np.mean(vv)))
        ds.close()
    out = {k: {d: round(float(sum(v) / len(v)), 4) for d, v in sorted(s.items())}
           for k, s in acc.items() if s}
    return out or None


def flow_metrics(obs: dict, mod: dict, warmup_days: int = WARMUP_DAYS):
    """NSE and KGE between two {date: mm/day} series on their common days.

    The first `warmup_days` are DISCARDED. A warm-started column dumps its
    prescribed initial storage in the opening days; including that measures
    initialization rather than hydrology, and on the Naches run a single day
    drove alpha to 6.1 and NSE to -36.9 on its own.
    """
    import numpy as np
    days_all = sorted(set(obs) & set(mod))
    days = days_all[warmup_days:] if warmup_days else days_all
    if len(days) < 100:
        return None
    o = np.array([obs[d] for d in days], float)
    m = np.array([mod[d] for d in days], float)
    nse = 1 - float(np.sum((m - o) ** 2)) / float(np.sum((o - o.mean()) ** 2))
    r = float(np.corrcoef(o, m)[0, 1])
    alpha = float(m.std() / o.std()) if o.std() > 0 else np.nan
    beta = float(m.mean() / o.mean()) if o.mean() > 0 else np.nan
    kge = 1 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return {"n_days": len(days), "warmup_days_excluded": warmup_days,
            "n_days_before_warmup_cut": len(days_all),
            "NSE": round(nse, 3), "KGE": round(kge, 3),
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
                elev_ft = _n(s.get("elevation_ft") or s.get("elevation"))
                out["stations"].append({
                    "name": (s.get("name") or trip)[:24], "triplet": trip,
                    "elevation_m": round(elev_ft * FT_TO_M, 1) if elev_ft else None,
                    "lat": _n(s.get("lat") or s.get("latitude")),
                    "lon": _n(s.get("lon") or s.get("longitude")),
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
            # science year = the LAST simulated year (spin-up years precede it)
            y = _n(cc.get("DATM_CLMNCEP_YR_END")) or _n(cc.get("DATM_CLMNCEP_YR_START"))
            if y:
                return int(y)
    return 1995


def build_validation(run_dir: Path, clients, cases_file="cases.json"):
    import numpy as np
    brief = json.loads((run_dir / "reception_brief.json").read_text())
    dom = brief.get("domain", {})
    bb = dom["bbox"]
    year = sim_year(run_dir)
    _cj  = json.loads((run_dir / "columns.json").read_text())
    cols = _cj["columns"] if isinstance(_cj, dict) else _cj
    _bands = _cj.get("bands") if isinstance(_cj, dict) else None
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

    # ── restrict the model to the GAUGE'S OWN CATCHMENT ────────────────────
    # The previous comparison scored a basin-wide ensemble against whatever
    # sub-catchment the gauge drains. Ask the USGS NLDI for the actual
    # contributing-area polygon and keep only the columns inside it.
    cols_by_id = {c["id"]: c for c in cols}
    weights_all = band_weights(cols, _bands)
    catchment = {"gauge_id": None, "rings": 0, "columns_inside": [],
                 "n_inside": 0, "restricted": False, "note": None}
    if daily_best:
        rings = fetch_gauge_basin(daily_best["id"])
        catchment["gauge_id"] = daily_best["id"]
        catchment["rings"] = len(rings or [])
        if rings:
            inside = [cid for cid, c in cols_by_id.items()
                      if _in_polygon(_n(c.get("lat")), _n(c.get("lon")), rings)]
            catchment["columns_inside"] = sorted(inside)
            catchment["n_inside"] = len(inside)
            catchment["restricted"] = len(inside) >= 3
            catchment["note"] = (
                f"{len(inside)} of {len(cols_by_id)} columns fall inside the "
                f"gauge's contributing area"
                + ("; the hydrograph is computed from those only."
                   if len(inside) >= 3 else
                   " — too few to represent the catchment, so the basin-wide "
                   "ensemble is used and the comparison stays CONTEXT-ONLY. "
                   "Sampling targeted at the gauged catchment would fix this."))
        else:
            catchment["note"] = "NLDI returned no basin polygon for this gauge"

    only = (set(catchment["columns_inside"]) if catchment["restricted"] else None)
    mod_daily = model_daily_runoff(run_dir, cases_file, weights=weights_all,
                                   only_columns=only)
    if daily_best and mod_daily:
        fm = flow_metrics(daily_best["mm_day"], mod_daily["mm_day"])
        if fm:
            hydro = {"gauge": daily_best["gauge"], "gauge_id": daily_best["id"],
                     "drainage_area_km2": daily_best["drainage_area_km2"],
                     "model_columns": mod_daily["n_columns"],
                     "area_weighted": mod_daily.get("weighted"),
                     "restricted_to_catchment": catchment["restricted"],
                     "obs_mm_day": daily_best["mm_day"],
                     "mod_mm_day": mod_daily["mm_day"], **fm}

    # ── DOMAIN MATCH ───────────────────────────────────────────────────────
    # The columns sample the whole basin; a gauge sees only its own catchment.
    # Naches is ~2861 km2 and American River near Nile drains 204 km2 — 7% of
    # it, and a wet headwater at that. Comparing a basin-wide ensemble mean to
    # that gauge is not a model error, it is a different question. Quantify the
    # overlap and refuse to SCORE absolute yield when it is poor.
    basin_km2 = _n(dom.get("area_km2"))
    gauge_km2 = _n((gauges["gauges"] or [{}])[0].get("drainage_area_km2"))
    frac = (gauge_km2 / basin_km2) if (basin_km2 and gauge_km2) else None
    comparable = bool(frac and frac >= 0.5)
    domain_match = {
        "basin_area_km2": basin_km2, "gauge_area_km2": gauge_km2,
        "gauge_fraction_of_basin": round(frac, 3) if frac else None,
        "absolute_yield_comparable": comparable,
        "note": (
            f"the gauge drains {frac:.1%} of the basin"
            if frac else "gauge or basin area unknown") + (
            " — absolute yield is NOT comparable; the gauged catchment is a "
            "sub-domain with its own precipitation regime. Treat yield as "
            "context and judge partitioning by the runoff RATIO instead."
            if not comparable else " — areas are close enough to compare yields."),
    }
    context = model_daily_context(run_dir, cases_file)

    # ── RUNOFF RATIO — the comparison that survives a forcing bias ─────────
    # Absolute yield conflates two errors: getting precipitation wrong and
    # partitioning it wrong. The ratio yield/P removes the first. The question
    # asked ("how does precipitation PARTITION") is a ratio question, so this
    # is the honest headline even when absolute yield is not comparable.
    #
    # Both ratios use the SAME modelled precipitation, because no independent
    # precipitation product is fetched for the gauged catchment. That makes the
    # comparison a partitioning test, not a forcing test — stated, not implied.
    ids_for_ratio = (catchment["columns_inside"] if catchment["restricted"]
                     else [r["case_name"] for r in ok])
    p_by_id = {r["case_name"]: _n(r["metrics"].get("precip_mm_yr")) for r in ok}
    y_by_id = {r["case_name"]: ((_n(r["metrics"].get("annual_recharge_mm_yr")) or 0)
                                + (_n(r["metrics"].get("annual_runoff_mm_yr")) or 0))
               for r in ok}
    sel_w = {i: weights_all.get(i, 0.0) for i in ids_for_ratio}
    p_mean = weighted_mean({i: p_by_id.get(i) for i in ids_for_ratio}, sel_w)
    y_mean = weighted_mean({i: y_by_id.get(i) for i in ids_for_ratio}, sel_w)
    obs_q_best = (gauges["gauges"][0]["specific_discharge_mm_yr"]
                  if gauges["gauges"] else None)
    runoff_ratio = {
        "columns_used": sorted(ids_for_ratio),
        "restricted_to_catchment": catchment["restricted"],
        "area_weighted": True,
        "precip_mm_yr": round(p_mean, 1) if p_mean else None,
        "model_yield_mm_yr": round(y_mean, 1) if y_mean else None,
        "model_ratio": (round(y_mean / p_mean, 3) if (p_mean and y_mean) else None),
        "observed_yield_mm_yr": obs_q_best,
        "observed_ratio": (round(obs_q_best / p_mean, 3)
                           if (p_mean and obs_q_best) else None),
        "note": "both ratios use the MODELLED precipitation; no independent "
                "catchment precipitation product is fetched, so this tests "
                "partitioning, not forcing.",
    }
    # A runoff ratio above 1 means more water left than fell — impossible for a
    # closed catchment over a year. When it appears on the OBSERVED side it is
    # not a model result at all: it means the precipitation used is not the
    # precipitation the gauged catchment received. Say so rather than letting a
    # nonsensical ratio be read as a partitioning verdict.
    _or = runoff_ratio.get("observed_ratio")
    if _or is not None and _or > 1.0:
        runoff_ratio["observed_ratio_valid"] = False
        runoff_ratio["note"] += (
            f" WARNING: the observed ratio is {_or:.2f} (>1) — the gauge yields "
            f"more than the {runoff_ratio['precip_mm_yr']:.0f} mm/yr of modelled "
            f"basin-mean precipitation. That is impossible for a closed "
            f"catchment, so the gauged headwater plainly receives far more "
            f"precipitation than the basin mean. The observed ratio is NOT "
            f"usable until precipitation over the gauged catchment is supplied.")
    else:
        runoff_ratio["observed_ratio_valid"] = True

    # ── pair point observations to their NEAREST column ────────────────────
    # A basin-wide distribution hides which site is being missed; pairing makes
    # each comparison local and answerable.
    def _near(lat, lon):
        cand = [(r, r.get("lat"), r.get("lon")) for r in ok
                if r.get("lat") is not None and r.get("lon") is not None]
        if lat is None or lon is None or not cand:
            return None
        return min(cand, key=lambda c: (c[1] - lat) ** 2 + (c[2] - lon) ** 2)[0]

    well_series = []
    for wl in wells["wells"]:
        n = _near(wl.get("lat"), wl.get("lon"))
        pts = wl.get("series") or []
        well_series.append({
            "id": wl["id"], "lat": wl.get("lat"), "lon": wl.get("lon"),
            "points": pts,
            "n_in_sim_year": sum(1 for q in pts
                                 if q["date"][:4] == str(year)),
            "nearest_column": n["case_name"] if n else None,
            "nearest_column_zwt_m": (_n(n["metrics"].get("water_table_depth_m"))
                                     if n else None)})

    swe_pairs = []
    for s in swe["stations"]:
        n = _near(s.get("lat"), s.get("lon"))
        swe_pairs.append({
            "station": s["name"], "elevation_m": s.get("elevation_m"),
            "obs_peak_swe_mm": s["peak_swe_mm"],
            "nearest_column": n["case_name"] if n else None,
            "model_peak_swe_mm": (_n(n["metrics"].get("peak_swe_mm")) if n else None),
            "column_elevation_m": (_n(n.get("elevation_m")) if n else None)})

    model_swe_by_elev = [
        {"column": r["case_name"], "elevation_m": _n(r.get("elevation_m")),
         "peak_swe_mm": _n(r["metrics"].get("peak_swe_mm"))}
        for r in ok
        if _n(r.get("elevation_m")) is not None
        and _n(r["metrics"].get("peak_swe_mm")) is not None]

    # column-mean water budget, so yield can be read as P - ET - dStorage
    wb_keys = ("precip_mm_yr", "et_mm_yr", "runoff_mm_yr", "drainage_mm_yr",
               "storage_change_mm", "closure_residual_mm_yr")
    wbs = [r["metrics"].get("water_budget") or {} for r in ok]
    water_budget_mean = {
        k: round(float(np.mean([b[k] for b in wbs if _n(b.get(k)) is not None])), 1)
        for k in wb_keys
        if any(_n(b.get(k)) is not None for b in wbs)}
    # precip lives on metrics, not inside water_budget — without it the budget
    # bars have no P to balance against.
    if "precip_mm_yr" not in water_budget_mean and precip:
        water_budget_mean["precip_mm_yr"] = round(float(np.mean(precip)), 1)

    # model SWE (present in runs since H2OSNO joined hist_fincl1)
    model_peak_swe = [_n(r["metrics"].get("peak_swe_mm")) for r in ok]
    model_peak_swe = [s for s in model_peak_swe if s is not None]

    _all_pts = [q for w in well_series for q in w["points"]]
    _yrs = sorted({q["date"][:4] for q in _all_pts})
    _yr_lo, _yr_hi = (_yrs[0], _yrs[-1]) if _yrs else (None, None)
    _n_in_year = sum(w["n_in_sim_year"] for w in well_series)

    targets = [
        {"variable": "water-table depth", "status": "compared",
         "obs": f"Fan 2013 at columns + {len(obs_wtd)} USGS wells with records",
         "result": (f"model ZWT median {med(model_zwt)} m vs Fan {med(fan_at_cols)} m "
                    f"vs observed wells {med(obs_wtd)} m (median; n={len(obs_wtd)})"),
         "note": ("distribution comparison (wells are not co-located with columns); the "
                  "column ZWT is a shallow/perched table — offsets from the deep regional "
                  "WTD are expected without groundwater coupling + spin-up. "
                  + (f"TEMPORAL MISMATCH: {_n_in_year} of "
                     f"{sum(len(w['points']) for w in well_series)} well measurements "
                     f"fall in {year} (records span {_yr_lo}-{_yr_hi}) — this is a "
                     f"CLIMATOLOGICAL comparison, not a year-matched one."
                     if _yr_lo else ""))},
        {"variable": "streamflow (water yield)",
         "status": ("compared" if (obs_q and comparable) else "context-only"),
         "obs": f"{len(obs_q)} in-domain gauges with {year} daily records ÷ drainage area",
         "result": (f"modeled yield {mean_yield} mm/yr vs observed specific discharge "
                    f"{', '.join(str(q) for q in obs_q)} mm/yr "
                    f"({', '.join(g['name'] for g in gauges['gauges'])})"
                    if obs_q else f"no in-domain gauge had {year} daily records"),
         "note": ((domain_match["note"] + " Model spread across columns is "
                   f"{min(yields):.0f}-{max(yields):.0f} mm/yr, so the ensemble "
                   "mean alone hides most of the signal.") if obs_q else None),
         "needs": (None if (obs_q and comparable)
                   else "gauge whose catchment matches the modelled domain, or "
                        "restrict columns to the gauged catchment (USGS NLDI)")},
        {"variable": "streamflow (daily hydrograph)",
         "status": "compared" if hydro else "context-only",
         "obs": (f"daily specific discharge at {hydro['gauge']} "
                 f"({hydro['drainage_area_km2']:.0f} km²), {hydro['n_days']} common days"
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
            "domain_match": domain_match,
            "catchment": catchment,
            "runoff_ratio": runoff_ratio,
            "band_weights": {k: round(v, 4) for k, v in weights_all.items()},
            "per_column_yield_mm_yr": [round(y, 1) for y in yields],
            "water_budget_mean": water_budget_mean,
            "well_series": well_series,
            "swe_pairs": swe_pairs,
            "model_swe_by_elevation": model_swe_by_elev,
            "context_series": context,
            "hydrograph": hydro,
            "swe_context": swe["stations"],
            "model_peak_swe_mm": model_peak_swe,
            "targets": targets}


# ── figures ──────────────────────────────────────────────────────────────────
# One figure per observable, each carrying its OWN verdict. A single blended
# panel let a weak comparison borrow credibility from a strong one, and hid
# which quantity actually disagreed.
_W = "#d95f0e"      # observed
_M = "#2c7fb8"      # model
_G = "#31a354"      # secondary observed


def _style(ax):
    ax.grid(alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)


def _dates(keys):
    from datetime import datetime
    return [datetime.strptime(k, "%Y-%m-%d") for k in keys]


def plot_hydrograph(val, out_path):
    """Daily shape + cumulative volume. The cumulative panel is the verdict:
    a 1-D column has no routing, so instantaneous timing is expected to be
    wrong, while total volume is a fair test."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    h = val.get("hydrograph")
    if not (h and h.get("obs_mm_day") and h.get("mod_mm_day")):
        return None
    days_all = sorted(set(h["obs_mm_day"]) & set(h["mod_mm_day"]))
    if not days_all:
        return None
    nwu = int(h.get("warmup_days_excluded") or 0)
    days = days_all[nwu:] if nwu else days_all
    o = np.array([h["obs_mm_day"][d] for d in days])
    m = np.array([h["mod_mm_day"][d] for d in days])
    x = _dates(days)
    x_wu = _dates(days_all[:nwu]) if nwu else []

    fig, ax = plt.subplots(2, 1, figsize=(11.5, 7), sharex=True)
    ax[0].plot(x, o, color=_W, lw=1.2, label=f"observed — {h['gauge']} "
               f"({h['drainage_area_km2']:.0f} km²)")
    ax[0].plot(x, m, color=_M, lw=1.2, label=f"model — column-mean QOVER+QDRAI "
               f"({h['model_columns']} cols)")
    if x_wu:
        for a in ax:
            a.axvspan(x_wu[0], x_wu[-1], color="#bbb", alpha=.35, zorder=0)
        ax[0].text(x_wu[0], ax[0].get_ylim()[1] * .92,
                   f" {nwu}-day warm-up\n discarded", fontsize=8, color="#555",
                   va="top")
    ax[0].set_ylabel("specific discharge (mm/day)")
    ax[0].legend(frameon=False, fontsize=8.5)
    scope = ("gauge catchment only" if h.get("restricted_to_catchment")
             else "basin-wide ensemble")
    wgt = "area-weighted" if h.get("area_weighted") else "unweighted"
    ax[0].set_title(f"Daily hydrograph — shape only; a 1-D column has no routing, "
                    f"so timing disagreement is expected\n"
                    f"model = {wgt} mean of {h['model_columns']} columns ({scope})",
                    fontweight="bold", fontsize=10.5)

    ax[1].plot(x, np.cumsum(o), color=_W, lw=1.8, label="observed")
    ax[1].plot(x, np.cumsum(m), color=_M, lw=1.8, label="model")
    ax[1].set_ylabel("cumulative depth (mm)")
    ax[1].legend(frameon=False, fontsize=8.5)
    gap = (np.sum(m) - np.sum(o)) / np.sum(o) * 100 if np.sum(o) else float("nan")
    ax[1].set_title(f"Cumulative volume — THE verdict: model {np.sum(m):.0f} mm vs "
                    f"observed {np.sum(o):.0f} mm ({gap:+.0f}%)",
                    fontweight="bold", fontsize=11)
    for a in ax:
        _style(a)
    dm = val.get("domain_match") or {}
    fig.suptitle("Streamflow — " + (dm.get("note") or ""), fontsize=9.5,
                 y=.99, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_yield(val, out_path):
    """Per-column yield spread against the gauge, plus the water-balance terms.
    The single 'model' bar it replaces averaged 13 very different columns."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    ys = val.get("per_column_yield_mm_yr") or []
    gauges = (val.get("streamflow_comparison") or {}).get("gauges") or []
    if not ys:
        return None
    dm = val.get("domain_match") or {}
    rr = val.get("runoff_ratio") or {}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6),
                           gridspec_kw={"width_ratios": [1, .8, 1.25]})

    x = np.random.default_rng(0).normal(0, .05, len(ys))
    ax[0].scatter(x, ys, s=48, color=_M, edgecolor="#222", zorder=3,
                  label=f"model columns (n={len(ys)})")
    ax[0].hlines(np.median(ys), -.28, .28, color=_M, lw=2.5)
    for g in gauges:
        ax[0].axhline(g["specific_discharge_mm_yr"], color=_W, lw=2, ls="--")
        ax[0].text(.32, g["specific_discharge_mm_yr"],
                   f"  {g['name']}\n  {g['drainage_area_km2']:.0f} km²",
                   va="center", fontsize=8, color=_W)
    ax[0].set_xlim(-.5, 1.1); ax[0].set_xticks([])
    ax[0].set_ylabel("water yield (mm/yr)")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    ok = dm.get("absolute_yield_comparable")
    ax[0].set_title("Water yield — model spread vs gauge\n"
                    + ("comparable domains" if ok else "DIFFERENT DOMAINS — context only"),
                    fontweight="bold", fontsize=10.5,
                    color=("#222" if ok else _W))

    # runoff ratio — the partitioning test, immune to precipitation bias
    if rr.get("model_ratio") is not None:
        bars = [("model", rr["model_ratio"], _M)]
        if rr.get("observed_ratio") is not None:
            bars.append(("gauge", rr["observed_ratio"], _W))
        bad = rr.get("observed_ratio_valid") is False
        ax[1].bar([b[0] for b in bars], [b[1] for b in bars],
                  color=[b[2] for b in bars], edgecolor="#222", width=.55,
                  hatch=["", "//"][:len(bars)] if bad else None)
        if bad:
            # Anchor to the impossibility line itself, on the left where the
            # short model bar leaves room — the previous placement rode on top
            # of the observed bar and collided with the title.
            ax[1].axhline(1.0, color="#222", ls=":", lw=1.2)
            # Headroom ABOVE the tallest bar, so the note cannot be overrun by
            # it. The previous placement sat at the impossibility line and the
            # observed bar grew straight through it (caught by the analyzer's
            # own review of the rendered figure).
            top = max(b[1] for b in bars) * 1.42
            ax[1].set_ylim(0, top)
            ax[1].text(0.5, top * .985,
                       "ratio > 1 is impossible — the gauged catchment\n"
                       "receives more P than the basin mean",
                       fontsize=7.5, color=_W, va="top", ha="center",
                       transform=ax[1].get_xaxis_transform(which="grid")
                       if False else ax[1].transData)
        for i, b in enumerate(bars):
            ax[1].text(i, b[1], f"{b[1]:.2f}", ha="center", va="bottom",
                       fontsize=9.5, fontweight="bold")
        ax[1].set_ylabel("runoff ratio (yield / P)")
        n_used = len(rr.get("columns_used") or [])
        ax[1].set_title(f"Runoff ratio — THE partitioning test\n"
                        f"immune to precipitation bias ({n_used} cols, "
                        f"{'catchment' if rr.get('restricted_to_catchment') else 'basin'})",
                        fontweight="bold", fontsize=10.5)
    else:
        ax[1].axis("off")
        ax[1].text(.5, .5, "runoff ratio unavailable", ha="center", va="center",
                   color="#888")

    terms = [("precip", "precip_mm_yr"), ("ET", "et_mm_yr"),
             ("runoff", "runoff_mm_yr"), ("drainage", "drainage_mm_yr"),
             ("Δstorage", "storage_change_mm")]
    wb = val.get("water_budget_mean") or {}
    vals = [wb.get(k) for _, k in terms]
    if any(v is not None for v in vals):
        cols = ["#4d4d4d", "#31a354", "#d95f0e", "#fdae6b", "#9ecae1"]
        ax[2].bar([t for t, _ in terms], [v or 0 for v in vals], color=cols,
                  edgecolor="#222")
        ax[2].axhline(0, color="#222", lw=.8)
        ax[2].set_ylabel("mm/yr")
        ax[2].set_title("Water balance (column mean) — yield is P − ET − Δstorage;\n"
                        "a large Δstorage means the year is not at equilibrium",
                        fontweight="bold", fontsize=10.5)
    else:
        ax[2].axis("off")
        ax[2].text(.5, .5, "water-budget terms unavailable", ha="center",
                   va="center", color="#888")
    for a in ax:
        _style(a)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_water_table(val, out_path):
    """The distribution comparison, plus each well's measurements through time
    against its nearest column. Wells are discrete field visits, not loggers —
    drawn as markers, never interpolated into a curve."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    w = val.get("wtd_comparison") or {}
    groups = [("model ZWT", w.get("model_zwt_m") or [], _M),
              ("Fan 2013\n(at columns)", w.get("fan_at_columns_m") or [], "#756bb1"),
              ("USGS wells\n(observed)", w.get("observed_wells_m") or [], _G)]
    if not any(g[1] for g in groups):
        return None
    series = [s for s in (val.get("well_series") or []) if s.get("points")]
    fig, ax = plt.subplots(1, 2 if series else 1,
                           figsize=(11.5 if series else 6, 4.8))
    axes = ax if series else [ax]

    rng = np.random.default_rng(0)
    for i, (lbl, v, c) in enumerate(groups):
        if not v:
            continue
        axes[0].scatter(i + rng.normal(0, .06, len(v)), v, s=46, color=c,
                        edgecolor="#222", zorder=3)
        axes[0].hlines(np.median(v), i - .28, i + .28, color=c, lw=2.5)
    axes[0].set_xticks(range(len(groups)))
    axes[0].set_xticklabels([g[0] for g in groups], fontsize=9)
    axes[0].set_yscale("log"); axes[0].invert_yaxis()
    axes[0].set_ylabel("water-table depth (m, log)")
    axes[0].set_title("Distribution — wells are NOT co-located with columns",
                      fontweight="bold", fontsize=10.5)

    if series:
        for s in series[:6]:
            d = _dates([p["date"] for p in s["points"]])
            axes[1].plot(d, [p["wtd_m"] for p in s["points"]], "o", ms=5,
                         color=_G, label="observed well" if s is series[0] else None)
            if s.get("nearest_column_zwt_m") is not None:
                axes[1].axhline(s["nearest_column_zwt_m"], color=_M, lw=1.2, ls="--",
                                label=("nearest column ZWT"
                                       if s is series[0] else None))
        axes[1].invert_yaxis()
        axes[1].set_ylabel("water-table depth (m)")
        axes[1].legend(frameon=False, fontsize=8)
        yrs = sorted({q["date"][:4] for s in series for q in s["points"]})
        n_in = sum(s.get("n_in_sim_year") or 0 for s in series)
        span = f"{yrs[0]}-{yrs[-1]}" if yrs else "?"
        axes[1].set_title(f"Wells through time vs nearest column\n"
                          f"{n_in} of {sum(len(s['points']) for s in series)} "
                          f"measurements in the simulated year (records span {span})",
                          fontweight="bold", fontsize=10,
                          color=("#222" if n_in else _W))
    for a in axes:
        _style(a)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_swe(val, out_path):
    """Each SNOTEL paired with its NEAREST column, and peak SWE against
    elevation — the axis where the physics lives. The range band this replaces
    said nothing about which station was being missed."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    pairs = val.get("swe_pairs") or []
    st = val.get("swe_context") or []
    if not (pairs or st):
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    if pairs:
        lbl = [p["station"][:14] for p in pairs]
        i = np.arange(len(pairs)); wdt = .38
        ax[0].bar(i - wdt/2, [p["obs_peak_swe_mm"] for p in pairs], wdt,
                  color=_W, edgecolor="#222", label="SNOTEL observed")
        ax[0].bar(i + wdt/2, [p.get("model_peak_swe_mm") or 0 for p in pairs], wdt,
                  color=_M, edgecolor="#222", label="nearest column")
        ax[0].set_xticks(i); ax[0].set_xticklabels(lbl, rotation=30, ha="right",
                                                   fontsize=8)
        ax[0].set_ylabel("peak SWE (mm)")
        ax[0].legend(frameon=False, fontsize=8.5)
        ax[0].set_title("Peak SWE — station vs its nearest column",
                        fontweight="bold", fontsize=10.5)
    else:
        ax[0].axis("off")

    oe = [(s.get("elevation_m"), s.get("peak_swe_mm")) for s in st]
    oe = [(e, v) for e, v in oe if e and v]
    me = val.get("model_swe_by_elevation") or []
    if oe:
        ax[1].scatter([e for e, _ in oe], [v for _, v in oe], s=54, color=_W,
                      edgecolor="#222", label="SNOTEL")
    if me:
        ax[1].scatter([r["elevation_m"] for r in me], [r["peak_swe_mm"] for r in me],
                      s=54, color=_M, edgecolor="#222", marker="s", label="model columns")
    ax[1].set_xlabel("elevation (m)"); ax[1].set_ylabel("peak SWE (mm)")
    ax[1].legend(frameon=False, fontsize=8.5)
    ax[1].set_title("Peak SWE vs elevation — is the lapse behaviour right?",
                    fontweight="bold", fontsize=10.5)
    for a in ax:
        _style(a)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_context(val, out_path):
    """The year in time — the seasonal story behind the annual numbers.

    NOT validation. Precipitation is model INPUT (comparing it to a station
    tests NLDAS, not ELM) and there is no in-basin flux tower for ET. Nothing
    here is scored; it exists so an annual total can be read as a sequence:
    snow accumulates, melts, and the melt pulse drives runoff and recharge.

    Fluxes are OVERLAID on one axis so their competition for the same water is
    visible; the slow states (snowpack, water table) sit beneath on their own
    scales.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    cx = val.get("context_series") or {}
    if not any(cx.get(k) for k in ("P", "ET", "runoff", "recharge", "SWE", "ZWT")):
        return None

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                           gridspec_kw={"height_ratios": [1.5, 1, 1]})

    # ── fluxes, overlaid ──
    for key, col, lbl, style in (
            ("P",        "#4d4d4d", "precipitation (forcing)", dict(lw=1.0, alpha=.55)),
            ("ET",       "#31a354", "evapotranspiration",      dict(lw=1.4)),
            ("runoff",   "#d95f0e", "runoff (QOVER)",          dict(lw=1.2)),
            ("recharge", "#2c7fb8", "recharge (QCHARGE)",      dict(lw=1.2))):
        s = cx.get(key)
        if not s:
            continue
        d = sorted(s)
        ax[0].plot(_dates(d), [s[k] for k in d], color=col, label=lbl, **style)
    ax[0].set_ylabel("mm / day")
    ax[0].legend(frameon=False, fontsize=8.5, ncol=4)
    ax[0].set_title("Water fluxes through the year — column mean. NOT scored: "
                    "precipitation is model INPUT and no in-basin flux tower "
                    "exists for ET.", fontweight="bold", fontsize=10.5)

    # ── snowpack ──
    s = cx.get("SWE")
    if s:
        d = sorted(s)
        v = [s[k] for k in d]
        ax[1].fill_between(_dates(d), v, color="#9ecae1", alpha=.8)
        ax[1].plot(_dates(d), v, color="#3182bd", lw=1.0)
        pk = int(np.argmax(v))
        ax[1].annotate(f"peak {v[pk]:.0f} mm", (_dates(d)[pk], v[pk]),
                       textcoords="offset points", xytext=(6, -10), fontsize=8,
                       color="#3182bd")
    ax[1].set_ylabel("SWE (mm)")
    ax[1].set_title("Snowpack — the melt pulse is what the fluxes above respond to",
                    fontweight="bold", fontsize=10)

    # ── water table ──
    s = cx.get("ZWT")
    if s:
        d = sorted(s)
        ax[2].plot(_dates(d), [s[k] for k in d], color="#756bb1", lw=1.3)
        ax[2].invert_yaxis()
    ax[2].set_ylabel("water table (m)")
    ax[2].set_title("Water-table depth — a flat line means the aquifer never "
                    "responded (no spin-up)", fontweight="bold", fontsize=10)

    for a in ax:
        _style(a)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_validation(val, out_path):
    """Write every validation figure beside `out_path`; returns the paths.

    Kept as the entry point ELMExpManager and the CLI already call, but it now
    emits one file per observable instead of a single blended panel.
    """
    out = Path(out_path)
    stem, sfx, d = out.stem, out.suffix or ".png", out.parent
    made = {}
    for name, fn in (("hydrograph", plot_hydrograph), ("yield", plot_yield),
                     ("water_table", plot_water_table), ("swe", plot_swe),
                     ("context", plot_context)):
        try:
            r = fn(val, d / f"{stem}_{name}{sfx}")
            if r:
                made[name] = r
        except Exception as e:
            print(f"   ⚠️  {name} figure failed: {e}")
    for k, v in made.items():
        print(f"   ✓ {k}: {Path(v).name}")
    return made


def main():
    ap = argparse.ArgumentParser(description="Validate a run against in-domain observations")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cases-file", default="cases.json",
                    help="case-dir list, for the daily hydrograph (default cases.json)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--replot", action="store_true",
                    help="re-render the figure from a stored validation.json (no MCP)")
    args = ap.parse_args()
    if args.replot:
        rd = Path(args.run_dir)
        vp = rd / "04_analysis" / "validation.json"
        val = json.loads(vp.read_text())
        plot_validation(val, rd / "04_analysis" / "validation.png")
        return

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
