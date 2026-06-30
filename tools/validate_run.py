#!/usr/bin/env python3
"""
Observation validation for a completed ELM run — fetch the IN-DOMAIN observations
via the MCP servers and confront them with the model, HONESTLY:

  • water-table depth — a REAL comparison: per-column model ZWT vs Fan (2013)
    regional WTD (and observed USGS wells where records exist).
  • streamflow / SWE — labelled CONTEXT, not validation: the observations are
    inventoried, but the single-column stack has no routed streamflow and no
    snow output to compare against (see each target's `status` + `needs`).

Reads <run-dir>/reception_brief.json (bbox), columns.json (Fan WTD per column)
and 04_analysis/hydro_summary.json (model state — run analyze_run.py first).
Writes 04_analysis/validation.json + validation.png. Hits live MCP sources.

    module load pytorch/2.8.0
    python3 tools/validate_run.py --run-dir <dir>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")


def _n(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fetch_inventory(clients, bb):
    """Defensive inventory of in-domain observations."""
    bs = f'{bb["min_lon"]},{bb["min_lat"]},{bb["max_lon"]},{bb["max_lat"]}'
    inv = {}
    try:
        sn = clients["snotel"].call_tool_json("get_snotel_stations", {
            "min_lon": bb["min_lon"], "min_lat": bb["min_lat"],
            "max_lon": bb["max_lon"], "max_lat": bb["max_lat"]}) or {}
        inv["snotel"] = {"n": sn.get("n_stations", len(sn.get("stations", []))),
                         "names": [s.get("name") for s in sn.get("stations", [])][:8]}
    except Exception as e:
        inv["snotel"] = {"n": None, "error": str(e)[:80]}
    try:
        sg = clients["usgs_water"].call_tool_json("get_monitoring_locations", {
            "bbox": bs, "site_type_code": "ST", "limit": 200}) or {}
        inv["streamgages"] = {"n": sg.get("numberReturned") or len(sg.get("features", []))}
    except Exception as e:
        inv["streamgages"] = {"n": None, "error": str(e)[:80]}
    try:
        gw = clients["usgs_water"].call_tool_json(
            "get_groundwater_sites", {"bbox": bs, "limit": 200}) or {}
        sites = gw.get("sites", [])
        depths, with_rec = [], 0
        for s in sites[:6]:                       # sample a few wells for actual records
            sid = s.get("id") or s.get("monitoring_location_id")
            if not sid:
                continue
            w = clients["usgs_water"].call_tool_json(
                "get_water_table_depth", {"monitoring_location_id": sid, "limit": 50}) or {}
            summ = w.get("summary") or {}
            if summ.get("n_obs"):
                with_rec += 1
                for k in ("mean_depth_m", "latest_depth_m", "median_depth_m"):
                    if _n(summ.get(k)) is not None:
                        depths.append(_n(summ.get(k))); break
        inv["gwwells"] = {"n": gw.get("n_sites", len(sites)),
                          "sampled": min(6, len(sites)), "with_records": with_rec,
                          "observed_wtd_m": depths}
    except Exception as e:
        inv["gwwells"] = {"n": None, "error": str(e)[:80]}
    return inv


def build_validation(run_dir: Path, clients):
    brief = json.load(open(run_dir / "reception_brief.json"))
    dom = brief.get("domain", {})
    bb = dom["bbox"]
    cols = json.load(open(run_dir / "columns.json"))
    cols = cols["columns"] if isinstance(cols, dict) else cols
    fan = {c["id"]: _n(c.get("fan_wtd_m")) for c in cols}
    hs = json.load(open(run_dir / "04_analysis" / "hydro_summary.json"))

    # per-column model state + the Fan reference at the same point
    pairs = []
    yields = []
    for r in hs["experiments"]:
        if r.get("status") != "ok":
            continue
        m = r["metrics"]
        zwt = _n(m.get("water_table_depth_m"))
        rech = _n(m.get("annual_recharge_mm_yr")) or 0.0
        runf = _n(m.get("annual_runoff_mm_yr")) or 0.0
        yields.append(rech + runf)
        f = fan.get(r["case_name"])
        if zwt is not None and f is not None:
            pairs.append({"col": r["case_name"], "model_zwt_m": round(zwt, 3),
                          "fan_wtd_m": round(f, 3)})

    import numpy as np
    inv = fetch_inventory(clients, bb)

    wtd = {}
    if pairs:
        mz = np.array([p["model_zwt_m"] for p in pairs])
        fz = np.array([p["fan_wtd_m"] for p in pairs])
        wtd = {"n_columns": len(pairs),
               "model_zwt_mean_m": round(float(mz.mean()), 2),
               "fan_wtd_mean_m": round(float(fz.mean()), 2),
               "mean_bias_m_model_minus_fan": round(float((mz - fz).mean()), 2),
               "by_column": pairs}

    mean_yield = round(float(np.mean(yields)), 1) if yields else None

    targets = [
        {"variable": "water-table depth", "obs": "Fan 2013 (regional) + USGS wells",
         "status": "compared",
         "result": (f"model ZWT mean {wtd.get('model_zwt_mean_m')} m vs Fan "
                    f"{wtd.get('fan_wtd_mean_m')} m (bias "
                    f"{wtd.get('mean_bias_m_model_minus_fan')} m)") if wtd else "no pairs",
         "note": "single-column ZWT is a shallow/perched table (no groundwater coupling, "
                 "no spin-up) — a large offset from the deep regional Fan WTD is EXPECTED; "
                 "this checks the deep-drainage assumption, it is not a calibration pass."},
        {"variable": "integrated streamflow", "obs": "USGS gages",
         "n_obs": inv.get("streamgages", {}).get("n"),
         "status": "context-only", "needs": "runoff routing / column aggregation",
         "result": f"modeled basin water yield (runoff+recharge) ≈ {mean_yield} mm/yr "
                   f"(reference; a single column produces no routed streamflow)"},
        {"variable": "snow water equivalent", "obs": "NRCS SNOTEL",
         "n_obs": inv.get("snotel", {}).get("n"),
         "status": "context-only", "needs": "snow/SWE history output + elevation-downscaled forcing",
         "result": "SNOTEL stations present, but the run has no SWE output to compare"},
    ]
    return {"domain": {"name": dom.get("name"), "huc": dom.get("huc"),
                       "area_km2": dom.get("area_km2")},
            "observation_inventory": inv,
            "wtd_comparison": wtd,
            "modeled_basin_yield_mm_yr": mean_yield,
            "targets": targets}


def plot_validation(val, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    wtd = val.get("wtd_comparison") or {}
    rows = wtd.get("by_column", [])
    if rows:
        mz = np.array([r["model_zwt_m"] for r in rows])
        fz = np.array([max(r["fan_wtd_m"], 0.05) for r in rows])
        ax[0].scatter(fz, mz, c="#2c7fb8", s=60, edgecolor="#222", zorder=3)
        lo, hi = 0.05, max(fz.max(), mz.max()) * 1.3
        ax[0].plot([lo, hi], [lo, hi], "--", color="#888", label="1:1")
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlabel("Fan 2013 regional WTD (m)")
        ax[0].set_ylabel("model ZWT (m)")
        ax[0].set_title("Water table: model vs Fan", fontweight="bold")
        ax[0].legend(frameon=False)
    inv = val.get("observation_inventory", {})
    labels = ["SNOTEL\n(SWE)", "stream gages\n(runoff)", "GW wells\n(WTD)", "wells w/\nrecords"]
    vals = [inv.get("snotel", {}).get("n") or 0,
            inv.get("streamgages", {}).get("n") or 0,
            inv.get("gwwells", {}).get("n") or 0,
            inv.get("gwwells", {}).get("with_records") or 0]
    ax[1].bar(range(4), vals, color=["#31a354", "#d95f0e", "#756bb1", "#9e9ac8"], edgecolor="#222")
    for i, v in enumerate(vals):
        ax[1].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labels, fontsize=9)
    ax[1].set_title("In-domain observations", fontweight="bold"); ax[1].set_ylabel("count")
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
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "04_analysis" / "hydro_summary.json").exists():
        sys.exit("run analyze_run.py first (no 04_analysis/hydro_summary.json)")

    from core.mcp_manager import MCPManager
    clients = MCPManager("mcp_config.json").get_all_clients()
    val = build_validation(run_dir, clients)

    print("\n" + "=" * 78)
    print(f"OBSERVATION VALIDATION — {val['domain'].get('name')} (HUC {val['domain'].get('huc')})")
    print("=" * 78)
    inv = val["observation_inventory"]
    print(f"in-domain obs: SNOTEL={inv.get('snotel',{}).get('n')}  "
          f"stream gages={inv.get('streamgages',{}).get('n')}  "
          f"GW wells={inv.get('gwwells',{}).get('n')} "
          f"({inv.get('gwwells',{}).get('with_records')}/{inv.get('gwwells',{}).get('sampled')} sampled have records)")
    print("-" * 78)
    for t in val["targets"]:
        tag = {"compared": "✓ COMPARED", "context-only": "· context"}.get(t["status"], t["status"])
        print(f"{tag:<12} {t['variable']}")
        print(f"             {t['result']}")
        if t.get("needs"):
            print(f"             blocked for rigorous validation — needs: {t['needs']}")
    print("=" * 78)

    out = run_dir / "04_analysis"
    (out / "validation.json").write_text(json.dumps(val, indent=2, default=str))
    if not args.no_plot:
        plot_validation(val, out / "validation.png")
    print(f"\nwritten to {out}/validation.json")


if __name__ == "__main__":
    main()
