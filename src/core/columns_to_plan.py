#!/usr/bin/env python3
"""
Tier-2 -> Tier-3 adapter: expander columns -> an executable ELM plan.

Turns the expander's columns.json (each column = a real (lat,lon) + full soil
profile, materialized from MCP data) into the CONDITIONS_COUPLERS plan that
ELMExpManager runs — one coupler per column, each carrying its OWN lat/lon +
soil_profile (the builder reads these per-coupler, ELM_CONFIG as fallback).
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def build_ledger(columns, yr_start, yr_end, soil_config, substrate,
                 forcing_period, finidat_map=None, period_source=None,
                 n_source="planner/expander") -> List[Dict[str, Any]]:
    """The assumptions ledger — every load-bearing choice, tagged user vs
    DEFAULT, so silent defaults cannot hide.

    Lives here rather than in main() because BOTH callers need it: the CLI and
    ELMExpManager. It used to be built only in main(), so integrated runs shipped
    an empty ledger and the interpreter — which is told to flag DEFAULT-sourced
    assumptions — had nothing to flag.
    """
    n_years = yr_end - yr_start + 1
    n_warm  = sum(1 for c in columns
                  if (finidat_map or {}).get(c.get("id"))) if finidat_map else 0
    if n_warm:
        src = next(iter(finidat_map.values()))
        src = src.get("source", "warm") if isinstance(src, dict) else "warm"
        init_value = (f"{src} warm start (finidat), {n_warm}/{len(columns)} columns"
                      + ("" if n_warm == len(columns) else " — the rest COLD"))
        init_source = "user"
    else:
        init_value, init_source = "cold start (uniform ELM default)", "DEFAULT"
    return [
        {"parameter": "simulation period", "value": f"{yr_start}-{yr_end}",
         "source": period_source or "DEFAULT",
         "note": "science year = last simulated year; the year's climatic "
                 "percentile is not characterized"},
        {"parameter": "spin-up", "value": (f"{n_years - 1} yr (in-run)" if n_years > 1
                                           else "none"),
         "source": "derived", "note": "recharge/storage terms are transient "
                                      "without >=3 spin-up years"},
        {"parameter": "initialization", "value": init_value, "source": init_source,
         "note": "initial water table controls recharge sign in 1-yr runs"},
        {"parameter": "soil configuration", "value": soil_config,
         "source": "DEFAULT" if soil_config == "native" else "user",
         "note": f"substrate={substrate}"},
        {"parameter": "forcing period class", "value": forcing_period,
         "source": "DEFAULT" if forcing_period == "baseline" else "user",
         "note": ""},
        {"parameter": "N (columns)", "value": len(columns), "source": n_source,
         "note": "materialized by the deterministic expander from the "
                 "planner's sampling strategy"},
    ]


def columns_to_elm_plan(columns: List[Dict[str, Any]],
                        forcing_period: str = "baseline",
                        yr_start: int = 1995,
                        yr_end: int = 1999,
                        soil_config: str = "native",
                        substrate: str = "extrapolate",
                        finidat_map: Dict[str, Any] = None,
                        period_source: str = None) -> Dict[str, Any]:
    """columns -> {CONDITIONS_COUPLERS, ELM_CONFIG, assumptions_ledger}.

    finidat_map: {col_id: warmstart.json entry} — attaches FINIDAT per column
    (warm start). Columns absent from the map stay cold, and the ledger records
    the split rather than implying the whole ensemble was warm-started.
    """
    stop_n = yr_end - yr_start + 1
    couplers: List[Dict[str, Any]] = []
    for c in columns:
        coupler = {
            "EXPERIMENT": c.get("id", f"col_{len(couplers) + 1:02d}"),
            "FORCING_PERIOD": forcing_period,
            "DATM_CLMNCEP_YR_START": str(yr_start),
            "DATM_CLMNCEP_YR_END": str(yr_end),
            "STOP_N": str(stop_n),
            "RUN_STARTDATE": f"{yr_start}-01-01",
            "SOIL_CONFIG": soil_config,
            "DESCRIPTION": f"{c.get('id', 'col')} @ band {c.get('band', '?')}, "
                           f"{c.get('elevation_m', '?')} m",
            "lat": c["lat"], "lon": c["lon"],
        }
        if soil_config == "native":
            coupler["SUBSTRATE"] = substrate
            if c.get("soil_profile"):
                coupler["soil_profile"] = c["soil_profile"]
        m = (finidat_map or {}).get(coupler["EXPERIMENT"])
        if m:
            coupler["FINIDAT"] = m["finidat"] if isinstance(m, dict) else m
            # A CONUS-subset warm start also supplies the gridcell's own
            # surfdata as the surface template, so fsurdat and finidat describe
            # the same cell (ELM's check_weights gate).
            if isinstance(m, dict) and m.get("surface_template"):
                coupler["SURFACE_TEMPLATE"] = m["surface_template"]
        couplers.append(coupler)
    return {
        "CONDITIONS_COUPLERS": couplers,
        "ELM_CONFIG": {"base_stop_option": "nyears", "base_rest_n": "1",
                       "base_rest_option": "nyears"},
        "assumptions_ledger": build_ledger(
            columns, yr_start, yr_end, soil_config, substrate, forcing_period,
            finidat_map=finidat_map, period_source=period_source),
    }


def main():
    ap = argparse.ArgumentParser(description="columns.json -> executable ELM plan.json")
    ap.add_argument("columns", help="path to columns.json (from the expander)")
    ap.add_argument("--out", default=None, help="plan output (default <dir>/elm_plan.json)")
    ap.add_argument("--forcing-period", default="baseline")
    ap.add_argument("--yr-start", type=int, default=1995)
    ap.add_argument("--yr-end", type=int, default=1999)
    ap.add_argument("--soil-config", default="native")
    ap.add_argument("--substrate", default="extrapolate")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N columns")
    ap.add_argument("--period-source", default=None,
                    choices=("user", "user-clamped", "reception", "DEFAULT"),
                    help="true origin of --yr-start/--yr-end for the ledger "
                         "(wrappers pass years explicitly, so the != default "
                         "heuristic would mislabel wrapper defaults as 'user')")
    ap.add_argument("--finidat-map", default=None,
                    help="warmstart.json (make_warmstart.py): col id -> finidat path")
    args = ap.parse_args()

    data = json.loads(Path(args.columns).read_text())
    cols = data.get("columns", data if isinstance(data, list) else [])
    if args.limit:
        cols = cols[:args.limit]
    dflt = {a.dest: a.default for a in ap._actions}
    period_source = (args.period_source if args.period_source
                     else ("user" if (args.yr_start != dflt.get("yr_start")
                                      or args.yr_end != dflt.get("yr_end"))
                           else "DEFAULT"))
    fmap = (json.loads(Path(args.finidat_map).read_text())
            if args.finidat_map else None)

    plan = columns_to_elm_plan(cols, args.forcing_period, args.yr_start,
                               args.yr_end, args.soil_config, args.substrate,
                               finidat_map=fmap, period_source=period_source)
    if args.limit:
        for a in plan["assumptions_ledger"]:
            if a["parameter"] == "N (columns)":
                a["source"] = "user"
    out = Path(args.out) if args.out else Path(args.columns).with_name("elm_plan.json")
    out.write_text(json.dumps(plan, indent=2))
    print(f"{len(plan['CONDITIONS_COUPLERS'])} columns -> {out}")


if __name__ == "__main__":
    main()
