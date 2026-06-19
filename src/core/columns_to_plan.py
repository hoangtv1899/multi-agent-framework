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


def columns_to_elm_plan(columns: List[Dict[str, Any]],
                        forcing_period: str = "baseline",
                        yr_start: int = 1995,
                        yr_end: int = 1999,
                        soil_config: str = "native",
                        substrate: str = "extrapolate") -> Dict[str, Any]:
    """columns -> {CONDITIONS_COUPLERS, ELM_CONFIG} (one coupler per column)."""
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
        couplers.append(coupler)
    return {
        "CONDITIONS_COUPLERS": couplers,
        "ELM_CONFIG": {"base_stop_option": "nyears", "base_rest_n": "1",
                       "base_rest_option": "nyears"},
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
    args = ap.parse_args()

    data = json.loads(Path(args.columns).read_text())
    cols = data.get("columns", data if isinstance(data, list) else [])
    if args.limit:
        cols = cols[:args.limit]
    plan = columns_to_elm_plan(cols, args.forcing_period, args.yr_start,
                               args.yr_end, args.soil_config, args.substrate)
    out = Path(args.out) if args.out else Path(args.columns).with_name("elm_plan.json")
    out.write_text(json.dumps(plan, indent=2))
    print(f"{len(plan['CONDITIONS_COUPLERS'])} columns -> {out}")


if __name__ == "__main__":
    main()
