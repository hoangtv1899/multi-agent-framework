#!/usr/bin/env python3
"""
Scout candidate watersheds for OBSERVATION coverage before committing a run.
Resolves each (name or HUC8 code) to its HUC8 + bbox and inventories the
in-domain observations — SNOTEL SWE stations, USGS stream gages, GW wells — plus
area and relief, so you can pick a data-rich basin (and, for less forcing
confound, a lower-relief one).

    module load pytorch/2.8.0
    python3 tools/scout_watersheds.py "Naches" "Yakima" "Walla Walla" 17030001
"""
import argparse
import sys

sys.path.insert(0, "src")
from core.mcp_manager import MCPManager


def scout(clients, query):
    terr = clients["terrain"]
    key = "huc" if query.isdigit() else "name"
    r = terr.call_tool_json("resolve_watershed", {key: query}) or {}
    bb = r.get("bbox")
    if not bb:
        return {"query": query, "error": "could not resolve"}
    # bbox may be a dict or "lon,lat,lon,lat" string
    if isinstance(bb, str):
        x0, y0, x1, y1 = (float(v) for v in bb.split(","))
        bb = {"min_lon": x0, "min_lat": y0, "max_lon": x1, "max_lat": y1}
    bs = f'{bb["min_lon"]},{bb["min_lat"]},{bb["max_lon"]},{bb["max_lat"]}'

    def count(server, tool, args, *keys):
        try:
            d = clients[server].call_tool_json(tool, args) or {}
            for k in keys:
                if d.get(k) is not None:
                    return d[k]
            return 0
        except Exception:
            return None

    snotel = count("snotel", "get_snotel_stations",
                   {"min_lon": bb["min_lon"], "min_lat": bb["min_lat"],
                    "max_lon": bb["max_lon"], "max_lat": bb["max_lat"]}, "n_stations")
    gages = count("usgs_water", "get_monitoring_locations",
                  {"bbox": bs, "site_type_code": "ST", "limit": 200}, "numberReturned")
    wells = count("usgs_water", "get_groundwater_sites", {"bbox": bs, "limit": 200}, "n_sites")
    bands = []
    try:
        bands = (terr.call_tool_json("elevation_summary", {**bb, "n": 60}) or {}).get(
            "elevation_bands") or []
    except Exception:
        pass
    relief = round(bands[-1]["elev_hi_m"] - bands[0]["elev_lo_m"]) if bands else None
    return {"query": query, "name": r.get("name"), "huc": r.get("huc"),
            "area_km2": r.get("area_km2"), "relief_m": relief,
            "snotel": snotel, "gages": gages, "wells": wells}


def main():
    ap = argparse.ArgumentParser(description="Scout watersheds for observation coverage")
    ap.add_argument("queries", nargs="+", help="watershed names and/or HUC8 codes")
    args = ap.parse_args()
    clients = MCPManager("mcp_config.json").get_all_clients()

    rows = [scout(clients, q) for q in args.queries]
    print("\n" + "=" * 92)
    print("WATERSHED OBSERVATION SCOUT")
    print("=" * 92)
    print(f"{'watershed':<22}{'HUC8':>10}{'area_km2':>10}{'relief_m':>9}"
          f"{'SNOTEL':>8}{'gages':>7}{'wells':>7}")
    print("-" * 92)
    for r in rows:
        if r.get("error"):
            print(f"{r['query']:<22}  {r['error']}")
            continue
        def s(x):
            return "-" if x is None else (f"{x:.0f}" if isinstance(x, float) else str(x))
        print(f"{(r['name'] or r['query'])[:21]:<22}{s(r['huc']):>10}{s(r['area_km2']):>10}"
              f"{s(r['relief_m']):>9}{s(r['snotel']):>8}{s(r['gages']):>7}{s(r['wells']):>7}")
    print("=" * 92)
    print("gages/wells are capped at 200 (a '200' means ≥200). Lower relief ⇒ less forcing confound.")


if __name__ == "__main__":
    main()
