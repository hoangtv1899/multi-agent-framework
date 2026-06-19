#!/usr/bin/env python3
"""
CONUS coverage sweep for the spatial MCP tools (terrain, usgs_water, fan_wtd).

This is a DIAGNOSTIC, not a unit test: it exercises every spatial tool against a
spread of CONUS reference sites (arid SW, humid SE, Plains, PNW, Rockies, NE …)
through the real MCPClient path, and prints a coverage table — None-rates, well
counts, Fan no-data — so you can see where coverage is thin BEFORE designing a
study there. The deterministic logic is covered by tests/test_mcp_tools.py.

Run from the project root with the MCP runtime env:
    module load pytorch/2.8.0
    python3 tools/mcp_conus_sweep.py                 # full report
    python3 tools/mcp_conus_sweep.py --max-sites 4   # quick subset
    python3 tools/mcp_conus_sweep.py --assert         # report + exit 1 on sanity failures

Note: each MCP call opens a fresh server session, so a full sweep is a couple of
minutes. Network is required for terrain + usgs_water (fan_wtd is local).
"""
import argparse
import sys

sys.path.insert(0, "src")
from core.mcp_manager import MCPManager

# CONUS reference sites spanning hydroclimatic regimes.
SITES = [
    ("Naches PNW, WA",     46.75, -120.70),
    ("Spokane, WA",        47.66, -117.43),
    ("Boulder Rockies, CO", 40.01, -105.27),
    ("Phoenix arid, AZ",   33.45, -112.07),
    ("Fresno CV, CA",      36.74, -119.79),
    ("Lincoln Plains, NE", 40.81, -96.70),
    ("Bismarck, ND",       46.81, -100.78),
    ("Houston Gulf, TX",   29.76, -95.37),
    ("Atlanta humid, GA",  33.75, -84.39),
    ("Tallahassee, FL",    30.44, -84.28),
    ("Minneapolis, MN",    44.98, -93.27),
    ("Burlington NE, VT",  44.48, -73.21),
]

# Named watersheds to exercise resolve_watershed (name or HUC).
WATERSHEDS = [("Naches", {"huc": "17030002"}),
              ("Brandywine-Christina", {"huc": "02040205"})]


def _bbox(lat, lon, d=0.25):
    return f"{lon - d},{lat - d},{lon + d},{lat + d}"


def sweep(mgr, sites, do_assert):
    terr = mgr.get_client("terrain")
    usgs = mgr.get_client("usgs_water")
    fan = mgr.get_client("fan_wtd")
    failures = []

    print("\n" + "=" * 84)
    print("CONUS MCP COVERAGE SWEEP")
    print("=" * 84)
    print(f"{'site':<22}{'elev_m':>9}{'#GWwells':>10}{'WTDobs':>8}{'fan_m':>9}  note")
    print("-" * 84)

    cov = {"elev": 0, "wells": 0, "wtd": 0, "fan": 0}
    for name, lat, lon in sites:
        elev = wells = wtd = fanm = None
        note = ""

        e = terr.call_tool_json("get_elevation", {"lat": lat, "lon": lon}) or {}
        elev = e.get("elevation_m")
        if elev is not None:
            cov["elev"] += 1
            if do_assert and not (-100 <= elev <= 5000):
                failures.append(f"{name}: elevation {elev} out of CONUS range")

        gw = usgs.call_tool_json("get_groundwater_sites", {"bbox": _bbox(lat, lon), "limit": 100}) or {}
        sites_list = gw.get("sites", [])
        wells = gw.get("n_sites")
        if wells:
            cov["wells"] += 1
            sid = sites_list[0].get("id")
            w = usgs.call_tool_json("get_water_table_depth", {"monitoring_location_id": sid}) or {}
            wtd = (w.get("summary") or {}).get("n_obs")
            if wtd:
                cov["wtd"] += 1
        else:
            note = "no GW wells in bbox"

        fr = fan.call_tool_json("get_fan_wtd", {"lat": lat, "lon": lon}) or {}
        fanm = fr.get("depth_to_water_m")
        if fanm is not None:
            cov["fan"] += 1
            if do_assert and fanm < 0:
                failures.append(f"{name}: Fan depth negative ({fanm})")
        elif fr.get("error"):
            note = (note + " | " if note else "") + "fan: " + fr["error"][:30]

        print(f"{name:<22}{_fmt(elev):>9}{_fmt(wells):>10}{_fmt(wtd):>8}"
              f"{_fmt(fanm):>9}  {note}")

    n = len(sites)
    print("-" * 84)
    print(f"coverage: elevation {cov['elev']}/{n} | GW wells {cov['wells']}/{n} | "
          f"WTD obs {cov['wtd']}/{n} | Fan {cov['fan']}/{n}")

    # watershed resolver check
    print("\nresolve_watershed:")
    for label, kw in WATERSHEDS:
        r = terr.call_tool_json("resolve_watershed", kw) or {}
        ok = bool(r.get("bbox"))
        print(f"  {label:<24} -> {r.get('name')} ({r.get('huc')}), "
              f"{r.get('area_km2')} km2  {'OK' if ok else 'FAILED'}")
        if do_assert and not ok:
            failures.append(f"resolve_watershed {label} returned no bbox")
    print("=" * 84)
    return failures


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(description="CONUS coverage sweep for spatial MCP tools")
    ap.add_argument("--max-sites", type=int, default=0, help="limit number of sites (quick run)")
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    help="exit 1 if sanity checks fail")
    args = ap.parse_args()

    sites = SITES[:args.max_sites] if args.max_sites else SITES
    mgr = MCPManager("mcp_config.json")
    failures = sweep(mgr, sites, args.do_assert)

    if args.do_assert:
        if failures:
            print(f"\nSANITY FAILURES ({len(failures)}):")
            for f in failures:
                print(f"  - {f}")
            sys.exit(1)
        print("\nall sanity checks passed.")
    print()


if __name__ == "__main__":
    main()
