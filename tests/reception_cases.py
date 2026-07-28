#!/usr/bin/env python3
"""
Reception evaluation suite — 15 fully-specified requests.

Every query names its watershed by HUC8 AND its simulation period, so reception
should never need to ask. Anything it asks about was already in the request.

Six cases share one basin and differ only in the year, so any difference
downstream is the YEAR doing work — observation coverage, forcing boundaries.
Nine cover contrasting regimes: dense gauging against arid, snow-dominated
against lowland, an endorheic basin with no outlet to gauge.

Two are deliberate traps, both otherwise well-formed so the period is the only
thing wrong:
    naches_2025      outside the forcing window entirely -> must REFUSE
    manitowoc_2224   straddles the upper bound          -> must CLAMP

Run:  python3 tests/reception_cases.py --list
      python3 tests/reception_cases.py --run [--only naches_1988,verde_2005]
"""

CASES = [
    # ── one watershed, six periods ──────────────────────────────────────────
    dict(id="naches_1988", huc="17030002", name="Naches", yr=(1988, 1988),
         why="the one window where the basin-OUTLET gauge reports (~100% of basin)",
         query="Quantify how precipitation partitions into runoff and recharge "
               "across the Naches watershed (HUC8 17030002) for 1988, and "
               "compare the results with available observations."),

    dict(id="naches_1995", huc="17030002", name="Naches", yr=(1995, 1995),
         why="same basin, only the 7% headwater gauge reports",
         query="Quantify how precipitation partitions into runoff and recharge "
               "across the Naches watershed (HUC8 17030002) for 1995, and "
               "compare the results with available observations."),

    dict(id="naches_2020", huc="17030002", name="Naches", yr=(2020, 2020),
         why="recent year; wells exist in the bbox but stopped reporting in 2003",
         query="For the Naches watershed (HUC8 17030002), simulate 2020 and "
               "report the annual water balance — runoff, recharge and ET — "
               "against whatever observations exist."),

    dict(id="naches_1979", huc="17030002", name="Naches", yr=(1979, 1979),
         why="FIRST runnable forcing year — lower boundary",
         query="What fraction of precipitation becomes recharge in the Naches "
               "watershed (HUC8 17030002) during 1979? Validate against "
               "observed streamflow where possible."),

    dict(id="naches_2023", huc="17030002", name="Naches", yr=(2023, 2023),
         why="LAST runnable forcing year — upper boundary",
         query="Estimate annual recharge and runoff for the Naches watershed "
               "(HUC8 17030002) in 2023, with an observational comparison."),

    dict(id="naches_2025", huc="17030002", name="Naches", yr=None,
         expect="refuse",
         why="TRAP: outside 1979-2023 entirely — must refuse, not clamp to 2023",
         query="Estimate annual recharge and runoff for the Naches watershed "
               "(HUC8 17030002) in 2025, with an observational comparison."),

    # ── nine basins, contrasting regimes ────────────────────────────────────
    dict(id="brandywine_2010", huc="02040205", name="Brandywine-Christina",
         yr=(2010, 2010),
         why="DENSE gauge network (25 reporting), low relief, mid-Atlantic",
         query="Quantify the annual water balance of the Brandywine-Christina "
               "basin (HUC8 02040205) for 2010 — how much precipitation leaves "
               "as runoff versus recharge — and compare with gauge observations."),

    dict(id="gunnison_2015", huc="14020002", name="Upper Gunnison",
         yr=(2015, 2015),
         why="high relief, snow-dominated, large basin (6245 km2)",
         query="For the Upper Gunnison basin (HUC8 14020002), simulate 2015 and "
               "quantify how snowmelt partitions into runoff and recharge, "
               "validated against available observations."),

    dict(id="verde_2005", huc="15060203", name="Lower Verde", yr=(2005, 2005),
         why="ARID southwest, ephemeral streams — expect thin observations",
         query="Quantify runoff and recharge in the Lower Verde basin "
               "(HUC8 15060203) for 2005, and report which observations are "
               "available to check them."),

    dict(id="smoky_2012", huc="16060004", name="Northern Big Smoky Valley",
         yr=(2012, 2012),
         why="ENDORHEIC Great Basin — no outlet to gauge, by definition",
         query="For the Northern Big Smoky Valley (HUC8 16060004), simulate 2012 "
               "and quantify the partitioning of precipitation into runoff, "
               "recharge and ET, with any observational comparison available."),

    dict(id="chicopee_2018", huc="01080204", name="Chicopee River",
         yr=(2018, 2018),
         why="humid New England, SMALL basin (1872 km2)",
         query="Quantify how precipitation partitions into runoff and recharge "
               "in the Chicopee River basin (HUC8 01080204) for 2018, and "
               "compare against observed streamflow."),

    dict(id="chattahoochee_2000", huc="03130002",
         name="Middle Chattahoochee", yr=(2000, 2000),
         why="southeast — NO SNOTEL, so one observable is legitimately absent",
         query="For HUC8 03130002 (Middle Chattahoochee), simulate the year 2000 "
               "and quantify runoff versus recharge, validated against "
               "available observations."),

    dict(id="centralcoast_1998", huc="18060006", name="Central Coastal",
         yr=(1998, 1998),
         why="Mediterranean climate, strong El Nino year",
         query="Quantify the annual water balance of the Central Coastal "
               "California basin (HUC8 18060006) for 1998, including runoff and "
               "recharge, with observational comparison."),

    dict(id="stvrain_2013", huc="10190005", name="St. Vrain", yr=(2013, 2013),
         why="Colorado Front Range; 2013 was the extreme flood year",
         query="For the St. Vrain basin (HUC8 10190005), simulate 2013 and "
               "quantify how precipitation partitions into runoff and recharge, "
               "compared against observed streamflow."),

    dict(id="manitowoc_2224", huc="04030101", name="Manitowoc-Sheboygan",
         yr=(2022, 2023), expect="clamp",
         why="TRAP: 2022-2024 straddles the upper bound — must clamp to 2022-2023",
         query="Quantify runoff and recharge for the Manitowoc-Sheboygan basin "
               "(HUC8 04030101) over 2022-2024, and compare with available "
               "observations."),
]


def check(case, pkg):
    """Grade one reception package. Returns (n_pass, [failure strings])."""
    fails = []
    route = (pkg or {}).get("route") or {}
    brief = (pkg or {}).get("brief") or {}
    dom = brief.get("domain") or {}
    rs = brief.get("run_settings") or {}
    period = rs.get("resolved_period") or {}
    obs = (pkg or {}).get("observations") or {}
    grid = (pkg or {}).get("grid") or {}
    expect = case.get("expect")

    def want(cond, msg):
        if not cond:
            fails.append(msg)

    if expect == "refuse":
        # The only acceptable outcomes: refuse outright, or flag it loudly.
        conflicts = " ".join(rs.get("conflicts") or []).lower()
        refused = (route.get("action") == "clarify"
                   or "cannot" in conflicts or "outside" in conflicts
                   or not period)
        want(refused, "2025 was accepted without refusing or flagging it")
        want(period.get("yr_start") != 2025,
             "resolved a period of 2025, which cannot be forced")
        return 2 - len(fails), fails

    want(route.get("action") == "design",
         f"route={route.get('action')}, expected design")
    want(str(dom.get("huc") or "").startswith(case["huc"][:8]),
         f"huc={dom.get('huc')}, expected {case['huc']}")
    want(dom.get("bbox"), "no bbox resolved")

    y0, y1 = case["yr"]
    want(period.get("yr_start") == y0 and period.get("yr_end") == y1,
         f"period={period.get('yr_start')}-{period.get('yr_end')}, expected {y0}-{y1}")
    if expect == "clamp":
        want(period.get("source") == "user-clamped",
             f"period source={period.get('source')}, expected user-clamped")
    else:
        want(period.get("source") == "user",
             f"period source={period.get('source')}, expected user "
             f"(the year was given, so it must not be a default)")

    want((rs.get("initialization") or {}).get("mode") == "warm",
         "initialization is not warm (warm is the default)")

    # the gather phase must have run and recorded provenance either way
    want(grid.get("n_in_basin"), "no DEM grid gathered")
    want(obs.get("streamflow") is not None, "observations were not gathered")
    want(any(p.get("tool", "").endswith("get_streamflow")
             for p in (pkg or {}).get("provenance") or []),
         "no streamflow provenance recorded")

    # the regression that started this: prose must not contradict the data
    n_bbox = ((obs.get("streamflow") or {}).get("n_in_bbox"))
    summ = ((brief.get("observations_summary") or {}).get("streamflow") or {})
    want(summ.get("n_in_bbox") == n_bbox,
         f"summary says {summ.get('n_in_bbox')} gauges, data says {n_bbox}")

    return 9 - len(fails), fails


if __name__ == "__main__":
    import argparse, json, sys, time
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="reception_eval.json")
    ap.add_argument("--model", default="claude-opus-4-8-project")
    a = ap.parse_args()

    if a.list or not a.run:
        for c in CASES:
            print(f"  {c['id']:<20} HUC {c['huc']}  {str(c['yr']):<14} {c['why']}")
        sys.exit(0)

    sys.path.insert(0, "src")
    from core.mcp_manager import MCPManager
    from agents.reception_llm import LLMReceptionAgent

    only = {s.strip() for s in a.only.split(",") if s.strip()}
    todo = [c for c in CASES if not only or c["id"] in only]
    clients = MCPManager("mcp_config.json").get_all_clients()

    results = []
    for c in todo:
        agent = LLMReceptionAgent(model=a.model, mcp_clients=clients,
                                  verbose=False, interactive=False)
        t0 = time.time()
        try:
            pkg = agent.process(c["query"])
            err = None
        except Exception as e:                       # noqa: BLE001
            pkg, err = {}, f"{type(e).__name__}: {e}"
        el = time.time() - t0
        n_ok, fails = check(c, pkg) if not err else (0, [err])
        results.append({"id": c["id"], "seconds": round(el, 1),
                        "passed": n_ok, "failures": fails,
                        "rounds": pkg.get("rounds"),
                        "package": pkg})
        mark = "ok  " if not fails else "FAIL"
        print(f"  {mark} {c['id']:<20} {el:6.1f}s  {n_ok} checks"
              + ("" if not fails else "  | " + "; ".join(fails[:2])))

    Path(a.out).write_text(json.dumps(results, indent=2, default=str))
    bad = [r for r in results if r["failures"]]
    print(f"\n  {len(results) - len(bad)}/{len(results)} cases clean -> {a.out}")
