#!/usr/bin/env python3
"""
Chain evaluation: reception -> planner -> sample_columns, over the 15 cases.

`reception_cases.py` stops at reception. This carries each case two stages
further and asks one question the earlier harness could not: **does the sampler
build the design the planner asked for?**

That question only became answerable on 2026-07-28, when `_enforce()` was deleted
from the planner. Before then it recomputed the column count and dropped the
station ids the planner had named, so the ids never reached plan.json. They have
survived since, and until 60d191f (2026-08-07) nothing read them: the sampler
knew `n_bands` and `n_columns` and ignored `per_band`, `n_validation` and
`validation[].stations` entirely.

The contract being checked is the planner's own, stated in planner.txt:

    n_columns = n_bands * per_band + n_validation + 2

    `n_validation` is the number of columns PINNED to observation stations.
    The sampler will place a column at each pinned station. The + 2 is the
    sampler's own water-table anchors, which it places whatever the plan says.

Every check is deterministic and reads only what the two stages emitted. No
compute is submitted; the cost is LLM calls plus MCP fetches on a login node.

Run:  python3 tests/chain_cases.py --run
      python3 tests/chain_cases.py --run --only gunnison_2015
      python3 tests/chain_cases.py --report workflow_outputs/chain_eval_XXXX
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from reception_cases import CASES                              # noqa: E402


def pin_rules(clients):
    """What a column may be pinned to, asked of the model server.

    THE HARNESS HAS TO APPLY THE SAMPLER'S OWN TEST, or it scores the pipeline
    against a rule the pipeline no longer follows — which it did on 2026-08-08,
    failing four basins for doing the right thing. That is exactly why this asks
    the server rather than keeping its own copy: a third statement of the rule
    is a third thing to go stale.
    """
    import expand_sampling as _exp
    return _exp.pinning_rules(
        clients["elm"].call_tool_json("describe_elm_capabilities", {}) or {})


# ─────────────────────────────────────────────────────────────────────────────
# CHECKS — each returns (ok, detail). A check that cannot run returns None.
# ─────────────────────────────────────────────────────────────────────────────

def check_chain(plan, reception, res, pinned, rules):
    """What the planner asked for, against what the sampler built."""
    out = []

    def add(name, ok, detail, note=False):
        """note=True records a FINDING ABOUT THE BASIN, not a verdict on the
        code. A basin with no co-locatable observation is a fact about where
        the instruments are; scoring it as a failure blames the pipeline for
        the state of the USGS network."""
        out.append({"check": name, "ok": bool(ok), "detail": detail,
                    "note": bool(note)})

    s = plan.get("sampling") or {}
    nb, pb = s.get("n_bands"), s.get("per_band")
    nv, nc = s.get("n_validation"), s.get("n_columns")

    add("plan.emits_sampling_fields",
        all(v is not None for v in (nb, pb, nv, nc)),
        f"n_bands={nb} per_band={pb} n_validation={nv} n_columns={nc}")

    # The planner's own arithmetic, from planner.txt.
    if None not in (nb, pb, nv, nc):
        # + 2 FOR THE SAMPLER'S WATER-TABLE ANCHORS. This check omitted them
        # and so failed every correct site plan: naches scored 5*3+2 = 17
        # against a plan of 19 that was right. planner.txt has carried the + 2
        # since the anchors were added; the harness had not.
        add("plan.n_columns_arithmetic", nc == nb * pb + nv + 2,
            f"{nb}*{pb}+{nv}+2 = {nb * pb + nv + 2}, plan says {nc}")

    # A `comparison: "unavailable"` entry names stations the planner has already
    # ruled out, so they are not candidates for a column. Counting them against
    # n_validation misreads a self-consistent plan as a contradictory one.
    def _ruled_out(e):
        return str(e.get("comparison") or "").strip().lower().startswith(
            "unavailable")

    entries = plan.get("validation") or []
    cited = [sid for e in entries if not _ruled_out(e)
             for sid in (e.get("stations") or [])]

    # Only meaningful when reception HAS stations to cite. Chattahoochee's
    # streamflow and water-table fetches were rate-limited (HTTP 429), so the
    # plan citing nothing was correct; scoring that as a planner failure would
    # have blamed the wrong box for a data-collection failure.
    # Only stations a 1-D column can be CO-LOCATED with count as citable. Two
    # earlier versions of this check blamed the wrong box: it fired on
    # chattahoochee, whose fetches were rate-limited (429), and then on
    # brandywine, which fetched 30 stations that are all stream gauges — not
    # pinnable since 11d, wells with null coordinates, no SNOTEL, no towers. In
    # both cases citing nothing was the correct plan.
    # A station OUTSIDE THE DIVIDE is not citable either, for the same reason a
    # gauge is not: the sampler will refuse the pin. Counting them made the
    # check fail four basins on 2026-08-08 for doing the right thing — smoky's
    # only in-basin observations are 2 stream gauges, its lone SNOTEL and all 25
    # of its wells lie outside, and the planner said so in as many words. The
    # check has to apply the same test the sampler does, or it scores the
    # pipeline against a rule the pipeline no longer follows.
    # And an AmeriFlux tower whose SERIES we cannot fetch validates nothing, so
    # a column there is spent for nothing. Chicopee has 3 towers inside the
    # divide and the planner declined all 3 — "values_available false" — which
    # is right until the account exists.
    import expand_sampling as _exp
    et_ok = ((reception.get("observations") or {}).get("et")
             or {}).get("values_available", False)
    idx = _exp._station_index(reception)
    pinnable = [k for k, v in idx.items()
                if v["station_variable"] in rules["pinnable"]
                and v.get("in_basin") is not False
                and (et_ok or v["station_variable"] != "et")]
    if pinnable:
        add("plan.cites_stations", bool(cited),
            f"{len(cited)} cited of {len(pinnable)} pinnable "
            f"({len(idx)} fetched): {cited}")
    elif idx:
        add("basin.has_colocatable_observation", False, note=True, detail=
            f"{len(idx)} stations fetched but NONE pinnable in basin — a 1-D "
            f"column cannot be co-located with any of them. Validation is "
            f"basin-aggregate, and no column is spent on it.")
    else:
        add("reception.fetched_any_station", False,
            "reception fetched NO stations — check ok/error, this may be a "
            "failed fetch rather than an empty basin")

    # planner.txt: "If a variable has no stations in observations_summary, emit
    # it with `stations: []` and `comparison: 'unavailable'`." Half-obeying it —
    # the verdict without emptying the list — is what made naches_1988 pin two
    # wells the planner had itself ruled out.
    half = [(e.get("variable"), e.get("stations"))
            for e in entries if _ruled_out(e) and (e.get("stations") or [])]
    add("plan.unavailable_has_empty_stations", not half,
        f"unavailable but still listing stations: {half}" if half
        else "no unavailable entry carries stations")

    # n_validation is DEFINED as the number of pinned columns.
    if nv is not None:
        add("plan.n_validation_matches_stations", nv == len(cited),
            f"n_validation={nv}, pinnable stations={len(cited)}")

    if res is None:
        return out

    cols = res.get("columns") or []
    pin_cols = [c for c in cols if c.get("pinned")]

    add("sampler.built_requested_count", len(cols) == nc,
        f"built {len(cols)}, plan asked {nc}")
    add("sampler.pinned_every_station", len(pin_cols) == len(pinned),
        f"pinned {len(pin_cols)} of {len(pinned)} resolved")

    # The whole point: a pinned column must sit ON its station, not near it.
    idx = {p["station_id"]: p for p in pinned}
    off = []
    for c in pin_cols:
        p = idx.get(c.get("station_id"))
        if p and (abs(c["lat"] - round(p["lat"], 5)) > 1e-6
                  or abs(c["lon"] - round(p["lon"], 5)) > 1e-6):
            off.append(c["station_id"])
    add("sampler.pinned_at_exact_coords", not off, f"off-station: {off}" if off
        else f"all {len(pin_cols)} on station coordinates")

    # per_band obeyed for the stratified columns, band by band.
    if pb:
        bad = [(b["band"], b["allocated"]) for b in (res.get("bands") or [])
               if b["grid_points"] > 0 and b["allocated"] != pb]
        add("sampler.per_band_obeyed", not bad,
            f"bands off per_band={pb}: {bad}" if bad else f"every band got {pb}")

    # How far the model surface sits from each instrument — the quantity
    # step1_compare_swe had to give up on.
    d = [(c["station_id"], c["dem_minus_station_m"]) for c in pin_cols
         if c.get("dem_minus_station_m") is not None]
    if d:
        worst = max(abs(x[1]) for x in d)
        add("sampler.dem_within_100m_of_station", worst <= 100,
            f"max |DEM - station| = {worst} m  {d}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ONE CASE
# ─────────────────────────────────────────────────────────────────────────────

def run_case(case, clients, models, verbose=False):
    import expand_sampling as exp
    from agents.reception_llm import LLMReceptionAgent
    from agents.planner import Planner

    rec_agent = LLMReceptionAgent(model=models["reception"], mcp_clients=clients,
                                  verbose=verbose, interactive=False)
    reception = rec_agent.process(case["query"])
    plan = Planner(model=models["planner"]).plan(reception)

    brief = reception.get("brief") or {}
    bbox = exp._bbox_from_brief(brief)
    n_total = exp._n_from_plan(plan)
    n_bands = exp._n_bands_from_plan(plan)
    per_band = exp._per_band_from_plan(plan)

    # Raises when the plan names a station reception never fetched. That is the
    # designed behaviour and a genuine result for this case, not a harness bug.
    pinned = exp._pinned_from_plan(plan, reception, pin_rules(clients))

    # Reception already fetched the DEM grid at the sampler's own resolution
    # (data_gather.GRID_N = 120) and clipped it to the WBD polygon, so reuse its
    # boundary rather than asking for the polygon a second time.
    boundary = ((reception.get("grid") or {}).get("boundary"))

    res = None
    if bbox and n_total:
        res = exp.expand(clients, bbox, n_total, n_bands or 4,
                         boundary=boundary, per_band=per_band, pinned=pinned)
        if res.get("error"):
            res = None
    return reception, plan, res, pinned


# ─────────────────────────────────────────────────────────────────────────────

class _GridStub:
    """The real terrain client with sample_elevation_grid served from disk.

    Reception already fetched that grid — at the sampler's own resolution, and
    clipped to the basin (data_gather.GRID_N = 120) — and every chain artifact
    stores it. Replaying from it makes the sampler stage deterministic and cheap:
    only the per-station 3DEP point queries go out, and they are the calls whose
    accuracy is the point. Re-running reception and the planner instead would
    change the plan under us, since both are LLM stages.
    """

    def __init__(self, terr, points):
        self._terr, self._points = terr, points

    def call_tool_json(self, tool, args):
        if tool == "sample_elevation_grid":
            return {"points": self._points}
        return self._terr.call_tool_json(tool, args)


def replay(d: Path, clients):
    """Re-run ONLY the sampler stage over saved artifacts, and re-score."""
    import json
    import expand_sampling as exp

    summary = []
    for f in sorted(d.glob("*.json")):
        if f.name in ("summary.json", "replay_summary.json") or ".replay." in f.name:
            continue
        art = json.loads(f.read_text())
        plan, rec = art.get("plan") or {}, art.get("reception") or {}
        cid = art["case"]["id"]
        grid = (rec.get("grid") or {})
        pts = grid.get("points") or []
        if not plan or not pts:
            print(f"  skip {cid} (no plan or no saved grid)")
            continue

        stub = {**clients, "terrain": _GridStub(clients["terrain"], pts)}
        res, pinned, err = None, [], None
        try:
            pinned = exp._pinned_from_plan(plan, rec, pin_rules(clients))
            bbox = exp._bbox_from_brief(rec.get("brief") or {})
            res = exp.expand(stub, bbox, exp._n_from_plan(plan),
                             exp._n_bands_from_plan(plan) or 4,
                             boundary=grid.get("boundary"),
                             per_band=exp._per_band_from_plan(plan),
                             pinned=pinned)
            if res.get("error"):
                err, res = res["error"], None
        except Exception as e:                                 # noqa: BLE001
            err = f"{type(e).__name__}: {e}"

        checks = check_chain(plan, rec, res, pinned, pin_rules(clients))
        n_ok = sum(1 for x in checks if x["ok"] or x.get("note"))
        bad = [x["check"] for x in checks if not (x["ok"] or x.get("note"))]
        (d / f"{cid}.replay.json").write_text(json.dumps(
            {"case": art["case"], "sampling": res, "pinned_resolved": pinned,
             "checks": checks, "error": err}, indent=2, default=str))

        before = art.get("checks") or []
        summary.append({"id": cid, "error": err,
                        "n_checks": len(checks), "n_ok": n_ok, "failed": bad,
                        "n_ok_before": sum(1 for x in before if x["ok"] or x.get("note")),
                        "n_checks_before": len(before),
                        "n_columns": (res or {}).get("n_columns"),
                        "design": (res or {}).get("sampling_design")})
        (d / "replay_summary.json").write_text(
            json.dumps(summary, indent=2, default=str))
        mark = "FAIL" if (err or bad) else "ok  "
        print(f"  {mark} {cid:<20} {n_ok}/{len(checks)} checks "
              f"(was {summary[-1]['n_ok_before']}/{summary[-1]['n_checks_before']})"
              + (f"  | {err[:60]}" if err else "")
              + (f"  | {', '.join(bad[:2])}" if bad else ""), flush=True)
    return summary


def main():
    import argparse, json, time
    from datetime import datetime as _dt

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="", help="summarise an existing run dir")
    ap.add_argument("--replay", default="",
                    help="re-run ONLY the sampler over a run dir's artifacts")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to pause BETWEEN cases, to stay under the "
                         "USGS rate limit (the 2026-08-07 run took a 429 on "
                         "chattahoochee and the planner reported it as an "
                         "absence of gauges)")
    ap.add_argument("--reception-model", default="claude-opus-4-8-project")
    ap.add_argument("--planner-model", default="claude-opus-4-8-project")
    a = ap.parse_args()

    if a.replay:
        from core.mcp_manager import MCPManager
        clients = MCPManager("mcp_config.json").get_all_clients()
        d = Path(a.replay)
        print(f"\nreplaying the sampler over {d}\n" + "=" * 72)
        s = replay(d, clients)
        print("=" * 72)
        gained = sum(r["n_ok"] for r in s) - sum(r["n_ok_before"] for r in s)
        print(f"{len(s)} cases | checks passed "
              f"{sum(r['n_ok'] for r in s)}/{sum(r['n_checks'] for r in s)} "
              f"(was {sum(r['n_ok_before'] for r in s)}/"
              f"{sum(r['n_checks_before'] for r in s)}, {gained:+d})")
        return
    if a.report:
        _report(Path(a.report))
        return
    if not a.run:
        for c in CASES:
            print(f"  {c['id']:<20} HUC {c['huc']}  {c['why']}")
        return

    from core.mcp_manager import MCPManager

    only = {s.strip() for s in a.only.split(",") if s.strip()}
    todo = [c for c in CASES if not only or c["id"] in only]
    # The two traps are reception-level (refuse / clamp) and never reach a
    # sampler, so they are not part of this chain.
    todo = [c for c in todo if not c.get("expect")]

    out = Path(a.out or f"workflow_outputs/chain_eval_{_dt.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    clients = MCPManager("mcp_config.json").get_all_clients()
    models = {"reception": a.reception_model, "planner": a.planner_model}

    print(f"\n{len(todo)} cases -> {out}\n" + "=" * 72)
    summary = []
    for n, c in enumerate(todo):
        if n and a.sleep:
            time.sleep(a.sleep)
        t0 = time.time()
        rec = plan = res = None
        pinned, err = [], None
        try:
            rec, plan, res, pinned = run_case(c, clients, models)
        except Exception as e:                                  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        el = time.time() - t0

        checks = (check_chain(plan or {}, rec or {}, res, pinned,
                              pin_rules(clients)) if plan else [])
        (out / f"{c['id']}.json").write_text(json.dumps({
            "case": c,
            "reception": {k: v for k, v in (rec or {}).items()
                          if k not in ("trace", "raw")},
            "plan": plan,
            "sampling": res,
            "pinned_resolved": pinned,
            "checks": checks,
            "error": err,
            "seconds": round(el, 1),
        }, indent=2, default=str))

        n_ok = sum(1 for x in checks if x["ok"] or x.get("note"))
        bad = [x["check"] for x in checks if not (x["ok"] or x.get("note"))]
        summary.append({"id": c["id"], "seconds": round(el, 1), "error": err,
                        "n_checks": len(checks), "n_ok": n_ok, "failed": bad,
                        "n_columns": (res or {}).get("n_columns"),
                        "design": (res or {}).get("sampling_design")})
        # written after EVERY case: a crash at case 12 must not cost the
        # eleven that already ran
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

        mark = "FAIL" if (err or bad) else "ok  "
        print(f"  {mark} {c['id']:<20}{el:6.1f}s  {n_ok}/{len(checks)} checks"
              + (f"  | {err[:70]}" if err else "")
              + (f"  | {', '.join(bad[:2])}" if bad else ""), flush=True)

    print("=" * 72)
    _report(out)


def _report(d: Path):
    import json
    s = json.loads((d / "summary.json").read_text())
    print(f"\n{d}  —  {len(s)} cases")
    tot_ok = sum(r["n_ok"] for r in s)
    tot = sum(r["n_checks"] for r in s)
    print(f"checks passed: {tot_ok}/{tot}")
    errs = [r for r in s if r["error"]]
    if errs:
        print(f"\nERRORED ({len(errs)}):")
        for r in errs:
            print(f"  {r['id']:<20} {r['error'][:100]}")
    from collections import Counter
    c = Counter(x for r in s for x in r["failed"])
    if c:
        print("\nFAILED CHECKS (count of cases):")
        for k, v in c.most_common():
            print(f"  {v:>3}  {k}")
    print("\nDESIGN BUILT:")
    for r in s:
        dd = r.get("design") or {}
        print(f"  {r['id']:<20} {str(r.get('n_columns')):>3} columns"
              f"  = {dd.get('n_pinned', '-')} pinned + {dd.get('n_stratified', '-')} stratified")


if __name__ == "__main__":
    main()
