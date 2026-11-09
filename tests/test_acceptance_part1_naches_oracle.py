"""Acceptance without model calls, part 1 of package G (2026-11-06): the five
tests of the plan's section 7 on runs that exist.

    pytest tests/test_acceptance_part1_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the files are on disk: the acceptance_g1 block of
tests/local_oracle_paths.json names a scratch folder (every suite and study
these tests make is written there and nowhere else) and the run directories of
the three experiment runs (naches_bgc_fix_warm_dry_summer, _nofire, _dry); the
Naches control's registry entry names its run, case and suite.

The tests are grouped by the plan's kinds. CONTRACT TESTS (tests 2-4) check
that calculations and eligibility behave as the contracts say. REGRESSION
TESTS (tests 1 and 5, and the values of the October hand analysis) check that
these runs keep their expected outputs; a regression expectation is not a
general requirement of a method. Wherever a report is needed, the Analyzer's
outputs are scripted (fakes): the plan, its review and the claims. No model
is called and nothing is submitted.

Each test writes what it measured to measured.json in the scratch folder, so
the values are reported whether the test passes or fails.
"""
import copy
import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import domain_access as DACC
import domain_controls as DCT
import domain_methods as M
import domain_suite as S
import gridded as G
from test_domain_access import Ctx, Fake
from test_domain_catalog_runs import server_change

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402
from agents.analysis import step4_report as S4           # noqa: E402

pytestmark = pytest.mark.localdata
CONFIG = Path(__file__).with_name("local_oracle_paths.json")
REGION = "watershed, vegetated"
VEG = "vegetated cells"
OUTSIDE = "outside the watershed"                 # the region these tests make
OUTSIDE_REGION = OUTSIDE + ", vegetated"           # as the suite names it
NORMAL = [2016, 2017, 2018, 2019, 2020]
YEARS = list(range(2015, 2024))
WARM_RUN = "warm_dry_summer"
WARM = {"id": "weather_scale", "params": {"temperature_K": 2.0, "precip_factor": 0.8,
                                          "months": [6, 7, 8], "region": "watershed"}}
NOTES = M._notes(None)
FIELDS = M.catalog_fields(NOTES)
WEATHER = [f for f in FIELDS if (NOTES.get(f) or {}).get("group") == "weather in"]
EVENTS = [f for f in FIELDS if (NOTES.get(f) or {}).get("type") == M.EVENT_TYPE]
JAN_MAY_2015 = ([M.period_label("month", 2015, m) for m in range(1, 6)]
                + [M.period_label("season", 2015, "MAM")])
WARMED_2015 = [M.period_label("month", 2015, m) for m in (6, 7, 8)] + \
    [M.period_label("season", 2015, "JJA")]
# the periods a run that starts on 1 January 2015 cannot hold
BEFORE_2015 = {M.period_label("wateryear", 2015), M.period_label("season", 2015, "DJF")}
IMPOSED_K = 2.0
EXACT_K = 1e-3          # float32 history, region means in float64: far below any physical change
WEEK = 7.0


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def cfg():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    c = json.loads(CONFIG.read_text()).get("acceptance_g1") or {}
    runs = {WARM_RUN: c.get("warm_run_dir"), "nofire": c.get("nofire_run_dir"),
            "dry": c.get("dry_run_dir")}
    if not c.get("scratch") or not all(v and Path(v).is_dir() for v in runs.values()):
        pytest.skip("no acceptance_g1 block with a scratch folder and the three run directories")
    entry = (DCT.load().get("controls") or {}).get("naches") or {}
    if not entry.get("suite_dir") or not (Path(entry["suite_dir"]) / "suite.json").is_file():
        pytest.skip("the naches registry entry names no built suite")
    scratch = Path(c["scratch"])
    scratch.mkdir(parents=True, exist_ok=True)
    mp = pytest.MonkeyPatch()
    mp.setenv(G.INDEX_CACHE_ENV, str(scratch / "index_cache"))
    yield {"scratch": scratch, "runs": runs, "entry": entry,
           "control": M.Suite(entry["suite_dir"])}
    mp.undo()


def _note(cfg, key, value):
    """What a test measured, kept in the scratch folder for the report."""
    p = cfg["scratch"] / "measured.json"
    d = json.loads(p.read_text()) if p.is_file() else {}
    d[key] = value
    p.write_text(json.dumps(d, indent=1, default=_plain_number))


def _plain_number(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _outside_field(cfg) -> Path:
    """The region of test 2, made from the registry's masks as the control's
    suite holds them: vegetated cells and not the watershed's."""
    p = cfg["scratch"] / "outside_watershed.nc"
    if p.is_file():
        return p
    with netCDF4.Dataset(cfg["control"].dir / S.MASKS_FILE) as d:
        d.set_auto_mask(False)
        lat, lon = np.asarray(d["lat"][:]), np.asarray(d["lon"][:])
    ctl = cfg["control"]
    out = ctl.mask(VEG) & ~ctl.mask(REGION)
    with netCDF4.Dataset(p, "w") as d:
        d.createDimension("lat", lat.size)
        d.createDimension("lon", lon.size)
        d.createVariable("lat", "f8", ("lat",))[:] = lat
        d.createVariable("lon", "f8", ("lon",))[:] = lon
        v = d.createVariable("outside_watershed", "f4", ("lat", "lon"))
        v[:] = out.astype(np.float32)
        v.long_name = ("1 where a cell is vegetated and outside the watershed polygon, made "
                       "from the Naches control suite's masks (vegetated cells and not "
                       "watershed, vegetated)")
    return p


def _entry_plus(cfg):
    plus = dict(cfg["entry"])
    plus["regions"] = dict(cfg["entry"]["regions"])
    plus["regions"][OUTSIDE] = {"field": str(_outside_field(cfg)), "var": "outside_watershed",
                                "op": ">", "value": 0.0, "over": "any",
                                "what": "vegetated cells outside the watershed polygon"}
    return plus


@pytest.fixture(scope="module")
def suites(cfg):
    """Each experiment run's suite, built (or reused) by the server's own
    builder in the scratch folder only; and the control and the warm, dry
    summer again with the region of test 2 added."""
    g, e = cfg["scratch"], cfg["entry"]
    plus = _entry_plus(cfg)
    want = {WARM_RUN: (e, g / "study_warm" / "03_results" / "suite" / WARM_RUN,
                       cfg["runs"][WARM_RUN]),
            "nofire": (e, g / "suites" / "nofire", cfg["runs"]["nofire"]),
            "dry": (e, g / "suites" / "dry", cfg["runs"]["dry"]),
            "warm_plus": (plus, g / "suites" / "warm_plus", cfg["runs"][WARM_RUN]),
            "control_plus": (plus, g / "suites" / "control_plus", e["run_dir"])}
    out = {"control": e["suite_dir"]}
    for name, (entry, d, rd) in want.items():
        r = S.build(entry, d, run_dir=rd)
        assert r["action"] in ("reused", "built", "added"), (name, r)
        out[name] = str(d)
    return out


def _open(dirs):
    """A `suite` for the contract tests: every call answered and logged in
    memory, with no plan (these tests read the methods' answers, not a
    study's)."""
    return DACC.SuiteAccess(dict(dirs), gate=lambda ask: {}, log=lambda rec: None,
                            normal_years=NORMAL)


def _by_year(acc, run, mid, y):
    """A measure's stored outcome of one year (timing.nc, through get_timing),
    in the shape timing_summary reads."""
    return {"method": mid, "version": M.METHODS[mid].version, "kind": M.TIMING_KIND[mid],
            "year": int(y), "period": M.period_label(M.TIMING_KIND[mid], int(y)),
            "variants": acc.get_timing(run, mid, int(y))}


def _variant(by_year, **settings):
    return next(k for k, v in by_year["variants"].items() if v["settings"] == settings)


def _grid_mask(n, cells):
    m = np.zeros(n, dtype=bool)
    m[cells] = True
    return m


# ── a study, as the chain leaves it, in the scratch folder ──────────────────
def _records(rd: Path, entry, runs):
    """The manager's records a study's checks read, naming the real run
    directories: runs is {name: (run_dir, change, years)}."""
    (rd / "01_inputs").mkdir(parents=True, exist_ok=True)
    (rd / "01_inputs" / "domain_control_facts.json").write_text(json.dumps(
        {"control": {"name": "naches", "run_dir": entry["run_dir"],
                     "case_dir": entry["case_dir"]}}))
    exp_runs = [{"run": "control", "reused": True, "changes": []}]
    chain, years = {}, []
    for name, (run_dir, change, ys) in runs.items():
        (rd / "domain_case" / name / "checks").mkdir(parents=True, exist_ok=True)
        for y in ys:
            (rd / "domain_case" / name / "checks" / f"y{y}.json").write_text(
                json.dumps({"run_dir": run_dir}))
        exp_runs.append({"run": name, "changes": [server_change(change)]})
        chain[name] = {"stages": [{"stage": "case", "check_pass": True},
                                  {"stage": "smoke", "check_pass": True}]
                       + [{"stage": f"y{y}", "check_pass": True} for y in ys]
                       + [{"stage": "report", "check_pass": None}]}
        years = sorted(set(years) | set(ys))
    (rd / "experiment.json").write_text(json.dumps({
        "kind": "whole_domain", "experiment": {"runs": exp_runs},
        "chain": {"years_planned": years, "years_passed": years, "runs": chain}}))


def _study(rd: Path, entry, runs, ctx_plan, plan, script):
    _records(rd, entry, runs)
    out = rd / "04_analysis"
    if out.exists():
        shutil.rmtree(out)                                  # a fresh log each session
    ctx = Ctx(rd, ctx_plan)
    AP.write_plan(ctx, out, client=Fake(plan))
    S3.rule_plan_follows_question(ctx, out, client=Fake({"follows": True, "why": "yes"}))
    assert AP.plan_passed(out)[0]
    spec = AP.access_spec(ctx)
    return ctx, out, AP.make_access(spec, script=script), spec


def _ctx_plan(question, runs, years, variables):
    return {"question": question, "goals": [question],
            "experiment": {"control": {"name": "naches", "reuse": True},
                           "runs": [{"run": n, "changes": [c]} for n, c in runs.items()],
                           "years": years,
                           "analysis": {"normal_years": NORMAL, "question_type": "exploratory",
                                        "regions": [{"name": REGION}], "variables": variables}}}


def _row(res, field, period):
    """A logged screen row's path prefix ("rows.12.")."""
    return next(k[:-len("field")] for k, v in res.items()
                if k.startswith("rows.") and k.endswith(".field") and v == field
                and res.get(k[:-len("field")] + "period") == period)


def _period_words(label: str) -> str:
    kind, y, part = M.parse_period(label)
    if kind == "month":
        return f"{CT.MONTHS[int(part) - 1]} {y}"
    if kind == "season":
        return {"DJF": "winter", "MAM": "spring", "JJA": "summer", "SON": "autumn"}[part] + \
            f" {y}"
    if kind == "wateryear":
        return f"water year {y}"
    return str(y)


def _fmt(v, f):
    return format(int(v), "d") if f == "d" else format(float(v), f)


def _claims(out, spec):
    """Scripted claims, written from the logged records: each slot is (record,
    path, format), and each claim carries every warning its records carry."""
    recs = CT.evidence_records(out)
    log = CT._read_log(out)
    claims = []
    for c in spec:
        vals = [log[int(rid[1:]) - 1]["result"][path] for rid, path, _ in c["slots"]]
        codes = {w["code"] for i in c["ev"] for w in recs[i]["warnings"]}
        codes |= {w["code"] for i in c["ev"] for ws in recs[i]["row_warnings"].values()
                  for w in ws}
        text = c["t"].format(*(_fmt(v, f) for v, (_, _, f) in zip(vals, c["slots"])),
                             **c.get("named", {}))
        claims.append({"claim": text, "finding_id": "f", "values": [float(v) for v in vals],
                       "evidence": list(c["ev"]), "evidence_class": c["cls"],
                       "warnings": sorted(codes), **c.get("kw", {})})
    return claims


def _report(ctx, claims, headline):
    return S4.build(ctx, {}, {"findings": []},
                    {"verdict": "sufficient", "headline": headline,
                     "claims": copy.deepcopy(claims), "open_causes": []}, ctx.run_dir)


def _page(rep, question):
    return S4.render_markdown({**rep, "question": question, "model": "elm",
                               "status": "completed", "kind": "whole_domain",
                               "answer": "As the claims say."})


def _shown(display):
    rows = [r for g in display["groups"].values() for r in g]
    return rows + list(display["events"]) + list(display["not_scored"])


def _untraced(held, logged):
    """The rows holding a value whose logged copy (the lookup's record, the
    evidence the trace reads) differs from what the screen returned."""
    out = []
    for i, r in enumerate(held):
        for k in DACC.SHOWN_ROW:
            got, want = logged.get(f"rows.{i}.{k}"), r.get(k)
            if not (got == want or (got is None and want is None)):
                out.append((i, r.get("field"), r.get("period"), k, got, want))
    return out


# ── the warm, dry summer's study (tests 1 and 2) ─────────────────────────────
PLAN_WARM = {
    "comparisons": [{"id": "C1", "what": "the warm, dry summer against the control",
                     "run": WARM_RUN, "against": "control", "years": [2015]}],
    "baseline": {"normal_years": NORMAL, "why": "the Planner's"},
    "analyses": [{"id": "P1", "question": "what changed in the run", "fields": ["GPP"],
                  "regions": [REGION], "periods": ["2015"], "methods": ["difference"],
                  "settings": {}, "why": "after the screen",
                  # its required output, apart from its scope (2026-11-08)
                  "outputs": [{"id": "O1", "kind": "difference", "run": WARM_RUN,
                               "field": "GPP", "period": "2015", "region": REGION}]}],
    "expectations": [], "competing": [],
    "screen": {"when": "first", "region": REGION, "year": 2015,
               "why": "an exploratory question"}}
Q_WARM = "What changed over the watershed in the warm, dry summer run of 2015?"


@pytest.fixture(scope="module")
def warm_study(cfg, suites):
    rd = cfg["scratch"] / "study_warm"
    ctx, out, acc, spec = _study(
        rd, cfg["entry"], {WARM_RUN: (cfg["runs"][WARM_RUN], WARM, [2015])},
        _ctx_plan(Q_WARM, {WARM_RUN: WARM}, [2015], ["TBOT", "GPP"]), PLAN_WARM, "fig_warm")
    assert set(spec["suites"]) == {"control", WARM_RUN}, spec["suites"]
    res = acc.screen(2015, REGION, run=WARM_RUN)
    log = CT._read_log(out)
    assert len(log) == 1 and "sha256" in log[0] and log[0].get("label") == AP.EXPLORATORY_LABEL
    return {"rd": rd, "ctx": ctx, "out": out, "result": res, "log": log}


# ── the 2022 exploratory study (tests 4 and 5) ───────────────────────────────
PLAN_2022 = {
    "comparisons": [{"id": "C1", "what": "2022 against the normal years", "run": "control",
                     "against": "normal years", "years": [2022]}],
    "baseline": {"normal_years": NORMAL, "why": "the Planner's"},
    "analyses": [{"id": "P1", "question": "when photosynthesis started in 2022", "fields": [],
                  "regions": [REGION], "periods": ["2022"],
                  "methods": ["photosynthesis_onset"], "settings": {},
                  "why": "a timing measure whose shift can move with its setting",
                  "outputs": [{"id": "O1", "kind": "photosynthesis_onset", "region": REGION,
                               "period": "2022"}]}],
    "expectations": [], "competing": [],
    "screen": {"when": "first", "region": REGION, "year": 2022,
               "why": "an exploratory question"}}
Q_2022 = "What changed in 2022 compared with the normal years?"


@pytest.fixture(scope="module")
def study_2022(cfg):
    rd = cfg["scratch"] / "study_2022"
    ctx, out, acc, spec = _study(rd, cfg["entry"], {},
                                 _ctx_plan(Q_2022, {}, [2022], ["GPP"]), PLAN_2022, "fig_2022")
    assert set(spec["suites"]) == {"control"}, spec["suites"]
    res = acc.screen(2022, REGION)                       # first: the exploratory screen
    onset = acc.photosynthesis_onset(2022, REGION)
    log = CT._read_log(out)
    assert [x["call"] for x in log] == ["screen", "photosynthesis_onset"], log
    assert all("sha256" in x and not x.get("refused") for x in log)
    lr = log[0]["result"]
    groups = list(res["display"]["groups"])[:2]          # the first two groups shown
    spec_c = []
    for i, g in enumerate(groups):
        top = res["display"]["groups"][g][0]
        p = _row(lr, top["field"], top["period"])
        spec_c.append({
            "t": "Exploratory: in the 2022 screen over the watershed, {field}'s "
                 "{when} row scored {} against {} normal years, a change of {} {units}.",
            "named": {"field": top["field"], "when": _period_words(top["period"]),
                      "units": top.get("change_units") or ""},
            "slots": [("L1", p + "score", ".4g"), ("L1", p + "n_normal", "d"),
                      ("L1", p + "change", ".4g")],
            "ev": ["L1"], "cls": "exploratory", "kw": {"headline": i == 0}})
    tr = log[1]["result"]
    alts = sorted((int(k.split(".")[1]), v) for k, v in tr.items()
                  if k.startswith("sensitivity.") and k.endswith(".variant"))
    sl = [("L2", "variants.default.shift.median", ".4g")] + \
        [("L2", f"sensitivity.{i}.median_shift", ".4g") for i, _ in alts]
    shifts = [tr[p] for _, p, _ in sl]
    frac = {lab: tr[f"variants.{lab}.settings.fraction"] for _, lab in alts}
    lo, hi = min(shifts), max(shifts)
    words = ", ".join(f"by {{}} days at fraction {M.format_value(frac[lab])}" for _, lab in alts)
    with_range = {
        "t": "Over the watershed, photosynthesis onset in 2022 shifted by a median of {} days "
             "against the normal years at the default fraction 0.2, " + words +
             ": across the settings the shift ranges from {} to {} days.",
        "slots": sl + [("L2", sl[shifts.index(lo)][1], ".4g"),
                       ("L2", sl[shifts.index(hi)][1], ".4g")],
        "ev": ["L2"], "cls": "observation"}
    without = {"t": "Over the watershed, photosynthesis onset in 2022 shifted by a median of {} "
                    "days against the normal years.",
               "slots": sl[:1], "ev": ["L2"], "cls": "observation"}
    claims = _claims(out, spec_c + [with_range])
    lone = _claims(out, [with_range, without])
    return {"rd": rd, "ctx": ctx, "out": out, "screen": res, "onset": onset, "log": log,
            "claims": claims, "pair": lone, "shifts": dict(zip(
                ["default"] + [lab for _, lab in alts], shifts))}


@pytest.fixture(scope="module")
def report_2022(study_2022):
    c = study_2022["claims"]
    rep = _report(study_2022["ctx"], c, c[0]["claim"])
    return {"rep": rep, "page": _page(rep, Q_2022)}


# ── the control's daily history, read once per year (test 3) ────────────────
@pytest.fixture(scope="module")
def daily(cfg):
    """Per year, from the control's daily history (h1): each vegetated
    watershed cell's TLAI peak, its low before the peak and whether every day
    holds a value; each vegetated cell outside the watershed's largest H2OSNO
    over January-September and over October-December, with the same
    completeness. Worked out here, apart from the methods."""
    ctl = cfg["control"]
    ws = np.flatnonzero(ctl.mask(REGION).ravel())
    oc = np.flatnonzero((ctl.mask(VEG) & ~ctl.mask(REGION)).ravel())
    s = G.open_stream(cfg["entry"]["run_dir"], "h1")
    leaf, snow = {}, {}
    for y in YEARS:
        a = M.read_year(s, y, ["TLAI", "H2OSNO"])
        x = a["TLAI"][:, ws]
        fin = np.isfinite(x)
        xx = np.where(fin, x, -np.inf)
        peak, at = xx.max(axis=0), xx.argmax(axis=0)
        idx = np.arange(x.shape[0])[:, None]
        low = np.where((idx <= at[None, :]) & fin, x, np.inf).min(axis=0)
        leaf[y] = {"complete": fin.all(axis=0), "peak": peak, "low": low}
        h = a["H2OSNO"][:, oc]
        k = sum(G.days_in_month(y, m, s.calendar) for m in range(1, 10))   # to 30 September
        part = {}
        for name, b in (("js", h[:k]), ("od", h[k:])):
            f = np.isfinite(b)
            part[name + "_max"] = np.where(f, b, -np.inf).max(axis=0)
            part[name + "_ok"] = f.all(axis=0)
        snow[y] = part
        del a, x, h
    return {"ws": ws, "out": oc, "leaf": leaf, "snow": snow, "n": ctl.weights.size}


# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT TESTS (tests 2-4): calculations and eligibility behave as the
# contracts say
# ─────────────────────────────────────────────────────────────────────────────

# ── test 2: no-change cases give no finding ──────────────────────────────────
def test_contract_2_warm_dry_summer_outside_the_watershed_gives_no_finding(cfg, suites):
    """The warm, dry summer over the vegetated cells outside the watershed (a
    region made here from the registry's masks). DECISION 3 (the user,
    2026-11-08): "no change" is required only for the periods available and
    comparable in both runs (both suites hold the period); a period missing
    from either run (water year 2015 and winter 2015, which start before
    1 January 2015) stays "period absent", with no score and no finding. No
    row is scored or shown."""
    run, ctl = M.Suite(suites["warm_plus"]), M.Suite(suites["control_plus"])
    want = cfg["control"].mask(VEG) & ~cfg["control"].mask(REGION)
    assert (ctl.mask(OUTSIDE_REGION) == want).all() and (run.mask(OUTSIDE_REGION) == want).all()
    held = [{p for v in s.manifest["periods"].values() for p in v} for s in (run, ctl)]
    both = [p for p in M.screen_periods(2015) if all(p in h for h in held)]
    missing = [p for p in M.screen_periods(2015) if p not in both]
    res = M.screen(run, OUTSIDE_REGION, 2015, NORMAL, control=ctl)
    rows = res["rows"]
    shown = _shown(res["display"])
    comparable = [r for r in rows if r["period"] in both]
    absent = [r for r in rows if r["period"] in missing]
    other = [(r["field"], r["period"], r["status"], r["change"]) for r in comparable
             if r["status"] != "no change"]
    absent_wrong = [(r["field"], r["period"], r["status"], r.get("score"), r.get("mark"))
                    for r in absent if r["status"] != "period absent"
                    or r.get("score") is not None or r.get("mark") not in (None, "")]
    absent_shown = [(r["field"], r["period"]) for r in shown if r["period"] in missing]
    _note(cfg, "test2.outside", {
        "cells": int(want.sum()), "rows": len(rows),
        "periods_in_both_runs": both, "periods_missing": missing,
        "rows_in_periods_both_hold": len(comparable),
        "no_change": sum(r["status"] == "no change" for r in comparable),
        "period_absent_rows": sum(r["status"] == "period absent" for r in absent),
        "missing_rows_not_absent_or_scored_or_marked": absent_wrong[:20],
        "other_statuses": other[:20], "n_other": len(other),
        "scored": sum(r.get("score") is not None for r in rows),
        "displayed": len(shown), "missing_rows_displayed": absent_shown})
    assert [(r["field"], r["period"]) for r in rows] == \
        [(f, p) for f in FIELDS for p in M.screen_periods(2015)]
    assert set(missing) == BEFORE_2015 and len(absent) == len(FIELDS) * len(BEFORE_2015)
    # the periods available and comparable in both runs: every row no change
    assert not other, f"{len(other)} rows are not 'no change'; first: {other[:5]}"
    # the missing periods: "period absent", with no score and no finding
    assert not absent_wrong, absent_wrong[:5]
    assert not absent_shown, absent_shown[:5]
    assert all(r.get("score") is None for r in rows)
    assert shown == []


def test_contract_2_warm_dry_summer_watershed_january_to_may_gives_no_finding(cfg, warm_study):
    """The warm, dry summer over the watershed in January-May (the months
    1-5 and spring, the periods inside January-May): every row marked no
    change, none scored or displayed."""
    res = warm_study["result"]
    rows = [r for r in res["rows"] if r["period"] in JAN_MAY_2015]
    other = [(r["field"], r["period"], r["status"], r["change"]) for r in rows
             if r["status"] != "no change"]
    shown = [(r["field"], r["period"]) for r in _shown(res["display"])
             if r["period"] in JAN_MAY_2015]
    _note(cfg, "test2.watershed_jan_may", {
        "rows": len(rows), "no_change": len(rows) - len(other), "not_no_change": other[:40],
        "n_not_no_change": len(other), "displayed": shown})
    assert len(rows) == len(FIELDS) * len(JAN_MAY_2015)
    assert not other, f"{len(other)} rows are not 'no change'; first: {other[:8]}"
    assert all(r.get("score") is None for r in rows)
    assert not shown


def test_contract_2_a_run_compared_with_itself_gives_a_complete_table_of_no_change(cfg):
    """The control compared with itself, every year whose 18 periods it
    holds (2016-2023): a complete table, every row no change, nothing shown;
    once through the `suite` door as well."""
    s = cfg["control"]
    seen, bad = {}, []
    for y in YEARS[1:]:
        res = M.screen(s, REGION, y, NORMAL, control=s)
        rows = res["rows"]
        other = [(r["field"], r["period"], r["status"]) for r in rows
                 if r["status"] != "no change"]
        complete = [(r["field"], r["period"]) for r in rows] == \
            [(f, p) for f in FIELDS for p in M.screen_periods(y)]
        seen[y] = {"rows": len(rows), "complete": complete, "not_no_change": other[:10],
                   "n_not_no_change": len(other),
                   "scored": sum(r.get("score") is not None for r in rows),
                   "displayed": len(_shown(res["display"]))}
        if not complete or other or seen[y]["scored"] or seen[y]["displayed"]:
            bad.append(y)
    acc = _open({"control": s.dir, "itself": s.dir})
    res = acc.screen(2022, REGION, run="itself")
    seen["2022 through the suite door"] = {
        "rows": len(res["rows"]), "no_change": sum(r["status"] == "no change"
                                                   for r in res["rows"])}
    _note(cfg, "test2.itself", seen)
    assert not bad, {y: seen[y] for y in bad}
    assert len(res["rows"]) == len(FIELDS) * 18
    assert all(r["status"] == "no change" for r in res["rows"])


# ── test 3: inapplicable methods fail explicitly ─────────────────────────────
def test_contract_3_leaf_green_up_below_the_mark_gives_a_reason_and_counts(cfg, daily):
    """Where a cell's leaf rise in a year is below the mark (worked out here
    from the daily TLAI), leaf_green_up gives "leaf area changes too little",
    never a date, at the default mark and at its two alternatives; and the
    region summary over those cells gives the reason and its count, no date."""
    acc = _open({"control": cfg["control"].dir})
    w = cfg["control"].weights
    counts, bad = {}, []
    for y in YEARS:
        L = daily["leaf"][y]
        rise = L["peak"] - L["low"]
        by = _by_year(acc, "control", "leaf_green_up", y)
        for mark in (0.5, 0.25, 0.75):
            lab = _variant(by, fraction=0.2, rise_mark=mark)
            below = L["complete"] & (L["peak"] > 0) & ~(rise >= mark * L["peak"])
            cells = daily["ws"][below]
            v = by["variants"][lab]
            counts.setdefault(lab, {})[y] = int(cells.size)
            if not (v["reason"][cells] == M.CODE["little_leaf_change"]).all() or \
                    not np.isnan(v["date"][cells]).all():
                bad.append((y, lab, "a cell below the mark has a date or another reason"))
            if cells.size:
                t = M.timing_summary("leaf_green_up", {y: by}, y,
                                     _grid_mask(daily["n"], cells), w)
                d = t["variants"][lab]["distribution"]
                if not (d["cells"] == cells.size and d["dates"] == 0 and d["median"] is None
                        and d["reasons"] == {"little_leaf_change": int(cells.size)}
                        and d["date_area_share"] == 0.0):
                    bad.append((y, lab, d["dates"], d["reasons"], d["median"]))
    _note(cfg, "test3.leaf_below_mark_cells", {"counts": counts, "problems": bad})
    assert not bad, bad[:5]
    assert sum(counts["default"].values()) > 0


def _snow_period(x):
    return M.DailyPeriod(kind="wateryear", year=2016, first_day=-91, data={"H2OSNO": x},
                         shape=(1, x.shape[1]))


def test_contract_3_snow_measures_on_synthetic_cells_that_never_hold_1_mm():
    """Synthetic cells that never hold 1 mm (no snow, a 0.5 mm peak, 0.999 mm
    all year) beside one with a snowpack: snow_peak and snow_gone give "no
    snow" and no date at every stored setting, and the region summary the
    reason and its count."""
    n = 365
    d = np.arange(n, dtype=np.float64)
    x = np.zeros((n, 4))
    x[:, 1] = 0.5 * np.sin(np.pi * d / n)
    x[:, 2] = 0.999
    x[:, 3] = np.interp(d, [0, 60, 150, 220, 364], [0.0, 0.0, 200.0, 0.0, 0.0])
    p = _snow_period(x)
    mask, w = np.array([True, True, True, False]), np.ones(4)
    for mid in ("snow_peak", "snow_gone"):
        res = M.METHODS[mid].run(p)
        assert len(res["variants"]) == len(M.METHODS[mid].variants())
        for lab, v in res["variants"].items():
            assert (v["reason"][:3] == M.CODE["no_snow"]).all(), (mid, lab, v["reason"])
            assert np.isnan(v["date"][:3]).all(), (mid, lab)
            assert v["reason"][3] == M.CODE["date"] and np.isfinite(v["date"][3]), (mid, lab)
        t = M.timing_summary(mid, {2016: res}, 2016, mask, w)
        for lab, rec in t["variants"].items():
            dd = rec["distribution"]
            assert dd["dates"] == 0 and dd["reasons"] == {"no_snow": 3}, (mid, lab, dd)
            assert dd["median"] is None and dd["q25"] is None and dd["q75"] is None


def test_contract_3_snow_measures_outside_the_watershed_where_cells_never_hold_1_mm(cfg, daily):
    """Vegetated cells outside the watershed that never hold 1 mm (or 5 mm,
    the alternative threshold) in a water year, from the daily H2OSNO: no
    snow and no date at that threshold, and the region summary gives the
    reason and its count. Which cells these are is reported, not required."""
    acc = _open({"control": cfg["control"].dir})
    w = cfg["control"].weights
    counts, bad = {}, []
    for y in YEARS[1:]:                       # water years 2016-2023: October-December held
        a, b = daily["snow"][y - 1], daily["snow"][y]
        big = np.maximum(a["od_max"], b["js_max"])
        ok = a["od_ok"] & b["js_ok"]
        for mid in ("snow_peak", "snow_gone"):
            by = _by_year(acc, "control", mid, y)
            for lab, v in by["variants"].items():
                thr = v["settings"]["threshold_mm"]
                cells = daily["out"][ok & (big < thr)]
                counts.setdefault(f"{mid} {lab}", {})[y] = int(cells.size)
                if not (v["reason"][cells] == M.CODE["no_snow"]).all() or \
                        not np.isnan(v["date"][cells]).all():
                    bad.append((mid, lab, y, "a snow-free cell has a date or another reason"))
                if cells.size:
                    t = M.timing_summary(mid, {y: by}, y, _grid_mask(daily["n"], cells), w)
                    dd = t["variants"][lab]["distribution"]
                    if not (dd["dates"] == 0 and dd["median"] is None
                            and dd["reasons"] == {"no_snow": int(cells.size)}):
                        bad.append((mid, lab, y, dd["dates"], dd["reasons"]))
    _note(cfg, "test3.snow_free_cells_outside", {"counts": counts, "problems": bad})
    assert not bad, bad[:5]
    assert sum(counts["snow_peak default"].values()) > 0, \
        "no vegetated cell outside the watershed is snow-free in any water year: the case " \
        "cannot run on real cells"


def test_contract_3_snow_peak_in_a_water_year_whose_october_to_december_the_run_lacks(cfg):
    """snow_peak in water year 2015 (the control starts on 1 January 2015):
    a reason ("before the record") and its count for every cell, never a
    date, over the watershed and over every vegetated cell."""
    acc = _open({"control": cfg["control"].dir})
    got, bad = {}, []
    v = acc.get_timing("control", "snow_peak", 2015)["default"]
    for reg in (REGION, VEG):
        r = acc.snow_peak(2015, reg)
        d = r["variants"]["default"]["distribution"]
        sh = r["variants"]["default"].get("shift")
        m = cfg["control"].mask(reg).ravel()
        got[reg] = {"variants": sorted(r["variants"]), "cells": d["cells"], "dates": d["dates"],
                    "reasons": d["reasons"], "median": d["median"],
                    "missing_time_records": d["missing_time_records"],
                    "shift_cells": None if sh is None else sh["cells"],
                    "cells_with_a_date": int(np.isfinite(v["date"][m]).sum())}
        if not (set(r["variants"]) == {"default"} and d["dates"] == 0 and d["median"] is None
                and d["date_area_share"] == 0.0
                and d["reasons"] == {"before_record": d["cells"]}
                and d["missing_time_records"]["cells"] == d["cells"]
                and (sh is None or (sh["cells"] == 0 and sh["median"] is None))
                and (v["reason"][m] == M.CODE["before_record"]).all()
                and np.isnan(v["date"][m]).all()):
            bad.append(reg)
    _note(cfg, "test3.snow_peak_wy2015", got)
    assert not bad, {k: got[k] for k in bad}


# ── test 4: stability under setting changes ──────────────────────────────────
SENS_KEYS = {"variant", "settings", "median_date", "default_median_date", "median_shift",
             "default_median_shift", "median_cell_change", "cells_with_both",
             "cell_change_area_share", "cell_change_area_km2", "eligible"}


def _eligible_ok(e, names, cells) -> bool:
    return (set(e) == set(names) | {"region_area_km2"}
            and sum(e[k]["cells"] for k in names) == cells
            and abs(sum(e[k]["area_share"] for k in names) - 1.0) < 1e-9)


def test_contract_4_every_timing_result_carries_its_sensitivity_record(cfg, suites):
    """snow_gone across 1 to 5 mm and 7 to 14 days, photosynthesis_onset
    across 10 % to 30 %, and every other measure: every timing result (the
    control's years against the normal years, each experiment run's years
    against the control) carries the sensitivity record of 3.4 with the
    change of eligibility between settings, and each setting's shift its
    joint eligibility. The cases whose median shift moves by more than a week
    between settings are listed."""
    stored = {mid: dict(d["variants"])
              for mid, d in M.Suite(suites["control"]).manifest["timing"]["measures"].items()}
    assert {v["threshold_mm"] for v in stored["snow_gone"].values()} == {1.0, 5.0}
    assert {v["days"] for v in stored["snow_gone"].values()} == {7, 14}
    assert {v["fraction"] for v in stored["photosynthesis_onset"].values()} == {0.1, 0.2, 0.3}
    acc = _open({k: suites[k] for k in ("control", WARM_RUN, "nofire", "dry")})
    calls = [("control", y, "normal") for y in YEARS] + \
        [(WARM_RUN, 2015, "control"), ("nofire", 2015, "control"), ("nofire", 2016, "control"),
         ("dry", 2016, "control")]
    moves, n, bad = [], 0, []
    for mid in M.TIMING_IDS:
        alts = set(stored[mid]) - {"default"}
        for run, y, against in calls:
            r = getattr(acc, mid)(y, REGION, run=run, against=against)
            n += 1
            where = (mid, run, y)
            if set(r["variants"]) != set(stored[mid]) or \
                    {s["variant"] for s in r["sensitivity"]} != alts:
                bad.append(where + ("variants or sensitivity records missing",))
            cells = r["variants"]["default"]["distribution"]["cells"]
            for s in r["sensitivity"]:
                if set(s) != SENS_KEYS or not _eligible_ok(
                        s["eligible"], ("both", "only_default", "only_alternative", "neither"),
                        cells):
                    bad.append(where + (s["variant"], "sensitivity record incomplete"))
                a, b = s["default_median_shift"], s["median_shift"]
                if a is not None and b is not None and abs(b - a) > WEEK and \
                        mid in ("snow_gone", "photosynthesis_onset"):
                    moves.append({"measure": mid, "run": run, "year": y, "against": against,
                                  "variant": s["variant"], "default_shift": a, "shift": b})
            for lab, v in r["variants"].items():
                if "shift" not in v or not _eligible_ok(
                        v["shift"]["eligible"], ("both", "only_this", "only_reference",
                                                 "neither"), cells):
                    bad.append(where + (lab, "shift or its joint eligibility missing"))
    _note(cfg, "test4.moves_over_a_week", moves)
    _note(cfg, "test4.timing_results_checked", {"n": n, "problems": bad})
    assert not bad, bad[:5]


def test_contract_4_the_report_cites_the_range_when_a_shift_moves_more_than_a_week(
        cfg, study_2022, report_2022):
    """photosynthesis_onset in 2022 moves by more than a week between its
    settings (3.4: from -3 to +15 days): the scripted report's claim gives the
    range across the settings, and it passes the trace and the numbers
    check."""
    sh = study_2022["shifts"]
    _note(cfg, "test4.onset_2022_shifts", sh)
    assert max(abs(v - sh["default"]) for v in sh.values()) > WEEK, sh
    rep = report_2022["rep"]
    t, nc = rep["evidence_trace"], rep["evidence_numbers"]
    i = next(k for k, c in enumerate(rep["claims"], 1) if c["evidence"] == ["L2"])
    assert i not in t["claims_failed"], t["problems"].get(i)
    assert nc["pass"], [(m["record"], m["path"], m["logged"], m["recomputed"])
                        for m in nc.get("mismatches", [])[:8]]
    assert "L2" in nc["records"]


def test_contract_4_the_trace_rejects_a_claim_without_the_range(cfg, study_2022):
    """The same shift stated at the default setting alone, with every warning
    its record carries: the trace must reject it, since the shift moves by
    more than a week between settings and the claim gives no range."""
    pair = copy.deepcopy(study_2022["pair"])
    t = CT.trace_evidence(pair, None, study_2022["out"], study_dir=study_2022["rd"])
    _note(cfg, "test4.trace_of_claim_without_range", {
        "claims": [c["claim"] for c in pair], "claims_failed": t["claims_failed"],
        "problems": {str(k): v for k, v in t["problems"].items()}})
    assert 1 not in t["claims_failed"], t["problems"].get(1)
    assert t["claims_failed"] == [2], \
        "the trace accepted a claim that states the 2022 onset shift at the default setting " \
        "alone, though the shift moves by more than a week between settings"


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION TESTS (tests 1 and 5, and the October values): these runs keep
# their expected outputs
# ─────────────────────────────────────────────────────────────────────────────

# ── test 1: known signals are found ──────────────────────────────────────────
def test_regression_1_warm_dry_summer_imposed_warming_is_in_the_table_and_the_display(
        cfg, warm_study):
    """The imposed +2 K (June-August, over the watershed): TBOT's rows for
    June, July, August and summer 2015 change by 2 K, and TBOT is in the
    weather group's display with that change."""
    res = warm_study["result"]
    tb = {r["period"]: r for r in res["rows"] if r["field"] == "TBOT"}
    shown = [r for r in res["display"]["groups"].get("weather in", []) if r["field"] == "TBOT"]
    _note(cfg, "test1.warm.tbot", {
        "rows": {p: {k: tb[p][k] for k in ("change", "status", "score")} for p in tb},
        "weather_display": res["display"]["groups"].get("weather in")})
    for p in WARMED_2015:
        assert tb[p]["change"] is not None and abs(tb[p]["change"] - IMPOSED_K) <= EXACT_K, \
            (p, tb[p]["change"])
    assert shown, res["display"]["groups"].get("weather in")
    assert abs(shown[0]["change"] - IMPOSED_K) <= EXACT_K, shown[0]


def test_regression_1_warm_dry_summer_responses_are_in_the_table_with_traced_values(
        cfg, warm_study):
    """Its responses are in the table with traced values: every row that holds
    a value is in the lookup's record as the screen returned it, fields beyond
    the weather input change, and a scripted report citing the imposed
    warming, the drier summer and two responses passes the evidence trace and
    the numbers check (every cited number made again from the history)."""
    res, log = warm_study["result"], warm_study["log"]
    lr = log[0]["result"]
    held = [r for r in res["rows"] if r.get("value") is not None or r.get("change") is not None]
    untraced = _untraced(held, lr)
    resp = [r for r in held if r["field"] not in WEATHER and r["change"] not in (None, 0.0)]
    spec = []
    for f in ("TBOT", "RAIN", "GPP", "QVEGT"):
        p = _row(lr, f, "2015-JJA")
        pct = lr.get(p + "percent")
        slots = [("L1", p + "change", ".4g")]
        t = ("Exploratory: in the 2015 screen over the watershed, {field} of the "
             "warm_dry_summer run in summer 2015 differed from the control's by {} {units}")
        if f != "TBOT" and isinstance(pct, float):
            slots.append(("L1", p + "percent", ".3g"))
            t += " ({} %)"
        spec.append({"t": t + ".", "named": {"field": f, "units": lr.get(p + "change_units")},
                     "slots": slots, "ev": ["L1"], "cls": "exploratory",
                     "kw": {"headline": f == "TBOT"}})
    claims = _claims(warm_study["out"], spec)
    rep = _report(warm_study["ctx"], claims, claims[0]["claim"])
    t, nc = rep["evidence_trace"], rep["evidence_numbers"]
    page = _page(rep, Q_WARM)
    _note(cfg, "test1.warm.responses", {
        "rows_with_values": len(held), "untraced": untraced[:10],
        "changed_beyond_weather": len(resp),
        "largest_scores": sorted(((r["field"], r["period"], r["score"]) for r in resp
                                  if r.get("score") is not None),
                                 key=lambda x: -abs(x[2]))[:8],
        "claims": [c["claim"] for c in claims], "trace_failed": t["claims_failed"],
        "trace_problems": {str(k): v for k, v in t["problems"].items()},
        "numbers_compared": nc.get("n_compared"), "numbers_pass": nc.get("pass"),
        "withheld": rep.get("withheld_because")})
    assert not untraced, untraced[:5]
    assert resp, "no field beyond the weather input changed"
    assert t["claims_failed"] == [] and t["headline"]["problems"] == [], t["problems"]
    assert nc["pass"], [(m["record"], m["path"], m["logged"], m["recomputed"])
                        for m in nc.get("mismatches", [])[:8]]
    assert not rep.get("withheld_because")
    assert not page.startswith("# WITHHELD") and "**Numbers check: passed.**" in page


def test_regression_1_nofire_event_rows_differ_from_control_where_the_control_had_fire(
        cfg, suites):
    """The no-fire run, 2015 and 2016, over the watershed and over every
    vegetated cell: every event row whose period had fire in the control is
    marked differs_from_control, and shown among the display's events."""
    run, ctl = M.Suite(suites["nofire"]), cfg["control"]
    seen, total = {}, 0
    for y in (2015, 2016):
        for reg in (REGION, VEG):
            res = M.screen(run, reg, y, NORMAL, control=ctl)
            ev = [r for r in res["rows"] if r["field"] in EVENTS]
            had = [r for r in ev if r["reference"] is not None and r["reference"] > 0]
            wrong = [(r["field"], r["period"], r["status"], r["mark"]) for r in had
                     if not (r["status"] == "marked" and r["mark"] == "differs_from_control")]
            shown = {(r["field"], r["period"]) for r in res["display"]["events"]}
            seen[f"{y} {reg}"] = {
                "event_rows": len(ev), "control_had_fire": len(had), "wrong": wrong,
                "not_shown": sorted({(r["field"], r["period"]) for r in had} - shown),
                "other_statuses": sorted({r["status"] for r in ev if r not in had}),
                "run_values_nonzero": sum(1 for r in ev if r["value"] not in (None, 0.0))}
            total += len(had)
    _note(cfg, "test1.nofire", seen)
    assert all(not v["wrong"] and not v["not_shown"] for v in seen.values()), seen
    assert total > 0, "the control had no fire in these years and regions"


def test_regression_1_dry_weather_input_for_2016_matches_the_controls_2015_but_1_january(cfg):
    """The repeated dry year: its weather input for 2016 (TBOT, RAIN and SNOW
    day by day, cell by cell, from the daily history; FSDS, held monthly
    only, month by month) equals the control's 2015 on every day but 1
    January, where one time step is carried over from the year before."""
    ctl_run, dry_run = cfg["entry"]["run_dir"], cfg["runs"]["dry"]
    sc, sd = G.open_stream(ctl_run, "h1"), G.open_stream(dry_run, "h1")
    daily_f = [f for f in WEATHER if f in sc.variables]
    a, b = M.read_year(sc, 2015, daily_f), M.read_year(sd, 2016, daily_f)
    out = {}
    for f in daily_f:
        same = (a[f] == b[f]) | (np.isnan(a[f]) & np.isnan(b[f]))
        days = (np.flatnonzero(~same.all(axis=1)) + 1).tolist()
        d0 = np.abs(a[f][0] - b[f][0])
        out[f] = {"days_that_differ": days, "cells_on_1_january": int((~same[0]).sum()),
                  "largest_difference_on_1_january": float(np.nanmax(d0)) if
                  np.isfinite(d0).any() else None, "units": sc.units(f)}
    del a, b
    mc, md = G.open_stream(ctl_run, "h0"), G.open_stream(dry_run, "h0")
    monthly = {}
    for f in WEATHER:
        x, y = mc.read(f, 2015).data, md.read(f, 2016).data
        same = (x == y) | (np.isnan(x) & np.isnan(y))
        monthly[f] = (np.flatnonzero(~same.reshape(same.shape[0], -1).all(axis=1)) + 1).tolist()
    _note(cfg, "test1.dry.weather", {"daily": out, "months_that_differ": monthly})
    assert set(daily_f) == {"TBOT", "RAIN", "SNOW"}, daily_f
    for f in daily_f:
        assert set(out[f]["days_that_differ"]) <= {1}, (f, out[f]["days_that_differ"][:10])
    assert set().union(*(set(out[f]["days_that_differ"]) for f in daily_f)) == {1}
    for f in WEATHER:
        assert set(monthly[f]) <= {1}, (f, monthly[f])


def test_regression_1_dry_snow_gone_is_earlier_than_the_controls_2016(cfg, suites):
    """The repeated dry year's snow melt-out (water year 2016) against the
    control's same year over the watershed: earlier. The direction is tested;
    the trial measured a median of 40 days."""
    acc = _open({"control": suites["control"], "dry": suites["dry"]})
    r = acc.snow_gone(2016, REGION, run="dry", against="control")
    got = {lab: {"median_shift": v["shift"]["median"], "q25": v["shift"]["q25"],
                 "q75": v["shift"]["q75"], "both": v["shift"]["eligible"]["both"]["cells"]}
           for lab, v in r["variants"].items()}
    _note(cfg, "test1.dry.snow_gone_shift_days", got)
    assert got["default"]["median_shift"] is not None and got["default"]["median_shift"] < 0


# ── test 5: discoverable without a prescribed rank ───────────────────────────
def test_regression_5_the_2022_screen_holds_every_field_and_period_with_traced_values(
        cfg, study_2022):
    """"What changed in 2022?" against 2016-2020: the screen's full table
    holds every catalog field and period, and every row that holds a value is
    in the lookup's record (the evidence the claim trace reads) as the screen
    returned it; its display rule is stated beside the table, in the screen's
    record too."""
    res, log, out = study_2022["screen"], study_2022["log"], study_2022["out"]
    rows = res["rows"]
    held = [r for r in rows if r.get("value") is not None or r.get("change") is not None]
    untraced = _untraced(held, log[0]["result"])
    rec = CT.evidence_records(out)["L1"]
    sr = json.loads((out / AP.SCREEN_FILE).read_text())
    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    _note(cfg, "test5.table", {"rows": len(rows), "rows_with_values": len(held),
                               "rows_in_the_evidence_record": len(
                                   [k for k in rec["rows"] if k.startswith("rows.")]),
                               "untraced": untraced[:10], "statuses": statuses,
                               "shown": [(r["field"], r["period"], r.get("score"))
                                         for r in _shown(res["display"])]})
    assert [(r["field"], r["period"]) for r in rows] == \
        [(f, p) for f in FIELDS for p in M.screen_periods(2022)]
    assert len(rows) == len(FIELDS) * 18
    assert all(r.get("status") for r in rows)
    assert not untraced, untraced[:5]
    assert len([k for k in rec["rows"] if k.startswith("rows.")]) == len(held)
    assert res["display_rule"] == M.DISPLAY_RULE and res["display"]["rule"] == M.DISPLAY_RULE
    assert sr["display_rule"] == M.DISPLAY_RULE and sr["scoring_rule"]
    assert sr["scope"]["rows"] == len(rows) and sr["scope"]["fields"] == FIELDS
    assert log[0]["label"] == AP.EXPLORATORY_LABEL and log[0]["planned"] is True


def test_regression_5_the_report_labels_screen_findings_exploratory(cfg, study_2022,
                                                                    report_2022):
    """A scripted report of two screen findings (the top rows of the first
    two groups shown) and the timing claim: it passes the trace and the
    numbers check; the page states the display rule beside the screen's
    table, labels the screen and its findings exploratory, and a headline
    from the screen without the label is rejected."""
    rep, page = report_2022["rep"], report_2022["page"]
    t, nc = rep["evidence_trace"], rep["evidence_numbers"]
    _note(cfg, "test5.report", {
        "claims": [c["claim"] for c in rep["claims"]], "trace_failed": t["claims_failed"],
        "trace_problems": {str(k): v for k, v in t["problems"].items()},
        "headline_problems": t["headline"]["problems"],
        "numbers_compared": nc.get("n_compared"), "numbers_pass": nc.get("pass"),
        "withheld": rep.get("withheld_because")})
    assert t["claims_failed"] == [] and t["headline"]["problems"] == [], t["problems"]
    assert nc["pass"], [(m["record"], m["path"], m["logged"], m["recomputed"])
                        for m in nc.get("mismatches", [])[:8]]
    assert not rep.get("withheld_because") and not page.startswith("# WITHHELD")
    head = f"## The screen ({AP.EXPLORATORY_LABEL})"
    rule = f"Display rule: {S4._plain(M.DISPLAY_RULE)}"
    table = "| shown | field | period | change | units | percent | score | status | mark |"
    a = page.index(head)
    assert a < page.index(rule, a) < page.index(table, a)
    assert f"Its findings are \"{AP.EXPLORATORY_LABEL}\"." in page
    for c in rep["claims"]:
        if c["evidence"] == ["L1"]:
            assert c["evidence_class"] == "exploratory"
            assert f"warning, {AP.EXPLORATORY_LABEL} (L1)" in page
    bad = copy.deepcopy(study_2022["claims"])
    unlabelled = bad[0]["claim"].replace("Exploratory: in", "In", 1)
    bad[0]["claim"] = unlabelled
    t2 = CT.trace_evidence(bad, unlabelled, study_2022["out"], study_dir=study_2022["rd"])
    assert any("is labelled exploratory" in p for p in t2["headline"]["problems"]), \
        t2["headline"]


# ── the October values (a regression check, not an answer key) ──────────────
QUOTED = {("GPP", "2015"): 1.1, ("GPP", "2021"): -17.8, ("GPP", "2022"): -5.9,
          ("GPP", "2015-07"): -18.7, ("GPP", "2021-07"): -26.8, ("GPP", "2022-05"): -22.5,
          ("H2OSNO", "2015-02"): -94.7, ("QRUNOFF", "2015-03"): -75.4}


def test_regression_the_october_values_over_the_watershed(cfg):
    """The values of section 7 for the watershed, vegetated region, from the
    2015-2023 regional table: the suite reproduces them to the printed
    precision."""
    got = {f"{f} {lab}": M.anomaly(cfg["control"], f, lab, NORMAL, region=REGION)["percent"]
           for f, lab in QUOTED}
    _note(cfg, "regression.october_values_percent", got)
    for (f, lab), pct in QUOTED.items():
        assert round(got[f"{f} {lab}"], 1) == pct, (f, lab, got[f"{f} {lab}"])
