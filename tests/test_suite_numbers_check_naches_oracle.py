"""The numbers check of a study with a suite, end to end on real data, no model (package F2).

    pytest tests/test_suite_numbers_check_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the files are on disk: the suite_f2 block of
tests/local_oracle_paths.json names a scratch folder (the study and the one
experiment run's suite are written there and nowhere else) and the run
directory of naches_bgc_fix_warm_dry_summer; the Naches control's registry
entry names its run, case and suite.

A SCRIPTED STUDY. A hypothesis question over the watershed: the plan is
written and reviewed by scripted fakes; the experiment run's 2015 suite is
built (once) with the server's own builder; the lookups go through the real
`suite` (the control's GPP anomaly over the watershed, the run's summer
difference, two maps and their overlap, the run's per-cell difference split
by elevation band, leaf green-up and snow melt-out against the normal years,
the run's growth peak against the control, then the screen); the claims are
a fake Analyzer's, each carrying its records' warnings; step 4 builds the
report, which runs the evidence trace and then the numbers check, reading the
history files only.

The valid study passes and publishes with every warning. Then one number at
a time is planted wrong in lookups.jsonl, and the claim that states it is
written again with the planted value and the planted record's warnings, so the
evidence trace still passes: only the numbers check can catch it, and it must
withhold the report naming the record, the value logged and the value made
again.
"""
import copy
import json
import os
import shutil
from pathlib import Path

import pytest

import domain_controls as DCT
import domain_suite as S
from test_domain_access import Ctx, Fake
from test_domain_catalog_runs import server_change

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import numbers_check as NC          # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402
from agents.analysis import step4_report as S4           # noqa: E402

pytestmark = pytest.mark.localdata
CONFIG = Path(__file__).with_name("local_oracle_paths.json")
RUN = "warm_dry_summer"
REGION = "watershed, vegetated"
NORMAL = [2016, 2017, 2018, 2019, 2020]
WARM = {"id": "weather_scale", "params": {"temperature_K": 2.0, "precip_factor": 0.8,
                                          "months": [6, 7, 8], "region": "watershed"}}
PLAN = {
    "comparisons": [{"id": "C1", "what": "2015 against the normal years", "run": "control",
                     "against": "normal years", "years": [2015]},
                    {"id": "C2", "what": "the warm, dry summer against the control", "run": RUN,
                     "against": "control", "years": [2015]}],
    "baseline": {"normal_years": NORMAL, "why": "the Planner's"},
    "analyses": [
        {"id": "P1", "question": "GPP in 2015", "fields": ["GPP"], "regions": [REGION],
         "periods": ["2015"], "methods": ["anomaly"], "settings": {}, "why": "asked",
         "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2015",
                      "region": REGION}]},
        {"id": "P2", "question": "the summer change", "fields": ["GPP"], "regions": [REGION],
         "periods": ["2015-JJA"], "methods": ["difference"], "settings": {}, "why": "the run",
         "outputs": [{"id": "O1", "kind": "difference", "run": RUN, "field": "GPP",
                      "period": "2015-JJA", "region": REGION}]},
        {"id": "P3", "question": "the soil water map", "fields": ["H2OSOI"], "regions": [],
         "periods": ["2015"], "methods": [], "settings": {}, "why": "to set beside GPP",
         "outputs": [{"id": "O1", "kind": "map", "field": "H2OSOI", "period": "2015"}]},
        {"id": "P4", "question": "where", "fields": [], "regions": [REGION], "periods": [],
         "methods": ["overlap", "elevation_split"], "settings": {}, "why": "where",
         "outputs": [{"id": "O1", "kind": "elevation_split", "region": REGION,
                      "inputs": ["P3.O1"]}]},
        {"id": "P5", "question": "when", "fields": [], "regions": [REGION], "periods": ["2015"],
         "methods": ["leaf_green_up", "snow_gone", "growth_peak"], "settings": {},
         "why": "when",
         "outputs": [{"id": "O1", "kind": "leaf_green_up", "region": REGION, "period": "2015"},
                     {"id": "O2", "kind": "snow_gone", "region": REGION, "period": "2015"},
                     {"id": "O3", "kind": "growth_peak", "region": REGION,
                      "period": "2015"}]}],
    "expectations": [{"id": "E1", "analysis": "P1", "field": "GPP", "sign": "higher",
                      "text": "GPP higher in 2015 than in the normal years"}],
    "competing": [{"text": "soil water", "fields": ["H2OSOI"], "analysis": "P1"}],
    "screen": {"when": "last", "region": REGION, "year": 2015, "why": "a hypothesis"}}
QUESTION = "Was GPP over the watershed higher in 2015 than in the normal years?"
OPEN = [{"cause": "whether the summer's warmth or its dryness lowered GPP",
         "proposed_run": {"changes": [{"id": "weather_scale",
                                       "params": {"temperature_K": 2.0, "months": [6, 7, 8],
                                                  "region": "watershed"}}]}}]


@pytest.fixture(scope="module")
def cfg():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    c = json.loads(CONFIG.read_text()).get("suite_f2") or {}
    if not c.get("scratch") or not Path(c.get("warm_run_dir") or "/nonexistent").is_dir():
        pytest.skip("no suite_f2 block with a scratch folder and the run directory")
    entry = (DCT.load().get("controls") or {}).get("naches") or {}
    if not entry.get("suite_dir") or not (Path(entry["suite_dir"]) / "suite.json").is_file():
        pytest.skip("the naches registry entry names no built suite")
    return {"scratch": Path(c["scratch"]), "warm": c["warm_run_dir"], "entry": entry}


def _ctx_plan():
    return {"question": QUESTION, "goals": ["GPP in 2015"],
            "experiment": {"control": {"name": "naches", "reuse": True},
                           "runs": [{"run": RUN, "changes": [WARM]}], "years": [2015],
                           "analysis": {"normal_years": NORMAL, "question_type": "hypothesis",
                                        "regions": [{"name": REGION}], "variables": ["GPP"]}}}


def _records_of_study(rd, entry, warm):
    """The manager's records the study's checks read, as the chain writes them,
    naming the real run directories; written only in the scratch folder."""
    (rd / "01_inputs").mkdir(parents=True, exist_ok=True)
    (rd / "01_inputs" / "domain_control_facts.json").write_text(json.dumps(
        {"control": {"name": "naches", "run_dir": entry["run_dir"],
                     "case_dir": entry["case_dir"]}}))
    (rd / "domain_case" / RUN / "checks").mkdir(parents=True, exist_ok=True)
    (rd / "domain_case" / RUN / "checks" / "y2015.json").write_text(json.dumps({"run_dir": warm}))
    (rd / "experiment.json").write_text(json.dumps({
        "kind": "whole_domain",
        "experiment": {"runs": [{"run": "control", "reused": True, "changes": []},
                                {"run": RUN, "changes": [server_change(WARM)]}]},
        "chain": {"years_planned": [2015], "years_passed": [2015],
                  "runs": {RUN: {"stages": [{"stage": "case", "check_pass": True},
                                            {"stage": "smoke", "check_pass": True},
                                            {"stage": "y2015", "check_pass": True},
                                            {"stage": "report", "check_pass": None}]}}}}))


# Each claim: its text with a slot per value, the (record, path, format) of each
# slot, its records, its class. The text is written from the records, so a
# planted record gives a claim that still matches it.
def _claims_spec(ids, rows):
    a, d, ov, sp, lg, sg, gp, sc = (ids[k] for k in ("anom", "diff", "ov", "split", "leaf",
                                                     "snow", "peak", "screen"))
    return [
        {"t": "Over the watershed, GPP in 2015 was {} gC/m^2/yr, {} gC/m^2/yr from its normal "
              "({} %), where {} of the area had values; the plan expected it higher.",
         "slots": [(a, "value", ".6g"), (a, "anomaly", ".4g"), (a, "percent", ".3g"),
                   (a, "covered", "g")],
         "ev": [a], "cls": "tested expectation", "kw": {"expectation": "E1", "headline": True}},
        {"t": "Over the watershed in summer 2015, GPP of the warm_dry_summer run differed from "
              "the control's by {} gC/m^2/season ({} %).",
         "slots": [(d, "difference", ".4g"), (d, "percent", ".3g")],
         "ev": [d], "cls": "observation", "kw": {}},
        {"t": "Across the watershed in 2015, the GPP anomaly and H2OSOI went together with a "
              "correlation of {}.",
         "slots": [(ov, "correlation", ".4g")],
         "ev": [ov, ids["map_a"], ids["map_b"]], "cls": "association", "kw": {}},
        {"t": "Of the run's summer 2015 GPP difference over the watershed, the band below "
              "1000 m held a share of {} of the total absolute change.",
         "slots": [(sp, "parts.below 1000 m.abs_change_share", ".4g")],
         "ev": [sp, ids["map_d"]], "cls": "observation", "kw": {}},
        {"t": "Over the watershed, leaf green-up in 2015 had a median date of day {}, on {} "
              "cells dated in both 2015 and the normal, with complete inputs on {} of the "
              "area.",
         "slots": [(lg, "variants.default.distribution.median", "g"),
                   (lg, "variants.default.shift.eligible.both.cells", "d"),
                   (lg, "variants.default.distribution.inputs_area_share", "g")],
         "ev": [lg], "cls": "observation", "kw": {}},
        {"t": "Over the watershed, snow was gone in water year 2015 by a median date of day {}.",
         "slots": [(sg, "variants.default.distribution.median", "g")],
         "ev": [sg], "cls": "observation", "kw": {}},
        {"t": "Over the watershed in 2015, the warm_dry_summer run's growth peak had a median "
              "date of day {}.",
         "slots": [(gp, "variants.default.distribution.median", "g")],
         "ev": [gp], "cls": "observation", "kw": {}},
        {"t": "In the 2015 screen over the watershed, GPP's 2015 row scored {} against {} "
              "normal years.",
         "slots": [(sc, rows["gpp"] + "score", ".4g"), (sc, rows["gpp"] + "n_normal", "d")],
         "ev": [sc], "cls": "exploratory", "kw": {}}]


def _fmt(v, f):
    return format(int(v), "d") if f == "d" else format(float(v), f)


def _render(spec, out):
    """The claims, written from the records in `out`, each carrying every
    warning its records carry (by code)."""
    recs = CT.evidence_records(out)
    log = CT._read_log(out)
    claims = []
    for c in spec:
        vals = []
        for rid, path, f in c["slots"]:
            vals.append(log[int(rid[1:]) - 1]["result"][path])
        codes = {w["code"] for i in c["ev"] for w in recs[i]["warnings"]}
        codes |= {w["code"] for i in c["ev"] for ws in recs[i]["row_warnings"].values()
                  for w in ws}
        claims.append({"claim": c["t"].format(*(_fmt(v, f) for v, (_, _, f) in
                                               zip(vals, c["slots"]))),
                       "finding_id": "f", "values": [float(v) for v in vals],
                       "evidence": list(c["ev"]), "evidence_class": c["cls"],
                       "warnings": sorted(codes), **c["kw"]})
    return claims


def _headline(claims):
    return claims[0]["claim"].split(";")[0] + "."


@pytest.fixture(scope="module")
def demo(cfg):
    mp = pytest.MonkeyPatch()
    mp.setenv("ELM_STREAM_INDEX_CACHE", str(cfg["scratch"] / "index_cache"))
    rd = cfg["scratch"] / "study"
    entry = cfg["entry"]
    _records_of_study(rd, entry, cfg["warm"])
    built = S.build(entry, rd / "03_results" / "suite" / RUN, run_dir=cfg["warm"])
    assert built["action"] in ("reused", "built", "added"), built
    out = rd / "04_analysis"
    if out.exists():
        shutil.rmtree(out)                                  # a fresh log each session
    ctx = Ctx(rd, _ctx_plan())
    AP.write_plan(ctx, out, client=Fake(PLAN))
    S3.rule_plan_follows_question(ctx, out, client=Fake({"follows": True, "why": "yes"}))
    assert AP.plan_passed(out)[0]
    spec = AP.access_spec(ctx)
    assert set(spec["suites"]) == {"control", RUN}, spec["suites"]
    acc = AP.make_access(spec, script="fig_demo")
    acc.anomaly("GPP", "2015", region=REGION)
    acc.difference(RUN, "GPP", "2015-JJA", region=REGION)
    a = acc.anomaly("GPP", "2015")                               # per cell
    b = acc.get_map("control", "H2OSOI", "2015")
    acc.overlap(a["anomaly"], b, REGION, by_band=True)
    d = acc.difference(RUN, "GPP", "2015-JJA")                   # per cell
    acc.elevation_split(d["difference"], REGION)
    acc.leaf_green_up(2015, REGION)
    acc.snow_gone(2015, REGION)
    acc.growth_peak(2015, REGION, run=RUN, against="control")
    # the framework closes the planned phase before the screen (the user's
    # decision on G part 2, 2026-11-08); the screen and anything after it are
    # after the close
    AP.close_planned_phase(out, when="before the screen", by="the test, as the framework")
    acc.screen(2015, REGION)                                     # last: found by screening
    log = CT._read_log(out)
    assert all("sha256" in x and not x.get("refused") for x in log), \
        [x.get("refused") for x in log]
    by = {}
    for i, x in enumerate(log, 1):
        k = x.get("keys") or {}
        by.setdefault((x["call"], k.get("field") or k.get("measure"), k.get("region")), f"L{i}")
    ids = {"anom": by[("anomaly", "GPP", REGION)], "diff": by[("difference", "GPP", REGION)],
           "map_a": by[("anomaly", "GPP", None)], "map_b": by[("get_map", "H2OSOI", None)],
           "map_d": by[("difference", "GPP", None)], "ov": by[("overlap", None, REGION)],
           "split": by[("elevation_split", None, REGION)],
           "leaf": by[("leaf_green_up", None, REGION)], "snow": by[("snow_gone", None, REGION)],
           "peak": by[("growth_peak", None, REGION)], "screen": by[("screen", None, REGION)]}
    res = log[int(ids["screen"][1:]) - 1]["result"]
    gpp = next(k[:-5] for k, v in res.items() if k.startswith("rows.") and k.endswith(".field")
               and v == "GPP" and res.get(k[:-5] + "period") == "2015")
    rows = {"gpp": gpp}
    spec = _claims_spec(ids, rows)
    claims = _render(spec, out)
    yield {"rd": rd, "out": out, "ctx": ctx, "ids": ids, "rows": rows, "spec": spec,
           "claims": claims, "cfg": cfg}
    mp.undo()


def _build(ctx, claims):
    return S4.build(ctx, {}, {"findings": []},
                    {"verdict": "sufficient", "headline": _headline(claims),
                     "claims": copy.deepcopy(claims), "open_causes": OPEN}, ctx.run_dir)


def _page(rep):
    return S4.render_markdown({**rep, "question": QUESTION, "model": "elm",
                               "status": "completed", "kind": "whole_domain",
                               "answer": "As the claims say."})


# ── the valid study ──────────────────────────────────────────────────────────

def test_the_valid_study_passes_the_trace_and_the_numbers_check_and_publishes(demo):
    rep = _build(demo["ctx"], demo["claims"])
    t, n = rep["evidence_trace"], rep["evidence_numbers"]
    assert t["claims_failed"] == [] and t["headline"]["problems"] == [], t["problems"]
    assert n["pass"], n.get("error") or [(m["record"], m["path"], m["logged"], m["recomputed"],
                                          m["why"]) for m in n["mismatches"][:12]]
    assert not rep.get("withheld_because")
    cited = {i for c in demo["claims"] for i in c["evidence"]}
    assert set(n["records"]) == cited
    calls = {r["call"] for r in n["records"].values()}
    assert calls >= {"anomaly", "difference", "overlap", "elevation_split", "leaf_green_up",
                     "snow_gone", "growth_peak", "screen"}
    page = _page(rep)
    assert not page.startswith("# WITHHELD")
    assert "**Numbers check: passed.**" in page
    codes = set()
    for c in rep["claims"]:
        for w in c["evidence_trace"]["warnings"]:
            assert f"warning, {w['label']} ({w['record']})" in page
            codes.add(w["code"])
    assert {"restricted_comparability", "eligibility", "provisional_settings",
            "found_by_screening"} <= codes, codes
    w = n.get("largest_relative_difference")
    print(f"\nvalid study: {n['n_compared']} numbers of {len(n['records'])} records made "
          f"again, the largest relative difference {w and w.get('rel_diff')}; reads: "
          f"{len(n['sources']['reads'])} (field, file) pairs")


# ── the planted faults ───────────────────────────────────────────────────────

def _plant(demo, name, rid, path, how):
    """A copy of the study with one number of one record planted wrong, and
    the claims written again from the planted records."""
    rd = demo["cfg"]["scratch"] / "planted" / name
    if rd.exists():
        shutil.rmtree(rd)
    (rd / "04_analysis").mkdir(parents=True)
    src = demo["rd"]
    for f in ("01_inputs", "domain_case"):
        shutil.copytree(src / f, rd / f)
    shutil.copy(src / "experiment.json", rd / "experiment.json")
    os.symlink(src / "03_results", rd / "03_results")
    for f in (src / "04_analysis").iterdir():
        if f.is_file() and f.name != NC.EVIDENCE_JSON:
            shutil.copy(f, rd / "04_analysis" / f.name)
    p = rd / "04_analysis" / "lookups.jsonl"
    lines = p.read_text().splitlines()
    rec = json.loads(lines[int(rid[1:]) - 1])
    old = rec["result"][path]
    rec["result"][path] = how(old)
    lines[int(rid[1:]) - 1] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n")
    claims = _render(demo["spec"], rd / "04_analysis")
    ctx = Ctx(rd, _ctx_plan())
    return ctx, claims, old, rec["result"][path]


FAULTS = [
    ("region_value", "anom", "value", lambda v: round(v * 1.02, 2)),
    ("anomaly_percent", "anom", "percent", lambda v: round(v + 0.7, 1)),
    ("screen_score", "screen", "{gpp}score", lambda v: round(v * 1.05 + 0.1, 3)),
    ("overlap_correlation", "ov", "correlation", lambda v: round(v + 0.05, 3)),
    ("band_share", "split", "parts.below 1000 m.abs_change_share",
     lambda v: round(min(v * 1.1 + 0.01, 0.99), 3)),
    ("timing_date", "leaf", "variants.default.distribution.median", lambda v: v + 3),
    ("eligibility_count", "leaf", "variants.default.shift.eligible.both.cells", lambda v: v + 7),
    ("coverage_share", "leaf", "variants.default.distribution.inputs_area_share",
     lambda v: round(v * 0.97, 2))]


@pytest.mark.parametrize("name,key,path,how", FAULTS, ids=[f[0] for f in FAULTS])
def test_a_number_planted_wrong_is_caught_only_by_the_numbers_check(demo, name, key, path, how):
    rid = demo["ids"][key]
    path = path.format(**demo["rows"])
    ctx, claims, old, new = _plant(demo, name, rid, path, how)
    assert new != old
    rep = _build(ctx, claims)
    t, n = rep["evidence_trace"], rep["evidence_numbers"]
    assert t["claims_failed"] == [], t["problems"]       # the trace alone passes
    assert not n["pass"]
    hit = [m for m in n["mismatches"] if m["record"] == rid and m["path"] == path]
    assert hit and hit[0]["logged"] == new, n["mismatches"][:4]
    assert len(n["mismatches"]) == 1, n["mismatches"]
    assert rep["withheld_because"] and rid in rep["withheld_because"]
    page = _page(rep)
    assert page.startswith("# WITHHELD")
    row = next(x for x in page.splitlines() if x.startswith(f"| {rid} (") and f"| {path} |" in x)
    assert NC._fmt(new) in row and NC._fmt(hit[0]["recomputed"]) in row
    print(f"\n{name}: {rid} {path} logged {new}, made again {hit[0]['recomputed']}")
