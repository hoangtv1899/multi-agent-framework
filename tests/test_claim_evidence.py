"""Evidence records and the claim trace of a study with a suite (package F1).

    pytest tests/test_claim_evidence.py -v

ONE SCRIPTED STUDY, NO MODEL. A hypothesis study over a synthetic suite (the
control and one run with one change, every server check passed): its plan
is written and reviewed by scripted fakes, its figures' lookups go through
the real `suite` (anomaly, growth_peak, difference, two maps and their
overlap, then the screen, then an amendment), and the lookups whose warnings
the small synthetic grid cannot produce (coverage, a timing comparison with
eligibility changes, restricted comparability and three reason codes, the
screen's near-constant, insufficient-support, not-scored and marked rows,
FPG, a percent-free field) are written by the same log, shaped as the
methods return them. The claims are a fake Analyzer's: valid, each carrying
the warnings of its records by code.

The valid study passes and renders with every warning beside its claim.
Each other test plants one fault in it and checks that the trace catches it
and the report is withheld: a number that does not match its record, a claim
with no record, a period or a region the record does not name, a causal word
on an association, a real-world "caused by", in-model attribution without a
one-change run, a screen finding as a hypothesis headline, an amended lookup
claimed as tested, a missing warning (one per family) and a forbidden
percent.
"""
import copy
import json
from pathlib import Path

import pytest

import domain_access as DAC
import domain_catalog as DCAT
import domain_suite as S
from test_domain_access import REGION, YEARS, Ctx, Fake
from test_domain_suite import FIELDS, make_control

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402
from agents.analysis import step4_report as S4           # noqa: E402

WARM = {"id": "weather_scale", "params": {"temperature_K": 2.0}, "summary": "warm the air",
        "meaning": "the response to air 2 K warmer, every other input the control's",
        "not_meaning": "a consistent change of climate"}
PLAN = {
    "comparisons": [{"id": "C1", "what": "2016 against the normal year", "run": "control",
                     "against": "normal years", "years": [2016]}],
    "baseline": {"normal_years": [2015], "why": "the Planner's"},
    "analyses": [
        {"id": "P1", "question": "GPP in 2016", "fields": ["GPP"], "regions": [REGION],
         "periods": ["2016"], "methods": ["anomaly", "difference"], "settings": {},
         "why": "what was asked",
         "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2016",
                      "region": REGION}]},
        {"id": "P2", "question": "the maps", "fields": ["GPP", "TOTECOSYSC"], "regions": [],
         "periods": ["2016"], "methods": [], "settings": {}, "why": "to draw them",
         "outputs": [{"id": "O1", "kind": "map", "field": "GPP", "period": "2016"},
                     {"id": "O2", "kind": "map", "field": "TOTECOSYSC", "period": "2016"}]},
        {"id": "P3", "question": "how they go together", "fields": [], "regions": [REGION],
         "periods": [], "methods": ["overlap"], "settings": {}, "why": "where",
         "outputs": [{"id": "O1", "kind": "overlap", "region": REGION,
                      "inputs": ["P2.O1", "P2.O2"]}]},
        {"id": "P4", "question": "the peak's date", "fields": [], "regions": [REGION],
         "periods": ["2016"], "methods": ["growth_peak"], "settings": {}, "why": "when",
         "outputs": [{"id": "O1", "kind": "growth_peak", "region": REGION,
                      "period": "2016"}]}],
    "expectations": [{"id": "E1", "analysis": "P1", "field": "GPP", "sign": "higher",
                      "text": "GPP higher in 2016 than in the normal year"}],
    "competing": [{"text": "soil water", "fields": ["H2OSOI"], "analysis": "P1"}],
    "screen": {"when": "last", "region": REGION, "year": 2016, "why": "a hypothesis"}}
HEADLINE = "Over the vegetated cells, GPP in 2016 was {v} gC/m^2/yr above its 2015 normal."
OPEN = [{"cause": "whether the warmer air raised GPP through a longer season",
         "proposed_run": {"changes": [{"id": "weather_scale",
                                       "params": {"temperature_K": 2.0, "months": [3, 4, 5],
                                                  "region": "watershed"}}]}}]


def _first(rec, paths):
    for p in paths:
        v = rec["values"].get(p)
        if v is not None:
            return p, v
    raise AssertionError(f"{rec['id']} holds none of {paths}: {sorted(rec['values'])[:20]}")


def _inject(out, script, call, keys, method, result, marks):
    """A lookup the synthetic grid cannot produce, logged by the access's own
    log with a result shaped as the method returns it."""
    AP.LookupLog(out, script=script)({
        "call": call, "keys": keys, "method": method, "settings": None,
        "sha256": DAC.fingerprint(result), "shape": DAC._shape(result),
        "result": DAC.flat_result(call, result), **marks})
    return f"L{len(CT._read_log(out))}"


def _codes(records, ids):
    return sorted({w["code"] for i in ids for w in records[i]["warnings"]})


@pytest.fixture(scope="module")
def scripted(tmp_path_factory):
    root = tmp_path_factory.mktemp("evidence")
    control, _ = make_control(root / "ctl", YEARS)
    S.build(control, root / "suite", fields=FIELDS)
    rd = root / "study"
    (rd / "01_inputs").mkdir(parents=True)
    (rd / "01_inputs" / "domain_control_facts.json").write_text(json.dumps(
        {"control": {"name": "synth", "run_dir": control["run_dir"]}}))
    warm, _ = make_control(root / "warm", YEARS)
    S.build(warm, rd / "03_results" / "suite" / "warm", fields=FIELDS)
    (rd / "experiment.json").write_text(json.dumps({
        "kind": "whole_domain",
        "experiment": {"runs": [{"run": "control", "reused": True, "changes": []},
                                {"run": "warm", "changes": [WARM]}]},
        "chain": {"years_planned": YEARS, "years_passed": YEARS,
                  "runs": {"warm": {"stages": [
                      {"stage": "case", "check_pass": True},
                      {"stage": "smoke", "check_pass": True},
                      {"stage": "y2015", "check_pass": True},
                      {"stage": "y2016", "check_pass": True},
                      {"stage": "report", "check_pass": None}]}}}}))
    mp = pytest.MonkeyPatch()
    entry = dict(control, suite_dir=str(root / "suite"))
    mp.setattr(AP, "_registry_entry", lambda name: entry if name == "synth" else {})
    plan = {"question": "Was GPP higher in 2016 than in the normal year?",
            "goals": ["GPP in 2016"],
            "experiment": {"control": {"name": "synth", "reuse": True},
                           "runs": [{"run": "warm", "changes": [WARM]}], "years": YEARS,
                           "analysis": {"normal_years": [2015], "question_type": "hypothesis",
                                        "regions": [{"name": REGION}], "variables": ["GPP"]}}}
    ctx = Ctx(rd, plan)
    out = rd / "04_analysis"
    out.mkdir()
    AP.write_plan(ctx, out, client=Fake(PLAN))
    S3.rule_plan_follows_question(ctx, out, client=Fake({"follows": True, "why": "yes"}))
    assert AP.plan_passed(out)[0]
    spec = AP.access_spec(ctx)
    # THE FIGURES' LOOKUPS, THROUGH THE REAL SUITE, in the hypothesis order
    acc = AP.make_access(spec, script="fig_gpp")
    acc.anomaly("GPP", "2016", region=REGION)
    acc.difference("warm", "GPP", "2016", region=REGION)
    a = acc.get_map("control", "GPP", "2016")
    b = acc.get_map("control", "TOTECOSYSC", "2016")
    acc.overlap(a, b, REGION)
    acc.growth_peak(2016, REGION)
    # the framework closes the planned phase before the screen (the user's
    # decision on G part 2, 2026-11-08); every line after it is after the close
    AP.close_planned_phase(out, when="before the screen", by="the test, as the framework")
    acc.screen(2016, REGION)                          # last: found by screening, not tested
    am = AP.append_amendment(out, "a field the screen showed", {
        "fields": ["TOTECOSYSC"], "regions": [REGION], "periods": ["2016"],
        "methods": ["anomaly"]})
    AP.make_access(spec, script="fig_more", amendment=am["id"]).anomaly(
        "TOTECOSYSC", "2016", region=REGION)
    # WHAT THE SMALL GRID CANNOT PRODUCE, logged by the same log
    planned = {"planned": True, "plan_items": ["P1"]}
    ids = {}
    ids["cov"] = _inject(out, "fig_soil", "anomaly", {"run": "control", "field": "H2OSOI",
                         "period": "2016", "region": REGION}, "anomaly", {
        "method": "anomaly", "field": "H2OSOI", "period": "2016", "region": REGION,
        "normal_years": [2015], "anomaly": -0.0123, "covered": 0.92,
        "normal_incomplete": [2015],
        "coverage": {"least_area_share": 0.92, "missing_cell_values": [2016],
                     "missing_time_records": []}}, planned)
    ids["snow"] = _inject(out, "fig_snow", "snow_gone", {"run": "control", "year": 2016,
                          "region": REGION, "against": "normal"}, "snow_gone", {
        "method": "snow_gone", "year": 2016, "against": "the normal years [2015]",
        "variants": {"default": {
            "distribution": {"median": 141.0, "inputs_area_share": 1.0,
                             "missing_time_records": {"cells": 0, "area_share": 0.0},
                             "reasons": {"snow_persists": 3, "insufficient_follow_up": 2,
                                         "records_missing": 1}},
            "shift": {"cells": 11, "area_share": 0.81, "median": -12.0,
                      "eligible": {"both": {"cells": 11, "area_share": 0.81},
                                   "only_year": {"cells": 2, "area_share": 0.09},
                                   "only_normal": {"cells": 1, "area_share": 0.04},
                                   "neither": {"cells": 4, "area_share": 0.06}},
                      "comparability": "restricted comparability"}}},
        "comparability": "restricted comparability: a partial-record date is in a shift"},
        {"planned": True, "plan_items": ["P4"]})
    rows = [{"field": "TSA", "group": "weather in", "period": "2016", "change": 0.31,
             "status": "near-constant normal", "near_constant": True},
            {"field": "QRUNOFF", "group": "runoff", "period": "2016", "change": 4.4,
             "status": "insufficient support"},
            {"field": "TOTVEGC", "group": "vegetation carbon", "period": "2016",
             "change": 123.0, "status": "not scored yet"},
            {"field": "FAREA_BURNED", "group": "fire", "period": "2016", "change": 0.0021,
             "status": "marked", "mark": "differs_from_control"}]
    ids["rows"] = _inject(out, "fig_screen_rows", "screen", {"run": "control", "year": 2016,
                          "region": REGION}, "screen", {
        "method": "screen", "region": REGION, "year": 2016, "normal_years": [2015],
        "against": "the normal years", "rows": rows,
        "display": {"rule": "shown first", "groups": {}, "events": [rows[3]],
                    "not_scored": [dict(rows[2], label="not scored")]}},
        {"planned": False, "plan_items": ["screen"], "label": AP.SCREENING_LABEL})
    ids["fpg"] = _inject(out, "fig_fpg", "anomaly", {"run": "control", "field": "FPG",
                         "period": "2016", "region": REGION}, "anomaly", {
        "method": "anomaly", "field": "FPG", "period": "2016", "region": REGION,
        "normal_years": [2015], "anomaly": -0.021, "covered": 1.0}, planned)
    ids["qch"] = _inject(out, "fig_qch", "anomaly", {"run": "control", "field": "QCHARGE",
                         "period": "2016", "region": REGION}, "anomaly", {
        "method": "anomaly", "field": "QCHARGE", "period": "2016", "region": REGION,
        "normal_years": [2015], "anomaly": -3.4, "percent": -12.0, "covered": 1.0}, planned)
    recs = CT.evidence_records(out)
    by = {}
    for r in recs.values():
        by.setdefault((r["call"], r.get("field") or r.get("measure")), r["id"])
    ids.update(anom=by[("anomaly", "GPP")], diff=by[("difference", "GPP")],
               map_a=by[("get_map", "GPP")], map_b=by[("get_map", "TOTECOSYSC")],
               ov=by[("overlap", None)], gp=by[("growth_peak", "growth_peak")],
               screen=[r["id"] for r in recs.values() if r["call"] == "screen"][0],
               amend=by[("anomaly", "TOTECOSYSC")])
    _, anom = _first(recs[ids["anom"]], ["anomaly"])
    _, dif = _first(recs[ids["diff"]], ["difference", "change", "value"])
    _, corr = _first(recs[ids["ov"]], ["correlation", "sign_shares.first_positive"])
    gpp, gpv = _first(recs[ids["gp"]], ["variants.default.distribution.median",
                                        "variants.default.distribution.cells"])
    _, amv = _first(recs[ids["amend"]], ["anomaly"])

    def claim(text, ev, cls, values, **kw):
        return {"claim": text, "finding_id": "f", "values": values, "evidence": ev,
                "evidence_class": cls, "warnings": _codes(recs, ev), **kw}
    claims = [
        claim(f"Over the vegetated cells, GPP in 2016 was {anom:.4g} gC/m^2/yr above its "
              f"2015 normal, consistent with the plan's expectation.", [ids["anom"]],
              "tested expectation", [anom], expectation="E1", headline=True),
        claim(f"Over the vegetated cells, the 2016 growth peak's median was {gpv:.4g}.",
              [ids["gp"]], "observation", [gpv]),
        claim("The 2016 screen over the vegetated cells listed GPP among its rows.",
              [ids["screen"]], "exploratory", []),
        claim(f"In the model, the change caused GPP over the vegetated cells in 2016 to "
              f"differ from the control by {dif:.4g}.", [ids["diff"]], "in-model attribution",
              [dif]),
        claim(f"Across the vegetated cells in 2016, GPP and TOTECOSYSC went together, with a "
              f"correlation of {corr:.4g}.", [ids["ov"], ids["map_a"], ids["map_b"]],
              "association", [corr]),
        claim(f"Over the vegetated cells, TOTECOSYSC in 2016 was {amv:.4g} from its 2015 "
              f"normal.", [ids["amend"]], "exploratory", [amv]),
        claim("Over the vegetated cells, H2OSOI in 2016 was 0.0123 below its 2015 normal, "
              "where 0.92 of the area had values.", [ids["cov"]], "observation", [-0.0123]),
        claim("Over the vegetated cells, snow was gone 12 days earlier in 2016 than in the "
              "2015 normal, on 11 cells.", [ids["snow"]], "observation", [-12.0]),
        claim("In the 2016 screen over the vegetated cells, TSA changed by 0.31, QRUNOFF by "
              "4.4, TOTVEGC by 123 and FAREA_BURNED by 0.0021.", [ids["rows"]], "exploratory",
              [0.31, 4.4, 123.0, 0.0021]),
        claim("Over the vegetated cells, FPG in 2016 was 0.021 below its 2015 normal.",
              [ids["fpg"]], "observation", [-0.021]),
        claim("Over the vegetated cells, QCHARGE in 2016 was 3.4 below its 2015 normal.",
              [ids["qch"]], "observation", [-3.4])]
    # the screen rows' own warnings, by code, as a correct claim carries them
    claims[8]["warnings"] = sorted(set(claims[8]["warnings"]) | {
        "near_constant", "insufficient_support", "not_scored_yet", "event_mark"})
    yield {"out": out, "rd": rd, "ctx": ctx, "claims": claims, "ids": ids, "records": recs,
           "headline": HEADLINE.format(v=f"{anom:.4g}"), "regions": S4.suite_regions(ctx)}
    mp.undo()


def _trace(sc, claims=None, headline=None, study_dir=None, open_causes=OPEN):
    cl = copy.deepcopy(claims if claims is not None else sc["claims"])
    t = CT.trace_evidence(cl, headline if headline is not None else sc["headline"], sc["out"],
                          study_dir=study_dir or sc["rd"], regions=sc["regions"],
                          open_causes=open_causes)
    return cl, t


def _with(sc, i, **kw):
    cl = copy.deepcopy(sc["claims"])
    cl[i].update(kw)
    return cl


def _report(sc, claims, t):
    return {"question": "Was GPP higher in 2016 than in the normal year?", "model": "elm",
            "status": "completed", "generated_at": "2026-11-05", "run_dir": str(sc["rd"]),
            "kind": "whole_domain", "headline": sc["headline"], "answer": "As the claims say.",
            "claims": claims, "caveats": [], "withheld": [], "provenance": {}, "cost": {},
            "figures": {"investigation": {"fig_gpp": str(sc["out"] / "fig_gpp.png")}},
            "evidence_trace": t, "open_causes": OPEN,
            **({"withheld_because": t["withheld_because"]} if t.get("withheld_because")
               else {})}


def _fails(t, i, words):
    probs = t["problems"].get(i + 1) or []
    assert t.get("withheld_because") and (i + 1) in t["claims_failed"], t.get("problems")
    assert any(words in p for p in probs), probs
    return probs


# ── the valid study ──────────────────────────────────────────────────────────

def test_the_valid_scripted_study_passes_and_renders_every_warning(scripted):
    cl, t = _trace(scripted)
    assert t["claims_failed"] == [] and t["headline"]["problems"] == [], t["problems"]
    assert "withheld_because" not in t and t["open_cause_problems"] == []
    # every warning family is on some record of the study, and on the page
    codes = {w["code"] for c in cl for w in c["evidence_trace"]["warnings"]}
    fam = {"coverage", "normal_left_out", "eligibility", "restricted_comparability",
           "reason:snow_persists",
           "reason:insufficient_follow_up", "reason:records_missing", "near_constant",
           "insufficient_support", "not_scored_yet", "event_mark", "provisional_settings",
           "found_by_screening", "not_planned", "fpg_configuration"}
    assert fam <= codes, fam - codes
    page = S4.render_markdown(_report(scripted, cl, t))
    assert not page.startswith("# WITHHELD")
    for c in cl:
        for w in c["evidence_trace"]["warnings"]:
            assert f"warning, {w['label']} ({w['record']})" in page
    for cls in CT.EVIDENCE_CLASSES:
        assert f"evidence class: {cls}" in page
    for words in ("## The maps and series the Analyzer drew", "## Timing", "eligible in both",
                  "## The screen (found by screening, not tested)", "Scoring rule:",
                  "Display rule:", CT.HEADLINE_RULE["hypothesis"], CT.IN_MODEL_WORDS,
                  "## Open causes and the runs that would settle them", "weather_scale"):
        assert words in page, words
    # the record names the lookup's hash, its comparison and its support
    r = t["records"][scripted["ids"]["anom"]]
    assert r["sha256"] and r["comparison"]["against"] == "the normal years"
    assert r["support"]["area_share"] == 1.0


def test_the_report_of_a_suite_study_runs_the_evidence_trace(scripted, monkeypatch):
    # Five of this study's records are written by hand (shaped as the methods
    # return them, values the small grid cannot produce), so the numbers check
    # of package F2 rightly cannot make them again from the history; it is
    # stood in for here, and has its own tests (tests/test_suite_numbers_check*.py).
    from agents.analysis import numbers_check as NCK
    monkeypatch.setattr(NCK, "check_evidence", lambda *a, **k: {
        "pass": True, "n_compared": 0, "mismatches": [], "records": {}})
    ctx = scripted["ctx"]
    rep = S4.build(ctx, {}, {"findings": []},
                   {"verdict": "sufficient", "headline": scripted["headline"],
                    "claims": copy.deepcopy(scripted["claims"]), "open_causes": OPEN},
                   scripted["rd"])
    assert "numbers_trace" not in rep and rep["evidence_trace"]["claims_failed"] == []
    assert not rep.get("withheld_because")


# ── the planted faults ───────────────────────────────────────────────────────

def test_a_number_that_does_not_match_its_record_is_rejected(scripted):
    c = scripted["claims"][0]
    cl = _with(scripted, 0, claim=c["claim"].replace(f"{c['values'][0]:.4g}", "52.5"),
               values=[52.5])
    cl, t = _trace(scripted, cl)
    _fails(t, 0, "the number 52.5 matches no value of its records")
    page = S4.render_markdown(_report(scripted, cl, t))
    assert page.startswith("# WITHHELD") and "FAILS: the number 52.5" in page


def test_a_claim_with_no_record_is_rejected(scripted):
    _, t = _trace(scripted, _with(scripted, 1, evidence=[]))
    _fails(t, 1, "no record: every empirical claim names the logged lookup")
    _, t = _trace(scripted, _with(scripted, 1, evidence=["L999"]))
    _fails(t, 1, "it names L999, which is no answered lookup")


def test_a_period_or_region_the_record_does_not_name_is_rejected(scripted):
    c = scripted["claims"][0]["claim"]
    _, t = _trace(scripted, _with(scripted, 0, claim=c.replace("in 2016", "in July 2016")))
    _fails(t, 0, "it names July, which none of its records names")
    _, t = _trace(scripted, _with(scripted, 0, claim=c.replace("2015 normal", "2018 normal")))
    _fails(t, 0, "it names 2018, which none of its records names")
    _, t = _trace(scripted, _with(scripted, 0, claim=c.replace(
        "Over the vegetated cells", "Over the box")))
    _fails(t, 0, "it names the region 'box', which none of its records names")


def test_a_causal_verb_on_an_association_is_rejected(scripted):
    c = scripted["claims"][4]["claim"]
    _, t = _trace(scripted, _with(scripted, 4, claim=c.replace("went together", "GPP drove "
                                                               "TOTECOSYSC")))
    _fails(t, 4, "the causal word \"drove\" goes beyond association")


def test_a_real_world_caused_by_is_rejected(scripted):
    c = scripted["claims"][0]["claim"]
    _, t = _trace(scripted, _with(scripted, 0, claim=c.replace(
        "consistent with the plan's expectation", "caused by the wet spring")))
    _fails(t, 0, "\"caused by\" names a cause")
    # not even for in-model attribution
    c = scripted["claims"][3]["claim"]
    _, t = _trace(scripted, _with(scripted, 3, claim=c + " It was caused by the warming."))
    _fails(t, 3, "\"caused by\" names a cause")


def test_in_model_attribution_without_a_one_change_run_is_rejected(scripted, tmp_path):
    e = json.loads((scripted["rd"] / "experiment.json").read_text())
    e["experiment"]["runs"][1]["changes"].append(dict(WARM, id="co2_ppmv"))
    (tmp_path / "experiment.json").write_text(json.dumps(e))
    _, t = _trace(scripted, study_dir=tmp_path)
    _fails(t, 3, "in-model attribution needs a difference from a run that differs from the "
                 "control by one named change")
    e["experiment"]["runs"][1]["changes"].pop()
    e["chain"]["runs"]["warm"]["stages"][0]["check_pass"] = False         # a check failed
    (tmp_path / "experiment.json").write_text(json.dumps(e))
    _, t = _trace(scripted, study_dir=tmp_path)
    _fails(t, 3, "confirmed by the server's checks")
    # and never on a lookup that is not a difference
    _, t = _trace(scripted, _with(scripted, 0, evidence_class="in-model attribution"))
    _fails(t, 0, "in-model attribution needs a difference")


def test_a_screen_finding_as_a_hypothesis_headline_is_rejected(scripted):
    cl = copy.deepcopy(scripted["claims"])
    cl[0]["headline"], cl[2]["headline"] = False, True
    cl, t = _trace(scripted, cl, headline="The 2016 screen listed GPP among its rows.")
    assert t["claims_failed"] == [] and t.get("withheld_because")
    assert any("claim 3 is exploratory, not tested" in p for p in t["headline"]["problems"])
    assert any("found by screening or not planned: never a hypothesis headline" in p
               for p in t["headline"]["problems"])
    assert S4.render_markdown(_report(scripted, cl, t)).startswith("# WITHHELD")


def test_an_amended_lookup_claimed_as_tested_is_rejected(scripted):
    _, t = _trace(scripted, _with(scripted, 5, evidence_class="tested expectation",
                                  expectation="E1"))
    probs = _fails(t, 5, "is not planned (amendment")
    assert any("never a tested expectation" in p for p in probs)
    # nor a screen finding
    _, t = _trace(scripted, _with(scripted, 2, evidence_class="tested expectation",
                                  expectation="E1"))
    _fails(t, 2, "was found by screening, not tested: never a tested expectation")


FAMILIES = [(6, "coverage"), (6, "normal_left_out"), (7, "eligibility"), (7, "restricted_comparability"),
            (7, "reason:snow_persists"), (7, "reason:insufficient_follow_up"),
            (7, "reason:records_missing"), (8, "near_constant"), (8, "insufficient_support"),
            (8, "not_scored_yet"), (8, "event_mark"), (1, "provisional_settings"),
            (2, "found_by_screening"), (5, "not_planned"), (9, "fpg_configuration")]


@pytest.mark.parametrize("i,code", FAMILIES, ids=[c for _, c in FAMILIES])
def test_a_missing_warning_is_rejected(scripted, i, code):
    assert code in scripted["claims"][i]["warnings"]
    cl = _with(scripted, i, warnings=[w for w in scripted["claims"][i]["warnings"]
                                      if w != code])
    _, t = _trace(scripted, cl)
    probs = _fails(t, i, f"({code}) its record")
    assert all("leaves out the warning" in p for p in probs), probs
    # carried in the claim's own words instead, it passes
    label = next(w["label"] for w in t["records"][cl[i]["evidence"][0]]["warnings"]
                 + [w for r in t["records"].values() for ws in r["row_warnings"].values()
                    for w in ws] if w["code"] == code)
    cl[i]["claim"] += f" ({label})"
    _, t = _trace(scripted, cl)
    assert t["claims_failed"] == [], t["problems"]


def test_a_forbidden_percent_is_rejected(scripted):
    assert DCAT.load_notes()["QCHARGE"]["percent"]["allowed"] is False
    _, t = _trace(scripted, _with(scripted, 10, claim="Over the vegetated cells, QCHARGE in "
                                  "2016 was 12 % below its 2015 normal.", values=[-12.0]))
    probs = _fails(t, 10, "a percent of QCHARGE, whose notes give none")
    assert len(probs) == 1                      # the number itself is the record's


def test_an_exploratory_headline_from_the_screen_is_labelled_exploratory(scripted,
                                                                         monkeypatch):
    plan = dict(AP.load_plan(scripted["out"]), question_type="exploratory")
    monkeypatch.setattr(AP, "load_plan", lambda out: plan)
    cl = copy.deepcopy(scripted["claims"])
    cl[0]["headline"], cl[2]["headline"] = False, True
    _, t = _trace(scripted, cl, headline="The 2016 screen listed GPP among its rows.")
    assert t["headline"]["rule"] == CT.HEADLINE_RULE["exploratory"]
    assert any("is labelled exploratory" in p for p in t["headline"]["problems"])
    _, t = _trace(scripted, cl, headline="Exploratory: the 2016 screen listed GPP among its "
                                         "rows.")
    assert t["headline"]["problems"] == []


# ── the range across settings (the user's decisions on G part 1, 2026-11-08) ─

def _onset(shifts):
    """A photosynthesis_onset lookup over the vegetated cells in 2016 against
    the 2015 normal, made by the method itself from two cells whose shift at
    each setting (default, 0.1, 0.3) is the one given."""
    import numpy as np
    import domain_methods as M
    sets = {"default": 0.2, "fraction=0.1": 0.1, "fraction=0.3": 0.3}

    def year(y, dates):
        v = {lab: {"settings": {"fraction": f, "window": 7}, "date": np.full(2, float(dates[lab])),
                   "reason": np.zeros(2, dtype=np.int64), "flags": np.zeros(2, dtype=np.int64)}
             for lab, f in sets.items()}
        return {"method": "photosynthesis_onset", "version": M.METHODS["photosynthesis_onset"].version,
                "period": str(y), "kind": "year", "year": y, "variants": v}
    normal = {"default": 100, "fraction=0.1": 90, "fraction=0.3": 110}
    by = {2015: year(2015, normal),
          2016: year(2016, {k: normal[k] + s for k, s in zip(sets, shifts)})}
    return M.timing_summary("photosynthesis_onset", by, 2016, np.ones(2, bool),
                            np.array([1.0, 1.0]), normal_years=[2015])


@pytest.fixture()
def ranged(scripted, tmp_path):
    """The scripted study, copied, with three timing lookups: a range over 7
    days whose settings reverse the direction, one within 7 days, and one
    within 7 days whose settings reverse the direction."""
    import shutil
    rd = tmp_path / "study"
    shutil.copytree(scripted["rd"], rd)
    out = rd / "04_analysis"
    keys = {"run": "control", "year": 2016, "region": REGION, "against": "normal"}
    marks = {"planned": True, "plan_items": ["P4"]}
    ids = {name: _inject(out, "fig_onset", "photosynthesis_onset", keys, "photosynthesis_onset",
                         _onset(sh), marks)
           for name, sh in (("wide", (-2, 4, 16)), ("narrow", (-2, -5, -1)),
                            ("flips", (-2, 3, -1)))}
    return {"rd": rd, "out": out, "ids": ids, "records": CT.evidence_records(out),
            "regions": scripted["regions"]}


def _onset_claim(rg, which, text, values, warnings=None):
    rec = rg["records"][rg["ids"][which]]
    return {"claim": text, "finding_id": "f", "values": values, "evidence": [rec["id"]],
            "evidence_class": "observation",
            "warnings": sorted({w["code"] for w in rec["warnings"]}) if warnings is None
            else warnings}


def _onset_trace(rg, claims):
    return CT.trace_evidence(copy.deepcopy(claims), None, rg["out"], study_dir=rg["rd"],
                             regions=rg["regions"])


ALONE = "Over the vegetated cells, photosynthesis onset in 2016 shifted by -2 days against " \
        "the 2015 normal."
RANGED = ("Over the vegetated cells, photosynthesis onset in 2016 shifted by -2 days at the "
          "default fraction 0.2, by 4 days at fraction 0.1 and by 16 days at fraction 0.3 "
          "against the 2015 normal.")


def test_the_record_holds_the_range_and_flags_a_reversal(ranged):
    a = ranged["records"][ranged["ids"]["wide"]]["across_settings"]
    assert a["range_required"] is True and a["direction_reversal"] is True
    assert (a["minimum"], a["maximum"], a["default"]) == (
        {"variant": "default", "median_shift": -2.0},
        {"variant": "fraction=0.3", "median_shift": 16.0},
        {"variant": "default", "median_shift": -2.0})
    assert a["spread_days"] == 18.0 and a["comparable_cells"] == 2
    assert a["evaluated"] == ["default", "fraction=0.1", "fraction=0.3"]
    codes = {w["code"] for w in ranged["records"][ranged["ids"]["wide"]]["warnings"]}
    assert "settings_reverse_direction" in codes
    n = ranged["records"][ranged["ids"]["narrow"]]["across_settings"]
    assert n["range_required"] is False and n["direction_reversal"] is False
    assert "settings_reverse_direction" not in {
        w["code"] for w in ranged["records"][ranged["ids"]["narrow"]]["warnings"]}


def test_a_shift_without_the_range_is_rejected_when_the_settings_differ_by_over_a_week(ranged):
    t = _onset_trace(ranged, [_onset_claim(ranged, "wide", ALONE, [-2.0])])
    assert t["claims_failed"] == [1], t["problems"]
    p = " ".join(t["problems"][1])
    assert "moves by 18 days across the settings, more than 7" in p
    assert "the default, -2 days at default" in p and "the largest, 16 days at fraction=0.3" in p
    # without the largest, or with the largest but not its setting
    t = _onset_trace(ranged, [_onset_claim(ranged, "wide", RANGED.replace(
        " and by 16 days at fraction 0.3", ""), [-2.0, 4.0])])
    assert t["claims_failed"] == [1] and "the largest, 16 days at fraction=0.3" in \
        " ".join(t["problems"][1])
    t = _onset_trace(ranged, [_onset_claim(ranged, "wide", RANGED.replace(
        "by 16 days at fraction 0.3", "by 16 days at another setting"), [-2.0, 4.0, 16.0])])
    assert t["claims_failed"] == [1] and "the largest" in " ".join(t["problems"][1])


def test_a_shift_with_the_default_the_smallest_and_the_largest_passes(ranged):
    t = _onset_trace(ranged, [_onset_claim(ranged, "wide", RANGED, [-2.0, 4.0, 16.0])])
    assert t["claims_failed"] == [], t["problems"]
    # the label of a setting names it too
    t = _onset_trace(ranged, [_onset_claim(ranged, "wide", RANGED.replace(
        "at fraction 0.3", "at fraction=0.3"), [-2.0, 4.0, 16.0])])
    assert t["claims_failed"] == [], t["problems"]
    # within 7 days the default alone is enough; a date is no shift
    t = _onset_trace(ranged, [_onset_claim(ranged, "narrow", ALONE, [-2.0]),
                              _onset_claim(ranged, "wide", "Over the vegetated cells, the median "
                                           "photosynthesis onset in 2016 was day 98.", [98.0])])
    assert t["claims_failed"] == [], t["problems"]


def test_a_claim_without_the_reversal_warning_is_rejected(ranged):
    for which, text, vals in (("wide", RANGED, [-2.0, 4.0, 16.0]), ("flips", ALONE, [-2.0])):
        rec = ranged["records"][ranged["ids"][which]]
        w = [x["code"] for x in rec["warnings"] if x["code"] != "settings_reverse_direction"]
        t = _onset_trace(ranged, [_onset_claim(ranged, which, text, vals, warnings=w)])
        assert t["claims_failed"] == [1], (which, t["problems"])
        assert any("(settings_reverse_direction)" in p for p in t["problems"][1]), t["problems"]
        # carried in its own words, it passes
        t = _onset_trace(ranged, [_onset_claim(ranged, which, text[:-1] + ": the "
                                               + CT.REVERSAL_LABEL + " of the shift.", vals,
                                               warnings=w)])
        assert t["claims_failed"] == [], (which, t["problems"])


def test_the_timing_table_shows_every_setting_the_comparable_cells_and_the_flag(ranged, scripted):
    cl = [_onset_claim(ranged, "wide", RANGED, [-2.0, 4.0, 16.0])]
    t = _onset_trace(ranged, cl)
    page = S4.render_markdown(dict(_report(scripted, cl, t), run_dir=str(ranged["rd"])))
    assert not page.startswith("# WITHHELD")
    i = ranged["ids"]["wide"]
    head = next(x for x in page.splitlines() if x.startswith("| record | measure |"))
    assert "| setting |" in head and "| comparable cells |" in head and \
        "| shift on the comparable cells (days) |" in head and "| range across settings |" in \
        head and "| settings reverse the direction |" in head
    rows = [x for x in page.splitlines() if x.startswith(f"| {i} | photosynthesis_onset |")]
    assert [x.split(" | ")[6] for x in rows] == ["default", "fraction=0.1", "fraction=0.3"]
    assert all(len(x.split(" | ")) == len(head.split(" | ")) for x in rows)
    assert all("| 2 cells (1) |" in x for x in rows)                  # beside each setting's own
    assert "| -2 (smallest) |" in rows[0] and "| 16 (largest) |" in rows[2]
    assert "-2 (default) to 16 (fraction=0.3), default -2: 18 days, more than 7" in rows[0]
    assert rows[0].split(" | ")[-2] != "" and "| yes |" in rows[0]
    n = [x for x in page.splitlines()
         if x.startswith(f"| {ranged['ids']['narrow']} | photosynthesis_onset |")]
    assert "within 7" in n[0] and "| no |" in n[0]


# ── the screen's whole table, beside its display rule (decision 4) ──────────

def test_the_screen_csv_is_the_recorded_table_row_for_row(scripted):
    import csv
    import hashlib
    import domain_methods as M
    out, sid = scripted["out"], scripted["ids"]["screen"]
    line = CT._read_log(out)[int(sid[1:]) - 1]
    c = line["screen_csv"]
    p = out / c["file"]
    assert c["file"] == f"screen_{sid}.csv" and p.is_file()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == c["sha256"]
    with p.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(M.SCREEN_COLUMNS) == c["columns"] and len(rows) == c["rows"]
    # the complete table: the screen made again is the one logged (its hash),
    # and the CSV holds every row of it with every column, in order
    res = M.screen(M.Suite(scripted["rd"].parent / "suite"), REGION, 2016, [2015])
    assert DAC.fingerprint(res) == line["sha256"]
    cell = lambda v: "" if v is None else (repr(float(v)) if isinstance(v, float)   # noqa: E731
                                           else str(v))
    assert len(rows) == len(res["rows"]) == len(M.catalog_fields()) * 18
    for got, want in zip(rows, res["rows"]):
        assert got == {k: cell(want.get(k)) for k in M.SCREEN_COLUMNS}, (want["field"],
                                                                        want["period"])
    # the lookup's record, row for row: every row holding a value, each shown column
    rec = line["result"]
    held = [r for r in rows if r["value"] != "" or r["change"] != ""]
    assert len(held) == len({k.split(".")[1] for k in rec if k.startswith("rows.")}) > 0
    for i, r in enumerate(held):
        for k in DAC.SHOWN_ROW:
            assert cell(rec.get(f"rows.{i}.{k}")) == r[k], (i, k)
    # the evidence record carries it, and the page links it beside the display rule
    recs = CT.evidence_records(out)
    assert recs[sid]["screen_csv"] == c
    cl, t = _trace(scripted)
    page = S4.render_markdown(_report(scripted, cl, t))
    link = f"is the CSV [04_analysis/{c['file']}]({c['file']})"
    a = page.index("Display rule:", page.index("## The screen"))
    assert a < page.index(link, a) < page.index("| shown | field | period |", a)
    # a screen written by hand, with no CSV, says so
    assert "The whole table: no CSV was written for this lookup." in page
