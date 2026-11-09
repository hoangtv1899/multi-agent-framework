"""The close of the planned-analysis phase (the user's decision on G part 2,
2026-11-08), on the synthetic suite of tests/test_domain_access.py, with no
model call.

    pytest tests/test_analyzer_g2_close.py -v

A hypothesis study's screen used to wait for every lookup its plan declared,
and question 1 of G part 2 never reached its screen. Now the framework closes
the planned phase explicitly, once: every analysis of the reviewed plan is
marked completed, completed after an earlier failure, failed or skipped, with
a reason, by its REQUIRED OUTPUTS (plan_contract, the user's decision of
2026-11-08 that replaced the count of every scope combination); the screen
may run only after; every lookup after the close is exploratory and never a
tested expectation or a hypothesis headline; records made before it keep
their marks; and an analysis the plan's expectations name is required, so a
required analysis skipped or failed keeps the study from acceptance success
(without withholding its report for that alone). The contract itself is
tested in tests/test_plan_contract.py.
"""
import copy

import pytest

import domain_access as DAC
from g2_replay_support import Replies, forbid_model_clients
from test_domain_access import REGION, built, study           # noqa: F401

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402
from agents.analysis import step4_report as S4           # noqa: E402


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


def _a(aid, fields, methods, outputs, periods=("2016",), regions=(REGION,)):
    return {"id": aid, "question": f"{aid}'s question", "fields": list(fields),
            "regions": list(regions), "periods": list(periods), "steps": [],
            "methods": list(methods), "settings": {}, "why": "asked",
            "outputs": [dict(o, id=f"O{i}") for i, o in enumerate(outputs, 1)]}


def _anom(field, period="2016"):
    return {"kind": "anomaly", "field": field, "period": period, "region": REGION}


PLAN = {
    "comparisons": [{"id": "C1", "what": "2016 against the normal year", "run": "control",
                     "against": "normal years", "years": [2016]}],
    "baseline": {"normal_years": [2015], "why": "the Planner's"},
    "analyses": [_a("P1", ["GPP"], ["anomaly"], [_anom("GPP")]),       # completed, required
                 _a("P2", ["GPP", "QCHARGE"], [],                      # one of two made: failed
                    [{"kind": "map", "field": "GPP", "period": "2016"},
                     {"kind": "map", "field": "QCHARGE", "period": "2016"}], regions=()),
                 _a("P3", ["QCHARGE"], ["anomaly"], [_anom("QCHARGE")]),  # skipped, stated
                 _a("P4", ["TOTECOSYSC"], ["anomaly"], [_anom("TOTECOSYSC")]),  # not carried
                 _a("P5", [], ["growth_peak"],                          # its figure failed
                    [{"kind": "growth_peak", "region": REGION, "period": "2016"}]),
                 _a("P6", ["GPP"], [], [{"kind": "map", "field": "GPP", "period": "2017"}],
                    periods=("2017",), regions=())],                    # refused by the data
    "expectations": [{"id": "E1", "analysis": "P1", "field": "GPP", "sign": "higher", "text": "a"},
                     {"id": "E2", "analysis": "P3", "field": "QCHARGE", "sign": "lower",
                      "text": "b"},
                     {"id": "E3", "analysis": "P5", "field": "GPP", "sign": "earlier",
                      "text": "c"}],
    "competing": [], "screen": {"when": "last", "region": REGION, "year": 2016, "why": "last"}}
FOLLOWS = {"follows": True, "why": "it follows", "problems": []}


def _approved(study, plan=PLAN, qt="hypothesis"):
    study["ctx"].plan["experiment"]["analysis"]["question_type"] = qt
    r = Replies({"a": [plan], "b": [FOLLOWS]}).clients()
    AP.write_plan(study["ctx"], study["out"], client=r["a"])
    S3.rule_plan_follows_question(study["ctx"], study["out"], client=r["b"])
    assert AP.plan_passed(study["out"])[0], AP.plan_passed(study["out"])
    return AP.make_access(AP.access_spec(study["ctx"]), script="fig")


def _planned_lookups(acc):
    acc.anomaly("GPP", "2016", region=REGION)                 # P1, all of it
    acc.get_map("control", "GPP", "2016")                     # P2, one of two
    with pytest.raises(DAC.AccessError):
        acc.get_map("control", "GPP", "2017")                 # P6: the suite holds no 2017


FIGURES = [{"id": "fig_p1", "plan_items": ["P1", "P2"], "ok": True, "error": None},
           {"id": "fig_p5", "plan_items": ["P5"], "ok": False,
            "error": "script failed: ZeroDivisionError: division by zero"}]
STATED = {"P3": "the recharge map needs a season the suite lacks"}


def test_the_hypothesis_screen_is_refused_before_the_close_and_allowed_after(study):
    acc = _approved(study)
    _planned_lookups(acc)
    with pytest.raises(DAC.AccessRefused, match="the planned-analysis phase is not closed yet"):
        acc.screen(2016, REGION)
    AP.close_planned_phase(study["out"], when="before the screen", by="the test")
    assert acc.screen(2016, REGION)["method"] == "screen"
    last = AP.lookups(study["out"])[-1]
    assert (last["planned"], last["label"], last["after_close"]) == \
        (False, AP.SCREENING_LABEL, True)


def test_each_analysis_is_completed_failed_or_skipped_with_its_reason(study):
    acc = _approved(study)
    _planned_lookups(acc)
    ph = AP.close_planned_phase(study["out"], when="before the screen", by="the test",
                                figures=FIGURES, skipped=STATED)
    rows = {r["id"]: r for r in ph["analyses"]}
    assert {k: r["status"] for k, r in rows.items()} == {
        "P1": "completed", "P2": "failed", "P3": "skipped", "P4": "skipped", "P5": "failed",
        "P6": "failed"}
    assert rows["P1"]["reason"] == "1 required output produced"
    assert rows["P1"]["produced"] == ["O1"] and rows["P1"]["outputs"][0]["record"] == "L1"
    # carried out by a figure that ran, yet one output never made: failed, not completed
    assert (rows["P2"]["produced"], rows["P2"]["unmet"]) == (["O1"], ["O2"])
    assert rows["P2"]["reason"] == ("O2 unmet: no planned lookup matched it (the figure "
                                    "'fig_p1' naming this analysis ran without it)")
    assert rows["P3"]["reason"] == f"step 2b's reason: {STATED['P3']}"
    assert rows["P4"]["reason"] == AP.NOT_CARRIED
    assert rows["P5"]["reason"] == ("O1 unmet: no planned lookup matched it; the figure "
                                    "'fig_p5' failed: script failed: ZeroDivisionError: "
                                    "division by zero")
    assert rows["P6"]["reason"].startswith("O1 unmet: L3 refused by the data:")
    assert rows["P6"]["outputs"][0]["attempts"][0]["result"] == "refused by the data"
    # the scope is listed beside, and decides nothing
    assert rows["P2"]["scope"]["fields"] == ["GPP", "QCHARGE"] and "never_made" not in rows["P2"]
    # required: the analyses the expectations name, with which
    assert {k for k, r in rows.items() if r["required"]} == {"P1", "P3", "P5"}
    assert rows["P3"]["required_by"] == ["E2"]
    assert ph["counts"] == {"completed": 1, "completed after an earlier failure": 0,
                            "failed": 3, "skipped": 2}
    assert ph["lookups_before"] == 3 and ph["closed"] == "before the screen"
    assert ph["plan_sha256"] == AP.plan_sha256(study["out"])
    # the acceptance blockers: the required ones skipped or failed, an optional never
    acc_ = AP.acceptance(ph)
    assert acc_["ok"] is False
    assert [(b["analysis"], b["status"]) for b in acc_["blockers"]] == [("P3", "skipped"),
                                                                        ("P5", "failed")]
    assert acc_["blockers"][0]["reason"] == rows["P3"]["reason"]


def test_a_lookup_after_the_close_is_exploratory_never_tested_never_a_headline(study):
    acc = _approved(study)
    _planned_lookups(acc)
    before = (study["out"] / AP.LOOKUPS_FILE).read_text()
    AP.close_planned_phase(study["out"], when="before the screen", by="the test")
    acc.anomaly("GPP", "2016", region=REGION)             # P1's scope, after the close
    acc.screen(2016, REGION)
    acc.anomaly("GPP", "2016", region=REGION)             # and after the screen
    # records made before the close keep their marks, byte for byte
    assert (study["out"] / AP.LOOKUPS_FILE).read_text().startswith(before)
    log = AP.lookups(study["out"])
    early, late, screen, later = log[0], log[3], log[4], log[5]
    assert early["planned"] is True and "after_close" not in early
    assert (late["planned"], late["label"], late["after_close"]) == (False, AP.EXPLORATORY_LABEL,
                                                                     True)
    assert "after_screen" not in late and later["after_screen"] is True
    assert screen["label"] == AP.SCREENING_LABEL
    recs = CT.evidence_records(study["out"])
    assert recs["L1"]["after_close"] is False and recs["L4"]["after_close"] is True
    assert "exploratory" in {w["code"] for w in recs["L4"]["warnings"]}
    assert all(w["code"] != "exploratory" for w in recs["L1"]["warnings"])
    after = [w for w in recs["L4"]["warnings"] if w["code"] == "exploratory"][0]
    assert after["reason"] == CT.AFTER_CLOSE_REASON
    plan = AP.load_plan(study["out"])
    v = recs["L4"]["values"]["anomaly"]

    def claim(rid, cls, **kw):
        return {"claim": f"Over the vegetated cells, GPP in 2016 was {v:.4g} from its 2015 "
                         f"normal.", "finding_id": "f", "values": [v], "evidence": [rid],
                "evidence_class": cls,
                "warnings": sorted({w["code"] for w in recs[rid]["warnings"]}), **kw}
    # as a tested expectation it fails, by its own rule, not only as not planned
    t = CT.check_claim(claim("L4", "tested expectation", expectation="E1"), recs, plan, {},
                       [REGION])
    assert any("L4 was made after the planned analyses were closed (planned_phase.json): "
               "never a tested expectation" in p for p in t["problems"]), t["problems"]
    # the same claim on the record made before the close passes
    ok = CT.check_claim(claim("L1", "tested expectation", expectation="E1"), recs, plan, {},
                        [REGION])
    assert ok["problems"] == [], ok["problems"]
    # as the headline of a hypothesis it fails
    cl = [claim("L4", "tested expectation", expectation="E1", headline=True)]
    tr = CT.trace_evidence(cl, "GPP rose.", study["out"], regions=[REGION])
    assert any("made after the planned analyses were closed: never a hypothesis headline" in p
               for p in tr["headline"]["problems"])
    # as an observation it must carry the after-close warning
    obs = claim("L4", "observation")
    obs["warnings"] = [w for w in obs["warnings"] if w != "exploratory"]
    t = CT.check_claim(obs, recs, plan, {}, [REGION])
    assert any("(exploratory) its record L4 carries" in p for p in t["problems"])
    # an amendment after the screen is labelled exploratory, and says it came after
    a1 = AP.append_amendment(study["out"], "the stock", {"fields": ["TOTECOSYSC"],
                                                         "methods": ["anomaly"]})
    assert (a1["label"], a1["after_screen"], a1["after_close"]) == (AP.EXPLORATORY_LABEL,
                                                                     True, True)


def test_a_line_after_the_close_without_the_mark_is_still_after_the_close(study):
    """By its line, not only by the gate's mark: a line logged after the
    lines the close counted is read as after the close."""
    acc = _approved(study)
    _planned_lookups(acc)
    AP.close_planned_phase(study["out"], when="before the screen", by="the test")
    AP.LookupLog(study["out"], script="by_hand")({
        "call": "anomaly", "keys": {"run": "control", "field": "GPP", "period": "2016",
                                    "region": REGION},
        "method": "anomaly", "settings": None, "sha256": "c" * 64, "planned": True,
        "plan_items": ["P1"], "result": {"anomaly": 1.0, "covered": 1.0}})
    rec = CT.evidence_records(study["out"])["L4"]
    assert rec["planned"] is True and rec["after_close"] is True
    assert "exploratory" in {w["code"] for w in rec["warnings"]}


def test_the_close_is_written_once_by_the_framework(study, monkeypatch):
    acc = _approved(study)
    _planned_lookups(acc)
    monkeypatch.setenv(AP.IN_SCRIPT_ENV, "1")
    with pytest.raises(AP.PlanNotApproved, match="never by a figure script"):
        AP.close_planned_phase(study["out"], when="before the screen", by="a script")
    monkeypatch.delenv(AP.IN_SCRIPT_ENV)
    first = AP.close_planned_phase(study["out"], when="before the screen", by="the test",
                                   figures=FIGURES, skipped=STATED)
    text = (study["out"] / AP.PLANNED_PHASE_FILE).read_text()
    again = AP.close_planned_phase(study["out"], when="after the last round", by="again",
                                   figures=[], skipped={})
    assert again == first and (study["out"] / AP.PLANNED_PHASE_FILE).read_text() == text
    # the figure scripts' environment says so
    from agents.analysis import script_runner as SR
    assert SR._child_env()[AP.IN_SCRIPT_ENV] == "1"


def test_an_exploratory_study_requires_nothing(study):
    plan = dict(copy.deepcopy(PLAN), expectations=[],
                screen={"when": "first", "region": REGION, "year": 2016, "why": "first"})
    acc = _approved(study, plan=plan, qt="exploratory")
    acc.screen(2016, REGION)
    ph = AP.close_planned_phase(study["out"], when="after the last round", by="the test")
    assert not any(r["required"] for r in ph["analyses"])
    a = AP.acceptance(ph)
    assert a["ok"] is True and a["blockers"] == [] and "no required analysis" in a["why"]


def test_step_4_records_the_phase_and_says_the_hypothesis_was_not_fully_tested(study,
                                                                             monkeypatch):
    from agents.analysis import numbers_check as NC
    # the numbers check of package F2 has its own tests; it is stood in for here
    monkeypatch.setattr(NC, "check_evidence", lambda *a, **k: {
        "pass": True, "n_compared": 0, "mismatches": [], "records": {}})
    acc = _approved(study)
    _planned_lookups(acc)
    AP.close_planned_phase(study["out"], when="before the screen", by="the test",
                           figures=FIGURES, skipped=STATED)
    recs = CT.evidence_records(study["out"])
    v = recs["L1"]["values"]["anomaly"]
    claim = {"claim": f"Over the vegetated cells, GPP in 2016 was {v:.4g} gC/m^2/yr from its "
                      f"2015 normal, consistent with the plan's expectation.",
             "finding_id": "fig_p1", "values": [v], "caveats": [], "evidence": ["L1"],
             "evidence_class": "tested expectation", "expectation": "E1",
             "warnings": sorted({w["code"] for w in recs["L1"]["warnings"]}), "headline": True}
    rep = S4.build(study["ctx"], {}, {"findings": []},
                   {"verdict": "sufficient", "headline": claim["claim"], "claims": [claim]},
                   study["rd"])
    assert rep["evidence_trace"]["claims_failed"] == [], rep["evidence_trace"]["problems"]
    assert not rep.get("withheld_because")             # not withheld for this alone
    assert rep["planned_phase"]["counts"] == {"completed": 1,
                                              "completed after an earlier failure": 0,
                                              "failed": 3, "skipped": 2}
    assert rep["acceptance"]["ok"] is False
    assert [b["analysis"] for b in rep["acceptance"]["blockers"]] == ["P3", "P5"]
    page = S4.render_markdown(rep)
    first = page.split("\n")[2]
    assert first.startswith("**The hypothesis was not fully tested.** 2 required analyses")
    assert "P3 was skipped (step 2b's reason: the recharge map" in first
    assert "This does not count as acceptance success." in first
    assert "## Planned analyses" in page
    assert "| P3 | skipped | yes (E2) | none | none | O1 | step 2b's reason:" in page
    assert "| P4 | skipped | no | none | none | O1 | no figure carried it out" in page
    assert "| P1 | completed | yes (E1) | O1 (L1) | none | none | 1 required output produced |" \
        in page
    # each expectation: its analysis's completion beside the tested claims that name it,
    # kept apart (completion is not support)
    from agents.analysis import plan_contract as PC
    assert f"{PC.SUPPORT_WORDS}." in page
    assert "| E1 | P1 | completed | none | claim 1 |" in page
    assert "| E2 | P3 | skipped | none | none |" in page
    # a study whose required analyses all completed has no such line
    rep2 = dict(rep, acceptance={"ok": True, "blockers": []})
    assert "**The hypothesis was not fully tested.**" not in S4.render_markdown(rep2)
