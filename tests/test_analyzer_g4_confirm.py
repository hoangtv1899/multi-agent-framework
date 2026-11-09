"""Before the third acceptance batch (2026-11-08): the framework's own calls
of the required outputs keep the plan's approval, the outputs' dependencies
and the screening order. Synthetic, with no model call.

    pytest tests/test_analyzer_g4_confirm.py -v

Plan approval: no call before the plan has passed its review, none after a
failed one, none once the plan changed after its review, and a study whose
plan fails twice makes none at all. Dependencies: every map is made before
the call that takes it, across analyses too, and a dependency not made (its
call refused, or its map empty because the method found it not estimable)
leaves the output that takes it unmet, with which input and why on record.
Screening order: in a hypothesis study the framework's calls all come
before the close and it runs no screen; in an exploratory one its screen is
the first lookup, recorded once. Step 2b is shown the calls, never what
they returned.
"""
import json

import pytest

from g2_replay_support import Replies, forbid_model_clients
from test_analyzer_g2_replay import (FOLLOWS, NO_EVIDENCE, E1, P_GPP, P_MAPS, P_QCH,  # noqa: F401
                                     SPEC_GPP, SPEC_SCREEN, _hypothesis_study, _plan, _run,
                                     _spec)
from test_domain_access import REGION, built, study           # noqa: F401

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import plan_contract as PC          # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402

HYPO = _plan("hypothesis", [P_GPP, P_MAPS, P_QCH], [E1])


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


def _approve(study, plan, qt="hypothesis", follows=True):
    study["ctx"].plan["experiment"]["analysis"]["question_type"] = qt
    c = Replies({"a": [plan], "b": [dict(FOLLOWS, follows=follows)]}).clients()
    assert AP.write_plan(study["ctx"], study["out"], client=c["a"])["problems"] == []
    S3.rule_plan_follows_question(study["ctx"], study["out"], client=c["b"])


def _none_made(study):
    return not (study["out"] / AP.LOOKUPS_FILE).exists() and \
        not (study["out"] / AP.REQUIRED_FILE).exists()


# ── plan approval ───────────────────────────────────────────────────────────

def test_no_framework_call_before_a_review_or_after_a_failed_one(study):
    _hypothesis_study(study)
    c = Replies({"a": [HYPO]}).clients()
    AP.write_plan(study["ctx"], study["out"], client=c["a"])
    with pytest.raises(AP.PlanNotApproved, match="has not been reviewed"):
        AP.produce_required(study["ctx"], study["out"])
    assert _none_made(study)
    S3.rule_plan_follows_question(study["ctx"], study["out"], client=Replies(
        {"b": [dict(FOLLOWS, follows=False, why="it reads the wrong year")]}).clients()["b"])
    with pytest.raises(AP.PlanNotApproved, match="did not pass"):
        AP.produce_required(study["ctx"], study["out"])
    assert _none_made(study)


def test_no_framework_call_once_the_plan_changed_after_its_review(study):
    _approve(study, HYPO)
    p = study["out"] / AP.PLAN_FILE
    edited = json.loads(p.read_text())
    edited["analyses"][0]["why"] = "edited after the review"
    p.write_text(json.dumps(edited, indent=1))
    with pytest.raises(AP.PlanNotApproved, match="changed after its review"):
        AP.produce_required(study["ctx"], study["out"])
    assert _none_made(study)


def test_a_plan_that_fails_twice_makes_no_framework_call_and_no_model_call_after(study):
    _hypothesis_study(study)
    bad = json.loads(json.dumps(HYPO))
    bad["analyses"][0]["methods"] = ["anomaly", "screen"]          # the check refuses it
    r = Replies({"step2a_plan": [bad], "step3_plan_review": [FOLLOWS],
                 "step2a_plan_rewrite": [bad], "step3_plan_review_rewrite": [FOLLOWS]})
    _run(study, r)
    assert r.labels_asked() == ["step2a_plan", "step3_plan_review", "step2a_plan_rewrite",
                                "step3_plan_review_rewrite"]
    assert _none_made(study)


# ── dependencies ────────────────────────────────────────────────────────────

def _deps_plan(gpp_period="2016"):
    # the overlap's analysis comes FIRST in the plan; its maps are others'
    return _plan("hypothesis", [
        {"id": "P1", "question": "together", "fields": [], "regions": [REGION], "periods": [],
         "steps": [], "methods": ["overlap"], "settings": {}, "why": "w",
         "outputs": [{"id": "O1", "kind": "overlap", "region": REGION,
                      "inputs": ["P2.O1", "P3.O1"]}]},
        {"id": "P2", "question": "the stock", "fields": ["TOTECOSYSC"], "regions": [],
         "periods": ["2016"], "steps": [], "methods": [], "settings": {}, "why": "w",
         "outputs": [{"id": "O1", "kind": "map", "field": "TOTECOSYSC", "period": "2016"}]},
        {"id": "P3", "question": "uptake", "fields": ["GPP"], "regions": [],
         "periods": [gpp_period], "steps": [], "methods": ["anomaly"], "settings": {},
         "why": "w",
         "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP", "period": gpp_period,
                      "region": None}]}],
        [dict(E1, analysis="P1")])


def test_every_map_is_made_before_the_call_that_takes_it(study):
    _approve(study, _deps_plan())
    rec = AP.produce_required(study["ctx"], study["out"])["record"]
    order = [r["label"] for r in rec["calls"]]
    assert order.index("P2.O1") < order.index("P1.O1") and order.index("P3.O1") < \
        order.index("P1.O1")
    lines = {r["label"]: r["line"] for r in rec["calls"]}
    ins = CT.derived_inputs(AP.lookups(study["out"]), lines["P1.O1"])
    assert [x["record"] for x in ins] == [f"L{lines['P2.O1']}", f"L{lines['P3.O1']}"]
    ph = AP.close_planned_phase(study["out"], when="before the screen", by="the test",
                                figures=[AP.required_attempt(rec, AP.load_plan(study["out"]))])
    assert {r["id"]: r["status"] for r in ph["analyses"]} == {
        "P1": PC.COMPLETED, "P2": PC.COMPLETED, "P3": PC.COMPLETED}


def test_a_dependency_not_made_leaves_its_dependent_unmet_and_says_why(study):
    """Its input's map is empty: the anomaly of a year the run lacks, which
    the method answers as not estimable. The overlap is not called, and the
    record, the close and acceptance all say so."""
    _approve(study, _deps_plan(gpp_period="2017"))
    rec = AP.produce_required(study["ctx"], study["out"])["record"]
    calls = {r["label"]: r for r in rec["calls"]}
    assert calls["P3.O1"]["ok"] is True                         # answered: not estimable
    assert calls["P1.O1"]["ok"] is False and calls["P1.O1"]["line"] is None
    assert calls["P1.O1"]["error"].startswith("not called: its input map P3.O1 (") and \
        "was not made" in calls["P1.O1"]["error"]
    assert all(x["call"] != "overlap" for x in AP.lookups(study["out"]))
    ph = AP.close_planned_phase(study["out"], when="before the screen", by="the test",
                                figures=[AP.required_attempt(rec, AP.load_plan(study["out"]))])
    rows = {r["id"]: r for r in ph["analyses"]}
    assert rows["P3"]["status"] == PC.COMPLETED and rows["P3"]["not_estimable"] == ["O1"]
    assert rows["P1"]["status"] == PC.FAILED and rows["P1"]["unmet"] == ["O1"]
    assert "the framework's required calls" in rows["P1"]["reason"] and \
        "P1.O1: not called: its input map P3.O1" in rows["P1"]["reason"]
    acc = AP.acceptance(ph)
    assert acc["ok"] is False and [b["analysis"] for b in acc["blockers"]] == ["P1"]


# ── screening order ─────────────────────────────────────────────────────────

def test_in_a_hypothesis_study_the_framework_calls_come_before_the_close_and_no_screen(study):
    _hypothesis_study(study)
    r = Replies({"step2a_plan": [HYPO], "step3_plan_review": [FOLLOWS],
                 "step2_investigate": [{"reasoning": "r", "notes": "", "figures": [
                     _spec("fig_screen", SPEC_SCREEN, ["screen"]),
                     _spec("fig_gpp", SPEC_GPP, ["P1", "P2"])]}],
                 "step3_interpret": [NO_EVIDENCE]})
    _run(study, r)
    log = AP.lookups(study["out"])
    ph = AP.load_planned_phase(study["out"])
    req = [i for i, x in enumerate(log, 1) if x.get("script") == AP.REQUIRED_SCRIPT]
    assert req and req == list(range(1, len(req) + 1))           # first, together
    assert max(req) <= ph["lookups_before"] and all(log[i - 1]["planned"] is True for i in req)
    assert not any(log[i - 1]["call"] == "screen" for i in req)  # the framework runs none
    scr = [i for i, x in enumerate(log, 1) if x.get("call") == "screen" and "sha256" in x]
    assert scr and min(scr) > ph["lookups_before"]               # the screen after the close
    assert log[scr[0] - 1]["script"] == "fig_screen"


def test_in_an_exploratory_study_its_screen_is_first_and_recorded_once(study):
    study["ctx"].plan["experiment"]["analysis"]["question_type"] = "exploratory"
    plan = _plan("exploratory", [dict(P_GPP, id="P1")])
    r = Replies({"step2a_plan": [plan], "step3_plan_review": [FOLLOWS],
                 "step2_investigate": [{"reasoning": "r", "notes": "", "figures": [
                     _spec("fig_screen", SPEC_SCREEN, ["screen"]),
                     _spec("fig_gpp", [{"kind": "bars", "outputs": ["P1.O1"]}], ["P1"])]}],
                 "step3_interpret": [NO_EVIDENCE]})
    loop = _run(study, r)
    log = AP.lookups(study["out"])
    assert (log[0]["call"], log[0]["script"], log[0]["label"]) == \
        ("screen", AP.REQUIRED_SCRIPT, AP.EXPLORATORY_LABEL)
    sr = json.loads((study["out"] / AP.SCREEN_FILE).read_text())
    assert sr["script"] == AP.REQUIRED_SCRIPT                    # by the first screen, once
    # drawn from the framework's own screen, never called again (a figure is a
    # specification since 2026-11-08): its record is the first line
    assert not [x for x in log[1:] if x.get("call") == "screen"]
    fs = [f for f in loop["investigation"]["findings"] if f["id"] == "fig_screen"][0]
    assert [x["id"] for x in fs["lookups"]] == ["L1"]


def test_step_2b_is_shown_the_calls_never_what_they_returned(study):
    _hypothesis_study(study)
    r = Replies({"step2a_plan": [HYPO], "step3_plan_review": [FOLLOWS],
                 "step2_investigate": [{"reasoning": "r", "notes": "", "figures": [
                     _spec("fig_gpp", SPEC_GPP, ["P1", "P2"])]}],
                 "step3_interpret": [NO_EVIDENCE]})
    _run(study, r)
    prompt = r.prompts("step2_investigate")[0]
    assert f'P1.O1: suite.anomaly("GPP", "2016", region="{REGION}")' in prompt
    val = AP.produce_required(study["ctx"], study["out"])["values"]["P1.O1"]["anomaly"]
    for shown in (repr(val), f"{val:.6g}", f"{val:.4g}"):
        assert shown not in prompt, shown
