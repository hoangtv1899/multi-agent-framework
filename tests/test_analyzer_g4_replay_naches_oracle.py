"""The recorded studies of the framework's third acceptance batch
(2026-11-08), fed back through the fixes that followed it, with no model
call. Real data: run with --runlocaldata; the studies are read from
tests/local_oracle_paths.json, key "g4_replay" (each copied to tmp_path,
never written in place).

    pytest tests/test_analyzer_g4_replay_naches_oracle.py --runlocaldata -v

All three were withheld. Question 1 on two claims: claim 2 named 2015 and
spring from a daily series whose dates were never read (a defect of the
trace, now fixed), claim 4 named summer from an annual record (the check
working, still failing). Its plan, both versions, named no timing measure
for a question about an early snow loss: the plan check's timing rule now
fails both, so the recorded replies stop the study before any lookup.
Question 2 on one claim, the "10 cm" that defines its soil-water field (a
defect, now fixed): all fourteen claims trace and its report publishes.
Question 3 on two failures that are the checks working (a hypothesis
headline from an observation; months written "6,7,8"): both remain, and
step 3 is now shown each parameter's form.
"""
import json
import shutil
from pathlib import Path

import pytest

from g2_replay_support import Replies, forbid_model_clients

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import step0_context as S0          # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402
from agents.analysis import step4_report as S4           # noqa: E402

pytestmark = pytest.mark.localdata
CONFIG = Path(__file__).resolve().parent / "local_oracle_paths.json"
# the code-only report's files, written before the Analyzer starts
BEFORE_THE_ANALYZER = ("DOMAIN_REPORT.md", "domain_report.json", "comparison.json")
# the timing rule as revised the same day: the measure of what "early" is
# about (snow loss: snow_gone) for the year asked, and no direction asked
TIMING_PROBLEMS = [
    "timing: the question's 'early' is about snow loss, but no analysis has snow_gone among "
    "its outputs for 2015"]


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


@pytest.fixture(scope="module")
def replay():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    d = (json.loads(CONFIG.read_text()).get("g4_replay") or {}).get("dir")
    if not d or not (Path(d) / "q2" / "04_analysis" / "lookups.jsonl").is_file():
        pytest.skip("no g4_replay block naming the recorded studies")
    return Path(d)


def _copy(replay, q, tmp_path):
    rd = tmp_path / q
    shutil.copytree(replay / q, rd)
    return rd


def _j(p):
    return json.loads(Path(p).read_text())


def _trace(rd):
    out = rd / "04_analysis"
    ctx = S0.load(str(rd))
    interp = _j(out / "interpretation.json")
    claims = [dict(c) for c in interp["claims"]]
    head = interp["headline"]
    if AP.load_plan(out).get("question_type") == "exploratory":
        head = S4.exploratory_headline(head)
    return CT.trace_evidence(claims, head, out, study_dir=rd, regions=S4.suite_regions(ctx),
                             open_causes=interp.get("open_causes") or [])


def _report(rd):
    out = rd / "04_analysis"
    return S4.build(S0.load(str(rd)), {}, _j(out / "investigation.json"),
                    _j(out / "interpretation.json"), rd)


def test_q1_claim_2_now_traces_on_its_daily_series_and_claim_4_still_fails(replay, tmp_path):
    rd = _copy(replay, "q1", tmp_path)
    out = rd / "04_analysis"
    recorded = _j(out / "analysis.json")["evidence_trace"]
    assert recorded["claims_failed"] == [2, 4]                           # as it ran
    assert recorded["problems"]["2"] == ["it names 2015, which none of its records names",
                                         "it names MAM, which none of its records names"]
    t = _trace(rd)
    assert t["claims_failed"] == [4]
    # claim 4 names summer and cites the year's anomaly only: strict, still failing
    assert t["problems"][4] == ["it names JJA, which none of its records names"]
    l4, l7 = t["records"]["L4"], t["records"]["L7"]
    assert (l4["call"], l4["step"]) == ("get_series", "day")
    assert l4["names"]["years"] == [str(y) for y in range(2015, 2024)]
    assert ("season", "MAM") in {tuple(p) for p in l4["names"]["periods"]}
    assert (l7["call"], l7["period"]) == ("anomaly", "2015") and \
        ("season", "JJA") not in {tuple(p) for p in l7["names"]["periods"]}
    rep = _report(rd)
    assert rep["withheld_because"] == "1 of 10 claims fail the evidence trace (claims 4)"
    assert (rep.get("evidence_numbers") or {}).get("pass") is True


def test_q1_the_timing_rule_fails_both_recorded_plans(replay, tmp_path):
    rd = _copy(replay, "q1", tmp_path)
    out = rd / "04_analysis"
    ctx = S0.load(str(rd))
    names = AP._catalog_names(ctx)
    ny = AP.planner_normal_years(ctx)
    v1, v2 = _j(out / AP.PLAN_V1_FILE), _j(out / AP.PLAN_FILE)
    assert v2["question"] == ("Did an early snow loss in 2015 accompany the summer GPP decline "
                              "over the watershed?")
    assert AP.timing_words(v2["question"]) == ["early"]
    assert AP.plan_problems(v1, *names, ny, "hypothesis") == v1["problems"] + TIMING_PROBLEMS
    assert v2["problems"] == [] and AP.plan_problems(v2, *names, ny, "hypothesis") == \
        TIMING_PROBLEMS
    # the review that passed it saw the gap in words, and only words
    rv = _j(out / "analysis_plan_review.json")
    assert rv["passed"] is True and rv["code_problems"] == []
    assert any("melt-out" in p for p in rv["model_problems"])


def test_q1_its_recorded_replies_now_stop_the_study_before_any_lookup(replay, tmp_path):
    """The planning stage again, from a fresh analysis folder, with the
    model's recorded replies given back: both plans fail the code check, the
    rewrite is told why, and no framework call and no figure is made."""
    rd = _copy(replay, "q1", tmp_path)
    out = rd / "04_analysis"
    rec = {n: (out / n).read_text() for n in ("step2a_round1_reply.txt", "step2a_round2_reply.txt",
                                               "step3_plan_review_round1_reply.txt",
                                               "step3_plan_review_round2_reply.txt")}
    for p in out.iterdir():
        if p.name not in BEFORE_THE_ANALYZER:
            p.unlink()
    r = Replies({"step2a_plan": [rec["step2a_round1_reply.txt"]],
                 "step3_plan_review": [rec["step3_plan_review_round1_reply.txt"]],
                 "step2a_plan_rewrite": [rec["step2a_round2_reply.txt"]],
                 "step3_plan_review_rewrite": [rec["step3_plan_review_round2_reply.txt"]]})
    S3.investigate_and_interpret(S0.load(str(rd)), out, max_rounds=2, with_images=False,
                                 clients=r.clients())
    assert r.labels_asked() == ["step2a_plan", "step3_plan_review", "step2a_plan_rewrite",
                                "step3_plan_review_rewrite"]
    assert _j(out / AP.PLAN_V1_FILE)["problems"][-len(TIMING_PROBLEMS):] == TIMING_PROBLEMS
    assert _j(out / AP.PLAN_FILE)["problems"] == TIMING_PROBLEMS
    rewrite = " ".join(r.prompts("step2a_plan_rewrite")[0].split())
    assert all(" ".join(p.split()) in rewrite for p in TIMING_PROBLEMS)
    assert not AP.plan_passed(out)[0]
    assert not (out / AP.LOOKUPS_FILE).exists() and not (out / AP.REQUIRED_FILE).exists()


def test_q2_and_q3_meet_the_timing_rule_as_they_ran(replay, tmp_path):
    for q in ("q2", "q3"):
        rd = _copy(replay, q, tmp_path)
        out = rd / "04_analysis"
        ctx = S0.load(str(rd))
        plan = _j(out / AP.PLAN_FILE)
        assert AP.timing_words(plan["question"]) == [], q
        assert AP.plan_problems(plan, *AP._catalog_names(ctx), AP.planner_normal_years(ctx),
                                plan["question_type"]) == [], q


def test_q2_every_claim_traces_and_its_report_publishes(replay, tmp_path):
    rd = _copy(replay, "q2", tmp_path)
    out = rd / "04_analysis"
    recorded = _j(out / "analysis.json")["evidence_trace"]
    assert recorded["claims_failed"] == [10] and recorded["problems"]["10"] == [
        "the number 10 cm matches no value of its records (L48)"]           # as it ran
    t = _trace(rd)
    assert t["claims_failed"] == [] and t["headline"]["problems"] == []
    l48 = t["records"]["L48"]
    assert l48["field"] == "SOILWATER_10CM" and \
        [tuple(q) for q in l48["field_quantities"]] == [(10.0, "cm")]
    rep = _report(rd)
    assert rep["evidence_trace"]["claims_failed"] == [] and not rep.get("withheld_because")
    assert (rep.get("evidence_numbers") or {}).get("pass") is True
    # its headline: the claim the reply marked (a reply recorded before the
    # ranking), its own traced text, labelled; the reply's own sentence kept
    interp = _j(out / "interpretation.json")
    assert rep["headline_selection"]["chosen"] == 1
    assert rep["headline"] == S4.EXPLORATORY_PREFIX + interp["claims"][0]["claim"]
    assert rep["headline_as_written"] == interp["headline"]


def test_q3_only_the_open_cause_remains_and_step_3_now_sees_the_months_form(replay, tmp_path):
    rd = _copy(replay, "q3", tmp_path)
    out = rd / "04_analysis"
    recorded = _j(out / "analysis.json")["evidence_trace"]
    assert recorded["claims_failed"] == []                               # as it ran
    t = _trace(rd)
    assert t["claims_failed"] == []
    assert t["headline"]["problems"] == [
        "claim 1 is observation, not tested: a hypothesis question's headline comes from "
        "tested claims only"]
    assert t["open_cause_problems"] == [
        "open cause 1: 'weather_scale' cannot be made as proposed: weather_scale "
        "months='6,7,8' is not a list of distinct months 1-12"]
    # what step 3 was shown then, and what it is shown now
    then = " ".join((out / "step3_round1_prompt.txt").read_text().split())
    assert "months (months)" in then and "a list of distinct month numbers" not in then
    now = " ".join(S3.suite_task().split())
    assert "months: a list of distinct month numbers 1-12, as [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, " \
        "11, 12] (the months it applies in, 1-12)" in now
    # the report's headline is no longer written: claims 1 and 2 (observations)
    # are passed over and claim 3, the first tested expectation, heads it; the
    # report is withheld for the open cause alone, the check working
    rep = _report(rd)
    sel = rep["headline_selection"]
    assert sel["chosen"] == 3 and [p["claim"] for p in sel["passed_over"]] == [1, 2]
    assert rep["headline"] == _j(out / "interpretation.json")["claims"][2]["claim"]
    assert rep["withheld_because"] == t["open_cause_problems"][0]


# ── the second set of decisions (2026-11-08): the clarification, the specs ──

def test_q1s_catalog_says_which_products_lack_wy2015_and_how_each_timing_measure_answers(
        replay, tmp_path):
    rd = _copy(replay, "q1", tmp_path)
    cat = " ".join(AP.catalog_text(S0.load(str(rd))).split())
    assert ("Suite periods left out, wateryear WY2015: WY2015 needs October-December 2014, "
            "which the run does not hold whole (its output starts in 2015), so it is left out, "
            "not made from nine months. Only the wateryear products lack it; snow_gone dates it "
            "as a partial-record estimate; snow_peak and half_runoff are not estimable.") in cat
    # a left-out season says nothing more: no timing measure is of a season's kind
    assert ("so it is left out, not made from two months. - Suite periods left out, "
            "wateryear WY2015:") in cat
    import domain_catalog as C
    assert C.MAX_CHARS - len(C.render_parts(C.build(AP._registry_entry("naches")))["fixed"]) \
        >= 0


def _draw(rd, panels, items, fid="fig"):
    from agents.analysis import step2_investigate as S2
    out = rd / "04_analysis"
    ctx = S0.load(str(rd))
    plan = AP.load_plan(out)
    f = {"id": fid, "question": "q", "why": "w", "scale": "overall", "plan_items": items,
         "amendment": None, "extra": [], "panels": panels}
    return S2._spec_figure(ctx, f, out, AP.produce_required(ctx, out), plan, 9, [])


def test_q1s_blank_panels_are_drawn_from_the_framework_outputs_as_a_specification(replay,
                                                                                 tmp_path):
    """The batch's third figure had two empty panels (the quarter split and
    the band split), seen only by step 3's eyes. As a specification on the
    same recorded outputs, every panel holds data, each names its record."""
    rd = _copy(replay, "q1", tmp_path)
    o, f, cav = _draw(rd, [{"kind": "map", "output": "P2.O5"},
                           {"kind": "overlap", "output": "P3.O1"},
                           {"kind": "bars", "outputs": ["P5.O1"]}], ["P3", "P5"],
                      fid="gpp_snow_overlap_and_bands")
    assert o["ok"] and cav is None
    assert [(p["kind"], p["status"]) for p in o["panels"]] == [
        ("map", "drawn"), ("overlap", "drawn"), ("bars", "drawn")]
    assert all(p["n"] > 0 for p in o["panels"])
    lines = {c["label"]: c["line"] for c in _j(rd / "04_analysis" / AP.REQUIRED_FILE)["calls"]}
    assert [x["id"] for x in f["lookups"]] == [f"L{lines[k]}" for k in ("P2.O5", "P3.O1",
                                                                        "P5.O1")]
    # the season's and the year's totals never share an axis: refused side by
    # side when this batch ran, and since the phase's close (one unit per
    # panel, 2026-11-08) split into a panel each, each with its record
    o, f, cav = _draw(rd, [{"kind": "bars", "outputs": ["P2.O1", "P2.O2"]}], ["P2"])
    assert o["ok"] and cav is None
    assert [(p["outputs"], p["split_from"]) for p in o["panels"]] == [(["P2.O1"], 1),
                                                                      (["P2.O2"], 1)]
    assert [x["id"] for x in f["lookups"]] == [f"L{lines[k]}" for k in ("P2.O1", "P2.O2")]


def test_q2s_timing_and_screen_are_drawn_from_its_recorded_outputs(replay, tmp_path):
    rd = _copy(replay, "q2", tmp_path)
    o, f, _ = _draw(rd, [{"kind": "timing", "output": "P7.O5"},
                         {"kind": "bars", "outputs": ["P7.O1", "P7.O2", "P7.O3", "P7.O6"]}],
                    ["P7"])
    assert o["ok"] and [p["status"] for p in o["panels"]] == ["drawn", "drawn"]
    o, f, _ = _draw(rd, [{"kind": "screen"}], ["screen"], fid="screen_fig")
    assert o["ok"] and o["panels"][0]["n"] > 0
    assert [x["id"] for x in f["lookups"]] == ["L1"]                 # the framework's screen
