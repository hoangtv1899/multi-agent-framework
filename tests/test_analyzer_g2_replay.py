"""The failures of G part 2 (2026-11-08), each replayed in a small scripted
study with no model call, through the real code paths.

    pytest tests/test_analyzer_g2_replay.py -v

Each test hands the real Analyzer a scripted reply of the shape that failed
in part 2 (the acceptance notes, kept out of the repository) and checks the fix:

  - claims that name no evidence record (question 1): step 3's brief now
    lists each finding's records and its template asks for evidence,
    evidence_class, expectation, warnings and headline; the trace still
    fails a claim with no record;
  - the screen listed as an analysis (question 2) and a setting no method
    has (question 3): the plan's task states both rules, the check still
    refuses both, the review gives the check's own reason, and one rewrite
    before any lookup either passes or stops the study with both versions'
    problems on its withheld page; step 3 is not asked then;
  - a percent of a field whose notes give none (question 1's BTRAN): step
    3's own audit strikes it;
  - dashes in the model's words (question 1's plan review): kept out of
    every record, the verbatim exchanges aside;
  - five labelled calls (question 1): the printed spend counts all five;
  - a hypothesis screen before the close of the planned phase: refused, and
    the figures are run in the order the close needs.

Every model reply is scripted (g2_replay_support.Replies, keyed by the
client's label); SimpleLLMClient raises if anything constructs one.
"""
import copy
import json
from pathlib import Path

import pytest

from g2_replay_support import Replies, forbid_model_clients, record_strings
from test_domain_access import REGION, built, study           # noqa: F401

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402
from agents.analysis import step4_report as S4           # noqa: E402

EM = "\u2014"


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


def _plan(qt, analyses, expectations=(), when=None):
    return {"comparisons": [{"id": "C1", "what": "2016 against the normal year", "run": "control",
                             "against": "normal years", "years": [2016]}],
            "baseline": {"normal_years": [2015], "why": "the Planner's"},
            "analyses": list(analyses), "expectations": list(expectations),
            "competing": [{"text": "soil water", "fields": ["H2OSOI"], "analysis": "P1"}],
            "screen": {"when": when or AP.SCREEN_WHEN[qt], "region": REGION, "year": 2016,
                       "why": "the question type's order"}}


# each with its required outputs, apart from its scope (plan_contract, 2026-11-08)
P_GPP = {"id": "P1", "question": "GPP in 2016", "fields": ["GPP"], "regions": [REGION],
         "periods": ["2016"], "steps": [], "methods": ["anomaly"], "settings": {},
         "why": "what was asked",
         "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2016",
                      "region": REGION}]}
P_MAPS = {"id": "P2", "question": "the maps", "fields": ["GPP", "QCHARGE"], "regions": [],
          "periods": ["2016"], "steps": [], "methods": [], "settings": {}, "why": "to draw them",
          "outputs": [{"id": "O1", "kind": "map", "field": "GPP", "period": "2016"},
                      {"id": "O2", "kind": "map", "field": "QCHARGE", "period": "2016"}]}
P_QCH = {"id": "P3", "question": "recharge in 2016", "fields": ["QCHARGE"], "regions": [REGION],
         "periods": ["2016"], "steps": [], "methods": ["anomaly"], "settings": {},
         "why": "a competing path",
         "outputs": [{"id": "O1", "kind": "anomaly", "field": "QCHARGE", "period": "2016",
                      "region": REGION}]}
E1 = {"id": "E1", "analysis": "P1", "field": "GPP", "sign": "higher",
      "text": "GPP higher in 2016 than in the normal year"}
HYPO = _plan("hypothesis", [P_GPP, P_MAPS, P_QCH], [E1])
# QUESTION 2'S SHAPE: the screen listed as an analysis as well as in its entry
EXPLO_V1 = _plan("exploratory", [dict(P_GPP, id="P1", methods=["screen"],
                                      outputs=[{"id": "O1", "kind": "map", "field": "GPP",
                                                "period": "2016"}]),
                                 dict(P_GPP, id="P2")])
EXPLO_V2 = _plan("exploratory", [dict(P_GPP, id="P2")])
# QUESTION 3'S SHAPE: a call's argument written as a setting keyed by no method
BY_BAND = _plan("hypothesis", [P_GPP, P_MAPS, {"id": "P5", "question": "maps together",
                                               "fields": [], "regions": [REGION],
                                               "periods": [], "steps": [],
                                               "methods": ["overlap"],
                                               "settings": {"by_band": True},
                                               "why": "where",
                                               "outputs": [{"id": "O1", "kind": "overlap",
                                                            "region": REGION,
                                                            "inputs": ["P2.O1", "P2.O2"]}]}],
                [E1])
FOLLOWS = {"follows": True, "why": "it addresses the question as asked", "problems": []}

FIG_GPP = '''
a = suite.anomaly("GPP", "2016", region="vegetated cells")
x = suite.get_map("control", "GPP", "2016")
draw_map(x, out_path=out_path, units="gC/m^2/yr")
result = {"anomaly": float(a["anomaly"]), "tripled": float(a["anomaly"]) * 3.0,
          "n": int(np.isfinite(x).sum())}
'''
FIG_QCH = '''
q = suite.anomaly("QCHARGE", "2016", region="vegetated cells")
x = suite.get_map("control", "QCHARGE", "2016")
draw_map(x, out_path=out_path, units="mm/yr")
result = {"anomaly": float(q["anomaly"]), "n": int(np.isfinite(x).sum())}
'''
FIG_SCREEN = '''
s = suite.screen(2016, "vegetated cells")
plt.bar(range(3), [1, 2, 3]); plt.savefig(out_path)
result = {"rows": len(s["rows"]), "n": 3}
'''


def _figure(fid, code, items, **kw):
    return {"id": fid, "question": f"what {fid} shows", "why": f"{fid} adds this",
            "scale": "overall", "plan_items": items, "amendment": None,
            "variables": ["GPP"], "code": code, **kw}


# A FIGURE IS A SPECIFICATION THE FRAMEWORK DRAWS (the user's decision of
# 2026-11-08, after the third acceptance batch; no script in this phase):
# each figure of a study with a suite names outputs by label in its panels.
# A figure that carries code (_figure) is refused before any call.
SPEC_GPP = [{"kind": "bars", "outputs": ["P1.O1"]}, {"kind": "map", "output": "P2.O1"}]
SPEC_SCREEN = [{"kind": "screen"}]


def _spec(fid, panels, items, **kw):
    return {"id": fid, "question": f"what {fid} shows", "why": f"{fid} adds this",
            "scale": "overall", "plan_items": items, "amendment": None, "extra": [],
            "panels": panels, **kw}


# question 1's shape: claims with a finding and values, no record and no class;
# worded to pass the readability gate, so step 3 is asked once
NO_EVIDENCE = {"claims": [{"claim": "Over the vegetated cells, carbon uptake (GPP) rose in "
                                    "2016.",
                           "finding_id": "fig_gpp", "values": [], "caveats": []}],
               "headline": "This experiment cannot answer it yet: carbon uptake (GPP) rose in "
                           "2016.",
               "answer": "Carbon uptake (GPP) rose.", "verdict": "sufficient", "feedback": ""}


def _hypothesis_study(study, plan=HYPO):
    study["ctx"].plan["experiment"]["analysis"]["question_type"] = "hypothesis"
    return study


def _run(study, replies, max_rounds=1):
    return S3.investigate_and_interpret(study["ctx"], study["out"], max_rounds=max_rounds,
                                        with_images=False, clients=replies.clients())


# ── fix 1: step 3 sees the records; the trace still needs them ──────────────

def test_claims_without_evidence_still_fail_and_the_brief_now_shows_each_record(study):
    _hypothesis_study(study)
    replies = Replies({"step2a_plan": [HYPO], "step3_plan_review": [FOLLOWS],
                       "step2_investigate": [{"reasoning": "the plan", "notes": "",
                                              "figures": [_spec("fig_gpp", SPEC_GPP,
                                                                ["P1", "P2"])]}],
                       "step3_interpret": [NO_EVIDENCE]})
    loop = _run(study, replies)
    inv = loop["investigation"]
    # the figure's finding; the framework's own findings of the required
    # outputs come first since 2026-11-08
    f = [x for x in inv["findings"] if x["id"] == "fig_gpp"][0]
    ids = [x["id"] for x in f["lookups"]]
    assert ids and all(x.startswith("L") for x in ids)
    brief = replies.prompts("step3_interpret")[0]
    recs = CT.evidence_records(study["out"])
    for i in ids:                                    # every record, with its warning codes
        assert f"          {i}: " in brief, i
        for w in recs[i]["warnings"]:
            assert f"{w['code']} (\"{w['label']}\")" in brief, (i, w["code"])
    # a record's value is listed under that record; a figure the framework
    # drew made no number of its own (2026-11-08: specifications, no script)
    sec = brief[brief.index("    id: fig_gpp"):]
    made = sec[sec.index("numbers the script made itself, a value of no record"):]
    assert made.split("\n")[1].strip() == "none"
    first = sec[sec.index(f"          {ids[0]}: "):]
    first = first[:first.index("\n          L") if "\n          L" in first else None]
    listed = first[first.index("its values in the finding's result: "):]
    assert f"P1.O1.anomaly = {f['result']['P1.O1']['anomaly']!r}" in listed.split("\n")[0]
    # the template asks for every part the trace reads
    for part in ("evidence", "evidence_class", "expectation", "warnings", "headline_rank"):
        assert f'"{part}": ' in brief, part
    assert S3.TASK_REPLY not in brief
    # and the trace is as strict as it was: a claim with no record fails
    rep = S4.build(study["ctx"], {}, inv, loop["interpretation"], study["rd"])
    t = rep["evidence_trace"]
    assert t["claims_failed"] == [1] and rep.get("withheld_because")
    assert any(p.startswith("no record: every empirical claim names the logged lookup")
               for p in t["problems"][1])


# ── fixes 2, 3 and the rewrite: the plan check's rules, its reason, one rewrite ─

def test_the_task_states_the_two_rules_questions_2_and_3_broke():
    flat = " ".join(AP.plan_task().split())
    assert "The screen is never an analysis's method: it is declared once, in the plan's own " \
           "\"screen\" entry" in flat
    assert "settings: keyed by method id, then setting name" in flat
    assert "a call's own arguments (overlap's by_band, a timing measure's against) are not " \
           "settings" in flat
    assert "- overlap: sign_share_limit, default 0.8, no alternative (share of the area)." in flat
    # the check is as strict as it was
    got = AP.plan_problems(BY_BAND, ["GPP", "QCHARGE"], [REGION], [2015], "hypothesis")
    assert got == ["P5: settings for no method 'by_band'"]
    got = AP.plan_problems(EXPLO_V1, ["GPP"], [REGION], [2015], "exploratory")
    assert got == ["P1: the screen is declared in the plan's screen entry, not as an analysis"]


def test_a_failed_plan_is_rewritten_once_before_any_lookup_and_the_rewrite_passes(study):
    """Question 2's shape: the review approves, code fails the plan; the
    rewrite gets the same check and the same review prompt, and only then
    does any lookup run."""
    figs = [_figure("fig_screen", FIG_SCREEN, ["screen"]), _figure("fig_gpp", FIG_GPP, ["P2"])]
    replies = Replies({"step2a_plan": [EXPLO_V1], "step3_plan_review": [FOLLOWS],
                       "step2a_plan_rewrite": [EXPLO_V2],
                       "step3_plan_review_rewrite": [FOLLOWS],
                       "step2_investigate": [{"reasoning": "screen first", "notes": "",
                                              "figures": figs}],
                       "step3_interpret": [NO_EVIDENCE]})
    loop = _run(study, replies)
    out = study["out"]
    assert replies.labels_asked() == ["step2a_plan", "step3_plan_review", "step2a_plan_rewrite",
                                      "step3_plan_review_rewrite", "step2_investigate",
                                      "step3_interpret"]
    v1, r1 = (json.loads((out / n).read_text()) for n in (AP.PLAN_V1_FILE, AP.REVIEW_V1_FILE))
    v2, r2 = AP.load_plan(out), AP.load_review(out)
    assert v1["problems"] == ["P1: the screen is declared in the plan's screen entry, not as "
                              "an analysis"] and r1["passed"] is False
    assert r1["follows"] is True and r1["code_problems"] == v1["problems"]
    assert v2["problems"] == [] and v2["attempt"] == 2 and v2["rewrite_of"] == AP.PLAN_V1_FILE
    assert r2["passed"] is True and r2["plan_sha256"] == AP.plan_sha256(out)
    # the first review still ties to the first plan, kept as it was
    import hashlib
    assert r1["plan_sha256"] == hashlib.sha256((out / AP.PLAN_V1_FILE).read_bytes()).hexdigest()
    # the rewrite's review: the same template, not told it is a rewrite
    p1, p2 = replies.prompts("step3_plan_review")[0], \
        replies.prompts("step3_plan_review_rewrite")[0]
    assert p1.endswith(S3.PLAN_REVIEW_ASK) and p2.endswith(S3.PLAN_REVIEW_ASK)
    assert "rewrite" not in p2.lower() and "first plan" not in p2.lower()
    # the rewrite's own prompt: the same inputs, the first plan and what was found
    rw = replies.prompts("step2a_plan_rewrite")[0]
    first = replies.prompts("step2a_plan")[0]
    assert rw.startswith(first) and "YOUR FIRST PLAN:" in rw
    assert "the code check: P1: the screen is declared in the plan's screen entry" in rw
    # both exchanges saved; lookups only after the rewrite passed
    for n in ("step2a_round1_reply.txt", "step2a_round2_reply.txt",
              "step3_plan_review_round1_reply.txt", "step3_plan_review_round2_reply.txt"):
        assert (out / n).is_file(), n
    log = AP.lookups(out)
    assert log and all(r["at"] >= r2["at"] for r in log)
    assert loop["interpretation"].get("skipped") is not True


def test_a_second_plan_still_wrong_stops_with_both_versions_on_the_page(study):
    replies = Replies({"step2a_plan": [EXPLO_V1], "step3_plan_review": [FOLLOWS],
                       "step2a_plan_rewrite": [EXPLO_V1],
                       "step3_plan_review_rewrite": [{"follows": False,
                                                      "why": "it reads the wrong year",
                                                      "problems": ["P2 reads 2016 only"]}]})
    loop = _run(study, replies, max_rounds=2)
    out = study["out"]
    # never a third attempt, step 2b and step 3 never asked
    assert replies.labels_asked() == ["step2a_plan", "step3_plan_review", "step2a_plan_rewrite",
                                      "step3_plan_review_rewrite"]
    with pytest.raises(AP.PlanNotApproved, match="there is no third attempt"):
        AP.rewrite_plan(study["ctx"], out, client=Replies({}).clients().get("x"))
    assert loop["interpretation"]["skipped"] is True and loop["n_rounds"] == 1
    ok, why = AP.plan_passed(out)
    # THE REAL REASON: the code check first, then the model review, labelled
    assert not ok and why.startswith("the analysis plan did not pass step 3's review: the code "
                                     "check found 1 problem: P1: the screen is declared")
    assert "the model review said the plan does not follow from the question (it reads the " \
           "wrong year) and listed 1 problem: P2 reads 2016 only" in why
    with pytest.raises(AP.PlanNotApproved, match="the code check found 1 problem"):
        AP.check_gate(out)
    cav = loop["investigation"]["caveats"][0]
    assert cav["id"] == "analysis_plan_not_approved" and "the code check found 1 problem: P1" in \
        cav["statement"]
    rep = S4.build(study["ctx"], {}, loop["investigation"], loop["interpretation"], study["rd"],
                   rounds=loop["rounds"], stopped_because=loop["stopped_because"],
                   steps={"investigate": True, "interpret": "skipped"})
    assert rep["status"] == S4.INSUFFICIENT and "evidence_trace" not in rep
    assert rep["withheld_because"].startswith("the analysis plan did not pass step 3's review: "
                                              "the code check found")
    assert [v["attempt"] for v in rep["analysis_plan"]["versions"]] == [1, 2]
    assert rep["acceptance"]["ok"] is False
    page = S4.render_markdown(rep)
    assert page.startswith("# WITHHELD: the analysis plan did not pass its review")
    assert page.count("- the code check: P1: the screen is declared in the plan's screen "
                      "entry, not as an analysis") == 2
    assert "## Version 1 (analysis_plan.v1.json)" in page and \
        "## Version 2 (analysis_plan.json)" in page
    assert "No lookup was made" in page and "No results were produced" not in page
    assert "the plan does not follow from the question (it reads the wrong year)" in page
    assert not AP.lookups(out)
    assert "skipped: interpret" in S4.summary(rep)


def test_a_rewrite_once_a_lookup_is_logged_is_refused(study):
    AP.write_plan(study["ctx"], study["out"], client=Replies({"a": [EXPLO_V1]}).clients()["a"])
    S3.rule_plan_follows_question(study["ctx"], study["out"],
                                  client=Replies({"a": [FOLLOWS]}).clients()["a"])
    assert AP.can_rewrite(study["out"])[0] is True
    # a lookup (refused, since the plan did not pass) is on the log
    acc = AP.make_access(AP.access_spec(study["ctx"]), script="early")
    with pytest.raises(Exception):
        acc.get_map("control", "GPP", "2016")
    ok, why = AP.can_rewrite(study["out"])
    assert not ok and "before any outcome is looked up" in why
    with pytest.raises(AP.PlanNotApproved, match="before any outcome is looked up"):
        AP.rewrite_plan(study["ctx"], study["out"],
                        client=Replies({"a": [EXPLO_V2]}).clients()["a"])
    assert not (study["out"] / AP.PLAN_V1_FILE).exists()


# ── fix 5: step 3's own audit applies the field notes' percent rule ─────────

def _btran(out, after=False):
    """A BTRAN anomaly, logged as the access logs one (the small grid has no
    BTRAN): 0.7815 against a normal of 0.8342, as question 1's L24."""
    AP.LookupLog(out, script="fig_water")({
        "call": "anomaly", "keys": {"run": "control", "field": "BTRAN", "period": "2016-JJA",
                                    "region": REGION},
        "method": "anomaly", "settings": None, "sha256": "b" * 64, "planned": True,
        "plan_items": ["P1"],
        "result": {"method": "anomaly", "field": "BTRAN", "period": "2016-JJA",
                   "region": REGION, "value": 0.7815, "normal": 0.8342,
                   "anomaly": -0.0527, "reference_range": 0.2609, "covered": 1.0}})
    return f"L{len(CT._read_log(out))}"


def test_step_3s_audit_strikes_a_percent_the_field_notes_forbid(tmp_path):
    out = tmp_path / "04_analysis"
    out.mkdir()
    rid = _btran(out)
    records = CT.evidence_records(out)
    assert records[rid]["percent_allowed"] is False
    pct = -0.0527 / 0.2609 * 100                       # the script's own arithmetic
    finding = {"id": "fig_water", "n": 9, "result": {"btran_pct": pct, "anomaly": -0.0527},
               "lookups": [{"id": rid}]}
    inv = {"findings": [finding]}
    text = (f"Water available to leaves (BTRAN) was {abs(pct):.1f} percent lower in summer "
            f"2016.")
    by_text = {"claim": text, "finding_id": "fig_water", "values": [pct], "caveats": []}
    by_record = dict(by_text, evidence=[rid])
    allowed = {"claim": "BTRAN fell by 0.0527 in summer 2016.", "finding_id": "fig_water",
               "values": [-0.0527], "caveats": []}
    a = S3.audit([by_text, by_record, allowed], inv, [], records=records)
    assert [c["claim"] for c in a["kept"]] == [allowed["claim"]]
    why = records[rid]["percent_why_not"]
    for s in a["struck"]:
        assert s["struck_by"] == "percent" and "a percent of BTRAN, whose notes give none" in \
            s["struck_because"] and why in s["struck_because"]
    # a study without records (no suite) is not judged by this rule
    assert len(S3.audit([by_text], inv, [])["kept"]) == 1
    # and the trace still fails it too, by the same helper
    t = CT.check_claim(dict(by_record, evidence_class="observation", warnings=[]), records,
                       {"question_type": "exploratory"}, {}, [REGION])
    assert any("a percent of BTRAN, whose notes give none" in p for p in t["problems"])


# ── fix 6: the spend counts every call; no dashes in the model's words ──────

def test_the_printed_spend_counts_every_analyzer_call(monkeypatch):
    import agents.llm_agent as LA
    rows = [{"label": lab, "model": "m", "seconds": 1.0, "prompt_tokens": 10 * i,
             "completion_tokens": i}
            for i, lab in enumerate(("step2a_plan", "step3_plan_review", "step2_investigate",
                                     "step3_interpret", "step3_interpret"), 1)]
    monkeypatch.setattr(LA, "USAGE_LOG", rows)
    llm = S4._llm_accounting(expect_calls=True)
    assert llm["analyzer"]["calls"] == 5 == llm["total"]["calls"]
    assert llm["analyzer"]["prompt_tokens"] + llm["analyzer"]["completion_tokens"] == 165
    rep = {"cost": {"llm": llm}, "provenance": {}, "claims": [], "withheld": []}
    assert "llm      : 5 calls, 165 tokens" in S4.summary(rep)
    page = S4.render_markdown(dict(rep, question="q", headline="h", answer="a"))
    assert "- model calls in the Analyzer: 5" in page
    for lab in ("step2a_plan", "step2a_plan_rewrite", "step3_plan_review",
                "step3_plan_review_rewrite", "step2_investigate", "step3_interpret"):
        assert lab in S4.ANALYZER_LABELS, lab


def test_the_models_words_reach_no_record_with_a_dash(study):
    """Every model reply carries em-dashes; every JSON and Markdown record
    the run writes is free of them, the verbatim exchanges aside."""
    _hypothesis_study(study)
    plan = copy.deepcopy(HYPO)
    plan["comparisons"][0]["what"] = f"2016 {EM} against the normal"
    plan["analyses"][0]["why"] = f"asked {EM} directly"
    plan["expectations"][0]["text"] = f"higher {EM} as asked"
    review = {"follows": True, "why": f"it follows {EM} plainly", "problems": []}
    am = {"why": f"the store {EM} too", "fields": ["TOTECOSYSC"], "methods": ["anomaly"]}
    fig = _spec("fig_gpp", SPEC_GPP, ["P1", "P2"], why=f"draws {EM} the map")
    fig_am = _spec("fig_more", [{"kind": "bars", "outputs": ["X1"]}], [], amendment=am,
                   extra=[{"id": "X1", "kind": "anomaly", "field": "TOTECOSYSC",
                           "period": "2016", "region": REGION}])
    step2 = {"reasoning": f"the plan {EM} in order", "notes": f"all {EM} done",
             "figures": [fig, fig_am], "skipped": [{"analysis": "P3", "why": f"later {EM} maybe"}]}
    step3 = dict(NO_EVIDENCE, feedback=f"more {EM} please",
                 open_causes=[{"cause": f"warmth {EM} or not",
                               "proposed_run": {"changes": [{"id": "weather_scale",
                                                             "params": {"temperature_K": 2.0}}]}}])
    replies = Replies({"step2a_plan": [plan], "step3_plan_review": [review],
                       "step2_investigate": [step2], "step3_interpret": [step3]})
    loop = _run(study, replies)
    rep = S4.build(study["ctx"], {}, loop["investigation"], loop["interpretation"], study["rd"])
    S4.write(rep, study["out"], slides=False)
    out = study["out"]
    written = [p for p in out.iterdir() if p.suffix in (".json", ".md", ".jsonl")]
    assert {AP.PLAN_FILE, AP.REVIEW_FILE, AP.AMENDMENTS_FILE, AP.PLANNED_PHASE_FILE,
            "investigation.json", "interpretation.json", "analysis.json"} <= \
        {p.name for p in written}
    model_words = [w for r in (plan, review, am, step2, step3)
                   for w in json.dumps(r, ensure_ascii=False).split('"') if EM in w]
    assert model_words
    for p in written:
        strings = record_strings(p)
        for w in model_words:
            assert not any(w in x for x in strings), (p.name, w)
        if p.name in (AP.PLAN_FILE, AP.REVIEW_FILE, AP.AMENDMENTS_FILE, AP.PLANNED_PHASE_FILE,
                      "investigation.json", "interpretation.json"):
            assert not any(EM in x for x in strings), p.name
    # the verbatim exchanges keep what was said
    assert EM in (out / "step3_plan_review_round1_reply.txt").read_text()
    # the plan's names untouched, its words plain, and the hash of what was written
    assert AP.load_plan(out)["comparisons"][0]["what"] == "2016, against the normal"
    assert AP.load_review(out)["plan_sha256"] == AP.plan_sha256(out)


# ── the close of the planned phase in step 2b's flow ────────────────────────

def test_the_screens_figure_runs_after_the_close_and_a_figure_with_code_is_refused(study):
    """The screen's figure runs after the close, wherever the model lists it.
    A hidden screen call is no longer possible: a figure is a specification,
    and one that carries code (here, a call of the screen hidden from a text
    search) is refused before any call (2026-11-08)."""
    _hypothesis_study(study)
    hidden = '''
s = getattr(suite, "scr" + "een")(2016, "vegetated cells")
plt.bar(range(3), [1, 2, 3]); plt.savefig(out_path)
result = {"n": 3}
'''
    figs = [_spec("fig_screen", SPEC_SCREEN, ["screen"]),          # listed first by the model
            dict(_figure("fig_hidden", hidden, ["P3"]), panels=[
                {"kind": "bars", "outputs": ["P3.O1"]}]),
            _spec("fig_gpp", SPEC_GPP, ["P1", "P2"])]
    replies = Replies({"step2a_plan": [HYPO], "step3_plan_review": [FOLLOWS],
                       "step2_investigate": [{"reasoning": "r", "notes": "", "figures": figs,
                                              "skipped": [{"analysis": "P3",
                                                           "why": "the recharge map waits"}]}],
                       "step3_interpret": [NO_EVIDENCE]})
    loop = _run(study, replies)
    out = study["out"]
    log = AP.lookups(out)
    ph = AP.load_planned_phase(out)
    # the figure with code made no call at all, and says why
    assert not [r for r in log if r.get("script") == "fig_hidden"]
    got = {f["id"]: f for f in loop["investigation"]["figure_outcomes"]}
    assert "a script is not run" in got["fig_hidden"]["error"]
    # the close came after every figure that does not run the screen, then the screen
    assert ph["closed"] == "before the screen" and "fig_screen" in ph["by"]
    scr = [i for i, r in enumerate(log, 1) if r.get("call") == "screen" and "sha256" in r]
    assert scr and scr[0] > ph["lookups_before"]
    assert log[scr[0] - 1]["label"] == AP.SCREENING_LABEL and log[scr[0] - 1]["after_close"]
    assert all(not r.get("after_close") for r in log[:ph["lookups_before"]])
    rows = {r["id"]: r for r in ph["analyses"]}
    assert rows["P1"]["status"] == "completed" and rows["P1"]["required"] is True
    # P3's required output was made by the framework's own calls before any
    # figure (2026-11-08): the figure with code was refused, and P3 is
    # completed all the same (plotting independence), the failure kept
    assert rows["P3"]["status"] == "completed" and rows["P3"]["produced"] == ["O1"]
    assert any(h["figure"] == "fig_hidden" and not h["ok"] for h in rows["P3"]["attempts"])
    outcomes = {f["id"]: f["ok"] for f in loop["investigation"]["figure_outcomes"]}
    assert outcomes == {AP.REQUIRED_SCRIPT: True, "fig_hidden": False, "fig_gpp": True,
                        "fig_screen": True}


# ── fix 4: the study record knows package F ─────────────────────────────────

def test_the_study_record_names_package_f_and_says_a_withheld_report_is_withheld(study,
                                                                                monkeypatch):
    from agents import study_record as SRD
    from agents.analysis import numbers_check as NC

    def stand_in(study_dir, claims, **k):
        """The numbers check of package F2 has its own tests; stood in for
        here, writing its record as it does."""
        rec = {"pass": True, "n_compared": 0, "mismatches": [], "records": {}}
        (Path(study_dir) / "04_analysis" / "evidence_numbers_check.json").write_text(
            json.dumps(rec))
        return rec
    monkeypatch.setattr(NC, "check_evidence", stand_in)
    _hypothesis_study(study)
    bad = copy.deepcopy(HYPO)
    bad["analyses"][0]["methods"] = ["anomaly", "screen"]          # fails; rewritten once
    am = {"why": "the carbon stock too", "fields": ["TOTECOSYSC"], "methods": ["anomaly"]}
    fig_more = _spec("fig_more", [{"kind": "bars", "outputs": ["X1"]}], [], amendment=am,
                     extra=[{"id": "X1", "kind": "anomaly", "field": "TOTECOSYSC",
                             "period": "2016", "region": REGION}])
    figs = [_spec("fig_gpp", SPEC_GPP, ["P1", "P2"]), fig_more,
            _spec("fig_screen", SPEC_SCREEN, ["screen"])]
    step2 = {"reasoning": "r", "notes": "n", "figures": figs,
             "skipped": [{"analysis": "P3", "why": "not this round"}]}
    replies = Replies({"step2a_plan": [bad], "step3_plan_review": [FOLLOWS],
                       "step2a_plan_rewrite": [HYPO], "step3_plan_review_rewrite": [FOLLOWS],
                       "step2_investigate": [step2], "step3_interpret": [NO_EVIDENCE]})
    loop = _run(study, replies)
    out = study["out"]
    recs = CT.evidence_records(out)
    gpp = [r for r in recs.values() if r["call"] == "anomaly" and r["field"] == "GPP"][0]
    v = gpp["values"]["anomaly"]
    good = {"claim": f"Over the vegetated cells, carbon uptake (GPP) in 2016 was {v:.4g} "
                     f"gC/m^2/yr from its 2015 normal, consistent with the plan's expectation.",
            "finding_id": "fig_gpp", "values": [v], "caveats": [], "evidence": [gpp["id"]],
            "evidence_class": "tested expectation", "expectation": "E1",
            "warnings": sorted({w["code"] for w in gpp["warnings"]}), "headline": True}
    interp = dict(loop["interpretation"],
                  claims=[good] + list(loop["interpretation"]["claims"]))
    rep = S4.build(study["ctx"], {}, loop["investigation"], interp, study["rd"])
    assert rep.get("withheld_because")                 # claim 2 names no record
    S4.write(rep, out, slides=False)
    names = {p.name for p in out.iterdir()}
    # (a figure is a specification since 2026-11-08: no script reads a
    # method's contract, so contract_reads.jsonl is not written here)
    for n in (AP.PLAN_FILE, AP.PLAN_V1_FILE, AP.REVIEW_FILE, AP.REVIEW_V1_FILE,
              AP.AMENDMENTS_FILE, AP.LOOKUPS_FILE, AP.PLANNED_PHASE_FILE, AP.SCREEN_FILE,
              "evidence_numbers_check.json", "fig_gpp.spec.json", "fig_more.spec.json"):
        assert n in names, n
    assert any(n.startswith("screen_L") and n.endswith(".csv") for n in names)
    page = SRD.render(study["rd"])
    for n in names:
        row = [x for x in page.splitlines() if x.startswith(f"| `04_analysis/{n}` |")]
        assert row and "no role on record" not in row[0], n
    # the withheld report and its claims, said as withheld
    sec = page[page.index("## The Analyzer"):page.index("## Every file in this folder")]
    assert "is **withheld**: " in sec and "Its report" not in sec
    assert "kept by the audit" not in sec
    assert "- claims withheld with the report: 2" in sec and sec.count("**(withheld)**") == 2
    assert f"evidence class: tested expectation; records: {gpp['id']}; the evidence trace: " \
           f"passes" in sec
    assert "evidence class: none; records: none; the evidence trace: fails (no record" in sec
    # the plan, both versions' reviews, the lookups, the phase, the screen, the classes
    assert "the first version, `04_analysis/analysis_plan.v1.json`, did not pass its review" in sec
    assert "its review, `04_analysis/analysis_plan_review.json`: **passed**" in sec
    assert "the code check: P1: the screen is declared in the plan's screen entry" in sec
    assert "- the lookups, `04_analysis/lookups.jsonl`: " in sec
    assert "after the close of the planned analyses" in sec
    assert "- the planned analyses, `04_analysis/planned_phase.json` (closed before the screen" \
        in sec and "each judged by its required outputs" in sec
    # step 2b did not draw P3 ("not this round"); the framework made its
    # required output all the same (2026-11-08), so P3 is completed
    assert "| P3 | completed | no | O1 | none | none | 1 required output produced |" in sec
    assert "- the screen, `04_analysis/screen_record.json`" in sec
    assert "- the claims' evidence classes: none 1, tested expectation 1" in sec
    assert "- acceptance: every required analysis completed" in sec
