"""The required outputs of an analysis (plan_contract, the user's decision of
2026-11-08 after G part 2's fixes): positive and negative tests, synthetic,
with no model call and no suite.

    pytest tests/test_plan_contract.py -v

An analysis's scope (its fields, regions, periods, steps and methods) says
what a script may look up; its required outputs say what it must produce.
Completion is proven only by a lookup that matches an output exactly, planned
under its analysis, made before the close, answered, by a figure that ran to
its end: never by a figure that ran, nor by a field read by another call.
Outputs produced, not estimable and unmet are kept apart, and completion is
never support.
"""
import inspect
import re

import pytest

from g2_replay_support import forbid_model_clients

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import plan_contract as PC          # noqa: E402
from agents.analysis import step2_investigate as S2      # noqa: E402

AP._elm()
import domain_methods as M                               # noqa: E402

R = "watershed"
FIELDS = ["GPP", "H2OSNO", "QRUNOFF"]
TIMING = M.TIMING_IDS


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


def _an(aid, outputs, fields=("GPP",), methods=("anomaly",), periods=("2015",),
        regions=(R,), steps=()):
    return {"id": aid, "question": "q", "fields": list(fields), "regions": list(regions),
            "periods": list(periods), "steps": list(steps), "methods": list(methods),
            "settings": {}, "why": "w", "outputs": outputs}


def _plan(*analyses, expectations=()):
    return {"comparisons": [], "baseline": {"normal_years": [2016], "why": "w"},
            "analyses": list(analyses), "expectations": list(expectations), "competing": [],
            "screen": {"when": "last", "region": R, "year": 2015, "why": "w"}}


def _problems(plan):
    return AP.plan_problems(plan, FIELDS, [R], [2016], "hypothesis")


ANOM = {"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2015", "region": R}


# ── the plan check: positive ────────────────────────────────────────────────

def test_a_plan_whose_analyses_name_their_outputs_passes():
    plan = _plan(
        _an("P1", [ANOM, {"id": "O2", "kind": "series", "field": "GPP", "region": R,
                          "step": "day"}], methods=("anomaly",), steps=("day",)),
        _an("P2", [{"id": "O1", "kind": "anomaly", "field": "H2OSNO", "period": "2015-MAM",
                    "region": None},
                   {"id": "O2", "kind": "anomaly", "field": "GPP", "period": "2015-JJA",
                    "region": None},
                   {"id": "O3", "kind": "overlap", "region": R, "inputs": ["O1", "O2"]}],
            fields=("GPP", "H2OSNO"), methods=("anomaly", "overlap"),
            periods=("2015-MAM", "2015-JJA")),
        _an("P3", [{"id": "O1", "kind": "map", "field": "GPP", "period": "2015"},
                   {"id": "O2", "kind": "elevation_split", "region": R, "inputs": ["P2.O2"]}],
            methods=("elevation_split",)),
        _an("P4", [{"id": "O1", "kind": "snow_gone", "region": R, "period": "WY2015"},
                   {"id": "O2", "kind": "growth_peak", "region": R, "period": "2015"}],
            fields=(), methods=("snow_gone", "growth_peak"), periods=("WY2015", "2015")),
        _an("P5", [{"id": "O1", "kind": "difference", "run": "nofire", "field": "GPP",
                    "period": "2015", "region": R}], methods=("difference",)))
    assert _problems(plan) == []


# ── the plan check: negative, one rule each ─────────────────────────────────

BAD = [
    ("no output named", _an("P1", []), "P1: names no required output"),
    ("an id twice", _an("P1", [ANOM, dict(ANOM)]), "output 'O1' is named twice"),
    ("no such kind", _an("P1", [dict(ANOM, kind="trend")]), "no output kind 'trend'"),
    ("the screen", _an("P1", [{"id": "O1", "kind": "screen", "region": R, "period": "2015"}]),
     "the screen is never an output"),
    ("a key its kind names left out", _an("P1", [{"id": "O1", "kind": "anomaly", "field": "GPP",
                                                  "region": R}]), "it lacks period"),
    ("a key no kind knows", _an("P1", [dict(ANOM, colour="red")]), "knows no colour"),
    ("a timing measure with no region",
     _an("P1", [{"id": "O1", "kind": "snow_gone", "region": None, "period": "2015"}],
         fields=(), methods=("snow_gone",)), "a snow_gone output names its region"),
    ("a difference of the control",
     _an("P1", [{"id": "O1", "kind": "difference", "run": "control", "field": "GPP",
                 "period": "2015", "region": R}], methods=("difference",)),
     "a difference names the run it compares with the control"),
    ("a timing period that is no year",
     _an("P1", [{"id": "O1", "kind": "snow_gone", "region": R, "period": "spring"}],
         fields=(), methods=("snow_gone",)), "is not a year"),
    ("a method the analysis does not run", _an("P1", [ANOM], methods=("difference",)),
     "its kind 'anomaly' is not among the analysis's methods"),
    ("a period outside its scope", _an("P1", [dict(ANOM, period="2015-JJA")]),
     "outside the analysis's scope: it declares no period '2015-JJA'"),
    ("a field outside its scope", _an("P1", [dict(ANOM, field="H2OSNO")]),
     "outside the analysis's scope: it declares no field 'H2OSNO'"),
    ("an overlap of one map",
     _an("P1", [dict(ANOM, region=None), {"id": "O2", "kind": "overlap", "region": R,
                                          "inputs": ["O1"]}], methods=("anomaly", "overlap")),
     "a overlap names 2 inputs"),
    ("an input no output of the plan",
     _an("P1", [{"id": "O1", "kind": "elevation_split", "region": R, "inputs": ["P9.O1"]}],
         methods=("elevation_split",)), "its input 'P9.O1' is no output of the plan"),
    ("an input that is no map",
     _an("P1", [ANOM, {"id": "O2", "kind": "elevation_split", "region": R, "inputs": ["O1"]}],
         methods=("anomaly", "elevation_split")), "its input 'O1' is not a map"),
]


@pytest.mark.parametrize("name,analysis,words", BAD, ids=[b[0] for b in BAD])
def test_the_plan_check_refuses_an_output_that_breaks_the_contract(name, analysis, words):
    probs = _problems(_plan(analysis))
    assert any(words in p for p in probs), probs


# ── what proves an output ───────────────────────────────────────────────────

def _line(call, keys, aid="P1", result=None, script="fig", **marks):
    ln = {"call": call, "method": None if call in ("get_map", "get_series") else call,
          "keys": dict(keys), "planned": True, "plan_items": [aid], "script": script,
          "sha256": "s" * 64, "result": result if result is not None else {}}
    ln.update(marks)
    return ln


A_LINE = _line("anomaly", {"run": "control", "field": "GPP", "period": "2015", "region": R},
               result={"anomaly": 12.5})


def _eval(a, lines, figures=(), stated=None):
    return PC.evaluate(a, lines, figures, stated, timing_ids=TIMING,
                       index=PC.plan_index({"analyses": [a]}))


def test_an_exact_match_is_produced_and_completes():
    r = _eval(_an("P1", [ANOM]), [A_LINE], [{"id": "fig", "plan_items": ["P1"], "ok": True,
                                             "round": 1, "lines": [1, 1]}])
    assert (r["status"], r["produced"], r["not_estimable"], r["unmet"]) == \
        (PC.COMPLETED, ["O1"], [], [])
    assert r["outputs"][0]["record"] == "L1" and r["reason"] == "1 required output produced"


def test_a_figure_that_ran_proves_nothing():
    """The figure names the analysis and runs to its end, but its lookup is a
    series of the same field: field access is not the output."""
    ser = _line("get_series", {"run": "control", "field": "GPP", "region": R, "step": "day"})
    r = _eval(_an("P1", [ANOM], steps=("day",)), [ser],
              [{"id": "fig", "plan_items": ["P1"], "ok": True, "round": 1, "lines": [1, 1]}])
    assert r["status"] == PC.FAILED and r["unmet"] == ["O1"] and r["produced"] == []
    assert "no planned lookup matched it (the figure 'fig' naming this analysis ran without " \
        "it)" in r["reason"]


@pytest.mark.parametrize("line,result", [
    (dict(A_LINE, keys=dict(A_LINE["keys"], period="2015-JJA")), None),    # another period
    (dict(A_LINE, keys=dict(A_LINE["keys"], region="elsewhere")), None),   # another region
    (_line("anomaly", {"run": "control", "field": "GPP", "period": "2015", "region": R},
           refused="the gate: the reviewed plan declares no ...", planned=None,
           plan_items=None), "refused by the gate"),
    (dict(A_LINE, planned=False, plan_items=[], amendment="A1"), "not planned"),
    (dict(A_LINE, after_close=True, planned=False, label=AP.EXPLORATORY_LABEL),
     "made after the close"),
    (dict(A_LINE, plan_items=["P2"]), "not under this analysis"),
    (dict(A_LINE, failed="ZeroDivisionError: division by zero"), "failed"),
    (dict({k: v for k, v in A_LINE.items() if k != "sha256"},
          refused="the period is absent from the run"), "refused by the data"),
], ids=["another period", "another region", "refused by the gate", "an amendment's",
        "after the close", "another analysis's", "failed", "refused by the data"])
def test_a_line_that_is_not_the_output_or_not_eligible_leaves_it_unmet(line, result):
    r = _eval(_an("P1", [ANOM]), [line])
    assert r["unmet"] == ["O1"] and r["status"] in (PC.FAILED, PC.SKIPPED)
    att = r["outputs"][0]["attempts"]
    if result is None:
        assert att == []                                   # not a call of the output at all
    else:
        assert [a["result"] for a in att] == [result]
        assert r["status"] == PC.FAILED                    # attempted, not met


def test_an_answer_counts_although_its_figure_then_failed_to_plot():
    """PLOTTING INDEPENDENCE (the user's requirement, confirmed 2026-11-08):
    the logged, answered, planned call is the output; the figure that made it
    and then failed to plot takes nothing away, and stays in the attempts."""
    r = _eval(_an("P1", [ANOM]), [A_LINE],
              [{"id": "fig", "plan_items": ["P1"], "ok": False, "round": 1, "lines": [1, 1],
                "error": "script failed: KeyError"}])
    assert r["status"] == PC.COMPLETED and r["produced"] == ["O1"]
    assert r["outputs"][0]["attempts"] == [{"line": "L1", "result": PC.PRODUCED, "why": None}]
    assert [(h["figure"], h["ok"]) for h in r["attempts"]] == [("fig", False)]
    assert r["failures"] == []                     # a failed plot is no failure of an output


# ── not estimable, apart from produced and unmet ────────────────────────────

def test_a_not_estimable_answer_is_kept_apart_and_completes_the_task():
    ne = dict(A_LINE, result={"anomaly": None, "why": "not estimable: no normal year has "
                                                      "this period"})
    tim = _line("snow_peak", {"run": "control", "year": 2015, "region": R, "against": "normal"},
                result={"variants.default.shift.median": None,
                        "across_settings.why": "no cell is eligible under every setting"})
    gone = _line("snow_gone", {"run": "control", "year": 2015, "region": R,
                               "against": "normal"},
                 result={"variants.default.shift.median": -61.4})
    a = _an("P1", [ANOM, {"id": "O2", "kind": "snow_peak", "region": R, "period": "WY2015"},
                   {"id": "O3", "kind": "snow_gone", "region": R, "period": "WY2015"}],
            methods=("anomaly", "snow_peak", "snow_gone"), periods=("2015", "WY2015"))
    r = _eval(a, [ne, tim, gone])
    assert (r["produced"], r["not_estimable"], r["unmet"]) == (["O3"], ["O1", "O2"], [])
    assert r["status"] == PC.COMPLETED
    assert r["reason"] == "1 required output produced, 2 not estimable"
    o = {x["id"]: x for x in r["outputs"]}
    assert o["O1"]["why"] == "not estimable: no normal year has this period"
    assert o["O2"]["why"] == "no cell is eligible under every setting"
    # acceptance: no blocker, the not-estimable outputs listed apart
    ph = {"question_type": "hypothesis",
          "analyses": [dict(r, required=True, required_by=["E1"])]}
    acc = AP.acceptance(ph)
    assert acc["ok"] is True and acc["blockers"] == []
    assert [(x["output"], x["record"]) for x in acc["not_estimable"]] == [("O1", "L1"),
                                                                         ("O2", "L2")]


def test_a_refusal_in_the_methods_own_words_not_estimable_is_not_estimable():
    ln = dict({k: v for k, v in A_LINE.items() if k != "sha256"},
              refused="not estimable: the period starts before the run's records")
    r = _eval(_an("P1", [ANOM]), [ln])
    assert r["not_estimable"] == ["O1"] and r["status"] == PC.COMPLETED


# ── completed after an earlier failure, every attempt kept ──────────────────

def test_completed_after_an_earlier_failure_keeps_the_attempt_history():
    """The output's own call failed first (a call the data refused), then
    one answered it: completed after an earlier failure, every attempt kept."""
    refused = dict({k: v for k, v in A_LINE.items() if k != "sha256"},
                   refused="the period is absent from the run")
    figs = [{"id": "fig", "plan_items": ["P1"], "ok": False, "round": 1, "lines": [1, 1],
             "error": "script failed: IndexError"},
            {"id": "fig", "plan_items": ["P1"], "ok": True, "round": 2, "lines": [2, 2]}]
    r = _eval(_an("P1", [ANOM]), [refused, dict(A_LINE)], figs)
    assert r["status"] == PC.COMPLETED_AFTER_FAILURE and r["produced"] == ["O1"]
    assert r["outputs"][0]["record"] == "L2"
    assert [a["result"] for a in r["outputs"][0]["attempts"]] == ["refused by the data",
                                                                  PC.PRODUCED]
    assert [(h["round"], h["ok"]) for h in r["attempts"]] == [(1, False), (2, True)]
    assert r["reason"].startswith("1 required output produced, after an earlier failure: "
                                  "L1 (O1) refused by the data: the period is absent")
    ph = {"question_type": "hypothesis", "analyses": [dict(r, required=True)]}
    acc = AP.acceptance(ph)
    assert acc["ok"] is True and [x["analysis"] for x in acc["recovered"]] == ["P1"]


def test_a_failure_after_the_output_was_met_is_kept_but_changes_nothing():
    figs = [{"id": "fig", "plan_items": ["P1"], "ok": True, "round": 1, "lines": [1, 1]},
            {"id": "fig", "plan_items": ["P1"], "ok": False, "round": 2, "lines": [2, 1],
             "error": "script failed: TypeError"}]
    r = _eval(_an("P1", [ANOM]), [A_LINE], figs)
    assert r["status"] == PC.COMPLETED
    assert [h["ok"] for h in r["attempts"]] == [True, False]       # still on record


# ── skipped: never attempted ────────────────────────────────────────────────

def test_skipped_with_step_2bs_reason_or_the_frameworks():
    assert _eval(_an("P1", [ANOM]), [], [], "the plan's map waits")["reason"] == \
        "step 2b's reason: the plan's map waits"
    r = _eval(_an("P1", [ANOM]), [], [])
    assert (r["status"], r["reason"]) == (PC.SKIPPED, PC.NOT_CARRIED)


# ── an overlap's inputs are maps from the outputs it names ──────────────────

def _maps_and_overlap(a_sha, b_sha):
    m1 = _line("anomaly", {"run": "control", "field": "H2OSNO", "period": "2015-MAM",
                           "region": None}, arrays={"anomaly": "h" * 64, "value": "v" * 64})
    m2 = _line("anomaly", {"run": "control", "field": "GPP", "period": "2015-JJA",
                           "region": None}, arrays={"anomaly": "g" * 64})
    ov = _line("overlap", {"region": R, "by_band": False, "a": a_sha, "b": b_sha},
               result={"correlation": 0.24})
    a = _an("P1", [{"id": "O1", "kind": "anomaly", "field": "H2OSNO", "period": "2015-MAM",
                    "region": None},
                   {"id": "O2", "kind": "anomaly", "field": "GPP", "period": "2015-JJA",
                    "region": None},
                   {"id": "O3", "kind": "overlap", "region": R, "inputs": ["O1", "O2"]}],
            fields=("H2OSNO", "GPP"), methods=("anomaly", "overlap"),
            periods=("2015-MAM", "2015-JJA"))
    return a, [m1, m2, ov]


@pytest.mark.parametrize("a_sha,b_sha,met", [
    ("h" * 64, "g" * 64, True),                     # the named maps, in order
    ("g" * 64, "h" * 64, False),                    # swapped: not the output named
    ("v" * 64, "g" * 64, False),                    # the value map, not the anomaly
    ("z" * 64, "g" * 64, False),                    # a map the script made itself
], ids=["in order", "swapped", "not the anomaly map", "the script's own map"])
def test_an_overlap_counts_only_on_the_maps_its_inputs_name(a_sha, b_sha, met):
    a, lines = _maps_and_overlap(a_sha, b_sha)
    r = _eval(a, lines)
    assert ("O3" in r["produced"]) is met and ("O3" in r["unmet"]) is (not met)


def test_an_input_of_another_analysis_by_its_plan_name():
    a, lines = _maps_and_overlap("h" * 64, "g" * 64)
    first = dict(a, id="P1", outputs=a["outputs"][:2])
    second = _an("P2", [dict(a["outputs"][2], inputs=["P1.O1", "P1.O2"])], fields=(),
                 methods=("overlap",), periods=())
    lines[2]["plan_items"] = ["P2"]
    plan = {"analyses": [first, second]}
    r = PC.evaluate(second, lines, timing_ids=TIMING, index=PC.plan_index(plan))
    assert r["produced"] == ["O3"] and r["status"] == PC.COMPLETED
    assert _problems(_plan(first, second)) == []


# ── completion is not support ───────────────────────────────────────────────

def test_completion_says_nothing_of_support():
    """An output whose value goes against the expectation completes its task
    as fully as one that goes with it; acceptance holds no verdict on
    support."""
    up = _eval(_an("P1", [ANOM]), [dict(A_LINE, result={"anomaly": 40.0})])
    down = _eval(_an("P1", [ANOM]), [dict(A_LINE, result={"anomaly": -40.0})])
    assert up["status"] == down["status"] == PC.COMPLETED
    acc = AP.acceptance({"question_type": "hypothesis",
                         "analyses": [dict(down, required=True, required_by=["E1"])]})
    assert acc["ok"] is True and acc["support"] == PC.SUPPORT_WORDS
    assert not any("support" in k and k != "support" for k in acc)


# ── the words the model reads come from the contract ────────────────────────

def test_the_tasks_say_the_contract_in_its_own_words():
    flat = " ".join(AP.plan_task().split())
    for words in (PC.RULE_WORDS, PC.PROOF_WORDS, PC.STATUS_WORDS, PC.REQUIRED_WORDS,
                  PC.SUPPORT_WORDS):
        assert " ".join(words.split()) in flat, words[:60]
    for k in list(PC.KINDS) + list(TIMING):
        assert k in flat, k
    assert " ".join(PC.STEP2_WORDS.split()) in " ".join(S2.SUITE_TASK.split())
    assert '"outputs": [{"id": "O1", "kind": "anomaly"' in flat


def test_every_key_the_contract_reads_of_an_output_is_one_it_names():
    src = inspect.getsource(PC.output_problems) + inspect.getsource(PC._same) + \
        inspect.getsource(PC.matches) + inspect.getsource(PC.as_lookup)
    read = set(re.findall(r'\bo\.get\("(\w+)"\)', src)) | set(re.findall(r'\bo\["(\w+)"\]', src))
    known = {"id", "kind", "settings"} | {n for d in list(PC.KINDS.values()) + [PC.TIMING]
                                          for n in list(d["names"]) + list(d["may"])}
    assert read and read <= known, read - known


# ── the open points, resolved (the user's decision of 2026-11-08) ───────────
# 1. NOT ESTIMABLE, MADE PLAIN: its task is complete and blocks nothing, and
#    the expectation it serves is listed as untestable on it, up front.

def test_a_not_estimable_required_output_is_named_untestable_up_front():
    from agents.analysis import step4_report as S4
    tim = _line("snow_peak", {"run": "control", "year": 2015, "region": R, "against": "normal"},
                result={"variants.default.shift.median": None,
                        "across_settings.why": "no cell is eligible under every setting"})
    a = _an("P2", [{"id": "O1", "kind": "snow_peak", "region": R, "period": "WY2015"}],
            fields=(), methods=("snow_peak",), periods=("WY2015",))
    row = dict(_eval(a, [dict(tim, plan_items=["P2"])]), required=True, required_by=["E2"])
    ph = {"question_type": "hypothesis", "analyses": [row],
          "expectations": [{"id": "E2", "analysis": "P2", "field": "H2OSNO",
                            "sign": "earlier", "text": "an earlier peak"}]}
    acc = AP.acceptance(ph)
    assert acc["ok"] is True and acc["blockers"] == []               # its task is complete
    assert acc["untestable"] == [{"expectation": "E2", "analysis": "P2", "none_estimable": True,
                                  "outputs": [{"output": "O1", "record": "L1",
                                               "why": "no cell is eligible under every "
                                                      "setting"}]}]
    line = S4.untestable_line({"acceptance": acc})
    assert line.startswith("**Part of the hypothesis cannot be tested from this run.** E2 "
                           "(analysis P2): its required output O1 is not estimable (L1: no cell "
                           "is eligible under every setting), so nothing in this run tests it")
    page = S4.with_planned_phase("# Q\n\nbody", {"acceptance": acc}, section=False)
    assert page.split("\n")[2] == line
    # produced outputs only: no such line
    ok = dict(_eval(_an("P1", [ANOM]), [A_LINE]), required=True)
    assert AP.acceptance({"question_type": "hypothesis", "analyses": [ok],
                          "expectations": [{"id": "E1", "analysis": "P1"}]})["untestable"] == []
    assert S4.untestable_line({"acceptance": {"untestable": []}}) is None


# 2. AN OVERLAP'S ORDER STAYS, AND STEP 2b IS TOLD THE EXACT CALLS: each
#    required output written as the call that produces it, from the contract.
#    Run on the synthetic suite, each listed call is accepted by the close.

from test_domain_access import REGION as SREG, built, study   # noqa: E402,F401

RT = {"comparisons": [], "baseline": {"normal_years": [2015], "why": "the Planner's"},
      "analyses": [
          {"id": "P1", "question": "GPP", "fields": ["GPP"], "regions": [SREG],
           "periods": ["2016"], "steps": ["month"], "methods": ["anomaly"], "settings": {},
           "why": "w",
           "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2016",
                        "region": SREG},
                       {"id": "O2", "kind": "anomaly", "field": "GPP", "period": "2016",
                        "region": None},
                       {"id": "O3", "kind": "series", "field": "GPP", "region": SREG,
                        "step": "month"}]},
          {"id": "P2", "question": "the stock's map", "fields": ["TOTECOSYSC"], "regions": [],
           "periods": ["2016"], "steps": [], "methods": [], "settings": {}, "why": "w",
           "outputs": [{"id": "O1", "kind": "map", "field": "TOTECOSYSC", "period": "2016"}]},
          {"id": "P3", "question": "where", "fields": [], "regions": [SREG], "periods": [],
           "steps": [], "methods": ["overlap", "elevation_split"],
           "settings": {"overlap": {"sign_share_limit": 0.7}}, "why": "w",
           "outputs": [{"id": "O1", "kind": "overlap", "region": SREG,
                        "inputs": ["P2.O1", "P1.O2"]},
                       {"id": "O2", "kind": "elevation_split", "region": SREG,
                        "inputs": ["P1.O2"]}]},
          {"id": "P4", "question": "when", "fields": [], "regions": [SREG], "periods": ["2016"],
           "steps": [], "methods": ["growth_peak"], "settings": {}, "why": "w",
           "outputs": [{"id": "O1", "kind": "growth_peak", "region": SREG, "period": "2016"}]}],
      "expectations": [{"id": "E1", "analysis": "P1", "field": "GPP", "sign": "higher",
                        "text": "GPP higher"}],
      "competing": [], "screen": {"when": "last", "region": SREG, "year": 2016, "why": "w"}}


def _rt_study(study):
    from g2_replay_support import Replies
    from agents.analysis import step3_interpret as S3
    study["ctx"].plan["experiment"]["analysis"]["question_type"] = "hypothesis"
    c = Replies({"a": [RT], "b": [{"follows": True, "why": "w", "problems": []}]}).clients()
    plan = AP.write_plan(study["ctx"], study["out"], client=c["a"])
    assert plan["problems"] == [], plan["problems"]
    S3.rule_plan_follows_question(study["ctx"], study["out"], client=c["b"])
    assert AP.plan_passed(study["out"])[0]
    return plan, AP.make_access(AP.access_spec(study["ctx"]), script="fig")


def _calls(plan, swap=False, default_settings=False):
    """The script a figure would write from the brief's list: each input map
    once, then each required output's call as listed."""
    index = PC.plan_index(plan)
    analyses = {a["id"]: a for a in plan["analyses"]}
    code, maps = [], {}
    for a in plan["analyses"]:
        for o in a["outputs"]:
            for x in o.get("inputs") or []:
                owner, oid = PC._owner(x, a["id"])
                label = f"{owner}.{oid}"
                if label not in maps:
                    maps[label] = f"map_{owner}_{oid}"
                    code.append(f"{maps[label]} = " + PC.map_code(index[label], analyses[owner],
                                                                   TIMING, AP._method_defaults))
    for a in plan["analyses"]:
        for o in a["outputs"]:
            if swap and o.get("kind") == "overlap":
                o = dict(o, inputs=list(reversed(o["inputs"])))
            if default_settings and o.get("kind") == "overlap":
                a = dict(a, settings={})
            code.append(PC.call_text(o, a, index, TIMING, AP._method_defaults, maps=maps,
                                     analyses=analyses))
    return code


def test_the_brief_lists_each_output_as_the_call_that_produces_it(study):
    from agents.analysis import step2_investigate as S2b
    plan, acc = _rt_study(study)
    brief = S2b.suite_brief(study["ctx"])
    flat = " ".join(brief.split())
    assert "THE REQUIRED OUTPUTS, EACH WITH THE CALL THAT PRODUCES IT" in flat
    assert f'P3.O1: suite.overlap(<the map of P2.O1: suite.get_map("control", "TOTECOSYSC", ' \
           f'"2016")>, <the map of P1.O2: suite.anomaly("GPP", "2016", region=None)' \
           f'["anomaly"]>, "{SREG}", settings={{\'sign_share_limit\': 0.7}})' in flat
    assert f'P4.O1: suite.growth_peak(2016, "{SREG}")' in flat
    assert " ".join(PC.STEP2_WORDS.split()) in " ".join(S2b.SUITE_TASK.split())


def test_every_listed_call_run_on_the_suite_is_accepted_by_the_close(study):
    plan, acc = _rt_study(study)
    code = _calls(plan)
    exec("\n".join(code), {"suite": acc})
    n = len(AP.lookups(study["out"]))
    figs = [{"id": "fig", "plan_items": ["P1", "P2", "P3", "P4"], "ok": True, "round": 1,
             "lines": [1, n]}]
    ph = AP.close_planned_phase(study["out"], when="before the screen", by="the test",
                                figures=figs)
    got = {f"{r['id']}.{o['id']}": o["outcome"] for r in ph["analyses"] for o in r["outputs"]}
    assert set(got) == {"P1.O1", "P1.O2", "P1.O3", "P2.O1", "P3.O1", "P3.O2", "P4.O1"}
    why = {f"{r['id']}.{o['id']}": (o["why"], o["attempts"]) for r in ph["analyses"]
           for o in r["outputs"] if o["outcome"] == PC.UNMET}
    assert all(v in (PC.PRODUCED, PC.NOT_ESTIMABLE) for v in got.values()), why
    assert all(r["status"] == PC.COMPLETED for r in ph["analyses"]), \
        [(r["id"], r["reason"]) for r in ph["analyses"]]


@pytest.mark.parametrize("swap,default_settings,result", [
    (True, False, None),                                 # the maps swapped: no call of it
    (False, True, "made with other settings"),           # its analysis's setting left out
], ids=["inputs swapped", "settings left at the defaults"])
def test_a_call_other_than_the_listed_one_leaves_the_output_unmet(study, swap,
                                                                 default_settings, result):
    plan, acc = _rt_study(study)
    exec("\n".join(_calls(plan, swap=swap, default_settings=default_settings)), {"suite": acc})
    n = len(AP.lookups(study["out"]))
    ph = AP.close_planned_phase(study["out"], when="before the screen", by="the test",
                                figures=[{"id": "fig", "plan_items": ["P1", "P2", "P3", "P4"],
                                          "ok": True, "round": 1, "lines": [1, n]}])
    p3 = {r["id"]: r for r in ph["analyses"]}["P3"]
    o1 = {o["id"]: o for o in p3["outputs"]}["O1"]
    assert o1["outcome"] == PC.UNMET and p3["status"] == PC.FAILED
    assert [a["result"] for a in o1["attempts"]] == ([] if result is None else [result])
    assert o1["settings"] == {"sign_share_limit": 0.7}


# 3. AN OUTPUT'S SETTINGS: its analysis's for its method, else the defaults,
#    unless it names its own; a map, a series and a timing measure name none.

OV = {"id": "O3", "kind": "overlap", "region": R, "inputs": ["O1", "O2"]}
MAPS2 = [{"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2015", "region": None},
         {"id": "O2", "kind": "anomaly", "field": "H2OSNO", "period": "2015", "region": None}]


def _ov_analysis(o, settings=None):
    return _an("P1", MAPS2 + [o], fields=("GPP", "H2OSNO"), methods=("anomaly", "overlap"),
               periods=("2015",)) | ({"settings": settings} if settings is not None else {})


SETTING_BAD = [
    ("a timing measure's", _an("P1", [{"id": "O1", "kind": "growth_peak", "region": R,
                                       "period": "2015", "settings": {"window": 15}}],
                               fields=(), methods=("growth_peak",)),
     "a growth_peak output names no settings (every stored setting of a timing measure comes "
     "with each of its results)"),
    ("a map's", _an("P1", [{"id": "O1", "kind": "map", "field": "GPP", "period": "2015",
                            "settings": {}}]), "a map output names no settings (it takes none)"),
    ("a setting the method lacks", _ov_analysis(dict(OV, settings={"width": 3})),
     "overlap has no setting 'width'"),
    ("a value its analysis does not allow", _ov_analysis(dict(OV, settings={
        "sign_share_limit": 0.6})), "it declares no setting overlap.sign_share_limit=0.6"),
    ("several values named, none chosen", _ov_analysis(
        OV, settings={"overlap": {"sign_share_limit": [0.6, 0.7]}}),
     "its analysis names several values of overlap's sign_share_limit"),
]


@pytest.mark.parametrize("name,analysis,words", SETTING_BAD, ids=[b[0] for b in SETTING_BAD])
def test_the_plan_check_holds_an_outputs_settings_to_the_contract(name, analysis, words):
    probs = _problems(_plan(analysis))
    assert any(words in p for p in probs), probs


def test_an_outputs_own_settings_allowed_by_its_analysis_pass_the_check():
    a = _ov_analysis(dict(OV, settings={"sign_share_limit": 0.6}),
                     settings={"overlap": {"sign_share_limit": [0.6, 0.7]}})
    assert _problems(_plan(a)) == []
    assert PC.expected_settings(a["outputs"][2], a, AP._method_defaults) == {
        "sign_share_limit": 0.6}
    # none named anywhere: the method's defaults
    b = _ov_analysis(OV)
    assert PC.expected_settings(b["outputs"][2], b, AP._method_defaults) == \
        {"sign_share_limit": 0.8}
