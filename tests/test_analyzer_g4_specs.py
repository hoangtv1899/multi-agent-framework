"""The second set of decisions after the third acceptance batch
(2026-11-08), positive and negative tests, synthetic, with no model call.

    pytest tests/test_analyzer_g4_specs.py -v

S   PLOTTING SPECIFICATIONS, with no fallback to a script in this phase. A
    figure of a study with a suite names outputs by label in panels of six
    kinds (figure_spec.PANEL_KINDS); the framework checks it against the
    plan before anything is called or drawn, makes the calls it adds (an
    extra output under its amendment, after the close in a hypothesis study;
    the plan's screen), and draws it. A figure carrying code is refused and
    no script runs. A panel whose output the method found not estimable
    shows the method's own words; a panel with nothing to draw, or whose
    output was not made, fails its figure and names the panel.
H   THE HEADLINE IS A VERIFIED CLAIM'S OWN TEXT: step 3 ranks its claims,
    and step 4's code takes the first that traced and that the question type
    allows. With none, and every other gate passed, the report is published
    with a headline saying plainly that it is inconclusive, under its own
    status, and acceptance is not met; with another gate failed, the report
    is withheld as before.
"""
import json

import numpy as np
import pytest

from g2_replay_support import Replies, forbid_model_clients
from test_analyzer_g2_replay import (FOLLOWS, NO_EVIDENCE, E1, _figure, _hypothesis_study,  # noqa: F401
                                     _plan, _run, _spec)
from test_domain_access import REGION, built, study           # noqa: F401

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import figure_spec as FS            # noqa: E402
from agents.analysis import plan_contract as PC          # noqa: E402
from agents.analysis import step2_investigate as S2      # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402
from agents.analysis import step4_report as S4           # noqa: E402

AP._elm()
import domain_methods as M                               # noqa: E402


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


# a plan with an output of every kind the synthetic suite holds
PLAN = _plan("hypothesis", [
    {"id": "P1", "question": "GPP in 2016", "fields": ["GPP"], "regions": [REGION],
     "periods": ["2016"], "steps": [], "methods": ["anomaly"], "settings": {}, "why": "w",
     "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2016",
                  "region": REGION}]},
    {"id": "P2", "question": "the maps", "fields": ["GPP", "TOTECOSYSC"], "regions": [],
     "periods": ["2016"], "steps": [], "methods": [], "settings": {}, "why": "w",
     "outputs": [{"id": "O1", "kind": "map", "field": "GPP", "period": "2016"},
                 {"id": "O2", "kind": "map", "field": "TOTECOSYSC", "period": "2016"}]},
    {"id": "P3", "question": "the daily course", "fields": ["GPP"], "regions": [REGION],
     "periods": [], "steps": ["day"], "methods": [], "settings": {}, "why": "w",
     "outputs": [{"id": "O1", "kind": "series", "field": "GPP", "region": REGION,
                  "step": "day"}]},
    {"id": "P4", "question": "the peak's date", "fields": [], "regions": [REGION],
     "periods": ["2016"], "steps": [], "methods": ["growth_peak"], "settings": {}, "why": "w",
     "outputs": [{"id": "O1", "kind": "growth_peak", "region": REGION, "period": "2016"}]},
    {"id": "P5", "question": "where", "fields": ["GPP"], "regions": [REGION],
     "periods": ["2016"], "steps": [], "methods": ["anomaly", "overlap", "elevation_split"],
     "settings": {}, "why": "w",
     "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2016",
                  "region": None},
                 {"id": "O2", "kind": "overlap", "region": REGION, "inputs": ["P2.O2", "P5.O1"]},
                 {"id": "O3", "kind": "elevation_split", "region": REGION,
                  "inputs": ["P5.O1"]}]}],
    [E1])
FIG_ALL = _spec("fig_all", [{"kind": "bars", "outputs": ["P1.O1"]},
                            {"kind": "map", "output": "P5.O1"},
                            {"kind": "series", "outputs": ["P3.O1"], "years": [2016],
                             "normal": True},
                            {"kind": "timing", "output": "P4.O1"}], ["P1", "P3", "P4", "P5"])
FIG_WHERE = _spec("fig_where", [{"kind": "overlap", "output": "P5.O2"},
                                {"kind": "bars", "outputs": ["P5.O3"]},
                                {"kind": "map", "output": "P2.O1"}], ["P2", "P5"])


def _go(study, figures, plan=PLAN, step3=NO_EVIDENCE, qt="hypothesis"):
    study["ctx"].plan["experiment"]["analysis"]["question_type"] = qt
    r = Replies({"step2a_plan": [plan], "step3_plan_review": [FOLLOWS],
                 "step2_investigate": [{"reasoning": "r", "notes": "", "figures": figures}],
                 "step3_interpret": [step3]})
    return r, _run(study, r)


def _outcomes(loop):
    return {o["id"]: o for o in loop["investigation"]["figure_outcomes"]}


def _finding(loop, fid):
    return [f for f in loop["investigation"]["findings"] if f["id"] == fid][0]


# ── S: the specification is checked before anything is called or drawn ──────

@pytest.mark.parametrize("panels,why", [
    ([{"kind": "pie", "outputs": ["P1.O1"]}], "no panel kind 'pie'"),
    ([{"kind": "bars", "outputs": ["P1.O1"], "colour": "red"}], "bars has no option 'colour'"),
    ([{"kind": "bars", "outputs": ["P9.O1"]}], "no output 'P9.O1'"),
    ([{"kind": "map", "output": "P1.O1"}], "P1.O1 is a region's anomaly, one number"),
    ([{"kind": "bars", "outputs": ["P5.O1"]}], "P5.O1 is a per-cell anomaly, a map"),
    ([{"kind": "timing", "output": "P1.O1"}], "timing does not draw a anomaly"),
    ([{"kind": "series", "outputs": ["P3.O1"], "years": ["2016"]}], "years=['2016'] is not a "
                                                                    "list of years"),
    ([{"kind": "bars", "outputs": ["P1.O1"], "value": "ratio"}], "value='ratio' is not"),
    ([{"kind": "bars", "outputs": ["P5.O3", "P1.O1"]}], "an elevation split is drawn alone"),
    ([{"kind": "map", "outputs": ["P2.O1"]}], "map names one \"output\""),
    ([{"kind": "screen"}, {"kind": "bars", "outputs": ["P1.O1"]}],
     "the screen panel is its figure's only panel"),
    ([{"kind": "bars", "outputs": ["P1.O1"]}] * 5, "5 panels; at most 4"),
    ([], "no panels"),
], ids=["a kind", "an option", "an output", "a regional anomaly as a map",
        "a per-cell anomaly as bars", "a kind its panel does not draw", "years as words",
        "a value", "a split beside others", "outputs for one", "the screen beside others",
        "five panels", "none"])
def test_a_specification_the_plan_does_not_allow_is_refused(panels, why):
    probs = FS.check_figure(_spec("fig_x", panels, ["P1"]), PLAN, M.TIMING_IDS)
    assert any(why in p for p in probs), probs


def test_an_extra_output_is_checked_for_its_shape():
    bad = _spec("fig_x", [{"kind": "bars", "outputs": ["X1"]}], [], extra=[
        {"id": "Y1", "kind": "anomaly"}, {"id": "X1", "kind": "nothing"},
        {"id": "X2", "kind": "anomaly", "field": "GPP"},
        {"id": "X3", "kind": "overlap", "region": REGION, "inputs": ["P5.O9", "P5.O1"]}])
    probs = " | ".join(FS.check_figure(bad, PLAN, M.TIMING_IDS))
    for why in ("an extra output's id is X1, X2, ..., not 'Y1'", "extra X1: no kind 'nothing'",
                "extra X2: names no period, region", "extra X3: no input 'P5.O9'",
                "panel 1: no output 'X1'"):
        assert why in probs, why


def test_a_figure_with_code_is_refused_and_no_script_runs(study, monkeypatch):
    from agents.analysis import script_runner as SR
    monkeypatch.setattr(SR, "run", lambda *a, **k: pytest.fail("a script ran"))
    _hypothesis_study(study)
    fig = dict(_figure("fig_code", "result = {'n': 3}", ["P1"]),
               panels=[{"kind": "bars", "outputs": ["P1.O1"]}])
    r, loop = _go(study, [fig])
    out = _outcomes(loop)["fig_code"]
    assert out["ok"] is False and "a script is not run" in out["error"]
    assert not list(study["out"].glob("*.py"))
    assert not [x for x in AP.lookups(study["out"]) if x.get("script") == "fig_code"]
    cav = [c for c in loop["investigation"]["caveats"] if c["id"] == "figure_failed:fig_code"]
    assert cav and "its specification was refused" in cav[0]["statement"]


# ── S: the framework draws a checked figure from the calls' own results ──────

def test_a_figure_of_every_panel_kind_is_drawn_from_the_framework_outputs(study):
    _hypothesis_study(study)
    r, loop = _go(study, [FIG_ALL, FIG_WHERE])
    got = _outcomes(loop)
    assert got["fig_all"]["ok"] and got["fig_where"]["ok"], got
    out = study["out"]
    lines = {c["label"]: c["line"] for c in json.loads(
        (out / AP.REQUIRED_FILE).read_text())["calls"]}
    for fid, labels in (("fig_all", ["P1.O1", "P5.O1", "P3.O1", "P4.O1"]),
                        ("fig_where", ["P5.O2", "P5.O3", "P2.O1"])):
        f = _finding(loop, fid)
        assert (out / f"{fid}.png").is_file() and (out / f"{fid}{S2.SPEC_SUFFIX}").is_file()
        assert [x["id"] for x in f["lookups"]] == [f"L{lines[x]}" for x in labels]
        assert set(f["result"]) == set(labels)
        assert f["n"] > 0 and [p["status"] for p in f["panels"]] == ["drawn"] * len(labels)
        assert f["script"].endswith(S2.SPEC_SUFFIX) and f["amendment"] is None
    # each result holds its record's citable values, the labels left out
    a = _finding(loop, "fig_all")["result"]["P1.O1"]
    rec = CT.evidence_records(out)[f"L{lines['P1.O1']}"]
    assert a == {p: v for p, v in rec["values"].items()
                 if p != "version" and not p.startswith("settings.")}
    assert "normal_years" not in json.dumps(a)
    # the figures drew; they made no call: every line is the framework's
    assert all(x["script"] == AP.REQUIRED_SCRIPT for x in AP.lookups(out)
               if x.get("call") not in ("lat_lon", "screen"))


def test_the_framework_writes_every_title_from_the_outputs_keys(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    seen = []
    orig = FS._title

    def keep(ax, refs, outs):
        orig(ax, refs, outs)
        seen.append(ax.get_title(loc="left").replace("\n", " "))
    FS._title = keep
    try:
        refs = {"P1.O1": {"ref": "P1.O1", "kind": "anomaly", "made": True,
                          "out": PLAN["analyses"][0]["outputs"][0], "line": 3,
                          "value": {"anomaly": 2.5, "change_units": "gC/m^2/yr"},
                          "outcome": PC.PRODUCED, "why": None}}
        got = FS.render_figure(_spec("f", [{"kind": "bars", "outputs": ["P1.O1"]}], ["P1"],
                                     question="GPP COLLAPSED", why="it fell"), refs,
                               tmp_path / "f.png")
    finally:
        FS._title = orig
    assert got["ok"] and seen == ["P1.O1: GPP anomaly 2016 vegetated cells"]


def _ref(ref, kind, value, outcome=PC.PRODUCED, why=None, out=None):
    return {"ref": ref, "kind": kind, "made": True, "line": 1, "value": value,
            "outcome": outcome, "why": why, "out": out or {"kind": kind}}


def test_a_not_estimable_output_is_drawn_as_the_methods_own_words(tmp_path):
    words = "not estimable: no normal year has this period"
    refs = {"P1.O1": _ref("P1.O1", "anomaly", {"anomaly": None, "why": words},
                          PC.NOT_ESTIMABLE, words)}
    got = FS.render_figure(_spec("f", [{"kind": "bars", "outputs": ["P1.O1"]}], ["P1"]), refs,
                           tmp_path / "f.png")
    assert got["ok"] and got["panels"][0]["status"] == "not estimable"
    assert got["panels"][0]["why"] == {"P1.O1": words}


@pytest.mark.parametrize("panel,refs,why", [
    ({"kind": "map", "output": "P2.O1"},
     {"P2.O1": _ref("P2.O1", "map", np.full((3, 3), np.nan))}, "P2.O1 holds no cell with a value"),
    ({"kind": "bars", "outputs": ["P1.O1", "P4.O1"]},
     {"P1.O1": _ref("P1.O1", "anomaly", {"anomaly": 2.0, "change_units": "mm"}),
      "P4.O1": _ref("P4.O1", "growth_peak", {"variants": {"default": {"shift": {"median": 3.0}}}})},
     "its bars are in different units (days, mm)"),
    ({"kind": "series", "outputs": ["P3.O1"], "years": [2017]},
     {"P3.O1": _ref("P3.O1", "series", {"labels": ["2016-01-01", "2016-01-02"],
                                        "values": np.array([1.0, 2.0]), "step": "day",
                                        "units": "gC"})}, "P3.O1 holds no value in 2017"),
    ({"kind": "bars", "outputs": ["P1.O1"]},
     {"P1.O1": dict(_ref("P1.O1", "anomaly", None), made=False,
                    error="refused: the plan declares no period '2017'")},
     "P1.O1 was not made: refused: the plan declares no period '2017'"),
], ids=["an empty map", "bars of two units", "a year the series lacks", "an output not made"])
def test_a_panel_with_nothing_to_draw_fails_its_figure_and_names_the_panel(tmp_path, panel,
                                                                         refs, why):
    ll = {"lat": np.arange(3.0), "lon": np.arange(3.0)}
    got = FS.render_figure(_spec("f", [panel], ["P1"]), refs, tmp_path / "f.png", lat_lon=ll)
    assert got["ok"] is False and got["error"] == f"panel 1 ({panel['kind']}): {why}"
    assert not (tmp_path / "f.png").exists()


def test_a_blank_panel_is_a_caveat_step_3_reads(study, monkeypatch):
    """A figure whose map holds no value: the caveat names the panel, and
    the analysis stays completed (plotting independence)."""
    _hypothesis_study(study)
    real = FS._draw_map

    def empty(plt, ax, r, opts, ll, where):
        return real(plt, ax, dict(r, value=np.full_like(np.asarray(r["value"], float), np.nan)),
                    opts, ll, where)
    monkeypatch.setattr(FS, "_draw_map", empty)
    r, loop = _go(study, [_spec("fig_map", [{"kind": "map", "output": "P2.O1"}], ["P2"])])
    cav = [c for c in loop["investigation"]["caveats"] if c["id"] == "figure_failed:fig_map"]
    assert cav and "panel 1 (map): P2.O1 holds no cell with a value" in cav[0]["statement"]
    rows = {x["id"]: x for x in AP.load_planned_phase(study["out"])["analyses"]}
    assert rows["P2"]["status"] == PC.COMPLETED
    brief = r.prompts("step3_interpret")[0]
    assert "P2.O1 holds no cell with a value" in brief


# ── S: the calls a figure adds ──────────────────────────────────────────────

def test_an_extra_output_is_made_after_the_close_under_its_amendment(study):
    _hypothesis_study(study)
    am = {"why": "the carbon stock too", "fields": ["TOTECOSYSC"], "methods": ["anomaly"]}
    more = _spec("fig_more", [{"kind": "bars", "outputs": ["X1"]}], [], amendment=am,
                 extra=[{"id": "X1", "kind": "anomaly", "field": "TOTECOSYSC",
                         "period": "2016", "region": REGION}])
    r, loop = _go(study, [more, _spec("fig_gpp", [{"kind": "bars", "outputs": ["P1.O1"]}],
                                      ["P1"])])
    out = study["out"]
    ph = AP.load_planned_phase(out)
    assert ph["closed"] == "before the screen" and "fig_more" in ph["by"]
    log = AP.lookups(out)
    x = [(i, ln) for i, ln in enumerate(log, 1) if ln.get("script") == "fig_more"
         and ln.get("call") == "anomaly"]
    assert len(x) == 1 and x[0][0] > ph["lookups_before"]
    ln = x[0][1]
    assert ln["amendment"] == AP.load_amendments(out)[0]["id"] and ln["planned"] is not True
    assert ln["after_close"] is True and _outcomes(loop)["fig_more"]["ok"]
    assert [y["id"] for y in _finding(loop, "fig_more")["lookups"]] == [f"L{x[0][0]}"]


def test_an_extra_output_outside_the_plans_and_its_amendments_scope_is_refused(study):
    _hypothesis_study(study)
    more = _spec("fig_more", [{"kind": "bars", "outputs": ["X1"]}], [],
                 extra=[{"id": "X1", "kind": "anomaly", "field": "TOTECOSYSC",
                         "period": "2016", "region": REGION}])
    r, loop = _go(study, [more])
    o = _outcomes(loop)["fig_more"]
    assert o["ok"] is False and o["error"].startswith("panel 1 (bars): X1 was not made: ")
    ref = [ln for ln in AP.lookups(study["out"]) if ln.get("script") == "fig_more"]
    assert ref and ref[0].get("refused")


def test_the_screen_is_run_after_the_close_for_its_figure_in_a_hypothesis_study(study):
    _hypothesis_study(study)
    r, loop = _go(study, [_spec("fig_screen", [{"kind": "screen"}], ["screen"]), FIG_ALL])
    log = AP.lookups(study["out"])
    ph = AP.load_planned_phase(study["out"])
    scr = [i for i, ln in enumerate(log, 1) if ln.get("call") == "screen" and "sha256" in ln]
    assert scr and scr[0] > ph["lookups_before"] and log[scr[0] - 1]["script"] == "fig_screen"
    # the synthetic suite's one normal year scores no row: the panel says so
    f = _finding(loop, "fig_screen")
    assert f["panels"][0]["status"] == "not estimable"
    assert f["panels"][0]["why"]["screen"].startswith("the screen scored no row:")


def test_step_2bs_task_states_every_panel_and_asks_for_no_code():
    task = S2.SUITE_TASK
    flat = " ".join(task.split())
    for k in FS.PANEL_KINDS.values():
        assert f'"{k.id}": ' in flat, k.id
        for o in k.options:
            assert f'"{o.name}": ' in flat, (k.id, o.name)
    assert '"code"' in task and "a \"code\" field refuses it, and no script is run" in flat
    assert not any(p.name in ("code", "variables") for p in S2.SUITE_FIGURE_PARTS)
    assert " ".join(PC.STEP2_WORDS.split()) in flat
    assert "each names the outputs it draws by label" in PC.STEP2_WORDS
    assert S2.task_for.__doc__ and S2.SUITE_TASK == S2.suite_task()


# ── H: the headline is a verified claim's own text ──────────────────────────

def _claims():
    return [{"claim": "Over the vegetated cells, GPP rose in 2016.", "evidence": ["L1"],
             "evidence_class": "observation"},
            {"claim": "Over the vegetated cells, GPP in 2016 was 12.5 above its normal.",
             "evidence": ["L1"], "evidence_class": "tested expectation", "expectation": "E1"},
            {"claim": "A third claim.", "evidence": ["L2"],
             "evidence_class": "tested expectation", "expectation": "E1"}]


def _trace(qt="hypothesis", failed=(), records=None):
    return {"question_type": qt, "claims_failed": list(failed),
            "records": records or {"L1": {"id": "L1"}, "L2": {"id": "L2"}}}


def test_the_headline_is_the_first_ranked_claim_that_traced_and_the_rule_allows():
    sel = CT.select_headline(_claims(), _trace(), [1, 3, 2])
    assert sel["chosen"] == 3 and sel["text"] == "A third claim."
    assert sel["passed_over"] == [{"claim": 1, "why": "claim 1 is observation, not tested: a "
                                                       "hypothesis question's headline comes "
                                                       "from tested claims only"}]
    assert CT.select_headline(_claims(), _trace(failed=[3]), [3, 2])["chosen"] == 2
    gone = _trace(records={"L1": {"id": "L1", "after_close": True}, "L2": {"id": "L2"}})
    sel = CT.select_headline(_claims(), gone, [2, 3])
    assert sel["chosen"] == 3 and "made after the planned analyses were closed" in \
        sel["passed_over"][0]["why"]


def test_an_exploratory_headline_may_be_any_traced_claim():
    assert CT.select_headline(_claims(), _trace("exploratory"), [1])["chosen"] == 1
    assert CT.select_headline(_claims(), _trace("exploratory", failed=[1]), [1])["chosen"] == 2


def test_the_order_is_the_rank_then_every_other_claim_and_a_recorded_reply_its_marks():
    cs = _claims()
    assert CT.headline_order(cs, [3, 3, 9, "1", True]) == [3, 1, 2]
    legacy = [dict(c, headline=(i == 2)) for i, c in enumerate(cs)]
    assert CT.headline_order(legacy, None) == [3, 1, 2]
    assert CT.headline_order(cs, None) == [1, 2, 3]


def test_no_eligible_claim_leaves_the_choice_to_the_report():
    sel = CT.select_headline(_claims()[:1], _trace(), [1])
    assert sel["chosen"] is None and sel["text"] is None and len(sel["passed_over"]) == 1


def test_step_3s_ranking_is_carried_onto_the_claims_the_audit_kept(study, monkeypatch):
    _hypothesis_study(study)
    # the ranking is what is tested here: its claims name no record, which the
    # audit's trace rule (2026-11-10) would strike, so that rule is set aside
    monkeypatch.setattr(S3, "_tracer", lambda ctx, records: None)
    claims = [dict(NO_EVIDENCE["claims"][0]),
              dict(NO_EVIDENCE["claims"][0], claim="Over the vegetated cells, carbon uptake "
                   "(GPP) rose in 2016.", finding_id="no_such_finding"),
              dict(NO_EVIDENCE["claims"][0], claim="Carbon uptake (GPP) over the vegetated "
                   "cells rose in 2016.")]
    reply = {"claims": claims, "headline_rank": [2, 3, 1], "answer": "Carbon uptake (GPP) rose.",
             "verdict": "sufficient", "feedback": ""}
    r, loop = _go(study, [_spec("fig_gpp", [{"kind": "bars", "outputs": ["P1.O1"]}], ["P1"])],
                  step3=reply)
    it = loop["interpretation"]
    assert it["headline"] is None and it["headline_rank_as_given"] == [2, 3, 1]
    assert len(it["claims"]) == 2 and it["headline_rank"] == [2, 1]   # claim 2 was struck
    assert not any(p.startswith("headline") for p in it["readability"]["problems_before"])
    task = S3.suite_task()
    assert '"headline_rank": ' in task and "THE HEADLINE IS NOT YOURS TO WRITE" in task


def _built(study, monkeypatch, claims, rank):
    from agents.analysis import numbers_check as NC
    monkeypatch.setattr(NC, "check_evidence", lambda d, c, **k: {
        "pass": True, "n_compared": 0, "mismatches": [], "records": {}})
    _hypothesis_study(study)
    r, loop = _go(study, [_spec("fig_gpp", [{"kind": "bars", "outputs": ["P1.O1"]}], ["P1"])])
    recs = CT.evidence_records(study["out"])
    gpp = [x for x in recs.values() if x["call"] == "anomaly" and x["field"] == "GPP"][0]
    v = gpp["values"]["anomaly"]
    made = []
    for cls in claims:
        made.append({"claim": f"Over the vegetated cells, carbon uptake (GPP) in 2016 was "
                              f"{v:.4g} gC/m^2/yr from its 2015 normal.",
                     "finding_id": "fig_gpp", "values": [v], "caveats": [],
                     "evidence": [gpp["id"]] if cls != "none" else [],
                     "evidence_class": "observation" if cls == "none" else cls,
                     "expectation": "E1" if cls == "tested expectation" else None,
                     "warnings": sorted({w["code"] for w in gpp["warnings"]})})
    interp = dict(loop["interpretation"], claims=made, headline_rank=rank, verdict="sufficient")
    return S4.build(study["ctx"], {}, loop["investigation"], interp, study["rd"],
                    steps={"investigate": True, "interpret": True})


def test_the_report_headline_is_the_chosen_claims_text(study, monkeypatch):
    rep = _built(study, monkeypatch, ["observation", "tested expectation"], [1, 2])
    assert not rep.get("withheld_because")
    assert rep["headline"] == rep["claims"][1]["claim"]
    assert rep["headline_selection"]["chosen"] == 2 and rep["status"] == S4.SUPPORTED


def test_no_eligible_claim_with_every_gate_passed_is_published_inconclusive(study, monkeypatch):
    rep = _built(study, monkeypatch, ["observation"], [1])
    assert not rep.get("withheld_because")
    assert rep["headline"] == S4.INCONCLUSIVE_HEADLINE["hypothesis"]
    assert rep["status"] == S4.INCONCLUSIVE and rep["headline_selection"]["inconclusive"]
    acc = rep["acceptance"]
    assert acc["ok"] is False and acc["inconclusive"] == S4.INCONCLUSIVE_WHY["hypothesis"]
    page = S4.render_markdown(rep)
    assert f"**{S4.INCONCLUSIVE_HEADLINE['hypothesis']}**" in page
    assert f"Acceptance: not met: {S4.INCONCLUSIVE_WHY['hypothesis']}" in \
        "\n".join(S4.planned_phase_lines(rep))


def test_no_eligible_claim_with_another_gate_failed_is_withheld(study, monkeypatch):
    rep = _built(study, monkeypatch, ["observation", "none"], [1, 2])
    assert rep.get("withheld_because") and rep["headline"] is None
    assert not rep["headline_selection"].get("inconclusive")
    assert rep["status"] != S4.INCONCLUSIVE
