"""The two fixes that prepare the Analyzer phase's close (2026-11-08),
synthetic, with no model call.

    pytest tests/test_analyzer_phase_close.py -v

U   ONE UNIT PER PANEL. Every figure that failed in the accepted fourth batch
    was a bars panel of several units, found only when drawing. The rule
    (figure_spec.UNIT_RULE) is now the check's, the task's words and the unit
    the brief lists beside each output (figure_spec.output_unit): a bars or
    series panel of several units is split into one panel per unit while the
    figure keeps to its four panels, and otherwise refused before anything
    is drawn, the units named. No output is dropped, two units never share
    an axis, no value is rescaled, and every output keeps its record.
C   A HEADLINE KEEPS ITS CLAIM'S CAVEATS. A headline drawn from a verified
    claim carries every warning the claim carries, in the registry's words,
    beside it wherever it is shown: the page, the slide, the plain text and
    the study record. Its claim, records and warnings are named with it. The
    exploratory and inconclusive labels stay.
"""
import json

import numpy as np
import pytest

from g2_replay_support import Replies, forbid_model_clients
from test_analyzer_g2_replay import (FOLLOWS, NO_EVIDENCE, E1, _hypothesis_study, _plan,  # noqa: F401
                                     _run, _spec)
from test_analyzer_g4_specs import PLAN, _go, _outcomes, _finding    # noqa: F401
from test_domain_access import REGION, built, study           # noqa: F401

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import figure_spec as FS            # noqa: E402
from agents.analysis import plan_contract as PC          # noqa: E402
from agents.analysis import step2_investigate as S2      # noqa: E402
from agents.analysis import step4_report as S4           # noqa: E402

AP._elm()
import domain_methods as M                               # noqa: E402


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


def _ref(ref, kind, value, line=1, out=None, made=True, outcome=PC.PRODUCED, **kw):
    return {"ref": ref, "kind": kind, "made": made, "line": line, "value": value,
            "outcome": outcome, "why": None, "out": out or {"kind": kind, "region": REGION},
            **kw}


def _anom(units, x=2.0):
    return {"anomaly": x, "percent": 5.0, "change_units": units}


TIMING = {"variants": {"default": {"shift": {"median": -3.0}}},
          "across_settings": {"minimum": {"median_shift": -4.0},
                              "maximum": {"median_shift": -2.0}}}


# ── U: the unit each output is drawn in ─────────────────────────────────────

def test_the_unit_of_each_kind_of_output():
    assert FS.output_unit(_ref("a", "anomaly", _anom("mm/season"))) == "mm/season"
    assert FS.output_unit(_ref("a", "anomaly", _anom("mm/season")), {"value": "percent"}) == \
        "percent"
    assert FS.output_unit(_ref("t", "growth_peak", TIMING)) == "days"
    assert FS.output_unit(_ref("s", "series", {"units": "gC/m^2/d"})) == "gC/m^2/d"
    split = _ref("e", "elevation_split", {"parts": {}}, out={"kind": "elevation_split",
                                                              "inputs": ["P5.O1"]},
                 input_units={"P5.O1": "gC/m^2/yr"})
    assert FS.output_unit(split) == "gC/m^2/yr"
    # over a map read as it is, whose unit no record holds: no unit is listed
    # (review of 2026-11-09: the brief said "unit: no unit")
    assert FS.output_unit(dict(split, input_units={"P5.O1": None})) is None
    assert FS.output_unit(_ref("a", "anomaly", None, made=False)) is None    # not made
    assert FS.output_unit(_ref("a", "anomaly", {"anomaly": 1.0})) == FS.NO_UNIT


# ── U: split, or refused, before anything is drawn ──────────────────────────

def _refs(*rows):
    return {r["ref"]: r for r in rows}


def test_a_mixed_unit_panel_is_split_into_one_panel_per_unit_in_order():
    refs = _refs(_ref("P1.O1", "anomaly", _anom("mm/season"), line=3),
                 _ref("P1.O2", "anomaly", _anom("mm/yr"), line=4),
                 _ref("P1.O3", "anomaly", _anom("mm/season"), line=5))
    f = _spec("f", [{"kind": "bars", "outputs": ["P1.O1", "P1.O2", "P1.O3"],
                     "value": "change"}], ["P1"])
    got, probs = FS.split_by_unit(f, refs)
    assert probs == []
    assert got["panels"] == [
        {"kind": "bars", "outputs": ["P1.O1", "P1.O3"], "value": "change", "split_from": 1,
         "asked_as": 1},
        {"kind": "bars", "outputs": ["P1.O2"], "value": "change", "split_from": 1,
         "asked_as": 1}]
    # nothing dropped, nothing added
    assert sorted(n for p in got["panels"] for n in p["outputs"]) == ["P1.O1", "P1.O2", "P1.O3"]
    assert f["panels"][0]["outputs"] == ["P1.O1", "P1.O2", "P1.O3"]   # the request untouched


def test_a_panel_of_one_unit_and_a_figure_without_bars_are_left_as_they_are():
    refs = _refs(_ref("P1.O1", "anomaly", _anom("mm")), _ref("P1.O2", "anomaly", _anom("mm")))
    f = _spec("f", [{"kind": "bars", "outputs": ["P1.O1", "P1.O2"]},
                    {"kind": "timing", "output": "P4.O1"}], ["P1"])
    assert FS.split_by_unit(f, refs) == (f, [])


def test_a_split_beyond_four_panels_is_refused_naming_the_units():
    refs = _refs(_ref("P1.O1", "anomaly", _anom("mm/season")),
                 _ref("P1.O2", "anomaly", _anom("mm/yr")),
                 _ref("P2.O1", "anomaly", _anom("K")), _ref("P2.O2", "anomaly", _anom("K")),
                 _ref("P3.O1", "growth_peak", TIMING))
    f = _spec("f", [{"kind": "bars", "outputs": ["P1.O1", "P1.O2"]},
                    {"kind": "bars", "outputs": ["P2.O1"]}, {"kind": "bars", "outputs": ["P2.O2"]},
                    {"kind": "timing", "output": "P3.O1"}], ["P1"])
    got, probs = FS.split_by_unit(f, refs)
    assert got is f and probs == [
        "its outputs are in different units within one panel (panel 1 (bars): mm/season "
        "(P1.O1), mm/yr (P1.O2)), and one panel per unit would give it 5 panels where a "
        "figure holds 4: put each unit's outputs in a panel of their own, in this figure or "
        "another"]


def test_an_output_not_made_stays_in_its_panels_first_part_and_is_still_named():
    refs = _refs(_ref("P1.O1", "anomaly", _anom("mm/season")),
                 _ref("P1.O2", "anomaly", None, made=False, error="refused by the gate"),
                 _ref("P1.O3", "anomaly", _anom("mm/yr")))
    f = _spec("f", [{"kind": "bars", "outputs": ["P1.O1", "P1.O2", "P1.O3"]}], ["P1"])
    got, probs = FS.split_by_unit(f, refs)
    assert [p["outputs"] for p in got["panels"]] == [["P1.O1", "P1.O2"], ["P1.O3"]]
    drawn = FS.render_figure(got, refs, "/tmp/never_written.png")
    assert drawn["ok"] is False and "P1.O2 was not made: refused by the gate" in drawn["error"]


def test_a_series_panel_of_two_units_is_split_and_never_shares_an_axis(tmp_path):
    lab = ["2016-01-01", "2016-01-02"]
    refs = _refs(_ref("P3.O1", "series", {"labels": lab, "values": np.array([1.0, 2.0]),
                                          "units": "gC/m^2/d", "step": "day"}),
                 _ref("P3.O2", "series", {"labels": lab, "values": np.array([3.0, 4.0]),
                                          "units": "mm", "step": "day"}))
    f = _spec("f", [{"kind": "series", "outputs": ["P3.O1", "P3.O2"]}], ["P3"])
    got, probs = FS.split_by_unit(f, refs)
    assert probs == [] and [p["outputs"] for p in got["panels"]] == [["P3.O1"], ["P3.O2"]]
    drawn = FS.render_figure(got, refs, tmp_path / "f.png")
    assert drawn["ok"] and [(p["panel"], p["split_from"], p["records"]) for p in
                            drawn["panels"]] == [(1, 1, ["L1"]), (2, 1, ["L1"])]
    # the renderer's own guard, should a panel of two units ever reach it
    bad = FS.render_figure(f, refs, tmp_path / "g.png")
    assert bad["ok"] is False and "its series are in different units (gC/m^2/d, mm)" in \
        bad["error"]


def test_the_renderer_still_refuses_bars_of_two_units_that_reach_it(tmp_path):
    refs = _refs(_ref("P1.O1", "anomaly", _anom("mm/season")),
                 _ref("P1.O2", "anomaly", _anom("mm/yr")))
    bad = FS.render_figure(_spec("f", [{"kind": "bars", "outputs": ["P1.O1", "P1.O2"]}],
                                 ["P1"]), refs, tmp_path / "f.png")
    assert bad["ok"] is False and "its bars are in different units (mm/season, mm/yr)" in \
        bad["error"]


# ── U: the independent review of 2026-11-09 ─────────────────────────────────

def test_an_output_that_draws_nothing_takes_no_unit_and_forces_no_split(tmp_path):
    # a bar in mm beside a not-estimable output in K and a K/yr output with no
    # value: the drawing shows one bar and two notes on one mm axis, as before
    # the split existed, so nothing is split and a full figure is not refused
    refs = _refs(_ref("P1.O1", "anomaly", _anom("mm")),
                 _ref("P1.O2", "anomaly", _anom("K"), outcome=PC.NOT_ESTIMABLE),
                 _ref("P1.O3", "anomaly", {"anomaly": None, "change_units": "K/yr"}),
                 _ref("P2.O1", "anomaly", _anom("K")), _ref("P3.O1", "anomaly", _anom("K")),
                 _ref("P5.O1", "anomaly", _anom("mm")))
    f = _spec("f", [{"kind": "bars", "outputs": ["P1.O1", "P1.O2", "P1.O3"]},
                    {"kind": "bars", "outputs": ["P2.O1"]}, {"kind": "bars", "outputs": ["P3.O1"]},
                    {"kind": "bars", "outputs": ["P5.O1"]}], ["P1"])
    assert FS.split_by_unit(f, refs) == (f, [])
    drawn = FS.render_figure(f, refs, tmp_path / "f.png")
    assert drawn["ok"], drawn.get("error")
    assert FS.drawn_unit(refs["P1.O2"]) is None and FS.drawn_unit(refs["P1.O3"]) is None
    assert FS.drawn_unit(refs["P1.O1"]) == "mm"


def test_a_failure_after_a_split_names_the_panel_as_it_was_asked(tmp_path):
    # asked: panel 1, bars in mm and days; panel 2, a series with no value.
    # Drawn as three panels, the failure still names panel 2
    lab = ["2016-01-01", "2016-01-02"]
    refs = _refs(_ref("P1.O1", "anomaly", _anom("mm")), _ref("P4.O1", "growth_peak", TIMING),
                 _ref("P3.O1", "series", {"labels": lab, "values": np.array([np.nan, np.nan]),
                                          "units": "mm", "step": "day"}))
    f = _spec("f", [{"kind": "bars", "outputs": ["P1.O1", "P4.O1"]},
                    {"kind": "series", "outputs": ["P3.O1"]}], ["P1"])
    got, probs = FS.split_by_unit(f, refs)
    assert probs == [] and [p["asked_as"] for p in got["panels"]] == [1, 1, 2]
    drawn = FS.render_figure(got, refs, tmp_path / "f.png")
    assert drawn["ok"] is False and drawn["error"].startswith("panel 2 (series): ")
    assert [(p["panel"], p["asked_as"]) for p in drawn["panels"]] == [(1, 1), (2, 1)]


def test_a_bar_axis_with_no_unit_reads_value_as_before():
    plt = FS._plt()
    fig, ax = plt.subplots()
    FS._draw_bars(ax, [_ref("P1.O1", "anomaly", {"anomaly": 1.0})], {}, "panel 1 (bars)")
    assert ax.get_ylabel() == "value"
    plt.close(fig)


def test_series_with_and_without_a_unit_are_split_and_the_guard_names_both(tmp_path):
    lab = ["2016-01-01", "2016-01-02"]
    refs = _refs(_ref("P3.O1", "series", {"labels": lab, "values": np.array([1.0, 2.0]),
                                          "units": "", "step": "day"}),
                 _ref("P3.O2", "series", {"labels": lab, "values": np.array([3.0, 4.0]),
                                          "units": "mm", "step": "day"}))
    f = _spec("f", [{"kind": "series", "outputs": ["P3.O1", "P3.O2"]}], ["P3"])
    got, probs = FS.split_by_unit(f, refs)
    assert probs == [] and [p["outputs"] for p in got["panels"]] == [["P3.O1"], ["P3.O2"]]
    bad = FS.render_figure(f, refs, tmp_path / "g.png")
    assert bad["ok"] is False and "its series are in different units (mm, no unit)" in \
        bad["error"]


def test_a_percent_panel_is_one_unit_and_the_rule_says_so():
    assert '(percent, for every output of a panel drawn with "value": "percent")' in \
        FS.UNIT_RULE
    refs = _refs(_ref("P1.O1", "anomaly", _anom("mm")), _ref("P1.O2", "anomaly", _anom("K")))
    f = _spec("f", [{"kind": "bars", "outputs": ["P1.O1", "P1.O2"], "value": "percent"}], ["P1"])
    assert FS.split_by_unit(f, refs) == (f, [])


# ── U: the task and the brief say the rule the check applies ────────────────

def test_the_task_states_the_unit_rule_from_the_checks_own_words():
    flat = " ".join(S2.SUITE_TASK.split())
    assert "ONE UNIT PER PANEL: " + " ".join(FS.UNIT_RULE.split()) in flat
    assert str(FS.MAX_PANELS) in FS.UNIT_RULE


def test_the_brief_lists_each_outputs_unit_by_the_same_rule_and_no_value(study):
    _hypothesis_study(study)
    r, loop = _go(study, [])
    prompt = r.prompts("step2_investigate")[0]
    units = AP.output_units(study["out"], AP.load_plan(study["out"]))
    vals = AP.produce_required(study["ctx"], study["out"])["values"]
    assert units["P1.O1"] == FS.output_unit({"kind": "anomaly", "value": vals["P1.O1"]})
    assert units["P4.O1"] == "days" and units["P3.O1"] == vals["P3.O1"]["units"]
    assert "P5.O1" not in units and "P2.O1" not in units          # maps: drawn by map panels
    flat = " ".join(prompt.split())
    assert f'P1.O1: suite.anomaly("GPP", "2016", region="{REGION}") (unit: {units["P1.O1"]})' \
        in flat
    v = float(vals["P1.O1"]["anomaly"])
    for shown in (repr(v), f"{v:.6g}", f"{v:.4g}"):
        assert shown not in prompt, shown


# ── U: end to end on the synthetic suite ────────────────────────────────────

def test_a_mixed_unit_figure_is_split_and_drawn_its_records_kept(study):
    _hypothesis_study(study)
    f = _spec("fig_mix", [{"kind": "bars", "outputs": ["P1.O1", "P4.O1"]}], ["P1", "P4"])
    r, loop = _go(study, [f])
    o = _outcomes(loop)["fig_mix"]
    assert o["ok"] and [(p["outputs"], p["split_from"]) for p in o["panels"]] == [
        (["P1.O1"], 1), (["P4.O1"], 1)]
    lines = {c["label"]: c["line"] for c in json.loads(
        (study["out"] / AP.REQUIRED_FILE).read_text())["calls"]}
    assert [x["id"] for x in _finding(loop, "fig_mix")["lookups"]] == [
        f"L{lines['P1.O1']}", f"L{lines['P4.O1']}"]
    spec = json.loads((study["out"] / f"fig_mix{S2.SPEC_SUFFIX}").read_text())
    assert spec["panels"] == f["panels"] and len(spec["panels_drawn"]) == 2


def test_a_mixed_unit_figure_over_the_limit_is_refused_before_drawing(study):
    _hypothesis_study(study)
    f = _spec("fig_over", [{"kind": "bars", "outputs": ["P1.O1", "P4.O1"]},
                           {"kind": "map", "output": "P2.O1"}, {"kind": "map", "output": "P5.O1"},
                           {"kind": "series", "outputs": ["P3.O1"]}], ["P1"])
    r, loop = _go(study, [f])
    o = _outcomes(loop)["fig_over"]
    assert o["ok"] is False and o["error"].startswith("refused before drawing: its outputs are "
                                                      "in different units within one panel")
    assert "would give it 5 panels where a figure holds 4" in o["error"]
    assert not (study["out"] / "fig_over.png").exists()
    cav = [c for c in loop["investigation"]["caveats"] if c["id"] == "figure_failed:fig_over"]
    assert cav and "refused before drawing" in cav[0]["statement"]


# ── C: the headline keeps its claim's caveats ───────────────────────────────

def test_a_headlines_caveats_are_its_claims_warnings_in_the_registrys_words():
    cav = CT.headline_caveats({"warnings": ["restricted_comparability", "flag:partial_record",
                                            "eligibility"]})
    assert cav == [
        {"code": "restricted_comparability", "label": "restricted comparability",
         "means": CT.WARNINGS["restricted_comparability"][1]},
        {"code": "flag:partial_record", "label": "partial record",
         "means": M.FLAGS["partial_record"][1]},
        {"code": "eligibility", "label": "eligible", "means": CT.WARNINGS["eligibility"][1]}]
    assert CT.headline_caveats({"warnings": []}) == []


def test_a_warning_its_claim_carries_only_in_words_is_a_caveat_too_each_once():
    # the trace lets a claim carry a record's warning by its label's words;
    # a label its sentence happens to hold is no caveat a reader sees, so the
    # caveats are the claim's codes, then its records' (the trace's own list)
    trace = {"warnings": [{"code": "provisional_settings", "record": "L7", "reason": "x",
                           "label": "provisional"},
                          {"code": "flag:two_peaks", "record": "L7", "reason": "y",
                           "label": "two peaks"}]}
    cav = CT.headline_caveats({"warnings": ["flag:two_peaks"], "evidence_trace": trace})
    assert [c["code"] for c in cav] == ["flag:two_peaks", "provisional_settings"]


def test_a_malformed_or_unregistered_warning_shows_no_caveat():
    assert CT.headline_caveats({"warnings": "none"}) == []
    assert CT.headline_caveats({"warnings": 3}) == []                # no crash
    assert CT.headline_caveats({"warnings": {"eligibility": True}}) == []
    assert [c["code"] for c in CT.headline_caveats(
        {"warnings": ["partial record", "eligibility", "eligibility"]})] == ["eligibility"]


def test_a_report_written_before_its_caveats_were_recorded_still_shows_them():
    rep = {"headline": "GPP peaked 12.6 days later.", "headline_selection": {"chosen": 1},
           "claims": [{"evidence": ["L3"], "warnings": ["provisional_settings"]}]}
    line = CT.headline_caveat_line(rep)
    assert line == ("Caveats it carries from claim 1 (L3): provisional (the method's settings "
                    "include provisional values, approved for integration only).")
    assert CT.headline_text(rep) == rep["headline"] + " " + line
    # no headline chosen (the inconclusive one): none
    assert CT.headline_caveat_line(dict(rep, headline_selection={"chosen": None})) is None


def _rep(headline, warnings, qt_label=""):
    return {"headline": qt_label + headline,
            "headline_selection": {"chosen": 2, "source": {"claim": 2, "evidence": ["L7"],
                                                           "warnings": warnings}},
            "headline_caveats": CT.headline_caveats({"warnings": warnings})}


def test_the_caveat_line_names_its_claim_and_records_and_the_plain_text_carries_it():
    rep = _rep("GPP peaked 12.6 days later.", ["provisional_settings"])
    line = CT.headline_caveat_line(rep)
    assert line == ("Caveats it carries from claim 2 (L7): provisional (the method's settings "
                    "include provisional values, approved for integration only).")
    assert CT.headline_text(rep) == "GPP peaked 12.6 days later. " + line
    assert CT.headline_caveat_line(_rep("x", [])) is None
    assert CT.headline_text(_rep("x", [])) == "x"


def _built(study, monkeypatch, warnings_of, qt="hypothesis", words=""):
    from agents.analysis import numbers_check as NC
    monkeypatch.setattr(NC, "check_evidence", lambda d, c, **k: {
        "pass": True, "n_compared": 0, "mismatches": [], "records": {}})
    study["ctx"].plan["experiment"]["analysis"]["question_type"] = qt
    plan = _plan("exploratory", PLAN["analyses"]) if qt == "exploratory" else PLAN
    r, loop = _go(study, [_spec("fig_peak", [{"kind": "timing", "output": "P4.O1"}], ["P4"])],
                  plan=plan, qt=qt)
    recs = CT.evidence_records(study["out"])
    peak = [x for x in recs.values() if x["call"] == "growth_peak"][0]
    # the synthetic peak's shift moves across its settings, so the claim gives
    # the default, the smallest and the largest, each with its setting
    a = peak["across_settings"]
    d, lo, hi = a["default"], a["minimum"], a["maximum"]
    claim = {"claim": f"Over the vegetated cells, growth peaked {d['median_shift']:g} days from "
                      f"its normal in 2016 at the default setting, {lo['median_shift']:g} days "
                      f"at {lo['variant']} and {hi['median_shift']:g} days at {hi['variant']}"
                      f"{words}.",
             "finding_id": "fig_peak", "values": [d["median_shift"], lo["median_shift"]],
             "caveats": [], "evidence": [peak["id"]], "evidence_class": "observation",
             "expectation": None, "warnings": warnings_of(peak)}
    interp = dict(loop["interpretation"], claims=[claim], headline_rank=[1],
                  verdict="sufficient")
    rep = S4.build(study["ctx"], {}, loop["investigation"], interp, study["rd"],
                   steps={"investigate": True, "interpret": True})
    return rep, peak, r


def test_the_headline_keeps_its_caveats_on_the_page_the_slide_and_in_plain_text(study,
                                                                               monkeypatch):
    rep, peak, _ = _built(study, monkeypatch,
                          lambda p: sorted({w["code"] for w in p["warnings"]}), qt="exploratory")
    codes = sorted({w["code"] for w in peak["warnings"]})
    assert codes, "the timing record carries warnings"
    assert not rep.get("withheld_because")
    assert rep["headline"].startswith(S4.EXPLORATORY_PREFIX)           # the label kept
    assert rep["headline_selection"]["source"] == {"claim": 1, "evidence": [peak["id"]],
                                                   "warnings": codes}
    line = CT.headline_caveat_line(rep)
    assert line and all(CT.warning_label(c) in line for c in codes)
    assert rep["headline_text"] == rep["headline"] + " " + line
    page = S4.render_markdown(rep)
    para = page[page.index("## The answer"):].split("\n\n")[1]
    assert para.startswith(f"**{rep['headline']}**") and f"*{line}*" in para
    S4.write(rep, study["out"], slides=True)
    from pptx import Presentation
    deck = Presentation(str(next(study["out"].glob("*.pptx"))))
    texts = [sh.text_frame.text for sh in deck.slides[1].shapes if sh.has_text_frame]
    i = texts.index(rep["headline"])
    assert texts[i + 1] == line                                      # right under it
    from agents import study_record as SRD
    sec = SRD.render(study["rd"])
    assert line in sec[sec.index("## The Analyzer"):]


def test_a_claim_carrying_its_warnings_in_words_heads_with_them_as_caveats(study,
                                                                          monkeypatch):
    # the claim lists no code; its sentence holds each label, which the trace
    # accepts, and the headline still shows each warning with its meaning
    rep0, peak, _ = _built(study, monkeypatch, lambda p: [], qt="exploratory")
    labels = sorted({w["label"] for w in peak["warnings"]})
    rep, peak, _ = _built(study, monkeypatch, lambda p: [], qt="exploratory",
                          words=" (" + "; ".join(labels) + ")")
    assert not rep.get("withheld_because")
    codes = list(dict.fromkeys(w["code"] for w in peak["warnings"]))
    assert rep["headline_selection"]["source"]["warnings"] == codes
    assert [c["code"] for c in rep["headline_caveats"]] == codes
    assert all(c["means"] for c in rep["headline_caveats"])


def _analysis(study):
    return json.loads((study["out"] / "analysis.json").read_text())


def test_an_older_record_rendered_again_shows_the_caveats_on_the_page_and_record(study,
                                                                               monkeypatch):
    rep, peak, _ = _built(study, monkeypatch,
                          lambda p: sorted({w["code"] for w in p["warnings"]}), qt="exploratory")
    line = CT.headline_caveat_line(rep)
    S4.write(rep, study["out"], slides=False)
    old = _analysis(study)
    for k in ("headline_caveats", "headline_text"):
        old.pop(k)
    old["headline_selection"].pop("source")
    assert CT.headline_caveat_line(old) == line
    page = S4.render_markdown(old)
    assert f"*{line}*" in page[page.index("## The answer"):]
    (study["out"] / "analysis.json").write_text(json.dumps(old))
    from agents import study_record as SRD
    sec = SRD.render(study["rd"])
    assert line in sec[sec.index("## The Analyzer"):]


def test_long_caveats_stay_whole_in_the_record_and_apart_from_the_text_on_the_slide(
        study, monkeypatch):
    from agents import study_record as SRD
    from agents.analysis.dashes import strip_dashes
    rep, peak, _ = _built(study, monkeypatch,
                          lambda p: sorted({w["code"] for w in p["warnings"]}), qt="exploratory")
    many = ["restricted_comparability", "flag:partial_record", "eligibility",
            "provisional_settings", "flag:two_peaks", "near_constant", "coverage",
            "settings_reverse_direction", "fpg_configuration", "insufficient_support"]
    rep["headline_caveats"] = CT.headline_caveats({"warnings": many})
    line = CT.headline_caveat_line(rep)
    assert len(line) + len(rep["headline"]) > 1200          # past the record's longest line
    S4.write(rep, study["out"], slides=True)
    sec = SRD.render(study["rd"])
    assert strip_dashes(line) in sec[sec.index("## The Analyzer"):]
    # the slide: the caveats under the headline, the answer under both, all on it
    from pptx import Presentation
    deck = Presentation(str(study["out"] / "analysis.pptx"))
    boxes = {sh.text_frame.text: sh for sh in deck.slides[1].shapes if sh.has_text_frame}
    h, c, a = boxes[rep["headline"]], boxes[line], boxes[str(rep["answer"])]
    assert h.top + h.height <= c.top and c.top + c.height <= a.top
    assert a.top + a.height <= deck.slide_height
    assert c.text_frame.paragraphs[0].runs[0].font.size.pt == 11     # a long line set smaller
    # each box at least as tall as its text's fewest possible lines (a
    # character at least half its size wide)
    w_in = (deck.slide_width - 2 * int(0.6 * 914400)) / 914400
    for box, size in ((h, 22), (c, 11)):
        n_min = -(-len(box.text_frame.text) // int(w_in * 72 / (size * 0.5)))
        assert box.height / 914400 >= n_min * size * 1.2 / 72, box.text_frame.text[:40]
    # held back with its report, the record still gives the caveats whole
    withheld = dict(_analysis(study), withheld_because="1 of 1 claims fail the evidence trace")
    (study["out"] / "analysis.json").write_text(json.dumps(withheld))
    sec = SRD.render(study["rd"])
    assert f"The headline held back with it: \"{rep['headline']}\" {strip_dashes(line)}" in sec


def test_a_headline_whose_claim_carries_no_warning_gets_no_caveat_line(study, monkeypatch):
    rep, peak, _ = _built(study, monkeypatch, lambda p: [], qt="exploratory")
    # a claim leaving out its record's warnings fails the trace, so no headline
    assert rep.get("withheld_because") and rep["headline"] is None
    assert CT.headline_caveat_line(rep) is None


def test_the_inconclusive_headline_has_no_claim_and_no_caveat(study, monkeypatch):
    rep, peak, _ = _built(study, monkeypatch, lambda p: sorted({w["code"] for w in
                                                                 p["warnings"]}))
    # a hypothesis question: an observation may not head it, every gate passes
    assert rep["headline"] == S4.INCONCLUSIVE_HEADLINE["hypothesis"]
    assert rep.get("headline_caveats") is None and CT.headline_caveat_line(rep) is None
    assert rep["headline_text"] == rep["headline"]


def test_a_withheld_page_shows_the_held_back_headline_with_its_caveats():
    rep = dict(_rep("GPP peaked 12.6 days later.", ["provisional_settings"]),
               withheld_because="1 of 2 claims fail the evidence trace (claims 2)",
               evidence_trace={"withheld_because": "1 of 2 claims fail", "claims_failed": [2],
                               "problems": {2: ["x"]}, "headline": {"problems": []},
                               "records": {}}, claims=[], question="q")
    page = CT.evidence_withheld_markdown(rep)
    assert "The headline held back: \"GPP peaked 12.6 days later.\" Caveats it carries from " \
           "claim 2 (L7): provisional" in page
