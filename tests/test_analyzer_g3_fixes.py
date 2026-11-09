"""The fixes after G part 2's run again (2026-11-08): positive and negative
tests, synthetic, with no model call.

    pytest tests/test_analyzer_g3_fixes.py -v

A1  step 2b's reply: a cut or unreadable reply is saved before the step says
    what happened, and the required outputs no longer ride on it: the
    framework makes their calls itself (plan_contract.call_spec), once, before
    step 2b's model call, so the figures only draw (`outputs`).
B1  a number is its type: "2000 m" is an elevation, never the year 2000.
B2  a derived record (an overlap, an elevation split) names the records of
    the maps it was made from, their years and seasons and, for a split, its
    field; never a region (region matching stays strict).
B3  a percent may be a record's share times 100, a share's only; such a
    percent is no percent change of a field, and the field's percent rule
    does not reach it, while every other percent still meets it.
C1  step 3 is shown the plan's expectations; C2 the runs the server can make.
D2  every exploratory headline is labelled, by the report's own code.
And the two earlier requirements, confirmed: an output that cannot be
estimated stays "not estimable" with the method's own words and no value,
and completion is independent of plotting.
"""
import json

import numpy as np
import pytest

from g2_replay_support import Replies, forbid_model_clients
from test_analyzer_g2_replay import (FOLLOWS, NO_EVIDENCE, P_GPP, P_MAPS, P_QCH, E1, _figure,  # noqa: F401
                                     _hypothesis_study, _plan, _run, _spec)
from test_domain_access import REGION, built, study           # noqa: F401

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import plan_contract as PC          # noqa: E402
from agents.analysis import step2_investigate as S2      # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402
from agents.analysis import step4_report as S4           # noqa: E402

AP._elm()
import domain_changes as DCH                             # noqa: E402

HYPO = _plan("hypothesis", [P_GPP, P_MAPS, P_QCH], [E1])


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


def _rec(rid="L1", field="GPP", region=REGION, values=None, years=("2016",), **kw):
    """An evidence record as claim_trace.evidence_records makes one."""
    r = {"id": rid, "lookup": int(rid[1:]), "call": "anomaly", "method": "anomaly",
         "measure": None, "keys": {"field": field, "region": region, "period": "2016"},
         "settings": None, "run": "control", "field": field, "period": "2016",
         "region": region, "step": None, "planned": True, "plan_items": ["P1"],
         "amendment": None, "label": None, "after_screen": None, "after_close": False,
         "values": dict(values or {}), "words": {}, "comparison": {"against": "the normal"},
         "support": {}, "across_settings": None, "warnings": [], "rows": {},
         "row_warnings": {}, "names": {"years": list(years), "periods": []},
         "percent_allowed": True, "percent_why_not": None}
    r.update(kw)
    return r


def _claim(text, values, rid="L1", cls="observation"):
    return {"claim": text, "finding_id": "f", "values": list(values), "evidence": [rid],
            "evidence_class": cls, "warnings": []}


def _check(c, recs, regions):
    return CT.check_claim(c, {r["id"]: r for r in recs}, {"question_type": "hypothesis"}, {},
                          list(regions))["problems"]


# ── B1: a number is its type ────────────────────────────────────────────────

@pytest.mark.parametrize("text,years", [
    ("the band at 2000 m and above", set()),
    ("from 1500 to 2000 m", set()),
    ("the 2000-2500 m band", set()),
    ("between 2000 and 2500 m", set()),
    ("in 2015 and 2016", {"2015", "2016"}),
    ("over 2015-2023, against 2016-2020", {"2015", "2023", "2016", "2020"}),
    ("2016 was 12.5 mm lower", {"2016"}),
], ids=["a band", "a range", "a hyphenated band", "between", "two years", "year ranges",
        "a year then a value"])
def test_years_are_typed_a_length_is_never_a_year(text, years):
    assert CT.years_named(text) == years


def test_a_band_above_2000_m_is_no_year_but_a_year_its_records_lack_still_fails():
    band = f"{REGION}, 2000 m and above"
    r = _rec(region=band, values={"anomaly": -12.5})
    ok = _check(_claim("In 2016 the band at 2000 m and above held 12.5 mm less.", [-12.5]),
                [r], [REGION, band])
    assert not any("names 2000" in p for p in ok), ok
    bad = _check(_claim("In 2017 the band at 2000 m and above held 12.5 mm less.", [-12.5]),
                 [r], [REGION, band])
    assert any("it names 2017, which none of its records names" in p for p in bad), bad


# ── B2 and B3 on the synthetic suite: a derived record's provenance, a share's percent ──

DER = _plan("hypothesis", [
    {"id": "P1", "question": "GPP's map", "fields": ["GPP"], "regions": [], "periods": ["2016"],
     "steps": [], "methods": ["anomaly"], "settings": {}, "why": "w",
     "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2016",
                  "region": None}]},
    {"id": "P2", "question": "the stock's map", "fields": ["TOTECOSYSC"], "regions": [],
     "periods": ["2016"], "steps": [], "methods": [], "settings": {}, "why": "w",
     "outputs": [{"id": "O1", "kind": "map", "field": "TOTECOSYSC", "period": "2016"}]},
    {"id": "P3", "question": "where", "fields": [], "regions": [REGION], "periods": [],
     "steps": [], "methods": ["overlap", "elevation_split"], "settings": {}, "why": "w",
     "outputs": [{"id": "O1", "kind": "overlap", "region": REGION, "inputs": ["P2.O1", "P1.O1"]},
                 {"id": "O2", "kind": "elevation_split", "region": REGION,
                  "inputs": ["P1.O1"]}]}],
    [dict(E1, analysis="P1")])


def _derived(study, arithmetic=False):
    _hypothesis_study(study)
    c = Replies({"a": [DER], "b": [FOLLOWS]}).clients()
    assert AP.write_plan(study["ctx"], study["out"], client=c["a"])["problems"] == []
    S3.rule_plan_follows_question(study["ctx"], study["out"], client=c["b"])
    acc = AP.make_access(AP.access_spec(study["ctx"]), script="fig")
    stock = acc.get_map("control", "TOTECOSYSC", "2016")                     # L1
    gpp = acc.anomaly("GPP", "2016", region=None)["anomaly"]                 # L2
    acc.overlap(stock * 2.0 if arithmetic else stock, gpp, REGION)          # L3
    acc.elevation_split(gpp, REGION)                                         # L4
    return CT.evidence_records(study["out"])


def test_a_derived_record_names_its_inputs_their_periods_and_a_splits_field(study):
    recs = _derived(study)
    ov, es = recs["L3"], recs["L4"]
    assert [(x["role"], x["record"], x["field"], x["period"], x["path"])
            for x in ov["inputs"]] == [("a", "L1", "TOTECOSYSC", "2016", None),
                                       ("b", "L2", "GPP", "2016", "anomaly")]
    assert ov["field"] is None and ov["fields"] == ["TOTECOSYSC", "GPP"]   # a correlation
    assert "2016" in ov["names"]["years"] and "2016" in es["names"]["years"]
    assert es["field"] == "GPP" and es["percent_allowed"] is True          # typed by its input
    assert "made from L2 (anomaly GPP 2016, its anomaly map)" in CT.describe_record(es)
    # a claim naming the maps' year now traces; a region its records lack does not
    r = ov["values"]["correlation"]
    warn = sorted({w["code"] for w in ov["warnings"]})          # every warning it carries
    ok = CT.check_claim(dict(_claim(f"In 2016 the stock and carbon uptake (GPP) changes went "
                                    f"together weakly, correlation {r:.3g}.", [r], "L3",
                                    "association"), warnings=warn), recs,
                        {"question_type": "hypothesis"}, {}, [REGION, "box, vegetated"])
    assert ok["problems"] == [], ok["problems"]
    bad = CT.check_claim(_claim(f"Over the box, vegetated region, the correlation was "
                                f"{r:.3g}.", [r], "L3", "association"), recs,
                         {"question_type": "hypothesis"}, {}, [REGION, "box, vegetated"])
    assert any("names the region 'box'" in p for p in bad["problems"]), bad["problems"]


def test_a_map_the_script_made_itself_is_named_so_in_the_provenance(study):
    recs = _derived(study, arithmetic=True)
    a = recs["L3"]["inputs"][0]
    assert a["record"] is None and "its own arithmetic" in a["why"]
    assert "2016" in recs["L3"]["names"]["years"]                # its other input's year


def test_a_percent_is_a_share_times_100_only_for_a_share(study):
    recs = _derived(study)
    es = recs["L4"]
    share = next((p, v) for p, v in es["values"].items() if p.endswith("abs_change_share")
                 and 0.05 < abs(v) < 0.95)
    band = share[0].split(".")[1]
    pct = f"{share[1] * 100:.3g}"
    ok = CT.check_claim(_claim(f"In 2016 the {band} band carried {pct} percent of the change "
                               f"in carbon uptake (GPP).", [share[1]], "L4"), recs,
                        {"question_type": "hypothesis"}, {}, [REGION])
    assert not any("matches no value" in p for p in ok["problems"]), ok["problems"]
    # a value that is no share does not match times 100
    mean = next(v for p, v in es["values"].items() if p.endswith(f"{band}.mean"))
    bad = CT.check_claim(_claim(f"In 2016 the {band} band's change was {mean * 100:.3g} "
                                f"percent.", [], "L4"), recs, {"question_type": "hypothesis"},
                         {}, [REGION])
    assert any("matches no value of its records" in p for p in bad["problems"]), \
        bad["problems"]


def test_a_shares_percent_is_outside_a_fields_percent_rule_and_others_are_not():
    r = _rec(field="BTRAN", percent_allowed=False,
             percent_why_not="a bounded index: its change is given within its bounds",
             values={"parts.1000-1500 m.abs_change_share": 0.5557,
                     "parts.1000-1500 m.mean": -0.052})
    assert CT.percent_problems("The 1000-1500 m band carried 55.6 percent of the change.",
                               [r]) == []
    assert CT.percent_problems("Soil water stress (BTRAN) fell 5.2 percent.", [r]) == [
        "a percent of BTRAN, whose notes give none: a bounded index: its change is given "
        "within its bounds"]


# ── C1 and C2: step 3 is shown what the trace requires ──────────────────────

def test_step_3_is_shown_the_plans_expectations_with_their_records():
    recs = {"L1": _rec(), "L2": _rec("L2", plan_items=["P1"], after_close=True),
            "L3": _rec("L3", plan_items=["P2"]), "L4": _rec("L4", planned=False)}
    lines = CT.expectation_lines({"expectations": [E1]}, recs)
    flat = " ".join(" ".join(lines).split())
    assert "THE PLAN'S EXPECTATIONS" in flat
    assert f"E1 (analysis P1, field GPP, expected higher): {E1['text']}" in flat
    assert "records carrying out its analysis: L1" in flat and "L2" not in flat.split(
        "records carrying out its analysis:")[1]
    assert CT.expectation_lines({"expectations": []}, recs) == []


def test_the_brief_of_a_hypothesis_study_holds_its_expectations(study):
    _hypothesis_study(study)
    fig = _figure("fig_gpp", "plt.plot([1, 2]); plt.savefig(out_path)\nresult = {'n': 3}",
                  ["P1"])
    r = Replies({"step2a_plan": [HYPO], "step3_plan_review": [FOLLOWS],
                 "step2_investigate": [{"reasoning": "r", "notes": "", "figures": [fig]}],
                 "step3_interpret": [NO_EVIDENCE]})
    _run(study, r)
    brief = " ".join(r.prompts("step3_interpret")[0].split())
    assert "THE PLAN'S EXPECTATIONS" in brief and "E1 (analysis P1, field GPP, expected " \
        "higher)" in brief


def test_step_3_is_shown_every_run_the_server_can_make():
    flat = " ".join(" ".join(CT.run_lines()).split())
    for c in DCH.declare()["changes"]:
        assert f"{c['id']}" in flat, c["id"]
    assert "THE RUNS THE SERVER CAN MAKE" in " ".join(S3.suite_task().split())
    ok, probs = CT.check_proposals([{"cause": "c", "proposed_run": {"changes": [
        {"id": "fire_off", "params": {}}]}}])
    assert probs == []
    _, probs = CT.check_proposals([{"cause": "c", "proposed_run": {"changes": [
        {"id": "snow_water_initial_increase", "params": {}}]}}])
    assert probs and "is not a run the server can make" in probs[0]


# ── D2: every exploratory headline is labelled, by the report ───────────────

def test_the_report_labels_every_exploratory_headline():
    assert S4.exploratory_headline("GPP fell 23.9 percent.") == \
        "Exploratory: GPP fell 23.9 percent."
    assert S4.exploratory_headline("Exploratory: GPP fell.") == "Exploratory: GPP fell."
    assert "labelled exploratory, every one" in CT.HEADLINE_RULE["exploratory"]


def test_an_exploratory_headline_without_the_label_fails_the_trace(study):
    study["ctx"].plan["experiment"]["analysis"]["question_type"] = "exploratory"
    plan = _plan("exploratory", [dict(P_GPP, id="P1")])
    c = Replies({"a": [plan], "b": [FOLLOWS]}).clients()
    AP.write_plan(study["ctx"], study["out"], client=c["a"])
    S3.rule_plan_follows_question(study["ctx"], study["out"], client=c["b"])
    acc = AP.make_access(AP.access_spec(study["ctx"]), script="fig")
    acc.screen(2016, REGION)
    acc.anomaly("GPP", "2016", region=REGION)
    v = CT.evidence_records(study["out"])["L2"]["values"]["anomaly"]
    cl = [dict(_claim(f"In 2016 carbon uptake (GPP) was {v:.3g} from its normal.", [v], "L2",
                      "exploratory"), warnings=["exploratory"], headline=True)]
    bare = CT.trace_evidence(cl, f"Carbon uptake (GPP) was {v:.3g} from its normal.",
                             study["out"], regions=[REGION])
    assert any("is labelled exploratory, and this one is not" in p
               for p in bare["headline"]["problems"]), bare["headline"]
    put = CT.trace_evidence(cl, S4.exploratory_headline(
        f"Carbon uptake (GPP) was {v:.3g} from its normal."), study["out"], regions=[REGION])
    assert put["headline"]["problems"] == [], put["headline"]


# ── A1: the framework makes the required outputs; a cut reply is saved ─────

def test_the_framework_makes_every_required_output_once_as_the_brief_writes_it(study):
    _hypothesis_study(study)
    c = Replies({"a": [HYPO], "b": [FOLLOWS]}).clients()
    AP.write_plan(study["ctx"], study["out"], client=c["a"])
    S3.rule_plan_follows_question(study["ctx"], study["out"], client=c["b"])
    got = AP.produce_required(study["ctx"], study["out"])
    assert set(got["values"]) == {"P1.O1", "P2.O1", "P2.O2", "P3.O1"}
    log = AP.lookups(study["out"])
    assert all(x["script"] == AP.REQUIRED_SCRIPT and x["planned"] is True for x in log)
    calls = {r["label"]: r["call"] for r in got["record"]["calls"]}
    assert calls["P1.O1"] == f'suite.anomaly("GPP", "2016", region="{REGION}")'
    assert got["record"]["lines"] == [1, len(log)]
    again = AP.produce_required(study["ctx"], study["out"])          # once: read back
    assert len(AP.lookups(study["out"])) == len(log) and set(again["values"]) == \
        set(got["values"])
    assert np.isfinite(np.asarray(again["values"]["P2.O1"], dtype=float)).any()


def test_an_exploratory_studys_screen_comes_first_then_its_outputs(study):
    study["ctx"].plan["experiment"]["analysis"]["question_type"] = "exploratory"
    plan = _plan("exploratory", [dict(P_GPP, id="P1")])
    c = Replies({"a": [plan], "b": [FOLLOWS]}).clients()
    AP.write_plan(study["ctx"], study["out"], client=c["a"])
    S3.rule_plan_follows_question(study["ctx"], study["out"], client=c["b"])
    got = AP.produce_required(study["ctx"], study["out"])
    log = AP.lookups(study["out"])
    assert [x["call"] for x in log] == ["screen", "anomaly"]
    assert log[0]["label"] == AP.EXPLORATORY_LABEL and (study["out"] / AP.SCREEN_FILE).is_file()
    assert set(got["values"]) == {"screen", "P1.O1"}


CUT = '{"reasoning": "r", "notes": "", "figures": [{"id": "fig_a", "code": "plt.plot('


class Cut:
    """A client whose reply was cut off at its output limit."""
    label = "step2_investigate"
    max_tokens = 32000

    def __init__(self):
        self.last_finish_reason, self.prompts = None, []

    def ask(self, messages, **kw):
        self.prompts.append(messages[0]["content"])
        self.last_finish_reason = "length"
        return CUT


def test_a_cut_reply_is_saved_and_the_required_outputs_stand(study):
    _hypothesis_study(study)
    c = Replies({"a": [HYPO], "b": [FOLLOWS]}).clients()
    AP.write_plan(study["ctx"], study["out"], client=c["a"])
    S3.rule_plan_follows_question(study["ctx"], study["out"], client=c["b"])
    inv = S2.investigate(study["ctx"], study["out"], client=Cut(), round_no=1)
    saved = (study["out"] / "step2_round1_reply.txt").read_text()
    assert saved == CUT                                           # the cut reply, verbatim
    cav = [x for x in inv["caveats"] if x["id"] == "step2b_reply_unreadable"]
    assert cav and "cut off at its output limit of 32000 tokens" in cav[0]["statement"] \
        and "step2_round1_reply.txt" in cav[0]["statement"]
    assert inv["reply_failed"].startswith("step 2b's reply was cut off")
    ids = {f["id"] for f in inv["findings"]}
    assert ids == {"required_p1", "required_p2", "required_p3"}
    f = [x for x in inv["findings"] if x["id"] == "required_p1"][0]
    assert f["lookups"] and set(f["result"]) == {"P1.O1"}
    assert f["made_by"].startswith("the framework")


def test_step_2b_names_its_own_output_ceiling():
    import inspect
    assert S2.STEP2_MAX_TOKENS == 32000
    assert "client.max_tokens = STEP2_MAX_TOKENS" in inspect.getsource(S2.propose)


def test_a_figure_reads_the_outputs_and_a_failed_plot_takes_nothing_away(study, monkeypatch):
    """Plotting independence: the outputs are the framework's calls; a figure
    draws them, and one that then fails to plot (its map's drawing raises)
    leaves its analysis completed, its failure kept. Since 2026-11-08 a
    figure is a specification the framework draws."""
    from agents.analysis import figure_spec as FS
    _hypothesis_study(study)

    def broke(plt, ax, r, opts, ll, where):
        raise FS.FigureFailed(f"{where}: the plot broke")
    monkeypatch.setattr(FS, "_draw_map", broke)
    r = Replies({"step2a_plan": [HYPO], "step3_plan_review": [FOLLOWS],
                 "step2_investigate": [{"reasoning": "r", "notes": "", "figures": [
                     _spec("fig_reads", [{"kind": "bars", "outputs": ["P1.O1"]}], ["P1"]),
                     _spec("fig_breaks", [{"kind": "map", "output": "P2.O1"}], ["P2"])]}],
                 "step3_interpret": [NO_EVIDENCE]})
    loop = _run(study, r)
    inv = loop["investigation"]
    fr = [f for f in inv["findings"] if f["id"] == "fig_reads"][0]
    assert fr["result"]["P1.O1"]["anomaly"] == pytest.approx(
        float(AP.produce_required(study["ctx"], study["out"])["values"]["P1.O1"]["anomaly"]))
    assert "panel 1 (map): the plot broke" in [o for o in inv["figure_outcomes"]
                                               if o["id"] == "fig_breaks"][0]["error"]
    ph = AP.load_planned_phase(study["out"])
    rows = {x["id"]: x for x in ph["analyses"]}
    assert rows["P2"]["status"] == PC.COMPLETED and rows["P2"]["produced"] == ["O1", "O2"]
    assert any(h["figure"] == "fig_breaks" and not h["ok"] for h in rows["P2"]["attempts"])
    # the figures make no required call: every required line is the framework's
    req = json.loads((study["out"] / AP.REQUIRED_FILE).read_text())
    lo, hi = req["lines"]
    assert all(x["script"] == AP.REQUIRED_SCRIPT for x in AP.lookups(study["out"])[lo - 1:hi])


# ── the earlier non-estimability requirement, confirmed ─────────────────────

def test_a_not_estimable_output_holds_no_value_and_no_claim_can_give_it_one():
    """Plan section 3.4 rule 2: "not reliably estimable, because ..." is a
    proper outcome with its reason; a method never fills in a value. The
    output is kept apart as not estimable, with the method's own words, and
    a claim stating a number for it matches no value of its record."""
    ne = {"call": "anomaly", "method": "anomaly", "planned": True, "plan_items": ["P1"],
          "keys": {"run": "control", "field": "GPP", "period": "2016", "region": REGION},
          "sha256": "s" * 64,
          "result": {"anomaly": None, "why": "not estimable: no normal year has this period"}}
    a = {"id": "P1", "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP",
                                  "period": "2016", "region": REGION}]}
    row = PC.evaluate(a, [ne], timing_ids=AP._elm() or __import__("domain_methods").TIMING_IDS)
    o = row["outputs"][0]
    assert (o["outcome"], o["record"], o["why"]) == \
        (PC.NOT_ESTIMABLE, "L1", "not estimable: no normal year has this period")
    assert row["not_estimable"] == ["O1"] and row["produced"] == [] and row["unmet"] == []
    rec = _rec(values={})                                         # no number in the record
    probs = _check(_claim("In 2016 carbon uptake (GPP) was 12.5 gC/m2 below normal.", [12.5]),
                   [rec], [REGION])
    assert any("matches no value of its records" in p for p in probs), probs
