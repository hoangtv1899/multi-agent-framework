"""The recipe "seasonal evolution and spatial context" (the user's decision of
2026-11-09), synthetic, with no model call.

    pytest tests/test_analysis_recipes.py -v

A   WHICH PARTS A QUESTION CALLS FOR is read from its own words, by code: the
    same question gets the same guidance whatever model plans it.
G   THE GUIDANCE names only output kinds the plan contract has, no place, no
    value and no result; it stands before the plan's task, apart from the
    methods view, and the plan check does not enforce it.
M   A MONTH IS A REGION'S VALUE ONLY: the periods rule now says so, a region's
    month anomaly is planned, produced, drawn and traced, and a per-cell
    output of a month is refused at the plan check.
C   COVERAGE AND THE REPORT'S LINES follow what the plan holds.
S   THE ANALYZER'S MODEL is one setting: the config's, else the environment's,
    else the default.
"""
import pytest

from g2_replay_support import forbid_model_clients
from test_analyzer_g2_replay import E1, _hypothesis_study, _plan, _spec   # noqa: F401
from test_analyzer_g4_specs import _go, _outcomes                          # noqa: F401
from test_domain_access import REGION, built, study                        # noqa: F401

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import analysis_recipes as AR       # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import figure_spec as FS            # noqa: E402
from agents.analysis import plan_contract as PC          # noqa: E402
from agents.analysis import step2_investigate as S2      # noqa: E402

Q_BOTH = "Did an early leaf loss in 2016 accompany the summer GPP decline?"
Q_NONE = "What changed in 2016 compared with the normal years?"
Q_PLACE = "Which elevation bands lost the most carbon in 2016?"


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


# ── A: which parts a question calls for ──────────────────────────────────────

def test_the_parts_a_question_calls_for_are_read_from_its_words():
    assert AR.applies(Q_BOTH) == {AR.SEASONAL: ["early", "summer"], AR.SPATIAL: ["accompany"]}
    assert AR.applies(Q_NONE) == {} and AR.plan_lines(Q_NONE) == []
    assert AR.applies(Q_PLACE) == {AR.SPATIAL: ["elevation", "bands"]}
    assert AR.applies("How did GPP evolve from March to June?") == {
        AR.SEASONAL: ["march", "june"]}
    # a word that only dates a period is no timing word (the timing rule's own reading)
    assert AR.applies("Was GPP lower in late 2016?") == {}
    assert AR.applies(None) == {}


# ── G: the guidance ──────────────────────────────────────────────────────────

def test_the_guidance_names_existing_output_kinds_and_no_place_value_or_result():
    text = " ".join(" ".join(AR.plan_lines(Q_BOTH)).split())     # its lines, as one
    assert f"THE ANALYSIS RECIPE FOR THIS QUESTION: {AR.NAME}." in text
    assert 'SEASONAL EVOLUTION (the question says "early", "summer")' in text
    assert 'SPATIAL CONTEXT (the question says "accompany")' in text
    assert '"2016-01" to "2016-12"' in text                # the question's own year
    for kind in ("anomaly", "difference", "elevation_split", "overlap"):
        assert kind in text and kind in PC.KINDS
    assert "series" in PC.KINDS and 'step "day"' in text
    assert "the plan check does not enforce it" in text and "expects no result" in text
    # no place, no value, no direction: the only digits are the question's year's
    low = text.lower()
    for word in ("naches", "watershed", "snow", "gpp", "earlier than", "lower than", "higher"):
        assert word not in low, word
    import re
    assert set(re.findall(r"\d+", text)) <= {"2016", "01", "12"}
    only = "\n".join(AR.plan_lines(Q_PLACE))
    assert "SPATIAL CONTEXT" in only and "SEASONAL EVOLUTION" not in only
    assert '"YYYY-01"' in "\n".join(AR.plan_lines("Did spring GPP fall?"))   # no year named


def test_the_plan_prompt_holds_the_recipe_before_the_task_only_when_one_applies(study):
    study["ctx"].plan["question"] = Q_BOTH
    inputs = AP.plan_inputs(study["ctx"])
    prompt = AP.plan_prompt(inputs)
    recipe = "\n".join(AR.plan_lines(Q_BOTH))
    assert inputs["recipe"] == recipe
    assert prompt.index("THE METHODS:") < prompt.index(recipe) < prompt.index(AP.PLAN_LEAD)
    assert recipe not in inputs["methods"]                 # apart from the methods view
    assert prompt.endswith(AP.plan_task())                 # the task itself is unchanged
    study["ctx"].plan["question"] = Q_NONE
    inputs = AP.plan_inputs(study["ctx"])
    assert inputs["recipe"] == "" and "ANALYSIS RECIPE" not in AP.plan_prompt(inputs)


# ── M: a month is a region's value only ──────────────────────────────────────

MONTHS = [f"2016-{m:02d}" for m in range(1, 13)]


def FS_ref(ref, out, value=1.0):
    """A made output as the figure code takes it: an anomaly of `value`."""
    return {"ref": ref, "kind": "anomaly", "made": True, "line": 1, "outcome": PC.PRODUCED,
            "why": None, "out": out, "value": {"anomaly": value, "change_units": "gC/m^2/month"}}


def _months_plan(extra=(), periods=()):
    return _plan("hypothesis", [
        {"id": "P1", "question": "GPP in 2016", "fields": ["GPP"], "regions": [REGION],
         "periods": ["2016"], "steps": [], "methods": ["anomaly"], "settings": {}, "why": "w",
         "outputs": [{"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2016",
                      "region": REGION}]},
        {"id": "P2", "question": "GPP month by month", "fields": ["GPP"], "regions": [REGION],
         "periods": MONTHS + list(periods), "steps": [], "methods": ["anomaly"],
         "settings": {}, "why": "w",
         "outputs": [{"id": f"O{i}", "kind": "anomaly", "field": "GPP", "period": m,
                      "region": REGION} for i, m in enumerate(MONTHS, 1)] + list(extra)}],
        [E1])


def test_the_periods_rule_says_a_month_is_a_regions_value_only():
    words = " ".join(next(r for r in AP.PLAN_RULES if r.id == "periods").text().split())
    assert "a month (2015-07) only for a region's value" in words
    assert "month maps are not in the suite" in words and "no months" not in words


def test_a_per_cell_output_of_a_month_is_refused_at_the_plan_check():
    ok = AP.plan_problems(_months_plan(), ["GPP"], [REGION], [2015], "hypothesis")
    assert ok == []
    bad = AP.plan_problems(_months_plan([
        {"id": "X1", "kind": "anomaly", "field": "GPP", "period": "2016-07", "region": None},
        {"id": "X2", "kind": "map", "field": "GPP", "period": "2016-07"}]),
        ["GPP"], [REGION], [2015], "hypothesis")
    assert bad == [
        "P2: output 'X1': month maps are not in the suite: a month (2016-07) is a region's "
        "value only, in an anomaly or a difference that names a region",
        "P2: output 'X2': month maps are not in the suite: a month (2016-07) is a region's "
        "value only, in an anomaly or a difference that names a region"]


def test_a_regions_months_are_planned_produced_drawn_as_months_and_traced(study):
    _hypothesis_study(study)
    fig = _spec("fig_months", [{"kind": "bars", "outputs": [f"P2.O{i}" for i in range(1, 13)]}],
                ["P2"])
    r, loop = _go(study, [fig], plan=_months_plan())
    assert AP.load_plan(study["out"])["problems"] == []
    o = _outcomes(loop)["fig_months"]
    assert o["ok"] and o["panels"][0]["n"] == 12 and len(o["panels"][0]["records"]) == 12
    recs = CT.evidence_records(study["out"])
    july = next(x for x in recs.values() if x.get("period") == "2016-07")
    assert july["names"]["periods"] == [("month", "07")]
    a = july["values"]["anomaly"]
    claim = {"claim": f"Over the vegetated cells, carbon uptake (GPP) in July 2016 was "
                      f"{abs(a):.3g} gC/m2 from its normal July.",
             "finding_id": "fig_months", "values": [a], "caveats": [],
             "evidence": [july["id"]], "evidence_class": "observation", "expectation": None,
             "warnings": sorted({w["code"] for w in july["warnings"]})}
    t = CT.check_claim(claim, recs, AP.load_plan(study["out"]), CT.study_runs(study["rd"]),
                       [REGION])
    assert t["problems"] == []
    wrong = dict(claim, claim=claim["claim"].replace("July", "August"))
    assert any("August" in p for p in CT.check_claim(
        wrong, recs, AP.load_plan(study["out"]), CT.study_runs(study["rd"]), [REGION])["problems"])


# ── T: a run of months is not a season's value (the trace, 2026-11-09) ──────

def test_a_range_of_months_that_spells_a_season_is_found_with_its_clause():
    runs = CT.month_runs("GPP ran below normal from June to August, largest in July.")
    assert runs == {"JJA": {"words": "June to August", "months": ["06", "07", "08"],
                            "aggregate": False}}
    assert CT.month_runs("The June to August total was low.")["JJA"]["aggregate"] is True
    assert CT.month_runs("Low from Dec to Feb.")["DJF"]["months"] == ["12", "01", "02"]
    assert CT.month_runs("Above normal from January to May.") == {}      # no season's span
    assert CT.seasons_named_outright("Below normal from June to August.") == set()
    assert CT.seasons_named_outright("In summer, from June to August (JJA).") == {"JJA"}


def test_a_run_of_months_stands_on_its_monthly_records_and_a_seasons_value_does_not(study):
    """Q1's fifth run (2026-11-09): "below normal from June to August, largest
    in July", on a record for every month, failed for naming JJA. It now
    traces; the season's own record is still needed whenever the claim names
    the season outright or gives the range as one quantity."""
    _hypothesis_study(study)
    season = {"id": "O13", "kind": "anomaly", "field": "GPP", "period": "2016-JJA",
              "region": REGION}
    fig = _spec("fig_months", [{"kind": "bars", "outputs": [f"P2.O{i}" for i in range(1, 13)]}],
                ["P2"])
    r, loop = _go(study, [fig], plan=_months_plan([season], ["2016-JJA"]))
    plan, runs = AP.load_plan(study["out"]), CT.study_runs(study["rd"])
    recs = CT.evidence_records(study["out"])
    by = {x["period"]: x for x in recs.values() if x.get("call") == "anomaly"}
    months = [by[m]["id"] for m in MONTHS]
    jja = by["2016-JJA"]

    def problems(text, evidence, values=()):
        c = {"claim": text, "finding_id": "fig_months", "values": list(values), "caveats": [],
             "evidence": evidence, "evidence_class": "observation", "expectation": None,
             "warnings": sorted({w["code"] for i in evidence for w in recs[i]["warnings"]})}
        return CT.check_claim(c, recs, plan, runs, [REGION])["problems"]

    run = "Month by month, GPP in 2016 ran one way from June to August, most in July."
    assert problems(run, months) == []                          # every month has its record
    assert problems(run, [by[m]["id"] for m in ("2016-06", "2016-07", "2016-08")]) == []
    # a month of the run without a record: named
    assert problems(run, [by["2016-06"]["id"], by["2016-07"]["id"]]) == [
        "it names June to August: none of its records names the season JJA, and as a run of "
        "months none names August"]
    # the season named outright still needs the season's record
    assert problems("In summer 2016 GPP ran one way, from June to August.", months) == [
        "it names JJA, which none of its records names"]
    assert problems("Over JJA 2016 GPP ran one way.", months) == [
        "it names JJA, which none of its records names"]
    # the range given as one quantity still needs the season's record
    assert problems("The June to August total of GPP in 2016 ran one way.", months) == [
        "it names JJA, which none of its records names"]
    # a season's value is no value of monthly records: the numbers rule holds
    a = jja["values"]["anomaly"]
    text = f"From June to August 2016, GPP was {abs(a):.3g} gC/m2 from its normal."
    assert any("matches no value of its records" in p for p in problems(text, months, [a]))
    assert problems(text, [jja["id"]], [a]) == []               # on the season's record


def test_a_run_of_one_fields_months_is_titled_once_and_its_ticks_are_the_months():
    outs = [{"kind": "anomaly", "field": "GPP", "period": m, "region": REGION} for m in MONTHS]
    run = FS.period_run(outs)
    assert run == {"ticks": list(FS.MONTH_TICKS), "what": "each month of 2016"}
    assert FS.period_run(outs[:2]) is None                          # two outputs: each named
    assert FS.period_run(outs[:3] + [dict(outs[3], field="NEE")]) is None
    assert FS.period_run([dict(o, region=None) for o in outs[:3]])["ticks"] == ["Jan", "Feb",
                                                                               "Mar"]
    seasons = [{"kind": "anomaly", "field": "GPP", "period": f"2016-{s}", "region": REGION}
               for s in ("MAM", "JJA", "SON")]
    assert FS.period_run(seasons) == {"ticks": ["MAM", "JJA", "SON"],
                                      "what": "each season of 2016"}
    # a panel of many bars: one precision, that of its largest value, and a
    # value that rounds to nothing is shown as 0, never as 1.9e-08
    assert FS.panel_decimals([-173.0, 31.6, -1.93e-08]) == 0
    assert FS.panel_decimals([-19.3, 1.4, -0.0412]) == 1 and FS.panel_decimals([0.0638]) == 4
    assert FS.value_text(-1.93e-08, 0) == "0" and FS.value_text(-0.0412, 1) == "0.0"
    assert FS.value_text(-19.26, 1) == "\u221219.3" and FS.value_text(-173.4, 0) == "\u2212173"
    plt = FS._plt()
    fig, ax = plt.subplots()
    refs = [FS_ref(f"P2.O{i}", o) for i, o in enumerate(outs, 1)]
    FS._draw_bars(ax, refs, {}, "panel 1 (bars)")
    assert [x.get_text() for x in ax.get_xticklabels()] == list("JFMAMJJASOND")
    plt.close(fig)
    fig, ax = plt.subplots()
    FS._title(ax, [f"P2.O{i}" for i in range(1, 13)],
              {f"P2.O{i}": o for i, o in enumerate(outs, 1)})
    assert " ".join(ax.get_title(loc="left").split()) == (
        f"P2.O1 to P2.O12: GPP anomaly, each month of 2016; {REGION}")
    plt.close(fig)


# ── C: coverage, and what step 3 is asked to say ─────────────────────────────

def test_coverage_and_the_reports_lines_follow_what_the_plan_holds():
    plan = _months_plan()
    cov = AR.coverage(Q_BOTH, plan)
    assert cov[AR.SEASONAL] == {"months": {"GPP": [f"{m:02d}" for m in range(1, 13)]},
                                "series": {}}
    assert cov[AR.SPATIAL] == {"maps": [], "splits": [], "overlaps": 0, "by_band": 0}
    said = "\n".join(AR.report_lines(Q_BOTH, plan))
    assert "THE COURSE THROUGH THE YEAR (months planned for GPP)" in said
    assert "WHERE IN THE REGION" not in said and "never a cause" in said
    # a plan with a map, its split and an overlap by band
    plan["analyses"].append({"id": "P3", "outputs": [
        {"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2016", "region": None},
        {"id": "O2", "kind": "map", "field": "TOTECOSYSC", "period": "2016"},
        {"id": "O3", "kind": "elevation_split", "region": REGION, "inputs": ["O1"]},
        {"id": "O4", "kind": "overlap", "region": REGION, "inputs": ["O1", "O2"],
         "by_band": True}]})
    assert AR.coverage(Q_BOTH, plan)[AR.SPATIAL] == {
        "maps": ["GPP", "TOTECOSYSC"], "splits": ["GPP"], "overlaps": 1, "by_band": 1}
    assert "WHERE IN THE REGION (maps split by band for GPP)" in "\n".join(
        AR.report_lines(Q_BOTH, plan))
    # a question that calls for no part, or a plan that carried out none: nothing is asked
    assert AR.report_lines(Q_NONE, plan) == [] and AR.coverage(Q_NONE, plan) == {}
    assert AR.report_lines(Q_BOTH, {"analyses": []}) == []


# ── S: the Analyzer's model is one setting ───────────────────────────────────

def test_the_analyzers_model_is_the_configs_else_the_environments_else_the_default(
        monkeypatch):
    monkeypatch.delenv(S2.MODEL_ENV, raising=False)
    assert S2.analyzer_model() == S2.DEFAULT_MODEL == "claude-opus-5-project"
    monkeypatch.setenv(S2.MODEL_ENV, " model-from-env ")
    assert S2.analyzer_model() == "model-from-env"
    assert S2.analyzer_model("model-from-config") == "model-from-config"
    monkeypatch.setenv(S2.MODEL_ENV, "")
    assert S2.analyzer_model(None) == S2.DEFAULT_MODEL
