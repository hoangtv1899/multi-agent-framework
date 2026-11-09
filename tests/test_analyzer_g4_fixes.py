"""The fixes after the framework's third acceptance batch (2026-11-08):
positive and negative tests, synthetic, with no model call.

    pytest tests/test_analyzer_g4_fixes.py -v

E1  a series names what its days hold: a day label ("2015-03-12") names its
    year, never its day as a month; a month only when every day of it is
    held, and a season only when its three months are held whole as one run,
    winter from the December before (revised the same day); every other
    label is read as before. A claim naming a year or a season its records
    do not hold still fails: period matching stays strict.
E2  a length that defines a field ("10 cm" of SOILWATER_10CM, "2 m" of TSA)
    is that field's label only where the claim writes it as the field's
    definition does ("the top 10 cm", "at 2 m"); "rose 10 cm" is an amount
    and a value to trace (revised the same day), as is the same number in
    another unit or about a field it does not define.
F   each parameter of the server's runs is shown to step 3 with its form and
    an example its check passes (domain_changes.PARAM_FORMS, beside the
    check); months written "6,7,8" still fail the check, and a run on the
    list proposed wrongly is said so, not called a run the server cannot make.
T   the plan check's timing rule, as revised the same day: each timing word
    of a question needs a timing measure of what the word is about (its
    clause's words), for a year the question names; an expectation of
    "earlier" or "later" is tested by a measure of its own field; and the
    rule asks the plan for no direction.
"""
import datetime as dt
import json

import pytest

from g2_replay_support import Replies, forbid_model_clients
from test_analyzer_g2_replay import (FOLLOWS, NO_EVIDENCE, P_GPP, P_MAPS, E1, _figure,  # noqa: F401
                                     _hypothesis_study, _plan, _run)
from test_domain_access import REGION, built, study           # noqa: F401

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402

AP._elm()
import domain_catalog as DCAT                            # noqa: E402
import domain_changes as DCH                             # noqa: E402

R = "watershed, vegetated"


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


def _days(a, b):
    d, e, out = dt.date.fromisoformat(a), dt.date.fromisoformat(b), []
    while d <= e:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def _records(tmp_path, lines):
    """Evidence records of a lookups log written as given (each line an
    answered, planned lookup)."""
    out = tmp_path / "04_analysis"
    out.mkdir(exist_ok=True)
    with open(out / AP.LOOKUPS_FILE, "w") as fh:
        for i, ln in enumerate(lines, 1):
            fh.write(json.dumps(dict({"at": "t", "script": AP.REQUIRED_SCRIPT, "method": None,
                                      "settings": None, "planned": True, "plan_items": ["P1"],
                                      "sha256": f"h{i}"}, **ln)) + "\n")
    return CT.evidence_records(out)


def _series(labels, field="H2OSNO", step="day", flat=True):
    res = {"field": field, "region": R, "step": step, "units": "mm"}
    if flat:
        res.update({f"labels.{i}": lab for i, lab in enumerate(labels)})
    else:
        res["labels"] = list(labels)
    return {"call": "get_series", "keys": {"run": "control", "field": field, "region": R,
                                           "step": step}, "result": res}


def _names(tmp_path, labels, **kw):
    r = _records(tmp_path, [_series(labels, **kw)])["L1"]
    return set(r["names"]["years"]), {tuple(p) for p in r["names"]["periods"]}


def _months(*mm):
    return {("month", m) for m in mm}


def _check(c, recs):
    return CT.check_claim(c, recs, {"question_type": "hypothesis"}, {}, [R])["problems"]


def _claim(text, values=(), rid="L1"):
    return {"claim": text, "finding_id": "f", "values": list(values), "evidence": [rid],
            "evidence_class": "observation", "warnings": []}


# ── E1: a series names what its days hold ───────────────────────────────────

def test_a_daily_series_names_the_years_months_and_seasons_its_days_hold(tmp_path):
    yrs, per = _names(tmp_path, _days("2015-01-01", "2016-12-31"))
    assert yrs == {"2015", "2016"}
    assert per == _months(*[f"{m:02d}" for m in range(1, 13)]) | {
        ("season", s) for s in ("DJF", "MAM", "JJA", "SON")}


@pytest.mark.parametrize("flat", [True, False], ids=["one key per label", "a list"])
def test_a_day_is_never_read_as_a_month(tmp_path, flat):
    days = _days("2015-03-01", "2015-03-31")                     # "2015-03-12" among them
    yrs, per = _names(tmp_path, days, flat=flat)
    assert yrs == {"2015"} and per == _months("03")


def test_a_month_is_named_only_when_every_day_of_it_is_held(tmp_path):
    assert _names(tmp_path, _days("2015-03-12", "2015-03-31"))[1] == set()
    assert _names(tmp_path, _days("2015-02-01", "2015-02-28"))[1] == _months("02")


@pytest.mark.parametrize("labels,seasons", [
    (_days("2015-03-01", "2015-03-31"), set()),                                 # March only
    (_days("2015-03-01", "2015-05-31"), {"MAM"}),
    (_days("2015-06-01", "2015-06-30") + _days("2016-07-01", "2016-08-31"), set()),
    (_days("2014-12-01", "2015-02-28"), {"DJF"}),                     # across the new year
    (_days("2015-01-01", "2015-02-28") + _days("2016-12-01", "2016-12-31"), set()),
    (_days("2015-01-01", "2015-02-28") + _days("2015-12-01", "2015-12-31"), set()),
    (_days("2015-01-01", "2015-12-31"), {"MAM", "JJA", "SON"}),
    (_days("2014-12-02", "2015-02-28"), set()),
], ids=["one month", "a whole spring", "a summer split across years",
        "a winter across the new year", "a winter's months years apart",
        "a year's January, February and December", "a calendar year",
        "a winter missing its first day"])
def test_a_season_is_named_only_when_one_years_days_hold_its_three_months(tmp_path, labels,
                                                                          seasons):
    _, per = _names(tmp_path, labels)
    assert {p[1] for p in per if p[0] == "season"} == seasons


def test_every_other_label_is_read_as_before(tmp_path):
    recs = _records(tmp_path, [
        _series(["2015-01", "2015-02"], step="month"),
        _series(["2015-DJF", "2015-MAM"], step="season"),
        {"call": "anomaly", "keys": {"run": "control", "field": "GPP", "period": "2015-JJA",
                                     "region": R}, "method": "anomaly",
         "result": {"anomaly": -1.5, "normal_years": [2016, 2017]}}])
    # a month's series names every month, a season's every season (their steps)
    assert set(recs["L1"]["names"]["years"]) == {"2015"} and \
        {tuple(p) for p in recs["L1"]["names"]["periods"]} == _months(
            *[f"{m:02d}" for m in range(1, 13)])
    assert {tuple(p) for p in recs["L2"]["names"]["periods"]} == {
        ("season", s) for s in ("DJF", "MAM", "JJA", "SON")}
    assert set(recs["L3"]["names"]["years"]) == {"2015", "2016", "2017"} and \
        {tuple(p) for p in recs["L3"]["names"]["periods"]} == {("season", "JJA")}


def test_a_claim_traces_on_what_its_series_holds_and_fails_on_what_it_lacks(tmp_path):
    """Question 1's claim 2 in small: a daily snow series of 2015, "by the
    start of spring". The same claim naming a year or a season the series
    does not hold still fails."""
    recs = _records(tmp_path, [_series(_days("2015-01-01", "2015-12-31"))])
    ok = "The daily snow-water curve for 2015 falls to near zero by the start of spring."
    assert _check(_claim(ok), recs) == []
    assert _check(_claim(ok.replace("2015", "2017")), recs) == [
        "it names 2017, which none of its records names"]
    winter = _records(tmp_path, [_series(_days("2015-01-01", "2015-02-28"))])
    assert _check(_claim(ok), winter) == ["it names MAM, which none of its records names"]


def test_a_daily_series_from_the_suite_names_its_years(study):
    """Through the gate on the synthetic suite: the series is logged one label
    per key, and its record names the suite's years and seasons."""
    _hypothesis_study(study)
    plan = _plan("hypothesis", [dict(P_GPP, steps=["day"], outputs=[
        {"id": "O1", "kind": "series", "field": "GPP", "region": REGION, "step": "day"}])],
        [E1])
    c = Replies({"a": [plan], "b": [FOLLOWS]}).clients()
    assert AP.write_plan(study["ctx"], study["out"], client=c["a"])["problems"] == []
    from agents.analysis import step3_interpret as S3
    S3.rule_plan_follows_question(study["ctx"], study["out"], client=c["b"])
    acc = AP.make_access(AP.access_spec(study["ctx"]), script="fig")
    ser = acc.get_series("control", "GPP", REGION, "day")
    assert len(ser["labels"]) == 730                       # the model's calendar: no leap day
    ln = AP.lookups(study["out"])[0]
    assert "labels.0" in ln["result"] and "labels" not in ln["result"]
    r = CT.evidence_records(study["out"])["L1"]
    assert r["names"]["years"] == ["2015", "2016"]
    assert {p[1] for p in r["names"]["periods"] if p[0] == "season"} == {
        "DJF", "MAM", "JJA", "SON"}


# ── E2: a length that defines a field is its label ──────────────────────────

def test_a_fields_defining_lengths_are_read_from_its_code_and_its_meaning():
    notes = DCAT.load_notes()
    assert CT.field_quantities("SOILWATER_10CM", notes) == [(10.0, "cm")]
    assert CT.field_quantities("TSOI_10CM", notes) == [(10.0, "cm")]
    assert CT.field_quantities("TSA", notes) == [(2.0, "m")]          # "air at 2 m"
    for f in ("GPP", "H2OSNO", "ZWT", "QCHARGE", None):
        assert CT.field_quantities(f, notes) == [], f


def _soil(tmp_path, field="SOILWATER_10CM"):
    return _records(tmp_path, [{"call": "anomaly", "method": "anomaly",
                                "keys": {"run": "control", "field": field,
                                         "period": "2022-JJA", "region": R},
                                "result": {"anomaly": 2.2575103364200544,
                                           "percent": 18.21707628512872,
                                           "normal_years": [2016, 2017]}}])


SOIL = "Summer water in the top 10 cm of soil was 2.26 kg per square metre, or 18.2 percent, " \
       "above the normal years."


def test_a_fields_own_depth_is_its_label_not_a_value(tmp_path):
    """Question 2's claim 10 in small."""
    recs = _soil(tmp_path)
    assert recs["L1"]["field_quantities"] == [(10.0, "cm")]
    assert _check(_claim(SOIL, [2.2575103364200544, 18.21707628512872]), recs) == []


@pytest.mark.parametrize("text,bad", [
    (SOIL.replace("10 cm", "20 cm"), "the number 20 cm"),          # a depth it does not define
    (SOIL.replace("10 cm", "10 mm"), "the number 10 mm"),          # another unit
    (SOIL + " The top 10 cm stayed wet.", None),                   # the label, said again
    (SOIL + " The 10-cm soil water is a layer.", None),            # as a compound
    (SOIL + " It rose 10 cm.", "the number 10 cm"),                # an amount: no label
    (SOIL + " The soil water held 10 cm more.", "the number 10 cm"),
    (SOIL.replace("18.2 percent", "10 percent"), "the number 10 percent"),
], ids=["another depth", "another unit", "the same label again", "a compound label",
        "an amount of the same length", "an amount before a comparative", "a percent"])
def test_another_depth_or_unit_is_still_a_value(tmp_path, text, bad):
    probs = _check(_claim(text, [2.2575103364200544]), _soil(tmp_path))
    if bad is None:
        assert probs == [], probs
    else:
        assert any(p.startswith(bad + " matches no value") for p in probs), probs


def test_a_length_is_its_fields_label_only_in_its_fields_own_phrase():
    """The false acceptance closed: the soil water "rose 10 cm" states an
    amount; "the top 10 cm", "10 cm of soil", "10-cm soil water", "at 2 m"
    and "the 2 m air temperature" are the fields' own phrases."""
    notes = DCAT.load_notes()
    soil, air = CT.field_labels("SOILWATER_10CM", notes), CT.field_labels("TSA", notes)
    assert soil == [{"value": 10.0, "unit": "cm", "before": ["top"], "after": ["of soil"],
                     "words": ["land", "liquid", "soil", "vegetated"]}]

    def label(text, labs):
        return [CT.written_as_label(text, end, float(w), lab) for w, _, end in
                CT._number_words(text) for lab in labs]
    for t in ("the top 10 cm", "10 cm of soil", "the 10-cm soil water"):
        assert label(t, soil) == [True], t
    for t in ("the soil water rose 10 cm.", "10 cm of water", "the 10 cm water"):
        assert label(t, soil) == [False], t
    for t in ("air temperature at 2 m", "the 2 m air temperature"):
        assert label(t, air) == [True], t
    assert label("the snowpack was 2 m deep", air) == [False]
    assert label("the top 10 cm held 10 cm more water", soil) == [True, False]


def test_a_depth_about_a_field_it_does_not_define_is_still_a_value(tmp_path):
    probs = _check(_claim(SOIL, [2.2575103364200544, 18.21707628512872]),
                   _soil(tmp_path, field="GPP"))
    assert any(p.startswith("the number 10 cm matches no value") for p in probs), probs


# ── F: each parameter's form ────────────────────────────────────────────────

def _params():
    return [(c, k, p) for c in DCH.declare()["changes"]
            for k, p in (c.get("params") or {}).items()]


def test_every_parameter_kind_has_a_form_and_its_example_passes_the_check():
    ps = _params()
    assert {p["kind"] for _, _, p in ps} == set(DCH.PARAM_FORMS)    # no form unused, none missing
    for c, k, p in ps:
        ex = DCH.param_example(p)
        assert DCH._param_problems({"id": c["id"], "params": {k: p}}, {k: ex}) == [], (c["id"], k)
        assert DCH.PARAM_FORMS[p["kind"]] in DCH.param_form(p)
        assert DCH.param_form(p).endswith(f", as {json.dumps(ex)}")


def test_step_3_is_shown_each_parameters_form_from_the_server():
    flat = " ".join(" ".join(CT.run_lines()).split())
    for c, k, p in _params():
        assert f"{k}: {DCH.param_form(p)}" in flat, (c["id"], k)
    assert "months: a list of distinct month numbers 1-12, as [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, " \
        "11, 12]" in flat
    assert "temperature_K: a number from -10 to 10 K, as 0.0" in flat
    assert "map: a map of model years to recorded years, as {\"2016\": 2015}" in flat


def test_months_written_as_text_still_fail_and_the_run_is_named_as_proposed_wrongly():
    bad = {"id": "weather_scale", "params": {"temperature_K": 2.0, "months": "6,7,8"}}
    with pytest.raises(DCH.ChangeError, match="is not a list of distinct months 1-12"):
        DCH.resolve(bad)
    _, probs = CT.check_proposals([{"cause": "c", "proposed_run": {"changes": [bad]}}])
    assert probs == ["open cause 1: 'weather_scale' cannot be made as proposed: weather_scale "
                     "months='6,7,8' is not a list of distinct months 1-12"]
    good = {"id": "weather_scale", "params": {"temperature_K": 2.0, "months": [6, 7, 8]}}
    assert CT.check_proposals([{"cause": "c", "proposed_run": {"changes": [good]}}])[1] == []
    _, probs = CT.check_proposals([{"cause": "c", "proposed_run": {"changes": [
        {"id": "snow_water_initial_increase", "params": {}}]}}])
    assert probs[0].startswith("open cause 1: 'snow_water_initial_increase' is not a run the "
                               "server can make: ")


# ── T: the plan check's timing rule ─────────────────────────────────────────

@pytest.mark.parametrize("q,words", [
    ("Did an early snow loss in 2015 accompany the summer GPP decline over the watershed?",
     ["early"]),
    ("Did snow melt 20 days later in 2015?", ["later"]),
    ("Did the snow melt out early?", ["melt out", "early"]),
    ("Was the meltout delayed?", ["meltout", "delayed"]),
    ("Did green-up come sooner in 2016?", ["green-up", "sooner"]),
    ("Was the timing of peak snow shifted?", ["timing"]),
    ("Did the onset of photosynthesis move?", ["onset"]),
    ("Was 2015 an early-melt year?", ["early"]),
    ("Did snow go late in the season?", ["late"]),
    ("What changed in 2022 compared with the normal years?", []),
    ("Which elevation bands lost the most snow in 2015, and did they also lose the most "
     "summer growth?", []),
    ("Did early summer GPP fall in 2015?", []),
    ("Did late-summer drought cut growth?", []),
    ("Was GPP lower than in earlier years?", []),
    ("Did GPP recover two years later?", []),
    ("Was late 2015 drier?", []),
    ("Did the late-May snow last?", []),
    ("When did the fire burn?", []),
], ids=lambda x: x if isinstance(x, str) else None)
def test_the_words_that_make_a_question_one_about_timing(q, words):
    assert AP.timing_words(q) == words


TIMING_Q = "Did growth peak earlier in 2016?"
P_PEAK = {"id": "P4", "question": "the peak's date", "fields": [], "regions": [REGION],
          "periods": ["2016"], "steps": [], "methods": ["growth_peak"], "settings": {},
          "why": "the question asks when",
          "outputs": [{"id": "O1", "kind": "growth_peak", "region": REGION, "period": "2016"}]}
P_MELT = {"id": "P5", "question": "the melt-out", "fields": [], "regions": [REGION],
          "periods": ["WY2016"], "steps": [], "methods": ["snow_gone"], "settings": {},
          "why": "the question asks when",
          "outputs": [{"id": "O1", "kind": "snow_gone", "region": REGION, "period": "WY2016"}]}
E_PEAK = {"id": "E2", "analysis": "P4", "field": "GPP", "sign": "earlier",
          "text": "growth peaks earlier in 2016 than in the normal year"}
GROWTH = ("timing: the question's 'earlier' is about growth and peak, but no analysis has "
          "growth_peak among its outputs for 2016")


def _timing(plan, question=TIMING_Q, qt="hypothesis"):
    return [p for p in AP.plan_problems(dict(plan, question=question), ["GPP", "QCHARGE"],
                                        [REGION], [2015], qt) if p.startswith("timing:")]


def test_a_timing_question_needs_a_measure_of_what_it_is_about():
    assert _timing(_plan("hypothesis", [P_GPP], [E1])) == [GROWTH]
    # a snow measure does not answer a question about growth's timing
    assert _timing(_plan("hypothesis", [P_GPP, P_MELT], [E1])) == [GROWTH]
    assert _timing(_plan("hypothesis", [P_GPP, P_PEAK], [E1])) == []


def test_the_measure_is_for_a_year_the_question_names():
    assert _timing(_plan("hypothesis", [P_GPP, P_PEAK], [E1]),
                   question="Did growth peak earlier in 2017?") == [
        GROWTH.replace("for 2016", "for 2017")]
    assert _timing(_plan("hypothesis", [P_GPP, P_PEAK], [E1]),
                   question="Did growth peak earlier than in the normal years?") == []


def test_the_rule_asks_for_no_direction():
    """The question says "earlier", and the plan states no earlier or later
    expectation: the rule asks for the measure only."""
    assert _timing(_plan("hypothesis", [P_GPP, P_PEAK], [E1])) == []
    assert "never for an expectation or a direction" in AP._timing_rule_words()


def test_each_timing_word_needs_the_measure_of_its_own_clause():
    q = "Did the snow melt out earlier, and did growth peak later in 2016?"
    assert [n["measures"] for n in AP.timing_needs(q)] == [["snow_gone"], ["growth_peak"]]
    assert _timing(_plan("hypothesis", [P_GPP, P_PEAK], [E1]), question=q) == [
        "timing: the question's 'melt out' is about snow and melt-out, but no analysis has "
        "snow_gone among its outputs for 2016"]
    assert _timing(_plan("hypothesis", [P_GPP, P_PEAK, P_MELT], [E1]), question=q) == []


@pytest.mark.parametrize("q,measures", [
    ("Did an early snow loss in 2015 accompany the summer GPP decline?", ["snow_gone"]),
    ("Was the timing of peak snow shifted?", ["snow_peak"]),
    ("Did the onset of photosynthesis move in 2022?", ["photosynthesis_onset"]),
    ("Did runoff peak sooner?", ["half_runoff"]),
    ("Did green-up come sooner in 2016?", ["leaf_green_up"]),
    ("Was the 2015 GPP decline early?", ["photosynthesis_onset", "growth_peak"]),
    ("Did the peak come earlier?", ["snow_peak", "growth_peak"]),
], ids=["snow loss", "peak snow", "photosynthesis onset", "runoff peak", "green-up",
        "a tie of the GPP measures", "a tie of the peaks"])
def test_the_most_specific_measure_is_the_one_a_timing_word_needs(q, measures):
    assert [n["measures"] for n in AP.timing_needs(q)] == [measures]


def test_a_timing_word_about_nothing_a_measure_dates_needs_none():
    assert AP.timing_needs("Did the 2016 drought arrive early?") == []
    assert _timing(_plan("hypothesis", [P_GPP], [E1]),
                   question="Did the 2016 drought arrive early?") == []


def test_an_exploratory_question_about_timing_needs_the_measure_too():
    assert _timing(_plan("exploratory", [P_GPP]), qt="exploratory") == [GROWTH]
    assert _timing(_plan("exploratory", [P_GPP, P_PEAK]), qt="exploratory") == []


def test_an_earlier_expectation_is_tested_by_a_measure_of_its_own_field():
    # without any timing word in the question, and with a measure elsewhere
    q = "What happened to growth in 2016?"
    wrong = dict(E_PEAK, analysis="P1")
    assert _timing(_plan("hypothesis", [P_GPP, P_PEAK], [E1, wrong]), question=q) == [
        "timing: E2 expects 'earlier' for GPP, but its analysis P1 has no "
        "photosynthesis_onset or growth_peak among its outputs"]
    snow = dict(E_PEAK, field="H2OSNO")                 # its analysis dates growth, not snow
    assert _timing(_plan("hypothesis", [P_GPP, P_PEAK], [E1, snow]), question=q) == [
        "timing: E2 expects 'earlier' for H2OSNO, but its analysis P4 has no snow_peak or "
        "snow_gone among its outputs"]
    recharge = dict(E_PEAK, field="QCHARGE")
    probs = _timing(_plan("hypothesis", [P_GPP, P_PEAK], [E1, recharge]), question=q)
    assert len(probs) == 1 and probs[0].startswith(
        "timing: E2 expects 'earlier' for QCHARGE, which no timing measure dates (H2OSNO: "
        "snow_peak, snow_gone; TLAI: leaf_green_up; GPP: photosynthesis_onset, growth_peak; "
        "QRUNOFF: half_runoff)")
    assert _timing(_plan("hypothesis", [P_GPP, P_PEAK], [E1, E_PEAK]), question=q) == []


def test_a_question_with_no_timing_word_meets_the_rule_as_before():
    for q in ("What happened to GPP in 2016?", "Did early summer GPP fall in 2016?", None):
        assert _timing(_plan("hypothesis", [P_GPP], [E1]), question=q) == []


def test_the_plan_task_states_the_timing_rule_from_the_checks_own_words():
    rule = next(r for r in AP.PLAN_RULES if r.id == "timing")
    assert rule.scope == "plan"
    task = " ".join(AP.plan_task().split())
    assert " ".join(rule.text().split()) in task
    for w in AP.TIMING_QUESTION_WORDS:
        assert w in rule.text()
    for mid, words in AP.TIMING_TOPICS.items():
        assert f"{mid} dates {', '.join(words)}" in task, mid
    AP._elm()
    import domain_methods as M
    assert set(AP.TIMING_TOPICS) == set(M.TIMING_IDS)          # every measure, none more
    for mid in M.TIMING_IDS:                                  # each dates what its name says
        assert any(w in mid.replace("_", " ") for w in AP.TIMING_TOPICS[mid]), mid


def test_a_timing_failure_takes_the_one_rewrite_and_the_rewrite_is_told_why(study):
    _hypothesis_study(study)
    study["ctx"].plan["question"] = TIMING_Q
    first = _plan("hypothesis", [P_GPP], [E1])
    second = _plan("hypothesis", [P_GPP, P_PEAK], [E1, E_PEAK])
    r = Replies({"step2a_plan": [first], "step3_plan_review": [FOLLOWS],
                 "step2a_plan_rewrite": [second], "step3_plan_review_rewrite": [FOLLOWS],
                 "step2_investigate": [{"reasoning": "r", "notes": "", "figures": []}],
                 "step3_interpret": [NO_EVIDENCE]})
    _run(study, r)
    v1 = json.loads((study["out"] / AP.PLAN_V1_FILE).read_text())
    assert [p for p in v1["problems"] if p.startswith("timing:")] == _timing(first) == [GROWTH]
    rewrite = " ".join(r.prompts("step2a_plan_rewrite")[0].split())
    assert GROWTH in rewrite
    assert AP.plan_passed(study["out"])[0] and AP.load_plan(study["out"])["problems"] == []
    calls = {x["label"]: x for x in AP.produce_required(study["ctx"], study["out"])[
        "record"]["calls"]}
    assert calls["P4.O1"]["ok"] is True


# ── the product-specific WY2015 clarification (2026-11-08, second set) ──────

def test_a_left_out_period_names_its_products_and_each_timing_measures_answer():
    import dataclasses
    import domain_catalog as C
    import domain_methods as M
    assert C.left_out_scope("wateryear", "WY2015") == (
        " Only the wateryear products lack it; snow_gone dates it as a partial-record "
        "estimate; snow_peak and half_runoff are not estimable.")
    # a kind no timing measure uses says nothing more (the catalog's room is small)
    assert C.left_out_scope("season", "2015-DJF") == ""
    # read from the contracts, never written: a start-of-record check moves a measure
    m = M.METHODS["snow_gone"]
    start = next(c for c in M.METHODS["snow_peak"].applicability
                 if c.id == "record_holds_the_start")
    M.METHODS["snow_gone"] = dataclasses.replace(m, applicability=m.applicability + (start,))
    try:
        moved = C.left_out_scope("wateryear", "WY2015")
    finally:
        M.METHODS["snow_gone"] = m
    assert moved == (" Only the wateryear products lack it; snow_peak and snow_gone and "
                     "half_runoff are not estimable.")
