"""A region's series keeps its values, each under its own period.

    pytest tests/test_series_values.py -v

A series by year, water year, season or month is one number per period. Its
record kept those numbers by a hash alone, as it keeps a map of every cell, so
a study whose plan asked for its yearly totals as series could draw them and
state none of them. The record now holds each value under its label, and a
series value is treated as any other number of a record (domain_access.
series_values). These tests hold the change to that, on a small made-up
control, with no model:

    the record      each value under its label and the covered shares beside
                    them, for the four steps whose labels are periods; a daily
                    series, a series that is too long, labels that are not
                    periods or are not each once keep the hash alone.
    what step 3     the framework's finding of a planned series holds its
    is shown        values, and they are listed under their record as values a
                    claim may state, with the rule that goes with them.
    the claim       a claim states a series value with its own year, and its
                    month or season: the value of one year under another's
                    name is refused, a count is not mistaken for a value, and
                    a record's coverage warning is carried.
    the numbers     each series value a claim states is made again from the
    check           history, as an anomaly's value is; one planted wrong is
                    caught, and a label the history does not hold cannot be
                    made again.

Every number comes from the synthetic control of tests/test_domain_suite.py.
"""
import json
import shutil

import numpy as np
import pytest

import domain_access as DAC
import domain_suite as S
from test_domain_access import Ctx, Fake
from test_domain_suite import FIELDS, make_control

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import numbers_check as NC          # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402
from agents.analysis import suite_check as SC            # noqa: E402

YEARS = [2015, 2016, 2017]
REGION = "vegetated cells"
STEPS = ("year", "wateryear", "season", "month", "day")
LINE = {step: i for i, step in enumerate(STEPS, 1)}      # each step's line of the log


def _series(i, step):
    return {"id": f"O{i}", "kind": "series", "field": "GPP", "region": REGION, "step": step}


PLAN = {
    "comparisons": [{"id": "C1", "what": "2016 against the normal year", "run": "control",
                     "against": "normal years", "years": [2016]}],
    "baseline": {"normal_years": [2015], "why": "the Planner's"},
    "analyses": [{"id": "P1", "question": "GPP in each year", "fields": ["GPP"],
                  "regions": [REGION], "periods": ["2016"], "steps": list(STEPS),
                  "methods": ["anomaly"], "settings": {}, "why": "what was asked",
                  "outputs": [_series(LINE[s], s) for s in STEPS]
                  + [{"id": "O6", "kind": "anomaly", "field": "GPP", "period": "2016",
                      "region": REGION}]}],
    "expectations": [{"id": "E1", "analysis": "P1", "field": "GPP", "sign": "higher",
                      "text": "GPP higher in 2016 than in the normal year"}],
    "competing": [{"text": "soil water", "fields": ["H2OSOI"], "analysis": "P1"}],
    "screen": {"when": "last", "region": REGION, "year": 2016, "why": "a hypothesis"}}


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("series")
    control, _ = make_control(root / "ctl", YEARS)
    S.build(control, root / "suite", fields=FIELDS)
    rd = root / "study"
    (rd / "01_inputs").mkdir(parents=True)
    (rd / "01_inputs" / "domain_control_facts.json").write_text(json.dumps(
        {"control": {"name": "synth", "run_dir": control["run_dir"],
                     "case_dir": control["case_dir"]}}))
    mp = pytest.MonkeyPatch()
    entry = dict(control, suite_dir=str(root / "suite"))
    mp.setattr(AP, "_registry_entry", lambda name: entry if name == "synth" else {})
    ctx = Ctx(rd, {"question": "How did GPP differ from year to year?",
                   "goals": ["GPP in each year"],
                   "experiment": {"control": {"name": "synth", "reuse": True}, "runs": [],
                                  "years": YEARS,
                                  "analysis": {"normal_years": [2015],
                                               "question_type": "hypothesis",
                                               "regions": [{"name": REGION}],
                                               "variables": ["GPP"]}}})
    out = rd / "04_analysis"
    out.mkdir()
    AP.write_plan(ctx, out, client=Fake(PLAN))
    S3.rule_plan_follows_question(ctx, out, client=Fake({"follows": True, "why": "yes"}))
    assert AP.plan_passed(out)[0], AP.plan_passed(out)
    made = AP.produce_required(ctx, out)                 # the framework's own calls
    assert all(c["ok"] for c in made["record"]["calls"]), made["record"]["calls"]
    log = [json.loads(x) for x in (out / "lookups.jsonl").read_text().splitlines()]
    assert [ln["keys"].get("step") for ln in log[:5]] == list(STEPS)
    yield {"rd": rd, "out": out, "ctx": ctx, "log": log, "made": made, "entry": dict(control),
           "records": CT.evidence_records(out), "plan": AP.load_plan(out)}
    mp.undo()


def _value(study, step, label):
    return study["log"][LINE[step] - 1]["result"][f"values.{label}"]


def _claim(study, text, steps=("year",), values=(), extra=(), warnings=()):
    return {"claim": text, "finding_id": "required_p1", "values": list(values),
            "evidence": [f"L{LINE[s]}" for s in steps] + list(extra),
            "evidence_class": "observation", "warnings": list(warnings)}


def _check(study, claim, records=None):
    return CT.check_claim(claim, records or study["records"], study["plan"], {}, [REGION])


# ── the record ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("step, labels", [
    ("year", ["2015", "2016", "2017"]),
    ("wateryear", ["WY2016", "WY2017"]),
    ("season", ["2015-MAM", "2016-DJF", "2017-SON"]),
    ("month", ["2015-01", "2016-07", "2017-12"])])
def test_the_record_holds_each_value_under_its_label(study, step, labels):
    ln = study["log"][LINE[step] - 1]
    got = study["made"]["values"][f"P1.O{LINE[step]}"]   # what the call returned
    res = ln["result"]
    kept = {k[len("values."):]: v for k, v in res.items() if k.startswith("values.")}
    assert list(kept) == list(got["labels"]) and set(labels) <= set(kept)
    for lab, v in zip(got["labels"], got["values"]):
        assert kept[lab] == float(v)                     # the number itself, not a rounding
    assert res["covered"] == [float(x) for x in got["covered"]]
    # the hash of what was returned, and of each array, is what it was
    assert ln["sha256"] == DAC.fingerprint(got)
    assert set(ln["arrays"]) == {"values", "covered"}
    assert res["step"] == step and res["field"] == "GPP" and res["region"] == REGION


def test_a_daily_series_keeps_its_hash_alone(study):
    ln = study["log"][LINE["day"] - 1]
    assert not [k for k in ln["result"] if k.startswith("values.")]
    assert "covered" not in ln["result"] or ln["result"]["covered"] is None
    assert set(ln["arrays"]) >= {"values"}
    assert CT.citable(ln["result"]) == {}


def test_a_series_value_is_the_number_a_single_period_call_gives(study):
    """The suite reads a region's value for a period from the file a series
    comes from, so the two are one number."""
    anomaly = study["log"][5]["result"]
    assert anomaly["value"] == _value(study, "year", "2016")
    assert anomaly["normal"] == _value(study, "year", "2015")


def _a_series(step="year", labels=("2015", "2016"), values=(1.5, 2.5), covered=(1.0, 1.0)):
    return {"field": "GPP", "region": REGION, "step": step, "labels": list(labels),
            "values": np.array(values, dtype=np.float64),
            "covered": None if covered is None else np.array(covered, dtype=np.float64),
            "units": "gC/m^2/yr", "how": "total"}


def test_what_series_values_keeps_and_what_it_does_not(monkeypatch):
    assert DAC.series_values(_a_series()) == {"values": {"2015": 1.5, "2016": 2.5},
                                              "covered": [1.0, 1.0]}
    # a period the suite holds no value for is kept as none, never as a number
    gap = DAC.series_values(_a_series(values=(np.nan, 2.5), covered=(np.nan, 0.5)))
    assert gap == {"values": {"2015": None, "2016": 2.5}, "covered": [None, 0.5]}
    assert DAC.series_values(_a_series(covered=None))["covered"] is None
    # none of these can be named by a period, each once
    assert DAC.series_values(_a_series(step="day", labels=("2015-01-01", "2015-01-02"))) is None
    assert DAC.series_values(_a_series(labels=("2015", "2015"))) is None
    assert DAC.series_values(_a_series(labels=("2015", "WY2016"))) is None
    assert DAC.series_values(_a_series(labels=("first", "second"))) is None
    assert DAC.series_values(_a_series(labels=("2015",))) is None        # one label, two values
    assert DAC.series_values(_a_series(labels=(), values=(), covered=())) is None
    assert DAC.series_values(np.array([1.5, 2.5])) is None
    assert DAC.series_values({"step": "year"}) is None
    monkeypatch.setattr(DAC, "SERIES_VALUES_KEPT", 1)
    assert DAC.series_values(_a_series()) is None                         # too long to list


def test_only_a_series_keeps_an_array_by_label():
    """A method's result that happens to hold labels and values keeps its
    arrays by their hashes, as before."""
    res = DAC.flat_result("anomaly", _a_series())
    assert not [k for k in res if k.startswith("values.")] and "covered" not in res
    res = DAC.flat_result("get_series", _a_series())
    assert res["values.2015"] == 1.5 and res["values.2016"] == 2.5
    assert res["covered"] == [1.0, 1.0] and res["labels"] == ["2015", "2016"]


# ── what step 3 is shown ─────────────────────────────────────────────────────

def test_the_frameworks_finding_holds_a_planned_series_values(study):
    f = AP.required_findings(study["out"], study["made"]["record"], study["plan"])[0]
    assert f["result"]["P1.O1"] == {f"values.{y}": _value(study, "year", str(y)) for y in YEARS}
    assert set(f["result"]["P1.O4"]) == {f"values.{y}-{m:02d}" for y in YEARS
                                         for m in range(1, 13)}
    assert "no numbers of its own" in f["result"]["P1.O5"]["returned"]
    assert "a daily one" in f["result"]["P1.O5"]["returned"]
    assert "a map" not in f["result"]["P1.O5"]["returned"]


def test_step_3_is_shown_the_values_under_their_record_with_the_rule(study):
    f = AP.required_findings(study["out"], study["made"]["record"], study["plan"])[0]
    lines = CT.finding_evidence_lines(f, study["records"])
    text = "\n".join(lines)
    v = _value(study, "year", "2016")
    assert f"P1.O1.values.2016 = {v!r}" in text
    assert f"P1.O2.values.WY2017 = {_value(study, 'wateryear', 'WY2017')!r}" in text
    # under the yearly series' own record, before the next record's line
    block = text.split("L1: get_series")[1].split("L2: get_series")[0]
    assert f"P1.O1.values.2016 = {v!r}" in block
    assert "a series by year: one value for each period, each stated with its own period" \
        in block
    # the rule itself is said once for the finding, above its records
    assert text.count(CT.SERIES_RULE) == 1
    assert text.index("of its series: " + CT.SERIES_RULE) < text.index("L1: get_series")
    # no value of a kept series is a number the script made itself
    made = text.split("numbers the script made itself")[1]
    assert "values." not in made
    # the daily series' record says nothing of values, and carries no rule
    day = text.split("L5: get_series")[1].split("L6: anomaly")[0]
    assert "a series by" not in day and "its values in the finding's result: none" in day


# ── the claim ────────────────────────────────────────────────────────────────

def test_a_series_value_stated_with_its_year_passes(study):
    a, b = _value(study, "year", "2016"), _value(study, "year", "2017")
    t = _check(study, _claim(
        study, f"Over the vegetated cells, GPP was {a:.5g} gC/m^2/yr in 2016 and {b:.5g} "
               f"gC/m^2/yr in 2017.", values=[a, b]))
    assert t["problems"] == [], t["problems"]
    assert {"L1:values.2016", "L1:values.2017"} <= set(t["matched"])


def test_two_years_joined_by_and_with_a_length_unit_are_both_years(study):
    """THE FIRST LIVE REPORT (2026-11-12) wrote its yearly values as "117 mm
    in water year 2017 and 25.7 mm in water year 2020". The trace's reading
    of years leaves out one that opens a range in a length unit ("2000 to
    2500 m"), and it took "2017 and 25.7 mm" for such a range, so true claims
    were struck for not naming the year they named. The unit is of no account
    to the trace, so the made-up control's values stand in with one."""
    a, b = _value(study, "year", "2016"), _value(study, "year", "2017")
    text = (f"Over the vegetated cells, the total was {a:.5g} mm in 2016 and {b:.5g} mm "
            f"in 2017.")
    assert "2016" not in CT.years_named(text)            # the older reading, as it is
    assert CT.years_written(text) == {"2016", "2017"}
    t = _check(study, _claim(study, text, values=[a, b]))
    assert t["problems"] == [], t["problems"]
    assert {"L1:values.2016", "L1:values.2017"} <= set(t["matched"])
    words = "it was 117 mm in water year 2017 and 25.7 mm in water year 2020"
    assert CT.label_named("WY2017", words) and CT.label_named("WY2020", words)
    assert not CT.label_named("WY2018", words)
    # a number with a unit straight after it is a value, never a year
    assert CT.years_written("a band from 2000 m to 2500 m in 2016") == {"2016"}


def test_the_value_of_one_year_under_anothers_name_is_refused(study):
    a = _value(study, "year", "2016")
    t = _check(study, _claim(
        study, f"Over the vegetated cells, GPP was {a:.5g} gC/m^2/yr in 2017.", values=[a]))
    assert len(t["problems"]) == 1, t["problems"]
    assert "is a series' value for 2016 (L1), a period the claim does not name" in \
        t["problems"][0]
    assert CT.SERIES_RULE in t["problems"][0]


def test_a_series_value_with_no_year_is_refused(study):
    a = _value(study, "year", "2016")
    t = _check(study, _claim(
        study, f"Over the vegetated cells, GPP was {a:.5g} gC/m^2/yr.", values=[a]))
    assert any("a period the claim does not name" in p for p in t["problems"]), t["problems"]


def test_a_water_year_is_named_by_its_year_or_by_its_label(study):
    a = _value(study, "wateryear", "WY2017")
    for words in ("in water year 2017", "in WY2017"):
        t = _check(study, _claim(
            study, f"Over the vegetated cells, GPP was {a:.5g} gC/m^2 {words}.",
            steps=("wateryear",), values=[a]))
        assert t["problems"] == [], (words, t["problems"])
    t = _check(study, _claim(
        study, f"Over the vegetated cells, GPP was {a:.5g} gC/m^2 in water year 2016.",
        steps=("wateryear",), values=[a]))
    assert any("a series' value for WY2017 (L2)" in p for p in t["problems"]), t["problems"]


def test_a_monthly_value_needs_its_month_and_its_year(study):
    jul = _value(study, "month", "2016-07")
    ok = _check(study, _claim(
        study, f"Over the vegetated cells, GPP in July 2016 was {jul:.5g} gC/m^2.",
        steps=("month",), values=[jul]))
    assert ok["problems"] == [], ok["problems"]
    for words in ("June 2016", "July 2017"):
        bad = _check(study, _claim(
            study, f"Over the vegetated cells, GPP in {words} was {jul:.5g} gC/m^2.",
            steps=("month",), values=[jul]))
        assert any("a series' value for 2016-07 (L4)" in p for p in bad["problems"]), \
            (words, bad["problems"])


def test_a_run_of_months_names_every_month_in_it(study):
    may, jul = _value(study, "month", "2016-05"), _value(study, "month", "2016-07")
    t = _check(study, _claim(
        study, f"Over the vegetated cells, from April to July 2016 GPP passed through "
               f"{may:.5g} gC/m^2 and reached {jul:.5g} gC/m^2.", steps=("month",),
        values=[may, jul]))
    assert t["problems"] == [], t["problems"]
    # "between April and July" is no range the trace reads, so May is not named
    t = _check(study, _claim(
        study, f"Over the vegetated cells, between April and July 2016 GPP passed through "
               f"{may:.5g} gC/m^2.", steps=("month",), values=[may]))
    assert any("a series' value for 2016-05 (L4)" in p for p in t["problems"]), t["problems"]
    assert CT.months_named("from November to February") == {"11", "12", "01", "02"}
    assert CT.months_named("in May, and again in Sept.") == {"05", "09"}
    assert CT.months_named("it may rise") == set()


def test_a_seasons_value_needs_its_season_and_its_year(study):
    jja = _value(study, "season", "2016-JJA")
    ok = _check(study, _claim(
        study, f"Over the vegetated cells, GPP in summer 2016 was {jja:.5g} gC/m^2.",
        steps=("season",), values=[jja]))
    assert ok["problems"] == [], ok["problems"]
    for words in ("summer 2017", "spring 2016"):          # another year, another season
        bad = _check(study, _claim(
            study, f"Over the vegetated cells, GPP in {words} was {jja:.5g} gC/m^2.",
            steps=("season",), values=[jja]))
        assert any("a series' value for 2016-JJA (L3)" in p for p in bad["problems"]), \
            (words, bad["problems"])
    # a winter is the year of its January
    assert CT.label_named("2016-DJF", "in winter 2016") is True
    assert CT.label_named("2016-DJF", "in winter 2015") is False


def test_a_number_another_record_gives_for_its_own_period_is_not_held_to_a_label(study):
    """The 2016 anomaly's own value is the series' 2016 value. A claim that
    rests on both states it as the anomaly's, whose record is of 2016 alone."""
    a = _value(study, "year", "2016")
    t = _check(study, _claim(
        study, f"Over the vegetated cells, GPP was {a:.5g} gC/m^2/yr in the year the plan "
               f"asked about.", values=[a], extra=["L6"]))
    assert t["problems"] == [], t["problems"]


def test_a_count_is_not_mistaken_for_a_series_value():
    rec = {"id": "L9", "call": "get_series"}
    hits = [(rec, "values.2019")]
    assert CT.series_period_problems("8", None, hits, "over the 8 water years") == []
    assert CT.series_period_problems("12", None, hits, "in all 12 cases") == []
    for word, unit in (("8", "mm"), ("8.0", None), ("153", None)):
        got = CT.series_period_problems(word, unit, hits, "it fell over the record")
        assert len(got) == 1 and "2019 (L9)" in got[0], (word, unit, got)
    # the label named, or a value that is no series value: no problem
    assert CT.series_period_problems("8", "mm", hits, "it was 8 mm in 2019") == []
    assert CT.series_period_problems("8", "mm", [(rec, "anomaly")], "it was 8 mm") == []
    assert CT.series_period_problems("8", "mm", [({"id": "L2", "call": "anomaly"},
                                                  "values.2019")], "it was 8 mm") == []


def test_where_several_labels_give_the_number_naming_one_is_enough():
    rec = {"id": "L9", "call": "get_series"}
    hits = [(rec, "values.2017"), (rec, "values.2019")]
    assert CT.series_period_problems("310", "mm", hits, "it was 310 mm in 2019") == []
    got = CT.series_period_problems("310", "mm", hits, "it was 310 mm in 2018")
    assert "2017 (L9), 2019 (L9)" in got[0]


def _with_a_gap(study, tmp_path):
    """The study's log with one more line: a yearly series whose second year
    covers nine tenths of the region's area, as the access would log it."""
    out = tmp_path / "04_analysis"
    shutil.copytree(study["out"], out)
    value = _a_series(labels=("2015", "2016"), values=(41.5, 52.5), covered=(1.0, 0.9))
    AP.LookupLog(out, script="fig")({
        "call": "get_series", "keys": {"run": "control", "field": "GPP", "region": REGION,
                                       "step": "year"}, "method": None, "settings": None,
        "sha256": DAC.fingerprint(value), "shape": DAC._shape(value),
        "result": DAC.flat_result("get_series", value), "planned": True, "plan_items": ["P1"]})
    recs = CT.evidence_records(out)
    return recs, sorted(recs, key=lambda i: int(i[1:]))[-1]


def test_a_series_that_covers_part_of_the_region_carries_its_warning(study, tmp_path):
    recs, rid = _with_a_gap(study, tmp_path)
    r = recs[rid]
    assert r["support"]["area_share"] == 0.9
    assert [w["code"] for w in r["warnings"]] == ["coverage"]
    assert "area share covered 0.9" in r["warnings"][0]["reason"]
    text = "Over the vegetated cells, GPP was 52.5 gC/m^2/yr in 2016."
    c = {"claim": text, "finding_id": "f", "values": [52.5], "evidence": [rid],
         "evidence_class": "observation", "warnings": []}
    bad = _check(study, c, records=recs)
    assert any("leaves out the warning" in p and "coverage" in p for p in bad["problems"])
    ok = _check(study, dict(c, warnings=["coverage"]), records=recs)
    assert ok["problems"] == [], ok["problems"]


# ── the numbers check ────────────────────────────────────────────────────────

def _stated(study):
    """Claims as the trace leaves them: each names its record and the values
    it states, by path."""
    return [{"claim": "the years", "evidence": ["L1"],
             "evidence_trace": {"matched": ["L1:values.2016", "L1:values.2017"]}},
            {"claim": "a water year, a season and a month", "evidence": ["L2", "L3", "L4"],
             "evidence_trace": {"matched": ["L2:values.WY2017", "L3:values.2016-JJA",
                                            "L3:values.2017-DJF", "L4:values.2016-07"]}},
            {"claim": "a figure of the days", "evidence": ["L5"]}]


def test_each_series_value_a_claim_states_is_made_again_from_the_history(study):
    rec = NC.check_evidence(study["rd"], _stated(study), registry=study["entry"], write=False)
    assert rec["pass"], rec.get("error") or rec["mismatches"][:5]
    assert rec["n_compared"] == 6 and rec["n_mismatched"] == 0
    assert rec["records"]["L1"]["compared"] == 2 and rec["records"]["L3"]["compared"] == 2
    assert rec["records"]["L2"]["compared"] == 1 and rec["records"]["L4"]["compared"] == 1
    assert "each the region's value for the period of its label" in \
        rec["records"]["L1"]["notes"][0]
    # the values no claim states are named as not made again, and are no fault
    left = {x["record"]: x for x in rec["not_recomputed"]}
    assert left["L1"]["paths"] == ["values.2015"] and left["L4"]["n"] == 35
    # the daily series logs no number, as before
    assert rec["records"]["L5"]["compared"] == 0
    assert "no claim states a value of this series" in rec["records"]["L5"]["note"]
    read = {r.split(": ")[1].split(" from ")[0] for r in rec["sources"]["reads"]}
    assert read == {"GPP"}, read


def test_a_series_no_claim_states_a_value_of_is_not_made_again(study):
    rec = NC.check_evidence(study["rd"], [{"claim": "a figure", "evidence": ["L1"]}],
                            registry=study["entry"], write=False)
    assert rec["pass"] and rec["n_compared"] == 0
    assert rec["records"]["L1"]["compared"] == 0 and "note" in rec["records"]["L1"]


def _planted(study, tmp_path, line, path, new):
    rd = tmp_path / "planted"
    (rd / "04_analysis").mkdir(parents=True)
    shutil.copytree(study["rd"] / "01_inputs", rd / "01_inputs")
    shutil.copy(study["out"] / AP.PLAN_FILE, rd / "04_analysis" / AP.PLAN_FILE)
    lines = (study["out"] / "lookups.jsonl").read_text().splitlines()
    rec = json.loads(lines[line - 1])
    rec["result"][path] = new
    lines[line - 1] = json.dumps(rec)
    (rd / "04_analysis" / "lookups.jsonl").write_text("\n".join(lines) + "\n")
    return rd


@pytest.mark.parametrize("line, path", [(1, "values.2016"), (2, "values.WY2017"),
                                        (3, "values.2017-DJF"), (4, "values.2016-07")])
def test_a_series_value_planted_wrong_in_its_record_is_caught(study, tmp_path, line, path):
    old = study["log"][line - 1]["result"][path]
    rd = _planted(study, tmp_path, line, path, old * 1.001)
    rec = NC.check_evidence(rd, _stated(study), registry=study["entry"], write=False)
    assert not rec["pass"] and len(rec["mismatches"]) == 1, rec["mismatches"]
    m = rec["mismatches"][0]
    assert m["record"] == f"L{line}" and m["path"] == path
    assert m["logged"] == old * 1.001 and m["recomputed"] == pytest.approx(old, rel=1e-12)
    why = NC.evidence_reason(rec)
    assert f"L{line}" in why and path in why


def test_a_value_no_claim_states_is_not_compared(study, tmp_path):
    """Only what a claim states is made again: the rest of a series is named
    as not made again, so a fault there is not a fault of the report."""
    old = _value(study, "year", "2015")
    rd = _planted(study, tmp_path, 1, "values.2015", old * 2.0)
    rec = NC.check_evidence(rd, _stated(study), registry=study["entry"], write=False)
    assert rec["pass"]
    assert {x["record"]: x for x in rec["not_recomputed"]}["L1"]["paths"] == ["values.2015"]


def test_a_stated_value_the_history_does_not_hold_fails_the_check(study, tmp_path):
    rd = _planted(study, tmp_path, 1, "values.2031", 12.5)
    claims = [{"claim": "a year the run never made", "evidence": ["L1"],
               "evidence_trace": {"matched": ["L1:values.2031"]}}]
    rec = NC.check_evidence(rd, claims, registry=study["entry"], write=False)
    assert not rec["pass"]
    m = rec["mismatches"][0]
    assert m["record"] == "L1" and m["path"] == "values.2031" and m["recomputed"] is None


def test_the_trace_and_the_check_agree_on_one_claim(study):
    """From the claim's words to the history: the trace matches the values a
    claim states to their labels, and the check makes those again."""
    a, b = _value(study, "year", "2016"), _value(study, "wateryear", "WY2017")
    c = _claim(study, f"Over the vegetated cells, GPP was {a:.5g} gC/m^2/yr in 2016 and "
                      f"{b:.5g} gC/m^2 in water year 2017.", steps=("year", "wateryear"),
               values=[a, b])
    c["evidence_trace"] = _check(study, c)
    assert c["evidence_trace"]["problems"] == []
    rec = NC.check_evidence(study["rd"], [c], registry=study["entry"], write=False)
    assert rec["pass"] and rec["n_compared"] == 2
    assert SC._cited([c], study["log"]) == {"L1": ["L1:values.2016".split(":")[1]],
                                            "L2": ["values.WY2017"]}
