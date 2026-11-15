"""Three false strikes of the claim trace, mended without letting a wrong
claim through (the report rebuild, package A7), and the texts the model is
shown, unchanged. Made-up records, no model call.

    pytest tests/test_report_rebuild_trace.py -v

1. A season's word over records that hold each of its months: kept when the
   records hold every month of the season in the year meant; still struck
   for some of the months, another year's months, a number under the
   season's name, the season given as one quantity, or the season's code.
2. The control run's own setting: the control's value of a setting a new run
   changes is a fact of the design, as the run's own value is; any other
   number is still struck.
3. A number in scientific notation is one number, in the trace and in step
   3's audit; one that is not its record's value is still struck.
"""
import hashlib
import json

import pytest

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402

REGION = "basin, vegetated"
RUN = "wet_run"


def _rec(rid, period, values, run=RUN, field="GPP"):
    """A difference record as claim_trace.evidence_records makes one."""
    names = CT._names({"period": period, "step": None}, {})
    return {"id": rid, "lookup": int(rid[1:]), "call": "difference", "method": "difference",
            "measure": None,
            "keys": {"run": run, "control": "control", "field": field, "period": period,
                     "region": REGION},
            "settings": None, "run": run, "field": field, "period": period, "region": REGION,
            "step": None, "planned": True, "plan_items": ["P1"], "amendment": None,
            "label": None, "after_screen": None, "after_close": False,
            "values": dict(values), "words": {}, "comparison": {"against": "the control"},
            "support": {}, "across_settings": None, "warnings": [], "rows": {},
            "row_warnings": {},
            "names": {"years": sorted(names["years"]), "periods": sorted(names["periods"])},
            "percent_allowed": True, "percent_why_not": None}


def _months(*periods):
    return [_rec(f"L{i}", p, {"difference": -0.25 * i, "percent": -0.5 * i})
            for i, p in enumerate(periods, 1)]


def _problems(text, recs, values=(), runs=None):
    c = {"claim": text, "finding_id": "f", "values": list(values),
         "evidence": [r["id"] for r in recs], "evidence_class": "observation", "warnings": []}
    return CT.check_claim(c, {r["id"]: r for r in recs}, {"question_type": "exploratory"},
                          runs or {}, [REGION])["problems"]


SPRING = ("2032-03", "2032-04", "2032-05")
WINTER = ("2031-12", "2032-01", "2032-02")


# ── 1. a season's word over a record for each of its months ─────────────────

@pytest.mark.parametrize("text,periods", [
    ("Plant growth stayed lower from September through the next spring.",
     ("2031-09",) + SPRING),
    ("In spring 2032 plant growth was lower in each month.", SPRING),
    ("In winter 2032 plant growth was lower.", WINTER),
    ("Plant growth was lower over the basin in spring.", SPRING),
], ids=["no year", "its year", "winter from the December before", "the basin"])
def test_a_season_word_over_its_months_is_kept(text, periods):
    assert _problems(text, _months(*periods)) == []


@pytest.mark.parametrize("text,periods,why", [
    ("Plant growth stayed lower from September through the next spring.",
     ("2031-09", "2032-03", "2032-05"), "some of its months"),
    ("In spring 2032 plant growth was lower.", ("2031-03", "2031-04", "2031-05", "2032-01"),
     "another year's months"),
    ("In winter 2032 plant growth was lower.", ("2032-12", "2032-01", "2032-02"),
     "the wrong December"),
    ("In spring 2031 and 2032 plant growth was lower.", ("2031-03", "2031-04") + SPRING,
     "one of two years"),
    ("In spring 2032 plant growth was 0.5 gC/m2 lower.", SPRING, "a month's number"),
    ("The spring 2032 total of plant growth was lower.", SPRING, "one quantity"),
    ("In MAM 2032 plant growth was lower.", SPRING, "the season's code alone"),
    ("In spring 2032 (MAM) plant growth was lower.", SPRING, "the season's code"),
], ids=["some months", "another year", "the wrong December", "one of two years",
        "a number under the season", "an aggregate", "the code alone", "the code too"])
def test_the_wrong_form_of_a_season_word_is_still_struck(text, periods, why):
    probs = _problems(text, _months(*periods))
    assert any("names MAM" in p or "names DJF" in p for p in probs), (why, probs)


def test_a_number_in_another_sentence_does_not_strike_the_season_word():
    """The rule looks at the sentence that names the season: a measured
    number in the next sentence is that sentence's own, and the season word
    still stands; the same number in the season's sentence still strikes."""
    recs = _months("2031-07", "2031-09", *SPRING)          # L1 is July, -0.25
    two = ("Plant growth stayed lower from September through the next spring. "
           "In July 2031 it was 0.25 gC/m2 lower.")
    assert _problems(two, recs, [-0.25]) == []
    one = "Plant growth stayed 0.25 gC/m2 lower from September through the next spring."
    assert any("names MAM" in p for p in _problems(one, recs, [-0.25]))


def _by_month(rid, years):
    """A record by month (step "month") that names its years, with no month
    label of its own."""
    r = _rec(rid, None, {"mean": 1.5})
    nm = CT._names({"period": None, "step": "month"}, {"years": list(years)})
    r.update(step="month", names={"years": sorted(nm["years"]), "periods": sorted(nm["periods"])})
    return r


def test_a_record_by_month_holds_every_month_of_its_years():
    assert _problems("In spring 2032 plant growth was lower.", [_by_month("L1", [2032])]) == []
    probs = _problems("In spring 2032 plant growth was lower.", [_by_month("L1", [2031])])
    assert any("names MAM" in p for p in probs), probs


def test_a_derived_record_holds_the_months_of_its_inputs():
    """An overlap or a split holds the periods of the records it was made
    from: each month of the season among them makes the word stand."""
    def derived(periods):
        r = _rec("L4", None, {"r": 0.5})
        r.update(call="overlap", method="overlap",
                 inputs=[{"id": f"L{i}", "period": p} for i, p in enumerate(periods, 1)])
        # its names, as claim_trace.evidence_records gives a derived record
        nms = [CT._names({"period": p, "step": None}, {}) for p in periods]
        r["names"] = {"years": sorted(set().union(*(n["years"] for n in nms))),
                      "periods": sorted(set().union(*(n["periods"] for n in nms)))}
        return r
    assert _problems("In spring 2032 plant growth was lower.", [derived(SPRING)]) == []
    probs = _problems("In spring 2032 plant growth was lower.", [derived(SPRING[:2])])
    assert any("names MAM" in p for p in probs), probs


@pytest.mark.parametrize("text,long", [
    ("It rose by 3.81234 \u00d7 10\u207b\u00b3 in 2031.", True),
    ("It rose by 3.8123 \u00d7 10\u207b\u00b3 in 2031.", False),
    ("It rose by 3.81234e-3 in 2031.", True),
    ("It rose by 0.00381 in 2031.", True),
    ("It rose by 0.0038 in 2031.", False),
], ids=["a five-decimal mantissa", "a four-decimal mantissa", "e-notation",
        "a plain five-decimal number", "a plain four-decimal number"])
def test_the_readability_gate_counts_a_mantissas_decimals(text, long):
    probs = S3.readability_problems(text, set(), "claim 1")
    assert any("has too many digits" in p for p in probs) is long, probs


def test_a_small_measured_value_is_held_at_its_own_precision():
    """A measured value below 1e-4 is written by Python in e-notation; it was
    read as having no decimals, so any written number within 0.5 of it
    passed. A wrong small number is now flagged and a right one kept."""
    assert S3._decimals(6.2e-05) == 6 and S3._decimals(1.25e-07) == 9
    assert S3._decimals(1.5e16) == 0 and S3._decimals(296.0) == 0
    assert S3._decimals("3.8e-3") == 4 and S3._decimals("3e5") == -5     # as written
    assert not S3.same_number("0.3", 6.2e-05)
    assert not S3.same_number("0.0002", 6.2e-05)
    for right in ("0.000062", "0.0001", "6.2e-5", "0.00006"):
        assert S3.same_number(right, 6.2e-05), right
    assert S3.number_problems("It fell by 0.000062 mm in 2031.", [-6.2e-05], "claim 1") == []
    assert S3.number_problems("It fell by 6.2 \u00d7 10\u207b\u2075 mm in 2031.", [-6.2e-05],
                              "claim 1") == []
    for wrong in ("0.3 mm", "0.0002 mm", "8.5e-5 mm"):
        bad = S3.number_problems(f"It fell by {wrong} in 2031.", [-6.2e-05], "claim 1")
        assert len(bad) == 1 and wrong in bad[0], (wrong, bad)
    # a larger value keeps the precision it always had
    assert S3.same_number("296", 296.3) and not S3.same_number("0.3", 0.011)


def test_a_season_held_by_its_own_record_is_named_as_before():
    r = _rec("L1", "2032-MAM", {"difference": -1.5})
    assert _problems("In spring 2032 plant growth was 1.5 gC/m2 lower.", [r], [-1.5]) == []


# ── 2. the control run's own setting ────────────────────────────────────────

CO2 = {"id": "co2_ppmv", "params": {"ppmv": 500}}


def test_the_settings_a_change_sets_are_the_servers_own_list():
    assert set(CT.change_settings("co2_ppmv")) == {"CCSM_CO2_PPMV", "co2_ppmv"}
    assert CT.change_settings("no_such_change") == []
    held = {"co2_ppmv": 300.0, "CCSM_CO2_PPMV": "300.0", "use_nofire": False,
            "stream_year_first_ndep": 1999}
    assert CT.control_settings([CO2], held) == {"co2_ppmv": 300.0, "CCSM_CO2_PPMV": 300.0}
    assert CT.control_settings([{"id": "fire_off", "params": {}}], held) == {}


def test_study_runs_carries_the_controls_value_of_each_changed_setting(tmp_path):
    (tmp_path / "experiment.json").write_text(json.dumps({
        "experiment": {"runs": [{"run": "control", "changes": []},
                                {"run": RUN, "changes": [CO2]}]},
        "control_facts": {"settings": {"co2_ppmv": 300.0, "CCSM_CO2_PPMV": "300.0",
                                       "stream_year_first_ndep": 1999}},
        "chain": {"runs": {RUN: {"stages": [{"stage": "case", "check_pass": True}]}}}}))
    runs = CT.study_runs(tmp_path)
    assert runs[RUN]["control_settings"] == {"CCSM_CO2_PPMV": 300.0, "co2_ppmv": 300.0}
    assert runs[RUN]["verified"] is True


RUNS = {RUN: {"changes": [CO2], "verified": True, "why": "w",
              "control_settings": {"co2_ppmv": 300.0, "CCSM_CO2_PPMV": 300.0}}}


def test_the_control_runs_setting_beside_the_runs_is_kept():
    r = _rec("L1", "2031", {"difference": 40.5})
    text = "With CO2 held at 500 ppm instead of 300 ppm, plant growth was 40.5 gC/m2 higher."
    assert _problems(text, [r], [40.5], RUNS) == []


@pytest.mark.parametrize("text,run,number", [
    ("With CO2 held at 500 ppm instead of 310 ppm, plant growth was 40.5 gC/m2 higher.", RUN,
     "310"),
    ("With the control run's CO2 at 300 ppm, plant growth was 40.5 gC/m2 higher.", "dry_run",
     "300"),
], ids=["another number", "a record of another run"])
def test_a_number_that_is_no_design_fact_is_still_struck(text, run, number):
    r = _rec("L1", "2031", {"difference": 40.5}, run=run)
    probs = _problems(text, [r], [40.5], dict(RUNS, dry_run={"changes": [], "verified": True}))
    assert [p for p in probs if "matches no value" in p] == [
        f"the number {number} matches no value of its records (L1)"], probs


# ── 3. a number in scientific notation ──────────────────────────────────────

@pytest.mark.parametrize("written", ["3.8 \u00d7 10\u207b\u00b3", "3.8 x 10^-3", "3.8\u00d710-3",
                                     "3.8e-3", "3.8 \u00d7 10^\u22123"])
def test_scientific_notation_is_one_number(written):
    assert [w for _, _, w in CT.sci_numbers(f"It rose by {written} in 2031.")] == ["3.8e-3"]
    assert S3._numbers(written) == ["0.0038"]
    r = _rec("L1", "2031-JJA", {"difference": 0.003812})
    text = f"The summer soil water index rose by {written} in 2031."
    assert _problems(text, [r], [0.003812]) == []


@pytest.mark.parametrize("written", ["3.9 \u00d7 10\u207b\u00b3", "3.8 \u00d7 10\u207b\u00b2",
                                     "3.8e-4"])
def test_a_scientific_number_that_is_not_its_records_value_is_still_struck(written):
    r = _rec("L1", "2031-JJA", {"difference": 0.003812})
    probs = _problems(f"The summer soil water index rose by {written} in 2031.", [r],
                      [0.003812])
    assert any("matches no value of its records" in p for p in probs), probs


def test_a_plain_product_is_not_scientific_notation():
    assert CT.sci_numbers("a 2 x 10 grid of 5 by 10 cells, 10-20 mm") == []
    assert CT._half_unit("3.8e-3") == pytest.approx(5e-5)
    assert CT._half_unit("8.2") == pytest.approx(0.05)


def test_step_3s_audit_reads_scientific_notation():
    assert S3.number_problems("rose by 3.8 \u00d7 10\u207b\u00b3 mm", [0.003812], "claim 1") == []
    bad = S3.number_problems("rose by 3.9e-3 mm", [0.003812], "claim 1")
    assert len(bad) == 1 and "3.9e-3 mm" in bad[0]
    assert S3.number_problems("rose by 3.8e-3 percent", [0.000038], "claim 1") == []
    long = S3.readability_problems("It rose by 3.81234e-3 in 2031.", set(), "claim 1")
    assert any("3.81234e-3 has too many digits" in p for p in long), long
    assert not S3.readability_problems("It rose by 3.8e-3 in 2031.", set(), "claim 1")


# ── the texts the model is shown, byte for byte as at fdb7f07 ───────────────
# Stage B of the report rebuild (2026-11-14) changed two of them on purpose:
# suite_task gained SUITE_FINDINGS, and the reply template's answer is at most
# five sentences, each part of the question with its key number. Every other
# text is as at fdb7f07.

def _sha(x):
    return hashlib.sha256(("\n".join(x) if isinstance(x, list) else x).encode()).hexdigest()


RECORD = {"id": "L3", "call": "difference", "method": "difference", "measure": None,
          "keys": {"run": "wet", "control": "control", "field": "GPP", "period": "2031-07",
                   "region": "box, vegetated"},
          "field": "GPP", "period": "2031-07", "region": "box, vegetated", "run": "wet",
          "plan_items": ["P1"], "label": None, "amendment": None, "after_close": False,
          "warnings": [], "row_warnings": {}, "percent_allowed": True, "percent_why_not": None,
          "values": {"difference": -1.5, "percent": -2.5}, "words": {},
          "comparison": {"against": "the control (control)", "run": "wet", "years": ["2031"]},
          "support": {"years": None, "area_share": 1.0, "n": None}, "across_settings": None}
FINDING = {"id": "f1", "lookups": [{"id": "L3"}], "result": {"d": -1.5, "own": 7.25}}
SHOWN = {
    "plan_task": (lambda: AP.plan_task(),
                  "a240978bda52b7c5b808745b23163e826b45af0ffe0d245edc1b2e88ce8c026e"),
    "step 3's task": (lambda: S3.TASK,
                      "27b226f277eb9cdb76cb5f9f1d018d27ed6dbc38b768d06943b4a1c5d8c206b6"),
    "suite_task": (lambda: S3.suite_task(),
                   "3488077337accb48aa473e31e14af63555da3bc3bd7f02f75c85f1f16ebb8115"),
    "SUITE_RULES": (lambda: S3.SUITE_RULES,
                    "8054184823b16a9f5b55cfead20a1451686ffb81903f495941b86a288fc08460"),
    "REWRITE": (lambda: S3.REWRITE,
                "54eef1739fcf03a1375edfea36807e0f6b11e953f6e9cb316606eede129280ca"),
    "VERDICT_ASK": (lambda: S3.VERDICT_ASK,
                    "a183d4f6f22f950e101530f1ee2627d5d1fef6b25572e843725e62c2a563d099"),
    "PLAN_REVIEW_ASK": (lambda: S3.PLAN_REVIEW_ASK,
                        "faa8bbfacdfa67fb4dd9aeb3f50cc2fc839270259fb8840a7c29477217f06c8a"),
    "claims_reply_lines": (lambda: CT.claims_reply_lines(),
                           "5adf885b4f45822cb7969e800e5434ba58312eb5baff56a57f115532a7d7bc9d"),
    "class_lines": (lambda: CT.class_lines(),
                    "4eb6c91cfddcd2f0ac1fd111e729677007782287305b98daa17f60dd545c2b42"),
    "run_lines": (lambda: CT.run_lines(),
                  "d7fd0460e35cbd22c03c043064d568a53e097f33f5cb303b83e43159ad550162"),
    "warning_lines": (lambda: CT.warning_lines(),
                      "79e6a2ce93f49a7277d2b376b6bbaac4b90d9d68a1afece6a564f7849b233606"),
    "finding_evidence_lines": (
        lambda: CT.finding_evidence_lines(FINDING, {"L3": RECORD}),
        "9119b2c36c5e57d1324873423c9c48488d73372b6b8767b6836ea3f23c7fd1ed")}


@pytest.mark.parametrize("name", list(SHOWN))
def test_the_text_the_model_is_shown_is_unchanged(name):
    render, sha = SHOWN[name]
    assert _sha(render()) == sha, name
