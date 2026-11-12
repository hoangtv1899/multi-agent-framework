"""Two things the analysis plan's check now holds or says (2026-11-12).

    pytest tests/test_plan_runs_and_baseline.py -v

1. THE STUDY'S OWN RUNS AND YEARS. The check was given the catalog's fields
   and regions and nothing of the study, so a plan whose outputs named a run
   the study does not have, or a year a run does not simulate, passed it. It
   is now given the study's runs with their years (analysis_plan.study_runs)
   and names both, when the plan is written.
2. A YEAR SET AGAINST NORMAL YEARS THAT CONTAIN IT. Where the Planner names no
   normal years the control's own stand in, so a study of one of those years
   compared it with a mean that held it, and nothing said so. The plan now
   says which years are so compared, and the report repeats it. Which
   baseline a study should use is not decided here: no value changes.

On the synthetic suite of tests/test_domain_access.py; every model call is a
scripted fake and every number is made up.
"""
import copy

import domain_methods as M
from test_domain_access import PLAN_REPLY, REGION, _approve, built, study  # noqa: F401

from agents.analysis import analysis_plan as AP
from agents.analysis import step4_report as S4

FIELDS, YEARS = ["GPP"], [2015]


def _plan(**over):
    plan = copy.deepcopy(PLAN_REPLY)
    plan.pop("question_type")
    plan.update(question="What happened to GPP in 2016?", **over)
    return plan


def _diff(run, period="2016"):
    return {"id": "P1", "question": "the run against the control", "fields": ["GPP"],
            "regions": [REGION], "periods": [period], "methods": ["difference"],
            "settings": {}, "why": "what was asked",
            "outputs": [{"id": "O1", "kind": "difference", "run": run, "field": "GPP",
                         "period": period, "region": REGION}]}


def _problems(plan, runs):
    return AP.plan_problems(plan, FIELDS, [REGION], YEARS, "exploratory", runs=runs)


RUNS = {"control": [2015, 2016], "wetter": [2015, 2016]}


# ── 1. the study's own runs and years ────────────────────────────────────────
class TestTheStudysRunsAndYears:

    def test_a_plan_of_the_studys_own_runs_and_years_passes(self):
        assert _problems(_plan(), RUNS) == []
        assert _problems(_plan(analyses=[_diff("wetter")]), RUNS) == []

    def test_an_output_of_a_run_the_study_does_not_have_is_named(self):
        got = _problems(_plan(analyses=[_diff("a_run_of_another_study")]), RUNS)
        assert got == ["P1: output 'O1': no run 'a_run_of_another_study' in this study "
                       "(its runs: control, wetter)"]

    def test_a_year_the_run_does_not_have_is_named(self):
        got = _problems(_plan(analyses=[_diff("wetter", "2031")]),
                        {"control": [2015, 2016, 2031], "wetter": [2015, 2016]})
        assert got == ["P1: output 'O1': 2031 is not a year of wetter (2015 to 2016)"]

    def test_a_difference_needs_the_year_of_the_control_too(self):
        got = _problems(_plan(analyses=[_diff("wetter", "2016")]),
                        {"control": [2015], "wetter": [2015, 2016]})
        assert "P1: output 'O1': 2016 is not a year of control (2015 to 2015)" in got
        assert not any("wetter" in p for p in got)          # the run itself has the year

    def test_an_output_that_names_no_run_is_the_controls(self):
        plan = _plan()
        plan["analyses"][0]["outputs"][0]["period"] = "2031"
        plan["analyses"][0]["periods"] = ["2031"]
        assert "P1: output 'O1': 2031 is not a year of control (2015 to 2016)" in \
            _problems(plan, RUNS)

    def test_a_season_a_month_and_a_water_year_are_read_by_their_year(self):
        for period in ("2031-JJA", "2031-07", "WY2031"):
            plan = _plan()
            plan["analyses"][0]["outputs"][0]["period"] = period
            plan["analyses"][0]["periods"] = [period]
            assert any("2031 is not a year of control" in p for p in _problems(plan, RUNS)), \
                period
        # a normal years' mean is of no one year
        plan = _plan()
        plan["analyses"][0]["outputs"][0]["period"] = "normal"
        plan["analyses"][0]["periods"] = ["normal"]
        assert not any("not a year of" in p for p in _problems(plan, RUNS))

    def test_a_comparison_is_held_to_them_too(self):
        plan = _plan(comparisons=[
            {"id": "C1", "what": "w", "run": "nope", "against": "control", "years": [2016]},
            {"id": "C2", "what": "w", "run": "wetter", "against": "control",
             "years": [2016, 2031]}])
        got = _problems(plan, RUNS)
        assert "comparison C1: no run 'nope' in this study (its runs: control, wetter)" in got
        assert "comparison C2: [2031] not among the years of wetter (2015 to 2016)" in got

    def test_a_run_whose_years_are_not_on_record_is_not_judged_by_year(self):
        plan = _plan(analyses=[_diff("wetter", "2031")])
        assert _problems(plan, {"control": None, "wetter": None}) == []
        # and a run that is not there is still named
        assert _problems(_plan(analyses=[_diff("nope")]), {"control": None}) != []

    def test_a_caller_that_gives_no_runs_gets_no_such_rule(self):
        plan = _plan(analyses=[_diff("a_run_of_another_study", "2031")])
        assert AP.plan_problems(plan, FIELDS, [REGION], YEARS, "exploratory") == []

    def test_the_task_states_the_rule_the_check_applies(self):
        task = " ".join(AP.plan_task().split())
        assert "runs: an output's run is control or a run of this study" in task
        assert "comparisons: each names control or a run of this study" in task

    def test_the_studys_runs_are_read_from_its_records(self, study):
        ctx = study["ctx"]
        assert AP.study_runs(ctx) == {"control": [2015, 2016]}
        ctx.plan["experiment"]["runs"] = [{"run": "control"}, {"run": "wetter"},
                                          {"name": "drier"}]
        assert AP.study_runs(ctx) == {"control": [2015, 2016], "wetter": [2015, 2016],
                                      "drier": [2015, 2016]}

    def test_a_plan_written_for_another_study_fails_when_it_is_written(self, study):
        """What the dry run of 2026-11-12 showed: such a plan used to pass."""
        review = _approve(study, reply=dict(PLAN_REPLY,
                                            analyses=[_diff("a_run_of_another_study")]))
        plan = AP.load_plan(study["out"])
        assert any("no run 'a_run_of_another_study' in this study" in p
                   for p in plan["problems"])
        assert review["passed"] is False and AP.plan_passed(study["out"])[0] is False

    def test_the_fixtures_own_plan_still_passes(self, study):
        assert _approve(study)["passed"] is True
        assert AP.load_plan(study["out"])["problems"] == []


# ── early and late: two checks, both kept ────────────────────────────────────
class TestEarlyAndLate:
    """EARLY: a run or a year the study does not have is named when the plan
    is written (the class above). LATE: a year the study did plan, whose
    output is not there once the runs are over, is refused by the data at the
    look-up and its analysis is closed as failed with the reason. The early
    check cannot see the second case, since the year IS one the study has, so
    this test sets nothing aside: the plan check runs as it does in a study."""

    def test_a_planned_year_whose_output_is_missing_is_handled_after_the_run(self, study):
        import domain_access as DAC
        import pytest
        ctx = study["ctx"]
        ctx.plan["experiment"]["analysis"]["question_type"] = "hypothesis"
        ctx.plan["experiment"]["runs"] = [{"run": "wetter", "changes": []}]   # planned
        assert AP.study_runs(ctx) == {"control": [2015, 2016], "wetter": [2015, 2016]}
        plan = dict(PLAN_REPLY, analyses=[
            PLAN_REPLY["analyses"][0],
            {"id": "P2", "question": "the run's map", "fields": ["GPP"], "regions": [],
             "periods": ["2016"], "methods": [], "settings": {}, "why": "asked",
             "outputs": [{"id": "O1", "kind": "map", "run": "wetter", "field": "GPP",
                          "period": "2016"}]}],
            screen={"when": "last", "region": REGION, "year": 2016, "why": "a hypothesis"})
        # early: nothing to refuse, the run and the year are the study's own
        assert _approve(study, reply=plan)["passed"] is True
        assert AP.load_plan(study["out"])["problems"] == []
        # late: the run's output was never made, so the data refuses the look-up
        acc = AP.make_access(AP.access_spec(ctx), script="fig")
        acc.anomaly("GPP", "2016", region=REGION)
        with pytest.raises(DAC.AccessError, match="no suite for run 'wetter'"):
            acc.get_map("wetter", "GPP", "2016")
        rows = {r["id"]: r for r in AP.close_planned_phase(
            study["out"], when="before the screen", by="the test")["analyses"]}
        assert rows["P1"]["status"] == "completed"
        assert rows["P2"]["status"] == "failed" and rows["P2"]["unmet"] == ["O1"]
        assert "refused by the data" in rows["P2"]["reason"]
        assert "no suite for run 'wetter'" in rows["P2"]["reason"]

    def test_the_same_year_outside_the_study_is_stopped_before_any_look_up(self, study):
        ctx = study["ctx"]
        ctx.plan["experiment"]["runs"] = [{"run": "wetter", "changes": []}]
        plan = dict(PLAN_REPLY, analyses=[
            {"id": "P2", "question": "the run's map", "fields": ["GPP"], "regions": [],
             "periods": ["2031"], "methods": [], "settings": {}, "why": "asked",
             "outputs": [{"id": "O1", "kind": "map", "run": "wetter", "field": "GPP",
                          "period": "2031"}]}])
        assert _approve(study, reply=plan)["passed"] is False
        assert "P2: output 'O1': 2031 is not a year of wetter (2015 to 2016)" in \
            AP.load_plan(study["out"])["problems"]
        import pytest
        with pytest.raises(AP.PlanNotApproved):
            AP.check_gate(study["out"])


# ── 2. a year set against normal years that contain it ───────────────────────
class TestAYearInsideItsOwnNormalYears:

    def test_it_is_found_in_a_comparison_an_anomaly_and_a_timing_measure(self):
        timing = sorted(M.TIMING_IDS)[0]
        base = {"baseline": {"normal_years": [2031, 2032, 2033]}}
        assert AP.baseline_inclusive(dict(base, comparisons=[
            {"against": "normal years", "years": [2032, 2040]}])) == [2032]
        assert AP.baseline_inclusive(dict(base, analyses=[{"outputs": [
            {"kind": "anomaly", "period": "2033-JJA"}]}])) == [2033]
        assert AP.baseline_inclusive(dict(base, analyses=[{"outputs": [
            {"kind": timing, "period": "WY2031"}]}])) == [2031]
        assert AP.baseline_inclusive(dict(base, analyses=[{"outputs": [
            {"kind": "anomaly", "period": "2031"}, {"kind": "anomaly", "period": "2033"}]}],
            comparisons=[{"against": "normal years", "years": [2032]}])) == [2031, 2032, 2033]

    def test_what_is_not_set_against_the_normal_years_does_not_count(self):
        timing = sorted(M.TIMING_IDS)[0]
        base = {"baseline": {"normal_years": [2031, 2032]}}
        for plan in (
                dict(base, comparisons=[{"against": "control", "years": [2031]}]),
                dict(base, analyses=[{"outputs": [{"kind": "difference", "period": "2031"}]}]),
                dict(base, analyses=[{"outputs": [{"kind": "series", "step": "day"}]}]),
                dict(base, analyses=[{"outputs": [{"kind": timing, "period": "2031",
                                                  "against": "control"}]}]),
                dict(base, analyses=[{"outputs": [{"kind": "anomaly", "period": "2040"}]}]),
                dict(base, analyses=[{"outputs": [{"kind": "anomaly", "period": "normal"}]}]),
                {"baseline": {}, "comparisons": [{"against": "normal years",
                                                  "years": [2031]}]}):
            assert AP.baseline_inclusive(plan) == [], plan

    def test_the_plan_says_it_and_it_is_no_problem(self, study):
        study["ctx"].plan["experiment"]["analysis"]["normal_years"] = [2015, 2016]
        reply = dict(PLAN_REPLY, baseline={"normal_years": [2015, 2016], "why": "the Planner's"})
        review = _approve(study, reply=reply)
        b = AP.load_plan(study["out"])["baseline"]
        assert b["includes_compared_years"] == [2016]
        assert b["inclusive"] == (
            "the normal years include 2016, a year this plan sets against them: an anomaly "
            "of such a year is against a mean that contains the year itself, and against "
            "the other normal years alone it would be larger")
        # a label: the plan passes as before, and its normal years are unchanged
        assert review["passed"] is True and b["normal_years"] == [2015, 2016]
        assert AP.load_plan(study["out"])["problems"] == []

    def test_the_reviewer_is_shown_it(self, study):
        study["ctx"].plan["experiment"]["analysis"]["normal_years"] = [2015, 2016]
        _approve(study, reply=dict(PLAN_REPLY, baseline={"normal_years": [2015, 2016],
                                                         "why": "the Planner's"}))
        prompt = (study["out"] / "step3_plan_review_round1_prompt.txt").read_text()
        assert "a mean that contains the year itself" in prompt

    def test_a_plan_whose_normal_years_hold_no_compared_year_says_nothing(self, study):
        _approve(study)
        b = AP.load_plan(study["out"])["baseline"]
        assert "inclusive" not in b and "includes_compared_years" not in b

    def test_several_years_are_said_in_the_plural(self):
        assert "2031, 2032, years this plan sets against them" in AP.INCLUSIVE_WORDS.format(
            years="2031, 2032", which="years")

    def test_the_report_repeats_it(self):
        row = {"id": "P1", "status": "completed", "required": False, "required_by": [],
               "outputs": [], "unmet": [], "reason": "1 required output produced"}
        phase = {"closed": "after the last round", "by": "the framework",
                 "lookups_before": 3, "counts": {"completed": 1}, "analyses": [row]}
        said = "the normal years include 2032, a year this plan sets against them"
        page = "\n".join(S4.planned_phase_lines(
            {"planned_phase": phase, "analysis_plan": {"baseline": {"inclusive": said}}}))
        assert f"On the normal years: {said}." in page
        assert page.index("On the normal years") < page.index("| analysis | status |")
        plain = "\n".join(S4.planned_phase_lines(
            {"planned_phase": phase, "analysis_plan": {"baseline": {}}}))
        assert "On the normal years" not in plain and "| P1 | completed |" in plain
