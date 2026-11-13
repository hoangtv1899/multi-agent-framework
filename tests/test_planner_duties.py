"""The Planner's duties for the rebuilt design page (2026-11-13).

    pytest tests/test_planner_duties.py -v

The page a person decides on shows a table of comparisons and the limits of
the answer. Two of its parts are written by a model and asked for here: the
Planner's limits for the question, and what each comparison tells the person.
These tests hold the words the two prompts carry and that the plan's check
asks for neither.
"""
import copy
from pathlib import Path

from agents.analysis import analysis_plan as AP

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ["vegetated cells"]
PLAN = {
    "question": "a made-up question", "question_type": "exploratory",
    "comparisons": [{"id": "C1", "what": "the run minus the control", "run": "control",
                     "against": "normal years", "years": [2033]}],
    "baseline": {"normal_years": [2032, 2034], "why": "made up"},
    "analyses": [{"id": "P1", "title": "Monthly change in plant growth",
                  "question": "In which months of 2033 does growth depart from normal?",
                  "tells": "When the response starts and how long it lasts",
                  "fields": ["LEAFX"], "regions": ["vegetated cells"], "periods": ["2033"],
                  "steps": [], "methods": ["anomaly"], "settings": {}, "why": "asked",
                  "outputs": [{"id": "O1", "kind": "anomaly", "field": "LEAFX",
                               "period": "2033", "region": "vegetated cells"}]}],
    "expectations": [], "competing": [],
    "screen": {"when": "first", "region": "vegetated cells", "year": 2033, "why": "first"},
    "problems": []}


def _flat(text):
    return " ".join(text.split())


def _planner():
    text = (ROOT / "src" / "agents" / "prompts" / "planner.txt").read_text()
    return text, _flat(text[text.index("THE WHOLE-DOMAIN PATH"):text.index("THE COLUMN BUDGET")])


class TestThePlannerNamesTheLimitsForTheQuestion:
    def test_it_is_asked_for_them_from_the_controls_known_errors(self):
        text, wd = _planner()
        assert "THE LIMITS FOR THIS QUESTION are yours to judge." in wd
        assert "name in domain_experiment.question_limits at most two limits" in wd
        assert "Read the control's `known_errors_words`" in wd
        assert "`evaluation`" not in wd          # measured numbers stay away from it
        assert '"question_limits": [' in text

    def test_each_says_the_error_in_words_and_its_effect_as_a_judgement(self):
        _, wd = _planner()
        assert "in the record's words and without numbers" in wd
        assert 'with "could" or "may"' in wd
        assert "Where the control lists no known error, or none bears on the question, the " \
               "list is empty" in wd

    def test_they_repeat_nothing_the_page_prints_by_itself_and_reach_the_report(self):
        _, wd = _planner()
        assert "Do not repeat a change's own limit or what the run does not contain" in wd
        assert "Copy each limit into caveats too, word for word" in wd

    def test_the_older_rules_on_observations_still_stand(self):
        _, wd = _planner()
        assert "No caveat carries an observation's numbers" in wd
        assert "controls[].observations" not in wd


class TestTheKnownErrorsReachThePlannerInWordsAlone:
    ENTRY = {"name": "valley", "description": "a made-up valley",
             "known_errors_words": ["It holds less snow than is measured and melts it earlier",
                                    "Its runoff is 30 percent too high",
                                    "  Its plant growth in summer   is lower than satellites show "],
             "page_summary": {"version": 1, "text": ["A made-up valley."],
                              "compared": ["Its snow is 40 percent of the measured."]},
             "evaluation": {"summary": "It holds 40 percent of the measured snow."}}

    def test_the_server_gives_the_statements_and_leaves_out_one_with_a_number(self):
        import domain_controls as DCT
        assert DCT.summary(self.ENTRY, None)["known_errors_words"] == [
            "It holds less snow than is measured and melts it earlier",
            "Its plant growth in summer is lower than satellites show"]
        assert DCT.summary({"name": "bare"}, None)["known_errors_words"] == []

    def test_the_planners_copy_keeps_them_and_loses_the_pages_summary(self):
        import domain_controls as DCT
        from agents import planner as PL
        row = DCT.summary(self.ENTRY, None)
        (got,) = PL._rulebook_for_prompt({"controls": {"controls": [row]}})["controls"]["controls"]
        assert got["known_errors_words"] == row["known_errors_words"]
        assert "page_summary" not in got and "evaluation" not in got
        assert "40 percent" not in str(got)            # no measured number in its copy
        assert row["page_summary"]["compared"]         # the row itself is not changed

    def test_reception_is_shown_neither(self):
        import domain_controls as DCT
        from agents import reception_llm as RL
        row = DCT.summary(self.ENTRY, None)
        (got,) = RL.controls_for_the_llm({"controls": {"controls": [row]}})["controls"]["controls"]
        for key in ("known_errors_words", "page_summary", "evaluation"):
            assert got[key] == RL.SHOWN_ON_THE_DESIGN_PAGE
        assert "snow" not in str(got) and "40 percent" not in str(got)


class TestEachComparisonSaysWhatItTells:
    def test_the_planning_call_is_asked_for_it_beside_the_title_and_the_question(self):
        names = [p.name for p in AP.PLAN_PARTS["analyses"]["item"]]
        assert names[:4] == ["id", "title", "question", "tells"]
        task = _flat(AP.plan_task())
        assert "tells (what its result tells the person who asked, as one phrase of at " \
               "most twelve plain words that begins with How, Whether, When or Where)" in task

    def test_the_analyses_that_answer_the_question_come_first(self):
        task = _flat(AP.plan_task())
        assert "Put first the analyses that answer the question most directly" in task
        assert "lists the first four" in task

    def test_a_plan_with_the_phrase_and_one_without_it_both_pass_the_check(self):
        with_it = copy.deepcopy(PLAN)
        without = copy.deepcopy(PLAN)
        del without["analyses"][0]["tells"]
        for plan in (with_it, without):
            got = AP.plan_problems(plan, ["LEAFX"], REGIONS, [2032, 2033, 2034], "exploratory")
            assert not [p for p in got if "tells" in p], got
        assert AP.plan_problems(with_it, ["LEAFX"], REGIONS, [2032, 2033, 2034],
                                "exploratory") == \
            AP.plan_problems(without, ["LEAFX"], REGIONS, [2032, 2033, 2034], "exploratory")
