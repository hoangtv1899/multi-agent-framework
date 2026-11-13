"""The short design page: what a person decides on, and nothing a decision
rests on left out (2026-11-12).

    pytest tests/test_design_page.py -v

The page was about 1,700 words for one new run and read like a form. It is
now short, in full sentences, with the design in full and the steps to make
the runs by hand in two files of their own. SHORT MUST NOT MEAN SILENT. These
tests hold the page to what the user asked of it:

  1. a difference between the question's own word and what the run does
     (rain asked, rain and snow changed) is said where the run is described;
  2. the places an analysis reads are given with their cells and area;
  3. a daily or monthly series is given with the years it covers;
  4. declining is not said to cost nothing;
  5. an analysis is listed by its short title, or by its question when the
     plan gave none, under the same id as in the details;
  6. every caveat, every finding of the model server, every remark of the
     plan's review and every thing decided without the person is on the page
     whole, and so are the conditions code states: an untested change, a
     change in effect in every simulated year, a year set against normal
     years that contain it, a run not made yet;
  7. the page says, where it shows a plan and where it asks, that approval of
     the analysis is not built yet.

Every name, year and number here is made up, and no test reaches a model.
"""
import copy
import json
from pathlib import Path

import pytest

import domain_changes as DC
from agents import analysis_preview as AV
from agents import design_page as DP
from agents import design_review as dr

RUN = "dry_summer_2033"
CHANGE = {"id": "weather_scale", "params": {"precip_factor": 0.7, "months": [6, 7, 8],
                                            "years": [2033]}}
QUESTION = ("What if June to August rainfall had been 30 percent lower in 2033 in the "
            "valley? How would plant growth respond?")
SUMMARY = {"version": 3, "written": "2031-02-03",
           "text": ["The prepared run is a model run of a made-up valley for 2031 to 2036.",
                    "Its snow is about half of what was measured."]}
CONTROL = {
    "name": "valley", "usable": True,
    "description": "the Valley control: a made-up box, weather from a made-up record",
    "page_summary": SUMMARY,
    "not_in_it_words": ["the observed floods: the run makes its own"],
    "evaluation": {"summary": "It holds half of the measured snow.", "source": "a note"},
    "suite": "version 1, built 2031-01-01 00:00 UTC, 0.10 GB: 3 fields",
    "facts": {"start_year": 2031, "output_years": [2031, 2036], "hours_per_model_year": 0.5,
              "grid": {"land_cells": 1200, "land_area_km2": 900.0},
              "fire": {"on": True, "sentence": "the fire is the model's own, lit by a "
                                               "made-up lightning file"}}}
CAVEATS = ["The difference between the runs is not a change of climate.",
           "The fire in both runs is the model's own."]
UNUSUAL = {"where": f"run {RUN}, change weather_scale", "what": "a made-up thing to note"}
REMARKS = ["The question says rainfall and the run scales snow too.",
           "The daily series name no period."]
INCLUSIVE = ("the normal years include 2033, a year this plan sets against them: an anomaly "
             "of such a year is against a mean that contains the year itself, and against "
             "the other normal years alone it would be larger")
PLAN = {
    "question": QUESTION, "question_type": "exploratory",
    "comparisons": [{"id": "C1", "what": "the run minus the control", "run": RUN,
                     "against": "control", "years": [2033]},
                    {"id": "C2", "what": "2033 against the normal years", "run": "control",
                     "against": "normal years", "years": [2033]}],
    "baseline": {"normal_years": [2032, 2033, 2034], "why": "made up",
                 "normal_years_from": "the control's own (the Planner named none)",
                 "includes_compared_years": [2033], "inclusive": INCLUSIVE},
    "analyses": [
        {"id": "P1", "title": "Monthly change in plant growth",
         "question": "In which months of 2033 does plant growth depart from the control, "
                     "and when does the departure begin and end?",
         "fields": ["LEAFX"], "regions": ["vegetated cells"],
         "periods": ["2033-06", "2033-07", "2033-08"], "steps": [], "methods": ["difference"],
         "settings": {}, "why": "asked",
         "outputs": [{"id": "O1", "kind": "difference", "run": RUN, "field": "LEAFX",
                      "period": "2033-06", "region": "vegetated cells"}]},
        {"id": "P2",
         "question": "How do the daily courses of growth in the two runs compare?",
         "fields": ["LEAFX", "SOILX"], "regions": ["vegetated cells", "valley floor"],
         "periods": [], "steps": ["day"], "methods": [], "settings": {}, "why": "asked",
         "outputs": [{"id": "O1", "kind": "series", "run": RUN, "field": "LEAFX",
                      "region": "vegetated cells", "step": "day"},
                     {"id": "O2", "kind": "series", "run": "control", "field": "LEAFX",
                      "region": "vegetated cells", "step": "day"}]}],
    "expectations": [],
    "competing": [{"title": "a nitrogen limit", "text": "nitrogen, not water, sets growth",
                   "fields": ["NITX"], "analysis": "P1"},
                  {"title": "the model's own fire", "text": "fire removes the leaves",
                   "fields": ["BURNX"], "analysis": "P2"}],
    "screen": {"when": "first", "region": "vegetated cells", "year": 2033, "why": "first"},
    "problems": []}
REVIEW = {"passed": True, "follows": True, "why": "it follows", "model_problems": REMARKS}
REGIONS = {"vegetated cells": {"cells": 1100, "area_km2": 830.4, "rule": "a made-up rule"},
           "valley floor": {"cells": 300, "area_km2": 210.0, "rule": "below a made-up line"}}


def _strategy(**over):
    st = {"archetype": "whole_domain", "goals": ["how growth responds"],
          "feasibility": {"verdict": "full", "why": "the change and the fields are there"},
          "requires": [],
          "domain_experiment": {"control": {"name": "valley", "reuse": True},
                                "runs": [{"name": RUN, "changes": [copy.deepcopy(CHANGE)]}],
                                "years": [2031, 2032, 2033],
                                "analysis": {"regions": [{"name": "vegetated cells"}],
                                             "variables": ["LEAFX"], "normal_years": [],
                                             "question_type": "exploratory"},
                                "caveats": list(CAVEATS)},
          "domain_rulebook": {"controls": {"controls": [copy.deepcopy(CONTROL)]},
                              "changes": DC.declare()}}
    st.update(over)
    return st


def _reception(question=QUESTION, **brief):
    b = {"intent": "design", "design_archetype": "whole_domain", "model": "elm",
         "domain": {"name": "valley"},
         "run_settings": {"requested_period": {"yr_start": 2033, "yr_end": 2033},
                          "conflicts": ["the question named no area, so the whole box is used"]},
         "known_data_gaps": ["a thing Reception noted"]}
    b.update(brief)
    return {"user_request": question, "brief": b}


PREVIEW = {"runs": [{"name": RUN}], "refused": [], "error": None, "problems": [],
           "check": {"unusual": [UNUSUAL], "cost": {"node_hours": 2.4}},
           "inputs_check": {"pass": True, "checks": {"active_cells": 1200}}}


@pytest.fixture
def folder(tmp_path):
    """A study folder with the records the page reads: the server's resolved
    words for the change, and an analysis plan with its review."""
    (tmp_path / "01_inputs").mkdir()
    meaning = DC.resolve(CHANGE)["meaning"]
    (tmp_path / dr.RESOLVED_FILE).write_text(json.dumps(
        {"check": {"runs": [{"name": RUN, "changes": [{"id": "weather_scale",
                                                       "meaning": meaning}]}]}}))
    adir = tmp_path / "plan"
    adir.mkdir()
    (adir / "analysis_plan.json").write_text(json.dumps(PLAN))
    (adir / "analysis_plan_review.json").write_text(json.dumps(REVIEW))
    return tmp_path


def _record(folder, strategy=None, **over):
    rec = {"status": AV.PLANNED, "attempts": 1, "regions": REGIONS,
           "design_sha256": dr.design_fingerprint(strategy or _strategy()),
           "plan_file": "plan/analysis_plan.json",
           "review_file": "plan/analysis_plan_review.json"}
    rec.update(over)
    return rec


def _page(folder, strategy=None, reception=None, preview=PREVIEW, record="default"):
    strategy = strategy or _strategy()
    rec = _record(folder, strategy) if record == "default" else record
    return DP.render(reception or _reception(), strategy, run_name="a_study", preview=preview,
                     run_dir=folder, analysis_preview=rec)


def _flat(text):
    return " ".join(text.split())


def _section(page, heading):
    body = page.split(f"## {heading}\n", 1)[1]
    return body.split("\n## ", 1)[0]


# ── the shape ────────────────────────────────────────────────────────────────
class TestTheShape:

    def test_six_parts_in_this_order_and_then_the_decision_line(self, folder):
        page = _page(folder)
        heads = [ln[3:] for ln in page.splitlines() if ln.startswith("## ")]
        assert heads == ["What will run", "The prepared run", "How the answer will be read",
                         "What the study cannot tell you", "Points to weigh", "Your decision"]
        assert page.startswith("# Design review: a_study\n\n**Your question.** " + QUESTION)
        assert page.rstrip().endswith(dr._PENDING)

    def test_it_is_short_and_carries_no_set_up_of_the_control(self, folder):
        page = _page(folder)
        assert DP.words(page) < 700, DP.words(page)
        for gone in ("lightning file", "against observations", "It holds half of the "
                     "measured snow", dr.APPENDIX_MARK, "## The code", "hist_fincl"):
            assert gone not in page, gone

    def test_a_design_of_another_kind_keeps_the_page_it_had(self):
        assert DP.render({"user_request": "q", "brief": {}}, {"archetype": "site"}) is None

    def test_changes_asked_at_the_page_are_shown_after_the_question(self, folder):
        asked = QUESTION + dr.REVISION_MARK + "make it 2034"
        page = _page(folder, reception=_reception(asked))
        assert "**Your question.** " + QUESTION in page
        assert "**Change 1 you asked for at this page.** make it 2034" in page

    def test_three_files_are_written_and_the_page_names_the_other_two(self, folder,
                                                                     monkeypatch):
        monkeypatch.setattr(dr, "_appendix", lambda *a, **k: [dr.APPENDIX_MARK, "", "steps"])
        (folder / "reception.json").write_text(json.dumps(_reception()))
        path = dr.write_review(folder, reception=_reception(), strategy=_strategy(),
                               preview=PREVIEW, analysis_preview=_record(folder))
        page = path.read_text()
        assert path.name == "DESIGN_REVIEW.md" and "## Your decision" in page
        assert "is in `DESIGN_DETAILS.md`." in page
        assert "The steps to make the runs by hand are in `RUN_BY_HAND.md`." in page
        details = (folder / DP.DETAILS_NAME).read_text()
        assert "## The design" in details and "against observations" in details
        assert dr._PENDING not in details and dr.APPENDIX_MARK not in details
        assert (folder / DP.BY_HAND_NAME).read_text().startswith(dr.APPENDIX_MARK)
        # the decision is recorded on the short page
        dr.record_decision(folder, "DECLINED at the terminal")
        assert "**Decision:** DECLINED at the terminal" in path.read_text()

    def test_with_nothing_to_make_by_hand_no_such_file_is_named_or_written(self, folder,
                                                                          monkeypatch):
        monkeypatch.setattr(dr, "_appendix", lambda *a, **k: [])
        path = dr.write_review(folder, reception=_reception(), strategy=_strategy(),
                               preview=None, analysis_preview=None)
        assert "RUN_BY_HAND.md" not in path.read_text()
        assert not (folder / DP.BY_HAND_NAME).exists()


# ── 1. where the run differs from the question's own word ────────────────────
class TestWhereTheRunDiffersFromTheQuestion:

    SAID = ("**Different from what you asked.** The question speaks of rain. This change "
            "multiplies rain and snow together, and the model server has no change of rain "
            "alone.")

    def test_rain_asked_and_rain_and_snow_changed_is_said_where_the_run_is_described(
            self, folder):
        page = _page(folder)
        assert self.SAID in _flat(_section(page, "What will run"))
        assert page.index("Different from what you asked") < page.index("## The prepared run")

    @pytest.mark.parametrize("question", [
        "What if June to August precipitation had been 30 percent lower in 2033?",
        "What if summer rain and snow had been 30 percent lower in 2033?",
        "What if summer rainfall and snowfall had been lower in 2033?"])
    def test_a_question_that_names_snow_or_precipitation_gets_no_such_line(self, folder,
                                                                         question):
        assert "Different from what you asked" not in _page(folder,
                                                           reception=_reception(question))

    def test_a_change_of_temperature_alone_gets_no_such_line(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"][0]["changes"][0]["params"] = {
            "temperature_K": 2.0, "months": [6, 7, 8]}
        assert "Different from what you asked" not in _page(folder, strategy=st)

    def test_the_sentence_is_the_model_servers_and_gives_no_order(self):
        ws = next(c for c in DC.declare()["changes"] if c["id"] == "weather_scale")
        note = ws["asked_otherwise"][0]
        assert note["param"] == "precip_factor" and note["is_not"] == 1.0
        assert "rainfall" in note["question_says"] and "precipitation" in note["and_not"]
        for order in ("Tell the person", "you must", "say so"):
            assert order not in note["say"]

    def test_the_rule_reads_whole_words_and_the_settings_default(self):
        menu = {c["id"]: c for c in DC.declare()["changes"]}
        runs = [{"name": "r", "changes": [{"id": "weather_scale",
                                           "params": {"precip_factor": 0.5}}]}]
        assert DP.differences("less rain in summer", runs, menu)
        assert DP.differences("a drier terrain", runs, menu) == []      # "rain" inside a word
        default = [{"name": "r", "changes": [{"id": "weather_scale",
                                              "params": {"temperature_K": 1.0}}]}]
        assert DP.differences("less rain in summer", default, menu) == []
        assert DP.differences("less rain", [{"name": "r", "changes": [{"id": "fire_off"}]}],
                              menu) == []


# ── 2. and 3. the places, and the years of a series ──────────────────────────
class TestThePlacesAndTheYears:

    def test_each_place_an_analysis_reads_is_given_with_its_cells_and_area(self, folder):
        sec = _section(_page(folder), "How the answer will be read")
        assert "The analyses read these places and no others. The modelled area has 1,200 " \
               "land cells and 900 km2." in _flat(sec)
        assert "| vegetated cells | 1,100 | 830 | a made-up rule |" in sec
        assert "| valley floor | 300 | 210 | below a made-up line |" in sec

    def test_a_place_with_no_size_on_record_says_so(self, folder):
        sec = _section(_page(folder, record=_record(folder, regions={})),
                       "How the answer will be read")
        assert "| vegetated cells | not on record | not on record | not on record |" in sec

    def test_a_daily_series_is_given_with_the_years_it_covers(self, folder):
        sec = _section(_page(folder), "How the answer will be read")
        row = next(ln for ln in sec.splitlines() if ln.startswith("| P2 |"))
        # the run's own years, and the prepared run's, which differ
        assert f"daily series over 2031 to 2033 for `{RUN}` and over 2031 to 2036 for the " \
               f"prepared run" in row

    def test_a_series_of_one_run_gives_one_span(self):
        a = {"outputs": [{"kind": "series", "run": RUN, "step": "month"}]}
        assert AV.series_words(a, {"years": [2031, 2032]}, CONTROL) == \
            "monthly series over 2031 to 2032"
        steps_only = {"steps": ["day"], "outputs": []}
        assert AV.series_words(steps_only, {"years": [2031]}, CONTROL) == \
            "daily series over 2031 to 2036"
        assert AV.series_words({"outputs": []}, {}, CONTROL) == ""

    def test_the_months_of_an_analysis_are_named(self, folder):
        sec = _section(_page(folder), "How the answer will be read")
        row = next(ln for ln in sec.splitlines() if ln.startswith("| P1 |"))
        assert "June to August 2033, by month" in row


# ── 4. the cost of a no ──────────────────────────────────────────────────────
class TestTheWordsOnCost:

    def test_declining_is_not_said_to_cost_nothing(self, folder):
        for page in (_page(folder),
                     _page(folder, strategy=_strategy(
                         domain_experiment=dict(_strategy()["domain_experiment"], runs=[])))):
            low = page.lower()
            for wrong in ("costs nothing", "no cost", "at no cost", "free of"):
                assert wrong not in low, wrong
            assert DP.NO_COMPUTE in _flat(page)

    def test_the_cost_of_the_runs_is_the_servers_estimate(self, folder):
        assert "The model server estimates the cost at about 2.4 node-hours." in \
            _section(_page(folder), "What will run")

    def test_a_design_of_the_control_alone_says_what_a_yes_starts(self, folder):
        st = _strategy(domain_experiment=dict(_strategy()["domain_experiment"], runs=[]))
        page = _page(folder, strategy=st, preview=None, record=None)
        assert "No new run is made. The prepared run's existing output is read for 2033." in \
            _flat(_section(page, "What will run"))
        said = _flat(_section(page, "Your decision"))
        assert "A yes starts the analysis of the prepared run's existing output." in said
        assert "the analysis itself uses paid model calls" in said


# ── 5. titles, with the question as the fallback ─────────────────────────────
class TestTitles:

    def test_an_analysis_is_listed_by_its_title_or_by_its_question(self, folder):
        sec = _section(_page(folder), "How the answer will be read")
        assert "| P1 | Monthly change in plant growth |" in sec
        assert "| P2 | How do the daily courses of growth in the two runs compare? |" in sec
        assert PLAN["analyses"][0]["question"] not in sec        # the title stands for it

    def test_the_details_keep_every_question_under_the_same_id(self, folder):
        rec = _record(folder)
        details = "\n".join(AV.page_lines(rec, folder, rec["design_sha256"]))
        page = _page(folder)
        for a in PLAN["analyses"]:
            assert f"| {a['id']} | {a['question']} |" in details
            assert f"| {a['id']} |" in page

    def test_other_explanations_are_named_by_their_titles_or_left_to_the_details(self, folder):
        sec = _flat(_section(_page(folder), "How the answer will be read"))
        assert "The plan also checks 2 other explanations. They are a nitrogen limit and " \
               "the model's own fire." in sec
        plan = copy.deepcopy(PLAN)
        del plan["competing"][1]["title"]
        (folder / "plan" / "analysis_plan.json").write_text(json.dumps(plan))
        sec = _flat(_section(_page(folder), "How the answer will be read"))
        assert "The plan also checks 2 other explanations, which the details list." in sec

    def test_the_planning_step_is_asked_for_titles_in_the_same_call(self):
        from agents.analysis import analysis_plan as AP
        names = [p.name for p in AP.PLAN_PARTS["analyses"]["item"]]
        assert names[:3] == ["id", "title", "question"]
        assert [p.name for p in AP.PLAN_PARTS["competing"]["item"]][0] == "title"
        task = _flat(AP.plan_task())
        assert "a short title of at most eight words" in task
        # a plan with no title is no worse a plan: the check asks for none
        plan = copy.deepcopy(PLAN)
        assert not [p for p in AP.plan_problems(
            plan, ["LEAFX", "SOILX"], list(REGIONS), [2032, 2033, 2034], "exploratory")
            if "title" in p]


# ── 6. nothing a decision rests on is left out ───────────────────────────────
class TestNothingADecisionRestsOnIsLeftOut:

    def test_every_caveat_is_on_the_page_whole(self, folder):
        sec = _section(_page(folder), "What the study cannot tell you")
        for c in CAVEATS:
            assert f"- {c}" in sec
        long = "A caveat " + "that goes on and on " * 40 + "to its very last word."
        st = _strategy()
        st["domain_experiment"]["caveats"] = CAVEATS + [long]
        assert _flat(long) in _flat(_page(folder, strategy=st))       # never cut

    def test_what_the_prepared_run_does_not_contain_is_said(self, folder):
        assert "This run does not contain the observed floods: the run makes its own." in \
            _flat(_section(_page(folder), "The prepared run"))

    def test_every_finding_of_the_model_server_is_on_the_page_whole(self, folder):
        sec = _flat(_section(_page(folder), "Points to weigh"))
        assert f"- The model server has a note on run {RUN}, change weather_scale. A " \
               f"made-up thing to note." in sec
        assert "- The model server checked the design and built every run without " \
               "submitting anything." in sec

    def test_every_remark_of_the_plans_review_is_on_the_page_whole(self, folder):
        sec = _section(_page(folder), "Points to weigh")
        assert "- The plan's review made 2 remarks." in sec
        for r in REMARKS:
            assert f"  - {r}" in sec

    def test_what_was_decided_without_the_person_is_on_the_page(self, folder):
        sec = _flat(_section(_page(folder), "Points to weigh"))
        assert "- Decided without you: The question named no area, so the whole box is " \
               "used." in sec
        assert "- Noted by Reception: A thing Reception noted." in sec

    def test_a_year_set_against_normal_years_that_contain_it_is_said_twice(self, folder):
        page = _page(folder)
        assert "sets 2033 of the prepared run against the normal years, 2032 to 2034, which " \
               "contain 2033." in _flat(_section(page, "How the answer will be read"))
        assert "- The normal years include 2033, a year this plan sets against them" in \
            _flat(_section(page, "Points to weigh"))

    def test_years_that_are_the_normal_years_themselves_are_said_once_and_plainly(
            self, folder):
        """A study of the normal years: the plain sentence in place of "2032 to
        2034 against the normal years, 2032 to 2034, which contain 2032, 2033
        and 2034". The condition stays under Points to weigh, in the plan's
        own words."""
        plan = copy.deepcopy(PLAN)
        plan["comparisons"] = [{"id": "C1", "what": "each year against the normal years",
                                "run": "control", "against": "normal years",
                                "years": [2032, 2033, 2034]}]
        plan["baseline"]["includes_compared_years"] = [2032, 2033, 2034]
        (folder / "plan" / "analysis_plan.json").write_text(json.dumps(plan))
        page = _page(folder)
        read = _flat(_section(page, "How the answer will be read"))
        assert "The plan sets each year of 2032 to 2034 of the prepared run against the mean " \
               "of those same years." in read
        assert "which contain" not in read
        assert "- The normal years include 2033, a year this plan sets against them" in \
            _flat(_section(page, "Points to weigh"))

    def test_the_servers_sentence_copied_into_the_caveats_is_printed_once(self, folder):
        """The Planner copies the server's sentence on rain and snow into its
        caveats. The page gives it once, where the run is described, and the
        caveat list points there. Every other caveat is printed whole."""
        said = ("The question speaks of rain. This change multiplies rain and snow together, "
                "and the model server has no change of rain alone.")
        st = _strategy()
        st["domain_experiment"]["caveats"] = CAVEATS + [said]
        page = _page(folder, strategy=st)
        assert _flat(page).count(said) == 1
        assert said in _flat(_section(page, "What will run"))
        sec = _section(page, "What the study cannot tell you")
        for c in CAVEATS:
            assert f"- {c}" in sec
        assert '- The difference from your question is given under "What will run".' in sec
        # a caveat in other words is not taken for the same sentence
        st["domain_experiment"]["caveats"] = CAVEATS + ["The run scales snow as well as rain."]
        sec = _section(_page(folder, strategy=st), "What the study cannot tell you")
        assert "- The run scales snow as well as rain." in sec
        assert "is given under" not in sec

    def test_a_run_not_made_yet_is_a_condition(self, folder):
        assert f"- `{RUN}` is not made yet, so the plan takes its outputs to be the prepared " \
               f"run's quantities over the planned years. Each year is checked after it " \
               f"runs." in _flat(_section(_page(folder), "Points to weigh"))

    def test_an_untested_change_is_said_in_the_servers_words(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"] = [{"name": "more_co2", "changes": [
            {"id": "co2_ppmv", "params": {"ppmv": 560.0}}]}]
        sec = _flat(_section(_page(folder, strategy=st, record=None), "What will run"))
        assert "No run with this change has been completed in this framework yet." in sec
        assert "its result is untested." in sec

    def test_a_change_in_effect_in_every_simulated_year_is_said(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"] = [{"name": "more_co2", "changes": [
            {"id": "co2_ppmv", "params": {"ppmv": 560.0}}]}]
        sec = _flat(_section(_page(folder, strategy=st, record=None), "What will run"))
        assert "The change `co2_ppmv` is in effect in every simulated year, 2031 to 2033, " \
               "and not only in the years reported. A reported year carries its effect from " \
               "2031 on." in sec
        # the change that names its years has no such sentence
        assert "in effect in every simulated year" not in _page(folder)

    def test_the_years_simulated_and_the_years_reported_are_both_said(self, folder):
        sec = _flat(_section(_page(folder), "What will run"))
        assert "A run starts from the prepared run's saved state at the beginning of 2031, " \
               "so each new run simulates 2031 to 2033. The numbers are reported for 2033, " \
               "the year you asked about." in sec

    def test_a_verdict_in_part_is_said_with_its_reason_and_what_is_missing(self, folder):
        st = _strategy(feasibility={"verdict": "partial",
                                    "why": "the second half needs a change that is not there"},
                       requires=["a change of the river's flow"])
        sec = _flat(_section(_page(folder, strategy=st), "Points to weigh"))
        assert "- The Planner judges that this design answers the question only in part. " \
               "The second half needs a change that is not there." in sec
        assert "- The question would also need this, which is not available: A change of " \
               "the river's flow." in sec

    def test_a_refused_design_says_why_and_offers_no_yes(self, folder):
        refused = {"refused": [{"where": f"run {RUN}", "what": "a name already taken"}],
                   "runs": [], "error": None}
        page = _page(folder, preview=refused, record=None)
        assert f"  - Run {RUN}: a name already taken." in _section(page, "Points to weigh")
        said = _flat(_section(page, "Your decision"))
        assert said.startswith("This design cannot run as it stands.")
        assert "A yes submits" not in said

    def test_a_design_judged_infeasible_offers_no_yes(self, folder):
        st = _strategy(feasibility={"verdict": "infeasible", "why": "no run can show it"})
        page = _page(folder, strategy=st, record=None)
        assert "- The Planner judges that this design cannot answer the question. No run " \
               "can show it." in _flat(_section(page, "Points to weigh"))
        assert "A yes submits" not in page

    def test_a_plan_that_did_not_pass_shows_every_problem_whole(self, folder):
        plan = dict(copy.deepcopy(PLAN), problems=["P1: no catalog field 'NOPE'",
                                                   "P2: another problem, in full"])
        (folder / "plan" / "analysis_plan.json").write_text(json.dumps(plan))
        sec = _section(_page(folder, record=_record(folder, status=AV.NOT_PASSED)),
                       "Points to weigh")
        assert "- The analysis plan did not pass its check or its review." in sec
        assert "  - P1: no catalog field 'NOPE'" in sec
        assert "  - P2: another problem, in full" in sec

    def test_what_the_details_hold_as_a_condition_the_page_holds_too(self, folder):
        """The page and the details are drawn from the same records. Every
        caveat, server finding, review remark and thing decided without the
        person that the details print is on the page."""
        st, rec = _strategy(), _record(folder)
        page = _flat(_page(folder, strategy=st, record=rec))
        details = _flat(dr.render(_reception(), st, run_name="a_study", preview=PREVIEW,
                                  run_dir=folder, analysis_preview=rec, decision=False,
                                  appendix=False))
        conditions = CAVEATS + REMARKS + [UNUSUAL["what"],
                                          "the question named no area, so the whole box is "
                                          "used", "a thing Reception noted"]
        for c in conditions:
            assert c in details, c
            assert c[0].upper() + c[1:] in page or c in page, c


# ── 7. approval of the analysis is not built yet ─────────────────────────────
class TestApprovalOfTheAnalysisIsNotBuilt:

    def test_the_page_says_so_where_it_shows_a_plan_and_where_it_asks(self, folder):
        page = _page(folder)
        assert AV.NOT_BUILT in _flat(_section(page, "How the answer will be read"))
        assert "It does not approve an analysis plan, since that approval is not built yet." \
            in _flat(_section(page, "Your decision"))

    def test_it_says_so_with_no_plan_on_the_page_as_well(self, folder):
        page = _page(folder, record=None)
        assert "The analysis is planned after the runs, so no analysis plan is on this " \
               "page." in _flat(_section(page, "How the answer will be read"))
        assert "that approval is not built yet" in _flat(_section(page, "Your decision"))

    @pytest.mark.parametrize("words", ["approved plan", "approves this analysis",
                                       "the analyzer carries out this plan", "is bound to"])
    def test_it_never_uses_the_words_of_a_later_stage(self, folder, words):
        assert words not in _flat(_page(folder)).lower()


# ── the prepared run's summary ───────────────────────────────────────────────
class TestThePreparedRunsSummary:

    def test_the_summary_is_printed_as_it_stands_with_its_version(self, folder):
        sec = _section(_page(folder), "The prepared run")
        for t in SUMMARY["text"]:
            assert t in sec
        assert "This summary of `valley` is version 3, written 2031-02-03. The run's full " \
               "record is in `DESIGN_DETAILS.md`." in _flat(sec)

    def test_a_control_with_no_summary_is_named_and_points_to_its_record(self, folder):
        st = _strategy()
        del st["domain_rulebook"]["controls"]["controls"][0]["page_summary"]
        sec = _flat(_section(_page(folder, strategy=st), "The prepared run"))
        assert "The prepared run is `valley`. The model server describes it as the Valley " \
               "control: a made-up box, weather from a made-up record." in sec
        assert "No short summary of it is on record." in sec

    def test_the_server_passes_the_summary_through(self):
        import domain_controls as D
        row = D.summary({"name": "x", "description": "d", "usable": True,
                         "page_summary": SUMMARY}, None)
        assert row["page_summary"] == SUMMARY
        assert D.summary({"name": "x", "description": "d"}, None)["page_summary"] is None

    def test_the_pages_own_words_do_not_call_runoff_river_flow(self):
        """The model makes runoff where it falls and routes none of it. The
        fixed words of the page never call it a river's flow; a control's own
        summary is its registry's to word."""
        for mod in (DP, AV):
            src = Path(mod.__file__).read_text().lower()
            for wrong in ("river flow", "streamflow", "discharge"):
                assert wrong not in src, (mod.__name__, wrong)


# ── the rules for the words, as the agents are given them ────────────────────
class TestTheAgentsAreGivenTheRules:

    def test_the_planner_and_the_planning_step_carry_the_rules_in_full(self):
        from agents.analysis import analysis_plan as AP
        from agents.prompts import load_prompt
        rules = load_prompt("house_style").rstrip()
        assert "HOW TO WRITE THE WORDS A PERSON WILL READ" in rules
        for rule in ("Give each sentence one job.", "Do not put a colon in the middle of a "
                     "sentence", "A limit or a caveat is one complete sentence",
                     "A title is at most eight words."):
            assert rule in " ".join(rules.split()), rule
        assert rules in AP.plan_task()

    def test_the_planner_is_sent_them_with_its_instructions(self):
        pytest.importorskip("openai")
        from agents.planner import Planner
        from agents.prompts import load_prompt
        p = Planner.__new__(Planner)
        seen = {}

        class LLM:
            def ask(self, messages, system_message=None):
                seen["system"] = system_message
                return "{}"
        p.llm, p.system, p.pinning, p.constraints, p.domain_rulebook = LLM(), "SYS", None, \
            None, None
        p.plan({"brief": {"design_archetype": "whole_domain"}})
        assert load_prompt("house_style").rstrip() in seen["system"]
