"""The Planner's duties for the rebuilt design page (2026-11-13).

    pytest tests/test_planner_duties.py -v

The page a person decides on shows a table of comparisons and the limits of
the answer. Two of its parts are written by a model and asked for here: the
Planner's limits for the question, and what each comparison tells the person.
These tests hold the words the two prompts carry and that the plan's check
asks for neither.

The rebuild of the same day is held here too. A whole-domain brief gets a
prompt of its own, code carries the model server's limits into the plan's
caveats, and the comparisons are listed before every page unless a switch
turns that off. Every brief, rulebook and number here is made up.
"""
import copy
import json
import re
from pathlib import Path

import pytest

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
    """The whole-domain prompt, whole and flattened. It moved out of
    planner.txt into a prompt of its own on 2026-11-13."""
    text = (ROOT / "src" / "agents" / "prompts" / "planner_domain.txt").read_text()
    return text, _flat(text)


class TestThePlannerNamesTheLimitsForTheQuestion:
    def test_it_is_asked_for_them_from_the_controls_known_errors(self):
        text, wd = _planner()
        assert "THE LIMITS FOR THIS QUESTION are yours to judge." in wd
        assert "name in domain_experiment.question_limits at most two limits" in wd
        assert "Read the control run's `known_errors_words`" in wd
        assert "`evaluation`" not in wd          # measured numbers stay away from it
        assert '"question_limits": [' in text

    def test_each_says_the_error_in_words_and_its_effect_as_a_judgement(self):
        _, wd = _planner()
        assert "in the record's words and without numbers" in wd
        assert 'with "could" or "may"' in wd
        assert "Where the control run lists no known error, or none bears on the question, " \
               "the list is empty" in wd

    def test_they_repeat_nothing_the_page_prints_by_itself_and_reach_the_report(self):
        _, wd = _planner()
        assert "Do not repeat a change's own limit or what the run does not contain" in wd
        # code copies them into the caveats now (planner.with_carried_caveats)
        assert "Code copies each limit into the caveats, so do not copy it there " \
               "yourself" in wd
        assert "Copy each limit into caveats too" not in wd

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


# ─────────────────────────────────────────────────────────────────────────────
# THE PLANNER'S REBUILD (2026-11-13)
# ─────────────────────────────────────────────────────────────────────────────
PROMPTS = ROOT / "src" / "agents" / "prompts"
DOMAIN_TEXT = (PROMPTS / "planner_domain.txt").read_text()
OTHER_TEXT = (PROMPTS / "planner.txt").read_text()
UNTESTED_WORDS = ("No run with this change has been completed in this framework yet. Its "
                  "settings are checked before the model is built; its result is untested")
RULEBOOK = {
    "controls": {"controls": [
        {"name": "valley", "known_errors_words": ["It melts its snow too early"],
         "not_in_it_words": ["the made-up floods: the run's floods are its own",
                             "  the made-up dams. "],
         "evaluation": {"summary": "its snow is 41 percent of the measured"},
         "facts": {"output_years": [2031, 2039],
                   "per_year": {"2032": {"gpp_gC_m2_yr": 812.5}}}},
        {"name": "bare", "not_in_it_words": []}]},
    "changes": {"demonstrated_false": UNTESTED_WORDS, "changes": [
        {"id": "dim_sun", "available": True, "demonstrated": True,
         "not_meaning": "a consistent change of the made-up sky: only the light is dimmer"},
        {"id": "dep_year", "available": True, "demonstrated": False,
         "params": {"year": {"kind": "integer"}},
         "not_meaning": "deposition beyond what the made-up dataset holds for {year}"},
        {"id": "lift_gas", "available": True, "demonstrated": False,
         "params": {"ppm": {"kind": "number", "default": 640}},
         "not_meaning": "a made-up gas rise of {ppm} ppm in the weather."},
        {"id": "no_tool", "available": False, "requires": ["a burn tool"],
         "not_meaning": "a burn that never happened"}]}}
NOT_SUN = ("The difference from the control run is not a consistent change of the made-up "
           "sky: only the light is dimmer.")
NOT_DEP = ("The difference from the control run is not deposition beyond what the made-up "
           "dataset holds for 2033.")
NOT_GAS = "The difference from the control run is not a made-up gas rise of 640 ppm in the weather."
UNTESTED_DEP = ("No run with the change dep_year has been completed in this framework yet, so "
                "its result is untested.")
UNTESTED_GAS = ("No run with the change lift_gas has been completed in this framework yet, so "
                "its result is untested.")
FLOODS = "The control run does not contain the made-up floods: the run's floods are its own."
DAMS = "The control run does not contain the made-up dams."


def _design(control="valley", runs=None):
    if runs is None:
        runs = [{"name": "dim", "changes": [{"id": "dep_year", "params": {"year": 2033.0}},
                                            {"id": "dim_sun"}]},
                {"name": "dim_gas", "changes": [{"id": "dim_sun"}, {"id": "lift_gas"}]}]
    return {"control": {"name": control, "reuse": True}, "runs": runs, "years": [2032, 2033]}


class _Model:
    """A scripted model client that keeps what it was sent."""

    def __init__(self, reply):
        self.reply, self.sent = reply, []

    def ask(self, messages, system_message=None):
        self.sent.append({"system": system_message, "content": messages[0]["content"]})
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


@pytest.fixture
def scripted(monkeypatch):
    """A Planner made through its own __init__, its model client scripted."""
    from agents import planner as PL

    def make(reply="{}", rulebook=None, constraints=None, pinning=None):
        model = _Model(reply)
        monkeypatch.setattr(PL, "SimpleLLMClient", lambda model_name=None, **k: model)
        p = PL.Planner(constraints=constraints, pinning=pinning)
        p.domain_rulebook = rulebook
        return p, model
    return make


WHOLE = {"brief": {"design_archetype": "whole_domain", "user_request": "a made-up question"}}
SITE = {"brief": {"design_archetype": "site", "user_request": "a made-up question"}}


class TestEachBriefGetsItsPrompt:

    @pytest.mark.parametrize("arche, name", [
        ("whole_domain", "planner_domain"), (" Whole_Domain ", "planner_domain"),
        ("site", "planner"), ("conceptual", "planner"), ("coupling", "planner"),
        (None, "planner"), ("", "planner")])
    def test_the_archetype_picks_the_prompt(self, arche, name):
        from agents import planner as PL
        assert PL.prompt_for({"design_archetype": arche}) == name
        assert PL.prompt_for(None) == PL.prompt_for({}) == "planner"

    def test_a_whole_domain_brief_is_sent_its_own_prompt_and_no_column_block(self, scripted):
        p, model = scripted(constraints={"fixed": ["a made-up column limit"]},
                            pinning={"pinnable": ["a made-up station kind"]},
                            rulebook=RULEBOOK)
        p.plan(WHOLE)
        system = model.sent[0]["system"]
        assert system.startswith(DOMAIN_TEXT)
        assert "WHOLE-DOMAIN RULEBOOK: the control runs and the change list" in system
        # the two column blocks describe single columns, not this study
        assert "DESIGN LIMITS" not in system and "a made-up column limit" not in system
        assert "PINNING RULES" not in system and "a made-up station kind" not in system
        assert "HOW TO WRITE THE WORDS A PERSON WILL READ" in system
        assert "prepared" not in system.split("WHOLE-DOMAIN RULEBOOK")[1].split("\n")[0]

    def test_any_other_brief_keeps_the_old_prompt_and_its_blocks(self, scripted):
        p, model = scripted(constraints={"fixed": ["a made-up column limit"]},
                            pinning={"pinnable": ["a made-up station kind"]})
        for reception in (SITE, {"brief": {}}):
            p.plan(reception)
            system = model.sent[-1]["system"]
            assert system.startswith(OTHER_TEXT)
            assert "DESIGN LIMITS" in system and "PINNING RULES" in system
            assert DOMAIN_TEXT[:200] not in system

    def test_a_planner_made_without_init_reads_the_prompt_itself(self):
        from agents.planner import Planner
        p = object.__new__(Planner)
        p.system, p.constraints, p.pinning, p.domain_rulebook = "SYS", None, None, None
        system, content = p.messages(WHOLE)
        assert system.startswith(DOMAIN_TEXT)
        assert p.messages(SITE)[0].startswith("SYS")
        assert content.endswith("Emit ONLY the JSON.")
        from agents import planner as PL
        assert (PROMPTS / f"{PL.DOMAIN_PROMPT}.txt").read_text() == DOMAIN_TEXT
        assert (PROMPTS / f"{PL.PROMPT}.txt").read_text() == OTHER_TEXT

    def test_measured_numbers_stay_out_of_the_whole_domain_prompt(self, scripted):
        p, model = scripted(rulebook=RULEBOOK)
        p.plan(WHOLE)
        system = model.sent[0]["system"]
        for gone in ("41 percent", "812.5", "gpp_gC_m2_yr", '"evaluation"'):
            assert gone not in system, gone
        assert "It melts its snow too early" in system


class TestTheWholeDomainPrompt:

    def test_it_stays_under_170_lines(self):
        assert len(DOMAIN_TEXT.splitlines()) < 170

    def test_its_instructions_are_in_one_place_only(self):
        assert "THE WHOLE-DOMAIN PATH" not in OTHER_TEXT
        assert "domain_experiment" not in OTHER_TEXT
        for moved in ("known_errors_words", "question_limits", "normal_years",
                      "FAREA_BURNED", "question_type", "vegetated cells"):
            assert moved not in OTHER_TEXT, moved
        pointer = [x for x in OTHER_TEXT.splitlines() + [""] if "planner_domain.txt" in x]
        assert len(pointer) == 1
        assert 'A "whole_domain" study is planned with planner_domain.txt' in _flat(OTHER_TEXT)
        assert '"archetype": "site" | "conceptual" | "coupling",' in OTHER_TEXT

    @pytest.mark.parametrize("phrase", [
        # the experiment is copied
        "Copy the `control`, `runs` and `years` of brief.domain_experiment into "
        "domain_experiment word for word.",
        "Nothing is sampled, so there are no bands, no pinned stations and no column budget.",
        # regions
        "Name regions the control run has (controls[].regions), or the standard "
        "\"vegetated cells\" and \"elevation bands\"",
        "makes them inside that region only.",
        "When a run's change applies in one region, report only that region or bands "
        "within it.",
        "A region that reaches outside it averages changed and unchanged cells.",
        "Always name the regions.",
        # variables
        "control run holds (facts.requested_fields). Each must be a single-level field.",
        "A field with levels, such as soil layers, has no one number per cell",
        "burned_area_km2 is not a history field.",
        "name FAREA_BURNED among the variables and the burned area in km2 comes with it.",
        # normal years
        "inside the control run's output years.",
        "use exactly those, less any year the question asks about.",
        "Choose other years only when the person named them.",
        "For any other question, give the empty list.",
        # years
        "Leave this key out.",
        "A year a run simulates is not reported unless the question asks about it.",
        "each inside domain_experiment.years, and then say why in a caveat.",
        # the question type
        "\"hypothesis\" only when the question proposes a testable expectation",
        "Naming a field alone is not a hypothesis",
        "Decide it from the question as asked, never after seeing data.",
        # the verdicts
        "full Every change a run needs is available.",
        "partial Part of the question needs a change whose `available` is false, or a field "
        "the control run lacks.",
        "and cannot be limited to them, because it has no `years` parameter and the run "
        "starts earlier.",
        "infeasible No run on this control run can answer the question.",
        "An infeasible design is stopped before any job.",
        # the answer is the model's
        "The answer is read from the model's output alone",
        "No caveat carries an observation's numbers.",
    ])
    def test_it_states_each_rule(self, phrase):
        assert phrase in _flat(DOMAIN_TEXT)

    def test_it_tells_the_planner_not_to_copy_the_servers_limits(self):
        wd = _flat(DOMAIN_TEXT)
        assert "The list is usually empty." in wd
        assert "Code carries the model server's limits to the page and the report in its " \
               "own words." in wd
        assert "Never copy them into your caveats or restate them in other words." in wd
        assert "A copy can lose the one word that turns a limit around." in wd
        # nothing left of the old order to copy them
        assert "copy each change's `not_meaning`" not in wd.lower()
        assert "written from the control's `not_in_it_words`" not in wd

    def test_the_reason_is_one_plain_sentence_with_a_shape_to_fill_and_no_example(self):
        wd = _flat(DOMAIN_TEXT)
        assert "feasibility.why is one sentence that the page prints directly under the " \
               "question." in wd
        assert "It names no tool, no change by its identifier, no field by its code name " \
               "and no part of the framework, because the person knows none of them." in wd
        assert "Its only numbers are the years and the months of the design." in wd
        # TWO SHAPES TO FILL IN, AND NO EXAMPLE (2026-11-13). An example about a
        # change the server's list holds is copied when a question is about
        # that change, so the prompt gives the sentence's shape with places to
        # fill, one for a new run and one for the control run alone.
        shapes = re.findall(r'"((?:A second run with|The control run holds) [^"]+)"', wd)
        assert len(shapes) == 2 and all("<" in x and ">" in x for x in shapes), shapes
        assert "This is the kind of sentence meant" not in wd
        for named in ("fire", "rain", "snow", "weather", "temperature", "CO2", "nitrogen"):
            assert all(named not in x for x in shapes), named

    def test_the_goals_are_three_plain_steps_at_most(self):
        wd = _flat(DOMAIN_TEXT)
        assert "goals are at most three steps toward the answer. Each is one plain sentence " \
               "that starts with a verb and names quantities in words, never by code names." \
               in wd

    @pytest.mark.parametrize("key", [
        '"archetype": "whole_domain"', '"goals": [', '"feasibility": {"verdict": "full" | '
        '"partial" | "infeasible"', '"why": "<one plain sentence',
        '"sampling": {"approach": "whole_domain", "n_columns": 1}', '"domain_experiment": {',
        '"control": {"name": "<brief.domain_experiment.control.name>", "reuse": true}',
        '"runs": [<copied from brief.domain_experiment.runs>]',
        '"years": [<copied from brief.domain_experiment.years>]', '"analysis": {"regions": [',
        '"variables": [', '"normal_years": [<int>, ...]',
        '"question_type": "hypothesis" | "exploratory"', '"question_limits": [', '"caveats": [',
        '"validation": []', '"requires": [', '"coupling": null'])
    def test_the_json_keeps_every_key_its_readers_read(self, key):
        assert key in DOMAIN_TEXT

    def test_the_words_follow_the_rules_for_a_person(self):
        wd = _flat(DOMAIN_TEXT)
        assert "\u2014" not in DOMAIN_TEXT and "\u2013" not in DOMAIN_TEXT
        assert "prepared" not in wd and "Analyzer" not in wd
        assert not re.search(r"\bthe control\b(?! run)", wd, re.IGNORECASE)


class TestTheCarriedCaveats:

    def test_the_words_and_their_order(self):
        from agents import planner as PL
        assert PL.carried_caveats(RULEBOOK, _design()) == [
            NOT_DEP, NOT_SUN, NOT_GAS, UNTESTED_DEP, UNTESTED_GAS, FLOODS, DAMS]

    def test_the_first_caveat_keeps_its_not(self):
        from agents import planner as PL
        first = PL.carried_caveats(RULEBOOK, _design(runs=[
            {"name": "dim", "changes": [{"id": "dim_sun"}]}]))[0]
        assert first.startswith("The difference from the control run is not a consistent "
                                "change of the made-up sky")

    def test_a_change_used_twice_is_said_once(self):
        from agents import planner as PL
        twice = _design(runs=[{"name": "a", "changes": [{"id": "lift_gas"}]},
                              {"name": "b", "changes": [{"id": "lift_gas"}]}])
        got = PL.carried_caveats(RULEBOOK, twice)
        assert got == [NOT_GAS, UNTESTED_GAS, FLOODS, DAMS]
        assert len(got) == len(set(got))

    def test_the_same_change_with_other_settings_says_each_thing(self):
        from agents import planner as PL
        got = PL.carried_caveats(RULEBOOK, _design(runs=[
            {"name": "a", "changes": [{"id": "dep_year", "params": {"year": 2033}}]},
            {"name": "b", "changes": [{"id": "dep_year", "params": {"year": 2035}}]}]))
        assert got[:3] == [NOT_DEP, NOT_DEP.replace("2033", "2035"), UNTESTED_DEP]

    def test_only_a_change_marked_not_demonstrated_is_called_untested(self):
        from agents import planner as PL
        got = PL.carried_caveats(RULEBOOK, _design(runs=[
            {"name": "a", "changes": [{"id": "dim_sun"}, {"id": "no_tool"}]}]))
        assert got == [NOT_SUN, "The difference from the control run is not a burn that "
                                "never happened.", FLOODS, DAMS]

    def test_a_control_run_with_nothing_missing(self):
        from agents import planner as PL
        assert PL.carried_caveats(RULEBOOK, _design(control="bare", runs=[
            {"name": "a", "changes": [{"id": "dim_sun"}]}])) == [NOT_SUN]
        # a control run the rulebook does not hold, and the control run alone
        assert PL.carried_caveats(RULEBOOK, _design(control="elsewhere", runs=[])) == []
        assert PL.carried_caveats(RULEBOOK, _design(runs=[])) == [FLOODS, DAMS]

    def test_a_placeholder_with_no_value_stays_as_written(self):
        from agents import planner as PL
        got = PL.carried_caveats(RULEBOOK, _design(runs=[
            {"name": "a", "changes": [{"id": "dep_year"}]}]))
        assert got[0] == ("The difference from the control run is not deposition beyond what "
                          "the made-up dataset holds for {year}.")

    def test_the_servers_words_are_put_on_one_line(self):
        """A line break or a doubled space in the server's words would break
        the caveat across lines of a page, or leave a gap in it."""
        from agents import planner as PL
        assert PL._clause("a made-up\nsky:  only\tthe light\r\n is dimmer. ") == \
            "a made-up sky: only the light is dimmer"
        rb = copy.deepcopy(RULEBOOK)
        rb["changes"]["changes"][0]["not_meaning"] = "a consistent change\nof the made-up  sky"
        rb["controls"]["controls"][0]["not_in_it_words"] = ["the made-up\n  floods",
                                                             " the made-up   dams. "]
        got = PL.carried_caveats(rb, _design(runs=[{"name": "dim",
                                                     "changes": [{"id": "dim_sun"}]}]))
        assert got == ["The difference from the control run is not a consistent change of the "
                       "made-up sky.", "The control run does not contain the made-up floods.",
                       DAMS]
        assert all("\n" not in c and "  " not in c for c in got)

    def test_no_rule_of_the_servers_list_asks_the_designer_for_a_caveat(self):
        """Code writes the caveat for a change no run has completed, so the
        list's rules no longer ask the designer to write it; a rule that did
        would have the Planner copy the server's words again."""
        import domain_changes as DC
        rules = DC.declare()["rules"]
        assert rules and not [r for r in rules if "caveat" in r.lower()]
        # the rule that tells the person stays
        assert any("demonstrated: false" in r and "untested" in r for r in rules)

    @pytest.mark.parametrize("rulebook, design", [
        (None, None), ({}, {}), (None, _design()), ({"changes": [], "controls": None}, _design()),
        (RULEBOOK, {"control": "valley", "runs": "dim"}),
        (RULEBOOK, {"runs": [None, {"changes": [None, {"id": 3}, {"no": "id"}]}]})])
    def test_a_record_of_another_form_gives_fewer_caveats_and_no_error(self, rulebook, design):
        from agents import planner as PL
        assert PL.carried_caveats(rulebook, design) == []


class TestThePlansCaveatsArePutTogetherByCode:

    def _planned(self):
        de = _design(runs=[{"name": "dim", "changes": [{"id": "dim_sun"}, {"id": "lift_gas"}]}])
        de["question_limits"] = ["The made-up error could matter.", {"text": "A second limit."}]
        de["caveats"] = [
            "a consistent change of the made-up sky: only the light is dimmer",   # the source
            NOT_GAS,                                                               # a carried one
            "  the made-up floods: the run's floods are its own. ",               # a source
            UNTESTED_WORDS,                                                        # a source
            "the made-up error could matter",                                     # a limit
            "Its own limit.", "its own limit", "", None]
        return de

    def test_the_order_the_planners_own_and_who_put_them_together(self):
        from agents import planner as PL
        de = self._planned()
        got = PL.with_carried_caveats(de, RULEBOOK)
        assert got["planner_caveats"] == ["Its own limit."]
        assert got["caveats"] == [NOT_SUN, NOT_GAS, UNTESTED_GAS, FLOODS, DAMS,
                                  "The made-up error could matter.", "A second limit.",
                                  "Its own limit."]
        assert got["caveats_by"] == "code" == PL.CAVEATS_BY
        assert got["question_limits"] == de["question_limits"]        # as the Planner wrote
        assert de["caveats"][5] == "Its own limit." and "caveats_by" not in de   # not changed

    def test_put_together_twice_is_put_together_once(self):
        from agents import planner as PL
        once = PL.with_carried_caveats(self._planned(), RULEBOOK)
        assert PL.with_carried_caveats(once, RULEBOOK) == once

    def test_a_plan_with_no_caveats_and_no_limits(self):
        from agents import planner as PL
        got = PL.with_carried_caveats(_design(runs=[]), RULEBOOK)
        assert got["caveats"] == [FLOODS, DAMS] and got["planner_caveats"] == []
        assert "question_limits" not in got
        # a control run the rulebook holds, with nothing missing and no run
        got = PL.with_carried_caveats(dict(_design(control="bare", runs=[]),
                                           caveats=["One of its own."]), RULEBOOK)
        assert got["caveats"] == got["planner_caveats"] == ["One of its own."]
        assert got["caveats_by"] == "code"

    @pytest.mark.parametrize("rulebook", [
        None, {}, "not a rulebook",
        # one that holds neither the design's control run nor a change its runs use
        {"controls": {"controls": [{"name": "elsewhere", "not_in_it_words": ["x"]}]},
         "changes": {"changes": [{"id": "other_knob", "not_meaning": "y"}]}}])
    def test_no_rulebook_for_the_design_leaves_it_as_the_planner_wrote_it(self, rulebook):
        """Code then carries none of the server's words. Marked as put
        together by code, the design would print no limit of the server at
        all, so a page prints the Planner's caveats as for an old record."""
        from agents import planner as PL
        de = dict(_design(runs=[{"name": "dim", "changes": [{"id": "dim_sun"}]}]),
                  caveats=["One of its own.", "A copied limit of the server."],
                  question_limits=["A made-up limit."])
        got = PL.with_carried_caveats(de, rulebook)
        assert got == de and got is not de
        assert "caveats_by" not in got and "planner_caveats" not in got
        # one put together before, with a rulebook, goes back to the Planner's own
        once = PL.with_carried_caveats(de, RULEBOOK)
        assert once["caveats_by"] == "code"
        again = PL.with_carried_caveats(once, rulebook)
        assert "caveats_by" not in again and "planner_caveats" not in again
        assert again["caveats"] == once["planner_caveats"]

    @pytest.mark.parametrize("design", [
        _design(control="valley", runs=[]),                         # the control run alone
        _design(control="elsewhere", runs=[{"name": "dim", "changes": [{"id": "dim_sun"}]}])])
    def test_a_rulebook_that_holds_the_control_run_or_a_change_carries(self, design):
        from agents import planner as PL
        got = PL.with_carried_caveats(design, RULEBOOK)
        assert got["caveats_by"] == "code" and got["caveats"]

    def test_plan_puts_them_together_for_a_whole_domain_plan_only(self, scripted):
        reply = {"archetype": "whole_domain", "domain_experiment": dict(
            _design(runs=[{"name": "dim", "changes": [{"id": "dim_sun"}]}]),
            caveats=["Its own limit."], question_limits=["A made-up limit."])}
        p, _ = scripted(reply=reply, rulebook=RULEBOOK)
        de = p.plan(WHOLE)["domain_experiment"]
        assert de["caveats"] == [NOT_SUN, FLOODS, DAMS, "A made-up limit.", "Its own limit."]
        assert de["planner_caveats"] == ["Its own limit."] and de["caveats_by"] == "code"
        p, _ = scripted(reply={"archetype": "site", "sampling": {"n_columns": 9}})
        assert p.plan(SITE) == {"archetype": "site", "sampling": {"n_columns": 9}}
        p, _ = scripted(reply="no JSON here")
        assert p.plan(WHOLE) == {}


class TestTheRoundAfterARefusal:

    def test_the_planner_is_shown_its_own_caveats_and_not_the_carried_ones(self, scripted):
        reply = {"archetype": "whole_domain", "domain_experiment": dict(
            _design(runs=[{"name": "dim", "changes": [{"id": "dim_sun"}]}]),
            caveats=["Its own limit."], question_limits=["A made-up limit."])}
        p, model = scripted(reply=reply, rulebook=RULEBOOK)
        first = p.plan(WHOLE)
        second = p.plan(WHOLE, findings=[{"where": "run dim", "what": "a made-up finding"}],
                        previous=dict(first, pinning={"x": 1}, domain_rulebook=RULEBOOK))
        content = model.sent[-1]["content"]
        assert p.FINDINGS_NOTE in content and "a made-up finding" in content
        shown = json.loads(content.split("THE DESIGN IT REFUSED:\n")[1]
                           .split("\n\nTHE SERVER'S FINDINGS:")[0])
        assert shown["domain_experiment"]["caveats"] == ["Its own limit."]
        assert shown["domain_experiment"]["question_limits"] == ["A made-up limit."]
        assert "caveats_by" not in shown["domain_experiment"]
        assert "planner_caveats" not in shown["domain_experiment"]
        assert "pinning" not in shown and "domain_rulebook" not in shown
        for carried in (NOT_SUN, FLOODS, DAMS):
            assert carried not in content
        # the changed design has its caveats put together again
        assert second["domain_experiment"]["caveats"] == first["domain_experiment"]["caveats"]
        # what the Planner was handed back is not changed by showing it
        assert first["domain_experiment"]["caveats_by"] == "code"

    def test_a_design_recorded_before_code_carried_them_is_shown_as_it_was(self):
        from agents import planner as PL
        old = dict(_design(), caveats=["an old caveat the Planner copied"])
        assert PL.as_the_planner_wrote_it(old) is old
        assert PL.as_the_planner_wrote_it(None) is None
        assert PL.as_the_planner_wrote_it(dict(old, caveats_by="code",
                                               planner_caveats=None))["caveats"] == []


class TestTheComparisonsAreListedBeforeEveryPage:
    """Both switches are on unless set to 0, false, no or off (interface I5)."""

    @pytest.mark.parametrize("module", ["analysis_preview", "analysis_handoff"])
    @pytest.mark.parametrize("value, on", [
        (None, True), ("", True), ("1", True), ("yes", True), ("on", True), ("true", True),
        ("anything", True), ("0", False), ("false", False), ("no", False), ("off", False),
        (" OFF ", False), ("False", False), ("No", False)])
    def test_each_switch(self, monkeypatch, module, value, on):
        import importlib
        mod = importlib.import_module(f"agents.{module}")
        monkeypatch.delenv(mod.SWITCH, raising=False)
        if value is not None:
            monkeypatch.setenv(mod.SWITCH, value)
        assert mod.enabled() is on

    def test_the_switch_names_stay(self):
        from agents import analysis_handoff as AH
        from agents import analysis_preview as AV
        assert AV.SWITCH == "IDEAS_PLAN_BEFORE_APPROVAL"
        assert AH.SWITCH == "IDEAS_CARRY_OUT_SHOWN_PLAN"

    @pytest.mark.parametrize("handoff, said", [
        (None, "If they pass their check, your yes covers them and they are carried out as "
               "listed"),
        ("off", "They are for you to read only. Your yes does not cover them")])
    def test_the_terminal_says_they_are_the_planners(self, monkeypatch, capsys, tmp_path,
                                                     handoff, said):
        import workflow as W
        from agents import analysis_handoff as AH
        from agents import analysis_preview as AV
        monkeypatch.delenv(AV.SWITCH, raising=False)
        monkeypatch.delenv(AH.SWITCH, raising=False)
        if handoff:
            monkeypatch.setenv(AH.SWITCH, handoff)
        monkeypatch.setattr(AV, "plan_before_approval",
                            lambda *a, **k: {"status": "planned", "elapsed_s": 0.1})
        co = W.WorkflowCoordinator.__new__(W.WorkflowCoordinator)
        rec = co._analysis_preview({}, {"domain_experiment": {}}, tmp_path, blocked=False)
        out = capsys.readouterr().out
        assert rec["status"] == "planned"
        assert "The Planner lists the comparisons before the page." in out and said in out
        assert "the comparisons: planned" in out
        assert "Analyzer" not in out and "preview" not in out and "analysis plan" not in out

    def test_with_the_switch_off_nothing_is_listed(self, monkeypatch, tmp_path):
        import workflow as W
        from agents import analysis_preview as AV
        monkeypatch.setenv(AV.SWITCH, "off")
        monkeypatch.setattr(AV, "plan_before_approval",
                            lambda *a, **k: pytest.fail("planned with the switch off"))
        co = W.WorkflowCoordinator.__new__(W.WorkflowCoordinator)
        assert co._analysis_preview({}, {"domain_experiment": {}}, tmp_path, False) is None
