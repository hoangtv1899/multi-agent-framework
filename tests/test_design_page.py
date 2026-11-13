"""The short design page: what a person decides on, and nothing a decision
rests on left out (2026-11-12, rebuilt section by section on 2026-11-13).

    pytest tests/test_design_page.py -v

The page was about 1,700 words for one new run and read like a form. The
hydrologist who reads it asked on 2026-11-12 for more figures, less text and
nothing taken too literally. The page is now, in this order:

  1. Your question
  2. The control run
  3. The new run: a table of the knobs and a figure of what the run is given
  4. How your question will be answered: a table of the comparisons
  5. What the answer will not cover
  6. Choices made for you
  7. Your decision: what each answer at the terminal does

SHORT MUST NOT MEAN SILENT. These tests hold the page to that:

  - a difference between the question's own word and what the run does
    (rain asked, rain and snow changed) is on the page;
  - a series is given with the years it covers;
  - declining is not said to cost nothing;
  - a comparison is listed by its short title, or by its fields where the
    plan gave none;
  - every limit of a knob, every limit of the Planner's, every finding of the
    model server's check and every thing decided without the person is on
    the page whole, and so are the conditions code states: an untested
    change, a change in effect in every simulated year, a year set against
    normal years that contain it;
  - what the short page leaves to DESIGN_DETAILS.md is there: every caveat,
    every finding of the server, every remark of the plan's review, the
    whole plan, a run not made yet;
  - where a plan is shown and the yes does not approve it, the page says so.

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
           "text": ["The control run is a model run of a made-up valley for 2031 to 2036.",
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
# the server's sentence where a run's change is wider than the question's word
SAID = ("The question speaks of rain. This change multiplies rain and snow together, and "
        "the model server has no change of rain alone.")
HEADS = ["The control run", "The new run", "How your question will be answered",
         "What the answer will not cover", "Choices made for you", "Your decision"]
FRAMEWORK_WORDS = ("analysis plan", "analyzer", "planning step", "coordinator",
                   "prepared run", "\u2014")


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


def _resolved(folder, runs, not_covered=None):
    """The server's check on disk: each change of each run with its meaning
    and, given `not_covered` ({change id: [sentences]}), its short limits."""
    out = []
    for r in runs:
        changes = []
        for c in r.get("changes") or []:
            rc = {"id": c["id"], "meaning": DC.resolve(c)["meaning"]}
            if not_covered is not None and c["id"] in not_covered:
                rc["not_covered"] = list(not_covered[c["id"]])
            changes.append(rc)
        out.append({"name": r["name"], "changes": changes})
    (folder / dr.RESOLVED_FILE).write_text(json.dumps({"check": {"runs": out}}))


@pytest.fixture
def folder(tmp_path):
    """A study folder with the records the page reads: the server's resolved
    words for the change, and an analysis plan with its review."""
    (tmp_path / "01_inputs").mkdir()
    _resolved(tmp_path, [{"name": RUN, "changes": [CHANGE]}])
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


def _page(folder, strategy=None, reception=None, preview=PREVIEW, record="default",
          given=None):
    strategy = strategy or _strategy()
    rec = _record(folder, strategy) if record == "default" else record
    return DP.render(reception or _reception(), strategy, run_name="a_study", preview=preview,
                     run_dir=folder, analysis_preview=rec, run_given=given)


def test_the_cost_and_the_limit_are_one_number_printed_one_way(folder):
    # the server's estimate with more figures than a person reads
    preview = copy.deepcopy(PREVIEW)
    preview["check"]["cost"]["node_hours"] = 1.58333333
    page = _page(folder, preview=preview)
    assert "estimates the cost at about 1.6 node-hours." in page
    assert "may use up to 1.6 node-hours." in page
    assert "1.58" not in page
    assert DP.hours_words(2) == "2" and DP.hours_words(0.75) == "0.8"


def _flat(text):
    return " ".join(text.split())


def _section(page, heading):
    body = page.split(f"## {heading}\n", 1)[1]
    return body.split("\n## ", 1)[0]


def _details(folder, strategy=None, preview=PREVIEW, record="default", reception=None):
    strategy = strategy or _strategy()
    rec = _record(folder, strategy) if record == "default" else record
    return dr.render(reception or _reception(), strategy, run_name="a_study", preview=preview,
                     run_dir=folder, analysis_preview=rec, decision=False, appendix=False)


def _plan(folder, **over):
    """The plan on disk, changed."""
    plan = copy.deepcopy(PLAN)
    plan.update(over)
    (folder / "plan" / "analysis_plan.json").write_text(json.dumps(plan))
    return plan


# what the new run is given (agents/run_figures), for the design above
MONTHS = [f"{y}-{m:02d}" for y in (2031, 2032, 2033) for m in range(1, 13)]
RAIN_ROW = {"knob": "Rain and snow", "change": "multiplied by 0.7", "when": "June to August "
            "of 2033", "where": "everywhere"}


def _given(*runs, drawn=None, **over):
    runs = runs or ({"name": RUN, "rows": [dict(RAIN_ROW)],
                     "panels": [{"title": "Rain and snow over the whole grid",
                                 "units": "mm/month", "step": "month", "labels": MONTHS,
                                 "control": [50.0] * 36, "run": [50.0] * 36,
                                 "changed": ["2033-06", "2033-07", "2033-08"]}],
                     "map": None, "problems": []},)
    rec = {"control": "valley", "years": [2031, 2032, 2033], "problems": [],
           "runs": list(runs)}
    rec.update(over)
    if drawn is None:
        drawn = {"runs": {r["name"]: {"series": f"design_figures/new_run_{r['name']}_series.png"
                                      if r.get("panels") else None, "map": None}
                          for r in runs}, "problems": []}
    return {"record": rec, "drawn": drawn}


# ── the shape ────────────────────────────────────────────────────────────────
class TestTheShape:

    def test_seven_parts_in_this_order_and_then_the_decision_line(self, folder):
        page = _page(folder)
        heads = [ln[3:] for ln in page.splitlines() if ln.startswith("## ")]
        assert heads == HEADS
        assert page.startswith("# Design review: a_study\n\n**Your question.** " + QUESTION)
        assert page.rstrip().endswith("---\n**Decision:** not yet made.")
        assert dr._PENDING == "**Decision:** not yet made."

    def test_the_order_holds_for_several_runs_no_run_and_a_design_that_cannot_run(
            self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"].append({"name": "no_fire", "changes": [{"id": "fire_off"}]})
        several = [ln[3:] for ln in _page(folder, strategy=st, record=None).splitlines()
                   if ln.startswith("## ")]
        assert several == [HEADS[0], "The new runs"] + HEADS[2:]
        none = _strategy()
        none["domain_experiment"]["runs"] = []
        refused = {"refused": [{"where": f"run {RUN}", "what": "a made-up refusal"}],
                   "runs": [], "error": None}
        for page in (_page(folder, strategy=none, preview=None, record=None),
                     _page(folder, preview=refused, record=None)):
            assert [ln[3:] for ln in page.splitlines() if ln.startswith("## ")] == HEADS

    def test_it_is_short_and_carries_no_set_up_of_the_control(self, folder):
        page = _page(folder, given=_given())
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
        assert "The design in full is in `DESIGN_DETAILS.md`. The steps to make the runs by " \
               "hand are in `RUN_BY_HAND.md`." in _flat(page)
        details = (folder / DP.DETAILS_NAME).read_text()
        assert "## The design" in details and "against observations" in details
        assert dr._PENDING not in details and dr.APPENDIX_MARK not in details
        assert (folder / DP.BY_HAND_NAME).read_text().startswith(dr.APPENDIX_MARK)

    def test_with_nothing_to_make_by_hand_no_such_file_is_named_or_written(self, folder,
                                                                          monkeypatch):
        monkeypatch.setattr(dr, "_appendix", lambda *a, **k: [])
        path = dr.write_review(folder, reception=_reception(), strategy=_strategy(),
                               preview=None, analysis_preview=None)
        assert "RUN_BY_HAND.md" not in path.read_text()
        assert not (folder / DP.BY_HAND_NAME).exists()


def _designs(folder):
    """The page for one run, several runs, no run, a design that cannot run,
    a plan the yes approves and a plan that did not pass."""
    st2 = _strategy()
    st2["domain_experiment"]["runs"].append({"name": "no_fire", "changes": [{"id": "fire_off"}]})
    none = _strategy()
    none["domain_experiment"]["runs"] = []
    refused = {"refused": [{"where": f"run {RUN}", "what": "a made-up refusal"}],
               "runs": [], "error": None}
    bound = _record(folder, handoff={"promised": True, "bound": True, "words": "x"})
    plan_problems = dict(copy.deepcopy(PLAN), problems=["P1: a made-up problem"])
    (folder / "plan" / "bad_plan.json").write_text(json.dumps(plan_problems))
    bad = _record(folder, status=AV.NOT_PASSED, plan_file="plan/bad_plan.json",
                  handoff={"promised": False, "why": "x"})
    return {"one run": _page(folder, given=_given()),
            "several runs": _page(folder, strategy=st2, record=None),
            "no run": _page(folder, strategy=none, preview=None, record=None),
            "cannot run": _page(folder, preview=refused, record=None),
            "plan approved": _page(folder, record=bound),
            "plan not passed": _page(folder, record=bad),
            "plan failed": _page(folder, record=_record(folder, status=AV.FAILED,
                                                        why="a made-up error",
                                                        handoff={"promised": False})),
            "no planning": _page(folder, record=_record(
                folder, status=AV.NOT_PLANNED,
                why="the control run has no results store to plan the comparisons from"))}


class TestTheWordsOfThePage:

    @pytest.mark.parametrize("words", ["approved plan", "approves this analysis",
                                       "the analyzer carries out this plan", "is bound to"])
    def test_it_never_uses_the_words_of_a_later_stage(self, folder, words):
        assert words not in _flat(_page(folder)).lower()

    def test_the_page_never_holds_the_words_of_the_framework(self, folder):
        for name, page in _designs(folder).items():
            low = page.lower()
            for w in FRAMEWORK_WORDS:
                assert w not in low, (name, w)

    def test_no_sentence_points_to_another_section(self, folder):
        for name, page in _designs(folder).items():
            for w in ("is given under", "see above", "under \"", "the section"):
                assert w not in page, (name, w)

    def test_declining_is_not_said_to_cost_nothing(self, folder):
        for name, page in _designs(folder).items():
            low = page.lower()
            for wrong in ("costs nothing", "no cost", "at no cost", "free of"):
                assert wrong not in low, (name, wrong)
        assert DP.NO_COMPUTE in _flat(_designs(folder)["one run"])


# ── 3. the new run ───────────────────────────────────────────────────────────
class TestTheNewRun:

    def test_one_run_one_knob_is_a_sentence_a_table_and_a_figure(self, folder):
        sec = _section(_page(folder, given=_given()), "The new run").strip()
        assert sec.startswith(
            f"We will make one new run, `{RUN}`. It starts from the control run's saved state "
            f"at the beginning of 2031 and simulates 2031 to 2033. The model server estimates "
            f"the cost at about 2.4 node-hours.\n\n")
        assert "| Knob | Change | When | Where |\n|---|---|---|---|\n| Rain and snow | " \
               "multiplied by 0.7 | June to August of 2033 | everywhere |\n" in sec
        assert f"![What the new run is given](design_figures/new_run_{RUN}_series.png)" in sec
        assert "The figure shows what the new run is given beside the control run, month by " \
               "month. The shaded months are the changed ones." in _flat(sec)
        # with a table, the server's long words for the change stay in the details
        assert "the difference from the control run is" not in sec
        assert sec.index("| Rain and snow |") < sec.index("![What the new run is given]")

    def test_two_knobs_are_two_rows(self, folder):
        warm = {"knob": "Air temperature", "change": "2 K warmer", "when": "June to August "
                "of 2033", "where": "everywhere"}
        run = dict(_given()["record"]["runs"][0], rows=[RAIN_ROW, warm])
        sec = _section(_page(folder, given=_given(run)), "The new run")
        rows = [ln for ln in sec.splitlines() if ln.startswith("| ") and "---" not in ln]
        assert rows == ["| Knob | Change | When | Where |",
                        "| Rain and snow | multiplied by 0.7 | June to August of 2033 | "
                        "everywhere |",
                        "| Air temperature | 2 K warmer | June to August of 2033 | "
                        "everywhere |"]

    def test_several_runs_have_a_run_column_and_their_own_figures(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"].append({"name": "no_fire", "changes": [{"id": "fire_off"}]})
        fire = {"name": "no_fire", "rows": [{"knob": "Fire", "change": "switched off",
                                             "when": "every year of the run",
                                             "where": "everywhere"}],
                "panels": [{"title": "Area the control run's own fire burns", "units": "km2",
                            "step": "year", "labels": ["2031", "2032", "2033"],
                            "control": [5.0, 6.0, 7.0], "run": [0.0, 0.0, 0.0],
                            "changed": ["2031", "2032", "2033"]}], "map": None, "problems": []}
        given = _given(_given()["record"]["runs"][0], fire)
        page = _page(folder, strategy=st, record=None, given=given)
        sec = _section(page, "The new runs")
        flat = _flat(sec)
        assert flat.startswith(f"We will make 2 new runs, `{RUN}` and `no_fire`. Each starts "
                               f"from the control run's saved state at the beginning of 2031")
        assert "| Run | Knob | Change | When | Where |" in sec
        assert f"| `{RUN}` | Rain and snow | multiplied by 0.7 |" in sec
        assert "| `no_fire` | Fire | switched off | every year of the run | everywhere |" in sec
        a, b = sec.index(f"**`{RUN}`**"), sec.index("**`no_fire`**")
        assert a < sec.index(f"new_run_{RUN}_series.png") < b < \
            sec.index("new_run_no_fire_series.png")
        assert "The figure shows what `no_fire` is given beside the control run, year by " \
               "year. The shaded years are the changed ones." in flat
        assert "We will compare each new run with the control run." in \
            _flat(_section(page, "How your question will be answered"))

    def test_no_record_keeps_the_servers_own_words_for_each_change(self, folder):
        sec = _flat(_section(_page(folder), "The new run"))
        meaning = DC.resolve(CHANGE)["meaning"].replace("the control", "the control run")
        assert f"In `{RUN}`, {meaning}." in sec
        assert "| Knob |" not in sec and "![" not in sec

    def test_a_record_of_another_design_is_not_used(self, folder):
        for other in (_given(control="another"), _given(years=[2033])):
            sec = _flat(_section(_page(folder, given=other), "The new run"))
            assert "| Knob |" not in sec and "![" not in sec
            assert f"In `{RUN}`, the difference from the control run is" in sec

    def test_a_run_the_record_does_not_hold_is_said_in_the_servers_words(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"].append({"name": "more_co2", "changes": [
            {"id": "co2_ppmv", "params": {"ppmv": 560.0}}]})
        sec = _flat(_section(_page(folder, strategy=st, record=None, given=_given()),
                             "The new runs"))
        assert f"| `{RUN}` | Rain and snow |" in sec and "`more_co2` |" not in sec
        assert "In `more_co2`, the difference from the control run is" in sec

    def test_a_run_with_rows_and_no_panel_gets_its_table_and_no_figure(self, folder):
        run = dict(_given()["record"]["runs"][0], panels=[])
        sec = _section(_page(folder, given=_given(run)), "The new run")
        assert "| Rain and snow |" in sec and "![" not in sec and "The figure" not in sec

    def test_a_figure_that_failed_leaves_the_table(self, folder):
        given = _given(drawn={"runs": {RUN: {"series": None, "map": None}},
                              "problems": ["the figure was not drawn: a made-up reason"]})
        sec = _section(_page(folder, given=given), "The new run")
        assert "| Rain and snow |" in sec and "![" not in sec
        assert "made-up reason" not in sec

    def test_a_map_is_shown_and_said_only_where_one_was_drawn(self, folder):
        drawn = {"runs": {RUN: {"series": "f/s.png", "map": "f/m.png"}}}
        sec = _flat(_section(_page(folder, given=_given(drawn=drawn)), "The new run"))
        assert "![Where the change applies](f/m.png)" in sec
        assert sec.endswith("The map shows the mean change in those months.")
        assert "map" not in _flat(_section(_page(folder, given=_given()), "The new run"))

    def test_a_record_that_breaks_the_page_code_leaves_the_servers_words(self, folder):
        broken = {"record": {"control": "valley", "years": [2031, 2032, 2033], "runs": 7},
                  "drawn": {"runs": "not a map"}}
        sec = _flat(_section(_page(folder, given=broken), "The new run"))
        assert f"In `{RUN}`, the difference from the control run is" in sec

    def test_the_control_run_alone(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"] = []
        sec = _flat(_section(_page(folder, strategy=st, preview=None, record=None),
                             "The new run"))
        assert sec == "No new run is made. The control run's existing output is read for 2033."

    def test_the_control_run_made_again(self, folder):
        st = _strategy()
        st["domain_experiment"].update(runs=[], control={"name": "valley", "reuse": False})
        sec = _flat(_section(_page(folder, strategy=st, record=None), "The new run"))
        assert sec == ("No run with a change is made. The control run is made again from its "
                       "own recipe and read for 2033. The model server estimates the cost at "
                       "about 2.4 node-hours.")

    def test_a_run_that_names_no_change_repeats_the_control_run(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"].append({"name": "same", "changes": []})
        sec = _flat(_section(_page(folder, strategy=st, record=None, given=_given()),
                             "The new runs"))
        assert "`same` names no change, so it repeats the control run." in sec

    def test_a_change_in_effect_in_every_simulated_year_is_said(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"] = [{"name": "more_co2", "changes": [
            {"id": "co2_ppmv", "params": {"ppmv": 560.0}}]}]
        sec = _flat(_section(_page(folder, strategy=st, record=None), "The new run"))
        assert "The change `co2_ppmv` is in effect in every simulated year, 2031 to 2033, " \
               "and not only in the years reported. A reported year carries its effect from " \
               "2031 on." in sec
        # the change that names its years has no such sentence
        assert "in effect in every simulated year" not in _page(folder)

    def test_the_cost_without_the_servers_estimate_is_the_controls_hours(self, folder):
        preview = dict(PREVIEW, check={"unusual": []})
        sec = _flat(_section(_page(folder, preview=preview), "The new run"))
        # 1 run, 3 simulated years, 0.5 node-hours a model year
        assert f"The model server estimates the cost at about {1 * 3 * 0.5} node-hours." in sec

    def test_what_moved_out_of_this_section(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"] = [{"name": "more_co2", "changes": [
            {"id": "co2_ppmv", "params": {"ppmv": 560.0}}]}]
        for page in (_page(folder), _page(folder, strategy=st, record=None)):
            sec = _section(page, "The new run")
            for gone in ("The numbers are reported", "Different from what you asked",
                         "No run with this change has been completed"):
                assert gone not in sec, gone


# ── 4. how your question will be answered ────────────────────────────────────
ANSWER = "How your question will be answered"


class TestHowYourQuestionWillBeAnswered:

    def test_the_opening_says_what_is_compared_and_the_years_reported(self, folder):
        sec = _flat(_section(_page(folder), ANSWER))
        assert sec.startswith(
            "We will compare the new run with the control run. We will also compare 2033 in "
            "the control run with its normal years, 2032 to 2034, which contain 2033. The "
            "numbers are reported for 2033, the year you asked about.")

    def test_a_question_about_the_control_run_alone_compares_its_years(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"] = []
        _plan(folder, comparisons=[PLAN["comparisons"][1]])
        sec = _flat(_section(_page(folder, strategy=st, record=_record(folder, st)), ANSWER))
        assert sec.startswith("We will compare 2033 in the control run with its normal "
                              "years, 2032 to 2034, which contain 2033.")
        # with no plan, the design's own normal years
        st["domain_experiment"]["analysis"]["normal_years"] = [2031, 2032]
        sec = _flat(_section(_page(folder, strategy=st, record=None), ANSWER))
        assert sec.startswith("We will compare 2033 in the control run with its normal years, "
                              "2031 to 2032. The numbers are reported for 2031 to 2033.")

    def test_years_that_are_the_normal_years_themselves_are_said_plainly(self, folder):
        _plan(folder, comparisons=[{"id": "C1", "what": "each year against the normal years",
                                    "run": "control", "against": "normal years",
                                    "years": [2032, 2033, 2034]}])
        sec = _flat(_section(_page(folder), ANSWER))
        assert "We will also compare each year of 2032 to 2034 in the control run with the " \
               "mean of those same years." in sec
        assert "which contain" not in sec

    def test_the_table_has_four_columns_in_the_readers_terms(self, folder):
        sec = _section(_page(folder), ANSWER)
        rows = [ln for ln in sec.splitlines() if ln.startswith("| ")]
        assert rows[0] == "| What we compare | When | How | What it tells us |"
        assert rows[1] == ("| Monthly change in plant growth | June to August of 2033 | the "
                           "new run minus the control run | "
                           + PLAN["analyses"][0]["question"] + " |")
        # no title: the fields, in words; a daily series with the years it covers
        assert rows[2] == ("| LEAFX and SOILX | each day of 2031 to 2033 | the two runs as "
                           "daily lines | " + PLAN["analyses"][1]["question"] + " |")
        for gone in ("| P1 |", "P2", "Status", "| Place |", "quantit", "1,100", "830"):
            assert gone not in sec, gone

    def test_what_it_tells_us_is_the_plans_own_field_where_it_gives_one(self, folder):
        plan = copy.deepcopy(PLAN)
        plan["analyses"][0]["tells"] = "How much growth falls when summer rain is cut"
        _plan(folder, analyses=plan["analyses"])
        sec = _section(_page(folder), ANSWER)
        assert "| the new run minus the control run | How much growth falls when summer rain " \
               "is cut |" in sec
        assert PLAN["analyses"][0]["question"] not in sec

    def _many(self, folder, n, main=()):
        a = PLAN["analyses"][0]
        many = [dict(copy.deepcopy(a), id=f"P{i}", title=f"Made-up comparison {i}",
                     **({"main": True} if i in main else {})) for i in range(1, n + 1)]
        _plan(folder, analyses=many)
        return _section(_page(folder), ANSWER)

    def test_at_most_four_and_the_rest_are_counted(self, folder):
        sec = self._many(folder, 7)
        shown = [i for i in range(1, 8) if f"| Made-up comparison {i} |" in sec]
        assert shown == [1, 2, 3, 4]
        assert "3 more comparisons are listed in `DESIGN_DETAILS.md`." in _flat(sec)
        sec = self._many(folder, 5)
        assert "1 more comparison is listed in `DESIGN_DETAILS.md`." in _flat(sec)
        sec = self._many(folder, 4)
        assert "more comparison" not in sec

    def test_the_main_ones_are_shown_where_the_plan_marks_them(self, folder):
        sec = self._many(folder, 6, main=(2, 5))
        shown = [i for i in range(1, 7) if f"| Made-up comparison {i} |" in sec]
        assert shown == [2, 5]
        assert "4 more comparisons are listed in `DESIGN_DETAILS.md`." in _flat(sec)
        sec = self._many(folder, 6, main=(1, 2, 3, 4, 5))
        assert [i for i in range(1, 7) if f"| Made-up comparison {i} |" in sec] == [1, 2, 3, 4]

    def test_a_conditional_comparison_is_named_with_its_reason(self, folder):
        sec = _flat(_section(_page(folder), ANSWER))
        assert 'The comparison "LEAFX and SOILX" gives figures and no number the report may ' \
               'state, since each of its outputs is a map or a daily series.' in sec
        # a supported one gets no sentence
        assert 'The comparison "Monthly change in plant growth"' not in sec

    def test_a_blocked_comparison_is_named_with_its_reason_without_its_id(self, folder):
        _plan(folder, problems=["P1: no catalog field 'NOPE'", "P1.O1: a made-up second one",
                                "a problem of the whole plan"])
        page = _page(folder, record=_record(folder, status=AV.NOT_PASSED))
        sec = _flat(_section(page, ANSWER))
        assert 'The comparison "Monthly change in plant growth" cannot be made as planned. No ' \
               "catalog field 'NOPE'. A made-up second one." in sec
        assert "These comparisons did not pass their check. This does not stop the design, " \
               "since they are chosen again after the run." in sec
        assert "- a problem of the whole plan" in _section(page, ANSWER)
        assert "P1" not in sec

    def test_other_explanations_by_their_titles_or_left_to_the_details(self, folder):
        sec = _flat(_section(_page(folder), ANSWER))
        assert "We will also check 2 other explanations, which are a nitrogen limit and the " \
               "model's own fire." in sec
        plan = copy.deepcopy(PLAN)
        del plan["competing"][1]["title"]
        _plan(folder, competing=plan["competing"])
        sec = _flat(_section(_page(folder), ANSWER))
        assert "We will also check 2 other explanations, which are listed in " \
               "`DESIGN_DETAILS.md`." in sec
        _plan(folder, competing=plan["competing"][:1])
        assert "We will also check one other explanation, which is a nitrogen limit." in \
            _flat(_section(_page(folder), ANSWER))

    def test_with_no_plan_the_comparisons_are_chosen_later_and_nothing_else(self, folder):
        sec = _flat(_section(_page(folder, record=None), ANSWER))
        assert sec == ("We will compare the new run with the control run. The numbers are "
                       "reported for 2033, the year you asked about. The comparisons are "
                       "chosen after the run, so none is listed here.")
        st = _strategy()
        st["domain_experiment"]["runs"].append({"name": "no_fire", "changes": [{"id": "fire_off"}]})
        assert "chosen after the runs, so none is listed here." in \
            _flat(_section(_page(folder, strategy=st, record=None), ANSWER))

    def test_a_plan_shown_and_not_approved_is_said_to_be_a_preview(self, folder):
        sec = _flat(_section(_page(folder), ANSWER))
        assert sec.endswith("These comparisons are a preview. Your yes does not approve them, "
                            "and they are chosen again after the run.")
        bound = _record(folder, handoff={"promised": True, "bound": True, "words": "x"})
        assert "preview" not in _section(_page(folder, record=bound), ANSWER)
        kept = _record(folder, handoff={"promised": True, "words": "x"})
        assert _flat(_section(_page(folder, record=kept), ANSWER)).endswith(
            "These comparisons are carried out as listed after the run.")

    def test_a_plan_that_could_not_be_written_or_was_not_planned(self, folder):
        failed = _record(folder, status=AV.FAILED, why="RuntimeError: a made-up error")
        assert _flat(_section(_page(folder, record=failed), ANSWER)).endswith(
            "The comparisons could not be planned before this page. The error was "
            "RuntimeError: a made-up error. This does not stop the design, since they are "
            "then chosen after the run.")
        none = _record(folder, status=AV.NOT_PLANNED, why="a made-up reason")
        assert _flat(_section(_page(folder, record=none), ANSWER)).endswith(
            "The comparisons are chosen after the run, so none is listed here. None could be "
            "planned before this page, since a made-up reason.")

    def test_a_plan_of_another_design_is_not_shown(self, folder):
        sec = _flat(_section(_page(folder, record=_record(folder, design_sha256="0" * 64)),
                             ANSWER))
        assert sec.endswith("The comparisons on record here were planned for another design, "
                            "so they are not shown.")
        assert "|" not in sec and "which contain" not in sec

    def test_a_plan_whose_file_cannot_be_read(self, folder):
        (folder / "plan" / "analysis_plan.json").write_text("{not json")
        assert _flat(_section(_page(folder), ANSWER)).endswith(
            "The list of comparisons could not be read, so none is shown here.")


class TestTheWordsOfTheTable:

    def test_every_kind_of_output_the_plan_contract_knows_has_its_words(self):
        from agents.analysis import plan_contract as PC
        said = {}
        for kind in list(PC.KINDS) + ["snow_gone"]:
            o = {"kind": kind, "run": RUN, "region": "r", "step": "month"}
            said[kind] = AV.how_words({"outputs": [o]})
            assert said[kind] not in ("", "not named") and "_" not in said[kind], (kind, said)
        assert len(set(said.values())) == len(said)                  # one phrase a kind
        assert AV.how_words({"outputs": [{"kind": "difference", "run": RUN, "region": "r"}]}) \
            == "the new run minus the control run"
        assert AV.how_words({"outputs": [{"kind": "anomaly", "region": "r"}]}) == \
            "the departure from the mean of the normal years"
        assert AV.how_words({"outputs": [{"kind": "map"}]}) == "a map over the grid"
        assert AV.how_words({"outputs": [{"kind": "elevation_split", "region": "r"}]}) == \
            "a split by elevation band"
        assert AV.how_words({"outputs": [{"kind": "overlap", "region": "r"}]}) == \
            "one map set against another"
        assert AV.how_words({"outputs": [{"kind": "snow_gone", "region": "r"}]}) == \
            "the date it happens in each year"

    def test_a_map_of_a_difference_is_said_cell_by_cell_and_several_kinds_are_joined(self):
        assert AV.how_words({"outputs": [{"kind": "difference", "run": RUN, "region": None}]}) \
            == "the new run minus the control run, cell by cell"
        both = {"outputs": [{"kind": "difference", "run": RUN, "region": "r"},
                            {"kind": "series", "run": RUN, "step": "month"},
                            {"kind": "series", "run": "control", "step": "month"},
                            {"kind": "difference", "run": RUN, "region": "s"}]}
        assert AV.how_words(both) == ("the new run minus the control run and the two runs as "
                                      "monthly lines")

    def test_a_difference_with_several_runs_names_its_run(self):
        assert AV.how_words({"outputs": [{"kind": "difference", "run": "b", "region": "r"}]},
                            n_runs=2) == "`b` minus the control run"

    @pytest.mark.parametrize("outs, words", [
        ([{"kind": "series", "run": "control", "step": "year"}], "the control run as a yearly line"),
        ([{"kind": "series", "run": RUN, "step": "month"}], "the new run as a monthly line"),
        ([{"kind": "series", "run": RUN, "step": "season"},
          {"kind": "series", "run": RUN, "step": "season"}], "the new run as seasonal lines"),
        ([{"kind": "series", "run": "a", "step": "day"}, {"kind": "series", "run": "b",
                                                          "step": "day"},
          {"kind": "series", "run": "control", "step": "day"}], "the runs as daily lines")])
    def test_a_series_is_said_with_its_runs_and_its_step(self, outs, words):
        assert AV.series_how(outs, 1 if len({o["run"] for o in outs} - {"control"}) <= 1
                             else 2) == words

    def test_the_months_of_several_years_are_said_once(self):
        months = [f"{y}-{m:02d}" for y in (2033, 2034) for m in (6, 7, 8)]
        assert AV.periods_words(months, grouped=True) == "June to August of 2033 and 2034"
        every = [f"{y}-{m:02d}" for y in (2031, 2032, 2033) for m in range(1, 13)]
        assert AV.periods_words(every, grouped=True) == "each month of 2031 to 2033"
        assert AV.periods_words(["2033-03", "2033-10"], grouped=True) == \
            "March and October of 2033"
        assert AV.periods_words(["2033-JJA", "2034-JJA", "2033"], grouped=True) == \
            "summer of 2033 and 2034; 2033"
        assert AV.periods_words(["WY2033", "WY2034", "normal"], grouped=True) == \
            "the water years 2033 and 2034; normal"
        # different months in different years are said apart
        assert AV.periods_words(["2033-06", "2034-07"], grouped=True) == \
            "June of 2033; July of 2034"
        # the details keep their own words
        assert AV.periods_words(months[:3]) == "June to August 2033, by month"

    def test_the_years_of_a_series_are_its_runs(self):
        a = {"outputs": [{"kind": "series", "run": RUN, "step": "month"}]}
        assert AV.when_words(a, {"years": [2033, 2034]}, CONTROL) == "each month of 2033 and 2034"
        # a series of the control run alone covers its whole output
        steps_only = {"steps": ["day"], "outputs": []}
        assert AV.when_words(steps_only, {"years": [2031]}, CONTROL) == \
            "each day of 2031 to 2036"
        assert AV.when_words({"outputs": []}, {}, CONTROL) == "not named"
        # periods named by the outputs alone
        timing = {"outputs": [{"kind": "snow_gone", "period": "2033", "region": "r"}]}
        assert AV.when_words(timing, {}, CONTROL) == "2033"


# ── 5. what the answer will not cover ────────────────────────────────────────
COVER = "What the answer will not cover"
CUT = ("The run cuts rain and snow only. Air temperature, humidity and sunshine stay as "
       "recorded, so it is not a full drought.")
FIRE_LIMIT = ("The run removes the model's own fire, which is unrelated in place and year to "
              "the fires that happened. It does not show the effect of an observed fire.")


class TestWhatTheAnswerWillNotCover:

    def test_the_target_three_points(self, folder):
        _resolved(folder, [{"name": RUN, "changes": [CHANGE]}], {"weather_scale": [CUT]})
        st = _strategy()
        st["domain_experiment"]["question_limits"] = [
            "The control run holds less snow than measured and melts it early. Plants in the "
            "model could depend on summer rain more than real plants do."]
        sec = _section(_page(folder, strategy=st), COVER)
        assert sec.strip().splitlines() == [
            f"- {CUT}",
            "- The control run holds less snow than measured and melts it early. Plants in the "
            "model could depend on summer rain more than real plants do. This is the Planner's "
            "judgement.",
            "- The control run does not contain the observed floods: the run makes its own."]

    def test_with_every_limit_of_a_knob_the_caveats_are_left_to_the_details(self, folder):
        _resolved(folder, [{"name": RUN, "changes": [CHANGE]}], {"weather_scale": [CUT]})
        page = _page(folder)
        for c in CAVEATS:
            assert c not in page
        details = _flat(_details(folder))
        for c in CAVEATS:
            assert c in details

    def test_a_limit_said_by_two_runs_is_printed_once(self, folder):
        st = _strategy()
        two = st["domain_experiment"]["runs"] + [{"name": "drier", "changes": [
            dict(copy.deepcopy(CHANGE), params={"precip_factor": 0.5})]},
            {"name": "no_fire", "changes": [{"id": "fire_off"}]}]
        st["domain_experiment"]["runs"] = two
        _resolved(folder, two, {"weather_scale": [CUT], "fire_off": [FIRE_LIMIT]})
        sec = _section(_page(folder, strategy=st, record=None), COVER)
        assert sec.count(CUT) == 1 and sec.count(FIRE_LIMIT) == 1
        assert sec.index(CUT) < sec.index(FIRE_LIMIT)

    def test_a_change_without_its_limit_brings_the_caveats_back_whole(self, folder):
        long = "A caveat " + "that goes on and on " * 40 + "to its very last word."
        st = _strategy()
        st["domain_experiment"]["caveats"] = CAVEATS + [long]
        st["domain_experiment"]["runs"].append({"name": "no_fire", "changes": [{"id": "fire_off"}]})
        _resolved(folder, st["domain_experiment"]["runs"], {"weather_scale": [CUT]})
        sec = _section(_page(folder, strategy=st, record=None), COVER)
        for c in CAVEATS:
            assert f"- {c}" in sec
        assert _flat(long) in _flat(sec)                                # never cut
        assert f"- {CUT}" in sec
        # an older record with no limit at all
        _resolved(folder, [{"name": RUN, "changes": [CHANGE]}])
        sec = _section(_page(folder), COVER)
        for c in CAVEATS:
            assert f"- {c}" in sec

    def test_a_record_of_the_wrong_shape_brings_the_caveats_back(self, folder):
        for bad in ({"check": ["not", "a", "map"]}, {"check": {"runs": 5}}, [1, 2]):
            (folder / dr.RESOLVED_FILE).write_text(json.dumps(bad))
            page = _page(folder)
            for c in CAVEATS:
                assert f"- {c}" in _section(page, COVER)
            assert f"`{RUN}`" in _section(page, "The new run")

    def test_a_question_about_the_control_run_alone_keeps_its_caveats(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"] = []
        sec = _section(_page(folder, strategy=st, preview=None, record=None), COVER)
        for c in CAVEATS:
            assert f"- {c}" in sec

    def test_the_planners_verdict_in_part_comes_first_with_its_reason(self, folder):
        _resolved(folder, [{"name": RUN, "changes": [CHANGE]}], {"weather_scale": [CUT]})
        st = _strategy(feasibility={"verdict": "partial",
                                    "why": "the second half needs a change that is not there"},
                       requires=["a change of the made-up river's course"])
        lines = _section(_page(folder, strategy=st), COVER).strip().splitlines()
        assert lines[0] == ("- The Planner judges that this design answers the question in "
                            "part. The second half needs a change that is not there.")
        assert lines[1] == ("- The question would also need something that is not available "
                            "here. A change of the made-up river's course.")
        assert lines[2] == f"- {CUT}"
        # a verdict of "full" is not on the short page
        assert "judges" not in _section(_page(folder), COVER)
        st = _strategy(feasibility={})
        assert "- The Planner gave no verdict on whether this design can answer the " \
               "question." in _section(_page(folder, strategy=st), COVER)

    def test_what_the_control_run_does_not_contain_is_said_once(self, folder):
        page = _page(folder)
        assert "- The control run does not contain the observed floods: the run makes its " \
               "own." in _flat(_section(page, COVER))
        assert page.count("the observed floods") == 1
        # where a printed caveat names it, the caveat says it
        st = _strategy()
        st["domain_experiment"]["caveats"] = CAVEATS + [
            "The control does not hold the observed floods, since its floods are its own."]
        page = _page(folder, strategy=st)
        assert page.count("the observed floods") == 1
        assert "- The control run does not hold the observed floods, since its floods are " \
               "its own." in _flat(_section(page, COVER))
        # and where a limit of the Planner's names it
        _resolved(folder, [{"name": RUN, "changes": [CHANGE]}], {"weather_scale": [CUT]})
        st = _strategy()
        st["domain_experiment"]["question_limits"] = ["The observed floods are not in the run."]
        page = _page(folder, strategy=st)
        assert page.count("bserved floods") == 1

    def test_a_caveat_that_is_the_servers_difference_is_printed_once(self, folder):
        st = _strategy()
        st["domain_experiment"]["caveats"] = CAVEATS + [SAID]
        page = _page(folder, strategy=st)
        assert _flat(page).count(SAID) == 1
        assert SAID in _flat(_section(page, "Choices made for you"))
        sec = _section(page, COVER)
        for c in CAVEATS:
            assert f"- {c}" in sec
        # a caveat in other words is not taken for the same sentence
        st["domain_experiment"]["caveats"] = CAVEATS + ["The run scales snow as well as rain."]
        assert "- The run scales snow as well as rain." in \
            _section(_page(folder, strategy=st), COVER)

    def test_with_nothing_to_say(self, folder):
        _resolved(folder, [{"name": RUN, "changes": [CHANGE]}], {"weather_scale": []})
        st = _strategy()
        st["domain_rulebook"]["controls"]["controls"][0]["not_in_it_words"] = []
        assert _flat(_section(_page(folder, strategy=st), COVER)) == "The design records no limit."


# ── 6. choices made for you ──────────────────────────────────────────────────
CHOICES = "Choices made for you"


class TestChoicesMadeForYou:

    def test_in_this_order(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"].append({"name": "more_co2", "changes": [
            {"id": "co2_ppmv", "params": {"ppmv": 560.0}}]})
        lines = _section(_page(folder, strategy=st, record=None), CHOICES).strip().splitlines()
        assert lines == [
            f"- {SAID}",
            "- The question named no area, so the whole box is used.",
            "- Noted by Reception: A thing Reception noted.",
            f"- The model server has a note on run {RUN}, change weather_scale. A made-up "
            f"thing to note.",
            "- The model server has a note on the change `co2_ppmv`. "
            + DC.declare()["demonstrated_false"].replace("the control", "the control run")
            + "."]

    @pytest.mark.parametrize("question", [
        "What if June to August precipitation had been 30 percent lower in 2033?",
        "What if summer rain and snow had been 30 percent lower in 2033?",
        "What if summer rainfall and snowfall had been lower in 2033?"])
    def test_a_question_that_names_snow_or_precipitation_gets_no_such_line(self, folder,
                                                                         question):
        assert SAID not in _flat(_page(folder, reception=_reception(question)))

    def test_a_change_of_temperature_alone_gets_no_such_line(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"][0]["changes"][0]["params"] = {
            "temperature_K": 2.0, "months": [6, 7, 8]}
        assert SAID not in _flat(_page(folder, strategy=st))

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

    def test_a_remark_that_makes_the_same_point_is_not_printed_again(self, folder):
        same = "The question says rainfall. The change multiplies rain and snow together."
        other = "The question says rainfall, so the whole summer is used."
        rec = _reception(run_settings={"requested_period": {"yr_start": 2033, "yr_end": 2033},
                                       "conflicts": [same, other]},
                         assumptions=["a made-up assumption"])
        sec = _section(_page(folder, reception=rec), CHOICES)
        assert same not in sec and f"- {other}" in sec
        assert "- A made-up assumption." in sec
        assert "Decided without you" not in sec
        # with no difference said, the same remark is printed
        rec["user_request"] = "What if summer precipitation had been lower in 2033?"
        assert f"- {same}" in _section(_page(folder, reception=rec), CHOICES)

    def test_a_refused_round_and_the_inputs_check_are_kept(self, folder):
        preview = dict(PREVIEW, rounds=[{"findings": [{"where": "run x", "what": "a made-up "
                                                       "reason"}], "design": "s.json"}],
                       inputs_check={"pass": False, "reasons": ["a made-up file is missing"]},
                       problems=["a made-up record could not be read"])
        sec = _section(_page(folder, preview=preview), CHOICES)
        assert "- The model server refused an earlier version of this design, and the Planner " \
               "changed it before this page:\n  - Run x: a made-up reason." in sec
        assert "- The input files failed their check. A made-up file is missing." in sec
        assert "- Not shown in the details: A made-up record could not be read." in sec

    def test_what_is_left_to_the_details(self, folder):
        page = _page(folder)
        details = _flat(_details(folder))
        for gone, kept in (
                ("can answer the question in full", "**Feasibility: full.**"),
                ("checked the design", "**Passed.**"),
                ("passed its check", "- Code's check and the model review: **passed**."),
                ("remark", f"- The model review's remarks: - {REMARKS[0]}"),
                ("not made yet", f"- `{RUN}` is not made yet, so the plan takes its outputs to "
                                 f"be the control run's quantities over the planned years. "
                                 f"Each year is checked after it runs.")):
            assert gone not in page, gone
            assert kept in details, kept
        for r in REMARKS:
            assert r not in page and r in details

    def test_with_nothing_to_say(self, folder):
        rec = _reception("What if summer precipitation had been lower in 2033?",
                         run_settings={}, known_data_gaps=[])
        page = _page(folder, reception=rec, preview=dict(PREVIEW, check={}))
        assert _flat(_section(page, CHOICES)) == "No choice was made for you."


# ── 7. your decision ─────────────────────────────────────────────────────────
DECISION = "Your decision"


class TestYourDecision:

    def test_the_three_answers_for_one_run(self, folder):
        sec = _section(_page(folder), DECISION)
        assert sec.split("\n---\n")[0].strip().splitlines() == [
            "Type one of these at the terminal.", "",
            "| You type | What happens |", "|---|---|",
            "| yes | The run is submitted. It may use up to 2.4 node-hours. |",
            "| no | Nothing is submitted, so no computing time is used. The model calls that "
            "prepared this page are already spent. |",
            "| a change, in your own words | The design is made again with your change, and "
            "you get a new page. Nothing is submitted. |", "",
            'An example of a change is "use other years" or "also switch the fire off".', "",
            "The design in full is in `DESIGN_DETAILS.md`. The steps to make the runs by hand "
            "are in `RUN_BY_HAND.md`."]

    def test_several_runs(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"].append({"name": "no_fire", "changes": [{"id": "fire_off"}]})
        assert "| yes | The runs are submitted. They may use up to 2.4 node-hours. |" in \
            _page(folder, strategy=st, record=None)

    def test_no_estimate_no_second_sentence(self, folder):
        assert "| yes | The run is submitted. |" in \
            _page(folder, preview=dict(PREVIEW, check={}))

    def test_a_yes_that_covers_the_comparisons_says_so(self, folder):
        bound = _record(folder, handoff={"promised": True, "bound": True, "words": "x"})
        assert "| yes | The run is submitted. It may use up to 2.4 node-hours. The comparisons " \
               "above are carried out as listed. |" in _page(folder, record=bound)
        assert "carried out as listed" not in _section(_page(folder), DECISION)

    def test_a_question_that_needs_no_run(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"] = []
        assert "| yes | The analysis of the control run's existing output starts. No model run " \
               "is made, and the analysis uses paid model calls. |" in \
            _page(folder, strategy=st, preview=None, record=None)

    def test_the_example_changes_name_no_year(self, folder):
        import re
        assert not re.search(r"\d{4}", DP.EXAMPLE)
        assert DP.EXAMPLE in _page(folder)

    CANNOT = ["", "This design cannot run as it stands.", "", "| You type | What happens |",
              "|---|---|", "| a change, in your own words | The design is made again with your "
              "change and checked again. |", "| nothing | The request stops. Nothing has been "
              "submitted. |"]

    def _cannot(self, page):
        sec = _section(page, DECISION)
        lines = sec.split("\n---\n")[0].strip().splitlines()
        assert "| yes |" not in sec and "| no |" not in sec
        k = lines.index("This design cannot run as it stands.")
        assert lines[k - 1:k + 6] == self.CANNOT
        return lines[:k - 1]

    def test_a_refused_design_says_why_first(self, folder):
        refused = {"refused": [{"where": f"run {RUN}", "what": "a name already taken"},
                               {"where": "the years", "what": "a made-up second reason"}],
                   "runs": [], "error": None}
        why = self._cannot(_page(folder, preview=refused, record=None))
        assert why == ["- The model server refuses this design as it stands:",
                       f"  - Run {RUN}: a name already taken.",
                       "  - The years: a made-up second reason."]
        # its notes are not on the page as notes of a check that passed
        assert "has a note on run" not in _page(folder, preview=dict(refused, check=PREVIEW["check"]),
                                               record=None)

    def test_a_design_that_could_not_be_built(self, folder):
        why = self._cannot(_page(folder, preview={"refused": [], "runs": [],
                                                  "error": "TimeoutError: a made-up wait"},
                                 record=None))
        assert why == ["- The design could not be built before this page. TimeoutError: a "
                       "made-up wait."]

    def test_a_design_judged_infeasible(self, folder):
        st = _strategy(feasibility={"verdict": "infeasible", "why": "no run can show it"})
        why = self._cannot(_page(folder, strategy=st, record=None))
        assert why == ["- The Planner judges that this design cannot answer the question. No "
                       "run can show it."]
        assert "judges" not in _section(_page(folder, strategy=st, record=None), COVER)

    def test_a_design_with_no_comparisons_a_yes_could_approve(self, folder):
        bad = _record(folder, status=AV.NOT_PASSED, handoff={"promised": False, "why": "x"})
        why = self._cannot(_page(folder, record=bad))
        assert why == ["- The comparisons did not pass their check, so a yes could approve "
                       "none of them."]
        sec = _flat(_section(_page(folder, record=bad), ANSWER))
        assert "These comparisons did not pass their check." in sec
        assert "This does not stop the design" not in sec
        failed = _record(folder, status=AV.FAILED, why="a made-up error",
                         handoff={"promised": False})
        assert self._cannot(_page(folder, record=failed)) == [
            "- The comparisons could not be planned before this page, so a yes could approve "
            "none."]

    def test_the_decision_line_is_replaced_on_a_new_page_and_on_an_old_one(self, folder):
        path = dr.write_review(folder, reception=_reception(), strategy=_strategy(),
                               preview=PREVIEW, analysis_preview=None)
        assert path.read_text().count(dr._PENDING) == 1
        dr.record_decision(folder, "DECLINED at the terminal")
        text = path.read_text()
        assert "**Decision:** DECLINED at the terminal at 20" in text
        assert dr._PENDING not in text
        # a page written before 2026-11-13 holds the old line, and it is replaced too
        old = "# Design review: x\n\n---\n" + dr._PENDING_BEFORE + "\n"
        assert "\u2014" in dr._PENDING_BEFORE and "coordinator" in dr._PENDING_BEFORE
        path.write_text(old)
        dr.record_decision(folder, "accepted at the terminal")
        text = path.read_text()
        assert "**Decision:** accepted at the terminal at 20" in text
        assert dr._PENDING_BEFORE not in text and "pending" not in text


# ── the details keep what the page leaves out ────────────────────────────────
class TestTheDetailsKeepEverything:

    def test_every_caveat_whole_however_many(self, folder):
        long = "A caveat " + "that goes on and on " * 40 + "to its very last word."
        many = [f"A made-up caveat number {k}." for k in range(1, 13)] + [long]
        st = _strategy()
        st["domain_experiment"]["caveats"] = many
        details = _flat(_details(folder, strategy=st))
        for c in many:
            assert _flat(c) in details, c[:40]

    def test_every_finding_of_the_server_whole(self, folder):
        def long(k):
            return f"finding {k} " + "that runs on " * 50 + f"to the end of finding {k}"
        preview = dict(PREVIEW, check={"unusual": [{"where": "years", "what": long(1)}],
                                       "cost": {"node_hours": 2.4}},
                       rounds=[{"findings": [{"where": "run x", "what": long(2)}],
                                "design": "s.json"}],
                       problems=[long(3)])
        details = _flat(_details(folder, preview=preview))
        for k in (1, 2, 3):
            assert _flat(long(k)) in details, k
        long = long(4)
        refused = {"refused": [{"where": "run x", "what": long + " refused"}], "runs": [],
                   "error": None}
        assert _flat(long + " refused") in _flat(_details(folder, preview=refused))
        error = {"refused": [], "runs": [], "error": "an error " + "that goes on " * 60 + "end"}
        assert _flat(error["error"]) in _flat(_details(folder, preview=error))

    def test_every_remark_of_the_review_and_the_whole_plan(self, folder):
        plan = copy.deepcopy(PLAN)
        plan["analyses"][0]["tells"] = "How much growth falls"
        # problems of the whole plan, not of one analysis: only the details'
        # list of problems can hold them (the status lines hold an analysis's)
        plan["problems"] = [f"a made-up problem of the plan, number {k}" for k in range(1, 12)]
        _plan(folder, analyses=plan["analyses"], problems=plan["problems"])
        details = _flat(_details(folder, record=_record(folder, status=AV.NOT_PASSED)))
        for r in REMARKS:
            assert r in details
        for a in plan["analyses"]:
            assert f"| {a['id']} | {a['question']} |" in details
        assert "| Monthly change in plant growth | How much growth falls |" in details
        for p in plan["problems"]:
            assert p in details
        for c in plan["competing"]:
            assert c["text"] in details
        assert INCLUSIVE in details

    def test_the_places_with_their_cells_and_area(self, folder):
        details = _details(folder)
        assert "| vegetated cells | 1,100 | 830 | a made-up rule |" in details
        assert "| valley floor | 300 | 210 | below a made-up line |" in details
        details = _details(folder, record=_record(folder, regions={}))
        assert "| vegetated cells | not on record | not on record | not on record |" in details

    def test_the_planners_limits_and_every_remark_of_reception(self, folder):
        st = _strategy()
        st["domain_experiment"]["question_limits"] = ["A made-up limit of the question."]
        many = [f"a made-up conflict {k}" for k in range(1, 9)]
        rec = _reception(run_settings={"conflicts": many}, known_data_gaps=["g1", "g2", "g3", "g4"])
        details = _details(folder, strategy=st, reception=rec)
        assert "A made-up limit of the question." in details
        for x in many + ["g1", "g2", "g3", "g4"]:
            assert x in details

    def test_what_the_new_runs_are_given_with_what_was_not_drawn(self, folder, monkeypatch):
        from agents import run_figures as RF
        given = _given()["record"]
        given["runs"][0]["problems"] = ["a made-up file is missing"]
        given["runs"][0]["panels"][0]["control"] = [None] * 36
        given["runs"][0]["panels"][0]["run"] = [None] * 36
        dr.write_review(folder, reception=_reception(), strategy=_strategy(), preview=PREVIEW,
                        analysis_preview=None, run_given=given)
        page = (folder / dr.REVIEW_NAME).read_text()
        details = (folder / DP.DETAILS_NAME).read_text()
        assert "| Rain and snow |" in page and "![" not in _section(page, "The new run")
        sec = _section(details, "What the new runs are given")
        assert f"- Not shown: `{RUN}`: a made-up file is missing" in sec
        assert f"- Not shown: the figure of `{RUN}` was not drawn" in sec
        assert details.index("## What the new runs are given") < \
            details.index("## Why this can answer it")
        assert json.loads((folder / RF.RECORD_FILE).read_text()) == given


class TestTheRecordOfWhatTheRunIsGiven:

    def test_it_is_drawn_kept_and_drawn_again_from_its_record(self, folder):
        from agents import run_figures as RF
        given = _given()["record"]
        dr.write_review(folder, reception=_reception(), strategy=_strategy(), preview=PREVIEW,
                        analysis_preview=None, run_given=given)
        first = (folder / dr.REVIEW_NAME).read_text()
        assert f"![What the new run is given](design_figures/new_run_{RUN}_series.png)" in first
        assert (folder / "design_figures" / RF.series_png(RUN)).stat().st_size > 5000
        assert RF.load(folder) == given
        dr.write_review(folder, reception=_reception(), strategy=_strategy(), preview=PREVIEW,
                        analysis_preview=None)                        # left out: read
        assert (folder / dr.REVIEW_NAME).read_text() == first

    def test_a_record_of_another_design_is_neither_drawn_nor_shown(self, folder):
        from agents import run_figures as RF
        for other in (dict(_given()["record"], control="another"),
                      dict(_given()["record"], years=[2033])):
            dr.write_review(folder, reception=_reception(), strategy=_strategy(),
                            preview=PREVIEW, analysis_preview=None, run_given=other)
            page = (folder / dr.REVIEW_NAME).read_text()
            details = (folder / DP.DETAILS_NAME).read_text()
            assert "| Knob |" not in page and "What the new runs are given" not in details
            assert not (folder / "design_figures" / RF.series_png(RUN)).exists()
            assert RF.load(folder) == other                 # kept as it was given

    def test_none_removes_an_earlier_record_and_the_page_has_no_table(self, folder):
        from agents import run_figures as RF
        dr.write_review(folder, reception=_reception(), strategy=_strategy(), preview=PREVIEW,
                        analysis_preview=None, run_given=_given()["record"])
        dr.write_review(folder, reception=_reception(), strategy=_strategy(), preview=PREVIEW,
                        analysis_preview=None, run_given=None)
        assert RF.load(folder) is None
        page = (folder / dr.REVIEW_NAME).read_text()
        assert "| Knob |" not in page and f"In `{RUN}`, the difference" in page

    def test_the_coordinator_passes_what_the_server_gives(self, monkeypatch, tmp_path):
        import workflow as wf
        from agents import run_figures as RF
        seen = {}
        monkeypatch.setattr(RF, "fetch", lambda plan, run_dir: seen.setdefault(
            "args", (plan, run_dir)) and {"made": "up"})
        co = object.__new__(wf.WorkflowCoordinator)
        assert co._run_given({"x": 1}, tmp_path) == {"made": "up"}
        assert seen["args"] == ({"x": 1}, tmp_path)

        def broke(plan, run_dir):
            raise RuntimeError("a made-up failure")
        monkeypatch.setattr(RF, "fetch", broke)
        assert co._run_given({"x": 1}, tmp_path) is None


# ── the control run's summary ───────────────────────────────────────────────
class TestTheControlRunsSummary:

    def test_the_summary_is_printed_as_it_stands_with_its_version(self, folder):
        sec = _section(_page(folder), "The control run")
        for t in SUMMARY["text"]:
            assert t in sec
        assert "This summary of `valley` is version 3, written 2031-02-03. The run's full " \
               "record is in `DESIGN_DETAILS.md`." in _flat(sec)

    def test_a_control_with_no_summary_is_named_and_points_to_its_record(self, folder):
        st = _strategy()
        del st["domain_rulebook"]["controls"]["controls"][0]["page_summary"]
        sec = _flat(_section(_page(folder, strategy=st), "The control run"))
        assert "The control run is `valley`. The model server describes it as the Valley " \
               "control: a made-up box, weather from a made-up record." in sec
        assert "No short summary of it is on record." in sec

    def test_the_server_passes_the_summary_through(self):
        import domain_controls as D
        row = D.summary({"name": "x", "description": "d", "usable": True,
                         "page_summary": SUMMARY}, None)
        assert row["page_summary"] == SUMMARY
        assert D.summary({"name": "x", "description": "d"}, None)["page_summary"] is None

    def test_the_servers_control_is_the_control_run_and_is_named_once(self):
        # the server says "the control"; a person reads "the control run"
        assert DP._control_run("the difference from the control is a factor, with every "
                              "other input the control's") == \
            "the difference from the control run is a factor, with every other input " \
            "the control run's"
        # words that already name it are not given the word twice
        assert DP._control_run("the control run's saved state, read by the control run") \
            == "the control run's saved state, read by the control run"
        assert "prepared run" not in DP._control_run("the control")
        # at the head of a sentence too
        assert DP._control_run("The control holds no floods. The control's snow is its own.") \
            == "The control run holds no floods. The control run's snow is its own."

    def test_the_pages_own_words_do_not_call_runoff_river_flow(self):
        """The model makes runoff where it falls and routes none of it. The
        fixed words of the page never call it a river's flow; a control's own
        summary is its registry's to word."""
        from agents import run_figures as RF
        for mod in (DP, AV, RF):
            src = Path(mod.__file__).read_text().lower()
            for wrong in ("river flow", "streamflow", "discharge"):
                assert wrong not in src, (mod.__name__, wrong)


# ── titles, with the fields as the fallback ──────────────────────────────────
class TestTitles:

    def test_the_details_keep_every_question_under_its_id(self, folder):
        rec = _record(folder)
        details = "\n".join(AV.page_lines(rec, folder, rec["design_sha256"]))
        for a in PLAN["analyses"]:
            assert f"| {a['id']} | {a['question']} |" in details

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
