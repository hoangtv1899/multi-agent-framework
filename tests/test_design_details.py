"""DESIGN_DETAILS.md: what the short design page leaves out (2026-11-13).

    pytest tests/test_design_details.py -v

The page was the old long page with new parts added to it, about 3,000 words of
bullet lists, sentences cut off and limits printed twice. The person who reads
it approved a draft that follows the short page's seven sections and holds only
what the short page leaves out, in tables and two figures. These tests hold
the page (agents/design_details.py), the Planner's direct answer on the short
page, and the wiring of the three pages in design_review.write_review:

  - each section with its record and without it, and a page written whatever
    record is missing, for a study made before the new records too;
  - the Planner's record of both forms (interface I4);
  - the control run alone, several runs, each verdict, a refused round, a list
    of comparisons that failed or did not pass, a conditional comparison;
  - every row of the table of what the old page printed: each is on one of
    the two pages, whole, or left out as bookkeeping;
  - the words of both pages;
  - the two modules of another piece (run_by_hand, places_figure) behind
    their guards, with made-up modules put in their place.

Every name, year and number here is made up, and no test reaches a model.
"""
import copy
import json
import re
import sys
import types
from pathlib import Path

import pytest

import domain_changes as DC
from agents import analysis_preview as AV
from agents import design_details as DD
from agents import design_page as DP
from agents import design_review as dr
from agents import study_record as SR
from test_control_figures import _rec as _figures_record
from test_design_page import (CAVEATS, CHANGE, PLAN, PREVIEW, QUESTION, RAIN_ROW, REMARKS,
                              REVIEW, RUN, SAID, _flat, _given, _plan, _reception, _record,
                              _section, _strategy, folder)  # noqa: F401

HEADS = ["Your question", "The control run", "The new run", "How your question will be answered",
         "What the answer will not cover", "Choices made for you", "Your decision"]
FULL = ("The run multiplies rain and snow by the same factor in a made-up way. The air "
        "temperature is not changed in this run.")
CUT = "The run cuts rain and snow only, in a made-up short limit."
SETUP = [{"label": "Area", "words": "900 km2 of made-up land"},
         {"label": "Grid", "words": "40 by 30 cells of about 1 km"}]
FLOODS = {"about": "floods",
          "lead": "The control run's floods are the model's own. The table sets them beside a "
                  "made-up record.",
          "corner": "Flooded area (km2)", "columns": ["2031", "2032", "2033"],
          "rows": [{"label": "The control run", "values": [12, 7, 1234]},
                   {"label": "A made-up record", "values": [3, 40.0, 0]}]}
NOT_IN_IT = ("the observed floods: the run makes its own; flooded area over the grid, 2031 12 "
             "against 3 km2")
FRAMEWORK = ("analysis plan", "analyzer", "planning step", "coordinator", "preview",
             "rulebook", "prepared run")


# ─────────────────────────────────────────────────────────────────────────────
# MADE-UP RECORDS
# ─────────────────────────────────────────────────────────────────────────────
def _resolved(folder, runs, not_covered=True, in_full=None, not_meaning=True):
    """The server's check on disk, each change resolved as the server does,
    with its short limit, and its fuller limit where `in_full` gives one
    ({change id: [sentences]})."""
    out = []
    for r in runs:
        changes = []
        for c in r.get("changes") or []:
            rc = DC.resolve({"id": c["id"], "params": dict(c.get("params") or {})})
            got = {"id": c["id"], "meaning": rc["meaning"]}
            if not_meaning:
                got["not_meaning"] = rc["not_meaning"]
            if not_covered:
                got["not_covered"] = [CUT] if c["id"] == "weather_scale" else \
                    list(rc.get("not_covered") or [])
            if in_full and c["id"] in in_full:
                got["in_full"] = list(in_full[c["id"]])
            changes.append(got)
        out.append({"name": r["name"], "changes": changes})
    (folder / dr.RESOLVED_FILE).write_text(json.dumps({"check": {"runs": out}}))


def _control_row(st):
    return st["domain_rulebook"]["controls"]["controls"][0]


def _new_strategy(**over):
    """The Planner's record since 2026-11-13: the caveats put together by
    code, the Planner's own caveats apart (interface I4)."""
    st = _strategy(**over)
    de = st["domain_experiment"]
    nm = DC.resolve(CHANGE)["not_meaning"]
    de["question_limits"] = ["The control run holds less snow than is measured, a made-up "
                             "limit for this question."]
    de["planner_caveats"] = ["A made-up caveat of the Planner's own."]
    de["caveats"] = [f"The difference from the control run is not {nm}.",
                     "The control run does not contain the observed floods."] + \
        de["question_limits"] + de["planner_caveats"]
    de["caveats_by"] = "code"
    st["feasibility"] = {"verdict": "full",
                         "why": "The model can cut summer rain over the valley, and the control "
                                "holds the plant growth the answer needs."}
    st["goals"] = ["Compare plant growth between the drier run and the control.",
                   "Trace the change through soil water."]
    return st


def _figures(record=None, drawn=None, years=(2033,)):
    rec = record if record is not None else dict(_figures_record(), setup=copy.deepcopy(SETUP),
                                                 tables=[copy.deepcopy(FLOODS)])
    if drawn is None:
        drawn = {"series": "design_figures/control_run_series.png",
                 "maps": "design_figures/control_run_maps.png", "problems": []}
    return {"record": rec, "drawn": drawn, "years": list(years)}


def _details(folder, strategy=None, reception=None, preview=PREVIEW, record="default",
             figures=None, given=None, places=None, by_hand=True):
    strategy = strategy or _strategy()
    rec = _record(folder, strategy) if record == "default" else record
    return DD.render(reception or _reception(), strategy, run_name="a_study", preview=preview,
                     run_dir=folder, analysis_preview=rec, by_hand=by_hand,
                     control_figures=figures, run_given=given, places=places)


def _short(folder, strategy=None, reception=None, preview=PREVIEW, record="default",
           figures=None, given=None):
    strategy = strategy or _strategy()
    rec = _record(folder, strategy) if record == "default" else record
    return DP.render(reception or _reception(), strategy, run_name="a_study", preview=preview,
                     run_dir=folder, analysis_preview=rec, control_figures=figures,
                     run_given=given)


def _heads(page):
    return [ln[3:] for ln in page.splitlines() if ln.startswith("## ")]


def _fake_module(monkeypatch, name, mod):
    """A made-up module of another piece in place of `agents.<name>`; None
    makes it absent, as it is in this clone."""
    import agents
    monkeypatch.setitem(sys.modules, f"agents.{name}", mod)
    if mod is None:
        monkeypatch.delattr(agents, name, raising=False)
    else:
        monkeypatch.setattr(agents, name, mod, raising=False)


def _places_module(fetched=None, fails=False):
    """A made-up agents/places_figure (interface I3) that records its calls."""
    mod = types.ModuleType("agents.places_figure")
    mod.RECORD_FILE = "01_inputs/places.json"
    mod.PNG = "design_figures/places_reported.png"
    mod.calls = {"fetch": [], "draw": []}

    def fetch(control_name, regions):
        mod.calls["fetch"].append((control_name, list(regions)))
        return copy.deepcopy(fetched)

    def save(run_dir, record):
        p = Path(run_dir) / mod.RECORD_FILE
        if record is None:
            if p.is_file():
                p.unlink()
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record))

    def load(run_dir):
        p = Path(run_dir) / mod.RECORD_FILE
        try:
            return json.loads(p.read_text()) if p.is_file() else None
        except ValueError:
            return None

    def of_design(record, control_name, regions):
        return isinstance(record, dict) and record.get("control") == control_name and \
            set(record.get("regions") or []) == set(regions)

    def draw(record, run_dir):
        mod.calls["draw"].append(record)
        if fails:
            raise RuntimeError("a made-up failure to draw")
        return {"png": mod.PNG, "problems": []}

    def page_lines(record, drawn):
        return ([f"![The places the numbers are reported for]({drawn['png']})", ""]
                if (drawn or {}).get("png") else [])
    for f in (fetch, save, load, of_design, draw, page_lines):
        setattr(mod, f.__name__, f)
    return mod


# the store's regions with elevation bands: two bands of the vegetated cells,
# and regions that are not its bands (no comma after the label, a rule that
# names no elevation, a band of another region)
BANDS = {
    "vegetated cells": {"cells": 1100, "area_km2": 830.4, "rule": "a made-up rule"},
    "valley floor": {"cells": 300, "area_km2": 210.0, "rule": "below a made-up line"},
    "vegetated cells, below 500 m": {"cells": 400, "area_km2": 300.0,
                                     "rule": "a made-up rule, TOPO below 500 m"},
    "vegetated cells, burned": {"cells": 50, "area_km2": 40.0, "rule": "a made-up rule, burned"},
    "vegetated cellsite, below 500 m": {"cells": 5, "area_km2": 4.0,
                                        "rule": "made up, TOPO below 500 m"},
    "valley floor, below 500 m": {"cells": 200, "area_km2": 150.0,
                                  "rule": "below a made-up line, TOPO below 500 m"},
    "vegetated cells, 500 m and above": {"cells": 700, "area_km2": 530.4,
                                         "rule": "a made-up rule, TOPO 500 m and above"}}


def _split_analyses():
    """The made-up list's comparisons, the first with a map split by
    elevation band over the vegetated cells."""
    analyses = copy.deepcopy(PLAN["analyses"])
    analyses[0]["outputs"] += [
        {"id": "O2", "kind": "difference", "run": RUN, "field": "LEAFX", "period": "2033-06",
         "region": None},
        {"id": "O3", "kind": "elevation_split", "region": "vegetated cells", "inputs": ["O2"]}]
    return analyses


def _places_record(regions=("vegetated cells", "valley floor"), control="valley"):
    return {"control": control, "regions": list(regions), "lat": [1.0, 2.0],
            "lon": [10.0, 11.0], "filled": [{"label": "valley floor", "cells": 3,
                                             "area_km2": 2.1}],
            "classes": [[0, 1], [1, 1]], "outlines": [], "problems": []}


@pytest.fixture
def full(folder):
    """A study with every new record: the fuller limit of the change, the
    control run's set-up and tables, and the Planner's record of the new form."""
    _resolved(folder, [{"name": RUN, "changes": [CHANGE]}], in_full={"weather_scale": [FULL]})
    return folder


# ─────────────────────────────────────────────────────────────────────────────
# THE SHAPE
# ─────────────────────────────────────────────────────────────────────────────
class TestTheShape:

    def test_the_short_pages_sections_in_its_order_and_a_line_that_says_so(self, full):
        page = _details(full, strategy=_new_strategy(), figures=_figures(), given=_given())
        assert _heads(page) == HEADS
        assert page.startswith("# Design details: a_study\n\nThis page holds what the short "
                               "page `DESIGN_REVIEW.md` leaves out. It has the same sections "
                               "in the same order. Your decision is recorded on the short "
                               "page.\n\n## Your question\n")
        assert dr._PENDING not in page and "**Decision:**" not in page

    def test_several_runs_name_the_section_for_them(self, full):
        st = _new_strategy()
        st["domain_experiment"]["runs"].append({"name": "no_fire", "changes": [{"id": "fire_off"}]})
        assert _heads(_details(full, strategy=st, record=None))[2] == "The new runs"

    def test_a_design_of_another_kind_has_no_such_page(self):
        assert DD.render({"user_request": "q", "brief": {}}, {"archetype": "site"}) is None
        assert DD.render({"user_request": "q", "brief": {}}, {"domain_experiment": "x"}) is None

    def test_an_empty_section_is_left_out_with_its_heading(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"] = []
        page = _details(folder, strategy=st, preview=None, record=None)
        assert "## The new run" not in page and "## Your question" in page
        assert "## Your decision" in page

    @pytest.mark.parametrize("broken", [
        {"control_figures": {"record": "not a map", "drawn": 5}},
        {"control_figures": {"record": {"setup": 7, "tables": "x", "maps": 3}, "drawn": {}}},
        {"run_given": {"record": {"control": "valley", "runs": 4}, "drawn": "x"}},
        {"places": {"record": [1], "drawn": None}},
        {"preview": {"check": "x", "inputs_check": 3, "rounds": "y", "runs": 5}},
        {"analysis_preview": {"status": "planned", "plan_file": 3}}])
    def test_a_record_of_the_wrong_shape_stops_no_page(self, folder, broken):
        kw = {"run_dir": folder, "preview": PREVIEW, "analysis_preview": None}
        kw.update(broken)
        page = DD.render(_reception(), _strategy(), "a_study", **kw)
        assert page.startswith("# Design details: a_study") and "## Your decision" in page

    def test_a_study_made_before_the_new_records_gets_a_correct_page(self, folder):
        # no set-up, no tables, no fuller limit, no caveats of the Planner's own,
        # no map: each part falls back on what the older records hold
        page = _details(folder, figures=_figures(record=_figures_record()))
        assert _heads(page) == HEADS
        flat = _flat(page)
        # no set-up rows on record: no set-up table, since the short page
        # describes the control run
        assert "| | The control run |" not in page and "lightning file" not in page
        assert "The difference from the control run is not " in flat
        for c in CAVEATS:
            assert c in _flat(_short(folder))                    # the short page keeps them


# ─────────────────────────────────────────────────────────────────────────────
# YOUR QUESTION, AND THE DIRECT ANSWER ON THE SHORT PAGE
# ─────────────────────────────────────────────────────────────────────────────
VERDICTS = [
    ("full", "This design can answer the question in full."),
    ("partial", "This design can answer part of the question."),
    ("infeasible", "This design cannot answer the question."),
    ("", "The Planner gave no verdict on whether this design can answer the question."),
    ("maybe", "The Planner gave no verdict on whether this design can answer the question.")]


class TestYourQuestion:

    def test_the_question_the_answer_and_the_steps(self, full):
        sec = _section(_details(full, strategy=_new_strategy()), "Your question")
        assert sec.strip().splitlines() == [
            QUESTION, "",
            "**The Planner's answer.** This design can answer the question in full. The model "
            "can cut summer rain over the valley, and the control run holds the plant growth "
            "the answer needs.", "",
            "The Planner reads the question in two steps.", "",
            "1. Compare plant growth between the drier run and the control run.",
            "2. Trace the change through soil water."]

    def test_one_step_or_none(self, folder):
        st = _strategy(goals=["only one"])
        assert "The Planner reads the question in one step.\n\n1. Only one." in \
            _details(folder, strategy=st)
        st = _strategy(goals=[])
        assert "reads the question in" not in _details(folder, strategy=st)

    @pytest.mark.parametrize("bad", [{"text": "a made-up reason ZQ"}, ["a made-up reason ZQ"],
                                     ("a made-up reason ZQ",)])
    def test_a_reason_or_a_step_of_the_wrong_shape_is_left_out(self, folder, bad):
        # never printed as Python text, on either page (2026-11-13)
        st = _strategy(feasibility={"verdict": "full", "why": bad},
                       goals=[bad, "a made-up step", 7, None])
        for page in (_short(folder, strategy=st), _details(folder, strategy=st)):
            assert "**The Planner's answer.** This design can answer the question in full.\n" \
                in page
            assert "made-up reason ZQ" not in page and "{" not in page
        assert "The Planner reads the question in one step.\n\n1. A made-up step.\n" in \
            _details(folder, strategy=st)
        # steps that are not a list are left out whole
        st = _strategy(goals={"text": "a made-up step ZQ"})
        page = _details(folder, strategy=st)
        assert "reads the question in" not in page and "made-up step ZQ" not in page

    def test_changes_asked_at_the_short_page_are_kept(self, folder):
        asked = QUESTION + dr.REVISION_MARK + "make it 2034"
        sec = _section(_details(folder, reception=_reception(asked)), "Your question")
        assert sec.lstrip().startswith(
            QUESTION + "\n\n**Change 1 you asked for at the short page.** make it 2034")

    @pytest.mark.parametrize("verdict, words", VERDICTS)
    def test_each_verdict_in_words_on_both_pages(self, folder, verdict, words):
        st = _strategy(feasibility={"verdict": verdict, "why": "a made-up reason"})
        line = f"**The Planner's answer.** {words} A made-up reason."
        short = _short(folder, strategy=st, record=None)
        assert short.count("**The Planner's answer.**") == 1
        assert short.count("A made-up reason.") == 1
        assert line in short and line in _details(folder, strategy=st, record=None)

    def test_the_answer_is_directly_under_the_question(self, folder):
        lines = _short(folder).splitlines()
        k = lines.index(f"**Your question.** {QUESTION}")
        assert lines[k + 2] == ("**The Planner's answer.** This design can answer the question "
                                "in full. The change and the fields are there.")
        assert lines[k + 4] == "## The control run"
        # after each change asked at the page, where there is one
        asked = QUESTION + dr.REVISION_MARK + "make it 2034"
        lines = _short(folder, reception=_reception(asked)).splitlines()
        k = lines.index("**Change 1 you asked for at this page.** make it 2034")
        assert lines[k + 2].startswith("**The Planner's answer.**")

    def test_no_reason_on_record_gives_the_verdict_alone(self, folder):
        st = _strategy(feasibility={"verdict": "partial"})
        assert "**The Planner's answer.** This design can answer part of the question.\n" in \
            _short(folder, strategy=st)
        assert "**The Planner's answer.** " + VERDICTS[3][1] + "\n" in \
            _short(folder, strategy=_strategy(feasibility="not a map"))

    def test_the_verdict_is_said_nowhere_else_on_the_short_page(self, folder):
        for v in ("partial", "infeasible", ""):
            st = _strategy(feasibility={"verdict": v, "why": "a made-up reason"})
            page = _short(folder, strategy=st, record=None)
            assert "judges" not in page and "no verdict" not in page.split("## ")[-1]
            assert page.count("gave no verdict") <= 1

    def test_what_the_design_requires_stays_on_the_short_page(self, folder):
        st = _strategy(requires=["a made-up river course"])
        sec = _section(_short(folder, strategy=st), "What the answer will not cover")
        assert "- The question would also need something that is not available here. A " \
               "made-up river course." in sec
        assert "made-up river course" not in _details(folder, strategy=st)


# ─────────────────────────────────────────────────────────────────────────────
# THE CONTROL RUN
# ─────────────────────────────────────────────────────────────────────────────
class TestTheControlRun:

    def test_the_set_up_the_maps_and_the_measurements(self, full):
        sec = _section(_details(full, figures=_figures()), "The control run")
        assert sec.lstrip().startswith(
            "| | The control run |\n|---|---|\n| Area | 900 km2 of made-up land |\n| Grid | 40 "
            "by 30 cells of about 1 km |\n")
        assert "![Maps of the control run's mean](design_figures/control_run_maps.png)" in sec
        assert "Each map is the average of 2031 to 2033 over the model's grid. The line is the " \
               "edge of the region `valley floor`." in sec
        assert "The figure on the short page sets the control run beside these measurements." \
            in sec
        rows = [ln for ln in sec.splitlines() if ln.startswith("| ") and "---" not in ln]
        assert rows[3:] == [
            "| Quantity | Measurement | Compared with |",
            "| Snow at two made-up pillows | a made-up network | `SNOWX`, the monthly mean in "
            "mm, over the 2 cells that hold the points |",
            "| Runoff and river flow | a made-up gauge | `RUNX`, the monthly total in mm/month, "
            "over valley floor |",
            "| Runoff and river flow | a made-up estimate | `RUNX`, the monthly total in "
            "mm/month, over valley floor |"]
        # how it compares in words is the short page's figure's, so not here
        assert "Against measurements" not in sec

    def test_without_set_up_rows_there_is_no_set_up_table(self, folder):
        # the facts' own sentences name files and raw settings, and the short
        # page already describes the control run (2026-11-13)
        st = _strategy()
        _control_row(st)["facts"]["grid"]["sentence"] = "40 by 30 made-up cells"
        for figures in (None, _figures(record=_figures_record()),
                        _figures(record=dict(_figures_record(), setup=[]))):
            sec = _section(_details(folder, strategy=st, figures=figures), "The control run")
            assert "| | The control run |" not in sec, figures
            for gone in ("| Years |", "| Grid |", "| Fire |", "lightning file", "made-up cells"):
                assert gone not in sec, (gone, figures)
        assert _section(_details(folder, strategy=st), "The control run").strip().splitlines() \
            == ["**Against measurements.** It holds half of the measured snow. The source is a "
                "note."]

    def test_a_set_up_row_of_the_wrong_form_is_left_out(self, folder):
        rec = dict(_figures_record(), setup=[{"label": "Area"}, "x", {"label": "Grid",
                                                                      "words": "made up"}])
        sec = _section(_details(folder, figures=_figures(record=rec)), "The control run")
        assert "| Grid | made up |" in sec and "| Area |" not in sec

    def test_a_bar_inside_a_table_cell_does_not_break_the_table(self, folder):
        rec = dict(_figures_record(), setup=[{"label": "Grid", "words": "40 | 30 made-up cells"}])
        sec = _section(_details(folder, figures=_figures(record=rec)), "The control run")
        assert "| Grid | 40 / 30 made-up cells |" in sec
        assert DD._cell("a | b\n c") == "a / b c"

    def test_figures_not_drawn_say_why_and_leave_the_tables(self, folder):
        drawn = {"series": None, "maps": None, "problems": ["the maps were not drawn: made up"]}
        sec = _section(_details(folder, figures=_figures(drawn=drawn)), "The control run")
        assert "![" not in sec and "| Area |" in sec
        assert "The model server holds these measurements beside the control run." in sec
        assert "- The maps were not drawn: made up." in sec
        # with no series on the short page, the words on how it compares
        assert "**Against measurements.**" in sec


# ─────────────────────────────────────────────────────────────────────────────
# THE NEW RUN
# ─────────────────────────────────────────────────────────────────────────────
class TestTheNewRun:

    def _table(self, sec):
        return [ln for ln in sec.splitlines() if ln.startswith("|")]

    def test_one_run_is_one_small_table(self, full):
        given = _given()
        given["record"]["runs"][0]["map"] = {"outline": [[1, 0], [1, 1]]}
        sec = _section(_details(full, given=given), "The new run")
        assert self._table(sec) == [
            "| | The new run |", "|---|---|",
            f"| Name | `{RUN}` |",
            "| Starts from | the control run's saved state of 1 January 2031 |",
            "| Simulates | 2031 to 2033 |",
            "| Change | rain and snow multiplied by 0.7 in June to August of 2033 |",
            "| Where | everywhere, 3 grid cells |",
            "| Everything else | as in the control run |",
            "| Cost | about 2.4 node-hours, as the model server estimates |"]
        assert sec.rstrip().endswith("The steps to make this run without the framework are in "
                                     "`RUN_BY_HAND.md`.")

    def test_a_knob_whose_name_begins_with_capitals_keeps_them(self, full):
        co2 = {"knob": "CO2", "change": "set to 560 ppm", "when": "every year of the run",
               "where": "everywhere"}
        run = dict(_given()["record"]["runs"][0], rows=[co2, dict(RAIN_ROW)])
        sec = _section(_details(full, given=_given(run)), "The new run")
        assert "| Change | CO2 set to 560 ppm in every year of the run and rain and snow " \
               "multiplied by 0.7 in June to August of 2033 |" in sec
        assert DD._lower_first("CO2") == "CO2" and DD._lower_first("Rain and snow") == \
            "rain and snow"

    def test_without_the_record_the_change_is_the_servers_words(self, full):
        sec = _flat(_section(_details(full), "The new run"))
        meaning = DC.resolve(CHANGE)["meaning"].replace("the control", "the control run")
        assert f"| Change | {DP._sentence(meaning)} |" in sec
        assert "| Where |" not in sec

    def test_several_runs_a_table_each_and_the_cost_once(self, full):
        st = _strategy()
        st["domain_experiment"]["runs"].append({"name": "no_fire", "changes": [{"id": "fire_off"}]})
        sec = _section(_details(full, strategy=st, record=None, by_hand=False), "The new runs")
        assert f"| | `{RUN}` |" in sec and "| | `no_fire` |" in sec
        assert "| Name |" not in sec and "| Cost |" not in sec
        assert sec.rstrip().endswith("The model server estimates the cost at about 2.4 "
                                     "node-hours for the runs.")
        assert "RUN_BY_HAND" not in sec

    def test_a_run_with_no_change_and_one_that_makes_the_control_run_again(self, full):
        st = _strategy()
        st["domain_experiment"].update(runs=[{"name": "same", "changes": []}],
                                       control={"name": "valley", "reuse": False})
        sec = _section(_details(full, strategy=st, record=None), "The new run")
        assert "| Change | none, so the run repeats the control run |" in sec
        assert "Starts from" not in sec
        assert "for the runs and the control run." in sec

    def test_the_control_run_alone_has_no_such_section(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"] = []
        assert "## The new run" not in _details(folder, strategy=st, record=None)


# ─────────────────────────────────────────────────────────────────────────────
# HOW YOUR QUESTION WILL BE ANSWERED
# ─────────────────────────────────────────────────────────────────────────────
ANSWER = "How your question will be answered"


class TestHowYourQuestionWillBeAnswered:

    def test_every_comparison_with_a_number_in_the_short_pages_columns(self, folder):
        sec = _section(_details(folder), ANSWER)
        assert sec.lstrip().startswith(
            "We will compare the new run with the control run for 2033. This table gives each "
            "comparison of the short page a number.\n\n")
        rows = [ln for ln in sec.splitlines() if ln.startswith("| ")]
        assert rows[0] == "| | What we compare | When | How | What it tells us |"
        assert rows[1].startswith("| 1 | Monthly change in plant growth | June to August of "
                                  "2033 | the new run minus the control run |")
        assert rows[2].startswith("| 2 | LEAFX and SOILX | each day of 2031 to 2033 | the two "
                                  "runs as daily lines |")

    def test_more_than_the_short_page_lists(self, folder):
        a = PLAN["analyses"][0]
        _plan(folder, analyses=[dict(copy.deepcopy(a), id=f"P{i}", title=f"Made-up {i}")
                                for i in range(1, 7)])
        sec = _section(_details(folder), ANSWER)
        assert "The short page lists the first four comparisons, and this table lists all six." \
            in sec
        _plan(folder, analyses=[dict(copy.deepcopy(a), id=f"P{i}", title=f"Made-up {i}",
                                     main=i in (2, 5)) for i in range(1, 7)])
        assert "The short page lists two comparisons, and this table lists all six." in \
            _section(_details(folder), ANSWER)
        assert "| 6 | Made-up 6 |" in sec

    def test_a_conditional_and_a_blocked_comparison_by_number(self, folder):
        _plan(folder, problems=["P1: no catalog field 'NOPE'"])
        sec = _flat(_section(_details(folder, record=_record(folder, status=AV.NOT_PASSED)),
                             ANSWER))
        assert "Comparison 1 cannot be made as planned. No catalog field 'NOPE'. Comparison 2 " \
               "gives figures and no number that the report may state, because each of its " \
               "outputs is a daily line." in sec

    def test_what_a_comparison_with_no_number_gives(self):
        def a(*outs):
            return {"outputs": list(outs)}
        day = {"kind": "series", "step": "day"}
        assert DD._no_number_kind(a(day, day)) == "a daily line"
        assert DD._no_number_kind(a({"kind": "map"}, {"kind": "difference"})) == "a map"
        assert DD._no_number_kind(a(day, {"kind": "map"})) == "a map or a daily line"
        assert DD._no_number_kind(None) == "a map or a daily line"

    def test_the_places_with_a_map_where_one_was_drawn(self, folder, monkeypatch):
        _fake_module(monkeypatch, "places_figure", _places_module())
        places = {"record": _places_record(), "drawn": {"png": "design_figures/p.png"}}
        sec = _section(_details(folder, places=places), ANSWER)
        assert "**Where the numbers are reported.** The comparisons read these places and no " \
               "others." in sec
        assert "| Place | Area (km2) | Cells | Which cells belong to it |" in sec
        assert "![The places the numbers are reported for](design_figures/p.png)" in sec
        assert sec.index("| `valley floor` |") < sec.index("![The places")
        # without the module, or with no map, the table alone
        _fake_module(monkeypatch, "places_figure", None)
        sec = _section(_details(folder, places=places), ANSWER)
        assert "| `valley floor` |" in sec and "![" not in sec
        assert "![" not in _section(_details(folder), ANSWER)

    def test_a_blank_line_follows_the_places_figure(self, monkeypatch):
        # the figure is the last line of its paragraph, whatever page_lines gives
        mod = _places_module()
        mod.page_lines = lambda record, drawn: [f"![The places]({drawn['png']})"]
        _fake_module(monkeypatch, "places_figure", mod)
        places = {"record": _places_record(), "drawn": {"png": "f/p.png"}}
        assert DD.places_figure_lines(places) == ["![The places](f/p.png)", ""]
        mod.page_lines = lambda record, drawn: [f"![The places]({drawn['png']})", ""]
        assert DD.places_figure_lines(places) == ["![The places](f/p.png)", ""]

    def test_the_elevation_bands_a_comparison_splits_by_are_places(self, folder,
                                                                   monkeypatch):
        # one comparison splits a region by elevation band and names the region
        # alone: the store's bands of that region are listed and mapped as well
        rec = _record(folder, regions=BANDS)
        assert "500 m" not in _section(_details(folder, record=rec), ANSWER)   # no split
        _plan(folder, analyses=_split_analyses())
        sec = _section(_details(folder, record=rec), ANSWER)
        rows = [ln for ln in sec.splitlines() if ln.startswith("| `")]
        assert rows == [
            "| `vegetated cells` | 830 | 1,100 | a made-up rule |",
            "| `vegetated cells, below 500 m` | 300 | 400 | a made-up rule, TOPO below 500 m |",
            "| `vegetated cells, 500 m and above` | 530 | 700 | a made-up rule, TOPO 500 m and "
            "above |",
            "| `valley floor` | 210 | 300 | below a made-up line |"]
        assert DD.plan_regions(rec, folder, rec["design_sha256"]) == [
            "vegetated cells", "vegetated cells, below 500 m", "vegetated cells, 500 m and above",
            "valley floor"]
        # the map is asked for with the bands, so it fills them inside the outline
        mod = _places_module(fetched=_places_record())
        _fake_module(monkeypatch, "places_figure", mod)
        dr.write_review(folder, reception=_reception(), strategy=_strategy(), preview=PREVIEW,
                        analysis_preview=rec)
        assert mod.calls["fetch"] == [("valley", DD.plan_regions(rec, folder,
                                                                 rec["design_sha256"]))]

    def test_the_normal_years_where_years_are_set_against_them(self, folder):
        sec = _flat(_section(_details(folder), ANSWER))
        assert "**The normal years.** The comparisons set years against the normal years " \
               "2032 to 2034, the control run's own, since the Planner named none. The normal " \
               "years include 2033, a year this plan sets against them" in sec
        _plan(folder, comparisons=[PLAN["comparisons"][0]],
              baseline={"normal_years": [2032, 2033, 2034]})
        sec = _flat(_section(_details(folder), ANSWER))
        assert "**The normal years.**" not in sec
        assert "spread from year to year over 2032 to 2034." in sec

    def test_what_the_comparisons_read_in_one_line(self, folder):
        sec = _section(_details(folder), ANSWER)
        assert "**What the comparisons read.** The two comparisons read two outputs of the " \
               "model. They are `LEAFX` and `SOILX`." in sec
        _plan(folder, analyses=[dict(copy.deepcopy(PLAN["analyses"][0]),
                                     fields=[f"F{i}" for i in range(13)])])
        assert "The comparison reads 13 outputs of the model." in \
            _section(_details(folder), ANSWER)

    def test_other_explanations_a_first_look_and_the_checks(self, folder):
        sec = _section(_details(folder), ANSWER)
        assert "| Explanation | Why it could matter |\n|---|---|\n| a nitrogen limit | " \
               "Nitrogen, not water, sets growth. |" in sec
        assert "**A first look at every quantity.** Every quantity is screened over " \
               "`vegetated cells` for 2033 before the comparisons. The report marks anything " \
               "that only this screen finds as found by screening and not as tested." in sec
        assert "**Checks on this list.** Code checked the Planner's list, and a second model " \
               "read it again against your question. Both passed." in sec
        assert f"The review made these remarks.\n\n- {REMARKS[0]}" in sec
        assert "The new run does not exist yet, so the list assumes that it will hold the same " \
               "quantities as the control run. Each year is checked after it runs. The full " \
               "list is in this study's folder, at `plan/analysis_plan.json`." in sec
        assert f"`{RUN}` is not made yet" not in sec

    def test_the_checks_in_one_paragraph_for_one_run_several_and_none(self, folder):
        (folder / "plan" / "analysis_plan_review.json").write_text(
            json.dumps(dict(REVIEW, model_problems=[])))
        one = ("**Checks on this list.** Code checked the Planner's list, and a second model "
               "read it again against your question. Both passed. The review made no remark. ")
        where = "The full list is in this study's folder, at `plan/analysis_plan.json`."
        assert one + "The new run does not exist yet, so the list assumes that it will hold " \
                     "the same quantities as the control run. Each year is checked after it " \
                     "runs. " + where in _section(_details(folder), ANSWER)
        st = _strategy()
        st["domain_experiment"]["runs"].append({"name": "no_fire", "changes": [{"id": "fire_off"}]})
        _plan(folder, comparisons=PLAN["comparisons"] + [
            dict(PLAN["comparisons"][0], id="C3", run="no_fire")])
        assert one + "The new runs do not exist yet, so the list assumes that they will hold " \
                     "the same quantities as the control run. Each year is checked after it " \
                     "runs. " + where in _section(_details(folder, strategy=st), ANSWER)
        # a list that reads the control run alone has no such sentence
        st["domain_experiment"]["runs"] = []
        _plan(folder, comparisons=[PLAN["comparisons"][1]])
        assert one + where in _section(_details(folder, strategy=st), ANSWER)
        assert "exist yet" not in _details(folder, strategy=st)

    def test_a_list_rewritten_once_and_one_that_did_not_pass(self, folder):
        (folder / "plan" / "analysis_plan_review.json").write_text(
            json.dumps(dict(REVIEW, model_problems=[])))
        sec = _section(_details(folder, record=_record(folder, attempts=2)), ANSWER)
        assert "Both passed on the list's second writing. The review made no remark." in sec
        bad = _record(folder, status=AV.NOT_PASSED, why="the analysis plan did not pass step "
                                                        "3's review: a made-up reason")
        sec = _section(_details(folder, record=bad), ANSWER)
        assert "The list did not pass. The review made no remark." in sec
        # the record's own summary repeats what is listed, in the framework's words
        assert "made-up reason" not in sec and "step 3" not in sec
        (folder / "plan" / "analysis_plan_review.json").write_text(
            json.dumps(dict(REVIEW, follows=False, model_problems=[])))
        sec = _section(_details(folder, record=dict(bad, attempts=2)), ANSWER)
        assert "The list did not pass on its second writing. The review found that it does " \
               "not follow from the question." in sec

    @pytest.mark.parametrize("given, said", [
        ("the Planner", "2032 to 2034, chosen by the Planner."),
        ("the control's own (the Planner named none)",
         "2032 to 2034, the control run's own, since the Planner named none."),
        ("none named", "2032 to 2034."),
        ("a made-up source", "2032 to 2034, a made-up source.")])
    def test_where_the_normal_years_come_from(self, folder, given, said):
        plan = copy.deepcopy(PLAN)
        _plan(folder, baseline=dict(plan["baseline"], normal_years_from=given))
        assert f"The comparisons set years against the normal years {said}" in \
            _flat(_section(_details(folder), ANSWER))

    @pytest.mark.parametrize("record", [
        None, {"status": AV.FAILED, "why": "a made-up error"},
        {"status": AV.NOT_PLANNED, "why": "no store"}, "other"])
    def test_no_list_shown_gives_the_designs_places(self, folder, record):
        if record == "other":
            record = _record(folder, design_sha256="0" * 64)
        st = _strategy()
        st["domain_experiment"]["analysis"]["regions"] = [
            {"name": "valley floor", "within": "the box", "why": "the control holds it"}]
        st["domain_experiment"]["analysis"]["normal_years"] = [2031, 2032]
        _control_row(st)["normal_years"] = [2032, 2033, 2034]
        sec = _section(_details(folder, strategy=st, record=record), ANSWER)
        assert "| `valley floor` within `the box` | The control run holds it. |" in sec
        st["domain_experiment"]["analysis"]["regions"] = ["valley floor", {"name": "slopes"}]
        assert "**Where the numbers are reported.** The Planner chose `valley floor` and " \
               "`slopes`." in _section(_details(folder, strategy=st, record=record), ANSWER)
        assert "**The normal years.** The design compares years with the normal years 2031 " \
               "and 2032. The control run's own normal years are 2032 to 2034, so the numbers " \
               "will differ from those of studies that used them." in _flat(sec)
        for gone in ("| 1 |", "Checks on this list", "LEAFX"):
            assert gone not in sec


# ─────────────────────────────────────────────────────────────────────────────
# WHAT THE ANSWER WILL NOT COVER
# ─────────────────────────────────────────────────────────────────────────────
COVER = "What the answer will not cover"


class TestWhatTheAnswerWillNotCover:

    def test_the_fuller_limit_and_the_controls_table(self, full):
        sec = _section(_details(full, strategy=_new_strategy(), figures=_figures()), COVER)
        assert sec.strip().splitlines() == [
            "The short page gives each limit in a sentence or two. This section gives more on "
            "them.", "",
            f"**The change in `{RUN}`.** {FULL}", "",
            "**The observed floods.** The control run's floods are the model's own. The table "
            "sets them beside a made-up record.", "",
            "| Flooded area (km2) | 2031 | 2032 | 2033 |", "|---|---|---|---|",
            "| The control run | 12 | 7 | 1,234 |", "| A made-up record | 3 | 40 | 0 |"]

    def test_a_change_with_no_fuller_limit_gives_its_not_meaning(self, folder):
        _resolved(folder, [{"name": RUN, "changes": [CHANGE]}])
        nm = DC.resolve(CHANGE)["not_meaning"].replace("the control", "the control run")
        assert f"**The change in `{RUN}`.** The difference from the control run is not {nm}." \
            in _flat(_section(_details(folder), COVER))

    def test_one_text_for_two_runs_is_said_once_and_two_changes_by_name(self, full):
        st = _new_strategy()
        st["domain_experiment"]["runs"] = [
            {"name": "a", "changes": [copy.deepcopy(CHANGE)]},
            {"name": "b", "changes": [copy.deepcopy(CHANGE), {"id": "fire_off"}]}]
        _resolved(full, st["domain_experiment"]["runs"], in_full={"weather_scale": [FULL]})
        sec = _section(_details(full, strategy=st, record=None), COVER)
        assert sec.count(FULL) == 1
        assert f"**The change `weather_scale` in `a` and `b`.** {FULL}" in sec
        assert "**The change `fire_off` in `b`.** The difference from the control run is not " \
               "the effect of an observed fire." in sec

    def test_a_table_of_the_wrong_shape_is_left_out_and_its_sentence_kept(self, folder):
        st = _strategy()
        _control_row(st)["not_in_it"] = [NOT_IN_IT]
        bad = dict(FLOODS, rows=[{"label": "x", "values": [1]}])
        sec = _section(_details(folder, strategy=st,
                                figures=_figures(record=dict(_figures_record(), tables=[bad]))),
                       COVER)
        assert "| Flooded area" not in sec
        assert "**The observed floods.** Flooded area over the grid, 2031 12 against 3 km2." in sec
        # a printed table of the same subject stands in for the sentence
        sec = _section(_details(folder, strategy=st, figures=_figures()), COVER)
        assert "2031 12 against 3 km2" not in sec and "| Flooded area (km2) |" in sec

    def test_an_item_the_short_page_has_no_words_for_is_kept_whole(self, folder):
        st = _strategy()
        _control_row(st)["not_in_it"] = ["the made-up glaciers"]
        assert "The control run does not contain the made-up glaciers." in \
            _section(_details(folder, strategy=st), COVER)

    def test_the_planners_caveats_of_an_older_record(self, folder):
        # every change has its short limit, so the short page leaves the caveats here,
        # less the server's own limit copied into them and the limits for the question
        _resolved(folder, [{"name": RUN, "changes": [CHANGE]}])
        st = _strategy()
        nm = DC.resolve(CHANGE)["not_meaning"]
        st["domain_experiment"]["question_limits"] = ["A made-up limit of the question."]
        st["domain_experiment"]["caveats"] = [nm] + CAVEATS + ["A made-up limit of the question."]
        sec = _section(_details(folder, strategy=st), COVER)
        assert "**The Planner's other limits.** The Planner also wrote these limits, which " \
               "the short page leaves out.\n\n" + \
               "\n".join(f"- {c}" for c in CAVEATS) in sec
        assert sec.count("consistent change of climate") == 1
        assert "made-up limit of the question" not in sec
        assert not any(c in _short(folder, strategy=st) for c in CAVEATS)

    def test_a_record_of_the_new_form_prints_the_servers_limits_from_its_records(self, full):
        st = _new_strategy()
        short = _section(_short(full, strategy=st), COVER)
        details = _section(_details(full, strategy=st), COVER)
        assert short.strip().splitlines() == [
            f"- {CUT}",
            f"- {st['domain_experiment']['question_limits'][0]} {DP.JUDGEMENT}",
            f"- A made-up caveat of the Planner's own. {DP.JUDGEMENT}",
            "- The control run does not contain the observed floods: the run makes its own."]
        assert "consistent change of climate" not in short + details
        assert "Planner's other limits" not in details
        # a change with no short limit of its own: still never the caveats
        _resolved(full, [{"name": RUN, "changes": [CHANGE]}], not_covered=False)
        short = _section(_short(full, strategy=st), COVER)
        assert "consistent change of climate" not in short
        assert "The difference from the control run is not a consistent change of climate" in \
            _flat(_section(_details(full, strategy=st), COVER))

    def test_the_planners_own_caveats_are_said_once(self):
        de = {"question_limits": ["A.", {"text": "B."}, None, "A."],
              "planner_caveats": ["b", "C."], "caveats_by": "code"}
        assert DP.planner_limits(de) == ["A.", "B.", "C."]
        assert DP.planner_limits(dict(de, caveats_by=None)) == ["A.", "B."]


# ─────────────────────────────────────────────────────────────────────────────
# CHOICES MADE FOR YOU
# ─────────────────────────────────────────────────────────────────────────────
CHOICES = "Choices made for you"


class TestChoicesMadeForYou:

    def test_the_servers_check_as_a_table(self, folder):
        sec = _section(_details(folder), CHOICES)
        assert sec.lstrip().startswith("The short page lists the choices. The model server's check "
                              "stands behind them.\n\n| The model server checked | Result |")
        rows = [ln for ln in sec.splitlines() if ln.startswith("| ") and "---" not in ln][1:]
        assert rows == [
            "| The design against its list of changes and the control run's facts | passed |",
            "| The run's recipe, its checks and its job scripts | built, and nothing was "
            "submitted |",
            "| The surface data, the grid file and the starting state, as one set | passed, "
            "1,200 cells |",
            f"| A note on run {RUN}, change weather_scale | A made-up thing to note. |"]

    def test_a_refusal_an_error_and_a_failed_input_check(self, folder):
        refused = {"refused": [{"where": "run x", "what": "a made-up refusal"}], "runs": []}
        assert "| The design against its list of changes and the control run's facts | " \
               "refused. Run x: a made-up refusal. |" in _details(folder, preview=refused)
        error = {"refused": [], "runs": [], "error": "a made-up error"}
        assert "| The design, built before this page | not built. A made-up error. |" in \
            _details(folder, preview=error)
        bad = dict(PREVIEW, inputs_check={"pass": False, "reasons": ["a file is gone"]})
        assert "| failed. A file is gone. |" in _details(folder, preview=bad)
        assert "## Choices made for you" not in _details(folder, preview=None, record=None,
                                                         strategy=_strategy(
                                                             domain_experiment=dict(
                                                                 _strategy()["domain_experiment"],
                                                                 analysis={})))

    def test_each_earlier_version_with_its_findings_whole(self, folder):
        long = "a made-up reason " + "that goes on " * 30 + "to its end"
        preview = dict(PREVIEW, rounds=[
            {"findings": [{"where": "run x", "what": long}], "design": "strategy.refused_1.json"},
            {"findings": ["one", "two"], "design": "strategy.refused_2.json"}])
        sec = _section(_details(folder, preview=preview), CHOICES)
        assert "**Earlier versions.** The model server refused 2 earlier versions of this " \
               "design, and the Planner changed it each time before this page." in sec
        assert "- The first version, kept as `strategy.refused_1.json`, was refused for this " \
               f"reason.\n  - Run x: {long}." in sec
        assert "- The second version, kept as `strategy.refused_2.json`, was refused for these " \
               "reasons.\n  - One.\n  - Two." in sec

    def test_the_tenth_version_is_the_last_with_a_word(self, folder):
        rounds = [{"findings": [f"made-up reason {i}"]} for i in range(1, 12)]
        sec = _section(_details(folder, preview=dict(PREVIEW, rounds=rounds)), CHOICES)
        assert "- The tenth version was refused for this reason.\n  - Made-up reason 10." in sec
        assert "- The number 11 version was refused" in sec
        assert DD._ordinal(1) == "first" and DD._ordinal(10) == "tenth"
        assert DD._ordinal(11) == "number 11" and DD._ordinal(0) == "number 0"

    def test_how_the_question_is_read(self, folder):
        assert DD.QUESTION_TYPE["exploratory"] in _section(_details(folder), CHOICES)
        st = _strategy()
        st["domain_experiment"]["analysis"]["question_type"] = "something"
        assert DD.NO_QUESTION_TYPE in _section(_details(folder, strategy=st), CHOICES)

    def test_a_remark_that_repeats_the_short_page_is_on_neither_page(self, folder):
        # the server's sentence on the short page makes its point, and the
        # remark would say it again in Reception's words (2026-11-13)
        same = "The question says rainfall. The change multiplies rain and snow together."
        rec = _reception(run_settings={"conflicts": [same, "another made-up remark"]})
        details, short = _details(folder, reception=rec), _short(folder, reception=rec)
        for page in (details, short):
            assert "says rainfall. The change" not in page
        assert "Reception" not in details
        assert SAID in _flat(_section(short, CHOICES))
        # a remark that makes another point stays on the short page
        assert "- Another made-up remark." in short
        assert "another made-up remark" not in details.lower()

    def test_with_no_choice_on_the_short_page_the_check_is_said_plainly(self, folder):
        rec = _reception("What if summer precipitation had been lower in 2033?",
                         run_settings={}, known_data_gaps=[])
        preview = dict(PREVIEW, check={})
        assert "No choice was made for you." in _short(folder, reception=rec, preview=preview)
        sec = _section(_details(folder, reception=rec, preview=preview), CHOICES)
        assert sec.lstrip().startswith("The model server checked the design before the short "
                                       "page was written.")


# ─────────────────────────────────────────────────────────────────────────────
# YOUR DECISION
# ─────────────────────────────────────────────────────────────────────────────
DECISION = "Your decision"
BOUND = {"promised": True, "bound": True, "words": "x"}


class TestYourDecision:

    def test_one_run_whose_comparisons_a_yes_approves(self, folder):
        sec = _section(_details(folder, record=_record(folder, handoff=BOUND)), DECISION)
        assert sec.strip().splitlines() == [
            "The model server checks the design and builds the run again from the same "
            "records after a yes. The run is then submitted as a chain of jobs, one for each "
            "year, and each year is checked before the next one starts. `RUN_BY_HAND.md` lists "
            "the jobs and their checks.", "",
            "The analysis carries out the two comparisons as they are listed here once the chain "
            "ends. The analysis stops, and its report says why, if a comparison cannot be "
            "carried out as listed.", "",
            "A no uses no computing time, and the records stay on disk."]

    def test_the_comparisons_a_yes_does_not_approve(self, folder):
        assert "The comparisons are chosen again once the chain ends, and they may differ from " \
               "this list." in _section(_details(folder), DECISION)
        kept = _record(folder, handoff={"promised": True, "words": "x"})
        assert "The record of the yes names the runs and the years and not this list." in \
            _flat(_section(_details(folder, record=kept), DECISION))
        assert "The comparisons are chosen once the chain ends." in \
            _section(_details(folder, record=None), DECISION)

    def test_several_runs_and_the_control_run_alone(self, folder):
        st = _strategy()
        st["domain_experiment"]["runs"].append({"name": "no_fire", "changes": [{"id": "fire_off"}]})
        sec = _flat(_section(_details(folder, strategy=st, record=None, by_hand=False), DECISION))
        assert sec.startswith("The model server checks the design and builds the runs again "
                              "from the same records after a yes. Each run is then submitted")
        assert "RUN_BY_HAND" not in sec and "once the chains end." in sec
        st["domain_experiment"]["runs"] = []
        sec = _flat(_section(_details(folder, strategy=st, preview=None,
                                      record=_record(folder, st, handoff=BOUND)), DECISION))
        assert sec.startswith("No model run is made after a yes. The control run's existing "
                              "output is read. The analysis carries out the two comparisons as "
                              "they are listed here after your yes.")

    def test_one_new_run_and_the_control_run_made_again(self, folder):
        # two runs are made: the new run and the control run again
        st = _strategy()
        st["domain_experiment"]["control"] = {"name": "valley", "reuse": False}
        sec = _flat(_section(_details(folder, strategy=st, record=None), DECISION))
        assert sec == ("The model server checks the design and builds the runs again from the "
                       "same records after a yes. Each run is then submitted as a chain of jobs, "
                       "one for each year, and each year is checked before the next one starts. "
                       "`RUN_BY_HAND.md` lists the jobs and their checks. The comparisons are "
                       "chosen once the chains end. A no uses no computing time, and the "
                       "records stay on disk.")
        assert "| yes | The runs are submitted. They may use up to 2.4 node-hours. |" in \
            _short(folder, strategy=st, record=None)

    @pytest.mark.parametrize("why", ["refused", "error", "infeasible", "not passed"])
    def test_a_design_that_cannot_run(self, folder, why):
        st, preview, rec = _strategy(), PREVIEW, None
        if why == "refused":
            preview = {"refused": [{"where": "x", "what": "y"}], "runs": []}
        elif why == "error":
            preview = {"refused": [], "runs": [], "error": "z"}
        elif why == "infeasible":
            st = _strategy(feasibility={"verdict": "infeasible", "why": "w"})
        else:
            rec = _record(folder, status=AV.NOT_PASSED, handoff={"promised": False, "why": "x"})
        assert _flat(_section(_details(folder, strategy=st, preview=preview, record=rec),
                              DECISION)) == ("Nothing can run until the design changes. "
                                             "Nothing has been submitted, and the records "
                                             "stay on disk.")


# ─────────────────────────────────────────────────────────────────────────────
# NOTHING A DECISION RESTS ON IS LOST BETWEEN THE TWO PAGES
# ─────────────────────────────────────────────────────────────────────────────
def _case(folder, kind):
    """(the arguments of both pages, the text that must be on one of them,
    which page holds it) for one row of the table in the builder's notes."""
    st, rec, pv, kw = _strategy(), "default", copy.deepcopy(PREVIEW), {}
    de = st["domain_experiment"]
    row = _control_row(st)
    if kind == "a change's short limit":
        _resolved(folder, [{"name": RUN, "changes": [CHANGE]}])
        return dict(strategy=st), CUT, "short"
    if kind == "a change's fuller limit":
        _resolved(folder, [{"name": RUN, "changes": [CHANGE]}], in_full={"weather_scale": [FULL]})
        return dict(strategy=st), FULL, "details"
    if kind == "a change's not meaning":
        _resolved(folder, [{"name": RUN, "changes": [CHANGE]}])
        return dict(strategy=st), "consistent change of climate", "details"
    if kind == "a limit for the question":
        de["question_limits"] = ["A made-up limit ZQ1."]
        return dict(strategy=st), "A made-up limit ZQ1.", "short"
    if kind == "a caveat of an older record":
        _resolved(folder, [{"name": RUN, "changes": [CHANGE]}])
        de["caveats"] = ["A made-up caveat ZQ2."]
        return dict(strategy=st), "A made-up caveat ZQ2.", "details"
    if kind == "a caveat the short page prints":
        de["caveats"] = ["A made-up caveat ZQ3."]
        return dict(strategy=st), "A made-up caveat ZQ3.", "short"
    if kind == "a caveat of the Planner's own":
        de.update(caveats_by="code", planner_caveats=["A made-up caveat ZQ4."])
        return dict(strategy=st), "A made-up caveat ZQ4.", "short"
    if kind == "what the control run does not contain":
        return dict(strategy=st), "the observed floods: the run makes its own", "short"
    if kind == "the numbers behind what it does not contain":
        row["not_in_it"] = [NOT_IN_IT]
        return dict(strategy=st), "2031 12 against 3 km2", "details"
    if kind == "the control run's tables":
        return dict(strategy=st, figures=_figures()), "| The control run | 12 | 7 | 1,234 |", \
            "details"
    if kind == "a note of the server":
        return dict(strategy=st), "A made-up thing to note.", "both"
    if kind == "a refused round":
        pv["rounds"] = [{"findings": [{"where": "run q", "what": "a made-up reason ZQ5"}],
                         "design": "s.json"}]
        return dict(preview=pv), "a made-up reason ZQ5", "both"
    if kind == "a refusal":
        return dict(preview={"refused": [{"where": "q", "what": "made-up ZQ6"}], "runs": []}), \
            "made-up ZQ6", "both"
    if kind == "an error of the build":
        return dict(preview={"refused": [], "runs": [], "error": "made-up ZQ7"}), \
            "made-up ZQ7", "both"
    if kind == "a failed check of the inputs":
        pv["inputs_check"] = {"pass": False, "reasons": ["made-up ZQ8"]}
        return dict(preview=pv), "made-up ZQ8", "both"
    if kind == "a part the server could not give":
        pv["problems"] = ["made-up ZQ9"]
        return dict(preview=pv), "made-up ZQ9", "short"
    if kind == "a remark of the review":
        return {}, REMARKS[1], "details"
    if kind == "a problem of code's check":
        _plan(folder, problems=["a made-up problem ZQ10 of the whole list"])
        return dict(record=_record(folder, status=AV.NOT_PASSED)), \
            "made-up problem ZQ10 of the whole list", "details"
    if kind == "a condition on a comparison":
        return {}, "Comparison 2 gives figures and no number that the report may state", \
            "details"
    if kind == "a year set against normal years that contain it":
        return {}, "a year this plan sets against them", "details"
    if kind == "what the design requires":
        st["requires"] = ["a made-up need ZQ11"]
        return dict(strategy=st), "A made-up need ZQ11.", "short"
    if kind == "a change no run has completed":
        de["runs"] = [{"name": "more_co2", "changes": [{"id": "co2_ppmv",
                                                        "params": {"ppmv": 560.0}}]}]
        return dict(strategy=st, record=None), "The model server has a note on the change " \
                                               "`co2_ppmv`.", "short"
    if kind == "a run not made yet":
        return {}, "The new run does not exist yet", "details"
    if kind == "a choice made without the person":
        return {}, "The question named no area, so the whole box is used.", "short"
    if kind == "a remark that makes a difference's point":
        # A REPEAT, NOT A LOSS (2026-11-13). The remark makes the point of the
        # server's sentence on the short page, in Reception's words and with the
        # server's identifiers, so neither page prints it. What it says is kept
        # by the server's sentence (TestChoicesMadeForYou holds that the remark
        # is on neither page).
        same = "The question says rainfall. The change multiplies rain and snow together."
        return dict(reception=_reception(run_settings={"conflicts": [same]})), SAID, "short"
    if kind == "a gap Reception noted":
        return {}, "Noted by Reception: A thing Reception noted.", "short"
    if kind == "how the question is read":
        return {}, DD.QUESTION_TYPE["exploratory"], "details"
    if kind == "a change in effect in every simulated year":
        de["runs"] = [{"name": "more_co2", "changes": [{"id": "co2_ppmv",
                                                        "params": {"ppmv": 560.0}}]}]
        return dict(strategy=st, record=None), "is in effect in every simulated year", "short"
    if kind == "the verdict and its reason":
        st["feasibility"] = {"verdict": "partial", "why": "made-up ZQ12"}
        return dict(strategy=st), "This design can answer part of the question. Made-up ZQ12.", \
            "both"
    if kind == "the Planner's steps":
        st["goals"] = ["a made-up step ZQ13"]
        return dict(strategy=st), "1. A made-up step ZQ13.", "details"
    if kind == "the other explanations":
        return {}, "Nitrogen, not water, sets growth.", "details"
    if kind == "the first look at every quantity":
        return {}, "found by screening and not as tested", "details"
    if kind == "the places with their cells, area and rule":
        return {}, "| `valley floor` | 210 | 300 | below a made-up line |", "details"
    if kind == "the cost":
        return {}, "2.4 node-hours", "both"
    if kind == "normal years unlike the control run's own":
        de["analysis"]["normal_years"] = [2031]
        row["normal_years"] = [2032, 2033]
        return dict(strategy=st, record=None), "so the numbers will differ", "details"
    if kind == "how the control run compares, with no figure":
        return {}, "It holds half of the measured snow.", "details"
    if kind == "where the list is kept":
        return {}, "`plan/analysis_plan.json`", "details"
    raise KeyError(kind)


LOST = ["a change's short limit", "a change's fuller limit", "a change's not meaning",
        "a limit for the question", "a caveat of an older record",
        "a caveat the short page prints", "a caveat of the Planner's own",
        "what the control run does not contain", "the numbers behind what it does not contain",
        "the control run's tables", "a note of the server", "a refused round", "a refusal",
        "an error of the build", "a failed check of the inputs",
        "a part the server could not give", "a remark of the review",
        "a problem of code's check", "a condition on a comparison",
        "a year set against normal years that contain it", "what the design requires",
        "a change no run has completed", "a run not made yet",
        "a choice made without the person", "a remark that makes a difference's point",
        "a gap Reception noted", "how the question is read",
        "a change in effect in every simulated year", "the verdict and its reason",
        "the Planner's steps", "the other explanations", "the first look at every quantity",
        "the places with their cells, area and rule", "the cost",
        "normal years unlike the control run's own",
        "how the control run compares, with no figure", "where the list is kept"]


@pytest.mark.parametrize("kind", LOST)
def test_nothing_a_decision_rests_on_is_lost(folder, kind):
    kw, text, where = _case(folder, kind)
    short_kw = {k: v for k, v in kw.items() if k != "figures"}
    short = _flat(_short(folder, **short_kw))
    details = _flat(_details(folder, **kw))
    text = _flat(text).lower()
    on = {"short": text in short.lower(), "details": text in details.lower()}
    if where == "both":
        assert on["short"] and on["details"], (kind, on)
    else:
        assert on[where], (kind, on)


# ─────────────────────────────────────────────────────────────────────────────
# THE WORDS OF BOTH PAGES
# ─────────────────────────────────────────────────────────────────────────────
def _designs(folder):
    """Both pages of many designs: (name, short page, details)."""
    out = []
    st2 = _new_strategy()
    st2["domain_experiment"]["runs"].append({"name": "no_fire", "changes": [{"id": "fire_off"}]})
    alone = _strategy()
    alone["domain_experiment"]["runs"] = []
    _control_row(alone)["not_in_it"] = [NOT_IN_IT]
    refused = {"refused": [{"where": f"run {RUN}", "what": "the control has a made-up fault"}],
               "runs": [], "rounds": [{"findings": ["the control's name"], "design": "s.json"}]}
    bound = _record(folder, handoff=BOUND)
    for name, kw in (("new form", dict(strategy=_new_strategy(), figures=_figures(),
                                       given=_given(), record=bound)),
                     ("old form", dict(figures=_figures(record=_figures_record()))),
                     ("several runs", dict(strategy=st2, record=None)),
                     ("control alone", dict(strategy=alone, preview=None,
                                            record=_record(folder, alone, handoff=BOUND))),
                     ("refused", dict(preview=refused, record=None)),
                     ("partial", dict(strategy=_strategy(feasibility={
                         "verdict": "partial", "why": "the control lacks a made-up field"}))),
                     ("infeasible", dict(strategy=_strategy(feasibility={
                         "verdict": "infeasible", "why": "no run shows it"}), record=None)),
                     ("not passed", dict(record=_record(folder, status=AV.NOT_PASSED,
                                                        handoff={"promised": False}))),
                     ("failed", dict(record=_record(folder, status=AV.FAILED, why="an error",
                                                    handoff={"promised": False})))):
        short_kw = {k: v for k, v in kw.items() if k != "figures"}
        out.append((name, _short(folder, **short_kw), _details(folder, **kw)))
    return out


class TestTheWords:

    def test_no_dash_no_cut_and_none_of_the_framework_words(self, full):
        for name, short, details in _designs(full):
            for page, which in ((short, "short"), (details, "details")):
                low = page.lower()
                for w in ("\u2014", "\u2013", "\u2026") + FRAMEWORK:
                    if which == "short" and w == "preview":
                        continue                    # held by test_design_page per design
                    assert w not in low, (name, which, w)

    def test_never_a_bare_control(self, full):
        # the table of comparisons on the short page included (2026-11-13): the
        # made-up list's question says "the control"
        for name, short, details in _designs(full):
            for page in (short, details):
                bare = re.findall(r"\bthe control\b(?! run)", page, re.IGNORECASE)
                assert not bare, (name, bare)

    def test_a_comparisons_own_words_never_say_a_bare_control(self, folder):
        # a comparison's title, question and what it tells are a model's words
        plan = copy.deepcopy(PLAN)
        plan["analyses"][0].update(title="Growth against the control",
                                   tells="How far the control leads")
        plan["analyses"][1]["question"] = "Does the control grow first?"
        plan["competing"][0]["title"] = "The control's own nitrogen"
        _plan(folder, analyses=plan["analyses"], competing=plan["competing"])
        short, details = _short(folder), _details(folder)
        for page in (short, details):
            for said in ("Growth against the control run", "How far the control run leads"):
                assert said in page, said
            assert not re.findall(r"\bthe control\b(?! run)", page, re.IGNORECASE)
        assert "| Does the control run grow first? |" in details
        assert '"The control run\'s own nitrogen"' in _flat(short)
        assert "| The control run's own nitrogen |" in details

    def test_the_direct_answer_once_on_the_short_page(self, full):
        for name, short, _ in _designs(full):
            assert short.count(DP.ANSWER_HEAD) == 1, name

    def test_the_words_a_person_reads_on_the_details_page_are_the_rules(self):
        # fixed sentences of the module: no colon in mid-sentence, no list in brackets
        for s in (DD.OPENING, DD.EVERYTHING_ELSE, DD.NO_COMPUTE, DD.NO_QUESTION_TYPE,
                  *DD.QUESTION_TYPE.values(), *DP.VERDICT_WORDS.values(), DP.NO_VERDICT):
            assert not re.search(r":\s", s) and "(" not in s, s

    def test_it_is_shorter_than_the_old_page(self, full):
        page = _details(full, strategy=_new_strategy(), figures=_figures(), given=_given())
        assert DP.words(page) < 1200, DP.words(page)


# ─────────────────────────────────────────────────────────────────────────────
# THE THREE PAGES, WIRED IN design_review.write_review
# ─────────────────────────────────────────────────────────────────────────────
class TestTheThreePages:

    def _write(self, folder, **kw):
        args = dict(reception=_reception(), strategy=_strategy(), preview=PREVIEW,
                    analysis_preview=_record(folder))
        args.update(kw)
        return dr.write_review(folder, **args)

    def test_the_details_page_is_the_new_one(self, full):
        self._write(full, strategy=_new_strategy(), control_figures=_figures()["record"])
        details = (full / DP.DETAILS_NAME).read_text()
        assert _heads(details) == HEADS
        assert "| Area | 900 km2 of made-up land |" in details
        assert "| The control run | 12 | 7 | 1,234 |" in details
        assert json.loads((full / "01_inputs" / "control_figures.json").read_text())["setup"] \
            == SETUP

    def test_another_kind_of_design_keeps_its_one_page(self, tmp_path):
        reception = {"user_request": "q", "brief": {"design_archetype": "site"}}
        dr.write_review(tmp_path, reception=reception, strategy={"archetype": "site"},
                        preview=None, analysis_preview=None)
        assert "# Design review" in (tmp_path / dr.REVIEW_NAME).read_text()
        assert not (tmp_path / DP.DETAILS_NAME).exists()
        assert not (tmp_path / DP.BY_HAND_NAME).exists()

    def test_run_by_hand_comes_from_its_module(self, folder, monkeypatch):
        mod = types.ModuleType("agents.run_by_hand")
        seen = []
        text = f"# How to make this run by hand\n\n{RUN}\n"
        mod.render = lambda preview, run_dir, strategy, brief: seen.append(
            (preview, Path(run_dir), strategy, brief)) or text
        _fake_module(monkeypatch, "run_by_hand", mod)
        path = self._write(folder)
        assert (folder / DP.BY_HAND_NAME).read_text() == text
        assert seen[0][0] == PREVIEW and seen[0][1] == folder
        assert seen[0][3] == _reception()["brief"]
        assert "The steps to make the run by hand are in `RUN_BY_HAND.md`." in \
            _flat(path.read_text())
        details = (folder / DP.DETAILS_NAME).read_text()
        assert "The steps to make this run without the framework are in `RUN_BY_HAND.md`." in \
            details
        assert "`RUN_BY_HAND.md` lists the jobs and their checks." in details

    @pytest.mark.parametrize("n, short, details", [
        (0, "How the control run was made is in `RUN_BY_HAND.md`.", None),
        (1, "The steps to make the run by hand are in `RUN_BY_HAND.md`.",
         "The steps to make this run without the framework are in `RUN_BY_HAND.md`."),
        (2, "The steps to make the runs by hand are in `RUN_BY_HAND.md`.",
         "The steps to make these runs without the framework are in `RUN_BY_HAND.md`.")])
    def test_the_sentence_that_names_run_by_hand_follows_the_new_runs(self, folder, n, short,
                                                                      details):
        st = _strategy()
        st["domain_experiment"]["runs"] = [
            {"name": RUN, "changes": [copy.deepcopy(CHANGE)]},
            {"name": "no_fire", "changes": [{"id": "fire_off"}]}][:n]
        pv = PREVIEW if n else None
        page = _flat(_short(folder, strategy=st, preview=pv, record=None))
        assert f"The design in full is in `DESIGN_DETAILS.md`. {short}" in page
        assert page.count("RUN_BY_HAND.md") == 1
        got = _flat(_details(folder, strategy=st, preview=pv, record=None))
        if details:
            assert details in got
        else:
            # no new run: the details name no file of steps, and the short page
            # says what it holds
            assert "RUN_BY_HAND" not in got

    def test_run_by_hand_ends_in_one_line_break(self, folder, monkeypatch):
        mod = types.ModuleType("agents.run_by_hand")
        _fake_module(monkeypatch, "run_by_hand", mod)
        for text in ("# Made up\n\nThe steps.", "# Made up\n\nThe steps.\n"):
            mod.render = lambda *a, text=text: text
            self._write(folder)
            assert (folder / DP.BY_HAND_NAME).read_text() == "# Made up\n\nThe steps.\n"

    def test_run_by_hand_with_nothing_to_say_writes_no_file(self, folder, monkeypatch):
        mod = types.ModuleType("agents.run_by_hand")
        mod.render = lambda *a: ""
        _fake_module(monkeypatch, "run_by_hand", mod)
        path = self._write(folder)
        assert not (folder / DP.BY_HAND_NAME).exists()
        assert "RUN_BY_HAND.md" not in path.read_text()
        assert "RUN_BY_HAND.md" not in (folder / DP.DETAILS_NAME).read_text()

    def test_a_page_the_module_cannot_write_is_written_as_before(self, folder, monkeypatch):
        mod = types.ModuleType("agents.run_by_hand")
        mod.render = lambda *a: (_ for _ in ()).throw(RuntimeError("made up"))
        _fake_module(monkeypatch, "run_by_hand", mod)
        self._write(folder)
        text = (folder / DP.BY_HAND_NAME).read_text()
        assert text.startswith("\n---\n\n" + dr.APPENDIX_MARK) and RUN in text

    def test_places_given_are_kept_and_drawn(self, folder, monkeypatch):
        mod = _places_module()
        _fake_module(monkeypatch, "places_figure", mod)
        self._write(folder, places=_places_record())
        assert json.loads((folder / mod.RECORD_FILE).read_text()) == _places_record()
        assert mod.calls["fetch"] == [] and len(mod.calls["draw"]) == 1
        assert "![The places the numbers are reported for](design_figures/places_reported.png)" \
            in (folder / DP.DETAILS_NAME).read_text()

    def test_places_left_out_are_read_or_asked_for_with_the_lists_regions(self, folder,
                                                                         monkeypatch):
        mod = _places_module(fetched=_places_record())
        _fake_module(monkeypatch, "places_figure", mod)
        self._write(folder)
        assert mod.calls["fetch"] == [("valley", ["vegetated cells", "valley floor"])]
        assert json.loads((folder / mod.RECORD_FILE).read_text()) == _places_record()
        self._write(folder)                                     # read, not asked again
        assert len(mod.calls["fetch"]) == 1 and len(mod.calls["draw"]) == 2
        # a record of another design is not drawn: the server is asked again
        (folder / mod.RECORD_FILE).write_text(json.dumps(_places_record(regions=["x"])))
        self._write(folder)
        assert len(mod.calls["fetch"]) == 2
        assert json.loads((folder / mod.RECORD_FILE).read_text()) == _places_record()

    def test_no_list_of_comparisons_no_map(self, folder, monkeypatch):
        mod = _places_module(fetched=_places_record())
        _fake_module(monkeypatch, "places_figure", mod)
        self._write(folder, analysis_preview=None)
        assert mod.calls == {"fetch": [], "draw": []}
        assert "![The places" not in (folder / DP.DETAILS_NAME).read_text()
        # a record given for another design is kept as given and not drawn
        self._write(folder, places=_places_record(control="another"))
        assert mod.calls["draw"] == []
        assert json.loads((folder / mod.RECORD_FILE).read_text())["control"] == "another"
        # None removes it
        self._write(folder, places=None)
        assert not (folder / mod.RECORD_FILE).exists()

    @pytest.mark.parametrize("how", ["absent", "fails"])
    def test_a_map_that_fails_never_stops_the_pages(self, folder, monkeypatch, how):
        _fake_module(monkeypatch, "places_figure", None if how == "absent" else
                     _places_module(fetched=_places_record(), fails=True))
        self._write(folder)
        details = (folder / DP.DETAILS_NAME).read_text()
        assert "| `valley floor` |" in details and "![The places" not in details

    def test_the_regions_are_those_the_list_names(self, folder):
        rec = _record(folder)
        assert DD.plan_regions(rec, folder, rec["design_sha256"]) == ["vegetated cells",
                                                                     "valley floor"]
        assert DD.plan_regions(rec, folder, "0" * 64) == []
        assert DD.plan_regions(None, folder, None) == []
        assert DD.plan_regions(dict(rec, status=AV.FAILED), folder, None) == []


class TestTheStudyRecord:

    @pytest.mark.parametrize("rel", [
        "DESIGN_DETAILS.md", "01_inputs/control_figures.json", "01_inputs/run_given.json",
        "01_inputs/places.json", "design_figures/places_reported.png",
        "design_figures/control_run_maps.png", "design_figures/new_run_x_series.png",
        "design_figures/anything_else.png"])
    def test_each_file_of_the_design_pages_has_a_role(self, rel):
        role = SR.file_role(rel)
        assert not role.startswith("no role"), rel
        for w in ("analysis plan", "preview", "\u2014"):
            assert w not in role, (rel, w)

    def test_the_details_role_says_what_the_page_holds(self):
        assert SR.file_role("DESIGN_DETAILS.md").startswith("what the short design page "
                                                            "leaves out")
        assert SR.file_role("design_figures/places_reported.png").startswith(
            "the map of the places")
