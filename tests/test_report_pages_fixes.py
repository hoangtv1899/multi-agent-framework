"""The report's pages after the join of the report rebuild: every figure in
its order, a finding that finds its comparison, words in place of codes on
the short pages, the screen's rules in plain sentences, short limits, pages
that never stop on an odd value, and the screen a redraw makes first.

Every record here is made up: its names, numbers and sentences belong to no
study.
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.analysis import report_pages as RP                   # noqa: E402
from agents.analysis import step4_report as S4                   # noqa: E402
from test_report_pages import (RUN, _redraw_tool, figures_record,  # noqa: E402,F401
                               make_report, pages, saved_study)


def _section(text, head, nxt):
    return text[text.index(head):text.index(nxt)]


def _results(text):
    return _section(text, "## 3. Results", "## 4. Limitations")


# ── every figure in its order, each table after the figure it follows ────────
def three_figures():
    """A split figure over time (Figures 1 and 2) and a figure over space,
    listed out of order, with Table 1 after the second figure over time and
    Table 2 after the figure over space."""
    return {"figures": [
        {"id": "over_space", "file": "figure_over_space.png", "number": 3,
         "heading": "Growth in each cell", "caption": "Each cell's growth.", "comparisons": [],
         "panels": [{"letter": "a", "field": "GPP"}]},
        {"id": "over_time", "file": "figure_over_time.png", "number": 1,
         "heading": "Water month by month", "caption": "Water, month by month.",
         "comparisons": [3], "panels": [{"letter": "a", "field": "RAIN"}]},
        {"id": "over_time_2", "file": "figure_over_time_2.png", "number": 2,
         "heading": "Plants month by month", "caption": "Plants, month by month.",
         "comparisons": [1, 2], "panels": [{"letter": "a", "field": "GPP"},
                                           {"letter": "b", "field": "SOILWATER_10CM"}]}],
        "tables": [
            {"id": "elevation", "number": 2, "follows": "over_space", "caption": "By band.",
             "corner": "Band", "columns": ["Area, km2"],
             "rows": [{"label": "Low", "cells": ["10"], "records": ["L40", "L6", "L2"]}]},
            {"id": "periods", "number": 1, "follows": "over_time_2", "caption": "By period.",
             "corner": "Quantity", "columns": ["Summer 2033"],
             "rows": [{"label": "Rain", "cells": ["1.5"], "panel": "1a", "records": ["L5"]},
                      {"label": "Growth", "cells": ["1.5"], "panel": "2a",
                       "records": ["L2"]}]}],
        "printed_records": ["L2", "L5", "L6"], "not_drawn": [], "problems": []}


def test_every_figure_in_its_order_and_each_table_after_its_figure(tmp_path):
    _, p = pages(tmp_path, figures=three_figures())
    res = _results(p["REPORT.md"])
    order = [res.index(x) for x in ("![Figure 1](figure_over_time.png)",
                                    "![Figure 2](figure_over_time_2.png)", "**Table 1.**",
                                    "![Figure 3](figure_over_space.png)", "**Table 2.**")]
    assert order == sorted(order)
    assert [ln for ln in res.splitlines() if ln.startswith("### ")][:3] == [
        "### 3.1 Water month by month", "### 3.2 Plants month by month",
        "### 3.3 Growth in each cell"]


def test_under_each_caption_the_comparisons_it_answers(tmp_path):
    _, p = pages(tmp_path, figures=three_figures())
    res = _results(p["REPORT.md"])
    assert "**Figure 1.** Water, month by month.\n\nThis figure answers comparison 3.\n" in res
    assert "**Figure 2.** Plants, month by month.\n\nThis figure answers comparisons 1 and " \
           "2.\n" in res
    # an empty list says nothing
    three = _section(res, "**Figure 3.**", "**Table 2.**")
    assert "answers" not in three
    assert RP.answers_line({"comparisons": [3, 7, 8]}) == \
        "This figure answers comparisons 3, 7 and 8."


def test_a_finding_whose_records_print_in_a_table_is_left_out(tmp_path):
    """Table 1 follows the second figure over time, but its row of panel 1a
    belongs to Figure 1: a finding of Figure 1 whose records all print in
    that row says nothing the table does not."""
    _, p = pages(tmp_path, figures=three_figures())
    assert "Rain was lower in July 2033." not in _results(p["REPORT.md"])
    assert RP.table_records(three_figures()) == {"L2", "L5", "L6", "L40"}
    assert "| 3 | Rain month by month |" in p["PLANNED_ANALYSES.md"]
    row = next(ln for ln in p["PLANNED_ANALYSES.md"].splitlines() if ln.startswith("| 3 |"))
    assert row.endswith("| Figure 1 and Table 1 |")          # every panel: the whole figure


def test_a_finding_sits_under_the_figure_that_draws_its_quantity(tmp_path):
    """Two figures answer comparisons 1 and 2; the finding on topsoil water
    goes under the one that draws it, the findings on growth under the
    first."""
    rf = {"figures": [
        {"id": "over_time", "file": "a.png", "number": 1, "caption": "A.", "comparisons": [1, 2],
         "panels": [{"letter": "a", "field": "GPP"}]},
        {"id": "over_time_2", "file": "b.png", "number": 2, "caption": "B.",
         "comparisons": [1, 2], "panels": [{"letter": "a", "field": "SOILWATER_10CM"}]}],
        "tables": []}
    report, adir = make_report(tmp_path, figures=rf)
    report["_plan"] = json.loads((adir / "analysis_plan.json").read_text())
    by_fig, others = RP.choose_findings(report, rf, RP.numbers_of(report["_plan"], report))
    assert by_fig == {"over_time": [1, 2], "over_time_2": [4]} and others == [(3, 5)]


# ── a finding must find its comparison ──────────────────────────────────────
NUMBERS = {"P1": 1, "P2": 2, "P3": 3}


def test_the_finding_names_its_comparison():
    c = {"finding_id": "required_p2", "evidence": ["L1"]}
    assert RP.claim_comparison(c, {"L1": {"plan_items": ["P1"]}}, NUMBERS) == 2


def test_the_records_cited_give_the_smallest_comparison():
    c = {"finding_id": "fig_x", "evidence": ["L1", "L2"]}
    records = {"L1": {"plan_items": ["P3"]}, "L2": {"plan_items": ["P3", "P1"]}}
    assert RP.claim_comparison(c, records, NUMBERS) == 1


def test_the_log_gives_a_records_plan_items_before_the_traces_copy():
    c = {"finding_id": "fig_x", "evidence": ["L4"]}
    assert RP.claim_comparison(c, {"L4": {}}, NUMBERS, lookups={"L4": ["P2", "screen"]}) == 2


def test_the_analyzers_own_figure_gives_it_when_the_records_do_not():
    c = {"finding_id": "fig_x", "evidence": ["L4"]}
    assert RP.claim_comparison(c, {"L4": {}}, NUMBERS, lookups={},
                               figure_items={"fig_x": ["P3", "P2"]}) == 2
    assert RP.claim_comparison(c, {"L4": {}}, NUMBERS) is None
    assert RP.claim_comparison("not a finding", {}, NUMBERS) is None


def _no_items(report):
    """The finding on L5 ("fig_rain") with no plan items in the trace."""
    report["evidence_trace"]["records"]["L5"]["plan_items"] = []
    return report


def _figure_of_2():
    return {"figures": [{"id": "over_time", "file": "figure_over_time.png", "number": 1,
                         "heading": "Soil month by month", "caption": "Soil.",
                         "comparisons": [2], "panels": []}],
            "tables": [], "printed_records": [], "not_drawn": [], "problems": []}


def test_route_through_the_lookups_log(tmp_path):
    report, adir = make_report(tmp_path, figures=_figure_of_2())
    _no_items(report)
    lines = [{"call": "screen", "plan_items": ["screen"]}] + \
        [{"call": "difference", "plan_items": ["P1"]}] * 3 + \
        [{"call": "difference", "plan_items": ["P2"]}]
    (adir / "lookups.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n\n")
    RP.write_pages(report, adir)
    res = _results((adir / "REPORT.md").read_text())
    assert "Rain was lower in July 2033." in _section(res, "**Figure 1.**", "Other comparisons")


def test_route_through_the_analyzers_own_figure(tmp_path):
    report, adir = make_report(tmp_path, figures=_figure_of_2())
    _no_items(report)
    (adir / "investigation.json").write_text(json.dumps({"figure_outcomes": [
        {"id": "fig_rain", "plan_items": ["P2"]}]}))
    RP.write_pages(report, adir)
    res = _results((adir / "REPORT.md").read_text())
    assert "Rain was lower in July 2033." in _section(res, "**Figure 1.**", "Other comparisons")


def test_a_finding_with_no_comparison_goes_to_other_comparisons(tmp_path):
    report, adir = make_report(tmp_path, figures=_figure_of_2())
    _no_items(report)
    RP.write_pages(report, adir)
    res = _results((adir / "REPORT.md").read_text())
    other = res[res.index("Other comparisons"):]
    assert "- Rain was lower in July 2033." in other
    # comparison 1 has no figure: its first kept finding
    assert "- Growth fell by 1.5 gC/m2 in summer 2033." in other


# ── no code on the short pages ──────────────────────────────────────────────
SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
RECORD = re.compile(r"\bL\d+\b")


def visible(text: str) -> str:
    """A page as its reader sees it: no link target, no code span."""
    text = re.sub(r"\]\([^)]*\)", "]", text)
    return re.sub(r"`[^`]*`", "", text)


def marked_rows():
    rows = {
        "display.events.0": {"field": "FAREA_BURNED", "group": "fire", "period": "2033-08",
                             "change": 2.5, "change_units": "km2", "score": None,
                             "status": "marked", "mark": "above_normal_maximum"},
        "display.events.1": {"field": "COL_FIRE_CLOSS", "group": "fire", "period": "2033",
                             "change": 1.5, "change_units": "gC/m^2/yr", "score": None,
                             "status": "marked", "mark": "differs_from_control"}}
    return rows


@pytest.mark.parametrize("figs", [None, "record"])
def test_no_snake_case_word_or_record_name_on_the_short_pages(tmp_path, figs):
    report, adir = make_report(tmp_path, screen_run=RUN,
                               figures=three_figures() if figs else None)
    report["evidence_trace"]["records"]["L1"]["rows"].update(marked_rows())
    report["evidence_trace"]["screen_record"]["scoring_rule"] = \
        "scored: the change over the spread (ddof 1), the catalog's order"
    RP.write_pages(report, adir)
    for name in ("REPORT.md", "SCREEN.md", "PLANNED_ANALYSES.md"):
        text = visible((adir / name).read_text())
        assert not SNAKE.search(text), (name, SNAKE.search(text))
        assert not RECORD.search(text), (name, RECORD.search(text))
        for word in ("ddof", "catalog's"):
            assert word not in text, (name, word)


def test_the_screens_marks_and_its_table_in_words(tmp_path):
    report, adir = make_report(tmp_path, screen_run=RUN)
    report["evidence_trace"]["records"]["L1"]["rows"].update(marked_rows())
    RP.write_pages(report, adir)
    s = (adir / "SCREEN.md").read_text()
    assert f"| {RP.MARK_WORDS['above_normal_maximum']} |" in s
    assert f"| {RP.MARK_WORDS['differs_from_control']} |" in s
    assert "[The whole table](screen_L1.csv) holds every quantity and period of the screen, " \
           "120 rows." in s


# ── the screen's rules: fixed plain sentences, held to the server's text ────
def _server_rules():
    """The server's own rule text: the scoring rule as a screen's record
    writes it, the display rule, and the outcomes of the screen method."""
    from agents.analysis import analysis_plan as AP
    AP._elm()
    import domain_access as DA                                      # noqa: N812
    import domain_methods as DM                                     # noqa: N812
    sr = DA._screen_record({"rows": [], "display_rule": DM.DISPLAY_RULE})["screen_record"]
    outcomes = " ".join(f"{k}: {v}" for k, v in DM.METHODS["screen"].outcomes)
    return sr["scoring_rule"] + " " + outcomes, sr["display_rule"]


# (the server's words, the page's words): each key fact on both sides
SCORING_FACTS = (
    ("the change over the normal years' standard deviation",
     "the change divided by the standard deviation of the normal years"),
    ("for an experiment, the run minus the control over the control's normal-year standard "
     "deviation", "the new run minus the control run, and the standard deviation is that of "
                  "the control run's normal years"),
    ("Two or more normal values, none of them short of the region's area",
     "at least two normal years whose values cover the whole region"),
    ("Normal years without the period (missing time records) lower the count shown",
     "lacks the period lowers the count of normal years shown"),
    ("near-constant share", "hardly vary gets no score"),
    ("A change of exactly zero is marked no change and gets no score",
     "A change of exactly zero gets no score."),
    ("stores and bounded indices", "Stores, such as plant carbon, and indices held between "
                                   "fixed bounds are not scored yet"),
    ("burned area in km2, carbon lost to fire", "Fire rows, the area burned and the carbon "
                                               "lost to fire, get no score."),
    ("above the normal years' largest", "above the largest value of the normal years"),
    ("an experiment's event row unlike the control's", "marked when it differs from the "
                                                       "control run"))
DISPLAY_FACTS = (
    ("per process group, the groups in the model's order",
     "by process group, the groups in the model's order"),
    ("the three rows with the largest absolute score, largest first",
     "three rows with the largest score in size, the largest first"),
    ("at most one per field (its highest-scored period; its other periods stay in the table)",
     "at most one row there, its period with the largest score, and its other periods stay"),
    ("a tie keeps the field that comes first in the catalog and, within a field, the period "
     "that comes first in the table", "A tie keeps the quantity that comes first in the "
                                      "model's list and, within a quantity, the period that "
                                      "comes first in the table."),
    ("a group without a scored row shows none", "A group without a scored row shows none."),
    ("Then every event row marked", "Then come all the marked fire rows, in the table's order."),
    ("its largest absolute change with its period", "comes its largest change with its period"),
    ("Every score is shown with the number of normal years behind it",
     "Every score is shown with the number of normal years behind it."),
    ("are not shown first and stay in the table",
     "never shown first and stays in the whole table"))


def test_the_rules_hold_the_key_facts_of_the_servers_rule_text():
    scoring, display = _server_rules()
    mine_s, mine_d = " ".join(RP.SCORING_SENTENCES), " ".join(RP.DISPLAY_SENTENCES)
    for server, page in SCORING_FACTS:
        assert server in scoring, server
        assert page in mine_s, page
    for server, page in DISPLAY_FACTS:
        assert server in display, server
        assert page in mine_d, page
    # one sentence each, plain
    for x in RP.SCORING_SENTENCES + RP.DISPLAY_SENTENCES:
        assert x.endswith(".") and ": " not in x and "(" not in x and "_" not in x, x


def test_the_rules_are_the_same_on_every_report_and_whole_on_the_details(tmp_path):
    texts = []
    for k, rule in enumerate(("one made-up rule; with a clause", "another made-up rule")):
        report, adir = make_report(tmp_path / str(k))
        report["evidence_trace"]["screen_record"]["scoring_rule"] = rule
        RP.write_pages(report, adir)
        s = (adir / "SCREEN.md").read_text()
        texts.append(s[s.index("## How the rows are scored"):])
        assert f"Scoring rule: {rule}" in (adir / "REPORT_DETAILS.md").read_text()
    assert texts[0] == texts[1]


# ── the screen the screen's record describes ────────────────────────────────
def test_the_screen_page_shows_the_screen_its_record_describes(tmp_path):
    report, adir = make_report(tmp_path, screen_run=RUN)
    recs = report["evidence_trace"]["records"]
    old = dict(recs["L1"], run="control", keys={"run": "control"}, sha256="aa")
    new = dict(recs["L1"], id="L7", run=RUN, keys={"run": RUN}, sha256="bb", planned=False,
               after_close=True, screen_csv={"file": "screen_L7.csv", "rows": 120})
    recs["L1"], recs["L7"] = old, new
    report["evidence_trace"]["screen_record"]["sha256"] = "bb"
    RP.write_pages(report, adir)
    s = (adir / "SCREEN.md").read_text()
    assert s.count("against the control run over") == 1 and "normal years 2031" not in s
    assert "(screen_L7.csv)" in s and "(screen_L1.csv)" not in s
    assert "made after the planned comparisons closed" in s
    d = (adir / "REPORT_DETAILS.md").read_text()
    screens = d[d.index("## The screens on record"):]
    assert "- record L1: run control" in screens and "not shown on SCREEN.md" in screens
    assert "- record L7: run wet_run" in screens and "not planned, made after the planned " \
                                                     "phase closed" in screens


def test_an_old_screen_record_with_its_run_in_its_keys():
    sr = {"keys": {"run": RUN}, "scope": {"year": 2033}}
    assert RP.screen_run_of(sr) == RUN
    assert RP.screen_run_of({"scope": {}}) == "control"
    assert RP.screen_run_of({"scope": {"run": "b"}, "keys": {"run": "a"}}) == "b"


# ── Limitations: plain, short ───────────────────────────────────────────────
def _design(sd, limits):
    (sd / "strategy.json").write_text(json.dumps({
        "domain_experiment": {"control": {"name": "box"}, "runs": [],
                              "question_limits": limits, "caveats_by": "code"},
        "domain_rulebook": {"controls": {"controls": [
            {"name": "box", "not_in_it_words": [
                "the made-up fires: the run's fire is the model's own"]}]},
            "changes": {"changes": []}}}))
    (sd / "reception.json").write_text(json.dumps({"user_request": "a made-up question"}))


def _later(sd):
    (sd / "experiment.json").write_text(json.dumps({"experiment": {"check": {"unusual": [
        {"where": "years", "what": "not demonstrated: the run starts from the control's saved "
                                   "state of 1 January 2033, and no run started from a later "
                                   "saved state has been completed in this framework yet. Its "
                                   "inputs are checked before the model is built; its start is "
                                   "untested"}]}}}))


def _limit_bullets(adir):
    text = (adir / "REPORT.md").read_text()
    return [ln[2:] for ln in _section(text, "## 4.", "## 5.").splitlines()
            if ln.startswith("- ")]


def test_limitations_are_plain_sentences(tmp_path):
    report, adir = make_report(tmp_path)
    _design(adir.parent, ["A made-up limit about snow. It could change the growth."])
    _later(adir.parent)
    RP.write_pages(report, adir)
    bullets = _limit_bullets(adir)
    for b in bullets:
        assert ": " not in b, b
        for word in ("model server", "framework", "has a note"):
            assert word not in b, (word, b)
    # the design page's two-sentence limit whole, with the Planner's judgement
    assert "A made-up limit about snow. It could change the growth. This is the Planner's " \
           "judgement." in bullets
    # a colon in mid-sentence: the words after it start a sentence of their own
    assert "The control run does not contain the made-up fires. The run's fire is the " \
           "model's own." in bullets
    # the later start: one plain sentence, its date the record's
    later = RP.LATER_START.format(date="1 January 2033")
    assert later in bullets and later.count(". ") == 0


def test_a_later_start_without_a_date_on_record():
    assert RP.later_start(Path("/nonexistent")) == []
    assert "{date}" not in RP.LATER_START_NO_DATE


# ── the twelve faults the review found uncaught ─────────────────────────────
def test_fault_1_a_finding_citing_comparisons_1_and_3_goes_to_1():
    c = {"finding_id": "fig_x", "evidence": ["L3", "L1"]}
    assert RP.claim_comparison(c, {"L1": {"plan_items": ["P1"]},
                                   "L3": {"plan_items": ["P3"]}}, NUMBERS) == 1


def test_fault_2_a_bracket_with_a_word_beside_a_name_keeps_its_words():
    assert RP.without_ids("Growth fell (L1, wider region).", {"L1"}) == \
        "Growth fell (L1, wider region)."
    assert RP.without_ids("Growth fell (L1, L2).", {"L1", "L2"}) == "Growth fell."


def test_fault_3_two_figures_answer_a_comparison_and_the_first_takes_its_finding(tmp_path):
    rf = {"figures": [{"id": "first", "number": 1, "comparisons": [2]},
                      {"id": "second", "number": 2, "comparisons": [2]}], "tables": []}
    report, adir = make_report(tmp_path, figures=rf)
    report["_plan"] = json.loads((adir / "analysis_plan.json").read_text())
    by_fig, _ = RP.choose_findings(report, rf, RP.numbers_of(report["_plan"], report))
    assert by_fig == {"first": [4], "second": []}


@pytest.mark.parametrize("share, said", [(1.0, False), (0.25, True)])
def test_fault_4_a_change_over_the_whole_region_is_no_limit(tmp_path, share, said):
    report, adir = make_report(tmp_path)
    (adir.parent / "experiment.json").write_text(json.dumps({"regional": {"changed_share": {
        RUN: {"region": "basin", "share": {"basin, vegetated": share}}}}}))
    RP.write_pages(report, adir)
    assert ("applies only" in "\n".join(_limit_bullets(adir))) is said


def test_fault_5_a_number_not_estimable_is_said(tmp_path):
    report, adir = make_report(tmp_path)
    report["planned_phase"]["analyses"][2]["outputs"][1]["outcome"] = "not estimable"
    RP.write_pages(report, adir)
    assert "| failed, 1 of 2 numbers produced and 1 not estimable |" in \
        (adir / "PLANNED_ANALYSES.md").read_text()


def test_fault_6_at_most_three_lines_under_next_steps(tmp_path):
    report, adir = make_report(tmp_path, withheld=[])
    cause = report["evidence_trace"]["open_causes"][0]
    report["evidence_trace"]["open_causes"] = [cause, dict(cause, cause="How warmth acts")]
    RP.write_pages(report, adir)
    nxt = _section((adir / "REPORT.md").read_text(), "## 5.", "## 6.")
    assert sum(ln.startswith("- ") for ln in nxt.splitlines()) == RP.MAX_NEXT == 3


def test_fault_7_only_a_whole_domain_report_gets_the_pages(tmp_path):
    report, adir = make_report(tmp_path)
    for kind in ("point", "", None):
        assert RP.applies(dict(report, kind=kind)) is False
        assert RP.write_pages(dict(report, kind=kind), adir) == {}
    assert not (adir / "REPORT.md").exists()


@pytest.mark.parametrize("n, words", [
    (1, "*1 sentence of the Analyzer's summary is left out here, because each gives a number "
        "that no kept finding gives. The details page shows it.*"),
    (2, "*2 sentences of the Analyzer's summary are left out here, because each gives a number "
        "that no kept finding gives. The details page shows them.*")])
def test_fault_8_the_sentences_left_out_in_their_number(tmp_path, n, words):
    report, adir = make_report(tmp_path)
    report["answer_kept"] = "Growth fell in both summers."
    report["answer_left_out"] = [{"sentence": f"A made-up sentence {k}.",
                                  "unsupported": ["9.9"]} for k in range(n)]
    RP.write_pages(report, adir)
    assert words in (adir / "REPORT.md").read_text()


def test_fault_9_the_methods_point_to_the_design_page(tmp_path):
    report, adir = make_report(tmp_path)
    (adir.parent / "DESIGN_REVIEW.md").write_text("a made-up design page")
    RP.write_pages(report, adir)
    m = _section((adir / "REPORT.md").read_text(), "## 2. Methods", "## 3.")
    assert "The design page, [DESIGN_REVIEW.md](../DESIGN_REVIEW.md), describes it and sets it " \
           "beside measurements." in m


def test_fault_10_the_numbers_not_estimable_are_counted_not_the_comparisons(tmp_path):
    report, adir = make_report(tmp_path)
    report["acceptance"] = {"ok": True, "blockers": [], "untestable": [
        {"analysis": "P3", "expectation": "E1", "outputs": ["O1", "O2"]}]}
    RP.write_pages(report, adir)
    assert "2 required numbers of comparison 3 could not be estimated" in \
        (adir / "REPORT.md").read_text()


def test_fault_11_the_summary_says_what_kind_of_question_it_was(tmp_path):
    _, p = pages(tmp_path)
    assert RP.QUESTION_KIND["exploratory"] in _section(p["REPORT.md"], "## 1.", "## 2.")


def test_fault_12_the_details_page_gives_the_sentences_left_out(tmp_path):
    report, adir = make_report(tmp_path)
    report["answer_kept"] = "Growth fell in both summers."
    report["answer_left_out"] = [{"sentence": "A made-up sentence.", "unsupported": ["9.9"]}]
    RP.write_pages(report, adir)
    d = (adir / "REPORT_DETAILS.md").read_text()
    assert "## The sentences of the summary left out of the report" in d
    assert "- \"A made-up sentence.\" It gives 9.9, which no kept finding's record gives." in d


# ── odd values never stop the pages ─────────────────────────────────────────
def test_odd_values_never_stop_the_pages(tmp_path):
    rf = three_figures()
    rf["figures"][1]["comparisons"] = ["a", None, 3.0, "1", True]
    rf["figures"].append("not a figure")
    rf["tables"][1]["rows"].append("not a row")
    report, adir = make_report(tmp_path, figures=rf)
    report["evidence_numbers"] = "x"
    report["claims"].append("not a finding")
    report["evidence_trace"]["records"]["L9"] = "not a record"
    written = RP.write_pages(report, adir)
    assert set(written) == {"REPORT.md", "REPORT_DETAILS.md", "PLANNED_ANALYSES.md",
                            "SCREEN.md"}
    text = (adir / "REPORT.md").read_text()
    assert [ln[3:] for ln in text.splitlines() if ln.startswith("## ")] == list(RP.SECTIONS)
    assert "This figure answers comparisons 1 and 3." in text
    from agents.analysis import numbers_check as NC
    assert NC.evidence_sentence(None, figures=True) in text


def test_a_part_that_fails_says_so_and_the_details_name_why(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise KeyError("made-up")
    monkeypatch.setattr(RP, "next_step_lines", boom)
    monkeypatch.setattr(RP, "render_planned", boom)
    _, p = pages(tmp_path)
    nxt = _section(p["REPORT.md"], "## 5.", "## 6.")
    assert "This section could not be written from the record." in nxt
    assert "This page could not be written from the record." in p["PLANNED_ANALYSES.md"]
    probs = p["REPORT_DETAILS.md"].split("## Parts of the pages that could not be written")[1]
    assert "REPORT.md, section 5. Next steps: KeyError" in probs
    assert "PLANNED_ANALYSES.md: KeyError" in probs


def test_the_one_page_sets_the_companions_of_an_earlier_pass_aside(tmp_path, monkeypatch):
    report, adir = make_report(tmp_path)
    for name in ("REPORT_DETAILS.md", "SCREEN.md", "PLANNED_ANALYSES.md"):
        (adir / name).write_text("an earlier page")

    def boom(*a, **k):
        raise OSError("made-up full disk")
    monkeypatch.setattr(RP, "write_pages", boom)
    S4.write(report, adir, slides=False)
    assert (adir / "REPORT.md").read_text() == S4.render_markdown(report)
    for name in ("REPORT_DETAILS.md", "SCREEN.md", "PLANNED_ANALYSES.md"):
        assert not (adir / name).exists() and list((adir / "superseded").rglob(name))


# ── a study with no design page ─────────────────────────────────────────────
def test_with_no_design_page_the_report_says_only_that(tmp_path):
    report, adir = make_report(tmp_path)
    RP.write_pages(report, adir)
    text = (adir / "REPORT.md").read_text()
    m = _section(text, "## 2. Methods", "## 3.")
    assert "The control run is the made-up box. This study holds no design page." in m
    assert "describes it" not in text and "does not hold" not in text
    assert "- This study holds no design page." in text[text.index("## 6."):]
    report["model_setup"]["control"] = {}
    assert RP.control_line(adir.parent, report) == \
        "No description of the control run is on record. This study holds no design page."


# ── the details page ────────────────────────────────────────────────────────
def test_a_table_rows_records_in_order_on_the_details_page(tmp_path):
    _, p = pages(tmp_path, figures=three_figures())
    d = p["REPORT_DETAILS.md"]
    assert "  - Low: records L2, L6, L40" in d
    assert RP.sorted_records(["L40", "L5", "L2", "L5", "x"]) == ["L2", "L5", "L40", "x"]


# ── tools/redraw_report.py: the screen first ────────────────────────────────
def test_the_redraw_screens_the_new_run_first_and_says_so(saved_study, monkeypatch, capsys):
    from agents.analysis import analysis_plan as AP
    report, adir = saved_study
    plan = json.loads((adir / "analysis_plan.json").read_text())
    plan["comparisons"] = [{"id": "C1", "run": RUN, "against": "control"}]
    (adir / "analysis_plan.json").write_text(json.dumps(plan))
    (adir / "screen_record.json").write_text(json.dumps({"sha256": "aa", "keys": {
        "run": "control"}}))
    calls = []

    def screen_study(study_dir):
        calls.append(Path(study_dir))
        return {"sha256": "bb", "scope": {"run": RUN, "year": 2033}}
    monkeypatch.setattr(AP, "screen_study", screen_study)
    monkeypatch.setattr(S4, "build", lambda *a, **k: dict(report))
    tool = _redraw_tool()
    tool.main([str(adir.parent)])
    assert calls == [adir.parent.resolve()]
    out = capsys.readouterr().out
    assert f"screen: made again; it sets the new run {RUN} against the control run in 2033" \
        in out
    assert "figures 0, tables 0" in out


def test_the_redraw_keeps_the_screen_of_a_control_run_alone(saved_study, monkeypatch):
    from agents.analysis import analysis_plan as AP
    report, adir = saved_study

    calls = []
    monkeypatch.setattr(AP, "screen_study", lambda study_dir: calls.append(study_dir))
    monkeypatch.setattr(S4, "build", lambda *a, **k: dict(report))
    out = _redraw_tool().redraw(adir.parent, stamp="S3")
    assert calls == [] and out["screen"]["made"] is False
    assert out["screen"]["said"] == ("screen: kept; a study of the control run alone screens "
                                     "its year against the normal years")


def test_a_screen_that_cannot_be_made_does_not_stop_the_redraw(saved_study, monkeypatch):
    from agents.analysis import analysis_plan as AP
    report, adir = saved_study
    plan = json.loads((adir / "analysis_plan.json").read_text())
    plan["comparisons"] = [{"id": "C1", "run": RUN, "against": "control"}]
    (adir / "analysis_plan.json").write_text(json.dumps(plan))

    def screen_study(study_dir):
        raise AP.PlanNotApproved("made-up: no store")
    monkeypatch.setattr(AP, "screen_study", screen_study)
    monkeypatch.setattr(S4, "build", lambda *a, **k: dict(report))
    out = _redraw_tool().redraw(adir.parent, stamp="S4")
    assert out["screen"]["made"] is False
    assert out["screen"]["said"].startswith("screen: NOT made again (PlanNotApproved: made-up")
    assert out["pages"][0] == "REPORT.md"


# ── what was not drawn, once ────────────────────────────────────────────────
def test_a_sentence_a_caption_says_is_not_said_again(tmp_path):
    rf = figures_record()
    rf["figures"][0]["caption"] = ("The new run minus the control run. Snow (SNOW) and wind "
                                   "(WIND) have no panel, because the new run does not change "
                                   "them.")
    rf["not_drawn"] = ["Snow (SNOW) has no panel over time, because the new run does not "
                       "change it.",
                       "The table by period holds 2 made-up periods, so autumn is not in it."]
    _, p = pages(tmp_path, figures=rf)
    res = _results(p["REPORT.md"])
    assert "Snow (SNOW) has no panel" not in res
    assert "The table by period holds 2 made-up periods, so autumn is not in it." in res
    assert "- not drawn: Snow (SNOW) has no panel over time" in p["REPORT_DETAILS.md"]


def test_the_panels_of_a_comparison_are_listed(tmp_path):
    rf = figures_record()
    rf["figures"][0]["panels"].append({"letter": "c", "field": "TLAI"})
    report, adir = make_report(tmp_path, figures=rf)
    plan = json.loads((adir / "analysis_plan.json").read_text())
    plan["analyses"][0]["fields"] = ["GPP", "RAIN"]
    (adir / "analysis_plan.json").write_text(json.dumps(plan))
    RP.write_pages(report, adir)
    assert "| Figure 1a, 1b and Table 1 |" in (adir / "PLANNED_ANALYSES.md").read_text()


def test_a_finding_on_two_maps_goes_to_the_figure_that_draws_both(tmp_path):
    rf = {"figures": [
        {"id": "a", "number": 1, "comparisons": [2], "panels": [{"letter": "a", "field": "GPP"}]},
        {"id": "b", "number": 2, "comparisons": [2],
         "panels": [{"letter": "a", "field": "GPP"}, {"letter": "b", "field": "FPG"}]}],
        "tables": []}
    report, adir = make_report(tmp_path, figures=rf)
    report["_plan"] = json.loads((adir / "analysis_plan.json").read_text())
    recs = report["evidence_trace"]["records"]
    recs["L4"] = dict(recs["L4"], field=None, call="overlap",
                      inputs=[{"record": "L2", "field": "GPP"}, {"record": "L9", "field": "FPG"}])
    by_fig, _ = RP.choose_findings(report, rf, RP.numbers_of(report["_plan"], report))
    assert by_fig == {"a": [], "b": [4]}
    # a figure that sets the two maps against each other, cell by cell, wins
    rf["figures"].append({"id": "c", "number": 3, "comparisons": [2], "panels": [
        {"letter": "a", "field": "GPP", "against": "FPG", "kind": "cells"}]})
    by_fig, _ = RP.choose_findings(report, rf, RP.numbers_of(report["_plan"], report))
    assert by_fig == {"a": [], "b": [], "c": [4]}
