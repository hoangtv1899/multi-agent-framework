"""The four pages of a whole-domain report (agents.analysis.report_pages).

A finished whole-domain study is written as REPORT.md in the order of a
paper, with SCREEN.md, PLANNED_ANALYSES.md and REPORT_DETAILS.md beside it.
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

RUN = "wet_run"
REGION = "basin, vegetated"


def _record(rid, field, period, plan_items, call="difference", **extra):
    return {"id": rid, "call": call, "method": call, "run": RUN, "field": field,
            "period": period, "region": REGION, "planned": True, "plan_items": plan_items,
            "keys": {"run": RUN, "field": field, "period": period, "region": REGION},
            "values": {"difference": 1.5, "percent": 2.5},
            "comparison": {"against": "the control (control)", "run": RUN},
            "warnings": [], **extra}


def _screen_rows():
    return {
        "display.groups.weather in.0": {"field": "RAIN", "group": "weather in", "period": "2033-07",
                                        "change": -12.5, "change_units": "mm/month",
                                        "percent": -30.0, "score": -2.25, "status": "scored",
                                        "n_normal": 4, "mark": None},
        "display.groups.leaves.0": {"field": "TLAI", "group": "leaves", "period": "2033-JJA",
                                    "change": -0.0125, "change_units": "m^2/m^2", "percent": None,
                                    "score": -1.5, "status": "scored", "n_normal": 4,
                                    "mark": None},
        "display.not_scored.0": {"field": "TOTVEGC", "group": "plant carbon", "period": "2033-10",
                                 "change": -4.5, "change_units": "gC/m^2", "percent": -0.5,
                                 "score": None, "status": "not scored yet", "mark": None},
        "rows.0": {"field": "RAIN", "period": "2033"}}


def make_report(tmp_path, screen_run=None, figures=None, new_run=True, withheld=None):
    """A made-up whole-domain report and its study folder; (report, 04_analysis)."""
    sd = tmp_path / "study_x"
    adir = sd / "04_analysis"
    adir.mkdir(parents=True)
    plan = {"question_type": "exploratory",
            "analyses": [
                {"id": "P1", "title": "Growth in the wet summers", "tells": "How growth moves",
                 "fields": ["GPP"]},
                {"id": "P2", "title": "Topsoil water", "tells": "How the soil holds the rain",
                 "fields": ["SOILWATER_10CM"]},
                {"id": "P3", "title": "Rain month by month", "tells": "How large the change is",
                 "fields": ["RAIN"]}],
            "competing": [{"title": "Warmth limits first", "text": "Warm weeks could limit growth",
                           "analysis": "P2"}]}
    (adir / "analysis_plan.json").write_text(json.dumps(plan))
    recs = {"L1": dict(_record("L1", None, "2033", ["screen"], call="screen"),
                       label="exploratory", rows=_screen_rows(),
                       screen_csv={"file": "screen_L1.csv", "rows": 120, "sha256": "ab" * 32}),
            "L2": _record("L2", "GPP", "2033-JJA", ["P1"]),
            "L3": _record("L3", "GPP", "2034-JJA", ["P1"]),
            "L4": _record("L4", "SOILWATER_10CM", "2033-JJA", ["P2"]),
            "L5": _record("L5", "RAIN", "2033-07", ["P3"]),
            "L6": _record("L6", "GPP", "2033", ["P1"])}
    scope = {"region": REGION, "year": 2033, "against": "the normal years",
             "normal_years": [2031, 2032, 2033, 2034]}
    if screen_run:
        scope = dict(scope, run=screen_run, against="the control")
    claims = [
        {"claim": "Growth fell by 1.5 gC/m2 in summer 2033 (limitation_structural_9).",
         "finding_id": "required_p1", "evidence": ["L2"], "values": [1.5],
         "caveats": ["limitation_structural_9"], "evidence_class": "observation"},
        {"claim": "Growth fell again in summer 2034.", "finding_id": "required_p1",
         "evidence": ["L3"], "values": [1.5], "caveats": [], "evidence_class": "observation"},
        {"claim": "Growth over the year 2033 fell a little.", "finding_id": "required_p1",
         "evidence": ["L6"], "values": [1.5], "caveats": [], "evidence_class": "observation"},
        {"claim": "Topsoil water was lower in summer 2033.", "finding_id": "required_p2",
         "evidence": ["L4"], "values": [1.5], "caveats": [], "evidence_class": "observation"},
        {"claim": "Rain was lower in July 2033.", "finding_id": "fig_rain",
         "evidence": ["L5"], "values": [1.5], "caveats": [], "evidence_class": "observation"}]
    for c in claims:
        c["evidence_trace"] = {"records": c["evidence"], "class": "observation",
                               "wording_allowed": "what the run's values were", "warnings": []}
    phase = {"closed": "after the last round", "by": "the framework", "file": "planned_phase.json",
             "lookups_before": 9, "plan_file": "analysis_plan.json", "plan_sha256": "cd" * 32,
             "question_type": "exploratory",
             "counts": {"completed": 2, "completed after an earlier failure": 0, "failed": 1,
                        "skipped": 0},
             "analyses": [
                 {"id": "P1", "status": "completed", "reason": "3 required outputs produced",
                  "outputs": [{"id": "O1", "outcome": "produced", "record": "L2"},
                              {"id": "O2", "outcome": "produced", "record": "L3"},
                              {"id": "O3", "outcome": "produced", "record": "L6"}]},
                 {"id": "P2", "status": "completed", "reason": "1 required output produced",
                  "outputs": [{"id": "O1", "outcome": "produced", "record": "L4"}]},
                 {"id": "P3", "status": "failed", "reason": "O2 unmet",
                  "outputs": [{"id": "O1", "outcome": "produced", "record": "L5"},
                              {"id": "O2", "outcome": "unmet", "record": None}]}]}
    report = {
        "kind": "whole_domain", "question": "What if summer rain had been larger in 2033?",
        "run_dir": str(sd), "model": "elm", "status": "supported",
        "generated_at": "2033-01-01T00:00:00+00:00",
        "answer": "Growth fell in both summers (limitation_structural_9). Rain was lower in July.",
        "headline": "Exploratory: Growth fell by 1.5 gC/m2 in summer 2033.",
        "claims": claims,
        "withheld": withheld if withheld is not None else [
            {"claim": "Growth fell most in spring.", "finding_id": "required_p1",
             "struck_by": "trace", "struck_because": "it names MAM, which no record names"}],
        "caveats": [
            {"id": "limitation_structural_9", "severity": "blocking",
             "statement": "a made-up structural limit of GPP numbers", "applies_to": "GPP"},
            {"id": "plan_caveat_1", "severity": "qualify",
             "statement": "The control run holds made-up snow."},
            {"id": "experiment_fact_1", "severity": "context", "statement": "a made-up fact"}],
        "figures": {"comparison": {}, "investigation": {"fig_rain": str(adir / "fig_rain.png")}},
        "evidence_trace": {"question_type": "exploratory", "records": recs,
                           "screen_record": {"scope": scope,
                                             "scoring_rule": "the change over the spread; a "
                                                             "zero change gets no score",
                                             "display_rule": "the largest first (one per field; "
                                                             "ties kept); then the rest"},
                           "wording": {"observation": "what the run's values were"},
                           "headline_rule": "made-up headline rule",
                           "open_causes": [{"cause": "How much of the fall comes from warmth",
                                            "changes": [{"id": "weather_scale",
                                                         "params": {"precip_factor": 1.5,
                                                                    "months": [6, 7, 8],
                                                                    "region": "basin"},
                                                         "meaning": "m", "not_meaning": "n"}]}]},
        "planned_phase": phase,
        "analysis_plan": {"passed": True, "handoff": {"used": True, "approved": True}},
        "acceptance": {"ok": True, "blockers": []},
        "model_setup": {"control": {"name": "box", "description": "the made-up box: one case"},
                        "runs": ([{"run": "control", "reused": True, "changes": []}]
                                 + ([{"run": RUN, "changes": [{"id": "weather_scale"}]}]
                                    if new_run else [])),
                        "years": {"reported": [2033, 2034]}},
        "cost": {"compute": {"runs": {RUN: 7200} if new_run else {},
                             "years_planned": [2033, 2034], "years_passed": [2033, 2034]}},
        "evidence_numbers": {"pass": True, "n_compared": 40,
                             "records": {"L2": {}, "L3": {}, "L4": {}, "L5": {}, "L6": {}},
                             "largest_relative_difference": {"rel_diff": 1e-12}},
        "numbers_check": None,
        "provenance": {"rounds": [{}], "stopped_because": "sufficient",
                       "audit": {"n_claims": 6, "n_struck": 1}}}
    if figures is not None:
        report["report_figures"] = figures
    return report, adir


def figures_record():
    return {"figures": [
        {"id": "over_time", "file": "figure_over_time.png", "number": 1,
         "heading": "Each quantity month by month", "caption": "The new run minus the control run.",
         "comparisons": [1, 3],
         "panels": [{"letter": "a", "field": "RAIN"}, {"letter": "b", "field": "GPP"}]}],
        "tables": [{"id": "periods", "number": 1, "follows": "over_time",
                    "caption": "The same changes over the periods.", "corner": "Quantity",
                    "columns": ["Summer 2033", "Year 2033"],
                    "rows": [{"label": "Growth (GPP), gC/m2", "cells": ["1.5 (2.5 %)", "1.5"],
                              "panel": "1b", "records": ["L2", "L6"]}]}],
        "printed_records": ["L2", "L6"], "not_drawn": ["No map was drawn, for a made-up reason"],
        "problems": []}


def pages(tmp_path, **kw):
    report, adir = make_report(tmp_path, **kw)
    written = RP.write_pages(report, adir)
    return report, {k: Path(v).read_text() for k, v in written.items()}


# ── REPORT.md ───────────────────────────────────────────────────────────────
def test_the_report_has_the_six_sections_of_a_paper_in_order(tmp_path):
    _, p = pages(tmp_path)
    heads = [ln[3:] for ln in p["REPORT.md"].splitlines() if ln.startswith("## ")]
    assert heads == list(RP.SECTIONS)
    assert p["REPORT.md"].startswith("# What if summer rain had been larger in 2033?")


CODE = re.compile(r"\b[PLOEC]\d+\b")


@pytest.mark.parametrize("figs", [None, "record"])
def test_no_code_or_record_name_on_the_three_short_pages(tmp_path, figs):
    report, p = pages(tmp_path, figures=figures_record() if figs else None)
    ids = RP.identifiers(report)
    assert {"limitation_structural_9", "required_p1", "L2", "P1", "O1"} <= ids
    for name in ("REPORT.md", "SCREEN.md", "PLANNED_ANALYSES.md"):
        text = p[name]
        assert not CODE.search(text), (name, CODE.search(text))
        for i in ids:
            assert not re.search(rf"(?<![\w.]){re.escape(i)}(?![\w])", text), (name, i)
    # the details page may show them
    assert "limitation_structural_9" in p["REPORT_DETAILS.md"] and "L2" in p["REPORT_DETAILS.md"]


def test_an_identifier_in_the_analyzers_words_is_taken_out_and_kept_on_the_details(tmp_path):
    _, p = pages(tmp_path)
    assert "Growth fell in both summers. Rain was lower in July." in p["REPORT.md"]
    assert "Growth fell in both summers (limitation_structural_9)." in p["REPORT_DETAILS.md"]


def test_nothing_of_the_old_page_is_lost(tmp_path):
    report, p = pages(tmp_path)
    every = "\n".join(p.values())
    for c in report["claims"]:
        assert c["claim"] in every
    for w in report["withheld"]:
        assert w["claim"] in every and w["struck_because"] in every
    for c in report["caveats"]:
        assert c["id"] in every and c["statement"] in every
    for a in json.loads((Path(report["run_dir"]) / "04_analysis" / "analysis_plan.json")
                        .read_text())["analyses"]:
        assert a["title"] in p["PLANNED_ANALYSES.md"]
    for x in _screen_rows().values():
        if "group" in x:
            assert any(x["field"] in ln and RP.period_words(x["period"]) in ln
                       for ln in p["SCREEN.md"].splitlines()), x
    assert "made-up headline rule" in p["REPORT_DETAILS.md"]          # the wording rules
    assert "screen_L1.csv" in p["SCREEN.md"]


def test_each_finding_is_said_once_on_the_short_pages(tmp_path):
    report, p = pages(tmp_path)
    short = p["REPORT.md"] + p["SCREEN.md"] + p["PLANNED_ANALYSES.md"]
    for c in report["claims"]:
        assert short.count(RP.without_ids(c["claim"], RP.identifiers(report))) <= 1


def test_the_results_choose_findings_by_rule(tmp_path):
    report, p = pages(tmp_path, figures=figures_record())
    text = p["REPORT.md"]
    res = text[text.index("## 3. Results"):text.index("## 4. Limitations")]
    assert "![Figure 1](figure_over_time.png)" in res
    # the caption, then the comparisons it answers on a line of their own
    assert "**Figure 1.** The new run minus the control run.\n\nThis figure answers " \
        "comparisons 1 and 3.\n" in res
    assert "**Table 1.** The same changes over the periods." in res
    assert "| Growth (GPP), gC/m2 | 1.5 (2.5 %) | 1.5 | 1b |" in res
    # comparison 1 belongs to Figure 1: its finding on L2 alone and its finding
    # on L6 alone print in Table 1, so they are left out; the one on L3 stays
    assert "Growth fell by 1.5 gC/m2 in summer 2033" not in res
    assert "Growth over the year 2033 fell a little." not in res
    assert "Growth fell again in summer 2034." in res
    # comparison 3 is answered by Figure 1 too, by its finding on its record
    assert "Rain was lower in July 2033." in res
    # comparison 2 has no figure: its first finding, in Other comparisons
    other = res[res.index("Other comparisons"):]
    assert "- Topsoil water was lower in summer 2033." in other
    assert "No map was drawn, for a made-up reason." in res


def test_at_most_two_findings_under_a_figure(tmp_path):
    report, adir = make_report(tmp_path, figures=dict(figures_record(), tables=[]))
    by_fig, others = RP.choose_findings(report, report["report_figures"],
                                        RP.numbers_of(json.loads(
                                            (adir / "analysis_plan.json").read_text()), report))
    assert by_fig["over_time"] == [1, 2]
    assert others == [(2, 4)]


def test_without_a_figures_record_the_results_are_the_key_findings(tmp_path):
    report, p = pages(tmp_path)
    res = p["REPORT.md"][p["REPORT.md"].index("## 3. Results"):
                         p["REPORT.md"].index("## 4. Limitations")]
    assert "![" not in res and "The main findings" in res
    assert res.count("\n- ") == 5


def test_methods_name_the_new_run_its_change_years_and_time(tmp_path):
    _, p = pages(tmp_path)
    m = p["REPORT.md"][p["REPORT.md"].index("## 2. Methods"):p["REPORT.md"].index("## 3.")]
    assert "| `wet_run` |" in m and "| 2033 and 2034 | 2 hours |" in m
    assert "Each year of the new run, 2033 and 2034, passed the checks of its job." in m
    assert "carried out as the design page listed them" in m
    assert "[PLANNED_ANALYSES.md](PLANNED_ANALYSES.md)" in m


def test_a_record_without_the_chains_years_says_so(tmp_path):
    report, adir = make_report(tmp_path)
    report["cost"]["compute"] = {"runs": {RUN: 7200}}
    RP.write_pages(report, adir)
    text = (adir / "REPORT.md").read_text()
    assert "The record holds no check of the years of the new run." in text
    assert "| `wet_run` |" in text and "| not on record | 2 hours |" in text


def test_a_study_of_the_control_run_alone(tmp_path):
    report, p = pages(tmp_path, new_run=False)
    text = p["REPORT.md"]
    assert "The control run alone, 2033 and 2034." in text
    assert "No new run was made. The control run's existing output was read for 2033 and " \
           "2034." in text
    assert "| New run |" not in text
    assert p["SCREEN.md"].splitlines()[4].startswith(
        "2033 against the normal years 2031 to 2034, over the region `basin`")


def test_limitations_are_short_and_say_how_many_more(tmp_path):
    _, p = pages(tmp_path)
    lim = p["REPORT.md"][p["REPORT.md"].index("## 4."):p["REPORT.md"].index("## 5.")]
    bullets = [ln for ln in lim.splitlines() if ln.startswith("- ")]
    # no design page's records: the Planner's caveat; then the binding one on GPP
    assert bullets == ["- The control run holds made-up snow.",
                       "- A made-up structural limit of GPP numbers."]
    assert "The details page gives 1 more caveat on record in full." in lim


def test_limitations_come_from_the_design_page_first_and_stop_at_six(tmp_path):
    report, adir = make_report(tmp_path)
    sd = adir.parent
    (sd / "strategy.json").write_text(json.dumps({
        "domain_experiment": {"control": {"name": "box"}, "runs": [],
                              "question_limits": [f"A made-up limit number {k}." for k in
                                                  range(1, 7)],
                              "caveats_by": "code"},
        "domain_rulebook": {"controls": {"controls": [{"name": "box"}]},
                            "changes": {"changes": []}}}))
    (sd / "reception.json").write_text(json.dumps({"user_request": "a made-up question"}))
    (sd / "experiment.json").write_text(json.dumps({"experiment": {"check": {"unusual": [
        {"where": "years", "what": "not demonstrated: the run starts from a later saved state"}
    ]}}}))
    text = RP.write_pages(report, adir) and (adir / "REPORT.md").read_text()
    lim = text[text.index("## 4."):text.index("## 5.")]
    bullets = [ln for ln in lim.splitlines() if ln.startswith("- ")]
    assert len(bullets) == RP.MAX_LIMITS
    assert bullets[0] == "- A made-up limit number 1. This is the Planner's judgement."
    # the two limits the cap leaves out are counted and printed on the details
    # page; the caveat one of them says is not counted a second time
    assert "The details page gives 2 more limits and 2 more caveats on record in full." in lim
    d = (adir / "REPORT_DETAILS.md").read_text()
    left = d[d.index("## Limits the report leaves out"):]
    assert "- A made-up structural limit of GPP numbers." in left
    assert "- The new run starts from a later saved state of the control run" in left


def test_the_later_start_and_the_changed_share_are_limits(tmp_path):
    report, adir = make_report(tmp_path)
    sd = adir.parent
    (sd / "experiment.json").write_text(json.dumps({
        "experiment": {"check": {"unusual": [
            {"where": "years", "what": "not demonstrated: the run starts from the control's "
                                       "saved state of 1 January 2033, and no run started from "
                                       "a later saved state has been completed"}]}},
        "regional": {"changed_share": {RUN: {"region": "basin",
                                             "share": {REGION: 0.25, "other": 0.5}}}}}))
    report["caveats"].append({"id": "experiment_unusual_1", "severity": "qualify",
                              "statement": "years: not demonstrated: the run starts from the "
                                           "control's saved state of 1 January 2033, and no run "
                                           "started from a later saved state has been "
                                           "completed"})
    RP.write_pages(report, adir)
    lim = (adir / "REPORT.md").read_text()
    lim = lim[lim.index("## 4."):lim.index("## 5.")]
    assert ("The new run's change applies only in the region `basin`, which covers 25 percent "
            "of the region `basin`, its vegetated land.") in lim
    # the note on a later start is one plain sentence, its date the record's
    assert "- " + RP.LATER_START.format(date="1 January 2033") + "\n" in lim
    assert "other" not in lim                       # a region no record uses
    assert "The details page gives 1 more caveat" in lim


def test_next_steps_say_the_run_in_words_never_its_settings(tmp_path):
    report, p = pages(tmp_path, withheld=[])
    nxt = p["REPORT.md"][p["REPORT.md"].index("## 5."):p["REPORT.md"].index("## 6.")]
    assert "- This study leaves open how much of the fall comes from warmth." in nxt
    assert "rain and snow times 1.5 in June to August over the region `basin`" in nxt
    assert "precip_factor" not in nxt and "{" not in nxt


def test_a_cause_is_not_quoted_when_the_audit_withheld_a_finding(tmp_path):
    _, p = pages(tmp_path)
    nxt = p["REPORT.md"][p["REPORT.md"].index("## 5."):p["REPORT.md"].index("## 6.")]
    assert "warmth" not in nxt and "leaves open whether one of these results" in nxt


def test_checks_and_records_say_what_the_second_reading_covered(tmp_path):
    from agents.analysis import numbers_check as NC
    report, adir = make_report(tmp_path)
    (adir.parent / "DESIGN_REVIEW.md").write_text("a made-up design page")
    RP.write_pages(report, adir)
    ch = (adir / "REPORT.md").read_text()
    ch = ch[ch.index("## 6."):]
    # the check's own sentence, unchanged; no figure was drawn
    assert NC.evidence_sentence(report["evidence_numbers"], figures=False) in ch
    assert "The audit kept 5 findings and left out 1." in ch
    assert RP.JUDGEMENT_WORDS in ch
    for link in ("(REPORT_DETAILS.md)", "(SCREEN.md)", "(PLANNED_ANALYSES.md)",
                 "(../DESIGN_REVIEW.md)"):
        assert link in ch


def test_checks_use_the_second_readings_own_sentence(tmp_path, monkeypatch):
    from agents.analysis import numbers_check as NC
    seen = []

    def sentence(rec, figures=True):
        seen.append(figures)
        return "A made-up sentence of the check."
    monkeypatch.setattr(NC, "evidence_sentence", sentence)
    _, p = pages(tmp_path, figures=figures_record())
    assert "A made-up sentence of the check." in p["REPORT.md"]
    assert seen == [True]                       # the report draws figures


# ── SCREEN.md ───────────────────────────────────────────────────────────────
def test_the_screen_of_a_new_run_says_what_it_compares(tmp_path):
    _, p = pages(tmp_path, screen_run=RUN)
    s = p["SCREEN.md"]
    assert ("The new run `wet_run` against the control run over the region `basin`, its "
            "vegetated land, every quantity and period of 2033.") in s
    assert RP.SCORE_WORDS in s and RP.LOOKING_WORDS in s
    scored = s[s.index("## The rows with a score"):s.index("## The rows not scored")]
    lines = scored.splitlines()
    # the empty Mark column is left out, whole lines compared; a quantity and
    # a unit in the words of field_words
    assert "| Group | Quantity | Period | Change | Units | Percent | Score | Normal years |" \
        in lines
    assert (f"| weather in | {RP.field_name('RAIN')} | July 2033 | -12.5 | "
            f"{RP.units_name('mm/month')} | -30 | -2.25 | 4 |") in lines
    assert (f"| leaves | {RP.field_name('TLAI')} | summer 2033 | -0.0125 | "
            f"{RP.units_name('m^2/m^2')} |  | -1.5 | 4 |") in lines
    assert (f"| {RP.field_name('TOTVEGC')} | October 2033 | -4.5 | {RP.units_name('gC/m^2')} "
            f"| -0.5 | not scored yet |") in s.splitlines()
    # the rules at the end, the fixed sentences, after the link to the table
    for x in RP.SCORING_SENTENCES + RP.DISPLAY_SENTENCES:
        assert f"- {x}\n" in s
    assert s.index("## How the rows are scored") > s.index("[The whole table](screen_L1.csv)")


def test_an_older_screen_record_is_read_as_the_control_runs(tmp_path):
    _, p = pages(tmp_path)
    assert "2033 against the normal years 2031 to 2034" in p["SCREEN.md"]
    assert "every quantity and period of the control run" in p["SCREEN.md"]


def test_no_screen_no_screen_page_and_an_earlier_one_is_set_aside(tmp_path):
    report, adir = make_report(tmp_path)
    del report["evidence_trace"]["records"]["L1"]
    (adir / "SCREEN.md").write_text("an earlier screen")
    written = RP.write_pages(report, adir)
    assert "SCREEN.md" not in written and not (adir / "SCREEN.md").exists()
    assert list((adir / "superseded").rglob("SCREEN.md"))
    assert "(SCREEN.md)" not in (adir / "REPORT.md").read_text()


# ── PLANNED_ANALYSES.md ─────────────────────────────────────────────────────
def test_the_planned_comparisons_in_words(tmp_path):
    _, p = pages(tmp_path, figures=figures_record())
    s = p["PLANNED_ANALYSES.md"]
    assert "approved with the yes" in s and "names it by its hash" in s
    assert ("| 1 | Growth in the wet summers | How growth moves | completed, 3 of 3 numbers "
            "produced | Figure 1b and Table 1 |") in s
    assert "| 2 | Topsoil water | How the soil holds the rain | completed, 1 of 1 numbers " \
           "produced | a finding in the Results |" in s
    assert "| 3 | Rain month by month | How large the change is | failed, 1 of 2 numbers " \
           "produced, 1 not produced | Figure 1a |" in s
    assert "- **Warmth limits first.** Warm weeks could limit growth. Comparison 2 reads it." \
        in s
    assert ("The planned phase closed after the last round. Of its 3 comparisons, 2 were "
            "completed and 1 failed.") in s


def test_a_comparison_shown_nowhere_on_the_report_points_to_the_details(tmp_path):
    report, adir = make_report(tmp_path)
    report["claims"] = report["claims"][:1]
    RP.write_pages(report, adir)
    s = (adir / "PLANNED_ANALYSES.md").read_text()
    assert "| a finding in the Results |" in s and "| the details page |" in s


def test_a_hypothesis_not_fully_tested_is_said_in_words(tmp_path):
    report, adir = make_report(tmp_path)
    report["acceptance"] = {"ok": False, "blockers": [
        {"analysis": "P3", "status": "failed", "reason": "O2 unmet"}]}
    report["planned_phase"]["expectations"] = [
        {"id": "E1", "analysis": "P3", "field": "RAIN", "sign": "-", "text": "Rain falls"}]
    RP.write_pages(report, adir)
    text = (adir / "REPORT.md").read_text()
    assert "**The hypothesis was not fully tested.** Comparison 3 failed, and the plan's " \
           "expectations need it." in text
    s = (adir / "PLANNED_ANALYSES.md").read_text()
    assert "| Rain falls | 3 | failed | 0 | 0 |" in s
    assert not CODE.search(text + s)


# ── REPORT_DETAILS.md ───────────────────────────────────────────────────────
def test_the_details_page_holds_everything_else(tmp_path):
    report, p = pages(tmp_path)
    d = p["REPORT_DETAILS.md"]
    for head in ("## The headline", "## Every kept finding, with its records (5)",
                 "## The findings left out, and why (1)", "## Every caveat in full",
                 "## How the claims are worded", "## What code checked, and what it did not",
                 "## The two numbers checks", "## The model set-up in full",
                 "## The remaining uncertainty, with the runs as the server takes them",
                 "## The planned phase as recorded",
                 "## The figures the Analyzer drew in its own step", "## Provenance"):
        assert head in d, head
    assert RP.FIXED_RULES in d
    assert "- fig_rain: fig_rain.png" in d                         # by file name, no path
    assert str(tmp_path) not in d.split("## The figures the Analyzer drew")[1]
    assert "- comparison 3 (P3): failed; O2 unmet" in d
    assert "weather_scale {\"months\": [6, 7, 8]" in d              # the settings, here only


# ── step4_report.write ──────────────────────────────────────────────────────
def test_write_gives_four_pages_for_a_whole_domain_report(tmp_path):
    report, adir = make_report(tmp_path)
    S4.write(report, adir, slides=False)
    for name in ("analysis.json", "REPORT.md", "REPORT_DETAILS.md", "SCREEN.md",
                 "PLANNED_ANALYSES.md"):
        assert (adir / name).is_file(), name
    assert "## 1. Summary" in (adir / "REPORT.md").read_text()


def test_a_withheld_report_keeps_todays_page_and_sets_earlier_pages_aside(tmp_path):
    report, adir = make_report(tmp_path)
    for name in ("REPORT_DETAILS.md", "SCREEN.md", "PLANNED_ANALYSES.md"):
        (adir / name).write_text("an earlier page")
    report["withheld_because"] = "a made-up reason"
    report["evidence_trace"]["withheld_because"] = "a made-up reason"
    S4.write(report, adir, slides=False)
    assert (adir / "REPORT.md").read_text() == S4.render_markdown(report)
    for name in ("REPORT_DETAILS.md", "SCREEN.md", "PLANNED_ANALYSES.md"):
        assert not (adir / name).exists()
        assert list((adir / "superseded").rglob(name))


def test_another_kind_of_report_keeps_its_one_page(tmp_path):
    report, adir = make_report(tmp_path)
    report["kind"] = None
    S4.write(report, adir, slides=False)
    assert (adir / "REPORT.md").read_text() == S4.render_markdown(report)
    assert not (adir / "REPORT_DETAILS.md").exists()


def test_pages_that_fail_leave_the_one_page(tmp_path, monkeypatch):
    report, adir = make_report(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("made-up failure")
    monkeypatch.setattr(RP, "write_pages", boom)
    S4.write(report, adir, slides=False)
    assert (adir / "REPORT.md").read_text() == S4.render_markdown(report)


def test_the_old_page_still_has_its_fourteen_sections(tmp_path):
    report, _ = make_report(tmp_path)
    page = S4.render_markdown(report)
    heads = [ln for ln in page.splitlines() if ln.startswith("## ")]
    assert heads[:4] == ["## The answer", "## Model setup",
                         "## What the run found (5 audited claims)",
                         "## Caveats that bind this answer"]
    assert "## The screen (exploratory)" in heads and heads[-1] == "## Provenance"


# ── the wiring in step4_report.build ────────────────────────────────────────
def _joined(monkeypatch, build):
    """report_figures.build and numbers_check.check_evidence replaced by
    made-up ones; the calls the second reading received."""
    from agents.analysis import numbers_check as NC
    from agents.analysis import report_figures as RF
    calls = []

    def check_evidence(study_dir, claims, registry=None, write=True, also=()):
        calls.append(list(also))
        return {"pass": True, "printed": sorted(also)}
    monkeypatch.setattr(NC, "check_evidence", check_evidence)
    monkeypatch.setattr(RF, "build", build)
    return calls


def test_the_printed_records_go_to_the_second_reading(tmp_path, monkeypatch):
    seen = {}

    def build(study_dir, plan, out_dir, ctx=None):
        seen["ctx"] = ctx
        return {"figures": [], "tables": [], "printed_records": ["L2", "L9"]}
    calls = _joined(monkeypatch, build)
    rf, rec = S4.figures_and_reading(tmp_path, {}, tmp_path, [{"evidence": ["L2"]}], ctx="c")
    assert rf["printed_records"] == ["L2", "L9"] and seen["ctx"] == "c"
    assert calls == [["L2", "L9"]] and rec == {"pass": True, "printed": ["L2", "L9"]}


def test_the_figures_never_stop_the_report(tmp_path, monkeypatch):
    def build(study_dir, plan, out_dir, ctx=None):
        raise ValueError("made-up")
    calls = _joined(monkeypatch, build)
    rf, rec = S4.figures_and_reading(tmp_path, {}, tmp_path, [])
    assert rf["figures"] == [] and "made-up" in rf["problems"][0]
    # the second reading still runs, over the cited records alone
    assert calls == [[]] and rec["pass"]


def test_a_figures_build_that_returns_no_record_gives_an_empty_one(tmp_path, monkeypatch):
    _joined(monkeypatch, lambda study_dir, plan, out_dir, ctx=None: None)
    rec = S4.report_figures(tmp_path, {}, tmp_path)
    assert rec["figures"] == [] and rec["printed_records"] == []
    assert rec["problems"] == ["the report's figures returned no record"]


def test_a_second_reading_that_raises_does_not_pass(tmp_path, monkeypatch):
    from agents.analysis import numbers_check as NC
    _joined(monkeypatch, lambda study_dir, plan, out_dir, ctx=None: {"printed_records": []})

    def boom(*a, **k):
        raise OSError("made-up")
    monkeypatch.setattr(NC, "check_evidence", boom)
    _, rec = S4.figures_and_reading(tmp_path, {}, tmp_path, [])
    assert rec["pass"] is False and rec["error"] == "OSError: made-up"


def test_no_join_guard_is_left():
    root = Path(__file__).resolve().parents[1]
    for rel in ("src/agents/analysis/report_pages.py", "src/agents/analysis/step4_report.py",
                "tools/redraw_report.py"):
        text = (root / rel).read_text()
        assert "# JOIN" not in text and "inspect.signature" not in text, rel


# ── words ───────────────────────────────────────────────────────────────────
def test_words_in_place_of_codes():
    assert RP.period_words("2033-09") == "September 2033"
    from agents.analysis import field_words as FW
    assert RP.period_words("2033-DJF") == FW.period_words("2033-DJF")
    assert RP.period_words("WY2033") == "water year 2033"
    assert RP.period_words("2033") == "the year 2033"
    assert RP.span_words([2033, 2034]) == "2033 and 2034"
    assert RP.span_words([2031, 2032, 2033]) == "2031 to 2033"
    assert RP.numbers_words([1, 2, 3, 4, 6, 7, 8]) == "comparisons 1 to 4 and 6 to 8"
    assert RP.numbers_words([5]) == "comparison 5"
    assert RP.numbers_words([3, 7, 8]) == "comparisons 3, 7 and 8"
    assert RP.numbers_words([1, 2]) == "comparisons 1 and 2"
    assert RP.region_words("watershed, vegetated") == "the watershed's vegetated land"
    assert RP.region_words("watershed, vegetated, 1000-1500 m") == \
        "the watershed's vegetated land at 1000-1500 m"


def test_no_dash_or_mid_sentence_colon_in_the_fixed_words(tmp_path):
    _, p = pages(tmp_path, figures=figures_record())
    for name in ("REPORT.md", "PLANNED_ANALYSES.md"):
        assert chr(0x2014) not in p[name] and chr(0x2013) not in p[name]
    fixed = [RP.SCORE_WORDS, RP.LOOKING_WORDS, RP.JUDGEMENT_WORDS, RP.SUPPORT_WORDS,
             RP.NO_CAUSE] + list(RP.QUESTION_KIND.values())
    for s in fixed:
        assert ":" not in s and chr(0x2014) not in s


# ── the study record ────────────────────────────────────────────────────────
def test_the_study_record_names_the_three_new_pages_after_the_report(tmp_path):
    from agents import study_record as SRD
    for name in ("REPORT_DETAILS.md", "SCREEN.md", "PLANNED_ANALYSES.md", "report_figures.json",
                 "figure_over_time.png", "figure_over_time_2.png", "report_lookups.jsonl"):
        assert "no role on record" not in SRD.file_role(f"04_analysis/{name}"), name
    roles = [p for p, _ in SRD.ROLES]
    assert roles.index("04_analysis/REPORT.md") < roles.index("04_analysis/REPORT_DETAILS.md")
    report, adir = make_report(tmp_path)
    S4.write(report, adir, slides=False)
    sec = SRD._analyzer_lines(adir.parent, adir.parent)
    i = next(k for k, ln in enumerate(sec) if ln.startswith("Its report: [REPORT.md]"))
    assert [ln.split(":")[0] for ln in sec[i + 1:i + 4]] == [
        "Beside it, everything the report leaves out", "Beside it, the first look at every "
        "quantity", "Beside it, each approved comparison and where the report shows it"]


# ── tools/redraw_report.py ──────────────────────────────────────────────────
def _redraw_tool():
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "tools" / "redraw_report.py"
    spec = importlib.util.spec_from_file_location("redraw_report", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def saved_study(tmp_path, monkeypatch):
    """A made-up saved study whose step 3 is on record, with no model to ask."""
    report, adir = make_report(tmp_path)
    (adir / "interpretation.json").write_text(json.dumps({"claims": report["claims"]}))
    old = dict(report, cost={"llm": {"analyzer": {"calls": 3}}})
    (adir / "analysis.json").write_text(json.dumps(old))
    (adir / "REPORT.md").write_text("the earlier report")
    (adir / "figure_over_time.png").write_text("an earlier figure")
    import agents.llm_agent as L

    class NoModel:
        def __init__(self, *a, **k):
            raise AssertionError("a model was asked")
    monkeypatch.setattr(L, "SimpleLLMClient", NoModel)
    from agents.analysis import step0_context
    monkeypatch.setattr(step0_context, "load", lambda rd: object())
    return report, adir


def test_the_redraw_sets_the_earlier_report_aside_and_writes_the_pages(saved_study, monkeypatch):
    report, adir = saved_study
    tool = _redraw_tool()
    seen = {}

    def build(ctx, comparison, investigation, interp, run_dir, **kw):
        seen.update(kw, claims=interp["claims"])
        return dict(report, cost={"llm": {"measured": False}})
    monkeypatch.setattr(S4, "build", build)
    out = tool.redraw(adir.parent, stamp="S1")
    before = adir / "before_S1"
    assert (before / "REPORT.md").read_text() == "the earlier report"
    assert (before / "figure_over_time.png").is_file() and (before / "analysis.json").is_file()
    assert out["pages"] == ["REPORT.md", "REPORT_DETAILS.md", "PLANNED_ANALYSES.md", "SCREEN.md"]
    new = json.loads((adir / "analysis.json").read_text())
    assert new["cost"]["llm"] == {"analyzer": {"calls": 3}}         # the earlier pass's spend
    assert new["redrawn"]["earlier"] == "04_analysis/before_S1/"
    assert seen["stopped_because"] == "sufficient" and len(seen["claims"]) == 5
    assert "redraw" in tool.__doc__ and "copy" in tool.__doc__.lower()


def test_a_redraw_that_fails_puts_the_earlier_report_back(saved_study, monkeypatch):
    _, adir = saved_study
    tool = _redraw_tool()

    def build(*a, **k):
        raise OSError("made-up full disk")
    monkeypatch.setattr(S4, "build", build)
    with pytest.raises(OSError):
        tool.redraw(adir.parent, stamp="S2")
    assert (adir / "REPORT.md").read_text() == "the earlier report"
    assert (adir / "analysis.json").is_file() and not (adir / "before_S2").exists()
