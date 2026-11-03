"""The reports where the packages meet: wording that fits the study, the
numbers check seen on the page, burned area as a table, and the records of
what a pass wrote (2026-11-03). No LLM.

    pytest tests/test_report_polish.py -v

Found by an end-to-end run of the three demos: every number reproduced, and
these are the records and words around them.
"""
import csv
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
from agents.analysis import domain_report as DR      # noqa: E402
from agents.analysis import numbers_check as N       # noqa: E402
from agents import study_record as SR                # noqa: E402
import test_domain_report as T                       # noqa: E402
import test_domain_months_report as M                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _build(tmp_path, table, runs=None, analysis=None, caveats=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rd = T._run_dir(tmp_path, table, runs or [T.CONTROL_ROW, T.NOFIRE],
                    analysis or T.ANALYSIS, caveats=caveats)
    DR.build(rd)
    return rd, (rd / "04_analysis" / "DOMAIN_REPORT.md").read_text()


def _event_table():
    return M.with_months(T._table(with_twin=False, with_event=True), event=True)


EVENT = {"regions": [{"name": "vegetated cells"}], "variables": ["GPP"],
         "normal_years": [2016]}


def _check_record(rd, ok=True, n=2340, w=3.33e-12):
    """A numbers check's record as numbers_check.check writes it, newer than
    the table it checked."""
    out = rd / "04_analysis"
    out.mkdir(parents=True, exist_ok=True)
    rec = {"kind": "numbers_check", "pass": ok, "n_compared": n,
           "n_mismatched": 0 if ok else 3, "checked_at": "2026-11-03T14:59:14",
           "largest_relative_difference": {"rel_diff": w, "section": "runs"}}
    (out / N.CHECK_JSON).write_text(json.dumps(rec))
    return rec


# ── 1. a passing check is seen ──────────────────────────────────────────────
class TestTheNumbersCheckOnThePages:

    def test_the_numbers_report_says_the_check_passed(self, tmp_path):
        rd = T._run_dir(tmp_path, T._table(), [T.CONTROL_ROW, T.NOFIRE], T.ANALYSIS)
        (rd / "03_results").mkdir()
        (rd / "03_results" / "regional.json").write_text(json.dumps(T._table()))
        _check_record(rd)
        DR.build(rd)
        page = (rd / "04_analysis" / "DOMAIN_REPORT.md").read_text().splitlines()
        assert ("**Numbers check: passed.** Every number in the answer and under The numbers "
                "is read from 03_results/regional.json (a percent there is one of its "
                "numbers divided by another), and every number in that table was made again "
                "from the model output by an independent reading before this page was "
                "written: 2,340 numbers compared, the largest relative difference 3.33e-12 "
                "(04_analysis/numbers_check.json).") in page
        rec = json.loads((rd / "04_analysis" / DR.REPORT_JSON).read_text())
        assert rec["numbers_check"]["n_compared"] == 2340

    def test_no_check_is_said_and_an_older_check_is_none(self, tmp_path):
        rd = T._run_dir(tmp_path, T._table(), [T.CONTROL_ROW, T.NOFIRE], T.ANALYSIS)
        DR.build(rd)
        none = ("**Numbers check: none on record.** There is no "
                "04_analysis/numbers_check.json for the current 03_results/regional.json, so "
                "its numbers were not made a second time.")
        assert none in (rd / "04_analysis" / "DOMAIN_REPORT.md").read_text()
        # a check older than the table checked an earlier table
        _check_record(rd)
        (rd / "03_results").mkdir()
        t = rd / "03_results" / "regional.json"
        t.write_text(json.dumps(T._table()))
        old = time.time() - 60
        os.utime(rd / "04_analysis" / N.CHECK_JSON, (old, old))
        assert N.summary(rd) is None
        DR.build(rd)
        assert none in (rd / "04_analysis" / "DOMAIN_REPORT.md").read_text()

    def test_the_analyzer_report_says_it_too(self):
        from agents.analysis import step4_report as S4
        rep = {"question": "q", "kind": "whole_domain", "model": "elm", "status": "supported",
               "headline": "h", "answer": "a", "claims": [], "caveats": [], "withheld": [],
               "provenance": {}, "cost": {},
               "numbers_check": {"pass": True, "n_compared": 81, "n_mismatched": 0,
                                 "largest_relative_difference": 1.2e-13,
                                 "record": "04_analysis/numbers_check.json"}}
        lines = S4.render_markdown(rep).splitlines()
        line = ("**Numbers check: passed.** This analysis reads its numbers from "
                "03_results/regional.json (each claim below shows how its values trace to "
                "it), and every number in that table was made again from the model output by "
                "an independent reading before this page was written: 81 numbers compared, "
                "the largest relative difference 1.2e-13 (04_analysis/numbers_check.json).")
        assert line in lines and lines.index(line) < lines.index("## The answer")
        # a column study's report has no such line
        rep.update(kind=None)
        assert "Numbers check" not in S4.render_markdown(rep)

    def test_the_study_record_states_the_verdict(self, tmp_path):
        rd = tmp_path / "run"
        (rd / "03_results").mkdir(parents=True)
        (rd / "03_results" / "regional.json").write_text(json.dumps(T._table()))
        _check_record(rd, ok=False)
        page = SR.render(rd)
        assert ("The numbers check, `04_analysis/numbers_check.json`: **did not pass**. 3 of "
                "2,340 numbers of `03_results/regional.json` did not reproduce when made "
                "again from the model output. Checked 2026-11-03T14:59:14.") in page
        (rd / "04_analysis" / N.CHECK_JSON).unlink()
        assert "The numbers check: none on record" in SR.render(rd)


class TestTheFileIndex:

    def test_every_file_of_the_numbers_has_a_role(self, tmp_path):
        rd = tmp_path / "run"
        for rel in ("04_analysis/numbers_check.json", "03_results/regional_monthly.csv",
                    "04_analysis/superseded/20261103_120000/REPORT.md", "ensemble_B.sbatch"):
            (rd / rel).parent.mkdir(parents=True, exist_ok=True)
            (rd / rel).write_text("x")
        page = SR.render(rd)
        assert "no role on record" not in page
        # job B's notice is written after this page: named, and not counted
        assert ("Written after this page: `STUDY_REPORT.txt`, the notice "
                "tools/notify_study.py writes when job B ends") in page
        assert "## Every file in this folder (5)" in page
        # one already in the folder is an earlier pass's
        (rd / "STUDY_REPORT.txt").write_text("OK")
        page = SR.render(rd)
        assert ("| `STUDY_REPORT.txt` | the notice tools/notify_study.py wrote when the study "
                "last ended; it is written after this page, so this one is from an earlier "
                "pass |") in page
        assert "Written after this page" not in page


# ── 2. burned area, 5. wording that fits the study ──────────────────────────
class TestBurnedAreaIsATable:

    def test_a_fire_study_tables_its_burned_area_in_km2(self, tmp_path):
        t = T._table()
        for r in t["runs"].values():
            r["fields"] = dict(r["fields"], burned_area_km2={
                "units": "km2", "how": "sum", "stream": "h0",
                "note": "the model's own burned area"})
            for reg, by in r["values"].items():
                by["burned_area_km2"] = {y: f * T.REGIONS[reg]["area_km2"]
                                         for y, f in by["FAREA_BURNED"].items()}
        for reg, d in t["differences"]["nofire"].items():
            d["burned_area_km2"] = {y: -x * T.REGIONS[reg]["area_km2"]
                                    for y, x in t["runs"]["control"]["values"][reg]
                                    ["FAREA_BURNED"].items()}
        rd, page = _build(tmp_path, t)
        sec = page[page.index("### burned_area_km2 (km2, total over the year)"):]
        sec = sec[:sec.index("\n### ", 1)]
        assert "| vegetated cells | 0.6 | 0.3 |" in sec
        assert "| watershed, vegetated | -0.24 (-100 %) | -0.12 (-100 %) |" in sec
        # after the burned fraction it is made from
        assert page.index("### FAREA_BURNED") < page.index("### burned_area_km2")
        rec = json.loads((rd / "04_analysis" / DR.REPORT_JSON).read_text())
        assert rec["variables"] == ["GPP", "FAREA_BURNED", "burned_area_km2"]

    def test_a_study_without_fire_has_none(self, tmp_path):
        rd, page = _build(tmp_path, T._table())
        assert "### burned_area_km2" not in page


class TestWordingThatFitsTheStudy:

    def test_the_control_alone_has_no_difference_column(self, tmp_path):
        rd, page = _build(tmp_path, _event_table(), runs=[T.CONTROL_ROW], analysis=EVENT)
        rows = list(csv.DictReader(open(rd / "03_results" / DR.MONTHLY_CSV)))
        assert "minus_control" not in rows[0]
        said = ("Every month and season of the control, by region and field, and the mean of "
                "each month and season over the normal years (2016): "
                "[03_results/regional_monthly.csv](../03_results/regional_monthly.csv). This "
                "study has no run beside the control, so the table has no column of "
                "differences from it.")
        assert page.count(said) == 2                # the numbers, and where they come from
        assert "with each run's difference from the control" not in page

    def test_a_run_beside_the_control_keeps_its_column(self, tmp_path):
        rd, page = _build(tmp_path, M.with_months(T._table()))
        rows = list(csv.DictReader(open(rd / "03_results" / DR.MONTHLY_CSV)))
        assert "minus_control" in rows[0]
        assert ("Every month and season of every run, region and field, with each run's "
                "difference from the control") in page

    def test_elevations_are_named_only_with_bands(self, tmp_path):
        rd, page = _build(tmp_path, M.with_months(T._table()))
        assert ("The vegetated cells are read from the control's surface file, "
                "surfdata_x.nc.") in page
        t = M.with_months(T._table())
        t["provenance"] = json.loads(json.dumps(t["provenance"]))
        t["provenance"]["regions"]["watershed, vegetated, 1000-1500 m"] = {
            "cells": 4, "area_km2": 2.0, "rule": "inside, vegetated, TOPO 1000-1500 m",
            "sources": ["w.geojson", "surfdata_x.nc"]}
        rd, page = _build(tmp_path / "b", t)
        assert ("The vegetated cells and the elevations are read from the control's surface "
                "file, surfdata_x.nc.") in page

    def test_each_caveat_says_whose_it_is(self, tmp_path):
        rd, page = _build(tmp_path, T._table(),
                          caveats=["the difference is the model's own fire, not a real one"])
        tail = page[page.index("## Caveats"):page.index("## Where the numbers come from")]
        heads = [x for x in tail.splitlines() if x.endswith(":") and not x.startswith("-")]
        assert heads == [
            "The model server's change list (what each run's difference does not mean):",
            "The plan (strategy.json, the Planner's caveats):",
            "The model server's check of the experiment:",
            "The model server's notes on the fields:"]
        # both are kept, each under its own source
        server = tail.index("- Run nofire, fire_off: its difference from the control is NOT")
        plan = tail.index("- the difference is the model's own fire, not a real one")
        assert tail.index(heads[0]) < server < tail.index(heads[1]) < plan
        rec = json.loads((rd / "04_analysis" / DR.REPORT_JSON).read_text())
        assert [g["source"] for g in rec["caveat_groups"]] == ["changes", "plan", "check",
                                                               "fields"]
        assert len(rec["caveats"]) == sum(len(g["items"]) for g in rec["caveat_groups"])

    def test_observations_an_older_plan_names_are_said_once(self, tmp_path):
        rd = T._run_dir(tmp_path, T._table(), [T.CONTROL_ROW, T.NOFIRE],
                        dict(T.ANALYSIS, observations=["modis_gpp", "modis_lai"]))
        (rd / "DESIGN_REVIEW.md").write_text("# the design")
        DR.build(rd)
        page = (rd / "04_analysis" / "DOMAIN_REPORT.md").read_text()
        line = ("- observations named by the plan (modis_gpp, modis_lai): not part of the "
                "answer, which is read from the model's output only; how the control compares "
                "with observations is shown before the design, on the control's card and on "
                "the design page (DESIGN_REVIEW.md)")
        assert page.count(line) == 1 and page.count("observations named by the plan") == 1
        rd, page = _build(tmp_path / "b", T._table())
        assert "observations named by the plan" not in page

    def test_a_kilogram_is_not_a_kelvin(self, tmp_path):
        assert DR._no_percent("K") and DR._no_percent("K/s")
        assert not DR._no_percent("kg/m2") and not DR._no_percent("kg m-2")
        assert not DR._no_percent("gC/m^2")


# ── the notice says which reports exist ─────────────────────────────────────
def _notify():
    spec = importlib.util.spec_from_file_location("notify_study",
                                                  ROOT / "tools" / "notify_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _folder(tmp_path, analyze, reports=()):
    rd = tmp_path / "run"
    (rd / "04_analysis").mkdir(parents=True)
    stages = {s: {"status": "done"} for s in ("materialize", "build_case_inputs",
                                              "build_cases", "run", "extract", "package")}
    stages["analyze"] = analyze
    (rd / "run_state.json").write_text(json.dumps({"model": "elm", "stages": stages}))
    (rd / "experiment.json").write_text("{}")
    (rd / "04_analysis" / "domain_report.json").write_text("{}")
    for r in reports:
        (rd / "04_analysis" / r).write_text("# r")
    return rd


class TestTheNoticeNamesItsReports:

    def test_the_numbers_report_alone(self, tmp_path):
        rd = _folder(tmp_path, {"status": "done",
                                "analyzer": "switched off (analysis_llm: false)"},
                     ["DOMAIN_REPORT.md"])
        body, rc = _notify().build(str(rd))
        assert rc == 0 and body.splitlines()[0] == (
            "OK: the numbers report is written; there is no Analyzer report from this pass "
            "(the Analyzer did not run: switched off (analysis_llm: false))")
        assert "reports: DOMAIN_REPORT.md (the numbers report)" in body
        assert "the analysis is written" not in body

    def test_an_analyzer_report_from_an_earlier_pass(self, tmp_path):
        rd = _folder(tmp_path, {"status": "done",
                                "analyzer": "switched off (analysis_llm: false)"},
                     ["DOMAIN_REPORT.md", "REPORT.md"])
        body, _ = _notify().build(str(rd))
        assert body.startswith("OK: the numbers report is written")
        assert ("reports: DOMAIN_REPORT.md (the numbers report); REPORT.md (the Analyzer's "
                "report, from an earlier pass: the Analyzer did not run in this one)") in body

    def test_both_reports_of_this_pass(self, tmp_path):
        rd = _folder(tmp_path, {"status": "done"}, ["DOMAIN_REPORT.md", "REPORT.md"])
        body, rc = _notify().build(str(rd))
        assert rc == 0 and "the analysis is written" in body.splitlines()[0]
        assert ("reports: DOMAIN_REPORT.md (the numbers report); REPORT.md (the Analyzer's "
                "report)") in body
