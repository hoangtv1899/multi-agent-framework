"""The study record: a run folder's STUDY_RECORD.md, drawn by code (2026-11-03).

    pytest tests/test_study_record.py -v

Synthetic run folders, built here file by file in the shapes the framework
writes, are rendered and the page is checked against them: the question and
Reception's exchange, the decision, the server's check, each run's jobs with
their verdicts and times, the numbers, the Analyzer, and an index of every
file. Three shapes the page must survive are built on purpose: the control
alone, a chain that stopped part way, and an analysis that was skipped. The
last class runs the manager's own pipeline (no LLM, no job submitted) to show
the one call site writes the page and cannot fail the study.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agents import study_record as SR                       # noqa: E402

QUESTION = "How would a warmer summer change plant growth over the basin in 2015?"
MARK = "\n\nRevision requested at the design review: "


def _w(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data, indent=1))


def _check(stage, job, ok=True, reasons=(), years=None):
    rec = {"stage": stage, "job": job, "pass": ok, "reasons": list(reasons)}
    if years is not None:
        rec["years"] = years
    return rec


def _year(y, ok=True, expected=(True,), ref_reasons=("TOTVEGC: bias -0.25 %",)):
    return {str(y): {"year": y, "pass": ok, "reasons": [],
                     "expected": [{"change": "warm", "pass": e} for e in expected],
                     "reference": {"pass": not ref_reasons, "gated": False,
                                   "reasons": list(ref_reasons)}}}


def _run_folder(tmp_path, name="study_a", runs=("warm",), years=(2015,), stop_at=None,
                analysis="done"):
    """A finished whole-domain study. `runs` are the runs besides the reused
    control; `stop_at` a year whose check fails (the chain stops there);
    `analysis` done, skipped or off."""
    rd = tmp_path / name
    _w(rd / "reception.json", {
        "route": {"action": "design", "questions": []},
        "brief": {"design_archetype": "whole_domain", "model": "elm",
                  "domain_experiment": {"control": {"name": "basin", "reuse": True},
                                        "runs": [{"name": r} for r in runs],
                                        "years": list(years), "unresolved": []}},
        "rounds": 5,
        "revision_of": {"user_answers": [{"question": "Which months?",
                                          "answer": "June to August"}],
                        "changes": ["Report it by elevation band."]},
        "user_request": QUESTION + MARK + "Report it by elevation band."})
    _w(rd / "DESIGN_REVIEW.md", "# Design review\n\nbody\n\n---\n"
       "**Decision:** accepted at the terminal at 2026-11-01T10:00:00.\n")
    _w(rd / "strategy.json", {"domain_experiment": {
        "control": {"name": "basin", "reuse": True},
        "runs": [{"name": r, "changes": [{"id": "weather_scale",
                                          "params": {"temperature_K": 2.0}}]}
                 for r in runs], "years": list(years)}})
    _w(rd / "01_inputs" / "domain_runs.json", {
        "ok": True, "control": {"name": "basin", "reuse": True},
        "runs": [{"name": r, "changes": ["weather_scale"]} for r in runs],
        "years": list(years),
        "cost": {"runs_executed": len(runs), "model_years": len(runs) * len(years),
                 "node_hours": 0.8 * len(runs) * len(years),
                 "note": "the control is reused: it spends nothing"},
        "unusual": []})
    _w(rd / "domain_case" / "inputs_check.json", {"pass": True, "reasons": []})
    job = 1000
    for r in runs:
        rec = rd / "domain_case" / r
        stages = ["case", "smoke"] + [f"y{y}" for y in years] + ["report"]
        jobs, status = [], ["2026-11-01 10:01:00 ARMED"]
        stopped = False
        for st in stages:
            job += 1
            jobs.append({"stage": st, "job_id": str(job)})
            if st == "report":
                break
            if stopped:
                status.append(f"REPORT basin_{r}_{st} job={job} state=CANCELLED+ "
                              f"elapsed=00:00:00 exit=0:0")
                continue
            if st == "case":
                _w(rec / "checks" / f"case_{job}.json", _check("case", job))
                _w(rec / "checks" / f"settings_{job}.json", _check("settings", job))
            elif st == "smoke":
                _w(rec / "checks" / f"smoke_{job}.json", _check("smoke", job))
            else:
                y = int(st[1:])
                bad = (y == stop_at)
                _w(rec / "checks" / f"{st}_{job}.json", _check(
                    st, job, ok=not bad,
                    reasons=[f"{y}: the run ended before its year did"] if bad else [],
                    years=_year(y, ok=not bad)))
                stopped = bad
            status.append(f"REPORT basin_{r}_{st} job={job} state="
                          f"{'FAILED' if st == f'y{stop_at}' else 'COMPLETED'} "
                          f"elapsed=00:{10 + len(status):02d}:00 exit=0:0")
        _w(rec / "chain.json", {"segments": [{"armed": "2026-11-01T10:01:00",
                                              "record_dir": "/elsewhere/not/read",
                                              "jobs": jobs, "ok": True}]})
        _w(rec / "STATUS.log", "\n".join(status) + "\n")
    _w(rd / "03_results" / "regional.json", {
        "regions": {"vegetated cells": {"cells": 13394, "area_km2": 7872.4,
                                        "rule": "PCT_NATVEG >= 100"},
                    "watershed": {"cells": 4127, "area_km2": 2425.3,
                                  "rule": "cell centre inside the polygon"}},
        "years": list(years),
        "runs": {"control": {"run_dir": "/model/control/run",
                             "fields": {"GPP": {"how": "sum", "units": "gC/m^2/yr"}},
                             "missing": {}, "problems": []},
                 **{r: {"run_dir": f"/model/{r}/run",
                        "fields": {"GPP": {"how": "sum", "units": "gC/m^2/yr"}},
                        "missing": {}, "problems": []} for r in runs}},
        "differences": {r: {} for r in runs}, "problems": []})
    _w(rd / "04_analysis" / "DOMAIN_REPORT.md",
       "# Q\n\n*status*\n\n## The answer\n\n**GPP over watershed differed by -25 gC/m^2/yr "
       "in 2015 (-3.89 %).**\n\n## What was run\n")
    _w(rd / "04_analysis" / "domain_report.json", {"status": "complete",
                                                   "headline": "the json headline"})
    stage = {"status": "done", "at": "2026-11-01T12:00:00"}
    if analysis == "done":
        _w(rd / "04_analysis" / "REPORT.md",
           "# Q\n\n## The answer\n\n**A warmer summer cuts growth by 3.9 percent.**\n\n"
           "More text.\n")
        _w(rd / "04_analysis" / "analysis.json", {
            "status": "supported", "verdict": "sufficient",
            "headline": "the analysis headline",
            "claims": [{"claim": "Growth is lower by 25.0 gC/m2."},
                       {"claim": "Soil water is lower by 0.162 kg/m2."}],
            "withheld": [{"claim": "Fire doubled.", "struck_by": "value_not_in_finding",
                          "struck_because": "2.0 is in no finding"}],
            "cost": {"llm": {"total": {"calls": 3, "prompt_tokens": 24010,
                                       "completion_tokens": 12230, "seconds": 135.01,
                                       "models": ["model-x"]}}}})
    elif analysis == "skipped":
        stage = {"status": "skipped", "reason": "no successful columns"}
    elif analysis == "off":
        stage = {"status": "done", "analyzer": "switched off (analysis_llm: false)"}
    _w(rd / "run_state.json", {"stages": {"analyze": stage}})
    _w(rd / "RUN_SUMMARY.json", {"experiments_total": 1 + len(runs)})
    return rd


def _page(rd, **kw):
    return SR.render(rd, **kw)


class TestTheQuestion:

    def test_the_question_its_revisions_and_the_answers_given(self, tmp_path):
        p = _page(_run_folder(tmp_path))
        assert f"> {QUESTION}" in p
        assert "Changes asked for at the design review (1):\n1. Report it by elevation band." in p
        assert "1. **Asked:** Which months?\n   **You answered:** June to August" in p
        assert "routed as **design**, read as a **whole-domain** study, for **ELM**, in 5 " in p
        assert "The brief: control **basin** (reused); runs **warm**; years 2015." in p

    def test_no_question_on_record_is_said_with_what_the_record_keeps(self, tmp_path):
        rd = _run_folder(tmp_path)
        rec = json.loads((rd / "reception.json").read_text())
        del rec["revision_of"]
        _w(rd / "reception.json", rec)
        p = _page(rd)
        assert "reception.json records no question asked of you. It keeps a question only " \
               "when Reception stopped to ask it" in p
        assert "**Asked:**" not in p

    def test_a_clarification_lists_the_questions_left_open(self, tmp_path):
        rd = _run_folder(tmp_path)
        _w(rd / "reception.json", {"route": {"action": "clarify",
                                             "questions": ["Which years?", "Which basin?"]},
                                   "brief": {}, "user_request": QUESTION})
        p = _page(rd)
        assert "Reception stopped to ask 2 question(s):\n- Which years?\n- Which basin?" in p
        assert "routed as **clarify**" in p

    def test_no_reception_record(self, tmp_path):
        rd = _run_folder(tmp_path)
        (rd / "reception.json").unlink()
        assert "There is no reception.json in this folder" in _page(rd)


class TestTheDesignAndTheCheck:

    def test_the_decision_line_and_a_link_to_the_page(self, tmp_path):
        rd = _run_folder(tmp_path)
        p = _page(rd)
        assert "The design page: [DESIGN_REVIEW.md](DESIGN_REVIEW.md)." in p
        assert "- **Decision:** accepted at the terminal at 2026-11-01T10:00:00." in p

    def test_the_link_is_relative_to_where_the_page_is_written(self, tmp_path):
        rd = _run_folder(tmp_path)
        p = _page(rd, out=tmp_path / "records" / "a.md")
        assert "[DESIGN_REVIEW.md](../study_a/DESIGN_REVIEW.md)" in p

    def test_a_correction_after_the_decision_is_named(self, tmp_path):
        rd = _run_folder(tmp_path)
        _w(rd / "PLAN_CORRECTED.md", "# The plan was corrected\n\nwhy\n")
        _w(rd / "strategy.as_accepted.json", {})
        p = _page(rd)
        assert "[PLAN_CORRECTED.md](PLAN_CORRECTED.md) (The plan was corrected)" in p
        assert "`strategy.as_accepted.json`" in p

    def test_a_run_before_the_preview_says_so(self, tmp_path):
        p = _page(_run_folder(tmp_path))
        assert "This run predates the server's check before the design page" in p
        assert ("When the runs were built (`01_inputs/domain_runs.json`): the server "
                "accepted the experiment; 1 run(s) to execute, 1 model year(s), about 0.8 "
                "node-hours") in p
        assert "passed its check (`domain_case/inputs_check.json`)" in p

    def test_the_preview_passed_after_a_refused_round(self, tmp_path):
        rd = _run_folder(tmp_path)
        _w(rd / "01_inputs" / "design_preview.json", {
            "made": "2026-11-01T09:00:00", "refused": [], "error": None, "problems": [],
            "runs": [{"name": "warm"}],
            "rounds": [{"design": "01_inputs/refused_1.json",
                        "findings": [{"where": "run warm", "what": "H2OSOI has levels"}]}],
            "check": {"unusual": [{"where": "years", "what": "one year only"}],
                      "cost": {"node_hours": 0.8}},
            "inputs_check": {"pass": True}})
        p = _page(rd)
        assert "predates" not in p
        assert "the server refused design 1 and the Planner changed it" in p
        assert "  - run warm: H2OSOI has levels" in p
        assert "- **Passed.** The server checked the design and built 1 run(s)" in p
        assert "- unusual: years: one year only" in p
        assert "- cost: about 0.8 node-hours" in p

    def test_a_refused_preview(self, tmp_path):
        rd = _run_folder(tmp_path)
        _w(rd / "01_inputs" / "design_preview.json", {
            "refused": [{"where": "the experiment", "what": "no such change"}], "runs": []})
        assert "- **Refused.** The server would not build the design:\n" \
               "  - the experiment: no such change" in _page(rd)


class TestTheRuns:

    def test_each_job_its_check_and_its_time(self, tmp_path):
        p = _page(_run_folder(tmp_path, years=(2015, 2016)))
        assert "### Run warm" in p
        assert "- the control with weather_scale (temperature_K 2.0)" in p
        assert "| case | 1001 | case passed, settings passed | COMPLETED | 00:11:00 |" in p
        assert "| y2016 | 1004 | y2016 passed | COMPLETED | 00:14:00 |" in p
        assert "| report | 1005 | none (it logs the job times) | - | - |" in p
        # 11 + 12 + 13 + 14 minutes
        assert "model jobs' time as the report stage logged it: 0:50:00 in all." in p
        assert "years planned: 2015, 2016; years whose check passed: 2015, 2016." in p
        assert "- every job's check passed." in p
        assert ("y2015 (job 1003): 2015 compared with the control (not counted in the "
                "verdict): TOTVEGC: bias -0.25 %") in p
        assert "y2015 (job 1003): the change's expected effects on 2015's output: 1 of 1 held" in p
        assert "case (job 1001), its settings check" not in p      # nothing to say

    def test_a_chain_that_stopped_part_way(self, tmp_path):
        p = _page(_run_folder(tmp_path, years=(2015, 2016, 2017), stop_at=2016))
        assert "| y2016 | 1004 | y2016 FAILED | FAILED |" in p
        assert "| y2017 | 1005 | no check record | CANCELLED+ |" in p
        assert "years planned: 2015, 2016, 2017; years whose check passed: 2015." in p
        assert ("**The chain stopped at stage y2016** (job 1004): its check: y2016 FAILED; "
                "the scheduler's state: FAILED.") in p
        assert "y2016 (job 1004): 2016: the run ended before its year did" in p

    def test_a_chain_whose_report_has_not_run(self, tmp_path):
        rd = _run_folder(tmp_path)
        _w(rd / "domain_case" / "warm" / "STATUS.log", "2026-11-01 ARMED\n")
        p = _page(rd)
        assert "| case | 1001 | case passed, settings passed | not logged | - |" in p
        assert "the report stage has logged no job times yet" in p

    def test_the_control_alone(self, tmp_path):
        p = _page(_run_folder(tmp_path, runs=(), years=(2015, 2021)))
        assert "- the control: **basin**, reused, so its existing output was read" in p
        assert ("- No run differs from the control: the study read the control's output "
                "alone, and no job was submitted.") in p
        assert "; nothing to execute (the control is reused: it spends nothing)." in p
        assert "### Run" not in p

    def test_a_run_never_armed(self, tmp_path):
        rd = _run_folder(tmp_path)
        import shutil
        shutil.rmtree(rd / "domain_case" / "warm")
        assert "- no chain was armed for it (there is no `domain_case/warm/chain.json`)." \
            in _page(rd)

    def test_a_sampled_study_is_counted_from_its_summary(self, tmp_path):
        rd = tmp_path / "cols"
        _w(rd / "RUN_SUMMARY.json", {"experiments_total": 4, "experiments_success": 3,
                                     "experiments_failed": 1})
        p = _page(rd)
        assert "No whole-domain run is on record in this folder" in p
        assert "- RUN_SUMMARY.json: 3 of 4 case(s) succeeded, 1 failed." in p

    def test_the_report_lines_both_shapes(self):
        t = ("x REPORT basin_case job=11 state=COMPLETED elapsed=1-00:00:05 exit=0:0\n"
             "x REPORT job=12 state=UNKNOWN (no accounting record: cancelled before it "
             "started)\n"
             "x REPORT basin_case job=11 state=COMPLETED elapsed=00:00:06 exit=0:0\n")
        r = SR.report_lines(t)
        assert r["11"]["elapsed"] == "00:00:06"           # the last line wins
        assert r["12"] == {"name": None, "state": "UNKNOWN", "elapsed": None, "exit": None}
        assert SR._seconds("1-00:00:05") == 86405


class TestTheNumbersAndTheAnalyzer:

    def test_the_numbers(self, tmp_path):
        p = _page(_run_folder(tmp_path))
        assert ("> GPP over watershed differed by -25 gC/m^2/yr in 2015 (-3.89 %).") in p
        assert "(status **complete**), says:" in p
        assert "- years: 2015" in p
        assert "- run **warm**, read from `/model/warm/run`" in p
        assert "- fields (1): GPP (gC/m^2/yr, total over the year)" in p
        assert "| vegetated cells | 13,394 | 7,872 | PCT_NATVEG >= 100 |" in p
        assert "- differences from the control, for: warm" in p

    def test_no_numbers(self, tmp_path):
        rd = _run_folder(tmp_path)
        (rd / "03_results" / "regional.json").unlink()
        (rd / "04_analysis" / "DOMAIN_REPORT.md").unlink()
        (rd / "04_analysis" / "domain_report.json").unlink()
        p = _page(rd)
        assert "No numbers report was written" in p
        assert "There is no 03_results/regional.json" in p

    def test_the_analyzer(self, tmp_path):
        p = _page(_run_folder(tmp_path))
        assert "The analysis stage: **done**, at 2026-11-01T12:00:00." in p
        assert "Its report: [REPORT.md](04_analysis/REPORT.md)." in p
        assert "> A warmer summer cuts growth by 3.9 percent." in p
        assert "- status **supported**; verdict **sufficient**" in p
        assert "- claims kept by the audit: 2; struck: 1" in p
        assert "  1. Growth is lower by 25.0 gC/m2." in p
        assert ("  - struck (value_not_in_finding): Fire doubled. Why: 2.0 is in no finding"
                in p)
        assert ("- model calls: 3, 24,010 tokens sent and 12,230 received, 135 s (model-x)"
                in p)

    def test_an_analysis_that_was_skipped(self, tmp_path):
        p = _page(_run_folder(tmp_path, analysis="skipped"))
        assert "## The Analyzer\n**Skipped.** no successful columns" in p
        assert "There is no Analyzer report" in p

    def test_an_analyzer_switched_off(self, tmp_path):
        p = _page(_run_folder(tmp_path, analysis="off"))
        assert "The Analyzer did not run: switched off (analysis_llm: false)." in p


class TestTheIndexAndTheRobustness:

    def test_every_file_once_with_its_role(self, tmp_path):
        rd = _run_folder(tmp_path)
        _w(rd / "notes_by_hand.txt", "x")
        path = SR.write_record(rd)
        p = path.read_text()
        files = sorted(q.relative_to(rd).as_posix() for q in rd.rglob("*") if q.is_file())
        assert f"## Every file in this folder ({len(files)})" in p
        for f in files:
            assert p.count(f"| `{f}` |") == 1, f
        assert "| `STUDY_RECORD.md` | this page |" in p
        assert "| `notes_by_hand.txt` | no role on record for this file |" in p
        assert "| `domain_case/warm/checks/y2015_1003.json` | one model year's check" in p
        assert "| `domain_case/warm/checks/settings_1001.json` | the check of the run's " \
               "settings" in p

    def test_a_page_written_elsewhere_is_not_in_the_index(self, tmp_path):
        rd = _run_folder(tmp_path)
        out = SR.write_record(rd, tmp_path / "records" / "a.md")
        assert out == tmp_path / "records" / "a.md"
        assert not (rd / SR.RECORD_NAME).exists()
        assert "STUDY_RECORD.md" not in out.read_text()

    def test_a_broken_record_costs_only_its_own_lines(self, tmp_path):
        rd = _run_folder(tmp_path)
        _w(rd / "03_results" / "regional.json", "{not json")
        _w(rd / "domain_case" / "warm" / "checks" / "smoke_1002.json", "{")
        p = _page(rd)
        assert "03_results/regional.json is in the folder but regional.json could not be " \
               "read" in p
        assert "| smoke | 1002 | smoke unreadable |" in p
        assert "**The chain stopped at stage smoke**" in p
        assert "- claims kept by the audit: 2; struck: 1" in p      # the rest is drawn

    def test_a_part_that_raises_is_named_and_the_page_still_drawn(self, tmp_path,
                                                                 monkeypatch):
        rd = _run_folder(tmp_path)

        def boom(*a):
            raise KeyError("x")
        monkeypatch.setattr(SR, "_numbers_lines", boom)
        p = _page(rd)
        assert "*This part could not be drawn: KeyError: 'x'*" in p
        assert "## The Analyzer" in p and "## Every file in this folder" in p

    def test_the_record_dir_named_in_a_chain_is_not_read(self, tmp_path):
        """chain.json names its record directory by absolute path; the page
        reads the run's folder in this one, so a moved or copied study renders
        the same."""
        rd = _run_folder(tmp_path)
        moved = tmp_path / "moved"
        rd.rename(moved)
        assert "| case | 1001 | case passed, settings passed | COMPLETED |" in _page(moved)

    def test_the_command_line(self, tmp_path, capsys):
        rd = _run_folder(tmp_path)
        assert SR.main([str(rd), "--out", str(tmp_path / "o" / "r.md")]) == 0
        assert capsys.readouterr().out.strip() == str(tmp_path / "o" / "r.md")
        assert (tmp_path / "o" / "r.md").read_text().startswith("# Study record: study_a")

    def test_no_language_model_is_imported(self):
        src = Path(SR.__file__).read_text()
        assert "llm" not in "".join(l for l in src.splitlines() if "import" in l).lower()

    def test_no_em_dash_in_what_the_page_writes(self, tmp_path):
        assert "—" not in _page(_run_folder(tmp_path, years=(2015, 2016), stop_at=2016))


# ─────────────────────────────────────────────────────────────────────
# THE ONE CALL SITE: the end of the manager's analyze stage
# ─────────────────────────────────────────────────────────────────────

from test_domain_manager import srv, mgr, _stages            # noqa: E402,F401
from test_domain_experiment import study                  # noqa: E402,F401
import test_domain_experiment as TDE                      # noqa: E402


class TestTheCallSite:

    def test_the_study_ends_with_its_record(self, study, tmp_path):
        m, cfg = TDE.TestAnExperimentThroughTheManager._with_history(study, tmp_path)
        m.execute_plan(dict(cfg["strategy"]), cfg)
        page = (m.run_dir / SR.RECORD_NAME).read_text()
        assert "## The runs" in page and "No run differs from the control" in page
        assert "| `RUN_SUMMARY.json` |" in page
        assert "| `04_analysis/DOMAIN_REPORT.md` |" in page
        assert _stages(m)["analyze"]["status"] == "done"

    def test_a_record_that_fails_cannot_fail_the_study(self, study, tmp_path, monkeypatch,
                                                         capsys):
        def boom(run_dir, out=None):
            raise RuntimeError("no page today")
        monkeypatch.setattr(SR, "write_record", boom)
        m, cfg = TDE.TestAnExperimentThroughTheManager._with_history(study, tmp_path)
        summary = m.execute_plan(dict(cfg["strategy"]), cfg)
        assert summary["run_directory"] == str(m.run_dir)
        assert not (m.run_dir / SR.RECORD_NAME).exists()
        assert _stages(m)["analyze"]["status"] == "done"
        assert "the study record could not be written (RuntimeError: no page today)" \
            in capsys.readouterr().out
