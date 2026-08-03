#!/usr/bin/env python3
"""Finding the runs that can be continued — and saying why the rest cannot.

The scan is deliberately plain code, not an agent: "is job 770680 finished" is
a question squeue answers exactly. What is pinned here is that it stays honest
about the two things a caller acts on — whether a run can be resumed, and which
run it is.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.resumable import (                                    # noqa: E402
    find_resumable, inspect_run, describe, _next_stage, _stages_for,
)


def _run(tmp_path, name="elm_run_20260801_120000", *, stages=None,
         model="elm", request=None, experiment=False, ledger=True):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    if ledger:
        (d / "run_state.json").write_text(json.dumps({
            "model": model, "run_dir": str(d),
            "updated": datetime.now().isoformat(),
            "stages": stages or {}}))
    if request is not None:
        (d / "reception.json").write_text(json.dumps(
            {"user_request": request}))
    if experiment:
        (d / "experiment.json").write_text('{"columns": []}')
    return d


def _done(*names):
    return {n: {"status": "done", "at": datetime.now().isoformat()}
            for n in names}


class TestWhatCountsAsResumable:

    def test_a_submitted_job_is(self, tmp_path):
        d = _run(tmp_path, stages={**_done("materialize", "build_case_inputs", "build_cases"),
                                   "run": {"status": "pending",
                                           "job_id": "770680"}})
        r = inspect_run(d)
        assert r["resumable"]
        assert r["job_id"] == "770680"
        assert "770680" in r["why"]

    def test_a_run_stopped_partway_is(self, tmp_path):
        d = _run(tmp_path, stages=_done("materialize", "build_case_inputs"))
        r = inspect_run(d)
        assert r["resumable"]
        assert r["next"] == "build_cases"       # ELM has one
        assert "build_case_inputs" in r["why"]

    def test_a_finished_run_is_not(self, tmp_path):
        """_package wrote experiment.json, which is what the run is FOR. A
        failed analyze leaves a finished study with no report, and re-running
        the pipeline is not how to get one."""
        d = _run(tmp_path, stages={**_done("materialize", "build_case_inputs", "build_cases",
                                           "run", "extract", "package"),
                                   "analyze": {"status": "failed"}})
        r = inspect_run(d)
        assert not r["resumable"]
        assert "complete" in r["why"]

    def test_a_run_with_no_ledger_is_not(self, tmp_path):
        """Re-entering one would find an empty ledger and redo everything,
        which is not resuming — so it is not offered as such."""
        d = _run(tmp_path, ledger=False, request="something")
        r = inspect_run(d)
        assert not r["resumable"]
        assert "ledger" in r["why"]

    def test_and_says_whether_it_at_least_finished(self, tmp_path):
        """Most of what is on disk predates the ledger. Someone choosing
        between those still needs to know which ones produced a result."""
        done = _run(tmp_path, "a_run_1", ledger=False, experiment=True)
        part = _run(tmp_path, "a_run_2", ledger=False)
        assert "complete" in inspect_run(done)["why"]
        assert "re-run" in inspect_run(part)["why"]


class TestItSaysWhichRunItIs:
    """Run dirs are named elm_run_20260801_132630 — timestamps, not topics.
    Two studies of the same watershed an hour apart are indistinguishable by
    name, so the scan carries what the run was ASKED to do."""

    def test_the_request_text_comes_along(self, tmp_path):
        d = _run(tmp_path, stages=_done("materialize"),
                 request="Quantify recharge in the Upper Gunnison")
        assert "Gunnison" in inspect_run(d)["request"]

    def test_a_missing_reception_is_not_fatal(self, tmp_path):
        d = _run(tmp_path, stages=_done("materialize"))
        r = inspect_run(d)
        assert r["request"] is None and r["resumable"]

    def test_corrupt_json_is_not_fatal(self, tmp_path):
        """These files were written by an earlier session, possibly one that
        was killed mid-write."""
        d = _run(tmp_path, stages=_done("materialize"))
        (d / "reception.json").write_text("{ this is not json")
        assert inspect_run(d)["request"] is None
        (d / "run_state.json").write_text("}{")
        r = inspect_run(d)
        assert not r["resumable"] and "ledger" in r["why"]

    def test_the_model_comes_off_the_ledger(self, tmp_path):
        """Resuming an ELM run with the default backend would build PFLOTRAN
        decks in an ELM directory."""
        d = _run(tmp_path, stages=_done("materialize"), model="pflotran")
        assert inspect_run(d)["model"] == "pflotran"

    def test_a_ledgerless_run_falls_back_to_its_name(self, tmp_path):
        d = _run(tmp_path, "pflotran_run_20260801_120000", ledger=False)
        assert inspect_run(d)["model"] == "pflotran"


class TestTheStageSequenceIsTheBackendsNotAGuess:

    def test_pflotran_has_no_case_build_stage(self, tmp_path):
        """NEEDS_CASE_BUILD = False — deck generation IS its build. Reading the
        ledger without asking the backend reports 'build_cases' as the outstanding
        stage of every interrupted PFLOTRAN study, forever."""
        assert "build_cases" not in _stages_for("pflotran")
        assert "build_cases" in _stages_for("elm")

    def test_so_a_pflotran_run_past_build_is_waiting_on_run(self, tmp_path):
        d = _run(tmp_path, model="pflotran",
                 stages=_done("materialize", "build_case_inputs"))
        assert inspect_run(d)["next"] == "run"

    def test_an_unknown_backend_assumes_every_stage(self, tmp_path):
        """Not a crash and not an empty sequence: an unrecognised model name
        should degrade to the fullest reading, which over-reports rather than
        silently declaring a run finished."""
        assert _stages_for("no-such-model") == _stages_for(None)

    def test_every_stage_the_scan_knows_about_is_actually_recorded(self, tmp_path):
        """The scan and execute_plan must not disagree about what comes next.

        Checked by RUNNING the pipeline, not by grepping for _mark calls: the
        stages are marked from inside _advance now, and a source grep passed
        for the wrong reason before and failed for the wrong reason after.
        """
        import types
        from core.exp_manager_base import ExperimentManagerBase

        class _Full(ExperimentManagerBase):
            MODEL = "fake"
            def _materialize(self, p, c):  return p
            def _build_case_inputs(self, p, c):        return [{"case_name": "c1"}]
            def _build_cases(self, e, c=None): return None
            def _run(self, e, c):          return {"c1": True}
            def _extract(self, e, plan=None, config=None):
                return types.SimpleNamespace(results=[], units={}, summary={})
            def _package(self, p, a, c):
                (self.run_dir / "experiment.json").write_text("{}")
            def _save_llm_input(self, p, a):  pass

        m = _Full(base_output_dir=str(tmp_path))
        m.execute_plan({}, {})
        recorded = set(m._load_state()["stages"])
        assert set(ExperimentManagerBase.STAGES) <= recorded, (
            f"the scan reasons about stages execute_plan never records: "
            f"{set(ExperimentManagerBase.STAGES) - recorded}")


class TestTheScan:

    def test_newest_first(self, tmp_path):
        for n in ("elm_run_20260101_000000", "elm_run_20260801_000000",
                  "elm_run_20260401_000000"):
            _run(tmp_path, n, stages=_done("materialize"))
        names = [r["name"] for r in find_resumable(str(tmp_path))]
        assert names == sorted(names, reverse=True)

    def test_non_runs_are_skipped(self, tmp_path):
        """workflow_outputs also holds eval outputs, probe scripts and
        scratch dirs that are not runs."""
        _run(tmp_path, stages=_done("materialize"))
        (tmp_path / "planner_eval").mkdir()
        (tmp_path / "notes.txt").write_text("hello")
        assert len(find_resumable(str(tmp_path))) == 1

    def test_include_all_carries_the_reasons(self, tmp_path):
        """'3 runs, all complete' is a different answer from 'no runs at all',
        and a caller that shows an empty list can say neither."""
        _run(tmp_path, "elm_run_20260801_000001", stages=_done("materialize"))
        _run(tmp_path, "elm_run_20260801_000002", ledger=False,
             experiment=True)
        assert len(find_resumable(str(tmp_path))) == 1
        every = find_resumable(str(tmp_path), include_all=True)
        assert len(every) == 2
        assert all(r["why"] for r in every)

    def test_a_missing_output_dir_is_empty_not_an_error(self, tmp_path):
        assert find_resumable(str(tmp_path / "nope")) == []

    def test_describe_does_not_say_the_job_twice(self, tmp_path):
        """It read 'job 770680 RUNNING; job 770680 is RUNNING'."""
        d = _run(tmp_path, stages={"run": {"status": "pending",
                                           "job_id": "770680"}})
        r = inspect_run(d)
        r["job_state"] = "RUNNING"
        assert describe(r).count("770680") == 1


class TestResumeIsWiredIntoTheCLI:

    def test_bare_resume_lists_and_does_not_run(self):
        """A bare --resume is often someone asking what is outstanding.
        Answering it by starting work is not what was asked."""
        src = (ROOT / "workflow.py").read_text()
        assert "print_listing" in src
        assert "'--resume'" in src or '"--resume"' in src

    def test_listing_needs_no_llm_and_no_mcp(self):
        """A scan is a filesystem read. Building the coordinator first would
        spend an LLM and six MCP servers to print a directory listing — and
        would fail outright if the gateway were down, which is exactly when
        someone is trying to recover a run."""
        src = (ROOT / "workflow.py").read_text()
        i_scan = src.index("from core.resumable import find_resumable")
        i_coord = src.index("coordinator = WorkflowCoordinator", i_scan)
        assert i_scan < i_coord

    def test_the_model_is_taken_from_the_ledger_not_the_flag(self):
        src = (ROOT / "workflow.py").read_text()
        assert 'model = rec.get("model") or self.model' in src

    def test_resume_is_not_set_for_a_fresh_run(self):
        """execute_plan's resume is opt-in, and _execute must not leak it into
        an ordinary run — that would silently reuse stale compute."""
        src = (ROOT / "workflow.py").read_text()
        assert "if resume:\n\t\t\tcfg['resume'] = True" in src


class TestAPendingCaseBuildIsFoundToo:
    """Since D1 the CIME case build is sbatch'd, so a study can be parked at
    `prepare` as easily as at `run`."""

    def test_found_and_named(self, tmp_path):
        d = _run(tmp_path, stages={**_done("materialize", "build_case_inputs"),
                                   "build_cases": {"status": "pending",
                                               "job_id": "880001"}})
        r = inspect_run(d)
        assert r["resumable"]
        assert r["stage"] == "build_cases"
        assert r["job_id"] == "880001"
        assert "build_cases" in r["why"], \
            "'your cases are being built' and 'your ensemble is simulating' " \
            "are hours apart in what happens next"

    def test_the_earliest_pending_stage_wins(self, tmp_path):
        """A ledger cannot honestly have two stages in flight — the run stops
        at the first. If one somehow does, report the earlier."""
        d = _run(tmp_path, stages={**_done("materialize", "build_case_inputs"),
                                   "build_cases": {"status": "pending",
                                               "job_id": "880001"},
                                   "run": {"status": "pending",
                                           "job_id": "770595"}})
        assert inspect_run(d)["stage"] == "build_cases"

    def test_a_pflotran_run_never_reports_a_pending_case_build(self, tmp_path):
        """PFLOTRAN has no prepare stage at all, so a stray entry for one is
        not something to wait on."""
        d = _run(tmp_path, model="pflotran",
                 stages={**_done("materialize", "build_case_inputs"),
                         "build_cases": {"status": "pending", "job_id": "880001"}})
        r = inspect_run(d)
        assert r["stage"] != "build_cases"


class TestReceptionMayChooseButNotInvent:
    """Option A: deterministic discovery, LLM routing.

    A run directory reception PRODUCED rather than SELECTED would either crash
    or — worse — resume a different study and report it as the one that was
    asked about. Run dirs are named by timestamp, so a model working from the
    name alone cannot tell two same-day studies apart.
    """

    def test_resume_is_a_route(self):
        src = (ROOT / "src" / "agents" / "reception_llm.py").read_text()
        assert '"resume": "resume"' in src
        prompt = (ROOT / "src" / "agents" / "prompts"
                  / "reception_agentic.txt").read_text()
        assert '"intent": "resume"' in prompt

    def test_the_prompt_forbids_inventing_a_run_dir(self):
        prompt = (ROOT / "src" / "agents" / "prompts"
                  / "reception_agentic.txt").read_text()
        assert "Do not construct one" in prompt
        assert "verbatim" in prompt

    def test_the_coordinator_checks_the_choice_against_the_scan(self):
        """The prompt asks; the allowlist enforces. A prompt alone is a
        request, not a constraint."""
        src = (ROOT / "workflow.py").read_text()
        i = src.index("def _workflow_resume")
        body = src[i:i + 3000]
        assert "find_resumable" in body
        assert "allowed" in body and "allowed.get(asked)" in body

    def test_an_off_list_choice_does_not_get_resumed(self, tmp_path, capsys,
                                                     monkeypatch):
        import workflow as wf
        c = wf.WorkflowCoordinator.__new__(wf.WorkflowCoordinator)
        c.default_output_dir = str(tmp_path)
        c.conversation_context = {}
        resumed = []
        c.resume_run = lambda rd: resumed.append(rd) or "ok"

        _run(tmp_path, "elm_run_20260801_000001",
             stages={**_done("materialize"), })
        _run(tmp_path, "elm_run_20260801_000002", stages=_done("materialize"))

        out = c._workflow_resume(
            {"route": {"prior_run_dir": "/invented/elm_run_19990101_000000"}})
        assert not resumed, "a fabricated run directory must not be resumed"
        assert "Which one" in out
        assert "not resumable" in capsys.readouterr().out

    def test_the_scan_is_offered_to_reception(self):
        """Reception cannot pick from a list it never sees."""
        src = (ROOT / "workflow.py").read_text()
        i = src.index("def _reception_context")
        body = src[i:i + 3000]
        assert "resumable_runs" in body
        assert "'request'" in body, \
            "without the request text the list is a column of timestamps"

    def test_a_broken_scan_does_not_break_asking_a_question(self, tmp_path):
        """Someone asking about hydrology should not be stopped by a resume
        scan that failed."""
        import workflow as wf
        c = wf.WorkflowCoordinator.__new__(wf.WorkflowCoordinator)
        c.default_output_dir = "/no/such/dir"
        c.conversation_context = {"last_focus": "recharge"}
        ctx = c._reception_context()
        assert ctx["prior_focus"] == "recharge"
        assert "resumable_runs" not in ctx
