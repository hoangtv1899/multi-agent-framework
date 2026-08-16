#!/usr/bin/env python3
"""The Analyzer as a sequencer: steps 0-4, and what happens when one fails.

The steps have their own tests. What is pinned here is the WIRING — that the
box calls them in dependency order, hands each the previous one's output, and
does not lose four working steps because a fifth failed.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.analyzer import Analyzer                       # noqa: E402


class _Ctx:
    columns = [{"case_name": "col_01"}]
    caveats = [{"id": "b", "severity": "blocking", "applies_to": "x",
                "statement": "y"}]
    plan: dict = {}
    data: dict = {}

    def series(self):
        return None


def _wire(monkeypatch, *, compare_raises=False, loop_raises=False,
          report_raises=False, context_raises=False):
    from agents.analysis import (step0_context, step1_compare,
                                 step3_interpret, step4_report)
    calls = []

    def ctx_load(_rd):
        calls.append("context")
        if context_raises:
            raise RuntimeError("no run dir")
        return _Ctx()

    def compare_all(ctx, out_dir, **kw):
        calls.append("compare")
        if compare_raises:
            raise RuntimeError("boom")
        return {"figures": {"a": "a.png"}, "caveats": []}

    def loop(ctx, out_dir, comparison=None, **kw):
        calls.append("loop")
        assert comparison is not None, "step 2/3 must receive step 1's record"
        if loop_raises:
            raise RuntimeError("boom")
        return {"investigation": {"n_succeeded": 2, "n_proposed": 3,
                                  "findings": []},
                "interpretation": {"verdict": "sufficient",
                                   "audit": {"n_claims": 2, "n_struck": 0}},
                "rounds": [{"round": 1}], "stopped_because": "sufficient",
                "n_rounds": 1}

    def build(*a, **k):
        calls.append("report")
        if report_raises:
            raise RuntimeError("boom")
        return {"verdict": "sufficient", "cost": {}, "provenance": {}}

    monkeypatch.setattr(step0_context, "load", ctx_load)
    monkeypatch.setattr(step1_compare, "compare_all", compare_all)
    monkeypatch.setattr(step3_interpret, "investigate_and_interpret", loop)
    monkeypatch.setattr(step4_report, "build", build)
    monkeypatch.setattr(step4_report, "write", lambda r, d: str(Path(d) / "analysis.json"))
    monkeypatch.setattr(step4_report, "summary", lambda r: "summary")
    return calls


class TestItSequencesTheSteps:

    def test_all_five_run_in_dependency_order(self, tmp_path, monkeypatch):
        calls = _wire(monkeypatch)
        s = Analyzer(str(tmp_path), verbose=False).run()
        assert calls == ["context", "compare", "loop", "report"]
        assert all(s["steps"].values())
        assert s["verdict"] == "sufficient"

    def test_it_writes_the_boundary_file(self, tmp_path, monkeypatch):
        _wire(monkeypatch)
        s = Analyzer(str(tmp_path), verbose=False).run()
        assert Path(s["report"]).name == "analysis.json"

    def test_results_is_accepted_but_unused(self, tmp_path, monkeypatch):
        """The manager passes its in-memory extraction object. Step 0 reads the
        packaged run off disk instead, so a live run and an archived one take
        exactly the same path — using the object would make them differ."""
        _wire(monkeypatch)
        s = Analyzer(str(tmp_path), verbose=False).run(results=object())
        assert all(s["steps"].values())


class TestOneFailedStepDoesNotLoseTheRest:
    """This box exists to report what a run shows, including that part of it
    could not be shown."""

    def test_a_failed_comparison_still_reaches_the_report(self, tmp_path, monkeypatch):
        calls = _wire(monkeypatch, compare_raises=True)
        s = Analyzer(str(tmp_path), verbose=False).run()
        assert s["steps"]["compare"] is False
        assert s["steps"]["report"] is True
        assert "report" in calls

    def test_a_failed_loop_still_reaches_the_report(self, tmp_path, monkeypatch):
        s_calls = _wire(monkeypatch, loop_raises=True)
        s = Analyzer(str(tmp_path), verbose=False).run()
        assert s["steps"]["investigate"] is False
        assert s["steps"]["report"] is True, "a report of what failed is still a report"

    def test_a_failed_report_does_not_raise(self, tmp_path, monkeypatch):
        _wire(monkeypatch, report_raises=True)
        s = Analyzer(str(tmp_path), verbose=False).run()
        assert s["steps"]["report"] is False and "seconds" in s

    def test_a_failed_context_ends_the_box(self, tmp_path, monkeypatch):
        """The one exception: without step 0 there is nothing for any later
        step to read, so continuing would only produce empty artifacts."""
        calls = _wire(monkeypatch, context_raises=True)
        s = Analyzer(str(tmp_path), verbose=False).run()
        assert s["steps"]["context"] is False
        assert calls == ["context"], "no later step should have been attempted"
        assert "error" in s


class TestTheFailureCrossesTheBoundary:
    """A step that failed is only a fact of the run if it is WRITTEN DOWN.

    `status["steps"]` was returned to the caller and never persisted, so an
    analysis.json from a run whose steps 2-3 raised was indistinguishable from
    one that finished and concluded nothing — same null verdict, same empty
    claims. The difference lived in a terminal nobody was watching.
    """

    def test_the_step_record_is_handed_to_the_report(self, tmp_path, monkeypatch):
        from agents.analysis import step4_report
        seen = {}
        _wire(monkeypatch, loop_raises=True)

        def capture(*a, **k):
            seen.update(k)
            return {"verdict": None, "cost": {}, "provenance": {}}

        monkeypatch.setattr(step4_report, "build", capture)
        Analyzer(str(tmp_path), verbose=False).run()
        assert seen.get("steps"), "step 4 cannot report a failure it is not told about"
        assert seen["steps"]["investigate"] is False

    def test_a_crashed_loop_writes_analysis_failed(self, tmp_path, monkeypatch):
        """End to end, through the real builder and the real writer."""
        from agents.analysis import step0_context, step1_compare, step3_interpret
        monkeypatch.setattr(step0_context, "load", lambda _rd: _Ctx())
        monkeypatch.setattr(step1_compare, "compare_all",
                            lambda ctx, out_dir, **kw: {"figures": {}, "caveats": [],
                                                        "findings": []})

        def boom(*a, **k):
            raise RuntimeError("gateway 503")

        monkeypatch.setattr(step3_interpret, "investigate_and_interpret", boom)
        st = Analyzer(str(tmp_path), verbose=False).run()

        rep = json.loads((tmp_path / "04_analysis" / "analysis.json").read_text())
        assert rep["status"] == "analysis_failed"
        assert rep["provenance"]["failed_steps"] == ["investigate", "interpret"]
        assert st["steps"]["report"] is True, "reporting the failure is still a report"

    def test_a_clean_run_writes_supported(self, tmp_path, monkeypatch):
        """The same path must not label a healthy run as failed."""
        from agents.analysis import step0_context, step1_compare, step3_interpret
        monkeypatch.setattr(step0_context, "load", lambda _rd: _Ctx())
        monkeypatch.setattr(step1_compare, "compare_all",
                            lambda ctx, out_dir, **kw: {"figures": {}, "caveats": [],
                                                        "findings": []})
        monkeypatch.setattr(step3_interpret, "investigate_and_interpret",
                            lambda ctx, out_dir, **kw: {
                                "investigation": {"n_succeeded": 1, "n_proposed": 1,
                                                  "findings": [{"id": "f1", "n": 9}]},
                                "interpretation": {"verdict": "sufficient",
                                                   "answer": "yes",
                                                   "audit": {"n_claims": 1,
                                                             "n_struck": 0},
                                                   "claims": [{"claim": "a",
                                                               "finding_id": "f1"}]},
                                "rounds": [{"round": 1}],
                                "stopped_because": "sufficient", "n_rounds": 1})
        Analyzer(str(tmp_path), verbose=False).run()
        rep = json.loads((tmp_path / "04_analysis" / "analysis.json").read_text())
        assert rep["status"] == "supported"
        assert rep["provenance"]["failed_steps"] is None


class TestTheManagersCallSiteStillBinds:

    def test_run_takes_results_and_config(self):
        import inspect
        p = inspect.signature(Analyzer.run).parameters
        assert "results" in p and "config" in p

    def test_the_manager_calls_it_that_way(self):
        """The call site moved to the BASE when execute_plan was lifted, so
        every backend gets the Analyzer for free rather than each remembering
        to call it. Asserted against the base for that reason."""
        src = (ROOT / "src" / "core" / "exp_manager_base.py").read_text()
        assert "Analyzer(str(self.run_dir)).run(" in src
        elm = (ROOT / "mcp" / "elm-mcp" / "src" / "elm_exp_manager.py").read_text()
        assert "def execute_plan" not in elm, \
            "ELM must not re-implement the stage sequence"


class TestTheStageSequenceIsSharedNotCopied:
    """execute_plan was lifted out of ELMExpManager so a second backend cannot
    re-implement it. The ordering it enforces — _package before the Analyzer,
    everything from _package on non-fatal — is structural, and two copies of a
    structural guarantee is one copy too many."""

    def test_every_backend_resolves_to_the_base(self):
        from core.exp_manager_base import ExperimentManagerBase as B
        from elm_exp_manager import ELMExpManager
        from core.pflotran_exp_manager import PFLOTRANExpManager
        for M in (ELMExpManager, PFLOTRANExpManager):
            assert M.execute_plan is B.execute_plan, M.__name__

    def test_backends_declare_their_stages_rather_than_stubbing(self):
        """A no-op _build_cases() reports "prepared nothing, successfully"."""
        from elm_exp_manager import ELMExpManager
        from core.pflotran_exp_manager import PFLOTRANExpManager
        assert ELMExpManager.NEEDS_CASE_BUILD and ELMExpManager.NEEDS_SCHEDULER
        assert not PFLOTRANExpManager.NEEDS_CASE_BUILD
        assert not PFLOTRANExpManager.NEEDS_SCHEDULER

    def test_a_missing_stage_raises_rather_than_returning_empty(self):
        """A manager without its compute stage must fail at the first call,
        not report an empty successful run."""
        from core.exp_manager_base import ExperimentManagerBase as B
        import pytest as _pt
        m = B.__new__(B)
        for stage, args in (("_build_case_inputs", ({}, {})), ("_run", ([], {})),
                            ("_extract", ([],))):
            with _pt.raises(NotImplementedError):
                getattr(m, stage)(*args)

    def test_the_two_backends_cannot_claim_each_others_plans(self):
        """Sharing a plan key would make an ELM plan look already-materialized
        to PFLOTRAN, which builds zero experiments WITHOUT raising."""
        from elm_exp_manager import ELMExpManager
        from core.pflotran_exp_manager import PFLOTRANExpManager
        elm, pf = ELMExpManager.__new__(ELMExpManager), \
                  PFLOTRANExpManager.__new__(PFLOTRANExpManager)
        elm_plan = {"CONDITIONS_COUPLERS": [1]}
        pf_plan = {"PFLOTRAN_CASES": [1]}
        assert elm._already_executable(elm_plan) and not pf._already_executable(elm_plan)
        assert pf._already_executable(pf_plan) and not elm._already_executable(pf_plan)

    def test_each_backend_declares_its_own_field_semantics(self):
        """Inherited semantics describe the WRONG run.

        FIELD_SEMANTICS sat on the base holding ELM's metrics, so a PFLOTRAN
        experiment.json advertised precip_mm_yr and QOVER-derived runoff
        fractions for a run that computes neither — and step 2 hands this map
        to an LLM as the authority on what the numbers mean.
        """
        from core.exp_manager_base import ExperimentManagerBase as B
        from elm_exp_manager import ELMExpManager
        from core.pflotran_exp_manager import PFLOTRANExpManager

        assert B.FIELD_SEMANTICS == {}, "the base must not supply a default"

        elm_from = {v for e in ELMExpManager.FIELD_SEMANTICS.values()
                    for v in (e.get("from") or [])}
        pf_from = {v for e in PFLOTRANExpManager.FIELD_SEMANTICS.values()
                   for v in (e.get("from") or [])}
        assert "QOVER" in elm_from and "QOVER" not in pf_from
        assert "LIQUID_SATURATION" in pf_from
        # Every PFLOTRAN entry names a metric _extract actually emits.
        assert "final_water_table_depth_m" in PFLOTRANExpManager.FIELD_SEMANTICS


class TestPFLOTRANExtractSpeaksTheSharedRowShape:
    """experiment.json is ONE contract for every backend.

    These pin the parts that a passing stage-level test would not: the row
    shape _package consumes, and the two places where a convenient-looking
    value would be a false statement about the subsurface.
    """

    TEC = ('TITLE = "  2.00000E+01 [y]"\n'
           'VARIABLES="X [m]","Y [m]","Z [m]","Liquid Pressure [Pa]",'
           '"Liquid Saturation","Material ID"\n'
           'ZONE T="2.00000E+01", STRANDID=1, SOLUTIONTIME=2.00000E+01, '
           'I=1, J=1, K=3, DATAPACKING=POINT\n'
           ' 5.0E-01  5.0E-01  1.0E+00  2.0E+05  {b}  1 \n'
           ' 5.0E-01  5.0E-01  5.0E+00  1.5E+05  5.0E-01  2 \n'
           ' 5.0E-01  5.0E-01  9.0E+00  1.0E+05  4.0E-01  3 \n')

    def _case(self, tmp_path, bottom_sat):
        d = tmp_path / "col_01"
        d.mkdir(parents=True)
        (d / "col_01-004.tec").write_text(self.TEC.format(b=bottom_sat))
        return [{"id": "col_01", "case_dir": d, "fan_wtd_m": 3.5,
                 "runtime_seconds": 0.3}]

    def _extract(self, exps):
        from core.pflotran_exp_manager import PFLOTRANExpManager
        m = PFLOTRANExpManager.__new__(PFLOTRANExpManager)
        return m._extract(exps)["rows"][0]

    def test_the_row_carries_what_package_consumes(self, tmp_path):
        row = self._extract(self._case(tmp_path, "1.0E+00"))
        for key in ("case_name", "status", "metrics", "variables"):
            assert key in row, f"_package reads {key}"
        assert row["status"] == "ok"
        assert row["runtime_seconds"] == 0.3, \
            "_run's timing must survive into experiment.json"

    def test_no_water_table_is_null_not_the_domain_bottom(self, tmp_path):
        """'No water table in the domain' and 'a water table at 9 m' are
        different statements. On the 2019 Gunnison sample 10 of 19 columns
        never saturate, and filling those in with domain_depth_m would put a
        fabricated water table into every depth-vs-elevation comparison."""
        row = self._extract(self._case(tmp_path, "6.0E-01"))
        assert row["metrics"]["final_water_table_depth_m"] is None
        assert row["metrics"]["water_table_in_domain"] is False
        assert row["metrics"]["domain_depth_m"] == 9.0

        wet = self._extract(self._case(tmp_path / "wet", "1.0E+00"))
        assert wet["metrics"]["final_water_table_depth_m"] == 8.0
        assert wet["metrics"]["water_table_in_domain"] is True

    def test_there_is_no_fake_daily_block(self, tmp_path):
        """Five yearly snapshots are not a daily series. Writing them under
        `daily` with invented dates would make step0.series() build a frame
        that looks like a hydrograph."""
        row = self._extract(self._case(tmp_path, "1.0E+00"))
        for blk in row["variables"].values():
            assert "daily" not in blk

    def test_profiles_carry_times_and_depths(self, tmp_path):
        row = self._extract(self._case(tmp_path, "1.0E+00"))
        p = row["profiles"]
        assert p["times_y"] == [20.0], "the time comes from the TITLE line"
        assert p["depth_m"] == [8.0, 4.0, 0.0], "positive DOWN from the surface"
        assert len(p["saturation"]) == len(p["times_y"])
        assert len(p["saturation"][0]) == len(p["depth_m"])

    def test_a_case_with_no_output_is_failed_not_absent(self, tmp_path):
        """A column that vanishes from the ensemble is a column nobody counts."""
        d = tmp_path / "empty"
        d.mkdir()
        row = self._extract([{"id": "col_09", "case_dir": d}])
        assert row["status"] == "failed" and row["reason"]

    def test_an_unattributable_mcp_answer_falls_back_instead_of_guessing(
            self, tmp_path):
        """The server's aggregate exit_codes are in COMPLETION order.

        as_completed yields whichever job finished first, so exit_codes[i] does
        not belong to decks[i]; only results_by_input maps an outcome to the
        deck that produced it. A server that omits that map still ran the
        columns, but nothing could say WHICH failed — and a row in
        experiment.json attributed to the wrong column is worse than a slower
        run. So an answer without the map must be refused, not interpreted.
        """
        from core.pflotran_exp_manager import PFLOTRANExpManager
        m = PFLOTRANExpManager.__new__(PFLOTRANExpManager)
        d = tmp_path / "col_01"
        d.mkdir()
        (d / "col_01.in").write_text("x")
        exps = [{"id": "col_01", "case_dir": d}]

        class _Client:
            def __init__(self, payload):
                self.payload = payload
            def call_tool_json(self, *a, **k):
                return self.payload

        # the lossy shapes: no map, empty map, an outright timeout (None)
        for payload in ({"exit_codes": [0], "validation_status": "success"},
                        {"results_by_input": {}},
                        None):
            assert m._run_via_mcp(exps, _Client(payload), 60, 1) is None

        # ...and _run RAISES on them rather than quietly running locally. A
        # degradation that is merely recorded lets a study be months old
        # before anyone notices the server stopped being used.
        import pytest as _pt
        with _pt.raises(RuntimeError, match="attributed to columns"):
            m._run(exps, {"mcp_clients": {"reaction": _Client(None)}})

        # the documented escape hatch still works
        out = m._run(exps, {"mcp_clients": {"reaction": _Client(None)},
                            "run_via_mcp": False})
        assert out and out[0]["run_via"] == "local"

        # and the shape it CAN attribute is used
        good = {"results_by_input": {str(d / "col_01.in"): {
            "exit_codes": [0], "validation_status": "success",
            "execution_time": 1.25, "output_files": []}}}
        (d / "col_01-000.tec").write_text("x")
        out = m._run_via_mcp(exps, _Client(good), 60, 1)
        assert out is not None
        assert out[0]["status"] == "completed"
        assert out[0]["runtime_seconds"] == 1.25
        assert out[0]["run_via"] == "mcp"

    def test_the_call_budget_is_sized_to_the_ensemble_and_then_restored(
            self, tmp_path):
        """The client's ceiling is per CALL; `limit` is per column.

        mcp_config defaults the client to 300 s, a figure sized for the
        binning tools. Nineteen columns four-wide at 900 s each is 4500 s, so
        a perfectly healthy ensemble would be abandoned at 300 s and then
        re-run locally — paying for it twice. And because the client is
        shared, a raised timeout must not leak into the next binning call,
        where it would hide a hang.
        """
        from core.pflotran_exp_manager import PFLOTRANExpManager
        m = PFLOTRANExpManager.__new__(PFLOTRANExpManager)
        exps = []
        for i in range(8):
            d = tmp_path / f"col_{i:02d}"
            d.mkdir()
            (d / f"col_{i:02d}.in").write_text("x")
            (d / f"col_{i:02d}-000.tec").write_text("x")
            exps.append({"id": f"col_{i:02d}", "case_dir": d})

        seen = {}

        class _Client:
            timeout = 300.0
            def call_tool_json(self, name, args):
                seen["timeout_during_call"] = self.timeout
                return {"results_by_input": {
                    str(next(pathlib.Path(e["case_dir"]).glob("*.in"))): {
                        "exit_codes": [0], "validation_status": "success",
                        "execution_time": 1.0, "output_files": []}
                    for e in exps}}

        import pathlib
        c = _Client()
        out = m._run_via_mcp(exps, c, limit=900, width=4)
        assert out is not None and len(out) == 8
        # 8 columns / 4 wide = 2 waves x 900 s (+ headroom) — well past 300
        assert seen["timeout_during_call"] >= 1800
        assert c.timeout == 300.0, "the shared client's budget must be restored"

    def test_results_stay_in_experiment_order_not_completion_order(
            self, tmp_path):
        """Columns run concurrently, so completion order is a race.

        The run record gets compared against columns.json by position often
        enough that rows shuffled by whichever column finished first would be
        a needless difference between two identical runs.
        """
        from core.pflotran_exp_manager import PFLOTRANExpManager
        import os
        m = PFLOTRANExpManager.__new__(PFLOTRANExpManager)

        exps = []
        for i in range(6):
            d = tmp_path / f"col_{i:02d}"
            d.mkdir()
            (d / f"col_{i:02d}.in").write_text("x")
            exps.append({"id": f"col_{i:02d}", "case_dir": d})

        # Later columns finish FIRST, so completion order reverses input order.
        def fake_one(e, exe, limit):
            import time
            time.sleep(0.05 * (6 - int(e["id"][-2:])))
            e.update(status="completed", runtime_seconds=0.0)
            return {**e}

        m._run_one = fake_one
        os.environ.setdefault("PFLOTRAN_EXECUTABLE", "/bin/true")
        out = m._run(exps, {"max_parallel": 6})
        assert [r["id"] for r in out] == [e["id"] for e in exps]

    def test_a_run_that_failed_is_not_resurrected_by_partial_output(
            self, tmp_path):
        """A timed-out column still leaves the snapshots it managed to write.

        Reading those produced a row marked 'ok', carrying metrics from a
        simulation that never reached its final time, while _run's own record
        said 'failed'. The partial output is real, but it is not the
        experiment that was asked for.
        """
        exps = self._case(tmp_path, "1.0E+00")
        exps[0].update(status="failed", reason="exceeded 300s — timestep "
                                               "collapse", runtime_seconds=300)
        row = self._extract(exps)
        assert row["status"] == "failed"
        assert "timestep collapse" in row["reason"]
        assert row["partial_output_files"] == 1, "say what was left behind"
        assert "metrics" not in row, "no metrics from an unfinished run"


class TestTheStageLedgerIsARecordNotAClaim:
    """run_state.json exists so a later invocation can know what is already
    done — an ELM ensemble is ~40 minutes in a queue, and a process that must
    block for it cannot be interrupted or resumed.

    Everything here is about the ledger being SAFE to add: it is bookkeeping,
    so no failure of it may take down a run that actually succeeded.
    """

    def _mgr(self, tmp_path):
        from core.pflotran_exp_manager import PFLOTRANExpManager
        return PFLOTRANExpManager(base_output_dir=str(tmp_path))

    def test_a_stage_is_recorded_with_its_status_and_time(self, tmp_path):
        m = self._mgr(tmp_path)
        m._mark("build_case_inputs", n_experiments=19)
        st = m._load_state()
        assert st["stages"]["build_case_inputs"]["status"] == "done"
        assert st["stages"]["build_case_inputs"]["n_experiments"] == 19
        assert st["stages"]["build_case_inputs"]["at"]
        assert st["model"] == "pflotran"

    def test_artifacts_lists_only_files_that_exist(self, tmp_path):
        """Recording a file that was never written would make the ledger a
        claim rather than a record, and a resume would trust it."""
        m = self._mgr(tmp_path)
        (m.run_dir / "columns.json").write_text("{}")
        assert m._artifacts("columns.json", "never_written.json") == \
            ["columns.json"]

    def test_marking_twice_updates_rather_than_duplicates(self, tmp_path):
        m = self._mgr(tmp_path)
        m._mark("run", status="pending", job_id="770595")
        m._mark("run", status="done", n_results=19)
        e = m._load_state()["stages"]["run"]
        assert e["status"] == "done" and e["n_results"] == 19
        assert e["job_id"] == "770595", "earlier fields must survive"

    def test_a_corrupt_ledger_degrades_to_a_fresh_run(self, tmp_path):
        """Not an exception. An unreadable ledger means 'nothing is known to
        be done', which is exactly a fresh run — the safe reading."""
        m = self._mgr(tmp_path)
        m._state_path().write_text("{ this is not json")
        st = m._load_state()
        assert st["stages"] == {}

    def test_a_ledger_of_the_wrong_shape_is_also_survivable(self, tmp_path):
        m = self._mgr(tmp_path)
        m._state_path().write_text('["a", "list", "not", "an", "object"]')
        assert m._load_state()["stages"] == {}

    def test_marking_never_raises_even_when_it_cannot_write(self, tmp_path):
        """The ledger must never be the reason a completed stage is lost."""
        m = self._mgr(tmp_path)
        m._state_path().mkdir()          # a directory where the file should be
        m._mark("build_case_inputs")                 # must not raise


class TestResumeSkipsWhatIsAlreadyDone:
    """Phase 2. The ledger stops being a record and starts being read.

    Verified live: a run killed after _build_case_inputs, re-entered against the same
    directory, reused materialize and build and carried through to a correct
    experiment.json — with _build_case_inputs having run exactly once in total.
    """

    def _mgr(self, tmp_path):
        from core.pflotran_exp_manager import PFLOTRANExpManager
        return PFLOTRANExpManager(base_output_dir=str(tmp_path))

    def test_resume_is_opt_in(self, tmp_path):
        """A caller re-running execute_plan usually means 'do it again'. Only
        a caller told to continue means 'skip what is done' — guessing wrong
        one way wastes an ensemble, the other silently reuses stale compute.
        """
        src = (ROOT / "src" / "core" / "exp_manager_base.py").read_text()
        assert 'resume = bool(config.get("resume"))' in src

    def test_build_comes_back_exactly_as_it_went_in(self, tmp_path):
        """ELM's cases.json is a list of case PATHS while _build_case_inputs returns a
        list of dicts, so reconstructing from it would be lossy. The base
        persists _build_case_inputs's own return value instead."""
        m = self._mgr(tmp_path)
        exps = [{"id": "col_01", "case_dir": tmp_path / "col_01", "n_cells": 9},
                {"id": "col_02", "case_dir": tmp_path / "col_02", "n_cells": 12}]
        m._save_case_inputs(exps)
        back = m._rehydrate_case_inputs()
        assert [e["id"] for e in back] == ["col_01", "col_02"]
        assert back[0]["n_cells"] == 9
        # Paths come back as strings; every consumer wraps them in Path()
        assert isinstance(back[0]["case_dir"], str)

    def test_extract_comes_back_from_experiment_json(self, tmp_path):
        """_package already writes the rows and their units verbatim, so
        _extract needs no separate persistence."""
        m = self._mgr(tmp_path)
        (m.run_dir / "experiment.json").write_text(json.dumps({
            "columns": [{"case_name": "col_01", "status": "ok",
                         "metrics": {"saturation_mean": 0.53}}],
            "variable_units": {"LIQUID_SATURATION": "-"}}))
        ns = m._rehydrate_extract()
        assert len(ns["rows"]) == 1
        assert ns["rows"][0]["metrics"]["saturation_mean"] == 0.53
        assert ns["units"]["LIQUID_SATURATION"] == "-"
        # was: assert ns.summary["units"] — the rehydrated namespace carried
        # a second place to find units because _package looked in two. The
        # contract has one, so assert what that guarded instead: the units
        # survive the round trip and reach the packager.
        assert m._variable_units(ns)["LIQUID_SATURATION"] == "-"

    def test_rehydration_returns_none_when_the_artifact_is_missing(self, tmp_path):
        """None means 'run the stage', which is the safe reading. Returning an
        empty list would mean 'the stage produced nothing', and the run would
        carry on with no experiments."""
        m = self._mgr(tmp_path)
        assert m._rehydrate_case_inputs() is None
        assert m._rehydrate_extract() is None
        assert m._rehydrate_materialize() is None

    def test_a_ledger_saying_done_with_no_artifact_still_reruns(self, tmp_path):
        """The ledger can outlive its artifacts — a cleaned scratch directory,
        a partial copy. Trusting it over the filesystem would skip a stage
        whose output no longer exists."""
        m = self._mgr(tmp_path)
        m._mark("build_case_inputs", n_experiments=19)
        assert (m._load_state()["stages"]["build_case_inputs"]["status"]) == "done"
        assert m._rehydrate_case_inputs() is None, \
            "no manifest on disk means the stage must run again"


# ─────────────────────────────────────────────────────────────────────
# Phase 3 — a run that SUBMITS instead of finishing
# ─────────────────────────────────────────────────────────────────────
from core.exp_manager_base import ExperimentManagerBase, Pending   # noqa: E402


class _Submits(ExperimentManagerBase):
    """A backend whose _run queues work rather than doing it.

    Stands in for ELM, whose real _run is 40 minutes of SLURM. Everything
    under test here is control flow in the base, so the compute is a stub.
    """
    MODEL = "fake"
    NEEDS_CASE_BUILD = False

    def __init__(self, *a, **kw):
        self.calls = []
        self.polls = []
        self.poll_returns = None          # None == the job is still running
        super().__init__(*a, **kw)

    def _materialize(self, plan, config):
        self.calls.append("materialize")
        return plan

    def _build_case_inputs(self, plan, config):
        self.calls.append("build_case_inputs")
        return [{"case_name": "col_01"}, {"case_name": "col_02"}]

    def _run(self, experiments, config):
        self.calls.append("run")
        if config.get("submit"):
            return Pending("770595", n_cases=len(experiments), queue="short")
        return {e["case_name"]: True for e in experiments}

    def _poll(self, record, experiments, config):
        self.calls.append("poll")
        self.polls.append(record)
        return self.poll_returns

    def _extract(self, experiments, plan=None, config=None):
        self.calls.append("extract")
        import types
        return types.SimpleNamespace(results=[], units={}, summary={})

    def _package(self, plan, analyzer, config):
        self.calls.append("package")

    def _save_llm_input(self, plan, analyzer):
        self.calls.append("llm_input")


class _SubmitsButCannotPoll(_Submits):
    """Submits and has no way to ask whether the job finished."""
    _poll = ExperimentManagerBase._poll


class TestASubmittedEnsembleStopsAndSaysSo:
    """Phase 3. _run may return a job id instead of results.

    The point of the whole ledger: a 40-minute ELM ensemble should not need a
    process sitting on a login node for the duration. _run submits, execute_plan
    records the id and returns, and a later session picks the study back up.
    """

    def _mgr(self, tmp_path, **kw):
        return _Submits(base_output_dir=str(tmp_path), **kw)

    def test_it_stops_before_extract(self, tmp_path):
        """There are no results yet. Extracting anyway would package an empty
        ensemble and report 0/2 — a queued run described as a failed one."""
        m = self._mgr(tmp_path)
        m.execute_plan({}, {"submit": True})
        assert m.calls == ["materialize", "build_case_inputs", "run"]
        assert "extract" not in m.calls
        assert "package" not in m.calls

    def test_the_summary_is_the_same_shape_a_finished_run_returns(self, tmp_path):
        """Callers read experiments_success/experiments_total off this dict.
        Handing them a different shape turns 'your job is queued' into an
        AttributeError three frames away."""
        m = self._mgr(tmp_path)
        s = m.execute_plan({}, {"submit": True})
        for k in ("run_directory", "experiments_total", "experiments_success",
                  "experiments_failed", "experiments", "model_type",
                  "total_runtime_seconds", "output_files"):
            assert k in s, f"pending summary is missing {k}"
        assert s["status"] == "pending"
        assert s["job_id"] == "770595"
        assert "--resume" in s["resume_command"]

    def test_a_queued_column_is_not_a_failed_column(self, tmp_path):
        """Nothing has been asked of these columns yet. Calling them failed
        would put 2 failures in the summary of a healthy job."""
        m = self._mgr(tmp_path)
        s = m.execute_plan({}, {"submit": True})
        assert s["experiments_failed"] == 0
        assert s["experiments_pending"] == 2
        assert s["experiments_success"] == 0
        assert [e["status"] for e in s["experiments"]] == ["pending", "pending"]

    def test_a_finished_run_still_reports_failures_as_failures(self, tmp_path):
        """The pending flag must not soften an ordinary run's accounting."""
        m = self._mgr(tmp_path)
        s = m.execute_plan({}, {})
        assert s["status"] == "completed"
        assert s["experiments_pending"] == 0
        assert s["experiments_success"] == 2

    def test_the_ledger_carries_the_job_id(self, tmp_path):
        """The id is the only thing that makes the run recoverable."""
        m = self._mgr(tmp_path)
        m.execute_plan({}, {"submit": True})
        run = m._load_state()["stages"]["run"]
        assert run["status"] == "pending"
        assert run["job_id"] == "770595"
        assert run["n_cases"] == 2, "Pending detail is kept for _poll"

    def test_resume_polls_instead_of_resubmitting(self, tmp_path):
        """Re-entering a submitted run must not queue a second ensemble."""
        m1 = self._mgr(tmp_path)
        m1.execute_plan({}, {"submit": True})

        m2 = _Submits(base_output_dir=str(tmp_path), run_dir=str(m1.run_dir))
        s = m2.execute_plan({}, {"submit": True, "resume": True})
        assert "poll" in m2.calls
        assert "run" not in m2.calls, "the job was already submitted"
        assert s["status"] == "pending", "_poll said it is still running"
        assert m2.polls[0]["job_id"] == "770595"

    def test_resume_carries_on_when_the_job_landed(self, tmp_path):
        """_poll returns what _run would have returned had it waited, and the
        pipeline continues from there."""
        m1 = self._mgr(tmp_path)
        m1.execute_plan({}, {"submit": True})

        m2 = _Submits(base_output_dir=str(tmp_path), run_dir=str(m1.run_dir))
        m2.poll_returns = {"col_01": True, "col_02": True}
        s = m2.execute_plan({}, {"submit": True, "resume": True})
        assert "extract" in m2.calls and "package" in m2.calls
        assert s["status"] == "completed"
        assert s["experiments_success"] == 2
        assert m2._load_state()["stages"]["run"]["status"] == "done"

    def test_build_is_not_redone_on_the_way_back(self, tmp_path):
        """The manifest is what stops a resume from paying for the build
        twice — the same property Phase 2 established, over a job."""
        m1 = self._mgr(tmp_path)
        m1.execute_plan({}, {"submit": True})
        m2 = _Submits(base_output_dir=str(tmp_path), run_dir=str(m1.run_dir))
        m2.poll_returns = {"col_01": True, "col_02": True}
        m2.execute_plan({}, {"submit": True, "resume": True})
        assert "build_case_inputs" not in m2.calls

    def test_a_backend_that_cannot_poll_still_records_the_id(self, tmp_path):
        """The job is in the queue by then. Raising would discard the one
        thing that makes it recoverable, so it warns and records."""
        m = _SubmitsButCannotPoll(base_output_dir=str(tmp_path))
        s = m.execute_plan({}, {"submit": True})
        assert s["job_id"] == "770595"
        assert m._load_state()["stages"]["run"]["job_id"] == "770595"

    def test_and_then_fails_loudly_on_resume(self, tmp_path):
        """Not silently forever-pending: a missing _poll is a bug in the
        backend, and it should read as one."""
        m1 = _SubmitsButCannotPoll(base_output_dir=str(tmp_path))
        m1.execute_plan({}, {"submit": True})
        m2 = _SubmitsButCannotPoll(base_output_dir=str(tmp_path),
                                   run_dir=str(m1.run_dir))
        with pytest.raises(NotImplementedError, match="770595"):
            m2.execute_plan({}, {"submit": True, "resume": True})

    def test_pending_is_a_type_not_a_status_key(self, tmp_path):
        """_run's success shapes are already loose — {case: bool} and a list of
        dicts. A dict with status='pending' would be indistinguishable from a
        backend that happened to key its results that way."""
        m = self._mgr(tmp_path)
        m.calls = []
        s = m.execute_plan({}, {})       # _run returns a plain dict
        assert s["status"] == "completed"
        assert "extract" in m.calls


class TestTheSchedulerIsAskedProperly:
    """_slurm_state is what _poll leans on, so what it does when SLURM will
    not answer matters more than what it does when SLURM will."""

    def test_no_answer_is_not_finished(self):
        """A squeue timeout or a purged sacct must never read as a completed
        ensemble — the caller decides what other evidence it trusts."""
        assert ExperimentManagerBase._slurm_state("99999999") is None

    def test_a_non_numeric_id_is_refused_without_shelling_out(self):
        assert ExperimentManagerBase._slurm_state("not-a-job") is None

    def test_running_states_are_the_ones_that_keep_waiting(self):
        active = ExperimentManagerBase.ACTIVE_JOB_STATES
        for s in ("PENDING", "RUNNING", "COMPLETING", "CONFIGURING"):
            assert s in active
        for s in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"):
            assert s not in active, f"{s} means SLURM is done with the job"


class TestAnEmptyEnsembleIsNotInterpreted:
    """The Analyzer is four LLM calls. On a 0/2 ensemble it spent 140 s and
    ~22 k tokens to conclude, correctly, that it had no data — steps 2 and 3
    are built to withhold claims when the evidence is absent, so they withheld,
    at full price. A cancelled or timed-out job now lands here routinely."""

    class _Empty(_Submits):
        def _run(self, experiments, config):
            self.calls.append("run")
            return {e["case_name"]: config.get("ok", False) for e in experiments}

    def test_skipped_when_nothing_succeeded(self, tmp_path):
        m = self._Empty(base_output_dir=str(tmp_path))
        s = m.execute_plan({}, {"ok": False})
        assert s["experiments_success"] == 0
        st = m._load_state()["stages"]["analyze"]
        assert st["status"] == "skipped"
        assert st["reason"] == "no successful columns"

    def test_still_run_when_something_did(self, tmp_path):
        m = self._Empty(base_output_dir=str(tmp_path))
        m.execute_plan({}, {"ok": True})
        assert m._load_state()["stages"]["analyze"]["status"] in ("done", "failed")

    def test_packaging_is_not_skipped(self, tmp_path):
        """experiment.json is the record that the run FAILED. Skipping it
        would lose the only account of what happened."""
        m = self._Empty(base_output_dir=str(tmp_path))
        m.execute_plan({}, {"ok": False})
        assert "package" in m.calls
        assert m._load_state()["stages"]["package"]["status"] == "done"

    def test_skipped_is_not_failed(self, tmp_path):
        """A reader of the ledger must not think the Analyzer crashed."""
        m = self._Empty(base_output_dir=str(tmp_path))
        m.execute_plan({}, {"ok": False})
        assert m._load_state()["stages"]["analyze"]["status"] != "failed"

    def test_the_written_report_is_skipped_too(self):
        """workflow.py's report agent is a second LLM call over the same
        nothing."""
        src = (ROOT / "workflow.py").read_text()
        assert "_NothingToReportOn" in src
        assert "if not run_summary.get('experiments_success')" in src


class _SubmitsPrepare(_Submits):
    """A backend whose CASE BUILD is the thing in the queue, not the ensemble.

    D1: an 8-10 minute CIME build is sbatch'd rather than run on a login node,
    so _build_cases hands back a job id exactly as _run does.
    """
    NEEDS_CASE_BUILD = True

    def _build_cases(self, experiments, config=None):
        self.calls.append("build_cases")
        if getattr(self, "prepare_submits", True):
            return Pending("880001", n_cases=len(experiments))
        return None

    def _poll(self, record, experiments, config):
        self.calls.append(f"poll:{record.get('stage')}")
        self.polls.append(record)
        if record.get("stage") == "build_cases":
            return self.prepare_poll_returns
        return self.poll_returns


class TestAnyStageMayHandBackAJobId:
    """Phase 3b. Phase 3 made the RUN stage job-shaped; D1 made prepare one
    too, which is a change to the base rather than to any backend."""

    def _mgr(self, tmp_path, **kw):
        m = _SubmitsPrepare(base_output_dir=str(tmp_path), **kw)
        m.prepare_poll_returns = None
        return m

    def test_a_queued_case_build_stops_the_run(self, tmp_path):
        m = self._mgr(tmp_path)
        s = m.execute_plan({}, {})
        assert s["status"] == "pending"
        assert s["job_id"] == "880001"
        assert "run" not in m.calls, "nothing may be submitted before the cases exist"
        assert "extract" not in m.calls

    def test_the_summary_says_WHICH_stage_is_waiting(self, tmp_path):
        """'job 880001 is queued' does not say whether the cases are being
        built or the ensemble is being simulated, and those are hours apart in
        what happens next."""
        m = self._mgr(tmp_path)
        s = m.execute_plan({}, {})
        assert s["pending_stage"] == "build_cases"

    def test_poll_is_told_which_stage_it_is_polling(self, tmp_path):
        """A backend answers differently for a CIME build than for an
        ensemble, and it cannot tell them apart from a job id."""
        m1 = self._mgr(tmp_path)
        m1.execute_plan({}, {})
        m2 = self._mgr(tmp_path, run_dir=str(m1.run_dir))
        m2.execute_plan({}, {"resume": True})
        assert m2.polls[0]["stage"] == "build_cases"
        assert "poll:build_cases" in m2.calls

    def test_resume_carries_on_when_the_build_landed(self, tmp_path):
        """A polled prepare returning case dirs must re-attach them, then the
        run proceeds to submit the ensemble."""
        m1 = self._mgr(tmp_path)
        m1.execute_plan({}, {})

        m2 = self._mgr(tmp_path, run_dir=str(m1.run_dir))
        m2.prepare_poll_returns = [{"case_name": "col_01", "case_dir": "/x/1"},
                                   {"case_name": "col_02", "case_dir": "/x/2"}]
        s = m2.execute_plan({}, {"resume": True})
        assert m2._load_state()["stages"]["build_cases"]["status"] == "done"
        assert "run" in m2.calls, "with the cases built, the ensemble may go"
        assert s["status"] == "completed"

    def test_build_cases_returning_none_is_not_a_stop(self, tmp_path):
        """None is a legitimate return from _build_cases. Conflating it with the
        STOP sentinel would end every ELM run at the prepare stage."""
        m = self._mgr(tmp_path)
        m.prepare_submits = False
        s = m.execute_plan({}, {})
        assert s["status"] == "completed"
        assert m._load_state()["stages"]["build_cases"]["status"] == "done"

    def test_both_stages_can_queue_in_turn(self, tmp_path):
        """prepare queues, resume collects it, run queues, resume collects
        that — two sessions' worth of waiting in one study."""
        m1 = self._mgr(tmp_path)
        assert m1.execute_plan({}, {})["pending_stage"] == "build_cases"

        m2 = self._mgr(tmp_path, run_dir=str(m1.run_dir))
        m2.prepare_poll_returns = [{"case_name": "col_01"}]
        s2 = m2.execute_plan({}, {"submit": True, "resume": True})
        assert s2["pending_stage"] == "run" and s2["job_id"] == "770595"

        m3 = self._mgr(tmp_path, run_dir=str(m1.run_dir))
        m3.poll_returns = {"col_01": True, "col_02": True}
        s3 = m3.execute_plan({}, {"submit": True, "resume": True})
        assert s3["status"] == "completed"
        assert m3.calls.count("poll:run") == 1
        assert "build_cases" not in m3.calls, "a finished build is not rebuilt"


class TestPreflightStopsBeforeTheModelIsAsked:
    """A run with no frame cannot be investigated, and discovering that inside
    the runner costs a model call plus one subprocess per proposed figure."""

    def _blocked_ctx(self):
        class C(_Ctx):
            def preflight(self):
                return {"frames": {}, "variables": [], "withheld": {},
                        "feasibility": None, "unmet_plan_targets": [],
                        "n_columns": 0,
                        "blocked": "this run packaged no columns, so there is "
                                   "nothing to investigate"}
        return C()

    def test_no_model_call_is_made(self, tmp_path, monkeypatch):
        from agents.analysis import step0_context, step3_interpret
        called = []
        monkeypatch.setattr(step0_context, "load",
                            lambda _rd: self._blocked_ctx())
        monkeypatch.setattr(step3_interpret, "investigate_and_interpret",
                            lambda *a, **k: called.append(1))
        Analyzer(str(tmp_path), verbose=False).run()
        assert called == [], "steps 2-3 must not run when there is no frame"

    def test_the_report_is_still_written(self, tmp_path, monkeypatch):
        """Returning early left a run with no boundary file and no account of
        why — the reader who opens the directory tomorrow gets nothing."""
        from agents.analysis import step0_context
        monkeypatch.setattr(step0_context, "load",
                            lambda _rd: self._blocked_ctx())
        st = Analyzer(str(tmp_path), verbose=False).run()
        assert st["steps"]["report"] is True
        rep = json.loads((tmp_path / "04_analysis" / "analysis.json").read_text())
        assert rep["provenance"]["preflight"]["blocked"]

    def test_a_deliberate_skip_is_not_a_crash(self, tmp_path, monkeypatch):
        """Steps 1-3 are marked False because they did not run — but they did
        not run because the run was inspected and found to hold nothing. That
        is a finding about the evidence, not a malfunction."""
        from agents.analysis import step0_context
        monkeypatch.setattr(step0_context, "load",
                            lambda _rd: self._blocked_ctx())
        Analyzer(str(tmp_path), verbose=False).run()
        rep = json.loads((tmp_path / "04_analysis" / "analysis.json").read_text())
        assert rep["status"] == "insufficient_evidence"
        assert rep["status"] != "analysis_failed"

    def test_the_answer_says_why_rather_than_being_null(self, tmp_path,
                                                        monkeypatch):
        from agents.analysis import step0_context
        monkeypatch.setattr(step0_context, "load",
                            lambda _rd: self._blocked_ctx())
        Analyzer(str(tmp_path), verbose=False).run()
        rep = json.loads((tmp_path / "04_analysis" / "analysis.json").read_text())
        assert rep["answer"] and "nothing to investigate" in rep["answer"]
