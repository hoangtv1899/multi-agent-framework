#!/usr/bin/env python3
"""The Analyzer as a sequencer: steps 0-4, and what happens when one fails.

The steps have their own tests. What is pinned here is the WIRING — that the
box calls them in dependency order, hands each the previous one's output, and
does not lose four working steps because a fifth failed.
"""
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
        elm = (ROOT / "src" / "core" / "elm_exp_manager.py").read_text()
        assert "def execute_plan" not in elm, \
            "ELM must not re-implement the stage sequence"


class TestTheStageSequenceIsSharedNotCopied:
    """execute_plan was lifted out of ELMExpManager so a second backend cannot
    re-implement it. The ordering it enforces — _package before the Analyzer,
    everything from _package on non-fatal — is structural, and two copies of a
    structural guarantee is one copy too many."""

    def test_every_backend_resolves_to_the_base(self):
        from core.exp_manager_base import ExperimentManagerBase as B
        from core.elm_exp_manager import ELMExpManager
        from core.pflotran_exp_manager import PFLOTRANExpManager
        for M in (ELMExpManager, PFLOTRANExpManager):
            assert M.execute_plan is B.execute_plan, M.__name__

    def test_backends_declare_their_stages_rather_than_stubbing(self):
        """A no-op _prepare() reports "prepared nothing, successfully"."""
        from core.elm_exp_manager import ELMExpManager
        from core.pflotran_exp_manager import PFLOTRANExpManager
        assert ELMExpManager.NEEDS_PREPARE and ELMExpManager.NEEDS_SCHEDULER
        assert not PFLOTRANExpManager.NEEDS_PREPARE
        assert not PFLOTRANExpManager.NEEDS_SCHEDULER

    def test_a_missing_stage_raises_rather_than_returning_empty(self):
        """A manager without its compute stage must fail at the first call,
        not report an empty successful run."""
        from core.exp_manager_base import ExperimentManagerBase as B
        import pytest as _pt
        m = B.__new__(B)
        for stage, args in (("_build", ({}, {})), ("_run", ([], {})),
                            ("_extract", ([],))):
            with _pt.raises(NotImplementedError):
                getattr(m, stage)(*args)

    def test_the_two_backends_cannot_claim_each_others_plans(self):
        """Sharing a plan key would make an ELM plan look already-materialized
        to PFLOTRAN, which builds zero experiments WITHOUT raising."""
        from core.elm_exp_manager import ELMExpManager
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
        from core.elm_exp_manager import ELMExpManager
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
        return m._extract(exps).results[0]

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
