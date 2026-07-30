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

    def test_pflotran_extract_refuses_rather_than_returning_nothing(self):
        """Pending the profiles-vs-depth-axis decision. Returning an empty
        results object would let the base package it and the Analyzer run on
        nothing — success reported over no data."""
        from core.pflotran_exp_manager import PFLOTRANExpManager
        import pytest as _pt
        m = PFLOTRANExpManager.__new__(PFLOTRANExpManager)
        with _pt.raises(NotImplementedError, match="depth-axis"):
            m._extract([])
