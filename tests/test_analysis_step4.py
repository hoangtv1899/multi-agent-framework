#!/usr/bin/env python3
"""Analyzer step 4 — the report.

Pure assembly, so all of it is testable with no API call. What is pinned here
is mostly what step 4 must NOT do: recompute, or quietly drop what the audit
rejected.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.analysis import step4_report as step4    # noqa: E402


class _Ctx:
    plan = {"question": "How does recharge behave?",
            "period": {"yr_start": 2019, "yr_end": 2019}}
    caveats = [{"id": "no_routing", "severity": "blocking",
                "applies_to": "runoff (QOVER)", "statement": "unrouted"}]
    data = {"spinup_dropped": {"days": 14, "timesteps_dropped": 112}}
    columns = [{"case_name": "col_01"}]

    def series(self):
        return None


INV = {"findings": [{"id": "f1", "n": 19, "figure": "/x/f1.png",
                     "script": "/x/f1.py", "variables": ["QOVER"],
                     "result": {"n": 19, "mean": 31.4}}],
       "caveats": [{"id": "figure_failed:f2", "severity": "context",
                    "statement": "f2 produced no result"}]}

INTERP = {"answer": "Recharge is small.", "verdict": "sufficient",
          "audit": {"n_claims": 2, "n_struck": 1},
          "claims": [{"claim": "Mean recharge is 31.4", "finding_id": "f1",
                      "caveats": ["no_routing"]}],
          "struck": [{"claim": "Recharge is about 30", "finding_id": "f1",
                      "struck_because": "states 30, not in finding f1"}]}


def _report(tmp_path):
    return step4.build(_Ctx(), {"caveats": [], "figures": {}}, INV, INTERP,
                       tmp_path, rounds=[{"round": 1}], stopped_because="sufficient")


class TestItAssemblesRatherThanRecomputes:

    def test_the_answer_comes_from_step_3_verbatim(self, tmp_path):
        assert _report(tmp_path)["answer"] == "Recharge is small."

    def test_only_claims_the_audit_kept_are_reported(self, tmp_path):
        r = _report(tmp_path)
        assert [c["claim"] for c in r["claims"]] == ["Mean recharge is 31.4"]

    def test_struck_claims_travel_with_their_reason(self, tmp_path):
        """Deleting them would make the report look like the reviewer never
        disagreed with anything."""
        w = _report(tmp_path)["withheld"]
        assert len(w) == 1 and "not in finding f1" in w[0]["struck_because"]

    def test_each_claim_carries_its_own_evidence(self, tmp_path):
        """A claim whose provenance has to be looked up elsewhere is a claim
        that will not be."""
        c = _report(tmp_path)["claims"][0]
        assert c["figure"] == "/x/f1.png" and c["script"] == "/x/f1.py"
        assert c["n"] == 19

    def test_caveats_from_every_source_are_gathered(self, tmp_path):
        ids = {c.get("id") for c in _report(tmp_path)["caveats"]}
        assert "no_routing" in ids            # from ctx
        assert "figure_failed:f2" in ids      # from step 2


class TestCostIsPartOfTheRecord:

    def test_llm_spend_is_attributed_per_step(self, tmp_path, monkeypatch):
        """A total alone cannot answer "what did step 2 cost" — which is the
        question asked when a run gets expensive."""
        import agents.llm_agent as la
        monkeypatch.setattr(la, "USAGE_LOG", [
            {"label": "step2_investigate", "model": "m", "seconds": 2.0,
             "prompt_tokens": 100, "completion_tokens": 900},
            {"label": "step3_interpret", "model": "m", "seconds": 1.0,
             "prompt_tokens": 800, "completion_tokens": 50}])
        by = _report(tmp_path)["cost"]["llm"]["by_step"]
        assert by["step2_investigate"]["completion_tokens"] == 900
        assert by["step3_interpret"]["prompt_tokens"] == 800

    def test_compute_is_read_back_not_recomputed(self, tmp_path):
        """The manager timed the runs and wrote the numbers down."""
        (tmp_path / "RUN_SUMMARY.json").write_text(json.dumps({
            "total_runtime_seconds": 1873.4, "experiments_total": 19,
            "experiments_success": 19,
            "experiments": [{"runtime_seconds": 90.0},
                            {"runtime_seconds": 110.0}]}))
        c = step4.build(_Ctx(), {}, INV, INTERP, tmp_path)["cost"]["compute"]
        assert c["total_runtime_seconds"] == 1873.4
        assert c["columns_succeeded"] == 19
        assert c["per_column_runtime_seconds"]["max"] == 110.0

    def test_a_missing_run_summary_is_empty_not_an_error(self, tmp_path):
        assert step4.build(_Ctx(), {}, INV, INTERP, tmp_path)["cost"]["compute"] == {}

    def test_compute_falls_back_to_experiment_json_during_a_live_run(
            self, tmp_path):
        """RUN_SUMMARY.json DOES NOT EXIST YET while the Analyzer is running.

        execute_plan writes it last, after the Analyzer it describes, so
        reading only that file made every live run report `compute: null` —
        ELM's included. The numbers appeared only when the Analyzer was re-run
        by hand against a finished directory, so the report claimed to account
        for compute and silently did not. experiment.json is written at stage
        4b, which the base guarantees runs BEFORE the Analyzer.
        """
        (tmp_path / "experiment.json").write_text(json.dumps({
            "columns_total": 19, "columns_succeeded": 19,
            "columns": [{"runtime_seconds": 0.3}, {"runtime_seconds": 0.5},
                        {"runtime_seconds": 0.4}]}))
        c = step4.build(_Ctx(), {}, INV, INTERP, tmp_path)["cost"]["compute"]
        assert c["columns_total"] == 19 and c["columns_succeeded"] == 19
        assert c["total_runtime_seconds"] == 1.2
        assert c["per_column_runtime_seconds"]["max"] == 0.5
        assert "experiment.json" in c["source"], "say which record it came from"

    def test_the_run_summary_still_wins_when_it_exists(self, tmp_path):
        """The fallback is for the live path only; a finished run has the
        fuller record and it must not be shadowed."""
        (tmp_path / "experiment.json").write_text(json.dumps({
            "columns_total": 3, "columns_succeeded": 3,
            "columns": [{"runtime_seconds": 0.3}]}))
        (tmp_path / "RUN_SUMMARY.json").write_text(json.dumps({
            "total_runtime_seconds": 1873.4, "experiments_total": 19,
            "experiments_success": 19, "experiments": []}))
        c = step4.build(_Ctx(), {}, INV, INTERP, tmp_path)["cost"]["compute"]
        assert c["total_runtime_seconds"] == 1873.4
        assert c["columns_total"] == 19

    def test_the_spinup_that_was_dropped_is_reported(self, tmp_path):
        """A series that does not start where the simulation did must say so."""
        p = _report(tmp_path)["provenance"]["spinup_dropped"]
        assert p["days"] == 14 and p["timesteps_dropped"] == 112

    def test_rounds_and_why_it_stopped_are_recorded(self, tmp_path):
        """Two rounds still judged insufficient is a different result from
        passing first time, and the answer alone does not show that."""
        p = _report(tmp_path)["provenance"]
        assert p["stopped_because"] == "sufficient" and len(p["rounds"]) == 1


class TestItIsTheBoundaryFile:

    def test_it_writes_analysis_json(self, tmp_path):
        p = step4.write(_report(tmp_path), tmp_path)
        assert Path(p).name == "analysis.json"
        assert json.loads(Path(p).read_text())["schema"] == "analysis/1"

    def test_the_summary_names_the_cost(self, tmp_path):
        s = step4.summary(_report(tmp_path))
        assert "llm" in s and "compute" in s and "verdict" in s


class TestAPendingSummaryIsNotAnAccounting:
    """Since Phase 3 a detached run writes RUN_SUMMARY.json and RETURNS. On
    resume that file is still the old one — execute_plan overwrites it with
    the real summary only at the very end, after the Analyzer has run.

    Caught end-to-end on job 770696: a 2/2 ensemble that took 9 minutes was
    reported as "2.9 s over 0/2 columns", which was the three seconds the
    submitting call took.
    """

    def _pending(self, d):
        return {"status": "pending", "job_id": "770696",
                "total_runtime_seconds": 2.9,
                "start_time": "2026-08-01T14:39:35",
                "end_time": "2026-08-01T14:39:38",
                "experiments_total": 2, "experiments_success": 0,
                "experiments_failed": 0, "experiments_pending": 2,
                "experiments": [{"case_name": "c1", "runtime_seconds": 0},
                                {"case_name": "c2", "runtime_seconds": 0}]}

    def test_it_falls_through_to_experiment_json(self, tmp_path):
        from agents.analysis.step4_report import _compute_accounting
        (tmp_path / "RUN_SUMMARY.json").write_text(
            json.dumps(self._pending(tmp_path)))
        (tmp_path / "experiment.json").write_text(json.dumps({
            "columns_total": 2, "columns_succeeded": 2,
            "columns": [{"case_name": "c1", "runtime_seconds": 270.0},
                        {"case_name": "c2", "runtime_seconds": 275.0}]}))
        acc = _compute_accounting(tmp_path)
        assert acc["columns_succeeded"] == 2, \
            "a submitted-but-not-finished summary must not be the accounting"
        assert acc["total_runtime_seconds"] == 545.0

    def test_a_completed_summary_is_still_preferred(self, tmp_path):
        from agents.analysis.step4_report import _compute_accounting
        (tmp_path / "RUN_SUMMARY.json").write_text(json.dumps({
            "status": "completed", "total_runtime_seconds": 600.0,
            "experiments_total": 2, "experiments_success": 2,
            "experiments_failed": 0,
            "experiments": [{"case_name": "c1", "runtime_seconds": 300.0}]}))
        (tmp_path / "experiment.json").write_text(json.dumps({
            "columns_total": 2, "columns_succeeded": 1, "columns": []}))
        acc = _compute_accounting(tmp_path)
        assert acc["total_runtime_seconds"] == 600.0
        assert acc["columns_succeeded"] == 2

    def test_a_summary_with_no_status_is_still_read(self, tmp_path):
        """Every run written before Phase 3 has no status key, and those are
        the fuller record — skipping them would be a regression."""
        from agents.analysis.step4_report import _compute_accounting
        (tmp_path / "RUN_SUMMARY.json").write_text(json.dumps({
            "total_runtime_seconds": 2406.0, "experiments_total": 19,
            "experiments_success": 19, "experiments_failed": 0,
            "experiments": []}))
        acc = _compute_accounting(tmp_path)
        assert acc["columns_succeeded"] == 19
