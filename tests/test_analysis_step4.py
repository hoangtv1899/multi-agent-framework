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
        # NAMED FOR WHAT IT TIMES. RUN_SUMMARY's total_runtime_seconds is
        # this PROCESS's wall time, not the model's: a detached ensemble
        # returns in ~3 s and a later --finalize times its own tail. It is
        # `framework_seconds` here, and the model's runtime is `ensemble`,
        # read from the scheduler.
        assert c["framework_seconds"] == 1873.4
        assert "total_runtime_seconds" not in c, \
            "the old name read as the model's runtime and was not"
        assert c["columns_succeeded"] == 19
        assert c["per_column_runtime_seconds"]["max"] == 110.0

    def test_a_missing_run_summary_is_empty_not_an_error(self, tmp_path):
        c = step4.build(_Ctx(), {}, INV, INTERP, tmp_path)["cost"]["compute"]
        # `ensemble` is always present and None when the scheduler has
        # nothing to say — absent would read as "not asked".
        assert c == {"ensemble": None}

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
        # Summed per-column MODEL time, so it is `model_seconds`; this path
        # has no framework figure to report and says so with None.
        assert c["model_seconds"] == 1.2
        assert c["framework_seconds"] is None
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
        assert c["framework_seconds"] == 1873.4
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
        assert acc["model_seconds"] == 545.0

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
        assert acc["framework_seconds"] == 600.0
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


# ═════════════════════════════════════════════════════════════════════
# A FAILURE AND A FINDING MUST NOT LOOK THE SAME
# ═════════════════════════════════════════════════════════════════════
# Before 2026-08-14 they did. A run whose steps 2-3 raised reached step 4 as
# `investigation={}, interpretation={}` and produced verdict=None with an empty
# claims list — byte-for-byte the shape of a run that finished and had nothing
# to say. The only record of the difference lived in the dict Analyzer.run()
# RETURNS, which nothing writes down, so the distinction survived exactly as
# long as someone was watching the terminal.
#
# This is the same failure the whole Analyzer is arranged against: a null that
# means "not computed" rendering identically to one that means "computed as
# nothing". These tests are here because it is easy to lose again — every one
# of them passes if `status` is deleted and `verdict` is read instead.

CRASHED = {"context": True, "compare": True,
           "investigate": False, "interpret": False}
RAN     = {"context": True, "compare": True,
           "investigate": True, "interpret": True}


def _built(tmp_path, investigation, interpretation, steps):
    return step4.build(_Ctx(), {"caveats": [], "figures": {}},
                       investigation, interpretation, tmp_path,
                       rounds=[], stopped_because=None, steps=steps)


class TestAFailedRunIsNotAnInconclusiveOne:

    def test_a_crashed_run_says_so(self, tmp_path):
        r = _built(tmp_path, {}, {}, CRASHED)
        assert r["status"] == "analysis_failed"
        assert r["provenance"]["failed_steps"] == ["investigate", "interpret"]

    def test_an_inconclusive_run_is_not_reported_as_a_crash(self, tmp_path):
        """The interpreter ran, looked, and judged the evidence too thin. That
        is a result, and it must not be filed as a malfunction."""
        r = _built(tmp_path, INV, dict(INTERP, verdict="insufficient"), RAN)
        assert r["status"] == "insufficient_evidence"
        assert r["provenance"]["failed_steps"] is None

    def test_the_two_are_distinguishable_without_reading_the_verdict(self, tmp_path):
        """Both carry no usable conclusion; only `status` separates them."""
        crashed = _built(tmp_path, {}, {}, CRASHED)
        thin = _built(tmp_path, INV, dict(INTERP, verdict="insufficient",
                                          claims=[]), RAN)
        assert crashed["claims"] == thin["claims"] == []
        assert crashed["status"] != thin["status"]

    def test_a_missing_step_record_is_treated_as_a_failure(self, tmp_path):
        """A caller that does not say what happened has not said it went well.
        Wrong in the safe direction: better a false alarm than a crash reported
        as a scientific judgement."""
        assert _built(tmp_path, {}, {}, None)["status"] == "analysis_failed"

    def test_a_crash_is_not_read_as_a_verdict(self, tmp_path):
        """verdict=None must never become `insufficient_evidence`: nothing
        judged anything."""
        assert _built(tmp_path, INV, {}, CRASHED)["status"] == "analysis_failed"

    def test_sufficient_with_nothing_surviving_the_audit_is_its_own_status(
            self, tmp_path):
        """The interpreter concluded and every claim was struck. That is not
        `supported`, and calling it so would publish a verdict with no claim
        behind it."""
        r = _built(tmp_path, INV, dict(INTERP, claims=[]), RAN)
        assert r["status"] == "no_supported_claims"
        assert r["verdict"] == "sufficient"

    def test_a_good_run_is_supported(self, tmp_path):
        assert _built(tmp_path, INV, INTERP, RAN)["status"] == "supported"

    def test_which_steps_ran_is_persisted(self, tmp_path):
        """The coordinator has always known this and always dropped it at the
        boundary."""
        assert _built(tmp_path, {}, {}, CRASHED)["provenance"]["steps"] == CRASHED

    def test_the_receipt_names_the_failed_steps(self, tmp_path):
        """The terminal line is what a person actually reads after a run."""
        s = step4.summary(_built(tmp_path, {}, {}, CRASHED))
        assert "analysis_failed" in s
        assert "investigate" in s and "interpret" in s


class TestZeroCostMeansUnmeasuredNotFree:
    """`calls: 0` is already the signature of a gateway failure (job 770905).
    It was also what a report assembled in a different process from the model
    calls produced — the job-B verification run reported 0 calls for an
    analysis that made four."""

    def test_a_verdict_with_no_usage_log_is_flagged_unmeasured(
            self, tmp_path, monkeypatch):
        import agents.llm_agent as la
        monkeypatch.setattr(la, "USAGE_LOG", [])
        llm = _built(tmp_path, INV, INTERP, RAN)["cost"]["llm"]
        assert llm["measured"] is False
        assert "note" in llm, "zeros must say they mean unmeasured"

    def test_the_receipt_says_unmeasured_rather_than_zero(
            self, tmp_path, monkeypatch):
        import agents.llm_agent as la
        monkeypatch.setattr(la, "USAGE_LOG", [])
        s = step4.summary(_built(tmp_path, INV, INTERP, RAN))
        assert "NOT MEASURED" in s

    def test_a_run_that_never_called_a_model_is_not_flagged(
            self, tmp_path, monkeypatch):
        """No verdict means nothing was asked, so an empty log is correct."""
        import agents.llm_agent as la
        monkeypatch.setattr(la, "USAGE_LOG", [])
        assert _built(tmp_path, {}, {}, CRASHED)["cost"]["llm"]["measured"] is True

    def test_the_analyzers_spend_excludes_the_rest_of_the_pipeline(
            self, tmp_path, monkeypatch):
        """Reception and the Planner run in the same process on an end-to-end
        run and do not label their clients, so `total` is pipeline-wide while
        `analyzer` is this box."""
        import agents.llm_agent as la
        monkeypatch.setattr(la, "USAGE_LOG", [
            {"label": "step2_investigate", "model": "m", "seconds": 2.0,
             "prompt_tokens": 100, "completion_tokens": 900},
            {"label": "step3_interpret", "model": "m", "seconds": 1.0,
             "prompt_tokens": 800, "completion_tokens": 50},
            {"label": None, "model": "m", "seconds": 5.0,          # the planner
             "prompt_tokens": 9000, "completion_tokens": 4000}])
        llm = _built(tmp_path, INV, INTERP, RAN)["cost"]["llm"]
        assert llm["total"]["calls"] == 3
        assert llm["analyzer"]["calls"] == 2
        assert llm["analyzer"]["prompt_tokens"] == 900

    def test_unattributed_is_the_unlabelled_remainder_not_everything(
            self, tmp_path, monkeypatch):
        """usage_totals(label=None) means "no filter", so this field used to be
        a second copy of `total`."""
        import agents.llm_agent as la
        monkeypatch.setattr(la, "USAGE_LOG", [
            {"label": "step2_investigate", "model": "m", "seconds": 2.0,
             "prompt_tokens": 100, "completion_tokens": 900},
            {"label": None, "model": "m", "seconds": 5.0,
             "prompt_tokens": 9000, "completion_tokens": 4000}])
        llm = _built(tmp_path, INV, INTERP, RAN)["cost"]["llm"]
        assert llm["unattributed"]["calls"] == 1
        assert llm["unattributed"]["prompt_tokens"] == 9000


class TestTheSlideDeckIsARenderingNotASecondReport:
    """analysis.json stays the one authoritative output. The deck reads it and
    lays it out — a deck that disagreed with the file it came from would be the
    worst artifact in the run, because it is the one that gets presented while
    the file is the one that gets checked."""

    def _deck(self, tmp_path, report):
        pytest.importorskip("pptx")
        from agents.analysis import step4_slides
        from pptx import Presentation
        p = step4_slides.build(report, tmp_path)
        assert p, "python-pptx is installed, so a deck was expected"
        return Presentation(p), p

    def _all_text(self, prs):
        return "\n".join(sh.text_frame.text for s in prs.slides
                         for sh in s.shapes if sh.has_text_frame)

    def test_the_answer_appears_verbatim(self, tmp_path):
        prs, _ = self._deck(tmp_path, _built(tmp_path, INV, INTERP, RAN))
        assert "Recharge is small." in self._all_text(prs)

    def test_every_claim_carries_the_finding_it_rests_on(self, tmp_path):
        """A slide without the id is a sentence with no evidence — which is
        what the audit exists to remove."""
        prs, _ = self._deck(tmp_path, _built(tmp_path, INV, INTERP, RAN))
        t = self._all_text(prs)
        assert "Mean recharge is 31.4" in t and "f1" in t

    def test_struck_claims_travel_with_the_rule_that_struck_them(self, tmp_path):
        rep = _built(tmp_path, INV,
                     dict(INTERP, struck=[{"claim": "Recharge is about 30",
                                           "finding_id": "f1",
                                           "struck_by": "declared_values",
                                           "struck_because": "states 30"}]),
                     RAN)
        prs, _ = self._deck(tmp_path, rep)
        t = self._all_text(prs)
        assert "Recharge is about 30" in t and "declared_values" in t

    def test_a_failed_run_says_so_on_the_title_slide(self, tmp_path):
        """A deck that opens with an answer for a run that did not complete is
        the failure this whole change was about."""
        prs, _ = self._deck(tmp_path, _built(tmp_path, {}, {}, CRASHED))
        first = "\n".join(sh.text_frame.text for sh in prs.slides[0].shapes
                          if sh.has_text_frame)
        assert "analysis failed" in first
        assert "investigate" in first, "name the steps that did not complete"

    def test_blocking_caveats_are_quoted_not_summarised(self, tmp_path):
        prs, _ = self._deck(tmp_path, _built(tmp_path, INV, INTERP, RAN))
        assert "unrouted" in self._all_text(prs)

    def test_the_json_is_written_even_if_the_deck_cannot_be(self, tmp_path,
                                                            monkeypatch):
        """The deck is a second rendering of an artifact already on disk. A
        failure in the presentation layer must not cost the run its result."""
        from agents.analysis import step4_slides
        monkeypatch.setattr(step4_slides, "build",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no")))
        p = step4.write(_report(tmp_path), tmp_path)
        assert Path(p).name == "analysis.json" and Path(p).is_file()

    def test_slides_can_be_turned_off(self, tmp_path):
        step4.write(_report(tmp_path), tmp_path, slides=False)
        assert not (tmp_path / "analysis.pptx").exists()
