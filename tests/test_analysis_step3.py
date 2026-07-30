#!/usr/bin/env python3
"""Analyzer step 3 — the audit and the loop bound.

The judgement needs an API call; the audit does not, and the audit is the part
that must not depend on a model behaving well. It is a pure function, so it is
pinned here in full.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.analysis import step2_investigate as step2   # noqa: E402
from agents.analysis import step3_interpret as step3     # noqa: E402

CAV = [{"id": "no_routing", "severity": "blocking",
        "applies_to": "runoff (QOVER)",
        "statement": "local generation, not routed discharge"},
       {"id": "context_only", "severity": "context",
        "applies_to": "anything", "statement": "x"}]

INV = {"findings": [{"id": "f1", "result": {"n": 19, "mean": 31.4},
                     "blocked_by": ["no_routing"]}]}


class TestTheAuditStrikesUnsupportedClaims:

    def test_a_claim_citing_no_real_finding_is_struck(self):
        r = step3.audit([{"claim": "Recharge is 31.4", "finding_id": "nope"}],
                        INV, CAV)
        assert r["kept"] == [] and "did not produce" in r["struck"][0]["struck_because"]

    def test_a_number_not_in_the_finding_is_struck(self):
        """Including a ROUNDED one. "about 30" cannot be traced back to 31.4 by
        anything downstream, which is the whole point of the citation."""
        r = step3.audit([{"claim": "Recharge is about 30 mm/yr",
                          "finding_id": "f1", "caveats": ["no_routing"]}],
                        INV, CAV)
        assert r["kept"] == []
        assert "does not appear in finding f1" in r["struck"][0]["struck_because"]

    def test_a_claim_ignoring_its_blocking_caveat_is_struck(self):
        r = step3.audit([{"claim": "Recharge is 31.4", "finding_id": "f1"}],
                        INV, CAV)
        assert r["kept"] == []
        assert "no_routing" in r["struck"][0]["struck_because"]

    def test_a_clean_claim_survives(self):
        r = step3.audit([{"claim": "Recharge is 31.4 across 19 columns",
                          "finding_id": "f1", "caveats": ["no_routing"]}],
                        INV, CAV)
        assert len(r["kept"]) == 1 and r["struck"] == []

    def test_years_are_not_treated_as_measurements(self):
        """Demanding provenance for "2019" would fire on prose that is fine."""
        r = step3.audit([{"claim": "In 2019 recharge was 31.4",
                          "finding_id": "f1", "caveats": ["no_routing"]}],
                        INV, CAV)
        assert len(r["kept"]) == 1

    def test_struck_claims_keep_their_reason(self):
        """The audit's own findings are evidence about the run, so a struck
        claim is recorded rather than quietly dropped."""
        r = step3.audit([{"claim": "x is 99", "finding_id": "f1"}], INV, CAV)
        assert r["struck"][0]["struck_because"]
        assert r["n_struck"] == 1 and r["n_claims"] == 1


class TestCaveatScopeMatching:
    """A first version searched each caveat's full statement and struck EVERY
    claim in a live run: limitation_structural_3 is scoped to the water table
    but mentions QDRAI in passing, so every drainage figure inherited it. An
    audit that strikes everything is indistinguishable from a broken one."""

    def test_it_matches_the_scope_not_the_prose(self):
        cav = [{"id": "wtd", "severity": "blocking",
                "applies_to": "water table (ZWT)",
                "statement": "ELM parameterizes lateral losses (QDRAI) locally"}]
        assert step2._blocked_by(["QDRAI"], cav) == []
        assert step2._blocked_by(["ZWT"], cav) == ["wtd"]

    def test_it_is_case_sensitive_so_variables_do_not_match_english(self):
        """SNOW the variable must not match "snow" the word, or every
        precipitation figure inherits the SWE-at-stations caveat."""
        cav = [{"id": "swe", "severity": "blocking",
                "applies_to": "snow (SWE) at stations", "statement": "x"}]
        assert step2._blocked_by(["SNOW"], cav) == []
        assert step2._blocked_by(["H2OSNO"], cav) == []

    def test_context_caveats_do_not_block(self):
        assert step2._blocked_by(["QOVER"], [
            {"id": "c", "severity": "context",
             "applies_to": "runoff (QOVER)", "statement": "x"}]) == []


class TestTheLoopIsBounded:

    def test_max_rounds_is_two(self):
        assert step3.MAX_ROUNDS == 2

    def test_it_stops_on_sufficient_without_a_second_round(self, monkeypatch, tmp_path):
        calls = []

        def fake_investigate(ctx, out_dir, **kw):
            calls.append(kw.get("round_no"))
            return {"round": kw.get("round_no"), "findings": [], "figures": [],
                    "n_succeeded": 0, "caveats": [], "notes": ""}

        monkeypatch.setattr(step2, "investigate", fake_investigate)
        monkeypatch.setattr(step3, "interpret", lambda *a, **k: {
            "verdict": "sufficient", "audit": {"n_claims": 1, "n_struck": 0},
            "feedback": None, "answer": "a", "claims": [], "struck": []})
        r = step3.investigate_and_interpret(_ctx(), tmp_path)
        assert calls == [1] and r["n_rounds"] == 1
        assert r["stopped_because"] == "sufficient"

    def test_it_revises_once_then_stops(self, monkeypatch, tmp_path):
        """A reviewer with an unbounded budget will always find something."""
        calls = []

        def fake_investigate(ctx, out_dir, **kw):
            calls.append(kw.get("feedback"))
            return {"round": kw.get("round_no"), "findings": [], "figures": [],
                    "n_succeeded": 0, "caveats": [], "notes": ""}

        monkeypatch.setattr(step2, "investigate", fake_investigate)
        monkeypatch.setattr(step3, "interpret", lambda *a, **k: {
            "verdict": "insufficient", "audit": {"n_claims": 1, "n_struck": 1},
            "feedback": "do better", "answer": "a", "claims": [], "struck": []})
        r = step3.investigate_and_interpret(_ctx(), tmp_path)
        assert r["n_rounds"] == 2, "must not exceed MAX_ROUNDS"
        assert calls == [None, "do better"], "round 2 must receive the feedback"
        assert "exhausted" in r["stopped_because"]

    def test_insufficient_with_no_actionable_feedback_stops_early(self, monkeypatch, tmp_path):
        monkeypatch.setattr(step2, "investigate", lambda ctx, out_dir, **kw: {
            "round": kw.get("round_no"), "findings": [], "figures": [],
            "n_succeeded": 0, "caveats": [], "notes": ""})
        monkeypatch.setattr(step3, "interpret", lambda *a, **k: {
            "verdict": "insufficient", "audit": {"n_claims": 0, "n_struck": 0},
            "feedback": None, "answer": "a", "claims": [], "struck": []})
        assert step3.investigate_and_interpret(_ctx(), tmp_path)["n_rounds"] == 1

    def test_an_unparseable_verdict_is_not_a_pass(self):
        """Defaulting a malformed verdict to 'sufficient' would let a broken
        reply end the review."""
        import inspect
        src = inspect.getsource(step3.interpret)
        assert 'verdict = "insufficient"' in src


class _ctx:
    plan = {"question": "q"}
    caveats: list = []
    data: dict = {}
    columns: list = []

    def series(self):
        return None


class TestIdentifiersAreNotMeasurements:
    """A live run struck a correct claim for "stating 01, 04, 05, 07" — the
    digits inside col_01, col_04, col_05, col_07, the names of the columns it
    was describing. Provenance is owed for measurements, not for names."""

    def test_column_ids_do_not_need_provenance(self):
        r = step3.audit([{"claim": "col_01 and col_04 both show 31.4",
                          "finding_id": "f1", "caveats": ["no_routing"]}],
                        INV, CAV)
        assert len(r["kept"]) == 1, r["struck"]

    def test_variable_names_with_digits_are_not_measurements(self):
        r = step3.audit([{"claim": "H2OSNO peaks at 31.4", "finding_id": "f1",
                          "caveats": ["no_routing"]}], INV, CAV)
        assert len(r["kept"]) == 1, r["struck"]

    def test_a_real_invented_number_is_still_struck(self):
        """The relaxation must not open a hole in the check it protects."""
        r = step3.audit([{"claim": "col_01 shows 99.9", "finding_id": "f1",
                          "caveats": ["no_routing"]}], INV, CAV)
        assert r["kept"] == [] and "99.9" in r["struck"][0]["struck_because"]


class TestRunLevelFactsAreQuotable:
    """A live run struck "across the 19 sampled columns, precipitation ranges
    ..." because 19 was in ctx, not in the cited finding. The check consulted
    the finding alone, so facts about the run had nowhere to be true."""

    def test_the_column_count_may_be_stated(self):
        r = step3.audit([{"claim": "Across 2 sampled columns, mean is 31.4",
                          "finding_id": "f1", "caveats": ["no_routing"]}],
                        INV, CAV, facts=step3.run_facts(_ctx_with_columns()))
        assert len(r["kept"]) == 1, r["struck"]

    def test_a_findings_own_n_may_be_stated(self):
        r = step3.audit([{"claim": "Over 19 points the mean is 31.4",
                          "finding_id": "f1", "caveats": ["no_routing"]}],
                        INV, CAV)
        assert len(r["kept"]) == 1, r["struck"]

    def test_an_invented_number_is_still_struck(self):
        r = step3.audit([{"claim": "Across 2 columns the mean is 77.7",
                          "finding_id": "f1", "caveats": ["no_routing"]}],
                        INV, CAV, facts=step3.run_facts(_ctx_with_columns()))
        assert r["kept"] == [] and "77.7" in r["struck"][0]["struck_because"]


def _ctx_with_columns():
    class C:
        columns = [{"case_name": "col_01"}, {"case_name": "col_02"}]
        plan = {"period": {"yr_start": 2019, "yr_end": 2019}}
        caveats: list = []
        data: dict = {}
    return C()
