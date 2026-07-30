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

    def test_a_declared_value_absent_from_the_finding_is_struck(self):
        r = step3.audit([{"claim": "Recharge is 30 mm/yr", "values": [30],
                          "finding_id": "f1", "caveats": ["no_routing"]}],
                        INV, CAV)
        assert r["kept"] == []
        assert "does not contain it" in r["struck"][0]["struck_because"]

    def test_prose_numbers_are_not_audited(self):
        """The rule needed six exemptions in a row — years, identifiers,
        labels, approximations, run facts, subset counts — each added after it
        struck a correct claim. Six patches on one rule is the rule being
        wrong. Only declared measurements are checked now."""
        r = step3.audit([{"claim": "In band 2, 16 of 19 columns sit below "
                                   "~3600 m and the mean is 31.4",
                          "values": [31.4], "finding_id": "f1",
                          "caveats": ["no_routing"]}], INV, CAV)
        assert len(r["kept"]) == 1, r["struck"]

    def test_a_claim_about_a_caveats_subject_must_carry_it(self):
        r = step3.audit([{"claim": "Runoff generation is 31.4",
                          "finding_id": "f1"}], INV, CAV)
        assert r["kept"] == []
        assert "no_routing" in r["struck"][0]["struck_because"]

    def test_a_claim_NOT_about_that_subject_does_not_owe_it(self):
        """The worst false positive, fixed. A five-variable figure inherits five
        caveats; enforcing all of them on every claim struck "no observational
        validation of the water table is possible" for not carrying a caveat
        scoped to RUNOFF. The figure bounds what is possible; the claim decides
        what applies."""
        r = step3.audit([{"claim": "No water-table observation exists here",
                          "finding_id": "f1"}], INV, CAV)
        assert len(r["kept"]) == 1, r["struck"]

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
        r = step3.audit([{"claim": "x is 99", "values": [99],
                          "finding_id": "f1"}], INV, CAV)
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


def _ctx_with_columns():
    class C:
        columns = [{"case_name": "col_01"}, {"case_name": "col_02"}]
        plan = {"period": {"yr_start": 2019, "yr_end": 2019}}
        caveats: list = []
        data: dict = {}
    return C()


class TestDeclaredValuesOnly:
    """run_facts still bounds what a DECLARED value may be — a claim declaring
    the column count as a measurement is quoting the record."""

    def test_a_declared_run_fact_is_allowed(self):
        r = step3.audit([{"claim": "All 2 columns were simulated", "values": [2],
                          "finding_id": "f1", "caveats": ["no_routing"]}],
                        INV, CAV, facts=step3.run_facts(_ctx_with_columns()))
        assert len(r["kept"]) == 1, r["struck"]

    def test_a_declared_finding_n_is_allowed(self):
        r = step3.audit([{"claim": "Over 19 points", "values": [19],
                          "finding_id": "f1", "caveats": ["no_routing"]}],
                        INV, CAV)
        assert len(r["kept"]) == 1, r["struck"]


def _ctx_with_columns():
    class C:
        columns = [{"case_name": "col_01"}, {"case_name": "col_02"}]
        plan = {"period": {"yr_start": 2019, "yr_end": 2019}}
        caveats: list = []
        data: dict = {}
    return C()


class TestCaveatsBindToTheClaimNotTheFigure:

    def test_scope_terms_split_variables_from_words(self):
        v, w = step3._scope_terms({"applies_to": "runoff (QOVER)"})
        assert v == {"QOVER"} and "runoff" in w

    def test_generic_scope_words_do_not_match_everything(self):
        """"any hydrograph or timing claim" must not fire on every sentence
        containing the word "claim"."""
        _v, w = step3._scope_terms({"applies_to": "any skill claim against these gauges"})
        assert "claim" not in w and "any" not in w
        assert "gauges" in w

    def test_a_variable_name_matches_case_sensitively(self):
        cav = [{"id": "swe", "severity": "blocking",
                "applies_to": "snow (SWE) at stations", "statement": "x"}]
        assert step3.required_caveats("SWE peaks in April", ["swe"], cav) == ["swe"]
        # word boundary: "snow depth" matches, "snowpack" does not. That is the
        # correct reading — a caveat scoped to snow at stations is about the
        # word, and sub-string matching would fire on unrelated compounds.
        assert step3.required_caveats("snow depth peaks", ["swe"], cav) == ["swe"]
        assert step3.required_caveats("the snowpack melts", ["swe"], cav) == []
        assert step3.required_caveats("recharge is small", ["swe"], cav) == []

    def test_only_candidates_from_the_figure_can_be_required(self):
        """A claim mentioning runoff does not owe a caveat if its figure never
        touched runoff — the figure still bounds the candidate set."""
        cav = [{"id": "r", "severity": "blocking",
                "applies_to": "runoff (QOVER)", "statement": "x"}]
        assert step3.required_caveats("runoff is high", [], cav) == []


