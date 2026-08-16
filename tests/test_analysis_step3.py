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




class TestARoundThatDiscardsWorkSaysSo:
    """A later round REPLACES the investigation wholesale — `investigation` is
    reassigned and only the last one reaches step 4. A round 2 that drops three
    of round 1's five findings used to lose them with nothing recording it.
    The prompt says "keep what worked"; whether it did was unknowable.
    """

    def _loop(self, monkeypatch, tmp_path, r1_ids, r2_ids, r1_kept, r2_kept):
        from agents.analysis import step2_investigate as s2
        seq = iter([
            {"round": 1, "n_succeeded": len(r1_ids), "caveats": [],
             "findings": [{"id": i, "n": 9, "result": {}} for i in r1_ids]},
            {"round": 2, "n_succeeded": len(r2_ids), "caveats": [],
             "findings": [{"id": i, "n": 9, "result": {}} for i in r2_ids]}])
        monkeypatch.setattr(s2, "investigate", lambda *a, **k: next(seq))
        verdicts = iter([("insufficient", r1_kept), ("sufficient", r2_kept)])
        def fake_interpret(ctx, comparison, investigation, out_dir, **kw):
            v, kept = next(verdicts)
            return {"verdict": v, "answer": "a", "feedback": "try again",
                    "claims": [], "struck": [],
                    "audit": {"n_claims": kept, "n_struck": 0}}
        monkeypatch.setattr(step3, "interpret", fake_interpret)
        return step3.investigate_and_interpret(_ctx(), tmp_path, comparison={})

    def test_findings_dropped_by_a_later_round_are_named(self, monkeypatch,
                                                         tmp_path):
        out = self._loop(monkeypatch, tmp_path,
                         ["a", "b", "c"], ["a"], 3, 4)
        r2 = out["rounds"][1]
        assert r2["findings_dropped_from_previous_round"] == ["b", "c"]

    def test_nothing_is_flagged_when_the_round_kept_everything(self, monkeypatch,
                                                               tmp_path):
        out = self._loop(monkeypatch, tmp_path,
                         ["a"], ["a", "b"], 1, 2)
        assert "findings_dropped_from_previous_round" not in out["rounds"][1]

    def test_a_round_that_kept_fewer_claims_is_recorded_as_a_regression(
            self, monkeypatch, tmp_path):
        """Not corrected — the reviewer judged this round's set and that
        judgement stands — but a reader comparing runs needs to know the extra
        round cost claims rather than earning them."""
        out = self._loop(monkeypatch, tmp_path, ["a"], ["a"], 5, 2)
        assert out["rounds"][1]["regressed"]
        assert out["regressed_rounds"] == [2]

    def test_an_improving_round_is_not_flagged(self, monkeypatch, tmp_path):
        out = self._loop(monkeypatch, tmp_path, ["a"], ["a"], 2, 5)
        assert "regressed" not in out["rounds"][1]
        assert out["regressed_rounds"] is None

    def test_every_round_records_which_findings_it_produced(self, monkeypatch,
                                                            tmp_path):
        out = self._loop(monkeypatch, tmp_path, ["a", "b"], ["c"], 2, 2)
        assert out["rounds"][0]["finding_ids"] == ["a", "b"]
        assert out["rounds"][1]["finding_ids"] == ["c"]


class TestTheThreeRulesHaveNames:
    """`cites`, `no_new` and `respects` were labels in the documentation and
    nowhere in the source. A reviewer sent to find the function implementing
    `no_new` found three anonymous blocks inside audit() and no such name. Docs
    pointing at symbols that do not exist are worse than docs with none: the
    reader concludes the code is elsewhere rather than that the name is fiction.
    """

    def test_every_rule_named_in_AUDIT_RULES_is_importable(self):
        for name in step3.AUDIT_RULES:
            assert callable(getattr(step3, f"rule_{name}", None)), \
                f"AUDIT_RULES names {name!r} but rule_{name} does not exist"

    def test_a_struck_claim_says_which_rule_struck_it(self):
        out = step3.audit([{"claim": "x", "finding_id": "nope"}],
                          {"findings": []}, [])
        assert out["struck"][0]["struck_by"] == "cites"

    def test_each_rule_is_attributed_correctly(self):
        findings = {"findings": [{"id": "f1", "n": 9, "result": {"mean": 31.4},
                                  "blocked_by": ["b1"]}]}
        caveats = [{"id": "b1", "severity": "blocking",
                    "applies_to": "runoff (QOVER)", "statement": "unrouted"}]
        cases = [
            ({"claim": "x", "finding_id": "ghost"}, "cites"),
            ({"claim": "x", "finding_id": "f1", "values": [99.9]},
             "declared_values"),
            ({"claim": "runoff is high", "finding_id": "f1", "values": []},
             "respects"),
        ]
        for claim, expected in cases:
            out = step3.audit([claim], findings, caveats)
            assert out["struck"][0]["struck_by"] == expected, claim

    def test_a_kept_claim_carries_no_rule(self):
        out = step3.audit(
            [{"claim": "the mean is 31.4", "finding_id": "f1",
              "values": [31.4], "caveats": []}],
            {"findings": [{"id": "f1", "n": 9, "result": {"mean": 31.4}}]}, [])
        assert out["kept"] and "struck_by" not in out["kept"][0]

    def test_the_rules_run_in_the_declared_order(self):
        """A claim failing two rules is attributed to the first. Otherwise the
        reported reason depends on dict ordering."""
        out = step3.audit(
            [{"claim": "runoff is 99.9", "finding_id": "ghost", "values": [99.9]}],
            {"findings": []}, [])
        assert out["struck"][0]["struck_by"] == step3.AUDIT_RULES[0]

    def test_each_rule_is_callable_on_its_own(self):
        """They are separable so a reader can test one without the loop."""
        assert step3.rule_cites({"finding_id": "a"}, {"a": {}}) is None
        assert step3.rule_cites({"finding_id": "b"}, {"a": {}})
        assert step3.rule_declared_values(
            {"values": [1.0]}, {"result": {"x": 1.0}}, set()) is None
        assert step3.rule_declared_values(
            {"values": [2.0]}, {"result": {"x": 1.0}}, set())
        assert step3.rule_respects({"claim": "snow"}, {"blocked_by": []}, [], {}) is None


class TestTheModelExchangeIsKept:
    """What the model was shown and what it said, verbatim, beside the run.

    Neither survived before 2026-08-14: step 2's reply came back as `raw`, was
    parsed, and dropped when the record was written; step 3 did not keep a name
    for it. Two questions about a finished run had no answer — "the model
    proposed five figures and four appeared, what happened to the fifth?" and
    "did the parser change what the model meant?" — and no end-to-end test
    could replay a round without paying for a live call.
    """

    def _interpret(self, tmp_path, reply):
        class FakeClient:
            label = None
            def ask(self, messages):
                FakeClient.seen = messages
                return reply
        inv = {"round": 1, "findings": [{"id": "f1", "n": 9, "question": "q",
                                         "result": {"mean": 1.0}}],
               "caveats": [], "figures": [], "notes": "n"}
        return step3.interpret(_ctx(), {}, inv, tmp_path,
                               client=FakeClient(), with_images=False), inv

    def test_the_reply_is_saved_verbatim(self, tmp_path):
        reply = ('```json\n{"claims": [], "answer": "a", '
                 '"verdict": "sufficient", "feedback": ""}\n```')
        out, _ = self._interpret(tmp_path, reply)
        p = Path(out["exchange"]["reply"])
        assert p.name == "step3_round1_reply.txt"
        assert p.read_text() == reply, "including the fence the parser strips"

    def test_the_prompt_is_saved_too(self, tmp_path):
        """The worst bug this pipeline has had was a silent cap on the evidence
        shown to the model. It was invisible because nobody could see what was
        sent."""
        out, _ = self._interpret(
            tmp_path, '{"claims": [], "answer": "a", "verdict": "sufficient"}')
        body = Path(out["exchange"]["prompt"]).read_text()
        assert "FINDINGS" in body and "id: f1" in body

    def test_the_prompt_saved_is_the_text_not_the_images(self, tmp_path):
        """`content` is the brief plus every figure base64-encoded. Writing
        that would put tens of megabytes of image bytes in the run directory
        and tell a reader nothing."""
        out, _ = self._interpret(
            tmp_path, '{"claims": [], "answer": "a", "verdict": "sufficient"}')
        body = Path(out["exchange"]["prompt"]).read_text()
        assert "base64" not in body and "image_url" not in body

    def test_a_round_can_be_replayed_from_what_was_saved(self, tmp_path):
        """The point of keeping it: the same reply parses to the same verdict,
        with no model call and no cost."""
        reply = ('{"claims": [{"claim": "the mean is 1.0", "finding_id": "f1",'
                 ' "values": [1.0], "caveats": []}], "answer": "a",'
                 ' "verdict": "sufficient"}')
        first, inv = self._interpret(tmp_path, reply)
        saved = Path(first["exchange"]["reply"]).read_text()

        class Replay:
            label = None
            def ask(self, messages):
                return saved
        again = step3.interpret(_ctx(), {}, inv, tmp_path,
                                client=Replay(), with_images=False)
        assert again["verdict"] == first["verdict"]
        assert len(again["claims"]) == len(first["claims"]) == 1

    def test_an_unwritable_directory_does_not_lose_the_round(self, tmp_path,
                                                             monkeypatch):
        """A run that produced figures is not lost because a log could not be
        written."""
        from agents.analysis import script_runner as sr
        monkeypatch.setattr(Path, "write_text",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
        assert sr.save_exchange(tmp_path, "step2", 1, "p", "r") == {}
