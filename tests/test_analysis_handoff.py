"""The plan shown on the design page, handed to the Analyzer (transitional).

    pytest tests/test_analysis_handoff.py -v

With the analysis preview on, a person reads an analysis plan before the yes.
With this stage's switch on as well, a plan that passed, shown for the design
that was then approved, is the plan the Analyzer carries out: it writes no
second one (src/agents/analysis_handoff.py). These tests hold that stage to
what it says and to what it must not do:

    the promise     it is made when the page is written and kept in the
                    preview's record, so the page and the Analyzer agree, and
                    a switch set or unset after the page changes nothing.
    the handoff     the plan carried out is the plan shown, byte for byte; the
                    Analyzer's planning step makes no model call; its gate
                    holds the plan to the hash its review recorded.
    blocked         another design approved, no approval, a plan or a review
                    that changed, a plan that did not pass, a study that holds
                    another plan: the promise cannot be kept, so the handoff
                    is blocked. The Analyzer writes no other plan, makes no
                    lookup and asks no model, what the study holds is left as
                    it was, and the report is a page that says so.
    no promise      a study whose page promised nothing plans as before.
    a resume        a plan carried out is decided once. A blocked handoff is
                    checked again, and goes through only when its cause is
                    gone.
    late            a year the plan names that a run did not produce is
                    closed as failed, as before, and the plan is not changed.
    the yes         since stage 4 it is of the plan too, and names it by its
                    hash (tests/test_approved_plan.py). A record made before
                    that is handed over as it was promised then.

Every design goes through the coordinator's own path, as in
tests/test_analysis_preview.py, on the synthetic suite of
tests/test_domain_access.py. Every model call is a scripted fake and every
number is made up.
"""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from test_analysis_preview import (BAD_PLAN, QUESTION, WARM, WARM_PLAN, Never, _clients, _co,
                                   _details, _first_part, _reception, _record, _strategy)
from test_domain_access import PLAN_REPLY, REGION, Ctx, Fake, built, study  # noqa: F401
from test_front_door_routes import _typed

from agents import analysis_handoff as AH
from agents import analysis_preview as AV
from agents import design_review as dr
from agents.analysis import analysis_plan as AP
from agents.analysis import step3_interpret as S3
from agents.analysis import step4_report as S4


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def both(monkeypatch, study):
    """Both switches set, on the synthetic control of the `study` fixture."""
    monkeypatch.setenv(AV.SWITCH, "1")
    monkeypatch.setenv(AH.SWITCH, "1")


def _approved(tmp_path, monkeypatch, plan=None, clients=None, answer="yes"):
    """A design taken through the coordinator to its page and answered."""
    rd = tmp_path / "design"
    rd.mkdir()
    plan = plan or _strategy()
    clients = clients or _clients(WARM_PLAN if plan["domain_experiment"]["runs"]
                                  else PLAN_REPLY)
    co = _co(plan, clients, ran=[] if answer == "yes" else None)
    _typed(monkeypatch, [answer])
    co._workflow_design_and_run(_reception(), str(tmp_path), rd)
    return rd, plan


def _after_the_runs(rd, plan, built):
    """The Analyzer's context once the runs are over, and its folder."""
    (rd / "01_inputs" / "domain_control_facts.json").write_text(json.dumps(
        {"control": {"name": "synth", "run_dir": built["control"]["run_dir"]}}))
    de = dict(plan["domain_experiment"])
    de["runs"] = [{"run": r["name"], "changes": []} for r in de.get("runs") or []]
    ctx = Ctx(rd, {"question": QUESTION, "goals": ["GPP in 2016"], "experiment": de})
    assert AP.suite_dirs(ctx)["control"] == str(built["suite"])
    return ctx, rd / "04_analysis"


def _shown(rd):
    return (rd / _record(rd)["plan_file"]).parent


def _no_call():
    return {AP.PLAN_LABEL: Never(), AP.REVIEW_LABEL: Never(),
            AP.REWRITE_LABEL: Never(), AP.REWRITE_REVIEW_LABEL: Never()}


def _no_model_at_all():
    """Every model the Analyzer's loop could ask, each one failing the test."""
    return dict(_no_call(), step2_investigate=Never(), step3_interpret=Never())


def _files(root):
    """Every file under `root` with its hash."""
    return {str(p.relative_to(root)): _sha(p) for p in sorted(Path(root).rglob("*"))
            if p.is_file()}


def _plan_changed_after_the_page(rd):
    p = _shown(rd) / AP.PLAN_FILE
    p.write_text(p.read_text().replace("GPP", "NEE", 1))


# ── the promise, made at the page ────────────────────────────────────────────
class TestThePromise:

    def test_with_the_preview_alone_nothing_is_promised_and_nothing_changes(
            self, monkeypatch, study, built, tmp_path):
        monkeypatch.setenv(AV.SWITCH, "1")
        monkeypatch.delenv(AH.SWITCH, raising=False)
        rd, plan = _approved(tmp_path, monkeypatch)
        assert "handoff" not in _record(rd)
        assert "These comparisons are a preview. Your yes does not approve them" in \
            " ".join(_first_part(rd).split())
        ctx, out = _after_the_runs(rd, plan, built)
        assert AH.hand_over(ctx, out) is None
        assert not out.exists() or not list(out.iterdir())      # no record, no plan

    @pytest.mark.parametrize("value, is_on", [("1", True), ("yes", True), ("on", True),
                                              ("0", False), ("", False), ("no", False)])
    def test_the_switch_is_a_plain_word(self, monkeypatch, value, is_on):
        monkeypatch.setenv(AH.SWITCH, value)
        assert AH.enabled() is is_on

    def test_it_is_off_unless_set(self, monkeypatch):
        monkeypatch.delenv(AH.SWITCH, raising=False)
        assert AH.enabled() is False and AH.promise(True) is None

    def test_a_plan_that_passed_is_promised_and_the_page_says_so(self, both, tmp_path,
                                                                monkeypatch):
        rd, _ = _approved(tmp_path, monkeypatch, plan=_strategy(runs=[WARM]))
        rec = _record(rd)
        assert rec["handoff"] == {"promised": True, "bound": True, "words": AH.APPROVED}
        assert rec["transitional"] == AH.APPROVED
        page = " ".join(_first_part(rd).split())
        # the short page says it in the yes row (2026-11-13); the details keep
        # the record's own words, with what happens if it cannot be done
        assert "The comparisons above are carried out as listed. |" in page
        assert "preview" not in page and "Analyzer" not in page
        assert "writes its plan again" not in page
        assert AH.APPROVED in " ".join(_details(rd).split())
        assert "After the runs the Analyzer carries the plan out as it stands and writes " \
               "no second one." in " ".join(_details(rd).split())
        assert "writes its plan again" not in _details(rd)

    def test_a_design_with_no_new_run_does_not_speak_of_runs_to_come(self, both, tmp_path,
                                                                    monkeypatch):
        rd, _ = _approved(tmp_path, monkeypatch)             # the control run alone
        rec = _record(rd)
        assert rec["handoff"] == {"promised": True, "bound": True,
                                  "words": AH.APPROVED_NO_RUN}
        page = " ".join(_first_part(rd).split())
        assert "| yes | The analysis of the control run's existing output starts." in page
        assert "The comparisons above are carried out as listed. |" in page
        assert "after the run" not in page.lower() and "writes its plan again" not in page
        assert AH.APPROVED_NO_RUN in " ".join(_details(rd).split())
        assert "If it cannot, the analysis stops and its report says why." in \
            " ".join(_details(rd).split())

    def test_a_plan_that_did_not_pass_is_not_promised(self, both, tmp_path, monkeypatch):
        rd, _ = _approved(tmp_path, monkeypatch,
                          clients=_clients(BAD_PLAN, rewrite=BAD_PLAN))
        rec = _record(rd)
        assert rec["status"] == AV.NOT_PASSED
        assert rec["handoff"] == {"promised": False, "why": AH.NOT_PASSED}
        page = " ".join(_first_part(rd).split())
        assert "carries the plan out" not in page and "A yes approves this plan" not in page

    def test_a_switch_changed_after_the_page_changes_nothing(self, both, built, tmp_path,
                                                             monkeypatch):
        """The analysis is the last line of a batch job, where a switch set in
        a login shell does not exist: the Analyzer reads the record."""
        rd, plan = _approved(tmp_path, monkeypatch)
        monkeypatch.delenv(AH.SWITCH)
        monkeypatch.delenv(AV.SWITCH)
        ctx, out = _after_the_runs(rd, plan, built)
        assert AH.hand_over(ctx, out)["used"] is True

    def test_a_switch_set_after_a_page_that_promised_nothing_changes_nothing(
            self, monkeypatch, study, built, tmp_path):
        monkeypatch.setenv(AV.SWITCH, "1")
        monkeypatch.delenv(AH.SWITCH, raising=False)
        rd, plan = _approved(tmp_path, monkeypatch)
        monkeypatch.setenv(AH.SWITCH, "1")
        ctx, out = _after_the_runs(rd, plan, built)
        assert AH.hand_over(ctx, out) is None


# ── the handoff ──────────────────────────────────────────────────────────────
class TestThePlanCarriedOutIsThePlanShown:

    def test_byte_for_byte_and_with_no_model_call(self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        src = _shown(rd)
        before = {p.name: _sha(p) for p in src.iterdir()}
        ctx, out = _after_the_runs(rd, plan, built)
        rec = AH.hand_over(ctx, out)
        assert rec["used"] is True and rec["why"] == AH.USED
        assert rec["plan_sha256"] == _record(rd)["plan_sha256"] == AP.plan_sha256(out)
        assert rec["design_sha256"] == dr.design_fingerprint(plan)
        assert rec["files"] == sorted(before)
        assert rec["approval"]["decision"]
        assert rec["recheck"] == {"against": "the runs as they finished", "problems": []}
        # every planning file is in the study's folder, the same bytes
        for name, sha in before.items():
            assert _sha(out / name) == sha, name
        # the shown files are left as they were
        assert {p.name: _sha(p) for p in src.iterdir()} == before
        # THE ANALYZER'S PLANNING STEP FINDS THEM AND ASKS NO MODEL
        review, ok = S3.plan_and_review(ctx, out, clients=_no_call())
        assert ok is True and review["passed"] is True
        assert AP.plan_sha256(out) == rec["plan_sha256"]
        AP.check_gate(out)                                # the gate opens on it
        assert json.loads((out / AH.RECORD_FILE).read_text()) == rec

    def test_the_analyzer_hands_itself_the_plan_before_it_plans(self, both, built, tmp_path,
                                                               monkeypatch):
        """The Analyzer's own loop makes the handoff, then its planning step,
        then its first round. Stopped at that round: by then the shown plan is
        in place and no planning call was made."""
        from agents.analysis import step2_investigate as S2

        class Stop(Exception):
            pass

        def first_round(*a, **k):
            raise Stop()
        rd, plan = _approved(tmp_path, monkeypatch)
        ctx, out = _after_the_runs(rd, plan, built)
        monkeypatch.setattr(S2, "investigate", first_round)
        with pytest.raises(Stop):
            S3.investigate_and_interpret(ctx, out, clients=_no_call())
        assert AP.plan_sha256(out) == _record(rd)["plan_sha256"]
        assert AH.load(out)["used"] is True

    def test_a_plan_changed_after_the_handoff_no_longer_passes_the_gate(
            self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        ctx, out = _after_the_runs(rd, plan, built)
        AH.hand_over(ctx, out)
        p = out / AP.PLAN_FILE
        p.write_text(p.read_text().replace("GPP", "NEE", 1))
        assert AP.plan_passed(out) == (False, "the analysis plan changed after its review")
        with pytest.raises(AP.PlanNotApproved):
            AP.check_gate(out)

    def test_a_plan_rewritten_once_is_handed_over_with_its_first_version(
            self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch,
                             clients=_clients(BAD_PLAN, rewrite=PLAN_REPLY))
        assert _record(rd)["attempts"] == 2 and AH.promised(_record(rd))
        ctx, out = _after_the_runs(rd, plan, built)
        rec = AH.hand_over(ctx, out)
        assert rec["used"] is True
        assert {AP.PLAN_FILE, AP.REVIEW_FILE, AP.PLAN_V1_FILE, AP.REVIEW_V1_FILE} <= \
            set(rec["files"])
        versions = AP.plan_versions(out)
        assert [v["passed"] for v in versions] == [False, True]
        assert AP.can_rewrite(out)[0] is False            # there is no third writing
        assert S3.plan_and_review(ctx, out, clients=_no_call())[1] is True

    def test_an_unattended_study_is_handed_its_plan_too(self, both, built, tmp_path):
        rd = tmp_path / "design"
        rd.mkdir()
        plan = _strategy()
        co = _co(plan, _clients(), interactive=False, ran=[])
        co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        assert AH.promised(_record(rd))
        ctx, out = _after_the_runs(rd, plan, built)
        assert AH.hand_over(ctx, out)["used"] is True


# ── when the promise cannot be kept ──────────────────────────────────────────
class TestWhenThePromiseCannotBeKept:
    """Each case: the handoff is blocked. Nothing is copied, the reason is on
    record, and nothing passes the Analyzer's gate, so no other plan is used
    and no lookup is made."""

    def _blocked(self, ctx, out, reason):
        rec = AH.hand_over(ctx, out)
        assert rec["used"] is False and rec["blocked"] is True and rec["files"] == []
        assert AH.blocked(rec) and reason in rec["why"], rec["why"]
        assert not (out / AP.PLAN_FILE).exists() and not (out / AP.REVIEW_FILE).exists()
        assert json.loads((out / AH.RECORD_FILE).read_text()) == rec
        # THE GATE REFUSES, with the handoff's reason and not "there is no plan"
        assert AP.plan_passed(out) == (False, AH.BLOCKED.format(why=rec["why"]))
        assert "writes no other plan in its place" in AP.plan_passed(out)[1]
        with pytest.raises(AP.PlanNotApproved, match="could not be handed to it"):
            AP.check_gate(out)
        return rec

    def test_another_design_was_approved(self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        other = _strategy(years=(2015,))
        dr.approve(rd, other, "accepted at the terminal")
        ctx, out = _after_the_runs(rd, plan, built)
        self._blocked(ctx, out, "the design that was approved is not the design this plan "
                                 "was written for")

    def test_no_approval_is_on_record(self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        (rd / dr.APPROVAL_FILE).unlink()
        ctx, out = _after_the_runs(rd, plan, built)
        self._blocked(ctx, out, "no approval of the design is on record")

    def test_the_plan_changed_after_the_page(self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        p = _shown(rd) / AP.PLAN_FILE
        p.write_text(p.read_text().replace("GPP", "NEE", 1))
        ctx, out = _after_the_runs(rd, plan, built)
        self._blocked(ctx, out, "the plan on disk is not the plan that was shown")

    def test_the_plan_is_gone(self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        (_shown(rd) / AP.PLAN_FILE).unlink()
        ctx, out = _after_the_runs(rd, plan, built)
        self._blocked(ctx, out, "the plan shown is no longer on disk")

    @pytest.mark.parametrize("damage, reason", [
        (lambda p: p.unlink(), "has not been reviewed"),
        (lambda p: p.write_text(json.dumps(dict(json.loads(p.read_text()), passed=False))),
         "did not pass step 3's review"),
        (lambda p: p.write_text(json.dumps(dict(json.loads(p.read_text()),
                                                plan_sha256="0" * 64))),
         "changed after its review")])
    def test_the_review_is_gone_failed_or_of_another_plan(self, both, built, tmp_path,
                                                          monkeypatch, damage, reason):
        rd, plan = _approved(tmp_path, monkeypatch)
        damage(_shown(rd) / AP.REVIEW_FILE)
        ctx, out = _after_the_runs(rd, plan, built)
        rec = self._blocked(ctx, out, "the plan shown no longer passes")
        assert reason in rec["why"]

    def test_a_promise_cannot_be_forged_for_a_plan_that_did_not_pass(
            self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch,
                             clients=_clients(BAD_PLAN, rewrite=BAD_PLAN))
        rec = _record(rd)
        rec["handoff"] = {"promised": True, "words": AH.CARRIED_OUT_NO_RUN}
        (rd / AV.RECORD_FILE).write_text(json.dumps(rec))
        ctx, out = _after_the_runs(rd, plan, built)
        self._blocked(ctx, out, AH.NOT_PASSED)

    def test_the_study_already_holds_another_plan(self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        ctx, out = _after_the_runs(rd, plan, built)
        own = Fake(dict(PLAN_REPLY, comparisons=[]))
        AP.write_plan(ctx, out, client=own)
        mine = _sha(out / AP.PLAN_FILE)
        assert mine != _record(rd)["plan_sha256"]
        rec = AH.hand_over(ctx, out)
        assert rec["used"] is False and rec["blocked"] is True
        assert "already holds planning files that differ" in rec["why"]
        assert AP.PLAN_FILE in rec["why"]
        assert _sha(out / AP.PLAN_FILE) == mine           # the plan there is untouched
        # THAT PLAN IS NOT THE ONE PROMISED, so it is not used either: its
        # files would pass by themselves, and the gate still refuses
        S3.rule_plan_follows_question(ctx, out, client=Fake(
            {"follows": True, "why": "it asks what was asked", "problems": []}))
        assert AP.plan_file_passed(out)[0] is True
        assert AP.plan_passed(out)[0] is False
        with pytest.raises(AP.PlanNotApproved):
            AP.check_gate(out)
        loop = S3.investigate_and_interpret(ctx, out, clients=_no_model_at_all(),
                                            client=Never())
        assert loop["interpretation"]["skipped"] is True
        assert _sha(out / AP.PLAN_FILE) == mine

    def test_a_design_that_was_declined_has_no_handoff(self, both, built, tmp_path,
                                                       monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch, answer="no")
        assert AH.promised(_record(rd))                   # the page promised, for a yes
        ctx, out = _after_the_runs(rd, plan, built)
        self._blocked(ctx, out, "no approval of the design is on record")

    def test_an_unreadable_preview_record_is_no_promise(self, both, built, tmp_path,
                                                        monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        (rd / AV.RECORD_FILE).write_text("{ not json")
        ctx, out = _after_the_runs(rd, plan, built)
        assert AH.hand_over(ctx, out) is None

    def test_it_never_raises_and_leaves_no_half_copied_plan(self, both, built, tmp_path,
                                                            monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        ctx, out = _after_the_runs(rd, plan, built)
        real = shutil.copyfile

        def fails_on_the_plan(src, dst, *a, **k):
            if Path(src).name == AP.PLAN_FILE:
                raise OSError("the disk is full")
            return real(src, dst, *a, **k)
        monkeypatch.setattr(AH.shutil, "copyfile", fails_on_the_plan)
        rec = AH.hand_over(ctx, out)
        assert rec["used"] is False and "the disk is full" in rec["why"]
        assert rec["blocked"] is True and AP.plan_passed(out)[0] is False
        left = sorted(p.name for p in out.iterdir())
        assert left == [AH.RECORD_FILE]                   # what was copied is taken back


# ── the Analyzer, when the handoff is blocked ────────────────────────────────
class TestTheAnalyzerStopsOnABlockedHandoff:
    """The user's decision of 2026-11-12: a study that promised the plan
    shown does not plan again by itself when that plan cannot be handed over.
    It keeps what it holds and reports the blocked handoff."""

    def _blocked_study(self, tmp_path, monkeypatch, built):
        rd, plan = _approved(tmp_path, monkeypatch)
        _plan_changed_after_the_page(rd)
        ctx, out = _after_the_runs(rd, plan, built)
        # what the earlier steps of a study leave behind, made up here
        (rd / "03_results").mkdir(exist_ok=True)
        (rd / "03_results" / "regional_numbers.csv").write_text("year,value\n2016,1.5\n")
        out.mkdir(exist_ok=True)
        (out / "domain_report.json").write_text(json.dumps({"rows": 3}))
        (out / "numbers_check.json").write_text(json.dumps({"mismatches": 0}))
        return rd, ctx, out

    def test_it_writes_no_other_plan_and_asks_no_model(self, both, built, tmp_path,
                                                       monkeypatch):
        rd, ctx, out = self._blocked_study(tmp_path, monkeypatch, built)
        planned = []
        monkeypatch.setattr(S3, "plan_and_review",
                            lambda *a, **k: planned.append(1) or pytest.fail(
                                "the Analyzer planned again by itself"))
        loop = S3.investigate_and_interpret(ctx, out, clients=_no_model_at_all(),
                                            client=Never())
        assert planned == []
        for name in (AP.PLAN_FILE, AP.REVIEW_FILE, AP.PLAN_V1_FILE, AP.LOOKUPS_FILE):
            assert not (out / name).exists(), name
        rec = AH.load(out)
        assert rec["blocked"] is True
        assert "the plan on disk is not the plan that was shown" in rec["why"]
        # one round that proposed nothing, and step 3 not asked
        assert loop["n_rounds"] == 1 and "plan_review" not in loop
        inv, itp = loop["investigation"], loop["interpretation"]
        assert inv["n_proposed"] == 0 and inv["findings"] == []
        assert inv["caveats"][0]["id"] == "analysis_plan_not_approved"
        assert "could not be handed to it" in inv["caveats"][0]["statement"]
        assert itp["skipped"] is True and itp["claims"] == []
        assert itp["skipped_because"] == AH.BLOCKED.format(why=rec["why"])
        assert loop["stopped_because"] == itp["skipped_because"] + ", so step 3 was not asked"

    def test_what_the_study_holds_is_left_as_it_was(self, both, built, tmp_path,
                                                    monkeypatch):
        rd, ctx, out = self._blocked_study(tmp_path, monkeypatch, built)
        before = _files(rd)
        S3.investigate_and_interpret(ctx, out, clients=_no_model_at_all(), client=Never())
        after = _files(rd)
        assert {k: after.get(k) for k in before} == before     # nothing changed or gone
        new = sorted(set(after) - set(before))
        assert new == sorted(f"04_analysis/{n}" for n in (
            AH.RECORD_FILE, "investigation.json", "interpretation.json"))
        # the plan that was shown is still where the page left it
        assert (_shown(rd) / AP.PLAN_FILE).is_file() and (_shown(rd) / AP.REVIEW_FILE).is_file()

    def test_the_report_is_a_page_that_says_the_handoff_is_blocked(self, both, built,
                                                                    tmp_path, monkeypatch):
        rd, ctx, out = self._blocked_study(tmp_path, monkeypatch, built)
        loop = S3.investigate_and_interpret(ctx, out, clients=_no_model_at_all(),
                                            client=Never())
        report = S4.build(ctx, {}, loop["investigation"], loop["interpretation"], rd)
        ap = report["analysis_plan"]
        assert ap["passed"] is False and ap["versions"] == []
        assert ap["handoff"]["blocked"] is True
        assert "could not be handed to it" in report["withheld_because"]
        assert "no lookup was made" in report["withheld_because"]
        assert not report.get("claims") and not report.get("headline")
        page = S4.render_markdown(report)
        assert page.startswith("# BLOCKED: the plan shown on the design page could not be "
                               "carried out")
        flat = " ".join(page.split())
        assert "the page said the Analyzer would carry it out as it stood" in flat
        assert "the plan on disk is not the plan that was shown" in flat
        assert "It wrote no other plan in its place, so no result was looked up and no " \
               "claim was made." in flat
        assert "Nothing was deleted or written over." in flat
        assert f"04_analysis/{AH.RECORD_FILE}" in flat
        assert str(_shown(rd).relative_to(rd)) in flat
        assert "did not pass its review" not in flat       # it is not the failed-plan page
        assert "\u2014" not in page
        S4.write(report, out, slides=False)
        assert (out / "REPORT.md").read_text() == page

    def test_a_study_whose_page_made_no_promise_plans_as_before(self, monkeypatch, study,
                                                               built, tmp_path):
        """LEGACY: the preview alone, with no promise on record. The Analyzer
        writes its own plan after the runs, by its own model call."""
        from agents.analysis import step2_investigate as S2

        class Stop(Exception):
            pass

        def first_round(*a, **k):
            raise Stop()
        monkeypatch.setenv(AV.SWITCH, "1")
        monkeypatch.delenv(AH.SWITCH, raising=False)
        rd, plan = _approved(tmp_path, monkeypatch)
        _plan_changed_after_the_page(rd)                  # of no account: nothing was promised
        ctx, out = _after_the_runs(rd, plan, built)
        own = _clients()
        monkeypatch.setattr(S2, "investigate", first_round)
        with pytest.raises(Stop):
            S3.investigate_and_interpret(ctx, out, clients=own)
        assert len(own[AP.PLAN_LABEL].prompts) == 1       # the Analyzer asked its model
        assert AP.plan_passed(out)[0] is True
        assert not (out / AH.RECORD_FILE).exists() and AH.why_blocked(out) is None


# ── a resume ─────────────────────────────────────────────────────────────────
class TestAResume:

    def test_the_decision_is_made_once(self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        ctx, out = _after_the_runs(rd, plan, built)
        first = AH.hand_over(ctx, out)
        files = {p.name: (_sha(p), p.stat().st_mtime_ns) for p in out.iterdir()}
        assert AH.hand_over(ctx, out) == first
        assert {p.name: (_sha(p), p.stat().st_mtime_ns) for p in out.iterdir()} == files

    def test_a_handoff_cut_short_before_its_record_is_finished(self, both, built, tmp_path,
                                                               monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        ctx, out = _after_the_runs(rd, plan, built)
        AH.hand_over(ctx, out)
        (out / AH.RECORD_FILE).unlink()                   # the plan is there, the record not
        there = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
        rec = AH.hand_over(ctx, out)
        assert rec["used"] is True and AP.plan_sha256(out) == rec["plan_sha256"]
        # what was in place is left alone, not written again
        assert {n: (out / n).stat().st_mtime_ns for n in there} == there

    def test_a_handoff_cut_short_before_the_plan_is_made_again(self, both, built, tmp_path,
                                                              monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        ctx, out = _after_the_runs(rd, plan, built)
        AH.hand_over(ctx, out)
        (out / AH.RECORD_FILE).unlink()
        (out / AP.PLAN_FILE).unlink()                     # the review is there, the plan not
        rec = AH.hand_over(ctx, out)
        assert rec["used"] is True and AP.plan_passed(out)[0] is True

    def test_a_blocked_handoff_stays_blocked_while_its_cause_stands(
            self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        (rd / dr.APPROVAL_FILE).unlink()
        ctx, out = _after_the_runs(rd, plan, built)
        first = AH.hand_over(ctx, out)
        again = AH.hand_over(ctx, out)
        assert first["blocked"] is True and again["blocked"] is True
        assert again["why"] == first["why"] and again["first_blocked_at"] == first["at"]
        assert AH.hand_over(ctx, out)["first_blocked_at"] == first["at"]
        assert not (out / AP.PLAN_FILE).exists() and AP.plan_passed(out)[0] is False

    def test_a_blocked_handoff_goes_through_once_its_cause_is_gone(
            self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        approval = (rd / dr.APPROVAL_FILE).read_text()
        (rd / dr.APPROVAL_FILE).unlink()
        ctx, out = _after_the_runs(rd, plan, built)
        first = AH.hand_over(ctx, out)
        assert first["blocked"] is True and AP.plan_passed(out)[0] is False
        (rd / dr.APPROVAL_FILE).write_text(approval)      # the cause is gone
        rec = AH.hand_over(ctx, out)
        assert rec["used"] is True and "blocked" not in rec
        assert rec["first_blocked_at"] == first["at"]     # the first stop stays on record
        assert AP.plan_sha256(out) == _record(rd)["plan_sha256"]
        assert AH.why_blocked(out) is None and AP.plan_passed(out)[0] is True
        assert S3.plan_and_review(ctx, out, clients=_no_call())[1] is True

    def test_a_plan_put_in_the_folder_meanwhile_keeps_it_blocked(
            self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        approval = (rd / dr.APPROVAL_FILE).read_text()
        (rd / dr.APPROVAL_FILE).unlink()
        ctx, out = _after_the_runs(rd, plan, built)
        first = AH.hand_over(ctx, out)
        AP.write_plan(ctx, out, client=Fake(dict(PLAN_REPLY, comparisons=[])))
        mine = _sha(out / AP.PLAN_FILE)                    # a plan that was not shown
        (rd / dr.APPROVAL_FILE).write_text(approval)
        rec = AH.hand_over(ctx, out)
        assert rec["blocked"] is True and rec["first_blocked_at"] == first["at"]
        assert "already holds planning files that differ" in rec["why"]
        assert _sha(out / AP.PLAN_FILE) == mine and AP.plan_passed(out)[0] is False


# ── late: what a run did not produce ─────────────────────────────────────────
class TestWhatARunDidNotProduce:

    def test_a_planned_year_that_was_never_made_is_closed_as_failed(self, both, built,
                                                                    tmp_path, monkeypatch):
        """The plan was written before the run, with the run's years as
        planned. The run made nothing. The plan is carried out as it stands,
        the data refuses the look-up, and the analysis is closed as failed."""
        import domain_access as DAC
        design = _strategy(runs=[WARM])
        design["domain_experiment"]["analysis"]["question_type"] = "hypothesis"
        reply = dict(WARM_PLAN, screen={"when": "last", "region": REGION, "year": 2016,
                                        "why": "a hypothesis"})
        rd, plan = _approved(tmp_path, monkeypatch, plan=design, clients=_clients(reply))
        ctx, out = _after_the_runs(rd, plan, built)
        rec = AH.hand_over(ctx, out)
        assert rec["used"] is True
        shown = _sha(_shown(rd) / AP.PLAN_FILE)
        acc = AP.make_access(AP.access_spec(ctx), script="fig")
        with pytest.raises(DAC.AccessError, match="no suite for run 'warm'"):
            acc.difference("warm", "GPP", "2016", region=REGION)
        rows = {r["id"]: r for r in AP.close_planned_phase(
            out, when="before the screen", by="the test")["analyses"]}
        assert rows["P1"]["status"] == "failed" and rows["P1"]["unmet"] == ["O1"]
        assert "no suite for run 'warm'" in rows["P1"]["reason"]
        assert _sha(out / AP.PLAN_FILE) == shown          # the plan was not changed

    def test_the_check_made_again_after_the_runs_is_recorded_and_changes_nothing(
            self, both, built, tmp_path, monkeypatch):
        """A run that ended a year short: the plan's own check, made again
        against the runs as they finished, names the year. The plan stays the
        plan shown, and the report says what the check found."""
        rd, plan = _approved(tmp_path, monkeypatch, plan=_strategy(runs=[WARM]))
        ctx, out = _after_the_runs(rd, plan, built)
        ctx.plan["experiment"]["years"] = [2015]
        rec = AH.hand_over(ctx, out)
        assert rec["used"] is True
        assert any("2016 is not a year of warm" in p for p in rec["recheck"]["problems"])
        assert AP.plan_sha256(out) == _record(rd)["plan_sha256"]
        said = " ".join(AH.report_lines(rec))
        assert "Checked again against the runs as they finished" in said
        assert "2016 is not a year of warm" in said


# ── what the report and the study's index say ────────────────────────────────
class TestWhatIsSaidAfterwards:

    ROW = {"id": "P1", "status": "completed", "required": False, "outputs": [], "unmet": []}

    def _section(self, handoff):
        return "\n".join(S4.planned_phase_lines({
            "planned_phase": {"analyses": [self.ROW], "counts": {}},
            "analysis_plan": {"handoff": handoff} if handoff else {}}))

    def test_a_plan_approved_and_carried_out_is_said_in_the_report(self):
        text = self._section({"used": True, "approved": True, "why": AH.USED,
                              "recheck": {"problems": []}})
        assert "This plan was shown on the design page and approved with the yes, and the " \
               "Analyzer carried it out as it stood." in text
        assert "the record of the yes names it by its hash" in text
        assert "was not tied to this plan" not in text

    def test_a_plan_carried_out_under_a_yes_that_did_not_name_it_is_said_so(self):
        """A study approved before the yes named plans (stage 3)."""
        text = self._section({"used": True, "why": AH.USED, "recheck": {"problems": []}})
        assert "This plan was shown on the design page before the yes, and the Analyzer " \
               "carried it out as it stood." in text
        assert "It was not tied to this plan, since approval of the analysis was not built " \
               "when this study was approved." in text
        assert "approved with the yes" not in text

    def test_a_blocked_handoff_is_said_with_its_reason(self):
        said = " ".join(AH.report_lines({"used": False, "blocked": True,
                                         "why": "a made-up reason"}))
        assert "The Analyzer could not be handed that plan: a made-up reason. It wrote no " \
               "other plan in its place" in said
        assert "wrote the plan below" not in said

    def test_a_reason_that_quotes_a_model_carries_no_dash_to_the_reader(self):
        said = " ".join(AH.blocked_lines({"used": False, "why": "a review said \u2014 no"}))
        assert "\u2014" not in said and "a review said" in said

    def test_a_study_with_no_promise_says_nothing(self):
        assert AH.report_lines(None) == [] and AH.report_lines({}) == []
        assert "design page" not in self._section(None)

    def test_the_report_record_carries_the_handoff(self, both, built, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        ctx, out = _after_the_runs(rd, plan, built)
        rec = AH.hand_over(ctx, out)
        assert AH.load(out) == rec and AH.load(tmp_path / "nowhere") == {}
        report = S4.build(ctx, {}, {"findings": []},
                          {"verdict": "sufficient", "headline": None, "claims": []}, rd)
        assert report["analysis_plan"]["passed"] is True
        assert report["analysis_plan"]["handoff"] == rec

    def test_a_study_with_no_promise_has_no_handoff_in_its_report(self, study):
        from test_domain_access import _approve
        _approve(study)
        report = S4.build(study["ctx"], {}, {"findings": []},
                          {"verdict": "sufficient", "headline": None, "claims": []},
                          study["rd"])
        assert report["analysis_plan"]["passed"] is True
        assert "handoff" not in report["analysis_plan"]
        assert not (study["out"] / AH.RECORD_FILE).exists()

    def test_the_study_record_names_the_handoffs_file(self):
        from agents import study_record as SR
        role = SR.file_role(f"04_analysis/{AH.RECORD_FILE}")
        assert "no role on record" not in role
        assert "shown on the design page before the yes" in role
        assert "blocked" in role and "writes no other plan" in role
