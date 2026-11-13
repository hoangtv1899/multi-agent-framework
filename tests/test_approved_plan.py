"""The yes is of the analysis plan too (stage 4 of the single study plan).

    pytest tests/test_approved_plan.py -v

With the analysis preview and the handoff both on, a page that shows a plan
that passed says a yes approves it together with the runs and the years. These
tests hold that to what it says:

    the record      the approval names the plan by its hash, apart from the
                    design's own fingerprint; with the preview alone it names
                    no plan and the page says the yes approves none.
    the handoff     the plan handed to the Analyzer is the plan the yes names;
                    a yes that names no plan, or another, blocks it.
    before a job    a design whose approved plan changed or is gone is not
                    run at all.
    older records   a study approved before the yes named plans is handed its
                    plan as it was promised then.
    no plan, no     a design whose plan did not pass, or could not be written,
    yes             has no yes, at the terminal or unattended; with the
                    preview alone it runs as before.
    the status      each analysis is supported, conditional or blocked, by
                    code, on the page and in the details.

Every design goes through the coordinator's own path on the synthetic suite,
as in tests/test_analysis_handoff.py. Every model call is a scripted fake and
every number is made up.
"""
import json
import re

import pytest

from test_analysis_handoff import (_after_the_runs, _approved, _no_call, _shown, both,  # noqa
                                   _sha)
from test_analysis_preview import (BAD_PLAN, WARM, WARM_PLAN, Fails, _clients, _co, _details,
                                   _first_part, _reception, _record, _strategy)
from test_domain_access import PLAN_REPLY, REGION, built, study  # noqa: F401
from test_front_door_routes import _decision, _typed

from agents import analysis_handoff as AH
from agents import analysis_preview as AV
from agents import design_review as dr
from agents.analysis import analysis_plan as AP
from agents.analysis import step3_interpret as S3


def _approval(rd):
    return json.loads((rd / dr.APPROVAL_FILE).read_text())


# ── the record of the yes ────────────────────────────────────────────────────
class TestTheYesNamesThePlan:

    def test_the_approval_names_the_plan_and_the_page_says_so(self, both, tmp_path,
                                                              monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch, plan=_strategy(runs=[WARM]))
        shown = _record(rd)
        got = _approval(rd)
        assert got["design_sha256"] == dr.design_fingerprint(plan)
        assert got["analysis_plan_sha256"] == shown["plan_sha256"] == _sha(rd / shown["plan_file"])
        assert got["analysis_plan_file"] == shown["plan_file"]
        assert got["covers"] == "the control, the runs, the years and the analysis plan shown"
        # the plan is no part of the design's fingerprint: the two are kept apart
        assert got["analysis_plan_sha256"] != got["design_sha256"]
        page = " ".join(_first_part(rd).split())
        assert "A yes approves this plan together with the runs and the years." in page
        assert "It approves the run, the years above and the analysis plan shown under " \
               "\"How the answer will be read\"." in page
        for gone in ("writes its own plan afterwards", "does not approve"):
            assert gone not in page, gone

    def test_a_design_with_no_new_run_is_approved_with_its_plan_too(self, both, tmp_path,
                                                                    monkeypatch):
        rd, _ = _approved(tmp_path, monkeypatch)             # the control run alone
        assert _approval(rd)["analysis_plan_sha256"] == _record(rd)["plan_sha256"]
        page = " ".join(_first_part(rd).split())
        assert "It approves the years above and the analysis plan shown under" in page
        assert "A yes approves this plan together with the years." in page

    def test_with_the_preview_alone_the_yes_names_no_plan(self, monkeypatch, study, tmp_path):
        monkeypatch.setenv(AV.SWITCH, "1")
        monkeypatch.delenv(AH.SWITCH, raising=False)
        rd, plan = _approved(tmp_path, monkeypatch)
        assert set(_approval(rd)) == {"decision", "when", "design_sha256"}
        page = " ".join(_first_part(rd).split())
        assert "It does not approve an analysis plan, since the Analyzer writes its own " \
               "plan afterwards." in page
        assert AV.NOT_BUILT in page and "A yes approves this plan" not in page

    def test_an_unattended_study_names_its_plan_too(self, both, tmp_path):
        rd = tmp_path / "design"
        rd.mkdir()
        co = _co(_strategy(), _clients(), interactive=False, ran=[])
        co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        got = _approval(rd)
        assert got["analysis_plan_sha256"] == _record(rd)["plan_sha256"]
        assert "unattended" in got["decision"]

    def test_a_plan_shown_for_another_design_is_not_named(self, both, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        other = _strategy(years=(2015,))
        dr.approve(rd, other, "accepted at the terminal")
        got = _approval(rd)
        assert got["design_sha256"] == dr.design_fingerprint(other)
        assert "analysis_plan_sha256" not in got and "covers" not in got


# ── the handoff is of the plan the yes names ─────────────────────────────────
class TestThePlanHandedOverIsThePlanApproved:

    def test_it_is_handed_over_and_the_record_says_it_was_approved(self, both, built,
                                                                   tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        ctx, out = _after_the_runs(rd, plan, built)
        rec = AH.hand_over(ctx, out)
        assert rec["used"] is True and rec["approved"] is True
        assert AP.plan_sha256(out) == _approval(rd)["analysis_plan_sha256"]
        said = " ".join(AH.report_lines(rec))
        assert "approved with the yes" in said and "names it by its hash" in said
        assert S3.plan_and_review(ctx, out, clients=_no_call())[1] is True

    @pytest.mark.parametrize("damage", ["no plan named", "another plan named"])
    def test_a_yes_that_does_not_name_the_plan_blocks_it(self, both, built, tmp_path,
                                                         monkeypatch, damage):
        rd, plan = _approved(tmp_path, monkeypatch)
        got = _approval(rd)
        if damage == "no plan named":
            got.pop("analysis_plan_sha256")
        else:
            got["analysis_plan_sha256"] = "0" * 64
        (rd / dr.APPROVAL_FILE).write_text(json.dumps(got))
        ctx, out = _after_the_runs(rd, plan, built)
        rec = AH.hand_over(ctx, out)
        assert rec["blocked"] is True and rec["why"] == AH.NOT_NAMED
        assert not (out / AP.PLAN_FILE).exists() and AP.plan_passed(out)[0] is False

    def test_a_record_made_before_the_yes_named_plans_is_handed_over_as_promised(
            self, both, built, tmp_path, monkeypatch):
        """STAGE 3'S RECORDS STILL WORK. A page of that stage promised its plan
        and said the yes did not approve it; its approval names no plan."""
        rd, plan = _approved(tmp_path, monkeypatch)
        shown = _record(rd)
        shown["handoff"] = {"promised": True, "words": AH.CARRIED_OUT_NO_RUN}
        (rd / AV.RECORD_FILE).write_text(json.dumps(shown))
        got = {k: v for k, v in _approval(rd).items()
               if k in ("decision", "when", "design_sha256")}
        (rd / dr.APPROVAL_FILE).write_text(json.dumps(got))
        assert AH.promised(shown) and not AH.bound(shown)
        assert AH.page_words(shown) == AH.CARRIED_OUT_NO_RUN
        assert dr.approval_problem(rd, plan) is None
        ctx, out = _after_the_runs(rd, plan, built)
        rec = AH.hand_over(ctx, out)
        assert rec["used"] is True and rec["approved"] is False
        assert "was not tied to this plan" in " ".join(AH.report_lines(rec))


# ── before any job ───────────────────────────────────────────────────────────
class TestAChangedPlanIsNotRun:

    def test_the_approved_plan_as_it_was_runs(self, both, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        assert dr.approved_plan_problem(rd) is None
        assert dr.approval_problem(rd, plan) is None
        assert dr.approval_problem(rd, plan, resume=True) is None

    def test_a_plan_changed_after_the_yes_stops_the_study(self, both, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        p = _shown(rd) / AP.PLAN_FILE
        p.write_text(p.read_text().replace("GPP", "NEE", 1))
        why = dr.approval_problem(rd, plan)
        assert "is not the plan that was approved (its file changed)" in why
        assert why.endswith("is not run")
        assert dr.approval_problem(rd, plan, resume=True) == why

    def test_a_plan_that_is_gone_stops_the_study(self, both, tmp_path, monkeypatch):
        rd, plan = _approved(tmp_path, monkeypatch)
        (_shown(rd) / AP.PLAN_FILE).unlink()
        assert "the analysis plan that was approved is no longer on disk" in \
            dr.approval_problem(rd, plan)

    def test_the_coordinator_hands_no_such_study_to_the_manager(self, both, tmp_path,
                                                                monkeypatch):
        import workflow as wf
        rd, plan = _approved(tmp_path, monkeypatch)
        p = _shown(rd) / AP.PLAN_FILE
        p.write_text(p.read_text().replace("GPP", "NEE", 1))
        co = object.__new__(wf.WorkflowCoordinator)
        co.model, co.mcp_clients, co.conversation_context = "elm", {}, {}
        monkeypatch.setattr(wf, "_manager_for",
                            lambda m: pytest.fail("a manager was asked for"))
        with pytest.raises(dr.DesignNotApproved, match="is not the plan that was approved"):
            co._execute(plan=plan, output_dir=str(tmp_path), run_dir=rd,
                        reception=_reception())

    def test_an_approval_that_names_no_plan_is_held_to_its_design_alone(self, monkeypatch,
                                                                         study, tmp_path):
        monkeypatch.setenv(AV.SWITCH, "1")
        monkeypatch.delenv(AH.SWITCH, raising=False)
        rd, plan = _approved(tmp_path, monkeypatch)
        p = _shown(rd) / AP.PLAN_FILE
        p.write_text(p.read_text().replace("GPP", "NEE", 1))     # a preview nobody approved
        assert dr.approved_plan_problem(rd) is None
        assert dr.approval_problem(rd, plan) is None


# ── a design with no plan to approve has no yes ──────────────────────────────
class TestNoPlanNoYes:

    def _design(self, tmp_path, monkeypatch, clients, answers, interactive=True):
        rd = tmp_path / "design"
        rd.mkdir()
        co = _co(_strategy(), clients, interactive=interactive, ran=None)
        asked = _typed(monkeypatch, answers) if interactive else []
        said = co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        return rd, said, asked

    def test_a_plan_that_did_not_pass_leaves_the_design_unapprovable(self, both, tmp_path,
                                                                     monkeypatch):
        rd, said, asked = self._design(tmp_path, monkeypatch,
                                       _clients(BAD_PLAN, rewrite=BAD_PLAN), ["yes"])
        why = AH.unapprovable(_record(rd))
        assert why == ("its analysis plan did not pass its check and its review, so there "
                       "is no analysis plan a yes could approve")
        # the prompt of a design that cannot run, and a yes typed at it runs nothing
        assert asked == ["This design cannot run as it is. Type a change, or press Enter "
                         "to stop: "]
        assert not (rd / dr.APPROVAL_FILE).exists()
        assert "not run: no analysis plan a yes could approve" in _decision(rd)
        assert "this design cannot be approved" in said and why in said
        page = " ".join(_first_part(rd).split())
        assert f"This design cannot be approved as it stands, since {why}." in page
        assert "It did not pass its check and its review. A design is approved together " \
               "with its analysis plan" in page
        assert "A yes submits" not in page and "A yes starts" not in page
        assert "blocked" in page                          # the analysis's own status
        assert "cannot be approved as it is" in " ".join(_details(rd).split())

    def test_a_plan_that_could_not_be_written_does_too(self, both, tmp_path, monkeypatch):
        rd, said, _ = self._design(tmp_path, monkeypatch, {AP.PLAN_LABEL: Fails()}, [""])
        rec = _record(rd)
        assert rec["status"] == AV.FAILED and rec["handoff"]["promised"] is False
        assert AH.unapprovable(rec) == ("its analysis plan could not be written, so there "
                                        "is no analysis plan a yes could approve")
        assert not (rd / dr.APPROVAL_FILE).exists()
        page = " ".join(_first_part(rd).split())
        assert "The analysis plan could not be written." in page
        assert "this design cannot be approved as it is" in page

    def test_unattended_it_is_not_run_either(self, both, tmp_path, monkeypatch):
        rd, said, _ = self._design(tmp_path, monkeypatch,
                                   _clients(BAD_PLAN, rewrite=BAD_PLAN), [],
                                   interactive=False)
        assert not (rd / dr.APPROVAL_FILE).exists()
        assert "not run: no analysis plan a yes could approve" in _decision(rd)
        assert "Nothing was submitted: this design cannot be approved" in said

    def test_a_change_typed_at_the_page_plans_the_design_again(self, both, tmp_path,
                                                               monkeypatch):
        rd = tmp_path / "design"
        rd.mkdir()
        co = _co(_strategy(), _clients(BAD_PLAN, rewrite=BAD_PLAN), ran=None)
        again = []
        co.process_request = lambda text, **kw: again.append(text) or "revised"
        _typed(monkeypatch, ["look at 2015 only"])
        assert co._workflow_design_and_run(_reception(), str(tmp_path), rd) == "revised"
        assert again and "look at 2015 only" in again[0]
        assert not (rd / dr.APPROVAL_FILE).exists()

    def test_with_the_preview_alone_such_a_design_runs_as_before(self, monkeypatch, study,
                                                                 tmp_path):
        """LEGACY: nothing is promised, so nothing is needed for a yes."""
        monkeypatch.setenv(AV.SWITCH, "1")
        monkeypatch.delenv(AH.SWITCH, raising=False)
        rd, plan = _approved(tmp_path, monkeypatch,
                             clients=_clients(BAD_PLAN, rewrite=BAD_PLAN))
        assert _record(rd)["status"] == AV.NOT_PASSED and "handoff" not in _record(rd)
        assert AH.unapprovable(_record(rd)) is None
        assert (rd / dr.APPROVAL_FILE).is_file()
        assert "This does not stop the design" in " ".join(_first_part(rd).split())

    def test_a_study_with_no_analysis_plan_at_all_is_not_held_to_one(self):
        assert AH.unapprovable(None) is None
        assert AH.unapprovable({"status": AV.NOT_PLANNED, "why": "no results store"}) is None
        assert AH.unapprovable({"status": AV.NOT_PLANNED,
                                "handoff": {"promised": False}}) is None
        assert AH.unapprovable({"status": AV.PLANNED,
                                "handoff": {"promised": True, "bound": True}}) is None
        assert AH.unapprovable({"status": AV.NOT_PASSED}) is None      # the switch was off


# ── the status of each analysis ──────────────────────────────────────────────
def _analysis(aid, outputs):
    return {"id": aid, "question": "a question", "outputs": outputs}


SERIES_DAY = {"id": "O1", "kind": "series", "field": "GPP", "region": REGION, "step": "day"}
SERIES_YEAR = dict(SERIES_DAY, step="year")
A_MAP = {"id": "O2", "kind": "map", "field": "GPP", "period": "2016"}
CELL_MAP = {"id": "O3", "kind": "anomaly", "field": "GPP", "period": "2016"}
A_NUMBER = dict(CELL_MAP, region=REGION)


class TestTheStatusOfEachAnalysis:

    def test_supported_conditional_and_blocked(self):
        plan = {"analyses": [_analysis("P1", [A_NUMBER, A_MAP]),
                             _analysis("P2", [A_MAP, SERIES_DAY, CELL_MAP]),
                             _analysis("P3", [SERIES_YEAR]),
                             _analysis("P4", [A_NUMBER]),
                             _analysis("P10", [A_NUMBER]),
                             _analysis("P5", [])],
                "problems": ["P4: no catalog field 'NOPE'", "P4.O1: outside its scope",
                             "P10: a year outside the study", "the plan names no screen"]}
        got = AV.analysis_support(plan)
        assert list(got) == ["P1", "P2", "P3", "P4", "P10", "P5"]
        assert got["P1"] == {"status": AV.SUPPORTED, "why": []}
        assert got["P2"] == {"status": AV.CONDITIONAL, "why": [AV.NO_NUMBER]}
        assert got["P3"]["status"] == AV.SUPPORTED          # a yearly series can be stated
        assert got["P4"] == {"status": AV.BLOCKED,
                             "why": ["P4: no catalog field 'NOPE'",
                                     "P4.O1: outside its scope"]}
        # P1 is not P10: a problem belongs to the analysis whose id it opens with
        assert got["P10"]["why"] == ["P10: a year outside the study"]
        assert got["P5"]["status"] == AV.SUPPORTED          # no output is no figure
        lines = AV.support_lines(plan)
        assert lines[0] == f"- P2 is conditional: {AV.NO_NUMBER}."
        assert lines[1].startswith("- P4 is blocked: P4: no catalog field 'NOPE'; ")
        assert len(lines) == 3 and AV.support_lines({"analyses": [_analysis("P1", [A_NUMBER])]}) == []

    def test_the_page_and_the_details_give_each_analysis_its_status(self, both, tmp_path,
                                                                    monkeypatch):
        figures = dict(PLAN_REPLY, analyses=[
            PLAN_REPLY["analyses"][0],
            {"id": "P2", "question": "where it happened", "fields": ["GPP"], "regions": [],
             "periods": ["2016"], "methods": [], "settings": {}, "why": "to draw it",
             "outputs": [{"id": "O1", "kind": "map", "field": "GPP", "period": "2016"}]}])
        rd, _ = _approved(tmp_path, monkeypatch, clients=_clients(figures))
        assert _record(rd)["status"] == AV.PLANNED
        page = _first_part(rd)
        assert "| | What it looks at | Where | When | Status |" in page
        rows = [ln for ln in page.splitlines() if re.match(r"\| P\d", ln)]
        assert rows[0].endswith("| supported |") and rows[1].endswith("| conditional |")
        assert f"- P2 is conditional: {AV.NO_NUMBER}." in page
        details = _details(rd)
        assert "| Required outputs | Status |" in details
        assert f"- P2 is conditional: {AV.NO_NUMBER}." in details

    def test_a_page_whose_analyses_are_all_supported_adds_one_word_to_each_row(
            self, both, tmp_path, monkeypatch):
        rd, _ = _approved(tmp_path, monkeypatch)
        page = _first_part(rd)
        rows = [ln for ln in page.splitlines() if re.match(r"\| P\d", ln)]
        assert rows and all(r.endswith("| supported |") for r in rows)
        assert " is conditional" not in page and " is blocked" not in page
