"""The analysis plan shown on the design page before the yes (2026-11-12).

    pytest tests/test_analysis_preview.py -v

TRANSITIONAL. With its switch set, the Analyzer's own planning step writes
its plan from a design before the page, and the page shows it. These tests
drive the coordinator's own path (_workflow_design_and_run), as a person at
the terminal would, and hold the stage to what it is:

  - off unless the switch is set, and then nothing changes;
  - a design of the control alone, and one with a new run;
  - a decline, a yes, a change typed at the page, an unattended run;
  - a planning call that fails, a plan that does not pass, a control with no
    results store, a design that cannot be approved;
  - a resume, and the Analyzer's own plan after the runs;
  - the yes is still of the control, the runs and the years, and the page
    says so wherever it shows a plan.

On the synthetic control and results store of tests/test_domain_access.py.
Every model call is a scripted fake and every name and number is made up.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from test_domain_access import MARK, PLAN_REPLY, REGION, Ctx, Fake, built, study  # noqa: F401
from test_front_door_routes import _Ran, _coordinator, _decision, _typed

from agents import analysis_preview as AV
from agents import design_review as dr
from agents.analysis import analysis_plan as AP
from agents.analysis import step3_interpret as S3

QUESTION = "What happened to GPP in 2016?"
FOLLOWS = {"follows": True, "why": "it asks what was asked",
           "problems": ["a remark of the review"]}
WARM = {"name": "warm", "changes": [{"id": "weather_scale",
                                     "params": {"temperature_K": 2.0, "months": [6, 7, 8]}}]}
WARM_PLAN = dict(
    PLAN_REPLY,
    comparisons=[{"id": "C1", "what": "warm minus the control", "run": "warm",
                  "against": "control", "years": [2016]}],
    analyses=[{"id": "P1", "question": "GPP in the warm run", "fields": ["GPP"],
               "regions": [REGION], "periods": ["2016"], "methods": ["difference"],
               "settings": {}, "why": "what was asked",
               "outputs": [{"id": "O1", "kind": "difference", "run": "warm", "field": "GPP",
                            "period": "2016", "region": REGION}]}])
BAD_PLAN = dict(PLAN_REPLY, analyses=[dict(PLAN_REPLY["analyses"][0], fields=["NOPE"])])
ASK = "To run this design, type yes. Type no to stop, or type a change: "


def _strategy(runs=(), years=(2015, 2016), verdict="full"):
    return {"archetype": "whole_domain", "goals": ["GPP in 2016"],
            "sampling": {"approach": "whole_domain", "n_columns": 1},
            "feasibility": {"verdict": verdict, "why": "the control holds the field"},
            "domain_experiment": {
                "control": {"name": "synth", "reuse": True}, "years": list(years),
                "runs": list(runs),
                "analysis": {"normal_years": [2015], "question_type": "exploratory",
                             "regions": [{"name": REGION}], "variables": ["GPP"]},
                "caveats": ["a made-up caveat"]}}


def _reception():
    return {"user_request": QUESTION,
            "brief": {"intent": "design", "design_archetype": "whole_domain", "model": "elm",
                      "domain": {"name": "synth"},
                      "run_settings": {"requested_period": {"yr_start": 2016, "yr_end": 2016},
                                       "resolved_period": {"yr_start": 2016, "yr_end": 2016,
                                                           "source": "user"}}}}


class Never:
    """A model that must not be asked."""
    def ask(self, messages, **kw):
        pytest.fail("the planning step was called when it should not have been")


class Fails:
    def ask(self, messages, **kw):
        raise RuntimeError("the gateway did not answer")


class ByDesign:
    """A scripted planner: the warm plan for a design with the warm run, the
    control's plan for one without."""
    def __init__(self):
        self.prompts = []

    def ask(self, messages, **kw):
        self.prompts.append(messages[0]["content"])
        return json.dumps(WARM_PLAN if "| warm |" in self.prompts[-1] else PLAN_REPLY)


def _clients(plan=PLAN_REPLY, rewrite=None):
    c = {AP.PLAN_LABEL: plan if hasattr(plan, "ask") else Fake(plan),
         AP.REVIEW_LABEL: Fake(FOLLOWS)}
    if rewrite is not None:
        c[AP.REWRITE_LABEL] = Fake(rewrite)
        c[AP.REWRITE_REVIEW_LABEL] = Fake(FOLLOWS)
    return c


def _co(plan, clients, interactive=True, ran=None):
    co = _coordinator(plan, interactive)
    co._analysis_preview_clients = clients
    if ran is None:
        co._execute = lambda **kw: pytest.fail("nothing was approved, so nothing runs")
    else:
        co._execute = lambda **kw: ran.append(kw) or (_ for _ in ()).throw(_Ran("stop here"))
    return co


def _first_part(run_dir):
    """The short page a person decides on, which the terminal shows whole."""
    return dr.decision_part((Path(run_dir) / dr.REVIEW_NAME).read_text())[0]


def _details(run_dir):
    """The design in full: the file the short page names for the rest."""
    return (Path(run_dir) / "DESIGN_DETAILS.md").read_text()


def _record(run_dir):
    return json.loads((Path(run_dir) / AV.RECORD_FILE).read_text())


@pytest.fixture
def on(monkeypatch, study):
    """The switch set, on the synthetic control of the `study` fixture."""
    monkeypatch.setenv(AV.SWITCH, "1")


# ── off unless the switch is set ─────────────────────────────────────────────
class TestOffUnlessTheSwitchIsSet:

    @pytest.mark.parametrize("value", [None, "", "0", "no", "off"])
    def test_nothing_is_planned_and_the_page_is_as_it_was(self, study, tmp_path,
                                                          monkeypatch, value):
        monkeypatch.delenv(AV.SWITCH, raising=False)
        if value is not None:
            monkeypatch.setenv(AV.SWITCH, value)
        rd = tmp_path / "design"
        rd.mkdir()
        co = _co(_strategy(), {AP.PLAN_LABEL: Never(), AP.REVIEW_LABEL: Never()})
        asked = _typed(monkeypatch, ["no"])
        out = co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        assert out.startswith("🛑 Design declined") and asked == [ASK]
        assert not (rd / AV.RECORD_FILE).exists() and not (rd / AV.FOLDER).exists()
        assert AV.HEADING not in (rd / dr.REVIEW_NAME).read_text()

    @pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
    def test_the_switch_is_a_plain_word(self, monkeypatch, value):
        monkeypatch.setenv(AV.SWITCH, value)
        assert AV.enabled() is True


# ── a design of the control alone ────────────────────────────────────────────
class TestTheControlAlone:

    def test_declined_the_plan_was_shown_and_nothing_else_happened(self, on, tmp_path,
                                                                   monkeypatch):
        rd = tmp_path / "design"
        rd.mkdir()
        (rd / "04_analysis").mkdir()
        (rd / "04_analysis" / "DOMAIN_REPORT.md").write_text(f"an outcome: {MARK}\n")
        clients = _clients()
        co = _co(_strategy(), clients)
        asked = _typed(monkeypatch, ["no"])
        out = co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        assert out.startswith("🛑 Design declined") and asked == [ASK]
        assert "DECLINED at the terminal" in _decision(rd)
        rec = _record(rd)
        assert rec["status"] == AV.PLANNED and rec["attempts"] == 1
        assert rec["design_sha256"] == dr.design_fingerprint(_strategy())
        # shown, short, on the page a person reads before the prompt
        page = _first_part(rd)
        flat = " ".join(page.split())
        assert "## How the answer will be read" in page and AV.NOT_BUILT in flat
        assert "The plan sets 2016 of the prepared run against the normal years, 2015." in flat
        assert "The plan has 1 analysis and reads 1 quantity." in flat
        assert f"| P1 | GPP in 2016 | {REGION} | the year 2016 |" in page
        assert "- The analysis plan passed its check and its review." in page
        assert "- The plan's review made 1 remark." in page
        assert "  - a remark of the review" in page
        assert page.index("## How the answer will be read") < page.index("## Your decision")
        # and in full in the details, with the words that say the yes does not cover it
        full = _details(rd)
        assert AV.HEADING in full and AV.NOT_BOUND in " ".join(full.split())
        assert "- Code's check and the model review: **passed**." in full
        assert "- Compared: C1: the prepared run against its normal years (2015), 2016." in full
        assert "- Read: 1 analysis, 1 required output, 1 quantity (GPP)." in full
        assert f"| P1 | GPP in 2016 | {REGION} | the year 2016 | 1 number |" in full
        assert "  - a remark of the review" in full and "  - soil water" in full
        # the plan is in the preview's folder, and nowhere the Analyzer writes
        plan_file = rd / rec["plan_file"]
        assert plan_file.is_file() and plan_file.parent.name == "04_analysis"
        assert str(Path(rec["plan_file"]).parent.parent) == f"{AV.FOLDER}/{rd.name}"
        assert not (rd / "04_analysis" / AP.PLAN_FILE).exists()
        assert not (rd / dr.APPROVAL_FILE).exists()
        # no record of the Experiment Manager was written or touched
        assert not (rd / "01_inputs" / "domain_experiment.json").exists()
        # the planning step was given the question and no outcome
        prompt = clients[AP.PLAN_LABEL].prompts[0]
        assert QUESTION in prompt and str(MARK) not in prompt
        assert f"study {rd.name}" in prompt            # the catalog names this study

    def test_unattended_it_is_planned_for_and_the_study_goes_on(self, on, tmp_path):
        rd = tmp_path / "design"
        rd.mkdir()
        ran = []
        co = _co(_strategy(), _clients(), interactive=False, ran=ran)
        co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        assert len(ran) == 1 and _record(rd)["status"] == AV.PLANNED
        assert "The plan has 1 analysis" in (rd / dr.REVIEW_NAME).read_text()
        assert AV.HEADING in _details(rd)
        assert "auto-continued (unattended run)" in _decision(rd)


# ── a design with a new run ──────────────────────────────────────────────────
class TestANewRun:

    def test_a_yes_runs_it_and_the_yes_is_still_of_the_runs_and_years(self, on, tmp_path,
                                                                      monkeypatch):
        rd = tmp_path / "design"
        rd.mkdir()
        plan, ran, clients = _strategy(runs=[WARM]), [], _clients(WARM_PLAN)
        co = _co(plan, clients, ran=ran)
        _typed(monkeypatch, ["yes"])
        co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        assert len(ran) == 1 and ran[0]["plan"] == plan
        # THE YES IS NOT BOUND TO THE ANALYSIS: the approval is what it was
        rec = json.loads((rd / dr.APPROVAL_FILE).read_text())
        assert set(rec) == {"decision", "when", "design_sha256"}
        assert rec["design_sha256"] == dr.design_fingerprint(plan)
        amended = copy.deepcopy(plan)
        amended["domain_experiment"]["analysis"]["variables"] = ["ANOTHER"]
        assert dr.approval_problem(rd, amended) is None
        # and the page said so, where it showed the plan and where it asked
        page = _first_part(rd)
        flat = " ".join(page.split())
        assert AV.NOT_BUILT in flat
        assert "It does not approve an analysis plan, since the Analyzer writes its own " \
               "plan afterwards." in flat
        assert "The plan compares `warm` with the prepared run for 2016." in flat
        assert f"| P1 | GPP in the warm run | {REGION} | the year 2016 |" in page
        assert "- `warm` is not made yet, so the plan takes its outputs to be the prepared " \
               "run's quantities over the planned years." in flat
        assert AV.NOT_BOUND in " ".join(_details(rd).split())
        assert "- Compared: C1: `warm` against the prepared run, 2016." in _details(rd)

    def test_the_planning_step_reads_the_run_as_planned_and_not_made(self, on, tmp_path,
                                                                     monkeypatch):
        rd = tmp_path / "design"
        rd.mkdir()
        clients = _clients(WARM_PLAN)
        co = _co(_strategy(runs=[WARM]), clients)
        _typed(monkeypatch, ["no"])
        co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        prompt = clients[AP.PLAN_LABEL].prompts[0]
        row = next(ln for ln in prompt.splitlines() if ln.startswith("| warm |"))
        # the server's own words for the change, the planned years, no store
        assert "the difference from the control is the response to air 2 K warmer" in row
        assert "planned 2015-2016" in row and "passed" not in row
        assert "suite not built" in row
        assert '"run": "warm"' in prompt and '"reused": false' in prompt
        assert _record(rd)["status"] == AV.PLANNED


# ── planning that fails ──────────────────────────────────────────────────────
class TestPlanningThatFails:

    def test_a_call_that_fails_is_said_and_stops_nothing(self, on, tmp_path, monkeypatch):
        rd = tmp_path / "design"
        rd.mkdir()
        ran = []
        co = _co(_strategy(), _clients(Fails()), ran=ran)
        asked = _typed(monkeypatch, ["yes"])
        co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        rec = _record(rd)
        assert rec["status"] == AV.FAILED
        assert rec["why"] == "RuntimeError: the gateway did not answer"
        page = " ".join(_first_part(rd).split())
        assert ("The plan could not be written. The error was RuntimeError: the gateway did "
                "not answer. This does not stop the design") in page
        assert AV.NOT_BUILT in page
        assert "The plan could not be written: RuntimeError: the gateway did not answer." \
            in " ".join(_details(rd).split())
        assert asked == [ASK] and len(ran) == 1          # the design is offered and runs

    def test_a_plan_that_fails_its_check_twice_is_shown_as_not_passed(self, on, tmp_path,
                                                                      monkeypatch):
        rd = tmp_path / "design"
        rd.mkdir()
        ran = []
        co = _co(_strategy(), _clients(BAD_PLAN, rewrite=BAD_PLAN), ran=ran)
        asked = _typed(monkeypatch, ["yes"])
        co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        rec = _record(rd)
        assert rec["status"] == AV.NOT_PASSED and rec["attempts"] == 2
        assert "did not pass step 3's review" in rec["why"]
        page = _first_part(rd)
        assert "- The analysis plan did not pass its check or its review. This does not stop " \
               "the design" in " ".join(page.split())
        assert "  - P1: no catalog field 'NOPE'" in page       # each problem, whole
        full = _details(rd)
        assert "- Code's check and the model review: **not passed**." in full
        assert "- What code's check found (" in full and "NOPE" in full
        assert asked == [ASK] and len(ran) == 1

    def test_a_plan_rewritten_once_is_shown_as_passed_on_its_second_writing(self, on, tmp_path,
                                                                            monkeypatch):
        rd = tmp_path / "design"
        rd.mkdir()
        co = _co(_strategy(), _clients(BAD_PLAN, rewrite=PLAN_REPLY))
        _typed(monkeypatch, ["no"])
        co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        rec = _record(rd)
        assert rec["status"] == AV.PLANNED and rec["attempts"] == 2
        assert "- The analysis plan passed its check and its review on its second writing." \
            in _first_part(rd)
        assert "**passed** on its second writing." in _details(rd)
        # the first plan and its review are kept, as after a run
        adir = (rd / rec["plan_file"]).parent
        assert (adir / "analysis_plan.v1.json").is_file()

    def test_a_control_with_no_results_store_is_not_planned_for(self, on, built, tmp_path,
                                                                monkeypatch):
        entry = dict(built["control"])                    # no suite_dir
        monkeypatch.setattr(AP, "_registry_entry", lambda name: entry)
        rd = tmp_path / "design"
        rd.mkdir()
        co = _co(_strategy(), {AP.PLAN_LABEL: Never(), AP.REVIEW_LABEL: Never()})
        _typed(monkeypatch, ["no"])
        co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        assert _record(rd)["status"] == AV.NOT_PLANNED
        assert "No plan was written, since the prepared run has no results store" in \
            " ".join(_first_part(rd).split())
        assert "No plan was written: the prepared run has no results store" in \
            " ".join(_details(rd).split())

    def test_a_design_that_cannot_be_approved_is_not_planned_for(self, on, tmp_path,
                                                                 monkeypatch):
        rd = tmp_path / "design"
        rd.mkdir()
        co = _co(_strategy(verdict="infeasible"),
                 {AP.PLAN_LABEL: Never(), AP.REVIEW_LABEL: Never()})
        asked = _typed(monkeypatch, [""])
        out = co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        assert "judged this design infeasible" in out
        assert asked == ["This design cannot run as it is. Type a change, or press Enter "
                         "to stop: "]
        assert not (rd / AV.RECORD_FILE).exists()
        assert AV.HEADING not in _details(rd)
        assert "no analysis plan is on this page" in (rd / dr.REVIEW_NAME).read_text()

    def test_whatever_goes_wrong_the_request_goes_on(self, on, tmp_path, monkeypatch):
        """Even an error outside the planning call: the page is written and
        the design is offered."""
        monkeypatch.setattr(AV, "plan_before_approval",
                            lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
        rd = tmp_path / "design"
        rd.mkdir()
        co = _co(_strategy(), _clients())
        asked = _typed(monkeypatch, ["no"])
        co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        assert asked == [ASK]
        assert "The plan could not be written. The error was ValueError: boom." in \
            " ".join(_first_part(rd).split())


# ── a change typed at the page ───────────────────────────────────────────────
class TestARevision:

    def test_each_design_gets_its_own_plan_in_its_own_folder(self, on, tmp_path,
                                                             monkeypatch):
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir(), second.mkdir()
        plans = {str(first): _strategy(), str(second): _strategy(runs=[WARM])}
        planner = ByDesign()
        ran = []
        co = _co(None, _clients(planner), ran=ran)
        co._plan = lambda result, run_dir, findings=None, previous=None: plans[str(run_dir)]
        co.process_request = lambda request, output_dir=None, revision=None: \
            co._workflow_design_and_run(_reception(), str(tmp_path), second)
        _typed(monkeypatch, ["add a run two degrees warmer in summer", "yes"])
        co._workflow_design_and_run(_reception(), str(tmp_path), first)
        assert len(planner.prompts) == 2                 # one planning for each design
        a, b = _record(first), _record(second)
        assert a["design_sha256"] == dr.design_fingerprint(plans[str(first)])
        assert b["design_sha256"] == dr.design_fingerprint(plans[str(second)])
        assert a["design_sha256"] != b["design_sha256"]
        assert a["plan_file"].startswith(f"{AV.FOLDER}/a/")
        assert b["plan_file"].startswith(f"{AV.FOLDER}/b/")
        assert "of the prepared run against the normal years" in \
            " ".join(_first_part(first).split())
        assert "The plan compares `warm` with the prepared run" in \
            " ".join(_first_part(second).split())
        assert "`warm`" not in _first_part(first) and "`warm`" not in _details(first)
        # the first design was replaced, never approved; the second ran
        assert "revision requested at the terminal" in _decision(first)
        assert not (first / dr.APPROVAL_FILE).exists()
        assert [str(k["run_dir"]) for k in ran] == [str(second)]


# ── a resume, and the Analyzer after the runs ────────────────────────────────
class TestAfterTheYes:

    def _approved(self, tmp_path, monkeypatch, plan=None):
        rd = tmp_path / "design"
        rd.mkdir()
        plan = plan or _strategy(runs=[WARM])
        # the scripted plan of the design in hand. A plan of the `warm` run
        # given for a design with no run fails its check, and the rewrite then
        # asked for has no scripted client: with the gateway's key in the
        # environment it was a real, paid call (found 2026-11-12; the root
        # conftest now fails any test that reaches the gateway).
        clients = _clients(WARM_PLAN if plan["domain_experiment"]["runs"] else PLAN_REPLY)
        co = _co(plan, clients, ran=[])
        _typed(monkeypatch, ["yes"])
        co._workflow_design_and_run(_reception(), str(tmp_path), rd)
        return rd, plan, clients

    def test_a_resume_does_not_plan_again_and_needs_no_preview(self, on, tmp_path,
                                                              monkeypatch):
        import workflow as wf
        rd, plan, clients = self._approved(tmp_path, monkeypatch)
        monkeypatch.setattr(wf, "_manager_for",
                            lambda model: (_ for _ in ()).throw(_Ran("the manager")))
        co = _coordinator(None, interactive=True)
        co._analysis_preview_clients = {AP.PLAN_LABEL: Never(), AP.REVIEW_LABEL: Never()}
        before = len(clients[AP.PLAN_LABEL].prompts)
        for resume in (True, False):
            with pytest.raises(_Ran):
                co._execute(plan=plan, output_dir=str(tmp_path), run_dir=rd,
                            reception=_reception(), resume=resume)
        assert len(clients[AP.PLAN_LABEL].prompts) == before
        # a study approved before this stage, with no preview on record, resumes too
        old = tmp_path / "older"
        old.mkdir()
        dr.approve(old, plan, "accepted at the terminal")
        with pytest.raises(_Ran):
            co._execute(plan=plan, output_dir=str(tmp_path), run_dir=old,
                        reception=_reception(), resume=True)

    def test_the_analyzer_still_writes_its_own_plan_after_the_runs(self, on, built, tmp_path,
                                                                   monkeypatch):
        """THE PREVIEW IS NOT HANDED TO THE ANALYZER. After the runs it plans in
        the study's own 04_analysis, by its own call, and the preview's files
        are left as they were."""
        rd, plan, _ = self._approved(tmp_path, monkeypatch, _strategy())
        shown = rd / _record(rd)["plan_file"]
        before = hashlib.sha256(shown.read_bytes()).hexdigest()
        (rd / "01_inputs" / "domain_control_facts.json").write_text(json.dumps(
            {"control": {"name": "synth", "run_dir": built["control"]["run_dir"]}}))
        ctx = Ctx(rd, {"question": QUESTION, "goals": ["GPP in 2016"],
                       "experiment": dict(plan["domain_experiment"])})
        assert AP.suite_dirs(ctx)["control"] == str(built["suite"])
        after_the_runs = Fake(PLAN_REPLY)
        out = rd / "04_analysis"
        AP.write_plan(ctx, out, client=after_the_runs)
        assert len(after_the_runs.prompts) == 1          # it asked: it did not reuse
        assert (out / AP.PLAN_FILE).is_file() and (out / AP.PLAN_FILE) != shown
        assert hashlib.sha256(shown.read_bytes()).hexdigest() == before
        # and its gate reads its own review, not the preview's
        assert AP.plan_passed(out) == (False, "the analysis plan has not been reviewed "
                                              "(step 3's review)")

    def test_a_page_drawn_again_shows_the_plan_of_its_own_design_only(self, on, tmp_path,
                                                                      monkeypatch):
        rd, plan, _ = self._approved(tmp_path, monkeypatch)
        (rd / "strategy.json").write_text(json.dumps(plan))
        dr.write_review(rd)
        assert "The plan compares `warm` with the prepared run" in \
            " ".join((rd / dr.REVIEW_NAME).read_text().split())
        assert "`warm` against the prepared run" in _details(rd)
        other = _strategy(runs=[WARM], years=(2015,))
        (rd / "strategy.json").write_text(json.dumps(other))
        dr.write_review(rd)
        for text in ((rd / dr.REVIEW_NAME).read_text(), _details(rd)):
            assert "was written for another design, so it is not shown" in " ".join(text.split())
            assert "against the prepared run, 2016" not in text
            assert "The plan compares" not in text
        # and a page the coordinator draws with no preview shows none, whatever is on disk
        dr.write_review(rd, analysis_preview=None)
        assert AV.HEADING not in _details(rd)
        assert "no analysis plan is on this page" in (rd / dr.REVIEW_NAME).read_text()


    def test_the_study_record_names_the_previews_files_for_what_they_are(self, on, tmp_path,
                                                                        monkeypatch):
        """The study's own index lists every file with its role. The plan
        shown before the yes is named a preview there, so it is never taken
        for the plan the Analyzer carried out."""
        from agents import study_record as SR
        rd, _, _ = self._approved(tmp_path, monkeypatch)
        made = sorted(p.relative_to(rd).as_posix() for p in (rd / "01_inputs").rglob("*")
                      if p.is_file() and "analysis_preview" in p.as_posix())
        assert len(made) >= 6
        for rel in made:
            role = SR.file_role(rel)
            assert "no role on record" not in role and "preview" in role, rel
        shown = SR.file_role(f"{AV.FOLDER}/design/04_analysis/{AP.PLAN_FILE}")
        assert "the plan the Analyzer carried out is 04_analysis/analysis_plan.json" in shown
        assert SR.file_role(f"04_analysis/{AP.PLAN_FILE}").startswith(
            "the recorded analysis plan (step 2a)")


# ── the planning is the Analyzer's own ───────────────────────────────────────
class TestThePlanningIsTheAnalyzersOwn:

    def test_both_callers_make_the_same_call(self, on, tmp_path, monkeypatch):
        import inspect
        seen = []
        real = S3.plan_and_review

        def spy(ctx, out_dir, **kw):
            seen.append((type(ctx).__name__, Path(out_dir)))
            return real(ctx, out_dir, **kw)
        monkeypatch.setattr(S3, "plan_and_review", spy)
        rd = tmp_path / "design"
        rd.mkdir()
        AV.plan_before_approval(_reception(), _strategy(), rd, clients=_clients())
        assert seen == [("PreRunContext", rd / AV.FOLDER / "design" / "04_analysis")]
        assert "plan_and_review(ctx, out_dir" in inspect.getsource(S3.investigate_and_interpret)

    def test_the_context_is_made_from_the_design_and_holds_no_result(self, on, tmp_path):
        rd = tmp_path / "design"
        rd.mkdir()
        ctx = AV.context(_reception(), _strategy(runs=[WARM]), rd)
        exp = ctx.plan["experiment"]
        assert [r["run"] for r in exp["runs"]] == ["control", "warm"]
        assert exp["runs"][0] == {"run": "control", "reused": True, "changes": []}
        change = exp["runs"][1]["changes"][0]
        assert change["id"] == "weather_scale" and "2 K warmer" in change["meaning"]
        assert change["params"]["precip_factor"] == 1.0      # the server's own default
        assert "not_meaning" in change
        assert exp["years"] == [2015, 2016] and ctx.plan["question"] == QUESTION
        for table in (ctx.regional, ctx.regional_monthly, ctx.regional_seasonal):
            with pytest.raises(RuntimeError, match="no result exists before a run"):
                table()
        assert AP.study_runs(ctx) == {"control": [2015, 2016], "warm": [2015, 2016]}
        assert Path(ctx.run_dir) == rd / AV.FOLDER / "design"
        assert sorted(p.name for p in rd.iterdir()) == ["01_inputs"]

    def test_it_never_raises(self, on, tmp_path):
        rd = tmp_path / "design"
        rd.mkdir()
        rec = AV.plan_before_approval(_reception(), {"domain_experiment": "not a design"}, rd)
        assert rec["status"] == AV.NOT_PLANNED and "not a whole-domain experiment" in rec["why"]
        bad = _strategy(runs=[{"name": "odd", "changes": [{"id": "no_such_change"}]}])
        rec = AV.plan_before_approval(_reception(), bad, rd, clients=_clients())
        assert rec["status"] == AV.FAILED and "no_such_change" in rec["why"]
        assert _record(rd) == rec


# ── the page's section, by itself ────────────────────────────────────────────
class TestThePagesSection:

    def test_no_record_no_section(self, tmp_path):
        assert AV.page_lines(None, tmp_path) == []

    @pytest.mark.parametrize("record", [
        {"status": AV.NOT_PLANNED, "why": "a reason"},
        {"status": AV.FAILED, "why": "an error"},
        {"status": AV.PLANNED, "plan_file": "nowhere.json", "review_file": "nowhere.json"},
        {"status": AV.NOT_PASSED, "plan_file": "nowhere.json", "review_file": "nowhere.json"}])
    def test_whatever_the_status_the_section_says_the_yes_does_not_cover_it(self, tmp_path,
                                                                           record):
        text = " ".join("\n".join(AV.page_lines(record, tmp_path)).split())
        assert AV.NOT_BOUND in text and "It does not approve this analysis" in text

    def test_it_never_says_the_analysis_is_approved_or_carried_out(self, on, tmp_path):
        """The words of a later stage must not appear in this one."""
        rd = tmp_path / "design"
        rd.mkdir()
        rec = AV.plan_before_approval(_reception(), _strategy(), rd, clients=_clients())
        text = " ".join("\n".join(AV.page_lines(rec, rd)).split()).lower()
        for words in ("approved plan", "approves this analysis", "will be carried out",
                      "is bound", "the analyzer carries out this plan"):
            assert words not in text, words

    def test_a_year_inside_its_own_normal_years_is_said_on_the_page(self, on, tmp_path):
        rd = tmp_path / "design"
        rd.mkdir()
        st = _strategy()
        st["domain_experiment"]["analysis"]["normal_years"] = [2015, 2016]
        reply = dict(PLAN_REPLY, baseline={"normal_years": [2015, 2016], "why": "w"})
        rec = AV.plan_before_approval(_reception(), st, rd, clients=_clients(reply))
        page = "\n".join(AV.page_lines(rec, rd))
        assert "- Normal years: 2015 to 2016; the Planner." in page
        assert "- On the normal years: the normal years include 2016, a year this plan " \
               "sets against them" in page

    def test_periods_and_outputs_in_words(self):
        assert AV.periods_words([f"2031-{m:02d}" for m in range(1, 13)] + ["2031-JJA", "2031"]) \
            == "each month of 2031, summer 2031 and the year 2031"
        assert AV.periods_words(["2031-06", "2031-07", "2031-08"]) == \
            "June to August 2031, by month"
        assert AV.periods_words(["WY2031", "normal"]) == "the water year 2031 and normal"
        # three or more years in a row are one span, where the first of them stood
        assert AV.periods_words(["2031", "2032", "2033", "WY2031", "WY2032", "WY2033"]) == \
            "the years 2031 to 2033 and the water years 2031 to 2033"
        assert AV.periods_words(["2031-JJA", "2031", "2033"]) == \
            "summer 2031, the year 2031 and the year 2033"
        assert AV.periods_words(["2031", "2031-JJA", "2032", "2033"]) == \
            "the years 2031 to 2033 and summer 2031"
        outs = [{"kind": "difference", "region": "r"}, {"kind": "anomaly", "region": "r"},
                {"kind": "difference", "region": None}, {"kind": "series"},
                {"kind": "overlap"}, {"kind": "elevation_split"}, {"kind": "elevation_split"}]
        assert AV.outputs_words(outs) == ("2 numbers, 1 map, 1 series, 1 comparison of two "
                                          "maps and 2 summaries by elevation band")
        assert AV.outputs_words([]) == "none"
