"""What the front door will not run, and what counts as a yes (2026-11-10).

    pytest tests/test_front_door_routes.py -v

A breadth test of the front half found two ways to a prompt that should not
have been there. A question about a watershed with no prepared control was
refused in words, then turned into a single-column design on "proceed anyway".
A question about a change the model does not have (pumping) ended at a design
its own Planner called infeasible, with a prompt to run it.

The rules these tests hold, each at more than one place:

  * this version runs whole-domain studies only (core/supported_routes),
    read by Reception before it gathers, by the coordinator before the page
    could offer a yes, and by the gate before compute;
  * an infeasible design has no yes, whatever kind of study it is;
  * only a typed yes approves, and an empty answer approves nothing;
  * an approval is of one design: the one on the page it was typed at.

Every place, year and name here is made up. No test reaches a language model,
a data server or a scheduler. The test session sets the developer's switch
(tests/conftest.py); each test of the rule removes it first.
"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("openai")   # reception_llm imports the tool loop

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core import supported_routes as sr                     # noqa: E402
from core import strategy_check                             # noqa: E402
from agents import design_review as dr                      # noqa: E402
from agents.reception_llm import LLMReceptionAgent          # noqa: E402


@pytest.fixture
def as_a_user(monkeypatch):
    """The front door as a user gets it: the developer's switch is not set."""
    monkeypatch.delenv(sr.SWITCH, raising=False)


# ── designs ──────────────────────────────────────────────────────────────────
def _site_brief():
    """A watershed with no prepared control, after the person said "proceed
    anyway": Reception's words are right and its archetype is single-column."""
    return {"intent": "design", "design_archetype": "site", "model": "elm",
            "domain": {"name": "Sandy Creek", "bbox": {"west": -1, "east": 0,
                                                       "south": 0, "north": 1}},
            "model_availability": "no prepared control exists for Sandy Creek",
            "run_settings": {
                "resolved_period": {"yr_start": 2031, "yr_end": 2031, "source": "user"},
                "conflicts": ["the user asked to proceed anyway with sampled columns"]}}


def _site_strategy(verdict="partial"):
    return {"archetype": "site",
            "sampling": {"approach": "elevation_bands", "n_columns": 3},
            "feasibility": {"verdict": verdict, "why": "sampled columns compute no growth"}}


def _domain_brief():
    return {"intent": "design", "design_archetype": "whole_domain", "model": "elm",
            "domain": {"name": "synthetic"},
            "run_settings": {"resolved_period": {"yr_start": 2031, "yr_end": 2032,
                                                 "source": "user"}}}


def _domain_strategy(run="warm", years=(2031, 2032), verdict="full"):
    return {"archetype": "whole_domain",
            "sampling": {"approach": "whole_domain", "n_columns": 1},
            "domain_experiment": {"control": {"name": "c", "reuse": True},
                                  "years": list(years), "runs": [{"name": run}]},
            "feasibility": {"verdict": verdict, "why": "the model has no such change"}}


def _reception(brief):
    return {"user_request": "a question", "brief": brief}


def _coordinator(plan, interactive):
    import workflow as wf
    co = object.__new__(wf.WorkflowCoordinator)
    co.conversation_context = {}
    co.interactive_reception = interactive
    co.model = "elm"
    co.mcp_clients = {}
    co._preview = lambda plan, run_dir, result, output_dir: None
    co._plan = lambda result, run_dir, findings=None, previous=None: plan
    return co


def _typed(monkeypatch, answers):
    """Answers typed at the prompt, in order; the prompts asked are returned."""
    asked, left = [], list(answers)

    def fake(prompt=""):
        asked.append(prompt)
        if not left:
            pytest.fail(f"the prompt was asked more often than expected: {asked}")
        return left.pop(0)
    monkeypatch.setattr("builtins.input", fake)
    return asked


def _decision(run_dir):
    return next(ln for ln in (Path(run_dir) / "DESIGN_REVIEW.md").read_text().splitlines()
                if ln.startswith("**Decision:**"))


# ── the rule ─────────────────────────────────────────────────────────────────
class TestTheRule:

    def test_a_whole_domain_study_is_run(self, as_a_user):
        assert sr.problem(_domain_brief()) is None
        assert sr.problem(_domain_brief(), _domain_strategy()) is None
        assert sr.problem({}, {"domain_case": {"recipe": "r"}}) is None

    @pytest.mark.parametrize("brief, strategy, kind", [
        (_site_brief(), None, "site"),
        (_site_brief(), _site_strategy(), "site"),
        ({"design_archetype": "conceptual"}, None, "conceptual"),
        ({}, {"sampling": {"approach": "factor_sweep"}}, "conceptual"),
        ({"design_archetype": "coupling"}, None, "coupling"),
        ({}, {"coupling": {"prior_experiment": "x"}}, "coupling"),
        ({}, None, "site"),
    ])
    def test_every_other_kind_is_set_aside(self, as_a_user, brief, strategy, kind):
        assert sr.kind(brief, strategy) == kind
        why = sr.problem(brief, strategy)
        assert "whole-domain studies on a prepared control only" in why
        assert "set aside in this version" in why and "?" not in why

    def test_the_switch_is_the_developers_way_in(self, monkeypatch):
        monkeypatch.setenv(sr.SWITCH, "1")
        assert sr.problem(_site_brief(), _site_strategy()) is None
        monkeypatch.setenv(sr.SWITCH, "yes please")     # only "1" is the switch
        assert sr.problem(_site_brief()) is not None


# ── the servers the front door starts ────────────────────────────────────────
CONFIGURED = ["gauges", "terrain", "other_model", "elm", "towers"]


class TestOnlyTheServersNeededAreStarted:

    def test_a_whole_domain_study_needs_its_model_server_only(self, as_a_user):
        assert sr.servers_to_start(CONFIGURED) == ["elm"]

    def test_the_switch_starts_them_all(self, monkeypatch):
        monkeypatch.setenv(sr.SWITCH, "1")
        assert sr.servers_to_start(CONFIGURED) is None

    def test_a_configuration_that_names_none_of_them_still_starts(self, as_a_user):
        assert sr.servers_to_start(["gauges", "terrain"]) is None
        assert sr.servers_to_start([]) is None

    def _manager(self, tmp_path, monkeypatch, select):
        from core import mcp_manager as MM
        started = []

        class Client:
            def __init__(self, server_name, command, args, timeout):
                started.append(server_name)
        monkeypatch.setattr(MM, "MCPClient", Client)
        cfg = tmp_path / "servers.json"
        cfg.write_text(json.dumps({"mcp_servers": {n: {"command": "x"} for n in CONFIGURED}}))
        return MM.MCPManager(str(cfg), select=select), started

    def test_the_manager_starts_only_what_the_rule_picks(self, as_a_user, tmp_path,
                                                         monkeypatch):
        m, started = self._manager(tmp_path, monkeypatch, sr.servers_to_start)
        assert started == ["elm"] and list(m.get_all_clients()) == ["elm"]
        assert m.not_started == ["gauges", "terrain", "other_model", "towers"]

    def test_with_no_rule_the_manager_starts_every_server(self, tmp_path, monkeypatch):
        m, started = self._manager(tmp_path, monkeypatch, None)
        assert started == CONFIGURED and m.not_started == []

    def test_the_coordinator_passes_the_rule(self):
        src = (ROOT / "workflow.py").read_text()
        assert "MCPManager(mcp_config_file, select=servers_to_start)" in src


# ── Reception: before a directory, before a fetch ────────────────────────────
class _Loop:
    def __init__(self, brief):
        self.brief = brief

    def run(self, system, message):
        return {"content": json.dumps(self.brief), "trace": [], "rounds": 1}


def _reception_agent(brief):
    agent = object.__new__(LLMReceptionAgent)
    agent.system, agent.loop = "", _Loop(brief)
    return agent


class _Minted(Exception):
    """Raised by the test's mint: Reception went on past the rule."""


def _mint(brief):
    raise _Minted


class TestReceptionStopsBeforeItGathers:

    @pytest.mark.parametrize("brief", [
        _site_brief(),
        {"intent": "design", "design_archetype": "conceptual", "model": "elm"},
        {"intent": "design", "design_archetype": "coupling", "model": "pflotran"},
    ])
    def test_a_kind_set_aside_mints_nothing_and_fetches_nothing(self, as_a_user, brief):
        """The live case: a watershed with no prepared control, and "proceed
        anyway". The mint is where the gather starts; it is never reached."""
        pkg = _reception_agent(brief).process("a question", run_dir=_mint)
        assert pkg["route"]["action"] == "unsupported"
        assert "set aside in this version" in pkg["route"]["why"]
        assert pkg["observations"] == {} and pkg["grid"] == {} and "run_dir" not in pkg

    def test_a_whole_domain_study_goes_on(self, as_a_user):
        with pytest.raises(_Minted):
            _reception_agent(_domain_brief()).process("a question", run_dir=_mint)

    def test_a_question_back_to_the_user_is_untouched(self, as_a_user):
        brief = {"intent": "clarification_needed", "questions": ["which year?"],
                 "design_archetype": "site"}
        pkg = _reception_agent(brief).process("a question", run_dir=_mint)
        assert pkg["route"]["action"] == "clarify"


# ── the coordinator ──────────────────────────────────────────────────────────
class TestTheCoordinatorOffersNoYes:

    def test_an_unsupported_route_ends_before_the_planner(self, as_a_user, tmp_path):
        import workflow as wf
        co = _coordinator(None, interactive=True)
        co.default_output_dir = str(tmp_path)
        co._reception_context = lambda: None
        co._plan = lambda *a, **k: pytest.fail("an unsupported request must not be planned")
        co._execute = lambda **kw: pytest.fail("an unsupported request must not run")

        class Front:
            def process(self, user_request, context=None, run_dir=None):
                return _reception_agent(_site_brief()).process(user_request,
                                                               context=context,
                                                               run_dir=run_dir)
        co.reception = Front()
        out = wf.WorkflowCoordinator.process_request(co, "proceed anyway",
                                                     output_dir=str(tmp_path))
        assert out.startswith("🛑 Nothing was designed or run")
        assert "no prepared control exists for Sandy Creek" in out
        assert list(tmp_path.iterdir()) == []          # no run directory was made

    def test_a_single_column_plan_has_no_yes_even_when_typed(self, as_a_user, tmp_path,
                                                             monkeypatch):
        """Should a single-column plan reach the coordinator all the same, a
        typed yes does not run it."""
        co = _coordinator(_site_strategy(), interactive=True)
        co._execute = lambda **kw: pytest.fail("a single-column design must not run")
        asked = _typed(monkeypatch, ["yes"])
        out = co._workflow_design_and_run(_reception(_site_brief()), str(tmp_path), tmp_path)
        assert asked == ["This design cannot run as it is. Type a change, or press "
                         "Enter to stop: "]
        assert out.startswith("🛑 Nothing was submitted: this version runs whole-domain")
        assert "not run: this version does not run this kind of study" in _decision(tmp_path)
        assert not (tmp_path / dr.APPROVAL_FILE).exists()

    def test_unattended_a_single_column_plan_stops_before_the_manager(self, as_a_user,
                                                                      tmp_path):
        co = _coordinator(_site_strategy(), interactive=False)
        co._execute = lambda **kw: pytest.fail("a single-column design must not run")
        out = co._workflow_design_and_run(_reception(_site_brief()), str(tmp_path), tmp_path)
        assert out.startswith("🛑 Nothing was submitted: this version runs whole-domain")
        assert not (tmp_path / dr.APPROVAL_FILE).exists()

    @pytest.mark.parametrize("interactive", [True, False])
    def test_an_infeasible_design_of_any_kind_has_no_yes(self, tmp_path, monkeypatch,
                                                         interactive):
        """With the developer's switch set, a single-column design is reachable;
        its Planner's own verdict still stops it. It was read for a
        whole-domain design only."""
        monkeypatch.setenv(sr.SWITCH, "1")
        co = _coordinator(_site_strategy("infeasible"), interactive=interactive)
        co._execute = lambda **kw: pytest.fail("an infeasible design must not run")
        asked = _typed(monkeypatch, ["yes"] if interactive else [])
        out = co._workflow_design_and_run(_reception(_site_brief()), str(tmp_path), tmp_path)
        assert out.startswith("🛑 The planner judged this design infeasible")
        assert all(a.startswith("This design cannot run as it is") for a in asked)
        assert "not run: the planner judged the design infeasible" in _decision(tmp_path)


# ── a question that cannot be answered ends as a statement ───────────────────
class TestCannotAnswerIsNotAClarification:

    def test_the_route_carries_the_reason(self):
        route = LLMReceptionAgent._route({"intent": "cannot_answer",
                                          "reason": "No pumping change exists here. "
                                                    "A what-if on fire can be run."})
        assert route["action"] == "unsupported" and route["questions"] == []
        assert route["why"] == ("No pumping change exists here. A what-if on fire can "
                                "be run")

    def test_the_person_reads_a_statement(self, as_a_user, tmp_path, capsys):
        import workflow as wf
        co = _coordinator(None, interactive=True)
        co.default_output_dir = str(tmp_path)
        co._reception_context = lambda: None
        co._plan = lambda *a, **k: pytest.fail("nothing is planned")

        class Front:
            def process(self, user_request, context=None, run_dir=None):
                return _reception_agent({"intent": "cannot_answer",
                                         "reason": "No pumping change exists here."}
                                        ).process(user_request, run_dir=run_dir)
        co.reception = Front()
        out = wf.WorkflowCoordinator.process_request(co, "double the pumping",
                                                     output_dir=str(tmp_path))
        assert out.startswith("🛑 Nothing was designed or run: No pumping change exists here.")
        assert "clarification" not in (out + capsys.readouterr().out).lower()

    def test_a_question_still_open_stays_a_clarification(self):
        route = LLMReceptionAgent._route({"intent": "clarification_needed",
                                          "questions": ["which year?"]})
        assert route["action"] == "clarify" and route["questions"] == ["which year?"]


# ── an unattended run submits jobs only on a yes given at launch ─────────────
def _control_only():
    plan = _domain_strategy()
    plan["domain_experiment"]["runs"] = []
    return plan


class TestAnUnattendedRunNeedsItsYesAtLaunch:

    def test_a_design_that_would_run_the_model_stops(self, as_a_user, tmp_path):
        co = _coordinator(_domain_strategy(), interactive=False)
        co._execute = lambda **kw: pytest.fail("no one approved this design")
        out = co._workflow_design_and_run(_reception(_domain_brief()), str(tmp_path), tmp_path)
        assert out.startswith("🛑 Nothing was submitted: this design would run the model, "
                              "and no one approved it.")
        assert "--yes" in out and "--ask" in out
        assert "not run: no one approved it" in _decision(tmp_path)
        assert not (tmp_path / dr.APPROVAL_FILE).exists()

    def test_with_yes_at_launch_it_runs_and_says_how(self, as_a_user, tmp_path):
        plan = _domain_strategy()
        co = _coordinator(plan, interactive=False)
        co.approve_unattended = True
        co._execute = _ran
        co._workflow_design_and_run(_reception(_domain_brief()), str(tmp_path), tmp_path)
        rec = json.loads((tmp_path / dr.APPROVAL_FILE).read_text())
        assert rec["decision"] == "approved at launch (--yes), unattended"
        assert rec["design_sha256"] == dr.design_fingerprint(plan)

    def test_a_design_that_only_reads_the_control_continues(self, as_a_user, tmp_path):
        co = _coordinator(_control_only(), interactive=False)
        co._execute = lambda **kw: (_ for _ in ()).throw(_Ran("read"))
        out = co._workflow_design_and_run(_reception(_domain_brief()), str(tmp_path), tmp_path)
        assert "no one approved" not in out
        assert "auto-continued (unattended run)" in _decision(tmp_path)
        assert (tmp_path / dr.APPROVAL_FILE).exists()

    def test_a_control_that_is_run_again_is_compute_too(self, as_a_user, tmp_path):
        plan = _control_only()
        plan["domain_experiment"]["control"] = {"recipe": "r", "reuse": False}
        co = _coordinator(plan, interactive=False)
        co._execute = lambda **kw: pytest.fail("no one approved this design")
        out = co._workflow_design_and_run(_reception(_domain_brief()), str(tmp_path), tmp_path)
        assert out.startswith("🛑 Nothing was submitted: this design would run the model")

    def test_the_flag_reaches_the_coordinator(self):
        src = (ROOT / "workflow.py").read_text()
        assert "'--yes'" in src and "approve_unattended    = args.yes" in src


# ── the gate before compute ──────────────────────────────────────────────────
class TestTheGate:

    def test_it_stops_a_kind_set_aside(self, as_a_user):
        report, _ = strategy_check.check(_reception(_site_brief()), _site_strategy())
        assert not report["ok"]
        assert any("set aside in this version" in s for s in report["stop"])

    def test_with_the_switch_the_route_itself_stops_nothing(self, monkeypatch):
        monkeypatch.setenv(sr.SWITCH, "1")
        report, _ = strategy_check.check(_reception(_site_brief()), _site_strategy())
        assert not any("set aside" in s for s in report["stop"])

    def test_it_stops_an_infeasible_design_of_any_kind(self, monkeypatch):
        monkeypatch.setenv(sr.SWITCH, "1")
        report, _ = strategy_check.check(_reception(_site_brief()),
                                         _site_strategy("infeasible"))
        assert any(s.startswith("the planner judged this study infeasible")
                   for s in report["stop"])

    def test_a_whole_domain_design_passes_as_before(self, as_a_user):
        report, _ = strategy_check.check(_reception(_domain_brief()), _domain_strategy())
        assert report["ok"], report["stop"]


# ── only a typed yes runs ────────────────────────────────────────────────────
class _Ran(Exception):
    pass


def _ran(**kw):
    raise _Ran(kw["plan"]["domain_experiment"]["runs"][0]["name"])


class TestOnlyATypedYesRuns:

    def test_the_prompt_names_no_default(self, as_a_user, tmp_path, monkeypatch):
        co = _coordinator(_domain_strategy(), interactive=True)
        co._execute = lambda **kw: pytest.fail("declined")
        asked = _typed(monkeypatch, ["no"])
        co._workflow_design_and_run(_reception(_domain_brief()), str(tmp_path), tmp_path)
        assert asked == ["To run this design, type yes. Type no to stop, or type a change: "]
        assert "[Y/n" not in asked[0] and "Enter" not in asked[0]

    def test_an_empty_answer_approves_nothing(self, as_a_user, tmp_path, monkeypatch,
                                              capsys):
        co = _coordinator(_domain_strategy(), interactive=True)
        co._execute = lambda **kw: pytest.fail("an empty answer must not run a design")
        asked = _typed(monkeypatch, ["", "   ", ""])
        out = co._workflow_design_and_run(_reception(_domain_brief()), str(tmp_path), tmp_path)
        assert len(asked) == co.BLANK_ASKS == 3
        assert out.startswith("🛑 No answer was typed, so the design was not approved")
        assert "NOT APPROVED: no answer at the terminal" in _decision(tmp_path)
        assert not (tmp_path / dr.APPROVAL_FILE).exists()
        assert "Nothing runs without a typed yes" in capsys.readouterr().out

    def test_a_yes_after_an_empty_answer_runs(self, as_a_user, tmp_path, monkeypatch):
        co = _coordinator(_domain_strategy(), interactive=True)
        co._execute = _ran
        _typed(monkeypatch, ["", "yes"])
        out = co._workflow_design_and_run(_reception(_domain_brief()), str(tmp_path), tmp_path)
        assert "_Ran" in out or "warm" in out
        assert "accepted at the terminal" in _decision(tmp_path)

    @pytest.mark.parametrize("word", ["yes", "YES", "y", " Yes "])
    def test_a_typed_yes_is_recorded_with_the_design(self, as_a_user, tmp_path,
                                                     monkeypatch, word):
        plan = _domain_strategy()
        co = _coordinator(plan, interactive=True)
        co._execute = _ran
        _typed(monkeypatch, [word])
        co._workflow_design_and_run(_reception(_domain_brief()), str(tmp_path), tmp_path)
        rec = json.loads((tmp_path / dr.APPROVAL_FILE).read_text())
        assert rec["decision"] == "accepted at the terminal"
        assert rec["design_sha256"] == dr.design_fingerprint(plan)


# ── an approval is of one design ─────────────────────────────────────────────
class TestAnApprovalIsOfOneDesign:

    def test_a_change_then_a_yes_runs_the_new_design_only(self, as_a_user, tmp_path,
                                                          monkeypatch):
        """Design A is on the page, a change is typed, design B is shown in its
        own folder and approved. A is never run and holds no approval."""
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir(), second.mkdir()
        plans = {str(first): _domain_strategy("warm", (2031, 2032)),
                 str(second): _domain_strategy("warm", (2031, 2032, 2033))}
        co = _coordinator(None, interactive=True)
        co._plan = lambda result, run_dir, findings=None, previous=None: plans[str(run_dir)]
        ran = []
        co._execute = lambda **kw: ran.append((str(kw["run_dir"]), kw["plan"])) or \
            (_ for _ in ()).throw(_Ran("stop here"))
        co.process_request = lambda request, output_dir=None, revision=None: \
            co._workflow_design_and_run(_reception(_domain_brief()), str(tmp_path), second)
        _typed(monkeypatch, ["make it one year longer", "yes"])
        co._workflow_design_and_run(_reception(_domain_brief()), str(tmp_path), first)
        assert ran == [(str(second), plans[str(second)])]
        assert "revision requested at the terminal" in _decision(first)
        assert not (first / dr.APPROVAL_FILE).exists()
        rec = json.loads((second / dr.APPROVAL_FILE).read_text())
        assert rec["design_sha256"] == dr.design_fingerprint(plans[str(second)])
        assert rec["design_sha256"] != dr.design_fingerprint(plans[str(first)])

    def _execute(self, co, run_dir, plan, **kw):
        return co._execute(plan=plan, output_dir=str(Path(run_dir).parent),
                           run_dir=Path(run_dir), reception=_reception(_domain_brief()), **kw)

    @pytest.fixture
    def no_manager(self, monkeypatch):
        """The manager is the next thing after the approval check: reaching it
        raises _Ran, and a design stopped by the check never gets there."""
        import workflow as wf

        def reached(model):
            raise _Ran("the manager was reached")
        monkeypatch.setattr(wf, "_manager_for", reached)

    def test_a_design_with_no_approval_does_not_reach_the_manager(self, as_a_user,
                                                                  tmp_path, no_manager):
        co = _coordinator(None, interactive=True)
        with pytest.raises(dr.DesignNotApproved, match="no approval of this design"):
            self._execute(co, tmp_path, _domain_strategy())

    def test_the_approved_design_reaches_it(self, as_a_user, tmp_path, no_manager):
        co, plan = _coordinator(None, interactive=True), _domain_strategy()
        dr.approve(tmp_path, plan, "accepted at the terminal")
        with pytest.raises(_Ran):
            self._execute(co, tmp_path, plan)

    @pytest.mark.parametrize("other", [
        _domain_strategy("warm", (2031, 2032, 2033)),       # other years
        _domain_strategy("warmer", (2031, 2032)),           # another run
    ])
    def test_a_design_other_than_the_one_approved_does_not(self, as_a_user, tmp_path,
                                                           no_manager, other):
        """The approval of the superseded design is no approval of this one."""
        co = _coordinator(None, interactive=True)
        dr.approve(tmp_path, _domain_strategy("warm", (2031, 2032)), "accepted at the terminal")
        with pytest.raises(dr.DesignNotApproved, match="not the design that was approved"):
            self._execute(co, tmp_path, other)

    def test_a_change_to_the_analysis_alone_keeps_the_approval(self, as_a_user, tmp_path,
                                                               no_manager):
        """The fingerprint is of what runs: a plan whose reported fields were
        amended afterwards is still the design approved."""
        co, plan = _coordinator(None, interactive=True), _domain_strategy()
        dr.approve(tmp_path, plan, "accepted at the terminal")
        amended = json.loads(json.dumps(plan))
        amended["domain_experiment"]["analysis"] = {"variables": ["A_FIELD"]}
        with pytest.raises(_Ran):
            self._execute(co, tmp_path, amended)

    @pytest.mark.parametrize("decision", [
        "DECLINED at the terminal",
        "revision requested at the terminal: 'make it one year longer'",
        "not run: the planner judged the design infeasible",
        "NOT APPROVED: no answer at the terminal",
    ])
    def test_a_design_that_was_not_approved_is_not_resumed(self, as_a_user, tmp_path,
                                                           no_manager, decision):
        plan = _domain_strategy()
        dr.write_review(tmp_path, reception=_reception(_domain_brief()), strategy=plan,
                        preview=None)
        dr.record_decision(tmp_path, decision)
        co = _coordinator(None, interactive=False)
        with pytest.raises(dr.DesignNotApproved, match="was not approved"):
            self._execute(co, tmp_path, plan, resume=True)

    def test_a_study_made_before_approvals_were_recorded_resumes(self, as_a_user, tmp_path,
                                                                 no_manager):
        plan = _domain_strategy()
        dr.write_review(tmp_path, reception=_reception(_domain_brief()), strategy=plan,
                        preview=None)
        dr.record_decision(tmp_path, "accepted at the terminal")
        co = _coordinator(None, interactive=False)
        with pytest.raises(_Ran):
            self._execute(co, tmp_path, plan, resume=True)
