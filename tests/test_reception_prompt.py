"""Reception's prompt, held to the code that reads its output and to what this
version of the framework runs.

    pytest tests/test_reception_prompt.py -v

THE PROMPT WAS REWRITTEN FOR WHOLE-DOMAIN STUDIES ALONE (2026-11-11). Until
then it was the prompt of four kinds of study, 735 lines, with a block
appended that told the model to disregard three of them. A prompt is checked
by nobody at run time, so these tests hold what can be held without a model:

  - it is one file, and short;
  - its endings are the ones the code routes, and no other;
  - the tools it names are the tools Reception is given, and no other;
  - it names no model, no control, no change and none of the studies set aside;
  - the brief it describes is read as a whole-domain study by the code;
  - the rules that live conversations taught are still in it.

What a model DOES with the prompt is shown only by live conversations; their
record is kept outside the repository.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "src" / "agents" / "prompts"
PROMPT = (PROMPTS / "reception_agentic.txt").read_text()
FLAT = " ".join(PROMPT.split())
TOOLS = {"list_servers", "describe_server", "describe_domain_controls",
         "describe_domain_changes"}


class FakeLoop:
    """Stands where the tool loop does: keeps what Reception hands it."""
    made = None

    def __init__(self, **kw):
        self.kw = kw
        self.local, self.tools = kw["local_tools"], []
        self.seen = []
        FakeLoop.made = self

    def run(self, system, msg):
        self.seen.append((system, msg))
        return {"content": '{"intent": "clarification_needed", "questions": ["q"]}',
                "trace": [], "rounds": 1, "messages": []}


@pytest.fixture
def reception(monkeypatch):
    import agents.reception_llm as R
    monkeypatch.setattr(R, "ToolLoopAgent", FakeLoop)
    return R.LLMReceptionAgent(model="x", mcp_clients={})


def test_it_is_one_file_and_well_under_half_of_what_it_was():
    assert not (PROMPTS / "reception_whole_domain_only.txt").exists()
    assert len(PROMPT) < 20000          # it was 46,000 characters, plus the override


def test_reception_loads_it_alone_whatever_the_switch_says(reception, monkeypatch):
    from core.supported_routes import SWITCH
    assert reception.system.strip() == PROMPT.strip()
    import agents.reception_llm as R
    for value in ("1", ""):
        monkeypatch.setenv(SWITCH, value)
        assert R.LLMReceptionAgent(model="x", mcp_clients={}).system.strip() == PROMPT.strip()


def test_its_endings_are_the_ones_the_code_routes():
    from agents.reception_llm import LLMReceptionAgent as R
    routed = {"design": "design", "cannot_answer": "unsupported",
              "clarification_needed": "clarify", "analyze_existing": "analyze_existing",
              "resume": "resume"}
    assert set(re.findall(r'"intent": "([a-z_]+)"', PROMPT)) == set(routed)
    for intent, action in routed.items():
        got = R._route({"intent": intent, "questions": ["q"], "reason": "r"})
        assert got["action"] == action, intent
    # each is introduced where the prompt lists how it may end
    head = PROMPT[:PROMPT.index("You do not answer the science")]
    for intent in routed:
        assert re.search(rf"^\s+{intent}\s", head, re.M), intent


def test_the_tools_it_names_are_the_tools_it_is_given(reception):
    import agents.reception_llm as R
    assert set(R.reception_tools({})) == TOOLS
    assert set(FakeLoop.made.local) == TOOLS
    assert FakeLoop.made.kw["allowlist"] == set()        # no data tool of any server
    named = set(re.findall(r"`(list_servers|describe_[a-z_]+|ask_user)`", PROMPT))
    assert named == TOOLS | {"ask_user"}
    # a tool it is not given is not mentioned, in any spelling
    for gone in ("describe_sweep", "terrain__", "resolve_watershed", "get_elevation"):
        assert gone not in PROMPT, gone


def test_it_names_no_model_no_control_no_change_and_no_study_set_aside():
    import domain_changes as DC
    low = PROMPT.lower()
    for word in ("elm", "pflotran", "e3sm"):
        assert not re.search(rf"\b{word}\b", low), word
    for change in DC.declare()["changes"]:
        assert not re.search(rf"\b{change['id']}\b", PROMPT), change["id"]
    # the studies set aside, and the fields only they wrote
    for gone in ("conceptual", "coupling", "sweep", "requested_n_columns", "held_fixed",
                 "heterogeneity", "initialization", "observations_summary", "HUC",
                 "from_model", "pinning"):
        assert gone not in PROMPT, gone
    # one archetype, written as a constant
    assert set(re.findall(r'"design_archetype": ([^,\n]+)', PROMPT)) == {'"whole_domain"'}


def test_the_brief_it_describes_is_read_as_a_whole_domain_study(monkeypatch):
    schema = PROMPT[PROMPT.index("design (the brief):"):]
    for key in ("intent", "design_archetype", "user_request", "model", "model_rationale",
                "model_availability", "domain_experiment", "control", "reuse", "runs",
                "changes", "id", "params", "settled_by", "years", "unresolved",
                "scientific_framing", "goals", "domain", "forcing_data", "run_settings",
                "requested_period", "resolved_period", "conflicts", "known_data_gaps",
                "notes"):
        assert f'"{key}"' in schema, key
    # a brief written to it, in made-up values, is whole-domain to every reader
    brief = {"intent": "design", "design_archetype": "whole_domain", "user_request": "q",
             "model": "m",
             "domain_experiment": {"control": {"name": "c", "reuse": True}, "runs": [],
                                   "years": [2001, 2002], "unresolved": []},
             "domain": {"name": "c", "area_km2": None, "source": "s"},
             "run_settings": {"requested_period": {"yr_start": 2001, "yr_end": 2002},
                              "resolved_period": {"yr_start": 2001, "yr_end": 2002,
                                                  "source": "user"}, "conflicts": []},
             "known_data_gaps": [], "notes": "n"}
    from core import supported_routes as SR
    from core.analysis_years import requested
    monkeypatch.delenv(SR.SWITCH, raising=False)
    assert SR.kind(brief) == "whole_domain" and SR.problem(brief) is None
    assert requested(brief) == [2001, 2002]


def test_a_design_reply_fetches_nothing_and_says_why(monkeypatch):
    """The reply the prompt asks for goes through process() with no data
    server called: a whole-domain study reads its control's own grid."""
    import agents.reception_llm as R
    from core import supported_routes as SR

    class Designs(FakeLoop):
        def run(self, system, msg):
            self.seen.append((system, msg))
            return {"content": ('{"intent": "design", "design_archetype": "whole_domain", '
                                '"model": "m", "domain": "c", "domain_experiment": '
                                '{"control": {"name": "c", "reuse": true}, "runs": [], '
                                '"years": [2001]}}'),
                    "trace": [], "rounds": 1, "messages": []}

    monkeypatch.delenv(SR.SWITCH, raising=False)
    monkeypatch.setattr(R, "ToolLoopAgent", Designs)
    pkg = R.LLMReceptionAgent(model="x", mcp_clients={}).process("q")
    assert pkg["route"]["action"] == "design"
    assert pkg["brief"]["domain"] == {"name": "c"}
    assert pkg["observations"] == {} and pkg["grid"] == {}
    assert "whole-domain experiment runs on its control's own grid" in \
        pkg["provenance"][-1]["skipped"]


def test_the_message_that_starts_the_conversation_is_about_what_is_run(reception):
    reception.process("a question")
    system, msg = FakeLoop.made.seen[-1]
    assert system.strip() == PROMPT.strip()
    assert msg.startswith("User request: a question")
    assert "Settle what is run with the person" in msg
    for old in ("Resolve the domain", "observations", "period is fixed"):
        assert old not in msg, old


def test_the_rules_that_live_conversations_taught_are_in_it():
    for rule in (
            # a menu read twice, and a menu never needed, were both paid for
            "Never call a tool twice with the same arguments",
            "In ONE step, on that server, call all three",
            # a cost multiplied by the model came out in the wrong unit
            "THE COST IS NOT YOURS TO COMPUTE",
            "no arithmetic of your own",
            # a change asked for two years was applied to every simulated year
            "unless the change's own parameters say which years it acts in",
            # a design was refused for a run name already used
            "never with a name in the control's `run_names_taken`",
            # a change no run has completed with was offered as proven
            "A change marked `demonstrated: false` may be designed",
            # an unsupported study was offered after "proceed anyway"
            'never offer single model columns, another model, or to "proceed anyway"',
            "When they decline, or tell you to stop, design nothing",
            # the start year a run needs was reported as a year asked about
            "exactly as the person stated or agreed them, never widened",
            "Do not add the control's start year to them",
            # one message, plain, and true of the design
            "Send ONE `ask_user` message before the brief",
            # an acknowledgement was put to the person as a question
            "`ask_user` is for a question, never for telling the person something",
            "Do not say control, whole-domain, brief or demonstrated",
            "No field name, tool name or JSON",
            'Never say "as you asked" for something you chose',
            "Check every concrete statement against the JSON you are about to emit",
            # a remark about the design, then the run's own record, was filed
            # as a data gap
            "record holds, and only that: [] otherwise",
            "It is no data gap: do not copy it into known_data_gaps",
            # a conclusion of its own was added to what the record says, so
            # the results are not shown to it and it says nothing about them
            "You are not shown the run's results",
            "No word on the run's results or on how good the run is, and no "
            "conclusion of your own"):
        assert rule in FLAT, rule
