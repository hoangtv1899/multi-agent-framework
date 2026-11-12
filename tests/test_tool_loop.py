"""Offline tests for the agentic tool-loop runtime `ToolLoopAgent.run()`.

`test_agentic.py` covers the static `_build_tools`/`_parse`; this covers the
*loop* — previously only exercised live by run_pipeline.py. We swap
SimpleLLMClient for a scripted fake (canned OpenAI-shaped responses) and feed
fake MCP clients that record dispatch, then assert: tool dispatch + result
feedback, the `tools=` every-round rule (Bedrock), unknown-tool handling,
ask_user (interactive + batch), multi-call rounds, and max-rounds truncation.

No network, no LLM. Run under: module load pytorch/2.8.0
"""
import json
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")   # tool_loop imports SimpleLLMClient -> openai

sys.path.insert(0, "src")
import agents.tool_loop as tl                      # noqa: E402
from agents.tool_loop import ToolLoopAgent         # noqa: E402


# ── fakes ──────────────────────────────────────────────────────────────────
class _FakeMCP:
    """Advertises tools and records every call_tool_json invocation."""
    def __init__(self, tools):
        self._tools = tools
        self.calls = []

    def list_tools_detailed(self):
        return self._tools

    def call_tool_json(self, tool, args):
        self.calls.append((tool, args))
        return {"ok": True, "tool": tool, "args": args}


def _msg(content=None, tool_calls=None):
    """An OpenAI-shaped assistant message. tool_calls: [(fq_name, args_dict), ...]."""
    tcs = None
    if tool_calls:
        tcs = [SimpleNamespace(id=f"call_{i}", type="function",
                               function=SimpleNamespace(name=name,
                                                        arguments=json.dumps(a)))
               for i, (name, a) in enumerate(tool_calls)]
    return SimpleNamespace(content=content, tool_calls=tcs)


def _resp(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _ScriptedLLM:
    """Drop-in for SimpleLLMClient: pops queued responses; records requests."""
    def __init__(self, scripted):
        self.model = "fake-model"
        self._queue = list(scripted)
        self.requests = []
        self.client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=self._create)))

    def _create(self, **kw):
        self.requests.append(kw)
        return _resp(self._queue.pop(0))


def _tool(name):
    return {"name": name, "description": f"does {name}",
            "parameters": {"type": "object", "properties": {}}}


def _make_agent(monkeypatch, scripted, clients=None, **kw):
    """Build a ToolLoopAgent whose LLM is the scripted fake."""
    llm = _ScriptedLLM(scripted)
    monkeypatch.setattr(tl, "SimpleLLMClient", lambda model: llm)
    if clients is None:
        clients = {"terrain": _FakeMCP([_tool("resolve_watershed")]),
                   "fan_wtd": _FakeMCP([_tool("get_fan_wtd")])}
    agent = ToolLoopAgent(model="fake", mcp_clients=clients, verbose=False, **kw)
    return agent, llm, clients


# ── tests ────────────────────────────────────────────────────────────────────
def test_no_tools_first_round_returns_immediately(monkeypatch):
    agent, _, _ = _make_agent(monkeypatch, [_msg(content="final text")])
    out = agent.run("sys", "hi")
    assert out["content"] == "final text"
    assert out["rounds"] == 1 and out["trace"] == [] and out["truncated"] is False


def test_dispatches_tool_then_returns_final(monkeypatch):
    scripted = [_msg(tool_calls=[("terrain__resolve_watershed", {"name": "Naches"})]),
                _msg(content='{"intent":"design"}')]
    agent, _, clients = _make_agent(monkeypatch, scripted)
    out = agent.run("sys", "frame the Naches")
    assert out["content"] == '{"intent":"design"}'
    assert out["rounds"] == 2 and out["truncated"] is False
    # the loop dispatched to the right MCP client with parsed args
    assert clients["terrain"].calls == [("resolve_watershed", {"name": "Naches"})]
    # and recorded a trace entry carrying the tool result back
    assert len(out["trace"]) == 1
    assert out["trace"][0]["tool"] == "terrain__resolve_watershed"
    assert out["trace"][0]["result"]["ok"] is True


def test_tools_passed_on_every_round(monkeypatch):
    """Bedrock/Claude rule: tools= must accompany every request, incl. follow-ups."""
    scripted = [_msg(tool_calls=[("terrain__resolve_watershed", {})]),
                _msg(content="done")]
    agent, llm, _ = _make_agent(monkeypatch, scripted)
    agent.run("sys", "go")
    assert len(llm.requests) == 2
    for req in llm.requests:
        assert req.get("tools") == agent.tools and req["tools"]


def test_multiple_tool_calls_in_one_round(monkeypatch):
    scripted = [_msg(tool_calls=[("terrain__resolve_watershed", {}),
                                 ("fan_wtd__get_fan_wtd", {"lat": 46.7})]),
                _msg(content="done")]
    agent, _, clients = _make_agent(monkeypatch, scripted)
    out = agent.run("sys", "go")
    assert len(out["trace"]) == 2                 # both calls in round 1
    assert all(t["round"] == 1 for t in out["trace"])
    assert clients["terrain"].calls and clients["fan_wtd"].calls


def test_unknown_tool_returns_error_not_crash(monkeypatch):
    scripted = [_msg(tool_calls=[("nope__bad", {})]), _msg(content="recovered")]
    agent, _, _ = _make_agent(monkeypatch, scripted)
    out = agent.run("sys", "go")
    assert out["content"] == "recovered"
    assert out["trace"][0]["result"] == {"error": "unknown tool nope__bad"}


def test_bad_tool_arguments_default_to_empty(monkeypatch):
    # arguments that aren't valid JSON must not crash the loop
    bad = SimpleNamespace(id="call_0", type="function",
                          function=SimpleNamespace(name="terrain__resolve_watershed",
                                                   arguments="{not json"))
    scripted = [SimpleNamespace(content=None, tool_calls=[bad]), _msg(content="ok")]
    agent, _, clients = _make_agent(monkeypatch, scripted)
    out = agent.run("sys", "go")
    assert out["content"] == "ok"
    assert clients["terrain"].calls == [("resolve_watershed", {})]


class _FakeTTY:
    """The loop reads the human's reply off a terminal opened at construction,
    not via input(). Under pytest there is no controlling terminal, so the test
    supplies one — patching builtins.input reaches nothing and the assertion
    then only proves the batch fallback it caused itself."""
    def __init__(self, line): self.line, self.reads = line, 0
    def readline(self): self.reads += 1; return self.line


def test_ask_user_interactive_uses_human_answer(monkeypatch):
    scripted = [_msg(tool_calls=[("ask_user", {"question": "WA or CA American River?"})]),
                _msg(content="done")]
    agent, _, _ = _make_agent(monkeypatch, scripted, interactive=True)
    tty = _FakeTTY("the Washington one\n")
    agent._tty = tty
    assert any(t["function"]["name"] == "ask_user" for t in agent.tools)
    out = agent.run("sys", "go")
    assert out["trace"][0]["tool"] == "ask_user"
    assert out["trace"][0]["result"]["answer"] == "the Washington one"
    assert tty.reads == 1, "the human must actually be read, not defaulted past"


@pytest.mark.parametrize("args", [{}, {"question": ""}, {"question": "   "},
                                  {"question": None}])
def test_an_empty_question_is_not_put_to_the_person(monkeypatch, args):
    """Seen live (2026-11-10): ask_user called with no arguments showed the
    person a bare prompt. The model is told; the terminal is never read."""
    scripted = [_msg(tool_calls=[("ask_user", args)]), _msg(content="done")]
    agent, _, _ = _make_agent(monkeypatch, scripted, interactive=True)
    tty = _FakeTTY("an answer to nothing\n")
    agent._tty = tty
    out = agent.run("sys", "go")
    assert tty.reads == 0
    assert "no question text" in out["trace"][0]["result"]["error"]
    assert out["content"] == "done"


def _tool_content(llm, k=0):
    """What the model was given back for its k-th tool call."""
    msgs = [m for m in llm.requests[-1]["messages"] if isinstance(m, dict)
            and m.get("role") == "tool"]
    return msgs[k]["content"]


def test_a_framework_tools_answer_is_read_whole_past_the_data_cap(monkeypatch):
    """The servers' menus come through framework tools and must be read whole:
    at the data tools' cap the list of changes had 2 characters to spare."""
    long = {"menu": "x" * (tl._MAX_TOOL_RESULT_CHARS + 3000)}
    local = {"read_menu": {"schema": {"type": "function", "function": {
        "name": "read_menu", "description": "d",
        "parameters": {"type": "object", "properties": {}}}}, "fn": lambda: long}}
    scripted = [_msg(tool_calls=[("read_menu", {})]), _msg(content="done")]
    agent, llm, _ = _make_agent(monkeypatch, scripted, local_tools=local)
    out = agent.run("sys", "go")
    assert "truncated" not in out["trace"][0]
    assert "[truncated]" not in _tool_content(llm)
    assert len(_tool_content(llm)) == len(json.dumps(long))


def _menu_tool(fn):
    return {"read_menu": {"schema": {"type": "function", "function": {
        "name": "read_menu", "description": "d",
        "parameters": {"type": "object",
                       "properties": {"server": {"type": "string"}}}}}, "fn": fn}}


def test_a_framework_tool_is_answered_once_for_the_same_arguments(monkeypatch):
    """A live reception read a twelve-thousand-character menu twice, and the
    second copy was carried in every later round. The prompt asks the model
    not to; the loop makes it so."""
    calls = []

    def read_menu(server=None):
        calls.append(server)
        return {"menu": "m" * 5000, "server": server}
    scripted = [_msg(tool_calls=[("read_menu", {"server": "a"})]),
                _msg(tool_calls=[("read_menu", {"server": "a"}),
                                 ("read_menu", {"server": "b"})]),
                _msg(content="done")]
    agent, llm, _ = _make_agent(monkeypatch, scripted, local_tools=_menu_tool(read_menu))
    out = agent.run("sys", "go")
    assert calls == ["a", "b"]                         # the repeat reached no tool
    again = out["trace"][1]["result"]
    assert again["repeated"] is True and "read it above" in again["note"]
    assert "read_menu" in again["note"] and len(json.dumps(again)) < 300
    assert out["trace"][2]["result"]["server"] == "b"  # other arguments are answered
    assert len(_tool_content(llm, 1)) < 300            # the menu is not carried twice


@pytest.mark.parametrize("fails", ["raises", "says"])
def test_a_framework_tool_that_failed_may_be_asked_again(monkeypatch, fails):
    state = {"n": 0}

    def read_menu(server=None):
        state["n"] += 1
        if state["n"] == 1:
            if fails == "raises":
                raise RuntimeError("not ready")
            return {"error": "not ready"}
        return {"menu": "m"}
    scripted = [_msg(tool_calls=[("read_menu", {"server": "a"})]),
                _msg(tool_calls=[("read_menu", {"server": "a"})]),
                _msg(tool_calls=[("read_menu", {"server": "a"})]),
                _msg(content="done")]
    agent, _, _ = _make_agent(monkeypatch, scripted, local_tools=_menu_tool(read_menu))
    out = agent.run("sys", "go")
    assert "not ready" in out["trace"][0]["result"]["error"]
    assert out["trace"][1]["result"] == {"menu": "m"}
    assert out["trace"][2]["result"]["repeated"] is True and state["n"] == 2


def test_a_statement_is_not_put_to_the_person(monkeypatch):
    """A live Reception, once the person had confirmed, used ask_user to say
    what it was about to do. The person saw a prompt with nothing to answer."""
    asked = []
    monkeypatch.setattr(tl, "_human_answer",
                        lambda q, tty, interactive: asked.append(q) or {"answer": "yes"})
    scripted = [_msg(tool_calls=[("ask_user", {"question": "Shall I read 2001 to 2003?"})]),
                _msg(tool_calls=[("ask_user", {"question": "Reading 2001 to 2003, no new "
                                                           "runs."})]),
                _msg(content="done")]
    agent, _, _ = _make_agent(monkeypatch, scripted, interactive=True)
    out = agent.run("sys", "go")
    assert asked == ["Shall I read 2001 to 2003?"]
    told = out["trace"][1]["result"]["error"]
    assert "asks none, so it was not shown to the person" in told
    assert "give your final answer now" in told


def test_the_person_may_be_asked_twice(monkeypatch):
    """Only a framework tool's answer is fixed for a conversation."""
    monkeypatch.setattr(tl, "_human_answer", lambda q, tty, interactive: {"answer": "yes"})
    scripted = [_msg(tool_calls=[("ask_user", {"question": "ok?"})]),
                _msg(tool_calls=[("ask_user", {"question": "ok?"})]),
                _msg(content="done")]
    agent, _, _ = _make_agent(monkeypatch, scripted, interactive=True)
    out = agent.run("sys", "go")
    assert [s["result"] for s in out["trace"]] == [{"answer": "yes"}, {"answer": "yes"}]


def test_when_to_ask_is_left_to_the_prompt():
    said = tl.ASK_USER_TOOL["function"]["description"]
    for old in ("HUC", "watershed", "SIMULATION PERIOD", "spin-up", "observations"):
        assert old not in said, old
    assert "as your instructions say" in said


def test_a_cut_is_said_and_marked_never_silent(monkeypatch, capsys):
    huge = {"menu": "x" * (tl._MAX_LOCAL_RESULT_CHARS + 500)}
    local = {"read_menu": {"schema": {"type": "function", "function": {
        "name": "read_menu", "description": "d",
        "parameters": {"type": "object", "properties": {}}}}, "fn": lambda: huge}}
    scripted = [_msg(tool_calls=[("read_menu", {})]), _msg(content="done")]
    agent, llm, _ = _make_agent(monkeypatch, scripted, local_tools=local)
    out = agent.run("sys", "go")
    assert out["trace"][0]["truncated"] == {"chars": len(json.dumps(huge)),
                                            "kept": tl._MAX_LOCAL_RESULT_CHARS}
    assert _tool_content(llm).endswith("[truncated]")
    assert "the rest was not read" in capsys.readouterr().out


def test_ask_user_without_a_terminal_does_not_hang(monkeypatch):
    """No terminal (batch, piped, nohup) must degrade to the sentinel."""
    scripted = [_msg(tool_calls=[("ask_user", {"question": "which basin?"})]),
                _msg(content="done")]
    agent, _, _ = _make_agent(monkeypatch, scripted, interactive=True)
    agent._tty = None
    out = agent.run("sys", "go")
    assert "non-interactive" in out["trace"][0]["result"]["answer"].lower()


def test_ask_user_batch_returns_sentinel(monkeypatch):
    scripted = [_msg(tool_calls=[("ask_user", {"question": "which HUC?"})]),
                _msg(content="done")]
    agent, _, _ = _make_agent(monkeypatch, scripted, interactive=False)
    out = agent.run("sys", "go")
    assert "non-interactive" in out["trace"][0]["result"]["answer"].lower()


def test_max_rounds_truncation(monkeypatch):
    # both rounds keep calling tools -> loop exhausts, then one final no-tool call
    scripted = [_msg(tool_calls=[("terrain__resolve_watershed", {})]),
                _msg(tool_calls=[("terrain__resolve_watershed", {})]),
                _msg(content="forced final")]
    agent, llm, _ = _make_agent(monkeypatch, scripted, max_rounds=2)
    out = agent.run("sys", "go")
    assert out["truncated"] is True and out["rounds"] == 2
    assert out["content"] == "forced final"
    assert len(llm.requests) == 3                 # 2 loop rounds + 1 forced final
