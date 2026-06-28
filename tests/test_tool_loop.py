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


def test_ask_user_interactive_uses_human_answer(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "the Washington one")
    scripted = [_msg(tool_calls=[("ask_user", {"question": "WA or CA American River?"})]),
                _msg(content="done")]
    agent, _, _ = _make_agent(monkeypatch, scripted, interactive=True)
    assert any(t["function"]["name"] == "ask_user" for t in agent.tools)
    out = agent.run("sys", "go")
    assert out["trace"][0]["tool"] == "ask_user"
    assert out["trace"][0]["result"]["answer"] == "the Washington one"


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
