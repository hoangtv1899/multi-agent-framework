"""A cut-off LLM reply is missing DATA, not bad syntax.

The gateway defaults to 4096 completion tokens. The planner's plan is longer,
so every call came back finish_reason="length", truncated mid-JSON. The parse
failed, the repair round asked the model to "fix" it, and the model invented the
part that never arrived — so `n_exploratory` came back 8, 11, 12, 12 and then
missing entirely across five runs of the SAME brief. The sampling design was
partly written by a JSON-repair prompt.

These pin the two halves of the fix: a cap large enough for the real documents,
and a refusal to repair truncation as though it were a formatting slip.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.llm_agent import LLMAgent, SimpleLLMClient  # noqa: E402


class _FakeLLM:
    """Stands in for SimpleLLMClient without touching the network."""
    def __init__(self, finish_reason=None, repair=None):
        self.last_finish_reason = finish_reason
        self.model = "fake"
        self._repair = repair
        self.repair_calls = 0

    def ask(self, messages, system_message=None):
        self.repair_calls += 1
        return self._repair


def _agent(llm):
    a = LLMAgent.__new__(LLMAgent)      # skip __init__: it builds a real client
    a.llm = llm
    return a


class TestCapIsLargeEnough:
    def test_the_default_is_not_the_provider_default(self, monkeypatch):
        """None meant 4096, which truncated every plan."""
        monkeypatch.setenv("PNNL_API_KEY", "x")
        c = SimpleLLMClient.__new__(SimpleLLMClient)
        SimpleLLMClient.__init__(c)
        assert c.max_tokens is not None
        assert c.max_tokens >= 8192

    def test_tool_loop_cap_clears_the_4096_wall(self):
        from agents.tool_loop import ToolLoopAgent
        import inspect
        default = inspect.signature(ToolLoopAgent.__init__).parameters["max_tokens"].default
        assert default > 4096


class TestTruncationIsNotRepaired:
    def test_a_cut_off_reply_raises_instead_of_being_repaired(self):
        """Repairing it would fabricate the missing fields, and a fabricated
        n_exploratory is indistinguishable from a designed one."""
        llm = _FakeLLM(finish_reason="length", repair='{"n_exploratory": 99}')
        a = _agent(llm)
        with pytest.raises(ValueError, match="cut off"):
            a.parse_json_resilient('{"sampling_strategy": {"n_explora')
        assert llm.repair_calls == 0, "must not ask the model to invent the tail"

    def test_the_error_names_the_actual_remedy(self):
        a = _agent(_FakeLLM(finish_reason="length"))
        with pytest.raises(ValueError) as e:
            a.parse_json_resilient('{"a": ')
        assert "max_tokens" in str(e.value)

    def test_a_genuine_syntax_slip_is_still_repaired(self):
        """Truncation is the only case that must not be repaired — a complete
        but malformed reply is exactly what the repair round is for."""
        llm = _FakeLLM(finish_reason="stop", repair='{"a": 1, "b": 2}')
        a = _agent(llm)
        assert a.parse_json_resilient('{"a": 1 "b": 2}') == {"a": 1, "b": 2}
        assert llm.repair_calls == 1

    def test_a_valid_reply_never_reaches_the_repair_round(self):
        llm = _FakeLLM(finish_reason="stop", repair="{}")
        a = _agent(llm)
        assert a.parse_json_resilient('{"a": 1}') == {"a": 1}
        assert llm.repair_calls == 0

    def test_unknown_finish_reason_does_not_block_repair(self):
        """finish_reason is absent on some gateway paths; absence must not be
        read as truncation or every malformed reply becomes fatal."""
        llm = _FakeLLM(finish_reason=None, repair='{"a": 1}')
        a = _agent(llm)
        assert a.parse_json_resilient('{"a" 1}') == {"a": 1}
        assert llm.repair_calls == 1
