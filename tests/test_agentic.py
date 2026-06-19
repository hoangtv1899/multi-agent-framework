"""
Offline unit tests for the agentic layer (no network):
  * ToolLoopAgent._build_tools — MCP tool specs -> OpenAI tool schemas (prefix,
    dispatch map, allowlist filtering)
  * LLMReceptionAgent._parse   — extract the brief JSON from a final message

The live loop itself is exercised by tools/run_pipeline.py.

Run under: module load pytorch/2.8.0
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("openai")   # tool_loop imports SimpleLLMClient -> openai

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.tool_loop import ToolLoopAgent          # noqa: E402
from agents.reception_llm import LLMReceptionAgent  # noqa: E402


class FakeClient:
    """Stands in for an MCPClient — only list_tools_detailed is needed."""
    def __init__(self, tools):
        self._tools = tools

    def list_tools_detailed(self):
        return self._tools


def _tool(name):
    return {"name": name, "description": f"does {name}",
            "parameters": {"type": "object", "properties": {}}}


class TestToolSchemas:
    def test_prefix_and_dispatch(self):
        clients = {
            "terrain": FakeClient([_tool("resolve_watershed"), _tool("get_elevation")]),
            "fan_wtd": FakeClient([_tool("get_fan_wtd")]),
        }
        tools, dispatch = ToolLoopAgent._build_tools(clients, None)
        names = {t["function"]["name"] for t in tools}
        assert names == {"terrain__resolve_watershed", "terrain__get_elevation",
                         "fan_wtd__get_fan_wtd"}
        assert dispatch["fan_wtd__get_fan_wtd"] == ("fan_wtd", "get_fan_wtd")
        for t in tools:                       # valid OpenAI tool shape
            assert t["type"] == "function"
            assert "parameters" in t["function"]

    def test_allowlist_filters(self):
        clients = {"terrain": FakeClient([_tool("a"), _tool("b")])}
        tools, dispatch = ToolLoopAgent._build_tools(clients, {"terrain__a"})
        assert [t["function"]["name"] for t in tools] == ["terrain__a"]
        assert "terrain__b" not in dispatch

    def test_missing_params_defaulted(self):
        clients = {"x": FakeClient([{"name": "t", "description": "d"}])}  # no parameters
        tools, _ = ToolLoopAgent._build_tools(clients, None)
        assert tools[0]["function"]["parameters"]["type"] == "object"


class TestReceptionParse:
    def test_parse_fenced_brief(self):
        b = LLMReceptionAgent._parse('```json\n{"intent":"design","design_archetype":"site"}\n```')
        assert b["intent"] == "design" and b["design_archetype"] == "site"

    def test_parse_with_prose_prefix(self):
        b = LLMReceptionAgent._parse(
            'Here is my output:\n{"intent":"clarification_needed","questions":["x"]}')
        assert b["intent"] == "clarification_needed"

    def test_parse_error_when_no_json(self):
        b = LLMReceptionAgent._parse("no json at all here")
        assert b["intent"] == "parse_error"
