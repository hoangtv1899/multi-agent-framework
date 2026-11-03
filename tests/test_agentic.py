"""
Offline unit tests for the agentic layer (no network):
  * ToolLoopAgent._build_tools — MCP tool specs -> OpenAI tool schemas (prefix,
    dispatch map, allowlist filtering)
  * LLMReceptionAgent._parse   — extract the brief JSON from a final message

The live loop itself is exercised by tools/run_pipeline.py.

Run under: module load pytorch/2.8.0
"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("openai")   # tool_loop imports SimpleLLMClient -> openai

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.tool_loop import (ToolLoopAgent, _human_answer,  # noqa: E402
                              ASK_USER_TOOL)
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


class TestAskUser:
    def test_batch_returns_sentinel(self):
        r = _human_answer("CA or WA American River?", None, interactive=False)
        assert "non-interactive" in r["answer"].lower()

    def test_no_terminal_falls_back_to_the_sentinel(self):
        """interactive=True is not enough — a batch job or a piped run has no
        terminal, and blocking forever on a read would hang the pipeline."""
        r = _human_answer("which basin?", None, interactive=True)
        assert "non-interactive" in r["answer"].lower()

    def test_ask_user_tool_schema(self):
        assert ASK_USER_TOOL["function"]["name"] == "ask_user"
        assert "question" in ASK_USER_TOOL["function"]["parameters"]["properties"]


class TestAnUnreadableFinalReply:
    """A live reception (2026-10-27) ended in a reply with no brief, and the
    study stopped on "I need some clarification:" with nothing under it."""

    def _agent(self, first, again):
        a = object.__new__(LLMReceptionAgent)
        a.system = "sys"

        class Msg:
            def __init__(self, c):
                self.content, self.tool_calls = c, None

        class Loop:
            def __init__(self):
                self.calls = []

            def run(self, system, msg):
                return {"content": first, "trace": [], "rounds": 1,
                        "messages": [{"role": "system", "content": system},
                                     {"role": "user", "content": msg}]}

            def _create(self, messages):
                self.calls.append(messages[-1]["content"])
                choice = type("C", (), {"message": Msg(again)})()
                return type("R", (), {"choices": [choice]})()

        a.loop = Loop()
        return a

    def test_it_is_asked_for_once_more(self):
        a = self._agent("Fine, I will proceed with that.",
                        '{"intent": "clarification_needed", "questions": ["Which years?"]}')
        pkg = a.process("q")
        assert pkg["route"]["action"] == "clarify"
        assert pkg["route"]["questions"] == ["Which years?"]
        assert len(a.loop.calls) == 1
        assert "could not be read as the final JSON" in a.loop.calls[0]

    def test_still_unreadable_says_so(self):
        a = self._agent("some text", "still some text")
        pkg = a.process("q")
        assert pkg["route"]["questions"] == [LLMReceptionAgent.UNREADABLE]
        assert len(a.loop.calls) == 1

    def test_a_readable_reply_is_not_asked_again(self):
        a = self._agent('{"intent": "clarification_needed", "questions": ["Q?"]}', "unused")
        a.process("q")
        assert a.loop.calls == []

    def test_a_place_given_as_a_name_becomes_a_record(self):
        """A live whole-domain question (2026-10-28) came back with
        "domain": "naches", and the gather died on dom.get before the
        design page."""
        a = self._agent(json.dumps({
            "intent": "design", "design_archetype": "whole_domain", "model": "elm",
            "domain": "naches",
            "domain_experiment": {"control": {"name": "naches", "reuse": True}}}), "unused")
        pkg = a.process("q")
        assert pkg["route"]["action"] == "design"
        assert pkg["brief"]["domain"] == {"name": "naches"}
        assert pkg["provenance"][0]["args"]["design_archetype"] == "whole_domain"

    def test_a_re_asked_brief_keeps_the_first_reply(self):
        """A REVIEW FINDING (2026-11-03): the second reply replaced the first
        in `raw`, and reception.json could not show that a re-ask made it."""
        first = "Fine, I will proceed with that. " * 20
        a = self._agent(first, '{"intent": "clarification_needed", "questions": ["Q?"]}')
        pkg = a.process("q")
        assert pkg["json_asked_again"] is True
        assert pkg["raw_first"] == first[:400] and len(pkg["raw_first"]) == 400
        (p,) = pkg["provenance"]
        assert p["first_reply"] == first[:400] and "asked for once more" in p["skipped"]

    def test_the_re_ask_record_survives_the_gather(self):
        a = self._agent("Fine, I will proceed.", json.dumps({
            "intent": "design", "design_archetype": "whole_domain", "model": "elm",
            "domain": {"name": "naches"},
            "domain_experiment": {"control": {"name": "naches", "reuse": True}}}))
        pkg = a.process("q")
        assert pkg["route"]["action"] == "design" and pkg["json_asked_again"] is True
        assert [p.get("first_reply") for p in pkg["provenance"]] == [
            "Fine, I will proceed.", None]
        assert pkg["provenance"][1]["args"]["design_archetype"] == "whole_domain"

    def test_a_brief_read_first_time_carries_no_re_ask_record(self):
        a = self._agent('{"intent": "clarification_needed", "questions": ["Q?"]}', "unused")
        pkg = a.process("q")
        assert "json_asked_again" not in pkg and "raw_first" not in pkg
        assert pkg["provenance"] == []

    def test_a_whole_domain_record_promises_no_observations(self):
        """A REVIEW FINDING (2026-11-03): the skip record said the experiment
        compares with gridded observations; since 2026-10-27 the answer is
        read from the model's output alone."""
        a = self._agent(json.dumps({
            "intent": "design", "design_archetype": "whole_domain", "model": "elm",
            "domain": {"name": "naches"},
            "domain_experiment": {"control": {"name": "naches", "reuse": True}}}), "unused")
        (p,) = a.process("q")["provenance"]
        assert "gridded observations" not in p["skipped"]
        assert "read from the model's output over the control's registered regions" \
            in p["skipped"]

    def test_a_clarification_with_no_question_says_what_to_do(self):
        r = LLMReceptionAgent._route({"intent": "clarification_needed"})
        assert r["action"] == "clarify"
        assert r["questions"] == [LLMReceptionAgent.NOTHING_TO_ASK]
        assert LLMReceptionAgent._route({"intent": "design"})["questions"] == []


class TestThePlannerPromptOnTheWholeDomain:
    """The planner's whole-domain section (review findings, 2026-11-03)."""

    def _section(self):
        text = (Path(__file__).resolve().parents[1] / "src" / "agents" / "prompts"
                / "planner.txt").read_text()
        return text[text.index("THE WHOLE-DOMAIN PATH"):
                    text.index("THE COLUMN BUDGET")]

    def test_burned_area_is_named_through_its_history_field(self):
        sec = self._section()
        assert "burned_area_km2 is not a history field: it is made from" in sec
        assert "Name FAREA_BURNED among the variables" in sec

    def test_the_caveat_is_written_from_the_words_without_numbers(self):
        sec = self._section()
        assert "`not_in_it_words`" in sec
        assert "`not_in_it`" not in sec and "`evaluation`" not in sec


def test_the_domain_doc_no_longer_sets_observations_in_the_analysis():
    """docs/ELM_DOMAIN_EXPERIMENTS.md described the analysis before
    2026-10-27 (review finding, 2026-11-03)."""
    doc = (Path(__file__).resolve().parents[1] / "docs"
           / "ELM_DOMAIN_EXPERIMENTS.md").read_text()
    assert "variables, observations and normal years" not in doc
    assert "beside the observed one" not in doc
    assert "the observed burned area per region" not in doc
    assert "set beside the model's own" not in doc
