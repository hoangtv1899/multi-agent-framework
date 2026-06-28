"""Offline tests for the dry-pipeline orchestration glue (no LLM, no MCP):

  * LLMReceptionAgent.process  — user-message assembly (request + optional
    conversation context) and brief parsing, driven by a fake tool-loop.
  * run_pipeline / run_session pure helpers — _parse_json, print_plan/show
    robustness on partial plans, context_from carry-forward, and run_planner
    prompt wiring (mocked LLM).

Run under: module load pytorch/2.8.0
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
pytest.importorskip("mcp")                     # run_pipeline imports core.mcp_manager

from agents.reception_llm import LLMReceptionAgent   # noqa: E402


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rp = _load("rp_mod", "tools/run_pipeline.py")
rs = _load("rs_mod", "tools/run_session.py")


# ── reception.process with a fake tool-loop ──────────────────────────────────
class _FakeLoop:
    def __init__(self, content):
        self._content = content
        self.tools = [{"function": {"name": "terrain__resolve_watershed"}}]
        self.seen = None

    def run(self, system, msg):
        self.seen = {"system": system, "msg": msg}
        return {"content": self._content, "trace": [{"tool": "x"}], "rounds": 3}


def _reception(content):
    rx = LLMReceptionAgent.__new__(LLMReceptionAgent)   # bypass __init__ (no LLM)
    rx.system = "SYSTEM PROMPT"
    rx.loop = _FakeLoop(content)
    return rx


def test_process_parses_brief_and_forwards_trace():
    rx = _reception('```json\n{"intent":"design","design_archetype":"site"}\n```')
    out = rx.process("study the Naches")
    assert out["brief"]["intent"] == "design"
    assert out["brief"]["design_archetype"] == "site"
    assert out["rounds"] == 3 and out["trace"] == [{"tool": "x"}]
    assert "intent" in out["raw"]                        # raw final message kept
    assert rx.loop.seen["system"] == "SYSTEM PROMPT"     # system prompt forwarded
    assert "study the Naches" in rx.loop.seen["msg"]     # request embedded


def test_process_injects_conversation_context():
    rx = _reception('{"intent":"design"}')
    rx.process("now couple PFLOTRAN", context={"prior_model": "ELM"})
    assert "CONVERSATION CONTEXT" in rx.loop.seen["msg"]
    assert "ELM" in rx.loop.seen["msg"]


def test_process_without_context_has_no_context_block():
    rx = _reception('{"intent":"design"}')
    rx.process("plain request")
    assert "CONVERSATION CONTEXT" not in rx.loop.seen["msg"]


def test_process_parse_error_on_non_json():
    rx = _reception("I could not identify a watershed.")
    assert rx.process("vague")["brief"]["intent"] == "parse_error"


def test_exposed_tools_lists_loop_tool_names():
    assert _reception('{"intent":"design"}').exposed_tools == ["terrain__resolve_watershed"]


# ── _parse_json (both entry points share the shape) ─────────────────────────
class TestParseJson:
    def test_plain(self):
        assert rp._parse_json('{"a":1}') == {"a": 1}

    def test_fenced(self):
        assert rp._parse_json('```json\n{"a":1}\n```') == {"a": 1}

    def test_prose_wrapped(self):
        assert rp._parse_json('reasoning... {"a":1} done')["a"] == 1

    def test_bad_returns_none(self):
        assert rp._parse_json("no json here") is None

    def test_session_parser_matches(self):
        assert rs._parse_json('```json\n{"a":1}\n```') == {"a": 1}


# ── plan printers must never crash on partial plans ─────────────────────────
def test_print_plan_handles_empty_and_full(capsys):
    rp.print_plan({})                                    # missing everything
    rp.print_plan({"scientific_decomposition": {"goals": ["g1"]},
                   "model_choice": {"design_archetype": "site"},
                   "sampling_plan": [{"n": 5, "group": "lowland", "reason": "wet"}],
                   "validation_design": [{"target_variable": "QDRAI",
                                          "observation_source": "USGS",
                                          "in_domain_available": True}],
                   "experiment_summary": {"total_columns": 5}})
    assert "PLAN SUMMARY" in capsys.readouterr().out


def test_show_handles_empty_and_coupling(capsys):
    rs.show(None)
    rs.show({"model_choice": {"design_archetype": "coupling"},
             "coupling_design": {"from_model": "ELM", "to_model": "PFLOTRAN",
                                 "driver": "QDRAI recharge flux"},
             "sampling_strategy": {"n_exploratory": 6, "approach": "transect"},
             "requires_capabilities": [{"capability": "pflotran_run"}]})
    out = capsys.readouterr().out
    assert "COUPLING DESIGN" in out and "ELM" in out and "PFLOTRAN" in out


# ── context_from carry-forward for multi-turn sessions ──────────────────────
def test_context_from_carries_domain_and_model():
    brief = {"domain": {"name": "Naches", "huc": "17030002", "area_km2": 3400}}
    plan = {"model_choice": {"primary_model": "ELM"},
            "sampling_strategy": {"approach": "elevation-stratified"}}
    ctx = rs.context_from("study Naches", brief, plan)
    assert ctx["prior_request"] == "study Naches"
    assert ctx["prior_model"] == "ELM"
    assert ctx["prior_domain"]["huc"] == "17030002"
    assert ctx["prior_design"] == "elevation-stratified"
    assert any("QDRAI" in o for o in ctx["prior_outputs"])


def test_context_from_tolerates_empty_plan():
    ctx = rs.context_from("req", {}, None)
    assert ctx["prior_model"] == "ELM"                   # default when no model_choice
    assert ctx["prior_domain"]["name"] is None


# ── run_planner prompt wiring (mocked LLM) ──────────────────────────────────
def test_run_planner_embeds_brief_and_returns_content(monkeypatch):
    captured = {}

    def fake_create(**kw):
        captured.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="PLAN TEXT"))])

    fake_llm = SimpleNamespace(model="m", client=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))))
    monkeypatch.setattr(rp, "SimpleLLMClient", lambda model: fake_llm)

    out = rp.run_planner({"intent": "design", "domain": {"name": "Naches"}},
                         "study the Naches", "model-x")
    assert out == "PLAN TEXT"
    user_msg = captured["messages"][-1]["content"]
    assert "study the Naches" in user_msg and "Naches" in user_msg
