"""LLM tier — real calls to the PNNL OpenAI-compatible endpoint.

Opt-in: skipped unless `pytest --runllm` (and PNNL_API_KEY set). Verifies the
endpoint round-trips, that a `tools=` request succeeds end-to-end for the
Bedrock-routed Claude model (the gateway 400s on tool round-trips without it),
and that the full agentic reception yields a parseable brief.

    module load pytorch/2.8.0 && pytest tests/test_live_llm.py --runllm [--runlive]
"""
import os
import sys

import pytest

sys.path.insert(0, "src")
pytest.importorskip("openai")

if not os.getenv("PNNL_API_KEY"):
    pytest.skip("PNNL_API_KEY not set", allow_module_level=True)

from agents.llm_agent import SimpleLLMClient      # noqa: E402

pytestmark = pytest.mark.llm
MODEL = "claude-opus-4-8-project"                  # project default (Bedrock-routed Claude)


def test_endpoint_roundtrip_returns_text():
    llm = SimpleLLMClient(model=MODEL)
    resp = llm.client.chat.completions.create(
        model=llm.model, max_tokens=64,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}])
    assert (resp.choices[0].message.content or "").strip()


def test_toolcall_request_does_not_400():
    """Confirms a tools= request succeeds for the project model — the gateway
    rejects Bedrock/Claude tool round-trips when tools= is omitted."""
    tool = {"type": "function", "function": {
        "name": "get_elevation", "description": "elevation at a point",
        "parameters": {"type": "object",
                       "properties": {"lat": {"type": "number"},
                                      "lon": {"type": "number"}},
                       "required": ["lat", "lon"]}}}
    llm = SimpleLLMClient(model=MODEL)
    resp = llm.client.chat.completions.create(
        model=llm.model, max_tokens=256, tools=[tool], tool_choice="auto",
        messages=[{"role": "user",
                   "content": "Elevation at lat 46.75, lon -120.70? Use the tool."}])
    msg = resp.choices[0].message
    assert msg.tool_calls or (msg.content or "").strip()      # responded; no 400


@pytest.mark.live          # also drives the live MCP servers
def test_reception_produces_parseable_brief():
    pytest.importorskip("mcp")
    from core.mcp_manager import MCPManager
    from agents.reception_llm import LLMReceptionAgent
    clients = MCPManager("mcp_config.json").get_all_clients()
    rx = LLMReceptionAgent(MODEL, clients, verbose=False, max_rounds=8)
    out = rx.process("Explore recharge vs runoff partitioning in the Naches "
                     "sub-watershed (HUC8 17030002) with ELM.")
    assert out["brief"].get("intent") in {"design", "clarification_needed",
                                          "analyze_existing"}
    assert out["rounds"] >= 1
