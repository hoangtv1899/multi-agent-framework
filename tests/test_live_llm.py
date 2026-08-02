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


# ─────────────────────────────────────────────────────────────────────
# The resume route, against the real model
# ─────────────────────────────────────────────────────────────────────
# Everything else about this route is tested with the LLM replaced by a
# fixture, which proves the plumbing and proves nothing about whether the
# model actually picks `resume` for a sentence a person would type. These do
# that, and only that.
#
# Two runs in the context, distinguishable ONLY by their request text — their
# directory names are timestamps a day apart. That is the real situation: if
# the model cannot use the request field it cannot tell them apart at all.

GUNNISON = "/qfs/.../workflow_outputs/elm_run_20260801_132630"
NACHES   = "/qfs/.../workflow_outputs/elm_run_20260731_094512"

_RESUMABLE = [
    {"run_dir": GUNNISON, "model": "elm", "stage": "run",
     "status": "run: job 770696 is RUNNING", "age": "14 min",
     "request": "Quantify how precipitation partitions into runoff and "
                "recharge across the Upper Gunnison watershed in Colorado "
                "(HUC8 14020002) for 2020"},
    {"run_dir": NACHES, "model": "elm", "stage": "prepare",
     "status": "prepare: job 880014 is PENDING", "age": "1 d",
     "request": "Explore recharge vs runoff partitioning in the Naches "
                "sub-watershed (HUC8 17030002)"},
]


def _route(request: str, context=None, clients=None):
    from agents.reception_llm import LLMReceptionAgent
    rx = LLMReceptionAgent(MODEL, clients or {}, verbose=False, max_rounds=4)
    out = rx.process(request, context=context)
    return out.get("route") or {}


@pytest.mark.llm
def test_asking_about_a_running_job_routes_to_resume():
    r = _route("is my Gunnison run done yet?",
               {"resumable_runs": _RESUMABLE})
    assert r.get("action") == "resume", \
        f"expected resume, got {r.get('action')} (llm_intent={r.get('llm_intent')})"


@pytest.mark.llm
def test_it_picks_the_run_the_request_text_names():
    """Directory names are timestamps. The `request` field is the only thing
    distinguishing two studies of different watersheds.

    Asserted UNCONDITIONALLY. Written first as `if it resumed, check the dir`,
    which would have gone quietly green the day the model stopped routing to
    resume at all — the silent-success shape. Verified live before tightening:
    the model returns the Gunnison directory verbatim.
    """
    r = _route("check on that Gunnison job for me",
               {"resumable_runs": _RESUMABLE})
    assert r.get("action") == "resume", f"routed to {r.get('action')}"
    assert r.get("prior_run_dir") == GUNNISON, \
        f"picked the wrong study: {r.get('prior_run_dir')!r}"


@pytest.mark.llm
def test_the_run_dir_is_copied_verbatim_not_constructed():
    """A paraphrased or completed path is what the coordinator's allowlist
    exists to reject — but it should not have to."""
    r = _route("carry on with the Naches run",
               {"resumable_runs": _RESUMABLE})
    assert r.get("action") == "resume", f"routed to {r.get('action')}"
    assert r.get("prior_run_dir") == NACHES, \
        f"run_dir was invented rather than selected: {r.get('prior_run_dir')!r}"


# Two runs of the SAME watershed on the same day — the case that motivated the
# allowlist. Their request text is identical; only the age differs.
_AMBIGUOUS = [
    {**_RESUMABLE[0], "run_dir": GUNNISON, "age": "14 min"},
    {**_RESUMABLE[0],
     "run_dir": "/qfs/.../workflow_outputs/elm_run_20260801_081500",
     "age": "6 h", "status": "run: job 770610 is RUNNING"},
]


@pytest.mark.llm
def test_two_identical_runs_are_not_guessed_between():
    """The failure the whole design guards against: picking one of two
    indistinguishable studies and reporting it as the one that was asked
    about. Verified live — the model returns run_dir null and the coordinator
    then lists both.
    """
    r = _route("is my Gunnison run done?", {"resumable_runs": _AMBIGUOUS})
    assert not r.get("prior_run_dir"), (
        f"guessed between two runs with identical request text: "
        f"{r.get('prior_run_dir')!r}")


@pytest.mark.llm
def test_an_extra_detail_does_disambiguate_them():
    """Declining to guess must not mean refusing to choose when the user has
    actually said which — otherwise the route is useless on a busy directory."""
    r = _route("resume the one from this morning",
               {"resumable_runs": _AMBIGUOUS})
    assert r.get("prior_run_dir") == _AMBIGUOUS[1]["run_dir"], \
        f"the older run was the one described: {r.get('prior_run_dir')!r}"


@pytest.mark.llm
def test_a_new_study_is_not_hijacked_by_the_resume_route():
    """The context always carries resumable_runs once anything is outstanding.
    A fresh design request must still route to design."""
    r = _route("Design a new 12-column ELM study of the Yakima basin "
               "(HUC8 17030003) for water year 2019.",
               {"resumable_runs": _RESUMABLE})
    assert r.get("action") != "resume", \
        "a new study was routed to resume because runs were outstanding"


@pytest.mark.llm
def test_nothing_outstanding_means_no_resume():
    """With an empty list there is nothing to continue, whatever the phrasing
    suggests."""
    r = _route("is my run done yet?", {"resumable_runs": []})
    assert r.get("action") != "resume" or not r.get("prior_run_dir"), \
        "resumed against an empty list"
