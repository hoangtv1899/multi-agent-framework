"""The front half for a whole-domain experiment: reception's two menu tools and
its gather, the planner's rulebook, the gate's period rule, the review page.

    pytest tests/test_domain_front.py -v

No LLM is called: the loops are fakes, and what is tested is what the
framework does around them (docs/ELM_DOMAIN_EXPERIMENTS.md, step 4).
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "mcp" / "elm-mcp" / "main.py"

CONTROLS = {"registry": "/r/domain_controls.json", "controls": [{
    "name": "naches", "description": "the Naches ELM-BGC control",
    "reproduced": {"years": [2015, 2016], "how": "bit for bit"},
    "regions": {"watershed": "the basin above the gauge"},
    "facts": {"start_year": 2015, "output_years": [2015, 2021],
              "hours_per_model_year": 0.79,
              "burned_area_km2": {"model": {"2021": 132.4}, "observed": {"2021": 343.8}}}}]}


class FakeServer:
    def __init__(self, tools, answers):
        self.tools, self.answers, self.calls = tools, answers, []

    def list_tools(self):
        return list(self.tools)

    def call_tool_json(self, tool, args):
        self.calls.append(tool)
        a = self.answers[tool]
        if isinstance(a, Exception):
            raise a
        return a


def _changes():
    import domain_changes as CH
    return CH.declare()


# ── reception's two menu tools ───────────────────────────────────────────────
class TestTheMenuTools:

    def test_the_servers_json_verbatim(self):
        from core.model_servers import describe_domain
        s = FakeServer(["describe_domain_controls", "describe_domain_changes"],
                       {"describe_domain_controls": CONTROLS,
                        "describe_domain_changes": _changes()})
        got = describe_domain({"elm": s}, "elm", "controls")
        assert got == {"server": "elm", "offers_whole_domain_experiments": True,
                       "controls": CONTROLS}
        assert describe_domain({"elm": s}, "elm", "changes")["changes"] == _changes()

    def test_a_server_without_the_tool_is_told_so_without_a_call(self):
        from core.model_servers import describe_domain
        s = FakeServer(["describe_pflotran_capabilities"], {})
        got = describe_domain({"pflotran": s}, "pflotran", "changes")
        assert got["offers_whole_domain_experiments"] is False and s.calls == []
        assert "Do NOT name controls, changes" in got["note"]

    def test_a_server_that_fails_says_why(self):
        from core.model_servers import describe_domain
        s = FakeServer(["describe_domain_controls"],
                       {"describe_domain_controls": RuntimeError("timed out")})
        got = describe_domain({"elm": s}, "elm", "controls")
        assert got["offers_whole_domain_experiments"] is False
        assert got["why"] == "RuntimeError: timed out"
        bad = FakeServer(["describe_domain_controls"],
                         {"describe_domain_controls": {"ok": False, "error": "no registry"}})
        assert describe_domain({"elm": bad}, "elm", "controls")["why"] == "no registry"
        assert "error" in describe_domain({}, "elm", "controls")

    def test_reception_is_offered_both(self):
        from core.model_servers import model_tools
        tools = model_tools({})
        assert {"describe_domain_controls", "describe_domain_changes"} <= set(tools)
        for n in ("describe_domain_controls", "describe_domain_changes"):
            assert tools[n]["schema"]["function"]["name"] == n


# ── reception's gather ───────────────────────────────────────────────────────
class _Loop:
    def __init__(self, brief):
        self.brief = brief

    def run(self, system, msg):
        return {"content": json.dumps(self.brief), "trace": [], "rounds": 1}


def test_a_whole_domain_brief_fetches_nothing(monkeypatch):
    pytest.importorskip("openai")
    import agents.reception_llm as R
    import core.data_gather as gather

    def refuse(*a, **k):
        raise AssertionError("a whole-domain reception fetched something")
    for name in dir(gather):
        if name.startswith("gather_") or name.startswith("fetch_"):
            monkeypatch.setattr(gather, name, refuse)
    brief = {"intent": "design", "design_archetype": "whole_domain", "model": "elm",
             "domain": {"name": "naches"},
             "domain_experiment": {"control": {"name": "naches", "reuse": True},
                                   "runs": [], "years": [2015, 2016]},
             "run_settings": {"resolved_period": {"yr_start": 2015, "yr_end": 2016}}}
    agent = R.LLMReceptionAgent.__new__(R.LLMReceptionAgent)
    agent.loop, agent.system, agent._clients = _Loop(brief), "", {}
    pkg = agent.process("how did the Naches behave after the 2015 drought?")
    assert pkg["route"]["action"] == "design"
    assert pkg["observations"] == {} and pkg["grid"] == {}
    (p,) = pkg["provenance"]
    assert p["args"] == {"design_archetype": "whole_domain", "control": "naches"}
    assert p["skipped"].startswith("no DEM grid, rain, water table or station observations")


# ── the planner's rulebook ───────────────────────────────────────────────────
class _LLM:
    def __init__(self):
        self.system = None

    def ask(self, messages, system_message=None):
        self.system = system_message
        return json.dumps({"archetype": "whole_domain"})


def test_the_rulebook_reaches_the_planner():
    pytest.importorskip("openai")
    from agents.planner import Planner
    p = Planner.__new__(Planner)
    p.llm, p.system, p.pinning, p.constraints = _LLM(), "PROMPT", None, None
    p.domain_rulebook = {"controls": CONTROLS, "changes": _changes()}
    p.plan({"brief": {"design_archetype": "whole_domain"}})
    assert "WHOLE-DOMAIN RULEBOOK: the prepared controls and the change list" in p.llm.system
    assert '"fire_off"' in p.llm.system and '"naches"' in p.llm.system
    p.domain_rulebook = None
    p.plan({"brief": {"design_archetype": "site"}})
    assert "WHOLE-DOMAIN RULEBOOK" not in p.llm.system


def test_the_rulebook_travels_with_the_plan(tmp_path, monkeypatch):
    pytest.importorskip("openai")
    import workflow as W
    s = FakeServer(["describe_domain_controls", "describe_domain_changes"],
                   {"describe_domain_controls": CONTROLS,
                    "describe_domain_changes": _changes()})
    co = W.WorkflowCoordinator.__new__(W.WorkflowCoordinator)
    co.mcp_clients, co.model = {"elm": s}, "elm"
    monkeypatch.setattr(co, "_pinning_block", lambda c, m: None, raising=False)
    monkeypatch.setattr(co, "_constraints_block", lambda c, m: None, raising=False)

    class P:
        pinning = constraints = domain_rulebook = None

        def plan(self, result):
            self.seen = self.domain_rulebook
            return {"archetype": "whole_domain"}
    co.planner = P()
    plan = co._plan({"brief": {"design_archetype": "whole_domain"}}, tmp_path)
    assert plan["domain_rulebook"] == {"controls": CONTROLS, "changes": _changes()}
    assert co.planner.seen == plan["domain_rulebook"]
    assert json.loads((tmp_path / "strategy.json").read_text())["domain_rulebook"] == \
        plan["domain_rulebook"]
    s.calls.clear()
    site = co._plan({"brief": {"design_archetype": "site"}}, tmp_path)
    assert "domain_rulebook" not in site and s.calls == []


# ── the gate ─────────────────────────────────────────────────────────────────
def test_a_whole_domain_study_needs_no_basin_and_carries_its_own_years():
    from core.strategy_check import check
    rec = {"brief": {"domain": {"name": "naches"}, "run_settings": {}},
           "grid": {}, "observations": {}}
    rep, _ = check(rec, {"archetype": "whole_domain", "domain_experiment": {
        "control": {"name": "naches", "reuse": True}, "runs": [], "years": [2015, 2016]}})
    assert rep["stop"] == []
    rep, _ = check(rec, {"archetype": "whole_domain", "domain_experiment": {
        "control": {"name": "naches", "reuse": True}, "runs": []}})
    assert any("no simulation period" in s for s in rep["stop"])


# ── the review page ──────────────────────────────────────────────────────────
def test_the_review_page_of_an_experiment():
    from agents.design_review import render
    brief = {"design_archetype": "whole_domain", "user_request": "what does the fire do?",
             "domain": {"name": "naches"},
             "run_settings": {"resolved_period": {"yr_start": 2015, "yr_end": 2016,
                                                  "source": "user"}}}
    strategy = {"archetype": "whole_domain",
                "domain_experiment": {
                    "control": {"name": "naches", "reuse": True},
                    "runs": [{"name": "nofire", "changes": [{"id": "fire_off"}]}],
                    "years": [2015, 2016],
                    "analysis": {"regions": [{"name": "watershed", "why": "the gauge's area"}],
                                 "normal_years": []},
                    "caveats": ["the model's own fire, not the 2021 fire"]},
                "domain_rulebook": {"controls": CONTROLS, "changes": _changes()},
                "feasibility": {"verdict": "full", "why": "every change is available"}}
    page = render({"brief": brief}, strategy, run_name="r")
    assert "whole-domain experiment" in page
    assert "- the control: **naches**; reused, so its existing output is compared and " \
           "nothing is spent on it; it is the Naches ELM-BGC control" in page
    assert "each run is built as its own case and run as a chain of one-year jobs" in page
    assert "builds real columns" not in page
    assert "- its output covers 2015–2021, from the saved state of 2015" in page
    assert "- run **nofire**: the control with fire_off" in page
    assert re.search(r"fire_off means the difference from the control is what the model's own "
                     r"fire does.*; it is NOT the effect of an observed fire", page)
    assert "- 2 model year(s), 2015–2016, for 1 run(s) to execute: about 1.6 node-hours" in page
    assert "- reported over: watershed (the gauge's area)" in page
    assert "- caveat: the model's own fire, not the 2021 fire" in page
    assert len(page.splitlines()) <= 60
    alone = dict(strategy, domain_experiment=dict(strategy["domain_experiment"], runs=[]))
    page = render({"brief": brief}, alone, run_name="r")
    assert "no run differs from the control: its output is read as it is, at no cost" in page


# ── what the prompts and the server say ──────────────────────────────────────
def test_the_prompts_name_no_change_and_no_control():
    """What can change, and which controls exist, is the server's to say: a
    prompt that names one is a capability claim nobody checks."""
    ids = [c["id"] for c in _changes()["changes"]]
    for name in ("reception_agentic.txt", "planner.txt"):
        text = (ROOT / "src" / "agents" / "prompts" / name).read_text()
        assert "whole_domain" in text
        assert not [i for i in ids if re.search(rf"\b{i}\b", text)], name
    rec = (ROOT / "src" / "agents" / "prompts" / "reception_agentic.txt").read_text()
    assert "STEP 2c: SETTLE THE EXPERIMENT" in rec and '"domain_experiment": {' in rec
    pl = (ROOT / "src" / "agents" / "prompts" / "planner.txt").read_text()
    assert "THE WHOLE-DOMAIN PATH: carry the experiment, choose the analysis" in pl


def test_the_server_says_columns_have_no_carbon_cycle():
    import importlib.util
    spec = importlib.util.spec_from_file_location("elm_mcp_front", SERVER)
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    fixed = json.loads(srv.describe_elm_capabilities())["constraints"]["fixed"]
    sp = [f for f in fixed if f["what"] == "a carbon and nitrogen cycle in sampled columns"]
    assert sp and "ELM%SP" in sp[0]["why"] and "describe_domain_controls" in \
        sp[0]["consequence"]
