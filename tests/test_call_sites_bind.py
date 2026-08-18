#!/usr/bin/env python3
"""Every call site in the coordinator must bind against the real signature.

The bug this exists for: workflow.py called
    self.reception.process(user_request=..., conversation_context=...)
while the method's parameter is named `context`. TypeError on the very first
request of an interactive session — after six MCP servers had started and the
user had typed their question.

Import checks cannot see it. A signature check that only walks self.<method>()
cannot see it either, because the call goes through an ATTRIBUTE first. This
walks both, resolving self.<agent> to the class the coordinator really builds.
"""
import ast
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _coordinator_class():
    import importlib.util as u
    spec = u.spec_from_file_location("wf", ROOT / "workflow.py")
    m = u.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.WorkflowCoordinator


def _agent_types():
    """self.<attr> → the class the coordinator actually constructs."""
    from agents.reception_llm import LLMReceptionAgent
    from agents.planner import Planner
    from agents.analysis_report_agent import AnalysisReportAgent
    return {"reception": LLMReceptionAgent,
            "planner":   Planner,
            "analyzer":  AnalysisReportAgent}


def _call_sites(path):
    """(lineno, owner_attr|None, method, n_positional, kwarg_names, has_star)"""
    tree = ast.parse(Path(path).read_text(), str(path))
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
            continue
        v = n.func.value
        star = (any(isinstance(a, ast.Starred) for a in n.args)
                or any(k.arg is None for k in n.keywords))
        kws = tuple(k.arg for k in n.keywords if k.arg)
        if isinstance(v, ast.Name) and v.id == "self":
            yield n.lineno, None, n.func.attr, len(n.args), kws, star
        elif (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
              and v.value.id == "self"):
            yield n.lineno, v.attr, n.func.attr, len(n.args), kws, star


def test_every_coordinator_call_site_binds():
    cls    = _coordinator_class()
    agents = _agent_types()
    problems = []

    for lineno, owner, method, npos, kws, star in _call_sites(ROOT / "workflow.py"):
        target_cls = cls if owner is None else agents.get(owner)
        if target_cls is None:
            continue                      # not an agent we can resolve
        fn = getattr(target_cls, method, None)
        if fn is None:
            problems.append(
                f"workflow.py:{lineno}  self."
                f"{owner + '.' if owner else ''}{method}() — "
                f"{target_cls.__name__} has no such method")
            continue
        if not callable(fn) or star:
            continue
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            continue
        try:
            sig.bind(*(["self"] + ["<v>"] * npos), **{k: "<v>" for k in kws})
        except TypeError as e:
            problems.append(
                f"workflow.py:{lineno}  self."
                f"{owner + '.' if owner else ''}{method}"
                f"({', '.join(kws)}) — {e}")

    assert not problems, "call sites that will TypeError at runtime:\n" + \
        "\n".join("  " + p for p in problems)


def test_reception_context_is_bounded():
    """conversation_context is dumped verbatim into reception's prompt, so
    what goes in is a token budget. last_analysis and last_plan are held in
    FULL and must not be forwarded."""
    import json
    cls = _coordinator_class()
    co  = object.__new__(cls)
    co.conversation_context = {}
    assert co._reception_context() is None, \
        "the first turn must emit no CONVERSATION CONTEXT block at all"

    co.conversation_context = {
        "last_run_dir": "/x/elm_run_1",
        "last_focus":   "partition precipitation",
        "last_plan":    {"sampling": {"n_columns": 19}, "archetype": "site",
                         "bulk": "y" * 50_000},
        "last_analysis": {"report": "z" * 200_000},
    }
    ctx = co._reception_context()
    blob = json.dumps(ctx)
    assert len(blob) < 2000, f"reception context ballooned to {len(blob)} chars"
    assert "z" * 100 not in blob, "the full analysis leaked into the prompt"
    assert "y" * 100 not in blob, "the full plan leaked into the prompt"
    assert ctx["prior_run_dir"] == "/x/elm_run_1"


# ─────────────────────────────────────────────────────────────────────────────
# Reception's gather calls bind against data_gather (2026-08-18)
# ─────────────────────────────────────────────────────────────────────────────
# The bug this exists for: an edit that retired gather_soil sliced
# data_gather.py from `def gather_soil(` to `def gather_subsurface(` and took
# gather_precipitation with it — the function defined between them. Reception
# went on calling `gather.gather_precipitation(...)` at line 246, the suite
# passed, the commit landed, and every site reception since would have died
# with AttributeError after the LLM rounds and the DEM fetch. Nothing walked
# the `gather.<name>` attribute accesses against the module. This does.
def _gather_names_called_by_reception():
    src = (ROOT / "src" / "agents" / "reception_llm.py").read_text()
    tree = ast.parse(src)
    out = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "gather"):
            out.add(n.attr)
    return out


def test_every_gather_call_in_reception_exists_in_data_gather():
    from core import data_gather
    names = _gather_names_called_by_reception()
    assert names, "reception_llm.py no longer calls any gather.* — check the alias"
    missing = sorted(n for n in names if not callable(getattr(data_gather, n, None)))
    assert not missing, (
        f"reception_llm.py calls gather.{missing} but data_gather has no such "
        f"function — the site path will raise AttributeError after the LLM "
        f"rounds. Restore it or stop calling it.")


def test_gather_calls_in_reception_bind():
    """The arguments reception passes must match the gather signature."""
    from core import data_gather
    src = (ROOT / "src" / "agents" / "reception_llm.py").read_text()
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "gather"):
            continue
        fn = getattr(data_gather, n.func.attr, None)
        if fn is None:
            continue                            # the test above reports it
        kws = {k.arg: None for k in n.keywords if k.arg}
        try:
            inspect.signature(fn).bind(*([None] * len(n.args)), **kws)
        except TypeError as e:
            pytest.fail(f"reception_llm.py:{n.lineno} gather.{n.func.attr}"
                        f"({len(n.args)} positional, {sorted(kws)}) does not "
                        f"bind: {e}")
