"""The list of tests set aside (tests/parked_tests.json), held to its rules.

A plain pytest run skips the tests in that list, so the list could hide a
defect if it grew without a reason, or if it named a test of code the
whole-domain workflow uses. These tests keep it honest:

- every entry has a known class and a reason, and names a test that exists;
- no test of a whole-domain module is in it (those are repaired, not set
  aside);
- the whole-domain modules import none of the modules that only set-aside
  paths use.

WHAT THE LAST CHECK RESTS ON. On 2026-11-10 every function the whole-domain
path called was recorded (Reception, the Planner and the design check in a
live study; the Experiment Manager and the Analyzer through their own tests),
and so was every function each failing test reached. The modules below as
SET_ASIDE_MODULES are the ones only the failing tests reached. Three files
serve both routes and are left out of the import check for that reason:
workflow.py, core/exp_manager_base.py and elm_exp_manager.py name set-aside
code on their single-column branches, and core/resumable.py names every
manager in its dispatch table.
"""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIST = json.loads((ROOT / "tests" / "parked_tests.json").read_text())
GROUPS = LIST["groups"]
ENTRIES = [(t, g["class"], g["reason"]) for g in GROUPS for t in g["tests"]] + \
          [(f, v["class"], v["reason"]) for f, v in LIST["files"].items()]

# Test files of the whole-domain path. No test in them may be set aside.
WHOLE_DOMAIN_TESTS = (
    "test_domain_", "test_analysis_", "test_analyzer_g", "test_analyzer_phase_close",
    "test_audit_trace_rule", "test_claim_evidence", "test_design_", "test_reception_",
    "test_report_views", "test_figure_bars", "test_numbers_check", "test_suite_numbers_check",
    "test_study_record", "test_resumable", "test_run_gate", "test_gridded_", "test_regions",
    "test_control_setup", "test_tool_loop", "test_parked_tests")

# Modules only set-aside paths load (the traces of 2026-11-10), by import name.
SET_ASIDE_MODULES = {
    "pflotran_exp_manager", "lambda_pflotran_exp_manager", "analyze_agentic",
    "analyze_run", "column_metrics", "columns_to_plan", "make_finidat_subset",
    "make_warmstart", "step2_derive", "limitations", "elm_results_analyzer"}

# The whole-domain modules: the server's whole-domain code and its readers,
# the Analyzer's steps, and the agents in front of them.
_SERVER = ROOT / "mcp" / "elm-mcp" / "src"
_ANALYSIS = ROOT / "src" / "agents" / "analysis"
WHOLE_DOMAIN_MODULES = sorted(
    list(_SERVER.glob("domain_*.py"))
    + [_SERVER / n for n in ("gridded.py", "regions.py", "run_gate.py")]
    + [p for p in _ANALYSIS.glob("*.py") if p.stem not in SET_ASIDE_MODULES]
    + [ROOT / "src" / "agents" / n for n in (
        "reception_llm.py", "planner.py", "design_review.py", "study_record.py",
        "analyzer.py", "tool_loop.py", "llm_agent.py")])


def _names(path: Path):
    """(classes, functions, methods by class) a test file defines."""
    tree = ast.parse(path.read_text(), str(path))
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    classes = {n.name: {m.name for m in n.body if isinstance(m, ast.FunctionDef)}
               for n in tree.body if isinstance(n, ast.ClassDef)}
    return funcs, classes


def _imported(path: Path):
    """Every module name a file imports, at any depth, by its last part; and
    the names handed as text to the two loaders this code base uses."""
    out = set()
    for n in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[-1] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                out.add(n.module.split(".")[-1])
            out |= {a.name for a in n.names}
        elif (isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant)
              and isinstance(n.args[0].value, str)):
            fn = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            if fn in ("_sibling_module", "import_module", "__import__"):
                out.add(n.args[0].value.split(".")[-1])
    return out


class TestTheListIsWellFormed:

    def test_every_entry_has_a_known_class_and_a_reason(self):
        assert ENTRIES
        for node, cls, reason in ENTRIES:
            assert cls in LIST["classes"], f"{node}: class {cls!r} is not one of the list's"
            assert len(reason.split()) >= 4, f"{node}: no reason a reader could act on"

    def test_no_test_is_listed_twice(self):
        nodes = [n for n, _c, _r in ENTRIES]
        assert len(nodes) == len(set(nodes))

    @pytest.mark.parametrize("node", [n for n, _c, _r in ENTRIES])
    def test_every_entry_names_a_test_that_exists(self, node):
        parts = node.split("::")
        path = ROOT / parts[0]
        assert path.is_file(), f"{parts[0]} is gone: drop it from the list"
        if len(parts) == 1:
            return
        funcs, classes = _names(path)
        name = parts[-1].split("[")[0]
        if len(parts) == 2:
            assert name in funcs, f"{node}: no such test function"
        else:
            assert name in classes.get(parts[1], set()), f"{node}: no such test"


class TestTheListCannotHideAWholeDomainDefect:

    def test_no_whole_domain_test_is_set_aside(self):
        bad = [n for n, _c, _r in ENTRIES
               if Path(n.split("::")[0]).name.startswith(WHOLE_DOMAIN_TESTS)]
        assert not bad, f"a test of the whole-domain path is repaired, not set aside: {bad}"

    def test_the_whole_domain_modules_exist(self):
        missing = [str(p.relative_to(ROOT)) for p in WHOLE_DOMAIN_MODULES if not p.is_file()]
        assert not missing and len(WHOLE_DOMAIN_MODULES) > 30, missing

    @pytest.mark.parametrize("path", WHOLE_DOMAIN_MODULES,
                             ids=lambda p: str(p.relative_to(ROOT)))
    def test_a_whole_domain_module_imports_no_set_aside_module(self, path):
        hit = _imported(path) & SET_ASIDE_MODULES
        assert not hit, (f"{path.relative_to(ROOT)} imports {sorted(hit)}, which only "
                         f"set-aside paths used on 2026-11-10: the tests of that code must "
                         f"come off tests/parked_tests.json and pass first")


class TestTheSwitch:

    def test_a_plain_run_skips_a_listed_test_and_says_why(self):
        """The reason a skipped test shows is the list's own, so a reader of
        the run's summary sees why it did not run."""
        import importlib.util as U
        spec = U.spec_from_file_location("root_conftest", ROOT / "conftest.py")
        C = U.module_from_spec(spec)
        spec.loader.exec_module(C)
        parked = C.parked_tests()
        node, cls, reason = ENTRIES[0]
        assert parked[node] == (cls, reason)
        assert set(parked) == {n for n, _c, _r in ENTRIES}
