"""docs/ARCHITECTURE.md must be what tools/make_architecture_doc.py writes now.

THE DOCUMENT IS GENERATED, SO IT CAN BE CHECKED (2026-11-03). It says which
components call a language model and with which model id, which MCP tools
exist and who calls them, where framework code loads a server's module without
MCP, which libraries are imported, and the order of a whole-domain study. All
of it is read from the source, so a change to the code that the document does
not reflect is a stale document, and this test fails naming the command that
refreshes it, the way tests/test_elm_vendor_sync.py does for the vendored files.

The other tests hold the generator to the principles the document exists to
show: the renderers reach no language model, and no key reaches the page.
"""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "tools" / "make_architecture_doc.py"
DOC = ROOT / "docs" / "ARCHITECTURE.md"
REFRESH = "python tools/make_architecture_doc.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("make_architecture_doc", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tree(gen):
    return gen.Tree()


@pytest.fixture(scope="module")
def text(gen):
    return gen.build()


def test_committed_document_is_current(text):
    assert DOC.is_file(), f"{DOC.relative_to(ROOT)} is missing. Write it with:\n    {REFRESH}"
    assert DOC.read_text() == text, (
        f"{DOC.relative_to(ROOT)} is out of date with the code it describes.\n"
        f"Refresh it with:\n    {REFRESH}\nand commit the result.")


def test_check_flag_reports_a_stale_copy(gen, tmp_path, text):
    stale = tmp_path / "ARCHITECTURE.md"
    stale.write_text(text.replace("## 6.", "## six.", 1))
    assert gen.main(["--check", "--out", str(stale)]) == 1
    stale.write_text(text)
    assert gen.main(["--check", "--out", str(stale)]) == 0


def test_no_em_dash(text):
    assert "—" not in text


def test_no_key_value_on_the_page(text):
    """The page names the variable a key is read from, never its value."""
    for var in ("PNNL_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(var)
        # a bare boolean, so a failure cannot print the value it found
        found = bool(v) and len(v) > 8 and v in text
        assert not found, f"the value of {var} is in the document"


def test_every_flow_function_exists(gen, tree):
    missing = [a for _, anchors in gen.FLOW for a in anchors if a not in tree.funcs]
    assert not missing, f"FLOW names functions that no longer exist: {missing}"


@pytest.mark.parametrize("key", [
    "src/agents/analysis/domain_report.py::build",
    "src/agents/design_review.py::write_review",
    "src/agents/analysis/step4_report.py::build",
    "mcp/elm-mcp/src/elm_exp_manager.py::ELMExpManager._regional",
    "mcp/elm-mcp/src/elm_exp_manager.py::ELMExpManager._package_experiment",
])
def test_renderers_and_numbers_reach_no_language_model(tree, key):
    """Renderers quote or count records, and the numbers are the server's
    code: no path from these functions reaches a model call."""
    assert key in tree.funcs, f"{key} no longer exists; update this test"
    assert key not in tree.llm_reach()


@pytest.mark.parametrize("key", [
    "src/agents/planner.py::Planner.plan",
    "src/agents/reception_llm.py::LLMReceptionAgent.process",
    "src/agents/analysis/step3_interpret.py::investigate_and_interpret",
])
def test_the_agents_are_found_to_reach_a_model(tree, key):
    """The call-graph walk sees the language-model agents it should."""
    assert key in tree.llm_reach()


def test_every_registered_server_in_the_repo_lists_its_tools(gen, tree):
    """Both tool forms are read: FastMCP's @mcp.tool and the low-level
    Server API's types.Tool list (daymet)."""
    for name, spec in gen.registry().items():
        rel = gen._entry_rel(spec)
        if rel is None:
            continue
        assert gen.served_tools(tree, rel), f"no tools read from {rel} ({name})"


def test_record_direction(gen, tree):
    """strategy.json is written where the coordinator saves the plan, and
    step 0 reads the three boundary records through run_layout."""
    plan = gen.records_of(tree, tree.funcs["workflow.py::WorkflowCoordinator._plan"])
    assert "strategy.json" in plan["writes"]
    load = gen.records_of(tree, tree.funcs["src/agents/analysis/step0_context.py::load"])
    assert {"reception.json", "strategy.json", "experiment.json"} <= load["reads"]
