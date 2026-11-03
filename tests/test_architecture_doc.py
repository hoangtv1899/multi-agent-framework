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


# ─────────────────────────────────────────────────────────────────────────────
# REVIEW FINDINGS, HELD (2026-11-03). Each test below failed on the first
# version of the generator and names the case a reader would have been misled by.
# ─────────────────────────────────────────────────────────────────────────────
def _row(text, first_cell):
    return next((l for l in text.splitlines() if l.startswith(f"| `{first_cell}` |")), "")


def _stage(text, number):
    lines = text.split("## 6.", 1)[1].splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(f"{number}. **"))
    out = []
    for l in lines[start + 1:]:
        if not l.startswith("   -"):
            break
        out.append(l)
    return "\n".join(out)


def test_a_tool_held_in_a_constant_is_credited_to_its_user(text):
    """sweep_menu.fetch calls call_tool_json(TOOL, {}) with TOOL a module
    constant; Reception's allowlist is a constant its __init__ uses."""
    assert "`src/core/sweep_menu.py::fetch`" in _row(text, "describe_conceptual_factors")
    assert "src/agents/reception_llm.py::LLMReceptionAgent.__init__" in _row(text, "get_elevation")
    assert "src/agents/reception_llm.py::LLMReceptionAgent.__init__" in _row(text, "resolve_watershed")


def test_a_prompt_naming_server__tool_counts(text):
    """reception_agentic.txt writes terrain__get_elevation; `_` is a word
    character, so a plain word boundary never matched it."""
    assert "reception_agentic.txt (prompt)" in _row(text, "get_elevation")


def test_a_description_naming_a_tool_is_not_a_caller(text):
    """FIELD_SEMANTICS says a field comes `from` build_conceptual_columns:
    a description, not a call."""
    assert "FIELD_SEMANTICS" not in _row(text, "build_conceptual_columns")


def test_reception_stage_names_its_tools(text):
    assert "get_elevation" in _stage(text, 1)


def test_records_through_a_module_loaded_by_path(text):
    """DC = self._sibling_module("domain_case"); rd / DC.CHAIN_FILE."""
    s7 = _stage(text, 7)
    assert "reads: `chain.json`" in s7
    assert "`execution_report.txt`" in s7.split("writes:", 1)[1]


def test_records_through_a_loop_and_an_imported_alias(text):
    """step 4 reads RUN_SUMMARY.json in `for name in (...)`, and
    DOMAIN_REPORT.md as `from ... import REPORT_MD as DOMAIN_MD`."""
    reads = _stage(text, 13).split("reads:", 1)[1].split("\n", 1)[0]
    assert "`RUN_SUMMARY.json`" in reads
    assert "`DOMAIN_REPORT.md`" in reads


def test_carried_reception_files_are_read_and_copied(gen, tree):
    r = gen.records_of(tree, tree.funcs["src/agents/reception_llm.py::_carry_prior_forward"])
    assert "conus2_subsurface.json" in r["reads"]
    assert "conus2_subsurface.json" in r["writes"]


def test_the_mcp_sdk_is_a_third_party_package(text):
    """The top-level mcp/ directory holds the servers; `import mcp` is the
    SDK, which requirements.txt lists first."""
    row = _row(text.split("## 5.", 1)[1], "mcp")
    assert row.startswith("| `mcp` | `mcp` |"), row
    assert "src/core/mcp_client.py" in row


def test_optional_requirements_are_listed_as_optional(text):
    libs = text.split("## 5.", 1)[1]
    assert _row(libs, "pptx").startswith("| `pptx` | optional: `python-pptx` |")
    assert _row(libs, "rasterio").startswith("| `rasterio` | optional: `rasterio` |")


def test_reply_limit_is_given_per_model_call(text):
    """Reception's tool loop sends its own 8192, not SimpleLLMClient's 16384."""
    assert "on every client" not in text
    assert "`src/agents/tool_loop.py::ToolLoopAgent._create`: 8192" in text
    assert "`src/agents/llm_agent.py::SimpleLLMClient.ask`: 16384" in text


def test_a_server_script_driving_an_agent_is_named(text):
    sec = text.split("### `src/agents/analyzer_agent.py`", 1)[1].split("\n### ", 1)[0]
    assert "Called from (code under `mcp/`): `mcp/elm-mcp/scripts/analyze_agentic.py::main`" in sec


def test_intro_names_everything_hand_written(text):
    intro = text.split("## 1.", 1)[0]
    assert "the `FLOW` list in section 6" in intro
    entries = [l for l in text.splitlines() if l.startswith("Entry:")]
    assert entries and not any("installed" in l for l in entries)
