"""No class may define the same method name twice.

Python accepts a duplicate `def` in a class body silently: the second one
wins and the first is simply gone. Nothing warns, and the failure surfaces
later as an argument-count error from a call site that looks correct.

This is not hypothetical. Renaming the pipeline's `_build` stage to
`_build_inputs` (commit 6ec9bb8) collided with ELM's existing
`_build_inputs()` — the per-column surface/domain pre-build. Two `def`s,
one class, no error; `self._build_inputs(config)` inside the stage started
calling the stage itself. The stage is now `_build_case_inputs` and the
helper `_build_column_inputs`, each named for the artifact it writes
(case_inputs.json, column_inputs.json).

An AST scan rather than a runtime check, because by the time the class
object exists the evidence has already been destroyed.
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
MODULES = sorted(SRC.rglob("*.py"))


def _duplicate_methods(tree):
    """(class_name, method_name, [lines]) for every name def'd more than once."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        seen = {}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seen.setdefault(item.name, []).append(item.lineno)
        for name, lines in seen.items():
            if len(lines) > 1:
                out.append((node.name, name, lines))
    return out


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_class_defines_a_method_twice(path):
    dupes = _duplicate_methods(ast.parse(path.read_text()))
    assert not dupes, "\n".join(
        f"{path.name}: {cls}.{name} defined {len(lines)}x at lines {lines} — "
        f"the earlier one is silently discarded"
        for cls, name, lines in dupes)


def test_the_scan_would_catch_a_shadowed_method():
    """The guard itself, against the exact shape that got through."""
    src = (
        "class ELMExpManager:\n"
        "    def _build_inputs(self, config): ...\n"
        "    def _build_inputs(self, plan, config): ...\n"
    )
    dupes = _duplicate_methods(ast.parse(src))
    assert dupes == [("ELMExpManager", "_build_inputs", [2, 3])]
