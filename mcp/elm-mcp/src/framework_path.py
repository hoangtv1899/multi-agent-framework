"""Where the multi-agent-framework checkout is, and how its modules reach sys.path.

elm-mcp is published as its own repository but is developed inside
multi-agent-framework, and it borrows a handful of that framework's modules.
This module is the single place that decides where they come from.

THE SERVER RUNS WITHOUT THE FRAMEWORK.  Its twelve tools need only six
framework modules, all of them self-contained: ``core.keyset``,
``core.model_agent_base``, ``core.basemap``, ``core.static_wtd``,
``agents.analysis.compare_common`` and ``figstyle``.  Verbatim copies live in
``src/_vendor/`` (see ``scripts/sync_vendor.py``), so a standalone checkout
works with nothing beside it.

THE EXPERIMENT MANAGER DOES NOT.  ``elm_exp_manager.py`` subclasses
``core.exp_manager_base``, a 2,000-line base class that ELM and PFLOTRAN
share and that reaches the framework's Analyzer and sampler.  Vendoring it
would fork a contract two models depend on, so it is not vendored, and the
code that needs it calls ``ensure_on_path(require=True)`` and fails plainly
without a framework.  The same goes for the analysis scripts, which need
``agents.analysis.step2_derive``, ``core.figure_registry`` and
``agents.analyzer_agent``.

ORDER MATTERS, and it is the point.  The framework is put on ``sys.path``
BEFORE ``_vendor``, so inside a multi-agent-framework checkout the framework's
own files execute and the vendored copies never do.  There is therefore one
source of truth during development, and the copies exist only for the
standalone case.  ``scripts/sync_vendor.py --check``, which the framework's
test suite runs, fails when a copy drifts.

Resolution order for the framework itself:

1. ``$IDEAS_FRAMEWORK_DIR``, if set.  Set it when this repository is checked
   out on its own and you want the real framework rather than the copies.
2. Three directories above this file, which is where the framework sits when
   elm-mcp is used in place at ``multi-agent-framework/mcp/elm-mcp/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ENV_VAR = "IDEAS_FRAMEWORK_DIR"

# This file is <framework>/mcp/elm-mcp/src/framework_path.py, so the framework
# root is three levels up.  It is counted ONCE, here.
_IN_PLACE_DEFAULT = Path(__file__).resolve().parents[3]

# Verbatim copies of the self-contained framework modules the server needs.
_VENDOR = Path(__file__).resolve().parent / "_vendor"

#: Modules the vendored fallback carries.  Anything else needs a real checkout.
VENDORED = (
    "core.keyset",
    "core.model_agent_base",
    "core.basemap",
    "core.static_wtd",
    "agents.analysis.compare_common",
    "figstyle",
)


def framework_dir() -> Path:
    """The multi-agent-framework checkout, as an absolute path.

    Does not check that it exists; see :func:`have_framework`.
    """
    return Path(os.environ.get(_ENV_VAR) or _IN_PLACE_DEFAULT).resolve()


def have_framework() -> bool:
    """True when :func:`framework_dir` really holds the framework source."""
    return (framework_dir() / "src" / "core").is_dir()


def _explain() -> str:
    where = os.environ.get(_ENV_VAR)
    if where:
        return f"{_ENV_VAR} is set to {where!r}, which does not hold the framework"
    return (
        f"{_ENV_VAR} is unset, so it was guessed as {framework_dir()} "
        "(correct only inside a multi-agent-framework checkout)"
    )


def _append(d: Path) -> None:
    if d.is_dir() and str(d) not in sys.path:
        # APPENDED, never inserted at 0: a name that exists both here and in
        # the framework must resolve to elm-mcp's own copy.
        sys.path.append(str(d))


def ensure_on_path(*, tools: bool = False, require: bool = False) -> Path | None:
    """Make the borrowed framework modules importable.

    Puts the framework's ``src/`` (and its ``tools/`` when ``tools=True``) on
    ``sys.path`` when a checkout is there, then ``_vendor`` behind it as the
    standalone fallback.  Returns the framework root, or None when running on
    the vendored copies alone.

    Pass ``require=True`` for code that needs modules ``_vendor`` does not
    carry, above all ``core.exp_manager_base``.  It then raises RuntimeError
    naming ``IDEAS_FRAMEWORK_DIR`` instead of letting the import fail later
    and less clearly.
    """
    root = framework_dir() if have_framework() else None

    if root is not None:
        # BOTH, always, whatever `tools` asked for. A caller that wanted only
        # src/ used to leave the vendored figstyle ahead of the framework's on
        # the path, so a later caller asking for tools/ appended behind it and
        # got the copy. The flag is kept for callers that read as documentation.
        _append(root / "src")
        _append(root / "tools")
    elif require:
        raise RuntimeError(
            f"This needs a full multi-agent-framework checkout, not just the "
            f"modules vendored into elm-mcp: {_explain()}.\n"
            f"The vendored fallback carries only {', '.join(VENDORED)}, which "
            f"is enough to run the MCP server but not the experiment manager "
            f"or the analysis scripts.\n"
            f"Set {_ENV_VAR} to a multi-agent-framework checkout."
        )

    # Always LAST, so it is a fallback and never a shadow. Re-appended rather
    # than skipped when already present, because an earlier call may have put
    # it on the path before the framework directories were known.
    if str(_VENDOR) in sys.path:
        sys.path.remove(str(_VENDOR))
    _append(_VENDOR)

    if root is None and not _VENDOR.is_dir():
        raise RuntimeError(
            f"Neither a framework checkout nor the vendored copies are "
            f"available: {_explain()}, and {_VENDOR} does not exist.\n"
            f"Restore it with: python3 scripts/sync_vendor.py --write"
        )
    return root


if __name__ == "__main__":
    import json

    where = os.environ.get(_ENV_VAR)
    report = {
        _ENV_VAR: where,
        "resolved": str(framework_dir()),
        "source": "environment" if where else "in-place default",
        "framework_present": have_framework(),
        "vendor_present": _VENDOR.is_dir(),
    }
    try:
        root = ensure_on_path(tools=True)
        report["mode"] = "framework" if root else "vendored (standalone)"
        report["server_tools"] = "available"
        report["experiment_manager"] = "available" if root else (
            "unavailable: needs a framework checkout")
    except RuntimeError as exc:
        report["mode"] = "broken"
        report["error"] = str(exc)
    print(json.dumps(report, indent=2))
