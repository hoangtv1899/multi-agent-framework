"""Where the multi-agent-framework checkout is, and how its modules reach sys.path.

elm-mcp is a component of multi-agent-framework, not a standalone package.  It
imports from the framework's ``src/`` (``core.keyset``, ``core.exp_manager_base``,
``core.model_agent_base``, ``core.limitations``, ``core.basemap``,
``core.static_wtd``, ``core.figure_registry``, ``agents.analysis.compare_common``,
``agents.analysis.step2_derive``, ``agents.analyzer_agent``) and ``figstyle`` from
its ``tools/``.  This module is the single place that decides where that checkout
lives.

Resolution order:

1. ``$IDEAS_FRAMEWORK_DIR``, if set.  Set it when this directory is checked out
   on its own, away from the framework.
2. Three directories above this file, which is where the framework sits when
   elm-mcp is used in place at ``multi-agent-framework/mcp/elm-mcp/``.

Before this module existed every caller counted parent directories for itself,
nine times over, at three different depths.  Those counts were correct only in
the monorepo: from a standalone checkout each one resolved above the repository,
``sys.path.insert`` accepted the non-existent directory without complaint, and
the failure surfaced much later as ``ModuleNotFoundError: No module named
'core'``.  ``ensure_on_path`` refuses the directory instead, and says which
environment variable fixes it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ENV_VAR = "IDEAS_FRAMEWORK_DIR"

# This file is <framework>/mcp/elm-mcp/src/framework_path.py, so the framework
# root is three levels up.  It is counted ONCE, here.
_IN_PLACE_DEFAULT = Path(__file__).resolve().parents[3]


def framework_dir() -> Path:
    """The multi-agent-framework checkout, as an absolute path.

    Does not check that it exists; see :func:`ensure_on_path`.
    """
    return Path(os.environ.get(_ENV_VAR) or _IN_PLACE_DEFAULT).resolve()


def ensure_on_path(*, tools: bool = False) -> Path:
    """Put the framework's ``src/`` on ``sys.path`` and return the framework root.

    Pass ``tools=True`` to also add its ``tools/`` directory, which is where
    ``figstyle`` lives.  Idempotent.

    Raises RuntimeError naming ``IDEAS_FRAMEWORK_DIR`` when the resolved
    directory does not hold the framework, rather than leaving a bad entry on
    ``sys.path`` for a later import to trip over.
    """
    root = framework_dir()
    src = root / "src"

    if not (src / "core").is_dir():
        where = os.environ.get(_ENV_VAR)
        how = (
            f"{_ENV_VAR} is set to {where!r}"
            if where
            else f"{_ENV_VAR} is unset, so it was guessed as {root} "
            "(correct only inside a multi-agent-framework checkout)"
        )
        raise RuntimeError(
            f"Cannot find the multi-agent-framework source at {src}: {how}.\n"
            "elm-mcp is a component of multi-agent-framework and needs that "
            f"checkout to import from. Set {_ENV_VAR} to point at it."
        )

    # APPENDED, never inserted at 0: a name that exists both here and in the
    # framework must resolve to elm-mcp's copy, which is what main.py's own
    # ordering has always arranged.  Appending makes that true whatever order
    # the callers happen to run in.
    wanted = [src] + ([root / "tools"] if tools else [])
    for d in wanted:
        if str(d) not in sys.path:
            sys.path.append(str(d))
    return root


if __name__ == "__main__":
    import json

    where = os.environ.get(_ENV_VAR)
    report = {
        _ENV_VAR: where,
        "resolved": str(framework_dir()),
        "source": "environment" if where else "in-place default",
        "in_place_default": str(_IN_PLACE_DEFAULT),
    }
    try:
        ensure_on_path(tools=True)
        report["usable"] = True
    except RuntimeError as exc:
        report["usable"] = False
        report["error"] = str(exc)
    print(json.dumps(report, indent=2))
