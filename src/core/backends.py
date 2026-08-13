#!/usr/bin/env python3
"""
The models the workflow can dispatch to
src/core/backends.py

    in   a model name ("elm", "pflotran")
    out  the Experiment Manager class that runs it

One table, so adding a third backend is a line here rather than an `if` in
every caller. workflow.py, the tools, and the tests all resolve a name through
this function, which means they cannot disagree about what "pflotran" refers to.

WHY THE IMPORT IS LAZY. Each manager pulls in its own stack — ELM reaches for
the CIME/E3SM tooling and the CONUS restart readers, PFLOTRAN for the deck
builder. Importing the table should not require every backend's dependencies to
be installed, or a PFLOTRAN-only environment could not run a PFLOTRAN study.

WHY NOT A `model` FLAG ON ONE MANAGER. The two backends differ in the stages
they have, not only in the code inside them: PFLOTRAN has no case-build step and
no scheduler, ELM has both. Those are declarations on the class
(NEEDS_CASE_BUILD, NEEDS_SCHEDULER, COUPLES_TO) that the base's execute_plan
reads, so the choice of model IS the choice of class.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# TEMPORARY, and it points the wrong way on purpose.
#
# ELM's code now lives in mcp/elm-mcp/src/ (docs/ELM_MCP_PLAN.md §9 phase 1b),
# which is where it belongs under the rule that anything requiring knowledge of
# ELM lives in ELM's server. But ELMExpManager is not deleted until phase 1e,
# and until then the framework still instantiates it — so for exactly that long
# the framework has to be able to import out of the server directory.
#
# This is the ONE edge from the framework into a model server, it exists only
# because a class is mid-deletion, and it goes when that class does. It is here
# rather than inside elm_exp_manager because this is the file that already knows
# which backend lives where, and one visible line is easier to remove than a
# hidden one.
_ELM_MCP_SRC = Path(__file__).resolve().parents[2] / "mcp" / "elm-mcp" / "src"
if _ELM_MCP_SRC.is_dir() and str(_ELM_MCP_SRC) not in sys.path:
    sys.path.append(str(_ELM_MCP_SRC))      # append, not insert: the framework's
                                            # own modules must still win a tie

# name -> (module path, class name). Lazy on purpose; see the docstring.
_BACKENDS: Dict[str, Tuple[str, str]] = {
    "elm":             ("elm_exp_manager",           "ELMExpManager"),
    "pflotran":        ("core.pflotran_exp_manager", "PFLOTRANExpManager"),
    "lambda-pflotran": ("core.lambda_pflotran_exp_manager",
                        "LambdaPFLOTRANExpManager"),
}

# Spellings that mean the same backend. Only for names a person would
# reasonably type for a model that EXISTS — never a near-miss of one that does
# not, which must still raise.
_ALIASES: Dict[str, str] = {
    "lambda_pflotran": "lambda-pflotran",
    "lambda":          "lambda-pflotran",
}

DEFAULT = "elm"


def names() -> List[str]:
    """Every dispatchable model name, for argparse choices and error text."""
    return sorted(_BACKENDS)


def get(name: str):
    """The Experiment Manager class for `name`.

    Raises on an unknown name rather than falling back to the default. A
    typo'd --model that quietly ran ELM would report a completed ELM study to
    someone who asked for PFLOTRAN, and every file in the run directory would
    agree with the wrong answer.
    """
    key = (name or "").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _BACKENDS:
        raise ValueError(
            f"unknown model {name!r} — known models: {', '.join(names())}")
    mod_path, cls_name = _BACKENDS[key]
    import importlib
    return getattr(importlib.import_module(mod_path), cls_name)


def config_for(name: str,
               base: Dict[str, Any],
               period: Dict[str, Any] = None,
               initialization: Dict[str, Any] = None) -> Dict[str, Any]:
    """Add the settings that only one backend understands.

    Kept beside the table rather than in the coordinator so that what a model
    needs travels with the model. `base` holds what every backend takes —
    brief, reception, strategy, mcp_clients — and is never modified in place.

    Passing a key a backend ignores is harmless; passing one it MISREADS is
    not, which is why warm_start does not reach PFLOTRAN. It has no restart
    file and no donor gridcell, and a run recording a warm start it never
    performed is a false statement about its own initial condition — which for
    these columns is the Fan 2013 water table, recorded in the assumptions
    ledger the manager writes.
    """
    cfg = dict(base)
    key = (name or "").strip().lower()

    # The period reception resolved. Every backend records it, because it is
    # what was ASKED about — not necessarily what was simulated. PFLOTRAN's own
    # simulated duration is a separate number and its ledger says so.
    if period:
        if period.get("yr_start"):
            cfg["yr_start"] = int(period["yr_start"])
        cfg["yr_end"] = int(period.get("yr_end")
                            or period.get("yr_start") or 1995)

    if key == "elm":
        # WARM IS THE DEFAULT. A cold single-column year starts from ELM's
        # generic state and spends the run relaxing out of it — measured on
        # this framework, recharge came out -0.18 mm/yr cold against 309 warm
        # on the SAME column. Cold is an explicit opt-out.
        # NO `source` KEY (2026-08-12). There is one warm start — the CONUS
        # 1-km restart subset — so a source was a choice with one option, and
        # inputs.py never read it anyway: reception emitted it, this passed it
        # on, and the warm start always did the CONUS subset regardless. A
        # setting that travels three components and changes nothing is worse
        # than no setting, because it reads like a lever.
        if (initialization or {}).get("mode") != "cold":
            cfg["warm_start"] = True

    elif key in ("pflotran", "lambda-pflotran"):
        # Defaults matching the standalone tool, so a deck built through the
        # workflow and one built from the command line are the same deck.
        cfg.setdefault("bottom", "fan")
        cfg.setdefault("years", 20.0)
        cfg.setdefault("depth_cap", 50.0)
        if key == "lambda-pflotran":
            # NOT a default: the LAMBDA deck NaNs at t=9.31 y, so 20 y is not a
            # duration this model can run. The manager clamps it and records
            # the clamp, which is why the ceiling is not silently applied here.
            cfg.setdefault("n_bins", 3)

    return cfg
