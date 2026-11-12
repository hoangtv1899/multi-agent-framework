#!/usr/bin/env python3
"""
Which kinds of study this version runs
src/core/supported_routes.py

THIS VERSION RUNS WHOLE-DOMAIN STUDIES ON A PREPARED CONTROL, AND NOTHING ELSE
(2026-11-10). The single-column routes (a sampled basin, a controlled sweep)
and the coupling route keep their code and their tests. They are set aside:
no request reaches them from the front door, whatever the person answers in
the dialog.

Why a rule in code. A live question about a watershed with no prepared control
was answered correctly at first ("no control exists there"), and then, on
"proceed anyway", turned into a single-column design with a prompt to run it.
A sentence in a prompt asks a language model to refuse; it cannot make it.

One reading of the rule, applied at three places:

  Reception      before it mints a run directory or fetches anything
  the coordinator  before the design page could offer a yes
  the gate       before compute (core/strategy_check), for every way in

A developer reaches the routes set aside by setting IDEAS_DEVELOPER_ROUTES=1
in the environment of the process. Nothing typed into the dialog can do that.
The test session sets it (tests/conftest.py), since most tests of those routes
predate this rule; the tests of the rule itself remove it.
"""
import os
from typing import Any, Dict, Optional

SWITCH = "IDEAS_DEVELOPER_ROUTES"
SUPPORTED = ("whole_domain",)

# THE SERVERS A SUPPORTED KIND OF STUDY NEEDS (2026-11-10). A whole-domain
# study reads its control and makes its runs through one model server, and
# fetches nothing. The front door therefore starts that server and no other.
# It used to start every configured server, so the start screen listed eight
# of them, another model's and six data servers among them, above a banner
# that says only whole-domain questions are answered; and Reception spent
# look-ups on places it would not study. Nothing is deleted: with the
# developer's switch every configured server starts, as before.
SERVERS = {"whole_domain": ("elm",)}

# What each kind is called when a person is told it will not run.
WORDS = {
    "site": "a study of single model columns sampled across a basin",
    "conceptual": "a controlled sweep of single model columns",
    "coupling": "a study that drives one model with another's output",
}


def developer_routes() -> bool:
    """Is the developer's switch set in this process?"""
    return os.environ.get(SWITCH, "").strip() == "1"


def servers_to_start(configured) -> Optional[list]:
    """The configured servers the front door starts, in the configuration's
    order. None means every one of them: with the developer's switch set, or
    when the configuration names none of the servers a supported study needs
    (an unusual configuration still starts, and the scope banner then says
    that no prepared run can be read)."""
    if developer_routes():
        return None
    needed = {s for k in SUPPORTED for s in SERVERS.get(k, ())}
    picked = [n for n in (configured or []) if n in needed]
    return picked or None


def _low(v: Any) -> str:
    return str(v or "").strip().lower()


def kind(brief: Optional[Dict[str, Any]] = None,
         strategy: Optional[Dict[str, Any]] = None) -> str:
    """The kind of study: "whole_domain", "conceptual", "coupling" or "site".

    The same reading as the gate's (strategy_check.check) and the base
    manager's (_is_whole_domain, _is_factor_sweep, _is_coupling): the
    archetype is the strategy's word, else Reception's, and the strategy's own
    blocks are each sufficient. A study named whole-domain that carries no
    case and no experiment is still read as whole-domain here, and the gate
    stops it for naming none.
    """
    brief, s = brief or {}, strategy or {}
    arche = _low(s.get("archetype") or brief.get("design_archetype"))
    approach = _low((s.get("sampling") or {}).get("approach"))
    if (arche == "whole_domain" or approach == "whole_domain"
            or s.get("domain_case") or s.get("domain_experiment")
            or isinstance(brief.get("domain_experiment"), dict)):
        return "whole_domain"
    if arche == "coupling" or s.get("coupling"):
        return "coupling"
    if arche == "conceptual" or approach == "factor_sweep":
        return "conceptual"
    return "site"


def problem(brief: Optional[Dict[str, Any]] = None,
            strategy: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """None when this version runs the study. Otherwise the reason, in words a
    person can act on. Never a question: there is no answer that runs it."""
    k = kind(brief, strategy)
    if k in SUPPORTED or developer_routes():
        return None
    return (f"this version runs whole-domain studies on a prepared control only. "
            f"This request was read as {WORDS.get(k, k)}, which is set aside in this "
            f"version, so nothing was designed, fetched or run. A question about a "
            f"place with a prepared control can be asked instead")
