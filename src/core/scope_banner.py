#!/usr/bin/env python3
"""
What this version answers, said before the first question
src/core/scope_banner.py

A person met the scope only after asking: the start screen listed the data
servers and nothing else, and the prepared run was introduced in Reception's
first reply. A question outside the scope then cost a conversation to refuse.

The banner says it first, in three statements:

  1. the prepared runs questions are asked of, each with its own description
     and the years it has output for;
  2. what a what-if run may change, with a mark on a change no run has
     completed with here;
  3. what is not answered.

EVERY WORD ABOUT A RUN OR A CHANGE IS THE MODEL SERVER'S OWN: its control
registry (describe_domain_controls) and its change list
(describe_domain_changes), the records Reception and the Planner read. This
module names no model, no place and no change, so it cannot go stale when a
control or a change is added. It calls no language model.
"""
import textwrap
from typing import Any, Dict, List, Optional

WIDTH = 98
RULE = "=" * 70
TITLE = "WHAT THIS VERSION ANSWERS"
MARK = "*"


def read(clients) -> Dict[str, Dict[str, Any]]:
    """{server: {"controls": [...], "changes": [...], "undemonstrated": words}}
    for every model server that offers whole-domain studies. A server that
    cannot be reached, or offers none, is left out; `problems` under the key
    "" lists why, so the banner can say that it could not read."""
    from core.model_servers import describe_domain, RUNNABLE
    out: Dict[str, Dict[str, Any]] = {}
    problems: List[str] = []
    for server in sorted(set(clients or {}) & set(RUNNABLE)):
        got = {w: describe_domain(clients, server, w) for w in ("controls", "changes")}
        if not all(g.get("offers_whole_domain_experiments") for g in got.values()):
            why = next((g.get("why") or g.get("error") for g in got.values()
                        if not g.get("offers_whole_domain_experiments")), None)
            # a server with no such tool is not a problem: it offers none
            if why and "has no" not in str(why):
                problems.append(f"{server}: {why}")
            continue
        ctl = got["controls"].get("controls") or {}
        chg = got["changes"].get("changes") or {}
        out[server] = {"controls": list(ctl.get("controls") or []),
                       "changes": list(chg.get("changes") or []),
                       "undemonstrated": chg.get("demonstrated_false")}
    if problems:
        out[""] = {"problems": problems}
    return out


def _span(years) -> Optional[str]:
    ys = [y for y in (years or []) if isinstance(y, int)]
    if not ys:
        return None
    return str(ys[0]) if ys[0] == ys[-1] else f"{ys[0]}–{ys[-1]}"


def _wrap(text: str, first: str = "", rest: str = "  ") -> List[str]:
    return textwrap.wrap(" ".join(str(text).split()), width=WIDTH,
                         initial_indent=first, subsequent_indent=rest) or [first.rstrip()]


def lines(menus: Dict[str, Dict[str, Any]], developer: bool = False) -> List[str]:
    """The banner, from what read() returned."""
    problems = (menus.get("") or {}).get("problems") or []
    servers = {s: m for s, m in menus.items() if s}
    runs = [(s, c) for s, m in servers.items() for c in m.get("controls") or []
            if isinstance(c, dict) and c.get("name")]
    L = [RULE, TITLE, RULE]
    if not runs:
        L += _wrap("Questions about a prepared whole-domain run, and what-if runs made from "
                   "it. No prepared run is registered on this machine, so no question can "
                   "be run here.")
        L += [f"  could not be read: {p}" for p in problems]
        return L + [RULE]

    # 1. the prepared runs
    L.append("Questions about " + ("this prepared run" if len(runs) == 1 else
                                   "these prepared runs")
             + ", and what-if runs made from " + ("it:" if len(runs) == 1 else "them:"))
    for server, c in runs:
        years = _span((c.get("facts") or {}).get("output_years"))
        L += _wrap(f"{c['name']} ({server}): {c.get('description') or 'no description'}"
                   + (f"; output for {years}" if years else "; its output years are not "
                                                           "on record"),
                   first="  ", rest="    ")

    # 2. what a what-if run may change
    untested = False
    for server, m in servers.items():
        items = []
        for ch in m.get("changes") or []:
            if not (isinstance(ch, dict) and ch.get("available")):
                continue
            mark = MARK if ch.get("demonstrated") is False else ""
            untested = untested or bool(mark)
            items.append(f"{ch.get('summary') or ch.get('id')}{mark}")
        if items:
            L += _wrap("A what-if run can: " + "; ".join(items) + ".")
    if untested:
        L += _wrap(f"{MARK} no run with this change has been completed here yet, so its "
                   f"result is untested.", first="  ", rest="    ")

    # 3. what is not answered
    L += _wrap("Not answered: another place or another model, a year outside the output, "
               "or a change that is not listed above.")
    for _, c in runs:
        for w in (c.get("not_in_it_words") or [])[:2]:
            L += _wrap(f"Not in {c['name']}: {w}.", first="  ", rest="    ")
    if developer:
        L += _wrap("The developer's switch is set: the kinds of study set aside in this "
                   "version can be reached in this session.")
    L += [f"  could not be read: {p}" for p in problems]
    return L + [RULE]


def text(clients, developer: Optional[bool] = None) -> str:
    """The banner as one string, read from the model servers now."""
    if developer is None:
        from core.supported_routes import developer_routes
        developer = developer_routes()
    return "\n".join(lines(read(clients), developer=developer))
