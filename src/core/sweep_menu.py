#!/usr/bin/env python3
"""
What a controlled sweep may vary, asked of the model that would run it
src/core/sweep_menu.py

    in   the MCP clients reception already holds
    out  the SWEEP MENU block substituted into the Reception prompt

THE FRAMEWORK DOES NOT KNOW WHAT ELM CAN VARY, and must not learn. Every entry
in this block is fetched from the model server at prompt-build time; nothing
here contains a factor name, a range, or a runtime key. A copy would be a
hand-maintained duplicate of RUNTIME_KEYS, correct on the day it was typed and
never checked again.

Written on the pattern of the forcing window, which is counted off the DATM
directory rather than asserted as a year range (it moved into
mcp/elm-mcp/src/forcing.py on 2026-08-17, since it is ELM's answer) — for the
same reason and with the same discipline: WHEN THE ANSWER CANNOT BE FETCHED, SAY SO
AND FORBID INVENTING ONE. An offered factor that does not exist costs a queue
slot and a designed study that cannot be built.

FAILS SOFT, ALWAYS. A site study needs none of this, and reception must not
stop working for every user because a model server is down. An unreachable
server produces a block that says the menu is unavailable, and the site path
never reads the block at all.
"""
from typing import Any, Dict, Optional

# The tool every model server exposes to answer this. A server without it
# simply has no sweep menu, which is a true statement about that server.
TOOL = "describe_conceptual_factors"

_UNAVAILABLE = (
    "  SWEEP MENU UNAVAILABLE — the {model} server did not answer\n"
    "  ({why}). Do NOT name factors, ranges or defaults from memory: what a\n"
    "  model can be told to vary is a property of that server's code, not\n"
    "  something to recall. Say plainly that the menu could not be read, ask\n"
    "  the user what they want to vary in their own words, and record in\n"
    "  run_settings.conflicts that the capability list was not verified."
)


def fetch(clients, model: str = "elm") -> Dict[str, Any]:
    """Ask the model server for its menu. Never raises; returns ok/error."""
    client = (clients or {}).get(model)
    if client is None:
        return {"ok": False, "why": f"no {model} client configured",
                "menu": None}
    try:
        out = client.call_tool_json(TOOL, {})
    except Exception as e:                       # noqa: BLE001 — reported
        return {"ok": False, "why": f"{type(e).__name__}: {e}"[:160],
                "menu": None}
    if not isinstance(out, dict) or out.get("error"):
        why = str((out or {}).get("error") or "tool returned no JSON object")
        return {"ok": False, "why": why[:160], "menu": None}
    return {"ok": True, "why": None, "menu": out}


def render_sweep_menu(clients, model: str = "elm") -> str:
    """The SWEEP MENU block. Empty-safe, and never invents a factor."""
    got = fetch(clients, model)
    if not got["ok"]:
        return _UNAVAILABLE.format(model=model, why=got["why"])

    menu = got["menu"] or {}
    factors = menu.get("factors") or []
    if not factors:
        return _UNAVAILABLE.format(model=model,
                                   why="the server offered no factors")

    lines = [f"  The {model} server can vary these, and NOTHING ELSE. Offer "
             f"them in the",
             f"  user's own terms; refuse anything not on this list rather "
             f"than designing",
             f"  a study that cannot be built.", ""]

    for i, f in enumerate(factors, 1):
        # THE MACHINE NAME FIRST, AND MARKED. The first version led with the
        # human label and never printed `name` at all — so a run settled a
        # perfectly good clay sweep and emitted factor "Soil texture", which
        # the server rejects because it offers "soil_texture". The label is for
        # talking to the user; the name is the only string that may be written
        # down, and the prompt has to be able to tell them apart.
        lines.append(f"  {i}. {f.get('label') or f.get('name')}")
        lines.append(f"     name: {f.get('name')}   <-- EMIT THIS EXACT STRING "
                     f"in factors[].name, never the label above")
        lines.append(f"     levels are {f.get('levels_are')}")
        sug = f.get("suggested_levels")
        if sug:
            lines.append(f"     SUGGEST: {sug}  ({len(sug)} columns) — offer "
                         f"this as a default the")
            lines.append(f"     user can accept or change, not as a question "
                         f"from nothing.")
        else:
            lines.append(f"     NO DEFAULT — this one must be asked; any "
                         f"value you pick would be")
            lines.append(f"     a scientific choice made on the user's behalf.")
        hf = f.get("holds_fixed") or []
        if hf:
            lines.append(f"     holds fixed: {', '.join(hf)}")
        if f.get("note"):
            lines.append(f"     {f['note']}")
        lines.append("")

    # WHAT IS TRUE OF EVERY SWEEP, whatever the user chooses. Printed before
    # the settings and the refusals because it is not a choice and not a
    # limitation — it is a property of the study they are about to design, and
    # they should hear it in the conversation rather than meet it in a caveat
    # attached to a finished result.
    for a in (menu.get("always_true") or []):
        lines += [f"  ALWAYS TRUE — {a.get('what')}.",
                  f"  {a.get('detail')}",
                  "  Say this while settling the design, not afterwards.", ""]

    grid = menu.get("soil_grid") or {}
    if grid.get("note"):
        lines += ["  THE SOIL COLUMN IS NOT CONFIGURABLE.",
                  f"  {grid['note']}",
                  "  If the user asks for a soil of a particular depth, this "
                  "is what to tell",
                  "  them — it is a misunderstanding to clear up, not a "
                  "setting to refuse.", ""]

    # SETTINGS THAT APPLY TO EVERY COLUMN, not things to sweep. Rendered right
    # after the factors because the most useful of them is easy to miss: the
    # menu reads as "here is what you may vary", and a user whose problem is
    # that the study should have NO location needs the option that removes it
    # from every column at once, not another thing to vary.
    opts = menu.get("held_fixed_options") or []
    if opts:
        lines.append("  APPLIED TO EVERY COLUMN AT ONCE — put these in "
                     "`held_fixed`, not in `factors`.")
        lines.append("  A factor makes columns DIFFER; these make them all the "
                     "same in one respect.")
        for o in opts:
            lines.append(f"    - held_fixed.{o.get('name')}: {o.get('takes')}")
            if o.get("does"):
                lines.append(f"      does: {o['does']}")
            if o.get("unlocks"):
                lines.append(f"      why it matters: {o['unlocks']}")
            if o.get("note"):
                lines.append(f"      {o['note']}")
        lines.append("")

    fills = menu.get("weather_fills") or []
    if fills:
        lines.append(f"  Weather fills the server accepts: {', '.join(fills)}. "
                     f"A spec is the string")
        lines.append(f"  \"copy\", or an object "
                     f"{{\"fill\": \"scale\", \"values\": {{\"PRECTmms\": 2.0}}}}. "
                     f"The variables are")
        lines.append(f"  {', '.join(sorted(menu.get('weather_variables') or {}))} "
                     f"— any other name is refused.")
        lines.append("")

    later = menu.get("not_yet_runnable") or []
    if later:
        lines.append("  DECLARED BUT NOT RUNNABLE — do NOT design a study "
                     "around one. But RAISE")
        lines.append("  one yourself the moment a user's question is the "
                     "question it answers:")
        lines.append("  saying nothing leaves them choosing between options "
                     "that do not include")
        lines.append("  the one they actually wanted.")
        for f in later:
            lines.append(f"    - {f.get('label') or f.get('name')}")
            if f.get("unlocks"):
                lines.append(f"      would allow: {f['unlocks']}")
            for b in (f.get("blocked_on") or [])[:1]:
                lines.append(f"      blocked on: {b}")
        lines.append("")

    cannot = menu.get("cannot_vary") or []
    if cannot:
        lines.append("  CANNOT BE VARIED AT ALL, and no amount of asking "
                     "changes it:")
        for c in cannot:
            lines.append(f"    - {c.get('what')} (needs {c.get('needs')})")
        lines.append("")

    n_min = menu.get("min_levels")
    if n_min:
        lines.append(f"  A sweep needs at least {n_min} levels per factor. "
                     f"One level is a single")
        lines.append(f"  run with a comparison implied and never made.")
        lines.append("")

    # THE NUMBER THE USER CAN ACTUALLY JUDGE. Everything else in this
    # conversation is a modelling choice they may have no view on; the column
    # count is a cost they can approve or refuse in one second.
    lines += ["  ALWAYS QUOTE THE COLUMN COUNT back before settling a design. "
              "Factors",
              "  multiply: 7 textures over 3 years is 21 columns, and 7 x 3 x "
              "3 is 63.",
              "  The largest run this framework has done is 19."]
    return "\n".join(lines)
