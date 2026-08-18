#!/usr/bin/env python3
"""
What a controlled sweep may vary, asked of the model that would run it
src/core/sweep_menu.py

    in   the MCP clients reception already holds, and a model name
    out  that server's menu, as JSON, or why it could not be read

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

def fetch(clients, model: str) -> Dict[str, Any]:
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


# render_sweep_menu WAS HERE and is deleted (2026-08-18). It rendered ELM's
# menu into prose and the reception prompt substituted it at construction —
# from a call hardcoded to model="elm", before any model was chosen, on every
# request. The menu is now fetched by model_servers.describe_sweep when the LLM
# asks for it, for the model it named, and handed over as the server's own
# JSON; the reading rules live in reception_agentic.txt STEP 2b, once.
