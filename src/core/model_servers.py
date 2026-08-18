#!/usr/bin/env python3
"""
The two questions reception asks about models, and answers itself
src/core/model_servers.py

    in   the MCP clients reception already holds
    out  two FRAMEWORK TOOLS the reception LLM calls when it wants them

THE FRAMEWORK DOES NOT KNOW WHICH SERVERS ARE MODELS, and must not be told.
There is no list of model names here and no mapping from a name to a server —
core/backends.py held both and was deleted on 2026-08-16.

WHAT CHANGED ON 2026-08-17, and why. This module used to render a finished
MENU: it decided which servers were models (a report carrying both `model` and
`pinning`), summarised each into a dozen lines, and pasted the result into the
prompt before the request was read. That made the choice cheap and it also made
it someone else's. Three things were wrong with it:

  - The RECOGNITION was the code's. A server whose report shaped its pinning
    differently was not a model, whatever it actually was, and nothing said so.
  - The SUMMARY was the code's. `_entry` kept five keys of eleven thousand
    characters and threw the rest away, so the chooser never saw a word the
    server wrote about itself.
  - The VERDICT was the code's. A server outside RUNNABLE got the sentence
    "Do not choose it" printed into the prompt, which is not evidence to weigh
    but an instruction to obey.

Now the model reads the servers itself: `list_servers` then `describe_server`,
as many as it wants, in whatever order it likes. The code's remaining job is to
answer honestly and to say which facts are the FRAMEWORK's rather than a
server's — `can_run_a_study` below is the only one, and it is labelled.
"""
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

# A tool by this name is a server describing itself. Used ONLY to tell the
# model where to look; it is not a test of whether something is a model. The
# obvious version of that test catches ameriflux, a tower database, and daymet,
# a rainfall service — which is the whole reason judging it is not code's job.
_DESCRIBE = re.compile(r"^describe_.+_capabilities$")

# WHAT THIS FRAMEWORK CAN ACTUALLY DRIVE, which is not the same as what can
# describe itself. A model server exposes tools; running a STUDY means walking
# the stages — materialize, build, run, extract, package — and writing the
# experiment record the Analyzer reads. That takes a manager beside the server
# (core/resumable._manager_for names them): ELM's since 2026-08-10, PFLOTRAN's
# since 2026-08-18, when it ran a Naches study from the request through package.
#
# This is the last remnant of core/backends.py and it is deliberately one line.
# It is HAND-MAINTAINED and nothing checks it: write a third manager and forget
# this line, and reception will go on refusing a model that now works.
RUNNABLE = frozenset({"elm", "pflotran"})


def _list_tools(client) -> Dict[str, Any]:
    """One server's tool names. Never raises; reports unreachable instead."""
    try:
        return {"reachable": True,
                "tools": [str(t) for t in (client.list_tools() or [])]}
    except Exception as e:                      # noqa: BLE001
        return {"reachable": False, "tools": [],
                "error": f"{type(e).__name__}: {e}"[:120]}


def _tools_of(clients, cache: Optional[Dict], name: str) -> Dict[str, Any]:
    """Cached tool names for one server.

    EVERY MCP CALL IS A SUBPROCESS. The client opens a fresh session per call
    and tears it down after, so asking a server what tools it has costs 1.5 to
    3.8 seconds — measured across the nine configured here — whether or not
    anything has changed. list_servers asks all nine; without this cache
    describe_server asked again, and a single reception paid for the same
    answer three or four times.
    """
    if cache is not None and name in cache:
        return cache[name]
    got = _list_tools(clients[name])
    if cache is not None:
        cache[name] = got
    return got


def _describe_tool_name(tools: List[str]) -> Optional[str]:
    return next((t for t in tools if _DESCRIBE.match(t)), None)


def list_servers(clients, cache: Optional[Dict] = None) -> Dict[str, Any]:
    """Every configured server, with no judgement about which are models.

    EVERY server, not a filtered set. Reception is choosing an instrument and
    the shortlist is part of that choice; handing it a pre-filtered list is
    the same as making the choice for it. A server with no describe tool is
    still listed, with its tool names, because a name can be evidence.

    ASKED IN PARALLEL. Nine servers at 1.5-3.8 s each is 18.5 s sequentially
    and the calls do not depend on one another — each spawns its own
    subprocess and its own event loop, so they overlap cleanly.
    """
    names = sorted(clients or {})
    if names:
        with ThreadPoolExecutor(max_workers=min(len(names), 12)) as pool:
            got = dict(zip(names, pool.map(
                lambda n: _tools_of(clients, cache, n), names)))
    else:
        got = {}

    out: List[Dict[str, Any]] = []
    for name in names:
        g = got[name]
        if not g.get("reachable"):
            out.append({"server": name, "reachable": False,
                        "error": g.get("error"),
                        "n_tools": 0, "tools": [], "describe_tool": None,
                        "can_run_a_study": name in RUNNABLE})
            continue
        tools = g["tools"]
        out.append({
            "server": name,
            "reachable": True,
            "n_tools": len(tools),
            "tools": tools,
            "describe_tool": _describe_tool_name(tools),
            # THE ONE FACT HERE THAT IS NOT THE SERVER'S. Labelled so it is
            # weighed as what it is: a statement about this repository today,
            # not about the model. A server can be excellent and still be false
            # here, and that is not a criticism of the server.
            "can_run_a_study": name in RUNNABLE,
        })
    return {
        "servers": out,
        "n_servers": len(out),
        "can_run_a_study_means": (
            "THE FRAMEWORK's answer, not the server's: whether anything here "
            "turns a sampling strategy into a finished study with that server "
            "— materialize, build, run, extract, and the experiment record the "
            "Analyzer reads. A server can be fully working and still be false "
            "here. It is hand-maintained and could be out of date."),
        "describe_tool_means": (
            "the server offers an account of itself. Call describe_server on "
            "it to read what it said. It is a place to look, NOT a sign that "
            "the server is a model — data services expose one too. WHERE THIS "
            "IS null THERE IS NOTHING TO READ: calling describe_server on it "
            "returns this same fact and nothing more, so do not spend a call "
            "on it."),
    }


def describe_server(clients, server: str,
                    cache: Optional[Dict] = None,
                    reports: Optional[Dict] = None) -> Dict[str, Any]:
    """One server's own account of itself, verbatim.

    UNSUMMARISED ON PURPOSE. The point of asking is to read what the server
    wrote, not what this file thought was worth keeping. ELM's report is 7,638
    characters and the loop truncates a tool result at 12,000, so one report
    per call fits and four in one call would not — which is why this takes a
    single name rather than returning them all.

    THREE THINGS ARE CACHED, and each was a measured cost. The tool list comes
    from list_servers rather than being fetched again (1.5-3.8 s). A server
    with no describe tool is answered WITHOUT being contacted at all, because
    list_servers already established that. And a report already fetched is
    returned as it was, flagged `already_asked`, rather than spending another
    2.5 s to receive the same characters — a reception was observed asking for
    pflotran twice and geology twice in one selection.
    """
    if (clients or {}).get(server) is None:
        return {"error": f"no server named {server!r} is configured",
                "configured": sorted(clients or {})}
    if reports is not None and server in reports:
        return {**reports[server], "already_asked": True,
                "note": ("you asked for this in this same conversation; this "
                         "is the same answer, not a new one.")}

    g = _tools_of(clients, cache, server)
    if not g.get("reachable"):
        return {"server": server, "reachable": False, "error": g.get("error"),
                "note": "the server could not be reached; it is not usable."}
    tool = _describe_tool_name(g["tools"])
    if not tool:
        # ANSWERED WITHOUT CONTACTING IT. list_servers already reported
        # describe_tool: null for this server, so a round trip here would buy
        # nothing but latency.
        return {"server": server, "describes_itself": False,
                "tools": g["tools"],
                "note": ("this server exposes no describe_*_capabilities tool, "
                         "so it has no account of itself to read — list_servers "
                         "already said so. Judge it from its tool names above, "
                         "or do not use it.")}
    try:
        rep = clients[server].call_tool_json(tool, {})
    except Exception as e:                      # noqa: BLE001
        return {"server": server, "describes_itself": True, "tool": tool,
                "error": f"{type(e).__name__}: {e}"[:200],
                "note": "the server did not answer; it is not usable now."}
    out = {"server": server, "tool": tool,
           "can_run_a_study": server in RUNNABLE,
           "report": rep}
    if reports is not None:
        reports[server] = out
    return out


def describe_sweep(clients, server: str,
                   cache: Optional[Dict] = None) -> Dict[str, Any]:
    """One server's sweep menu — the factors a controlled study may vary.

    The server's JSON, verbatim; the reading rules are in the reception prompt
    (STEP 2b), stated once for every model rather than rendered per model.
    A server without the tool, or one that will not answer, gets a plain
    statement rather than a default: what a model can be told to vary is a
    property of its code, and inventing a factor costs a queue slot and a
    designed study that cannot be built.

    ANSWERED FROM THE TOOL LIST WHEN THE TOOL IS ABSENT — the same courtesy
    describe_server extends: list_servers already fetched every server's tool
    names, so a server with no describe_conceptual_factors is told so without
    a round trip that could only come back "unknown tool".
    """
    from core.sweep_menu import TOOL, fetch
    if (clients or {}).get(server) is None:
        return {"error": f"no server named {server!r} is configured",
                "configured": sorted(clients or {})}
    g = _tools_of(clients, cache, server)
    if g.get("reachable") and TOOL not in (g.get("tools") or []):
        return {"server": server, "offers_a_sweep": False,
                "why": f"this server has no {TOOL} tool",
                "note": ("this server offers no controlled sweep. Do NOT name "
                         "factors, ranges or defaults from memory; say so to "
                         "the user, ask what they want to vary in their own "
                         "words, and record in run_settings.conflicts that "
                         "the capability list was not verified.")}
    got = fetch(clients, server)
    if not got["ok"]:
        return {"server": server, "offers_a_sweep": False, "why": got["why"],
                "note": ("this server offers no controlled sweep, or would not "
                         "answer. Do NOT name factors, ranges or defaults from "
                         "memory; say the menu could not be read, ask the user "
                         "what they want to vary in their own words, and "
                         "record in run_settings.conflicts that the capability "
                         "list was not verified.")}
    return {"server": server, "offers_a_sweep": True, "menu": got["menu"]}


# ─────────────────────────────────────────────────────────────────────────────
# THE TOOLS, as the reception loop takes them
# ─────────────────────────────────────────────────────────────────────────────
def model_tools(clients) -> Dict[str, Any]:
    """`local_tools` for ToolLoopAgent: the three questions, bound to clients.

    THE CACHES LIVE HERE, one set per reception. Deliberately not module-level:
    a long-lived process that added or restarted a server would keep answering
    from a stale picture, and "what is installed right now" is the one thing
    these tools exist to get right. Within a single reception nothing changes,
    so caching there is free.
    """
    tool_cache: Dict[str, Any] = {}
    report_cache: Dict[str, Any] = {}
    return {
        "list_servers": {
            "schema": {
                "type": "function",
                "function": {
                    "name": "list_servers",
                    "description": (
                        "Every MCP server configured right now, with its tool "
                        "names and whether it offers an account of itself. "
                        "Call this FIRST when you need to choose a model — it "
                        "is the only way to know what is installed today, and "
                        "no server can answer it. It makes no judgement about "
                        "which servers are models; that is yours to make."),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            "fn": lambda: list_servers(clients, cache=tool_cache),
        },
        "describe_server": {
            "schema": {
                "type": "function",
                "function": {
                    "name": "describe_server",
                    "description": (
                        "One server's own account of itself, verbatim and "
                        "unsummarised: what it models, whether it is ready, "
                        "and what one of its columns may be compared against. "
                        "Call it on every server that could plausibly answer "
                        "the question — reading two and comparing them is the "
                        "point. One server per call. Do NOT call it on a "
                        "server whose describe_tool was null in list_servers: "
                        "that already IS the answer. Do not call it twice for "
                        "the same server; the reply will not change."),
                    "parameters": {
                        "type": "object",
                        "properties": {"server": {
                            "type": "string",
                            "description": "the exact name from list_servers"}},
                        "required": ["server"],
                    },
                },
            },
            "fn": lambda server: describe_server(
                clients, server, cache=tool_cache, reports=report_cache),
        },
        # WHAT A CONTROLLED SWEEP MAY VARY, asked of the model the LLM chose —
        # AFTER it chose (2026-08-18). This used to be RENDERED INTO THE PROMPT
        # at construction, from a call hardcoded to "elm", so every reception
        # carried 5,871 characters of ELM's factors under a heading saying
        # they came from "the model server just now" — before any model had
        # been named, and for a model that may not offer a sweep at all. The
        # same class of bug the pinning block had before 08-17. Now it is one
        # more question the LLM asks, of one server, only on a conceptual
        # request, and the answer is the server's JSON, unrendered.
        "describe_sweep": {
            "schema": {
                "type": "function",
                "function": {
                    "name": "describe_sweep",
                    "description": (
                        "What a CONTROLLED SWEEP may vary in one model, from "
                        "that model's server: the factors it can be told to "
                        "vary (machine `name`, human `label`, what the levels "
                        "are, suggested levels or none), what is held fixed, "
                        "what is true of every sweep, and what cannot be "
                        "varied. Call it ONLY for a conceptual request, on "
                        "the model you chose in describe_server, and read "
                        "factors[].name as the only string you may write into "
                        "the brief. A server with no sweep says so; that is a "
                        "true statement about that server, not a failure."),
                    "parameters": {
                        "type": "object",
                        "properties": {"server": {
                            "type": "string",
                            "description": "the exact name from list_servers"}},
                        "required": ["server"],
                    },
                },
            },
            "fn": lambda server: describe_sweep(clients, server,
                                                cache=tool_cache),
        },
    }
