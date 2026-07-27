#!/usr/bin/env python3
"""
Agentic tool-loop runtime — lets an LLM drive the MCP tools itself.

This is the GENERIC runtime behind the agentic reception agent. It has NO domain
logic: it exposes the MCP servers' tools to the model as OpenAI function-calling
tools, runs the call/result loop, and returns the model's final message plus a
full tool trace. All "what to fetch / when to stop" decisions live in the LLM.

Key gateway rule (PNNL/Bedrock): `tools=` is passed on EVERY request, including
the follow-ups that feed tool results back, or Bedrock-routed Claude 400s.

Usage:
    agent = ToolLoopAgent(model, mcp_clients, allowlist={"terrain__resolve_watershed", ...})
    out = agent.run(system_prompt, user_message)
    out["content"]   # final text (e.g. the brief JSON)
    out["trace"]     # [{round, tool, args, result}, ...]
"""
import json
import sys
from typing import Any, Dict, List, Optional

from agents.llm_agent import SimpleLLMClient

_MAX_TOOL_RESULT_CHARS = 12000   # cap what we feed back per tool result


def _short(obj: Any, n: int = 140) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + "…"


# Human-in-the-loop tool — only exposed when interactive=True.
ASK_USER_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": ("Ask the human one clarifying question and get their "
                        "answer. Use for a decision that is THEIRS to make and "
                        "that you cannot settle from the request + tools:\n"
                        "  - which of several watersheds, or which HUC scale\n"
                        "  - a vague or ambiguous location\n"
                        "  - THE SIMULATION PERIOD, whenever the request did "
                        "not state one. Never default it silently when this "
                        "tool is available: the period sets compute cost, "
                        "spin-up and which observations exist, and the user is "
                        "the one who knows which year they mean.\n"
                        "Always present concrete candidates and your "
                        "recommendation, so the human can just confirm."),
        "parameters": {"type": "object",
                       "properties": {"question": {"type": "string"}},
                       "required": ["question"]},
    },
}


BATCH_SENTINEL = ("(non-interactive — no human available; use your best "
                  "judgment and record the assumption in the brief)")


def _open_tty():
    """Return (readable_terminal_stream, reason) for prompting the human.

    Must be called ONCE at startup, before any MCP call: each MCP tool call
    runs asyncio.run() which spawns a server subprocess and tears down the
    event loop, and that churn makes a later open('/dev/tty') fail with ENXIO
    even though it was openable at startup. So we grab a terminal stream early
    and hold it. Two sources, in order:
      1. a fresh open of /dev/tty (works when this process has a controlling
         terminal — the direct `python run_pipeline.py` case);
      2. fd 0, when a caller already connected it to the terminal (the shell
         wrapper runs us with `< /dev/tty`). An inherited, already-open fd
         survives the churn even when re-opening /dev/tty would not.
    """
    try:
        return open("/dev/tty", "r+"), None
    except OSError as e:
        reason = repr(e)
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            return sys.stdin, None
    except (OSError, ValueError):
        pass
    return None, reason


def _human_answer(question: str, tty, interactive: bool) -> Dict[str, Any]:
    """Prompt the human on the pre-opened terminal, else the batch sentinel."""
    print(f"\n  ❓ {question}")
    if interactive and tty is not None:
        try:
            sys.stderr.write("  your answer> ")   # prompt to stderr (works for
            sys.stderr.flush()                     # both /dev/tty and fd 0)
            line = tty.readline()                  # '' only on real EOF (Ctrl-D)
            if line:
                return {"answer": line.strip() or "(no answer given)"}
            print("  (end of input — using best-judgment fallback)")
        except (OSError, EOFError) as e:
            print(f"  (could not read your answer: {e!r})")
    return {"answer": BATCH_SENTINEL}


class ToolLoopAgent:
    """Runs an LLM <-> MCP tool-calling loop. Domain-agnostic."""

    def __init__(self,
                 model: str,
                 mcp_clients: Dict[str, Any],
                 allowlist: Optional[set] = None,
                 max_rounds: int = 8,
                 max_tokens: int = 8192,
                 verbose: bool = True,
                 interactive: bool = False):
        self.llm = SimpleLLMClient(model=model)
        self.mcp_clients = mcp_clients
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.interactive = interactive
        self._tty = None
        if interactive:                       # grab the terminal BEFORE MCP churn
            self._tty, reason = _open_tty()
            if self._tty is None:
                print("⚠️  interactive requested, but no controlling terminal is "
                      f"available ({reason}).\n    Clarifying questions will be "
                      "auto-answered with best-judgment defaults. To answer them "
                      "yourself, run this directly in a terminal (not piped, "
                      "nohup, or a job step).")
        self.tools, self.dispatch = self._build_tools(mcp_clients, allowlist)
        if interactive:
            self.tools.append(ASK_USER_TOOL)

    # ── build OpenAI tool schemas from MCP servers ───────────────────────
    @staticmethod
    def _build_tools(clients: Dict[str, Any],
                     allowlist: Optional[set]):
        """Return (openai_tools, dispatch_map). Tool names are server-prefixed
        as `server__tool` to avoid collisions; dispatch_map maps that back to
        (server, tool)."""
        tools, dispatch = [], {}
        for server, client in clients.items():
            for info in client.list_tools_detailed():
                fq = f"{server}__{info['name']}"
                if allowlist is not None and fq not in allowlist:
                    continue
                tools.append({
                    "type": "function",
                    "function": {
                        "name": fq,
                        "description": info.get("description") or "",
                        "parameters": info.get("parameters")
                        or {"type": "object", "properties": {}},
                    },
                })
                dispatch[fq] = (server, info["name"])
        return tools, dispatch

    @staticmethod
    def _assistant_dict(msg) -> Dict[str, Any]:
        d = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        return d

    def _create(self, messages):
        # tools= passed every round (Bedrock/Claude requirement).
        r = self.llm.client.chat.completions.create(
            model=self.llm.model, messages=messages,
            tools=self.tools, tool_choice="auto",
            max_tokens=self.max_tokens,
        )
        # A cut-off reply is not a formatting problem: the brief silently loses
        # its trailing fields, and downstream that looks like the model chose to
        # omit them. Say so rather than letting it pass as a finished answer.
        if (getattr(r, "choices", None)
                and getattr(r.choices[0], "finish_reason", None) == "length"):
            print(f"   ⚠️  reply hit the {self.max_tokens}-token cap and was cut "
                  f"off — raise max_tokens; trailing fields may be missing")
        return r

    # ── the loop ─────────────────────────────────────────────────────────
    def run(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}]
        trace: List[Dict[str, Any]] = []

        for rnd in range(1, self.max_rounds + 1):
            msg = self._create(messages).choices[0].message
            messages.append(self._assistant_dict(msg))

            if not msg.tool_calls:
                return {"content": msg.content or "", "trace": trace,
                        "rounds": rnd, "messages": messages, "truncated": False}

            for tc in msg.tool_calls:
                fq = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                if fq == "ask_user":
                    result = _human_answer(args.get("question", ""),
                                           self._tty, self.interactive)
                elif fq not in self.dispatch:
                    result = {"error": f"unknown tool {fq}"}
                else:
                    server, tool = self.dispatch[fq]
                    result = (self.mcp_clients[server].call_tool_json(tool, args)
                              or {"error": "tool returned no data"})
                trace.append({"round": rnd, "tool": fq, "args": args, "result": result})
                if self.verbose:
                    print(f"   [tool] {fq}({_short(args, 80)}) -> {_short(result)}")

                payload = json.dumps(result, default=str)
                if len(payload) > _MAX_TOOL_RESULT_CHARS:
                    payload = payload[:_MAX_TOOL_RESULT_CHARS] + " …[truncated]"
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": payload})

        # Hit max rounds — ask for a final answer without more tools.
        messages.append({"role": "user",
                         "content": "Stop calling tools and give your final answer now."})
        msg = self._create(messages).choices[0].message
        return {"content": msg.content or "", "trace": trace,
                "rounds": self.max_rounds, "messages": messages, "truncated": True}
