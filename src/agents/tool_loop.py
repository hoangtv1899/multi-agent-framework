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
from typing import Any, Dict, List, Optional

from agents.llm_agent import SimpleLLMClient

_MAX_TOOL_RESULT_CHARS = 12000   # cap what we feed back per tool result


def _short(obj: Any, n: int = 140) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + "…"


class ToolLoopAgent:
    """Runs an LLM <-> MCP tool-calling loop. Domain-agnostic."""

    def __init__(self,
                 model: str,
                 mcp_clients: Dict[str, Any],
                 allowlist: Optional[set] = None,
                 max_rounds: int = 8,
                 max_tokens: int = 4000,
                 verbose: bool = True):
        self.llm = SimpleLLMClient(model=model)
        self.mcp_clients = mcp_clients
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.tools, self.dispatch = self._build_tools(mcp_clients, allowlist)

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
        return self.llm.client.chat.completions.create(
            model=self.llm.model, messages=messages,
            tools=self.tools, tool_choice="auto",
            max_tokens=self.max_tokens,
        )

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
                if fq not in self.dispatch:
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
