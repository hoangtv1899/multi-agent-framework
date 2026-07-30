#!/usr/bin/env python3
"""
Simple LLM Agent using OpenAI API for PNNL
"""
import os
import re
import json
import openai
from typing import List, Dict, Any


# Every LLM call in the process, appended here as it completes. A module-level
# log rather than per-client state because the steps construct their own
# clients internally: step 4 has to account for tokens spent by objects it
# never sees. Each entry carries the label the caller set, so cost is
# attributable to a STEP rather than only to a total.
USAGE_LOG: List[Dict[str, Any]] = []


def usage_totals(label: str = None) -> Dict[str, Any]:
    """Aggregate USAGE_LOG, optionally for one label."""
    rows = [r for r in USAGE_LOG if label is None or r.get("label") == label]
    return {"calls": len(rows),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rows),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in rows),
            "seconds": round(sum(r.get("seconds") or 0.0 for r in rows), 2),
            "models": sorted({r.get("model") for r in rows if r.get("model")})}


class SimpleLLMClient:
    """Wrapper around OpenAI client for PNNL API."""

    def __init__(self, model: str = "claude-opus-4-8-project"):
        api_key = os.getenv("PNNL_API_KEY")
        if not api_key:
            raise ValueError("PNNL_API_KEY environment variable not set")
        self.client = openai.OpenAI(
            api_key  = api_key,
            base_url = "https://ai-incubator-api.pnnl.gov",
        )
        self.model = model
        # reproducibility knobs (None = provider default, preserves old behavior)
        self.temperature = None
        self.seed = None
        # NOT None. The gateway defaults to 4096 completion tokens, and the
        # planner's plan is longer than that: every call came back
        # finish_reason="length", truncated mid-JSON. The parse then failed, and
        # the repair round asked the model to "fix" JSON that was not malformed
        # but INCOMPLETE -- so it invented the missing tail. Measured over five
        # runs of the same brief, n_exploratory came back 8, 11, 12, 12 and then
        # missing entirely: the sampling design was partly written by a
        # JSON-repair prompt rather than by the planner.
        self.max_tokens = 16384
        self.label = None                    # which pipeline step is spending
        self.last_response_model = None      # provider-reported model version
        self.last_finish_reason = None       # "length" == the reply was cut off

    def ask(self,
            messages:       List[Dict[str, str]],
            system_message: str = None) -> str:
        """Send messages to LLM and get response."""
        import time as _time
        _t0 = _time.time()
        if system_message:
            messages = [{"role": "system",
                         "content": system_message}] + messages
        kwargs = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception:
            # some gateway backends reject optional sampling params — retry bare
            if "temperature" in kwargs or "seed" in kwargs:
                kwargs.pop("temperature", None)
                kwargs.pop("seed", None)
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise
        self.last_response_model = getattr(response, "model", None) or self.model
        choice = (getattr(response, "choices", None) or [None])[0]
        self.last_finish_reason = getattr(choice, "finish_reason", None)
        u = getattr(response, "usage", None)   # exact token accounting when the
        self.last_usage = ({"prompt_tokens": u.prompt_tokens,      # gateway
                            "completion_tokens": u.completion_tokens}
                           if u else None)     # returns it; None otherwise
        USAGE_LOG.append({
            "label": self.label,
            "model": self.last_response_model,
            "seconds": round(_time.time() - _t0, 2),
            "finish_reason": self.last_finish_reason,
            **(self.last_usage or {"prompt_tokens": None,
                                   "completion_tokens": None})})
        return response.choices[0].message.content


class LLMAgent:
    """Simple LLM-based agent."""

    def __init__(self,
                 name:           str,
                 system_message: str,
                 model:          str = "claude-opus-4-8-project"):
        self.name           = name
        self.system_message = system_message
        self.llm            = SimpleLLMClient(model=model)
        self.conversation_history = []

    def respond(self, user_message: str) -> str:
        """Get agent response to user message."""
        self.conversation_history.append({
            "role":    "user",
            "content": user_message
        })
        response = self.llm.ask(
            messages       = self.conversation_history,
            system_message = self.system_message
        )
        self.conversation_history.append({
            "role":    "assistant",
            "content": response
        })
        return response

    def ask_with_system(self,
                        user_message:   str,
                        system_message: str) -> str:
        """
        One-shot call with a custom system message.
        Does NOT affect conversation history — stateless.
        Use this instead of swapping self.system_message.
        """
        return self.llm.ask(
            messages       = [{"role": "user", "content": user_message}],
            system_message = system_message
        )

    @staticmethod
    def _strip_json_comments(s: str) -> str:
        """Remove // line comments that sit OUTSIDE string literals.

        The prompt schemas are annotated with `//` comments (see
        reception_agentic.txt), so models regularly echo them into their
        output. Naive regex stripping would corrupt URLs ("http://…") and any
        legitimate slashes inside strings, so track string state properly.
        """
        out, in_str, esc, i, n = [], False, False, 0, len(s)
        while i < n:
            c = s[i]
            if in_str:
                out.append(c)
                if esc:            esc = False
                elif c == "\\":    esc = True
                elif c == '"':     in_str = False
                i += 1
                continue
            if c == '"':
                in_str = True
                out.append(c)
                i += 1
                continue
            if c == "/" and i + 1 < n and s[i + 1] == "/":
                while i < n and s[i] != "\n":      # drop to end of line
                    i += 1
                continue
            out.append(c)
            i += 1
        return "".join(out)

    @staticmethod
    def _escape_raw_newlines(s: str) -> str:
        """Escape literal newlines/tabs that appear INSIDE string literals."""
        out, in_str, esc = [], False, False
        for c in s:
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                if not esc and c in "\n\r\t" and in_str:
                    out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[c])
                    continue
            elif c == '"':
                in_str = True
            out.append(c)
        return "".join(out)

    def parse_json(self, response: str) -> Dict[str, Any]:
        """
        Shared utility — extract JSON from LLM response.
        Handles markdown fences and leading/trailing text.
        Inherited by all agents — do not duplicate in subclasses.

        Applies a cascade of repairs for the LLM slips seen in practice:
        markdown fences, `//` comments echoed from the prompt schemas,
        trailing commas, and raw newlines inside strings. Each repair is
        additive, so the most-repaired form is tried last.
        """
        text = re.sub(r"```json\s*", "", response)
        text = re.sub(r"```\s*$",    "", text).strip()
        start = text.find("{")
        end   = text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON found in response: {text[:100]}")

        frag       = text[start:end + 1]
        no_comment = self._strip_json_comments(frag)
        no_trail   = re.sub(r",\s*([}\]])", r"\1", no_comment)
        escaped    = self._escape_raw_newlines(no_trail)

        last = None
        for candidate in (frag, no_comment, no_trail, escaped):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                last = e
        # Surface the offending region — "line 77 column 6" alone is useless
        # when the caller never sees the raw response.
        pos  = getattr(last, "pos", 0) or 0
        near = escaped[max(0, pos - 120): pos + 120]
        raise ValueError(
            f"Could not parse JSON after repairs ({last}). Near: …{near}…"
        ) from last

    def parse_json_resilient(self, response: str) -> Dict[str, Any]:
        """parse_json(), plus ONE LLM self-repair round if that fails.

        The deterministic repairs in parse_json() cover fences, echoed `//`
        comments, trailing commas and raw newlines. They deliberately do NOT
        guess at structural slips (a missing comma between elements), because
        inserting punctuation heuristically can silently change the data.
        For those, ask the model to fix its own output — the same approach
        the tools/ CLIs already use.

        Only the JSON is regenerated, not the reasoning, so this is cheap and
        cannot change the scientific content of a valid response.
        """
        truncated = getattr(self.llm, "last_finish_reason", None) == "length"
        try:
            return self.parse_json(response)
        except Exception as first:
            if truncated:
                # Repairing a truncated reply means asking the model to INVENT
                # the part that never arrived, and the invention is then
                # indistinguishable from a designed value. Fail loudly instead.
                raise ValueError(
                    "LLM reply was cut off (finish_reason='length'), so the JSON "
                    "is incomplete rather than malformed. Raise max_tokens — "
                    "repairing this would fabricate the missing fields, not "
                    "recover them."
                ) from first
            print(f"   ⚠️  JSON parse failed ({str(first)[:90]}…) — "
                  f"asking the model to repair it")
            try:
                fixed = self.llm.ask(
                    [{"role": "user",
                      "content": "The following was meant to be a single JSON "
                                 "object but does not parse. Return ONLY the "
                                 "corrected JSON — identical content, no "
                                 "commentary, no markdown fence.\n\n" + response}],
                    system_message="You repair malformed JSON. Output valid "
                                   "JSON only, preserving every value exactly.",
                )
                return self.parse_json(fixed)
            except Exception as second:
                raise ValueError(
                    f"JSON unparseable even after LLM repair. "
                    f"original={first} ; repair={second}"
                ) from first

    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []

    def __str__(self):
        return f"Agent({self.name})"